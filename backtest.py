"""
回测模块
- 回测引擎
- 指标计算
- 交易记录
"""

import numpy as np
import pandas as pd
from datetime import timedelta


class Backtest:
    """回测引擎"""
    
    def __init__(self, initial_capital=1000000, rebalance_cost=0.001, slippage=0.001):
        self.initial_capital = initial_capital
        self.rebalance_cost = rebalance_cost
        self.slippage = slippage
        self.portfolio_values = []
        self.positions_history = []
        self.trades_history = []
        self.rebalance_records = []
        
    def run(self, price_data_dict, strategy, start_date, end_date, rebalance_period=10,
            market_timing=None, market_index_code='000001', etf_names=None):
        """
        运行回测
        
        Parameters:
            price_data_dict: {sec_code: DataFrame} 价格数据字典
            strategy: 策略实例（SimpleMomentumStrategy）
            start_date: 开始日期
            end_date: 结束日期
            rebalance_period: 调仓周期（交易日数，默认10个交易日）
            market_timing: 市场择时实例（可选，用于根据市场状态调整仓位）
            market_index_code: 市场指数代码（默认上证指数000001）
            etf_names: {sec_code: 缩写} ETF代码到缩写的映射（可选，用于调仓记录输出）
        
        Returns:
            dict: 回测指标
        """
        # 获取基准ETF数据（沪深300ETF和上证50ETF）
        from data import get_etf_hist
        self.benchmark_data = {}
        try:
            self.benchmark_data['hs300'] = get_etf_hist('510300', start_date.strftime('%Y-%m-%d'), 
                                                         end_date.strftime('%Y-%m-%d'))
        except:
            self.benchmark_data['hs300'] = None
        
        try:
            self.benchmark_data['sh'] = get_etf_hist('510050', start_date.strftime('%Y-%m-%d'), 
                                                      end_date.strftime('%Y-%m-%d'))
        except:
            self.benchmark_data['sh'] = None
        
        # 获取市场指数数据（用于市场择时）
        self.market_index_code = market_index_code
        if market_timing is not None:
            from data import get_etf_hist
            try:
                self.market_data = get_etf_hist(market_index_code, start_date.strftime('%Y-%m-%d'), 
                                                end_date.strftime('%Y-%m-%d'))
            except:
                self.market_data = None
                market_timing = None  # 无法获取市场数据时禁用择时
        
        # 生成调仓日期（使用列表而不是set，避免哈希随机化导致的不确定性）
        all_dates_list = []
        for df in price_data_dict.values():
            for idx in df.index.tolist():
                if idx not in all_dates_list:
                    all_dates_list.append(idx)
        trading_dates = sorted([d for d in all_dates_list if pd.to_datetime(d) >= start_date])
        
        # 按调仓周期分组
        rebalance_dates = self._get_rebalance_dates(trading_dates, period=rebalance_period)
        capital = self.initial_capital
        current_positions = {}
        
        LOT_SIZE = 100  # 一手100股
        # 交易总成本：手续费 + 滑点
        buy_cost_rate = self.rebalance_cost + self.slippage
        sell_cost_rate = self.rebalance_cost + self.slippage
        
        # ========== 运行回测 ==========
        print(f"\n{'='*50}")
        print("回测执行")
        print(f"{'='*50}")
        
        prev_rebalance_date = None
        prev_portfolio_value = self.initial_capital
        
        for rebalance_date in rebalance_dates:
            rebalance_date = pd.to_datetime(rebalance_date)
            
            # 获取当日收盘价（用于组合估值、策略计算和交易执行）
            close_prices = {}
            for sec_code, df in price_data_dict.items():
                if rebalance_date in df.index:
                    close_prices[sec_code] = df.loc[rebalance_date, 'close']
            
            if len(close_prices) == 0:
                continue
            
            # 计算持仓价值（使用收盘价）
            portfolio_value = capital
            for sec_code, shares in current_positions.items():
                if sec_code in close_prices:
                    portfolio_value += shares * close_prices[sec_code]
            
            # 记录组合价值
            self.portfolio_values.append({
                'date': rebalance_date,
                'value': portfolio_value
            })
            
            # 记录调仓前的持仓（用于计算调仓变化）
            positions_before = current_positions.copy()
            
            # 策略调仓 - 使用调仓日之前的数据（收盘价计算策略）
            hist_price_dict = {}
            for sec_code, df in price_data_dict.items():
                hist_df = df[df.index < rebalance_date]
                # 多期限动量需要至少120天数据
                min_lookback = 120 if getattr(strategy, 'use_multi_momentum', False) else strategy.momentum_lookback
                if len(hist_df) >= min_lookback:
                    hist_price_dict[sec_code] = hist_df
            
            if len(hist_price_dict) > 0:
                factor_df = strategy.calculate_factors(hist_price_dict)
                selected = strategy.select_etfs(factor_df)
                weights = strategy.calculate_weights(selected)
                
                if len(weights) > 0:
                    # 计算新持仓（使用收盘价）
                    new_positions = {}
                    
                    # 根据市场择时决定仓位
                    position_ratio = 0.95  # 默认95%仓位
                    if market_timing is not None and self.market_data is not None:
                        # 获取调仓日之前的市场数据
                        market_hist = self.market_data[self.market_data.index < rebalance_date]
                        if len(market_hist) > 0:
                            market_prices = market_hist['close']
                            position_ratio = market_timing.get_position_ratio(market_prices)
                    
                    target_capital = portfolio_value * position_ratio
                    
                    for sec_code, weight in weights.items():
                        if sec_code in close_prices:
                            target_value = target_capital * weight
                            # 按一手100股取整
                            raw_shares = int(target_value / close_prices[sec_code])
                            lot_shares = (raw_shares // LOT_SIZE) * LOT_SIZE
                            if lot_shares > 0:
                                new_positions[sec_code] = lot_shares
                    
                    # 执行调仓（使用收盘价，含手续费和滑点）
                    old_positions_set = set(current_positions.keys())
                    new_positions_set = set(new_positions.keys())
                    
                    # 卖出不再持有的（按代码排序确保确定性）
                    for sec_code in sorted(old_positions_set - new_positions_set):
                        if sec_code in close_prices:
                            sell_value = current_positions[sec_code] * close_prices[sec_code]
                            capital += sell_value * (1 - sell_cost_rate)
                            self.trades_history.append({
                                'date': rebalance_date,
                                'sec_code': sec_code,
                                'action': 'sell',
                                'shares': current_positions[sec_code],
                                'price': close_prices[sec_code]
                            })
                    
                    # 调整持仓（按代码排序确保确定性，避免set遍历顺序受哈希随机化影响）
                    for sec_code in sorted(new_positions_set):
                        new_shares = new_positions[sec_code]
                        old_shares = current_positions.get(sec_code, 0)
                        
                        if new_shares > old_shares:
                            buy_shares = new_shares - old_shares
                            buy_value = buy_shares * close_prices[sec_code]
                            cost = buy_value * (1 + buy_cost_rate)
                            # 确保有足够现金，严格控制不超买
                            if cost <= capital:
                                capital -= cost
                                self.trades_history.append({
                                    'date': rebalance_date,
                                    'sec_code': sec_code,
                                    'action': 'buy',
                                    'shares': buy_shares,
                                    'price': close_prices[sec_code]
                                })
                            else:
                                # 现金不足时，按可用资金计算可买手数
                                available_shares = int(capital / (close_prices[sec_code] * (1 + buy_cost_rate)))
                                available_shares = (available_shares // LOT_SIZE) * LOT_SIZE
                                if available_shares > 0:
                                    actual_cost = available_shares * close_prices[sec_code] * (1 + buy_cost_rate)
                                    capital -= actual_cost
                                    # 更新目标份额
                                    new_positions[sec_code] = old_shares + available_shares
                                    self.trades_history.append({
                                        'date': rebalance_date,
                                        'sec_code': sec_code,
                                        'action': 'buy',
                                        'shares': available_shares,
                                        'price': close_prices[sec_code]
                                    })
                                else:
                                    # 买不起一手，保持原持仓
                                    new_positions[sec_code] = old_shares
                        elif new_shares < old_shares:
                            sell_shares = old_shares - new_shares
                            sell_value = sell_shares * close_prices[sec_code]
                            capital += sell_value * (1 - sell_cost_rate)
                            self.trades_history.append({
                                'date': rebalance_date,
                                'sec_code': sec_code,
                                'action': 'sell',
                                'shares': sell_shares,
                                'price': close_prices[sec_code]
                            })
                    
                    current_positions = new_positions
            
            # 记录调仓详情（计算本调仓周期收益和基准收益）
            period_return = (portfolio_value / prev_portfolio_value - 1) * 100 if prev_portfolio_value > 0 else 0
            
            hs300_return = 0
            sh_return = 0
            if prev_rebalance_date is not None:
                if self.benchmark_data['hs300'] is not None:
                    hs300_df = self.benchmark_data['hs300']
                    if prev_rebalance_date in hs300_df.index and rebalance_date in hs300_df.index:
                        hs300_return = (hs300_df.loc[rebalance_date, 'close'] / 
                                        hs300_df.loc[prev_rebalance_date, 'close'] - 1) * 100
                
                if self.benchmark_data['sh'] is not None:
                    sh_df = self.benchmark_data['sh']
                    if prev_rebalance_date in sh_df.index and rebalance_date in sh_df.index:
                        sh_return = (sh_df.loc[rebalance_date, 'close'] / 
                                     sh_df.loc[prev_rebalance_date, 'close'] - 1) * 100
            
            # 计算调仓变化
            positions_after = current_positions.copy()
            added = list(set(positions_after.keys()) - set(positions_before.keys()))
            removed = list(set(positions_before.keys()) - set(positions_after.keys()))
            changed = [code for code in positions_after.keys() 
                       if code in positions_before and positions_before[code] != positions_after[code]]
            
            # 格式化持仓为"缩写:仓位比例"
            def format_positions(positions, prices, total_value):
                if total_value <= 0 or not positions:
                    return ''
                items = []
                for code, shares in sorted(positions.items()):
                    name = etf_names.get(code, code) if etf_names else code
                    if code in prices:
                        ratio = shares * prices[code] / total_value * 100
                        items.append(f"{name}:{ratio:.0f}%")
                    else:
                        items.append(f"{name}:0%")
                return ', '.join(items)
            
            self.rebalance_records.append({
                'date': rebalance_date,
                'prev_date': prev_rebalance_date,
                'portfolio_value': round(portfolio_value, 2),
                'period_return': round(period_return, 2),
                'hs300_return': round(hs300_return, 2),
                'sh_return': round(sh_return, 2),
                'positions_before': format_positions(positions_before, close_prices, portfolio_value),
                'positions_after': format_positions(positions_after, close_prices, portfolio_value),
                'added': ','.join(added),
                'removed': ','.join(removed),
                'changed': ','.join(changed),
                'num_positions': len(positions_after)
            })
            
            prev_rebalance_date = rebalance_date
            prev_portfolio_value = portfolio_value
        
        # 计算指标
        metrics = self._calculate_metrics()
        
        return metrics
    
    def _get_rebalance_dates(self, dates, period=10):
        """
        根据调仓周期获取调仓日期
        
        Parameters:
            dates: 日期列表
            period: 调仓周期（交易日数，默认10个交易日）
        
        Returns:
            list: 调仓日期列表
        """
        if len(dates) == 0:
            return []
        
        dates = sorted([pd.to_datetime(d) for d in dates])
        # 每隔 period 个交易日选择一个调仓日
        rebalance_dates = [dates[i] for i in range(0, len(dates), period)]
        return rebalance_dates
    
    def _calculate_metrics(self):
        """计算回测指标"""
        if len(self.portfolio_values) == 0:
            return {}
        
        df = pd.DataFrame(self.portfolio_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
        
        # 日收益率
        df['return'] = df['value'].pct_change()
        df['cum_return'] = df['value'] / self.initial_capital - 1
        
        # 计算整体指标
        total_days = (df.index[-1] - df.index[0]).days
        annual_return = (1 + df['cum_return'].iloc[-1]) ** (365 / max(total_days, 1)) - 1
        annual_vol = df['return'].std() * np.sqrt(52)
        risk_free = 0.03
        sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0
        
        df['cummax'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        win_rate = (df['return'] > 0).sum() / len(df['return'].dropna()) if len(df['return'].dropna()) > 0 else 0
        
        metrics = {
            'total_return': df['cum_return'].iloc[-1],
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'portfolio_df': df,
            'trades_history': self.trades_history
        }
        
        return metrics
    
    def get_trades_summary(self):
        """获取交易汇总"""
        if len(self.trades_history) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades_history)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def get_rebalance_records(self):
        """获取调仓记录"""
        if len(self.rebalance_records) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.rebalance_records)
        df['date'] = pd.to_datetime(df['date'])
        df['prev_date'] = pd.to_datetime(df['prev_date'])
        return df
    
    def save_rebalance_records(self, file_path='rebalance_records.csv'):
        """保存调仓记录到CSV文件"""
        df = self.get_rebalance_records()
        if len(df) > 0:
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"调仓记录已保存至: {file_path}")


def calculate_benchmark_return(benchmark_df, start_date, end_date):
    """计算基准收益率"""
    if len(benchmark_df) == 0:
        return 0
    
    benchmark_df = benchmark_df[(benchmark_df.index >= start_date) & (benchmark_df.index <= end_date)]
    if len(benchmark_df) == 0:
        return 0
    
    initial_value = benchmark_df['close'].iloc[0]
    final_value = benchmark_df['close'].iloc[-1]
    return (final_value / initial_value - 1) * 100
