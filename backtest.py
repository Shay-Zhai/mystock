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
    
    def __init__(self, initial_capital=1000000, rebalance_cost=0.001, slippage=0.001,
                 stop_loss_config=None):
        self.initial_capital = initial_capital
        self.rebalance_cost = rebalance_cost
        self.slippage = slippage
        self.portfolio_values = []
        self.positions_history = []
        self.trades_history = []
        self.rebalance_records = []
        # 止损配置
        self.stop_loss_config = stop_loss_config or {'mode': 'none'}
        
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
        
        # 本轮调仓的交易汇总（费用、调仓金额）
        rebalance_turnover = 0.0  # 调仓金额：min(卖出额, 买入额) * 2 + |卖出额-买入额|
        rebalance_total_cost = 0.0  # 费用（手续费+滑点）
        
        # ========== 运行回测 ==========
        print(f"\n{'='*50}")
        print("回测执行")
        print(f"{'='*50}")
        
        # 构建调仓日集合
        rebalance_date_set = set(pd.to_datetime(d) for d in rebalance_dates)
        
        # 止损配置
        sl_cfg = self.stop_loss_config
        sl_mode = sl_cfg.get('mode', 'none')
        # 单票止损：从持仓最高收盘价回撤超过该百分比则止损
        sl_individual_dd = sl_cfg.get('individual_drawdown_pct', 0.10)
        sl_dd_half = sl_cfg.get('portfolio_dd_half', 0.18)
        sl_dd_clear = sl_cfg.get('portfolio_dd_clear', 0.25)
        
        # 止损状态
        resting = False  # 整体止损18%后休息状态
        position_high_prices = {}  # 各ETF持仓期间最高收盘价
        portfolio_high_value = self.initial_capital  # 组合净值最高点
        portfolio_half_sold = False  # 12%减半止损是否已触发（每次回撤事件仅触发一次）
        
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
        
        prev_rebalance_date = None
        prev_portfolio_value = self.initial_capital
        
        # 遍历所有交易日（逐日检查止损，调仓日执行策略）
        for trade_date in trading_dates:
            trade_date = pd.to_datetime(trade_date)
            
            # 获取当日收盘价
            close_prices = {}
            for sec_code, df in price_data_dict.items():
                if trade_date in df.index:
                    close_prices[sec_code] = df.loc[trade_date, 'close']
            
            if len(close_prices) == 0:
                continue
            
            # 计算当日组合净值
            portfolio_value = capital
            for sec_code, shares in current_positions.items():
                if sec_code in close_prices:
                    portfolio_value += shares * close_prices[sec_code]
            
            # 更新持仓ETF最高收盘价
            for sec_code in list(current_positions.keys()):
                if sec_code in close_prices:
                    if sec_code not in position_high_prices:
                        position_high_prices[sec_code] = close_prices[sec_code]
                    else:
                        position_high_prices[sec_code] = max(position_high_prices[sec_code], close_prices[sec_code])
            
            # 更新组合净值最高点
            if portfolio_value > 0:
                portfolio_high_value = max(portfolio_high_value, portfolio_value)
            
            # ===== 止损检查（有持仓且非休息状态） =====
            if len(current_positions) > 0 and not resting:
                # 单票止损：从持仓最高收盘价回撤超过阈值百分比
                if sl_mode in ('individual', 'both'):
                    for sec_code in sorted(current_positions.keys()):
                        if sec_code not in close_prices or sec_code not in position_high_prices:
                            continue
                        high_price = position_high_prices[sec_code]
                        current_price = close_prices[sec_code]
                        if high_price <= 0:
                            continue
                        drawdown_pct = (high_price - current_price) / high_price
                        if drawdown_pct > sl_individual_dd:
                            sell_value = current_positions[sec_code] * current_price
                            trade_cost = sell_value * sell_cost_rate
                            capital += sell_value - trade_cost
                            self.trades_history.append({
                                'date': trade_date, 'sec_code': sec_code, 'action': 'sell',
                                'shares': current_positions[sec_code], 'price': current_price,
                                'turnover': round(sell_value, 2), 'cost': round(trade_cost, 2)
                            })
                            del current_positions[sec_code]
                            del position_high_prices[sec_code]

                    # 重新计算组合净值
                    portfolio_value = capital
                    for sec_code, shares in current_positions.items():
                        if sec_code in close_prices:
                            portfolio_value += shares * close_prices[sec_code]
                
                # 整体止损：账户回撤超过阈值
                if sl_mode in ('portfolio', 'both') and len(current_positions) > 0:
                    portfolio_drawdown = (portfolio_high_value - portfolio_value) / portfolio_high_value if portfolio_high_value > 0 else 0

                    if portfolio_drawdown > sl_dd_clear:
                        # 回撤超18%：全部清仓，休息到下个调仓周期
                        for sec_code in sorted(current_positions.keys()):
                            if sec_code in close_prices:
                                sell_value = current_positions[sec_code] * close_prices[sec_code]
                                trade_cost = sell_value * sell_cost_rate
                                capital += sell_value - trade_cost
                                self.trades_history.append({
                                    'date': trade_date, 'sec_code': sec_code, 'action': 'sell',
                                    'shares': current_positions[sec_code], 'price': close_prices[sec_code],
                                    'turnover': round(sell_value, 2), 'cost': round(trade_cost, 2)
                                })
                        current_positions = {}
                        position_high_prices = {}
                        resting = True
                        portfolio_half_sold = False  # 重置减半标志
                    elif portfolio_drawdown > sl_dd_half and not portfolio_half_sold:
                        # 回撤超12%：每个标的卖出一半（仅触发一次）
                        portfolio_half_sold = True
                        for sec_code in sorted(current_positions.keys()):
                            if sec_code in close_prices:
                                half_shares = (current_positions[sec_code] // 2 // LOT_SIZE) * LOT_SIZE
                                if half_shares > 0:
                                    sell_value = half_shares * close_prices[sec_code]
                                    trade_cost = sell_value * sell_cost_rate
                                    capital += sell_value - trade_cost
                                    self.trades_history.append({
                                        'date': trade_date, 'sec_code': sec_code, 'action': 'sell',
                                        'shares': half_shares, 'price': close_prices[sec_code],
                                        'turnover': round(sell_value, 2), 'cost': round(trade_cost, 2)
                                    })
                                    current_positions[sec_code] -= half_shares
                                    if current_positions[sec_code] <= 0:
                                        del current_positions[sec_code]
                                        if sec_code in position_high_prices:
                                            del position_high_prices[sec_code]
                    elif portfolio_drawdown < sl_dd_half:
                        # 回撤恢复到12%以下：重置减半标志，允许下次再次触发
                        portfolio_half_sold = False
            
            # ===== 调仓日处理 =====
            if trade_date not in rebalance_date_set:
                continue
            
            # 重新计算组合净值（止损可能改变了持仓）
            portfolio_value = capital
            for sec_code, shares in current_positions.items():
                if sec_code in close_prices:
                    portfolio_value += shares * close_prices[sec_code]
            
            # 记录组合价值
            self.portfolio_values.append({
                'date': trade_date, 'value': portfolio_value
            })
            
            # 重置本轮调仓统计
            rebalance_turnover = 0.0
            rebalance_total_cost = 0.0
            total_sell_value = 0.0
            total_buy_value = 0.0
            
            # 计算周期收益和基准收益
            period_return = (portfolio_value / prev_portfolio_value - 1) * 100 if prev_portfolio_value > 0 else 0
            hs300_return = 0
            sh_return = 0
            if prev_rebalance_date is not None:
                if self.benchmark_data['hs300'] is not None:
                    hs300_df = self.benchmark_data['hs300']
                    if prev_rebalance_date in hs300_df.index and trade_date in hs300_df.index:
                        hs300_return = (hs300_df.loc[trade_date, 'close'] /
                                        hs300_df.loc[prev_rebalance_date, 'close'] - 1) * 100
                if self.benchmark_data['sh'] is not None:
                    sh_df = self.benchmark_data['sh']
                    if prev_rebalance_date in sh_df.index and trade_date in sh_df.index:
                        sh_return = (sh_df.loc[trade_date, 'close'] /
                                     sh_df.loc[prev_rebalance_date, 'close'] - 1) * 100
            
            # 休息状态：跳过本次调仓，解除休息
            if resting:
                resting = False
                portfolio_high_value = portfolio_value
                position_high_prices = {}
                portfolio_half_sold = False  # 重置减半标志
                self.rebalance_records.append({
                    'date': trade_date, 'prev_date': prev_rebalance_date,
                    'portfolio_value': round(portfolio_value, 2),
                    'period_return': round(period_return, 2),
                    'hs300_return': round(hs300_return, 2), 'sh_return': round(sh_return, 2),
                    'positions_before': '', 'positions_after': '',
                    'added': '', 'removed': '', 'changed': '', 'num_positions': 0,
                    'turnover': 0, 'turnover_rate': 0, 'total_cost': 0, 'cost_rate': 0
                })
                prev_rebalance_date = trade_date
                prev_portfolio_value = portfolio_value
                continue
            
            # 记录调仓前持仓
            positions_before = current_positions.copy()
            
            # 策略调仓 - 使用调仓日之前的数据
            hist_price_dict = {}
            for sec_code, df in price_data_dict.items():
                hist_df = df[df.index < trade_date]
                if getattr(strategy, 'use_multi_momentum', False):
                    min_lookback = 120
                else:
                    min_lookback = max(strategy.momentum_lookback + getattr(strategy, 'momentum_skip', 0) + 1,
                                       strategy.trend_ma)
                if len(hist_df) >= min_lookback:
                    hist_price_dict[sec_code] = hist_df
            
            if len(hist_price_dict) > 0:
                factor_df = strategy.calculate_factors(hist_price_dict)
                selected = strategy.select_etfs(factor_df)
                weights = strategy.calculate_weights(selected)
                
                if len(weights) > 0:
                    new_positions = {}
                    position_ratio = 0.95
                    if market_timing is not None and self.market_data is not None:
                        market_hist = self.market_data[self.market_data.index < trade_date]
                        if len(market_hist) > 0:
                            position_ratio = market_timing.get_position_ratio(market_hist['close'])
                    
                    target_capital = portfolio_value * position_ratio
                    for sec_code, weight in weights.items():
                        if sec_code in close_prices:
                            raw_shares = int(target_capital * weight / close_prices[sec_code])
                            lot_shares = (raw_shares // LOT_SIZE) * LOT_SIZE
                            if lot_shares > 0:
                                new_positions[sec_code] = lot_shares
                    
                    old_positions_set = set(current_positions.keys())
                    new_positions_set = set(new_positions.keys())
                    
                    # 卖出不再持有的
                    for sec_code in sorted(old_positions_set - new_positions_set):
                        if sec_code in close_prices:
                            sell_value = current_positions[sec_code] * close_prices[sec_code]
                            trade_cost = sell_value * sell_cost_rate
                            capital += sell_value - trade_cost
                            total_sell_value += sell_value
                            rebalance_total_cost += trade_cost
                            self.trades_history.append({
                                'date': trade_date, 'sec_code': sec_code, 'action': 'sell',
                                'shares': current_positions[sec_code], 'price': close_prices[sec_code],
                                'turnover': round(sell_value, 2), 'cost': round(trade_cost, 2)
                            })
                            if sec_code in position_high_prices:
                                del position_high_prices[sec_code]
                    
                    # 调整持仓
                    for sec_code in sorted(new_positions_set):
                        new_shares = new_positions[sec_code]
                        old_shares = current_positions.get(sec_code, 0)
                        
                        if new_shares > old_shares:
                            buy_shares = new_shares - old_shares
                            buy_value = buy_shares * close_prices[sec_code]
                            cost = buy_value * (1 + buy_cost_rate)
                            trade_cost = buy_value * buy_cost_rate
                            if cost <= capital:
                                capital -= cost
                                total_buy_value += buy_value
                                rebalance_total_cost += trade_cost
                                self.trades_history.append({
                                    'date': trade_date, 'sec_code': sec_code, 'action': 'buy',
                                    'shares': buy_shares, 'price': close_prices[sec_code],
                                    'turnover': round(buy_value, 2), 'cost': round(trade_cost, 2)
                                })
                            else:
                                available_shares = int(capital / (close_prices[sec_code] * (1 + buy_cost_rate)))
                                available_shares = (available_shares // LOT_SIZE) * LOT_SIZE
                                if available_shares > 0:
                                    actual_buy_value = available_shares * close_prices[sec_code]
                                    actual_cost = actual_buy_value * (1 + buy_cost_rate)
                                    actual_trade_cost = actual_buy_value * buy_cost_rate
                                    capital -= actual_cost
                                    total_buy_value += actual_buy_value
                                    rebalance_total_cost += actual_trade_cost
                                    new_positions[sec_code] = old_shares + available_shares
                                    self.trades_history.append({
                                        'date': trade_date, 'sec_code': sec_code, 'action': 'buy',
                                        'shares': available_shares, 'price': close_prices[sec_code],
                                        'turnover': round(actual_buy_value, 2), 'cost': round(actual_trade_cost, 2)
                                    })
                                else:
                                    new_positions[sec_code] = old_shares
                        elif new_shares < old_shares:
                            sell_shares = old_shares - new_shares
                            sell_value = sell_shares * close_prices[sec_code]
                            trade_cost = sell_value * sell_cost_rate
                            capital += sell_value - trade_cost
                            total_sell_value += sell_value
                            rebalance_total_cost += trade_cost
                            self.trades_history.append({
                                'date': trade_date, 'sec_code': sec_code, 'action': 'sell',
                                'shares': sell_shares, 'price': close_prices[sec_code],
                                'turnover': round(sell_value, 2), 'cost': round(trade_cost, 2)
                            })
                    
                    current_positions = new_positions
                    
                    # 为新建仓ETF初始化最高价
                    for sec_code in current_positions:
                        if sec_code in close_prices and sec_code not in position_high_prices:
                            position_high_prices[sec_code] = close_prices[sec_code]
            
            # 记录调仓详情
            positions_after = current_positions.copy()
            added = list(set(positions_after.keys()) - set(positions_before.keys()))
            removed = list(set(positions_before.keys()) - set(positions_after.keys()))
            changed = [code for code in positions_after.keys()
                       if code in positions_before and positions_before[code] != positions_after[code]]
            
            rebalance_turnover = (total_sell_value + total_buy_value) / 2
            cost_rate = rebalance_total_cost / portfolio_value * 100 if portfolio_value > 0 else 0
            turnover_rate = rebalance_turnover / portfolio_value * 100 if portfolio_value > 0 else 0
            
            self.rebalance_records.append({
                'date': trade_date, 'prev_date': prev_rebalance_date,
                'portfolio_value': round(portfolio_value, 2),
                'period_return': round(period_return, 2),
                'hs300_return': round(hs300_return, 2), 'sh_return': round(sh_return, 2),
                'positions_before': format_positions(positions_before, close_prices, portfolio_value),
                'positions_after': format_positions(positions_after, close_prices, portfolio_value),
                'added': ','.join(added), 'removed': ','.join(removed),
                'changed': ','.join(changed), 'num_positions': len(positions_after),
                'turnover': round(rebalance_turnover, 2), 'turnover_rate': round(turnover_rate, 2),
                'total_cost': round(rebalance_total_cost, 2), 'cost_rate': round(cost_rate, 2)
            })
            
            prev_rebalance_date = trade_date
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
