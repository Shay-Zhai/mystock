"""
回测模块
- 回测引擎（支持训练/测试分离）
- 指标计算
- 交易记录
"""

import numpy as np
import pandas as pd
from datetime import timedelta


class Backtest:
    """回测引擎（支持训练/测试分离）"""
    
    def __init__(self, initial_capital=1000000, rebalance_cost=0.001):
        self.initial_capital = initial_capital
        self.rebalance_cost = rebalance_cost
        self.portfolio_values = []
        self.positions_history = []
        self.trades_history = []
        self.train_end_date = None
        
    def run(self, price_data_dict, strategy, start_date, end_date, rebalance_freq='biweekly',
            market_timing=None, market_index_code='000001'):
        """
        运行回测
        
        Parameters:
            price_data_dict: {sec_code: DataFrame} 价格数据字典
            strategy: 策略实例（MultiFactorETFStrategy或SimpleMomentumStrategy）
            start_date: 开始日期
            end_date: 结束日期
            rebalance_freq: 'weekly' 或 'biweekly'（默认双周）
            market_timing: 市场择时实例（可选，用于根据市场状态调整仓位）
            market_index_code: 市场指数代码（默认上证指数000001）
        
        Returns:
            dict: 回测指标（包含训练期和测试期）
        """
        self.train_end_date = strategy.train_end_date
        
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
        
        # 生成调仓日期
        all_dates = set()
        for df in price_data_dict.values():
            all_dates.update(df.index.tolist())
        trading_dates = sorted([d for d in all_dates if pd.to_datetime(d) >= start_date])
        
        # 按周分组（根据调仓频率）
        weekly_dates = self._get_weekly_dates(trading_dates, freq=rebalance_freq)
        
        capital = self.initial_capital
        current_positions = {}
        
        # ========== 训练期：记录因子数据（仅多因子策略） ==========
        if self.train_end_date is not None:
            print(f"\n{'='*50}")
            print("训练期数据收集")
            print(f"{'='*50}")
            print(f"训练期: {start_date.strftime('%Y-%m-%d')} 至 {self.train_end_date.strftime('%Y-%m-%d')}")
            
            train_week_count = 0
            for week_dates in weekly_dates:
                if len(week_dates) == 0:
                    continue
                
                rebalance_date = week_dates[-1]
                
                # 只处理训练期内的数据
                if pd.to_datetime(rebalance_date) > self.train_end_date:
                    continue
                
                # 记录训练数据
                if len(price_data_dict) > 0:
                    strategy.record_training_data(price_data_dict, rebalance_date)
                    train_week_count += 1
            
            print(f"收集调仓次数: {train_week_count}")
            
            # 训练：计算因子IC
            strategy.train()
        
        # ========== 测试期：运行回测 ==========
        print(f"\n{'='*50}")
        print("回测执行")
        print(f"{'='*50}")
        test_start = None
        test_end = None
        
        for week_dates in weekly_dates:
            if len(week_dates) == 0:
                continue
            
            rebalance_date = week_dates[-1]
            
            # 获取当日价格
            current_prices = {}
            for sec_code, df in price_data_dict.items():
                if rebalance_date in df.index:
                    current_prices[sec_code] = df.loc[rebalance_date, 'close']
            
            if len(current_prices) == 0:
                continue
            
            # 计算持仓价值
            portfolio_value = capital
            for sec_code, shares in current_positions.items():
                if sec_code in current_prices:
                    portfolio_value += shares * current_prices[sec_code]
            
            # 判断是否在测试期
            is_test_period = (self.train_end_date is None) or (pd.to_datetime(rebalance_date) > self.train_end_date)
            
            # 记录组合价值
            self.portfolio_values.append({
                'date': rebalance_date,
                'value': portfolio_value,
                'is_test': is_test_period
            })
            
            # 更新测试期起止日期
            if is_test_period:
                if test_start is None:
                    test_start = rebalance_date
                test_end = rebalance_date
            
            # 执行交易（简单策略全程交易，多因子策略只在测试期交易）
            if is_test_period:
                # 策略调仓 - 使用调仓日之前的数据
                hist_price_dict = {}
                for sec_code, df in price_data_dict.items():
                    hist_df = df[df.index < rebalance_date]
                    if len(hist_df) >= strategy.momentum_lookback:
                        hist_price_dict[sec_code] = hist_df
                
                if len(hist_price_dict) > 0:
                    factor_df = strategy.calculate_factors(hist_price_dict)
                    selected = strategy.select_etfs(factor_df)
                    weights = strategy.calculate_weights(selected)
                    
                    if len(weights) > 0:
                        # 计算新持仓
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
                            if sec_code in current_prices:
                                target_value = target_capital * weight
                                new_positions[sec_code] = int(target_value / current_prices[sec_code])
                        
                        # 调仓
                        old_positions_set = set(current_positions.keys())
                        new_positions_set = set(new_positions.keys())
                        
                        # 卖出不再持有的
                        for sec_code in old_positions_set - new_positions_set:
                            if sec_code in current_prices:
                                sell_value = current_positions[sec_code] * current_prices[sec_code]
                                capital += sell_value * (1 - self.rebalance_cost)
                                self.trades_history.append({
                                    'date': rebalance_date,
                                    'sec_code': sec_code,
                                    'action': 'sell',
                                    'shares': current_positions[sec_code],
                                    'price': current_prices[sec_code]
                                })
                        
                        # 调整持仓
                        for sec_code in new_positions_set:
                            new_shares = new_positions[sec_code]
                            old_shares = current_positions.get(sec_code, 0)
                            
                            if new_shares > old_shares:
                                buy_shares = new_shares - old_shares
                                buy_value = buy_shares * current_prices[sec_code]
                                cost = buy_value * (1 + self.rebalance_cost)
                                # 确保有足够现金，严格控制不超买
                                if cost <= capital:
                                    capital -= cost
                                    self.trades_history.append({
                                        'date': rebalance_date,
                                        'sec_code': sec_code,
                                        'action': 'buy',
                                        'shares': buy_shares,
                                        'price': current_prices[sec_code]
                                    })
                                else:
                                    # 现金不足时，按比例买入
                                    available_shares = int(capital / (current_prices[sec_code] * (1 + self.rebalance_cost)))
                                    if available_shares > 0:
                                        actual_cost = available_shares * current_prices[sec_code] * (1 + self.rebalance_cost)
                                        capital -= actual_cost
                                        # 更新目标份额
                                        new_positions[sec_code] = old_shares + available_shares
                                        self.trades_history.append({
                                            'date': rebalance_date,
                                            'sec_code': sec_code,
                                            'action': 'buy',
                                            'shares': available_shares,
                                            'price': current_prices[sec_code]
                                        })
                            elif new_shares < old_shares:
                                sell_shares = old_shares - new_shares
                                sell_value = sell_shares * current_prices[sec_code]
                                capital += sell_value * (1 - self.rebalance_cost)
                                self.trades_history.append({
                                    'date': rebalance_date,
                                    'sec_code': sec_code,
                                    'action': 'sell',
                                    'shares': sell_shares,
                                    'price': current_prices[sec_code]
                                })
                        
                        current_positions = new_positions
        
        # 计算指标
        metrics = self._calculate_metrics()
        
        if test_start and test_end:
            print(f"测试期: {test_start.strftime('%Y-%m-%d')} 至 {test_end.strftime('%Y-%m-%d')}")
        
        return metrics
    
    def _get_weekly_dates(self, dates, freq='biweekly'):
        """
        将日期按周分组
        
        Parameters:
            dates: 日期列表
            freq: 'weekly' 或 'biweekly'（双周）
        """
        if len(dates) == 0:
            return []
        
        dates = sorted([pd.to_datetime(d) for d in dates])
        weeks = []
        current_week = []
        current_week_start = dates[0] - timedelta(days=dates[0].weekday())
        
        for d in dates:
            week_start = d - timedelta(days=d.weekday())
            if week_start != current_week_start:
                if len(current_week) > 0:
                    weeks.append(current_week)
                current_week = [d]
                current_week_start = week_start
            else:
                current_week.append(d)
        
        if len(current_week) > 0:
            weeks.append(current_week)
        
        # 如果是双周调仓，只保留奇数周（或偶数周）
        if freq == 'biweekly':
            weeks = [w for i, w in enumerate(weeks) if i % 2 == 0]
        
        return weeks
    
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
        
        # 分离训练期和测试期
        train_df = df[~df['is_test']].copy() if 'is_test' in df.columns else df.copy()
        test_df = df[df['is_test']].copy() if 'is_test' in df.columns else pd.DataFrame()
        
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
        
        # 测试期单独指标
        if len(test_df) > 0:
            test_days = (test_df.index[-1] - test_df.index[0]).days
            test_annual_return = (1 + test_df['cum_return'].iloc[-1]) ** (365 / max(test_days, 1)) - 1
            test_annual_vol = test_df['return'].std() * np.sqrt(52)
            test_sharpe = (test_annual_return - risk_free) / test_annual_vol if test_annual_vol > 0 else 0
            
            test_df['cummax'] = test_df['value'].cummax()
            test_df['drawdown'] = (test_df['value'] - test_df['cummax']) / test_df['cummax']
            test_max_dd = test_df['drawdown'].min()
            test_calmar = test_annual_return / abs(test_max_dd) if test_max_dd != 0 else 0
            test_win_rate = (test_df['return'] > 0).sum() / len(test_df['return'].dropna()) if len(test_df['return'].dropna()) > 0 else 0
            
            metrics['test_return'] = test_df['cum_return'].iloc[-1]
            metrics['test_annual_return'] = test_annual_return
            metrics['test_annual_volatility'] = test_annual_vol
            metrics['test_sharpe_ratio'] = test_sharpe
            metrics['test_max_drawdown'] = test_max_dd
            metrics['test_calmar_ratio'] = test_calmar
            metrics['test_win_rate'] = test_win_rate
            metrics['test_portfolio_df'] = test_df
        
        return metrics
    
    def get_trades_summary(self):
        """获取交易汇总"""
        if len(self.trades_history) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades_history)
        df['date'] = pd.to_datetime(df['date'])
        return df


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
