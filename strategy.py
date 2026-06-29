"""
策略核心模块
"""

import numpy as np
import pandas as pd


class SimpleMomentumStrategy:
    """
    简单动量策略（优化版）
    基于过去N周收益率选股，添加趋势过滤和动量阈值
    支持多期限动量合成（可选）
    支持三种动量调整模式：原始、波动率调整、下行偏差调整
    """
    
    def __init__(self, n_portfolio=5, momentum_lookback=12, volatility_lookback=12,
                 momentum_threshold=0.0, trend_ma=20, max_volatility=0.05,
                 use_multi_momentum=False, momentum_mode='raw', momentum_skip=0,
                 weight_method='inv_vol', min_volatility=0.0, max_weight=1.0):
        """
        Parameters:
            n_portfolio: 持仓ETF数量
            momentum_lookback: 动量回看周期数（单期限模式使用）
            volatility_lookback: 波动率回看周期数（用于权重计算）
            momentum_threshold: 动量阈值，只选择动量大于此值的ETF（默认0，即只选上涨的）
            trend_ma: 趋势均线周期（用于趋势过滤）
            max_volatility: 最大波动率阈值，超过此值的ETF不选
            use_multi_momentum: 是否启用多期限动量合成（20日40% + 60日40% + 120日20%）
            momentum_mode: 动量模式
                - 'raw': 原始动量
                - 'vol_adjusted': 波动率调整动量（动量/波动率）
                - 'downside_adjusted': 下行偏差调整动量（动量/下行偏差）
            momentum_skip: 动量跳过周期数（去除最近N期，避免短期反转效应，0=不跳过）
            weight_method: 权重分配方法
                - 'equal': 等权
                - 'inv_vol': 波动率倒数（风险平价简化版，默认）
                - 'momentum': 动量加权（动量值归一化）
                - 'inv_vol_momentum': 动量×波动率倒数（综合风险与动量信号）
            min_volatility: 波动率地板（inv_vol权重时，volatility取max(vol, min_volatility)），
                           用于避免低波动资产（如国债）被分配过大权重。0=不启用
            max_weight: 单标的权重上限（归一化后截断+重分配）。1.0=不限制
        """
        self.n_portfolio = n_portfolio
        self.momentum_lookback = momentum_lookback
        self.volatility_lookback = volatility_lookback
        self.momentum_threshold = momentum_threshold
        self.trend_ma = trend_ma
        self.max_volatility = max_volatility
        self.use_multi_momentum = use_multi_momentum
        self.momentum_mode = momentum_mode
        self.momentum_skip = momentum_skip
        self.weight_method = weight_method
        self.min_volatility = min_volatility
        self.max_weight = max_weight
    
    def _calculate_raw_momentum(self, prices):
        """计算原始动量"""
        if self.use_multi_momentum:
            # 多期限动量合成：20日(40%) + 60日(40%) + 120日(20%)
            momentum_20 = (prices.iloc[-1] / prices.iloc[-20]) - 1 if len(prices) >= 20 else np.nan
            momentum_60 = (prices.iloc[-1] / prices.iloc[-60]) - 1 if len(prices) >= 60 else np.nan
            momentum_120 = (prices.iloc[-1] / prices.iloc[-120]) - 1 if len(prices) >= 120 else np.nan
            
            if not np.isnan(momentum_20) and not np.isnan(momentum_60) and not np.isnan(momentum_120):
                return momentum_20 * 0.4 + momentum_60 * 0.4 + momentum_120 * 0.2
            elif not np.isnan(momentum_60):
                return momentum_60
            elif not np.isnan(momentum_20):
                return momentum_20
            else:
                return np.nan
        else:
            # 单期限动量（支持跳过最近N期，避免短期反转效应）
            skip = self.momentum_skip
            need = self.momentum_lookback + skip + 1
            if len(prices) < need:
                return np.nan
            # 动量 = prices[-(skip+1)] / prices[-(skip+lookback+1)] - 1
            end_price = prices.iloc[-(skip + 1)]
            start_price = prices.iloc[-(skip + self.momentum_lookback + 1)]
            return (end_price / start_price) - 1
    
    def _calculate_volatility(self, prices, lookback=None):
        """计算波动率"""
        if lookback is None:
            lookback = self.volatility_lookback
        if len(prices) < lookback + 1:
            return np.nan
        returns = prices.pct_change().dropna().tail(lookback)
        return returns.std()
    
    def _calculate_downside_deviation(self, prices, lookback=60):
        """计算下行偏差（只考虑负收益）"""
        if len(prices) < lookback + 1:
            return np.nan
        returns = prices.pct_change().dropna().tail(lookback)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return 0  # 没有负收益，风险低
        
        return negative_returns.std()
    
    def _calculate_momentum(self, prices):
        """计算动量（根据模式调整）"""
        raw_momentum = self._calculate_raw_momentum(prices)
        
        if np.isnan(raw_momentum):
            return np.nan, np.nan, np.nan  # raw, volatility, downside
        
        # 计算波动率和下行偏差
        volatility = self._calculate_volatility(prices)
        downside_dev = self._calculate_downside_deviation(prices)
        
        # 根据模式返回调整后的动量
        if self.momentum_mode == 'raw':
            adjusted_momentum = raw_momentum
        elif self.momentum_mode == 'vol_adjusted':
            # 波动率调整动量
            if not np.isnan(volatility) and volatility > 0:
                adjusted_momentum = raw_momentum / volatility
            else:
                adjusted_momentum = raw_momentum
        elif self.momentum_mode == 'downside_adjusted':
            # 下行偏差调整动量
            if not np.isnan(downside_dev) and downside_dev > 0:
                adjusted_momentum = raw_momentum / downside_dev
            else:
                adjusted_momentum = raw_momentum
        else:
            adjusted_momentum = raw_momentum
        
        return adjusted_momentum, volatility, downside_dev
    
    def _calculate_trend(self, prices):
        """计算趋势强度（价格相对于均线）"""
        if len(prices) < self.trend_ma:
            return np.nan
        ma = prices.tail(self.trend_ma).mean()
        return (prices.iloc[-1] / ma) - 1
    
    def calculate_factors(self, price_data_dict):
        """计算各ETF的动量、波动率和趋势"""
        results = []
        if self.use_multi_momentum:
            min_lookback = 120
        else:
            min_lookback = max(self.momentum_lookback + self.momentum_skip + 1, self.trend_ma)
        
        for sec_code, df in price_data_dict.items():
            if len(df) < min_lookback:
                continue
            
            momentum, volatility, downside_dev = self._calculate_momentum(df['close'])
            trend = self._calculate_trend(df['close'])
            
            if np.isnan(momentum):
                continue
            
            results.append({
                'sec_code': sec_code,
                'close': df['close'].iloc[-1],
                'momentum': momentum,
                'volatility': volatility if not np.isnan(volatility) else 0.02,
                'downside_deviation': downside_dev if not np.isnan(downside_dev) else 0.01,
                'trend': trend if not np.isnan(trend) else 0
            })
        return pd.DataFrame(results)
    
    def select_etfs(self, factor_df):
        """选股：动量排序 + 趋势过滤 + 波动率上限"""
        if len(factor_df) == 0:
            return []
        
        df = factor_df.copy()
        
        # 根据动量模式决定排序指标
        if self.momentum_mode == 'raw':
            # 原始动量模式：计算风险调整动量（动量/波动率）
            df['sort_metric'] = df['momentum'] / df['volatility'].replace(0, np.nan)
            df['sort_metric'] = df['sort_metric'].fillna(0)
        else:
            # 已调整动量模式：直接使用调整后的动量排序
            df['sort_metric'] = df['momentum']
        
        # 1. 趋势过滤：只选择趋势向上的ETF（价格高于均线）
        df = df[df['trend'] > 0]
        
        # 2. 波动率过滤：避免波动率过高的ETF
        df = df[df['volatility'] <= self.max_volatility]
        
        # 3. 动量阈值过滤：只选择动量大于阈值的ETF
        df = df[df['momentum'] > self.momentum_threshold]
        
        # 如果过滤后ETF不足，放宽条件
        if len(df) < self.n_portfolio:
            df_filtered = factor_df[factor_df['trend'] > 0].copy()
            if self.momentum_mode == 'raw':
                df_filtered['sort_metric'] = df_filtered['momentum'] / df_filtered['volatility'].replace(0, np.nan)
                df_filtered['sort_metric'] = df_filtered['sort_metric'].fillna(0)
            else:
                df_filtered['sort_metric'] = df_filtered['momentum']
            df_filtered = df_filtered[df_filtered['momentum'] > self.momentum_threshold]
            if len(df_filtered) >= self.n_portfolio:
                df = df_filtered
        
        if len(df) < self.n_portfolio:
            df_filtered = factor_df[factor_df['trend'] > 0].copy()
            if self.momentum_mode == 'raw':
                df_filtered['sort_metric'] = df_filtered['momentum'] / df_filtered['volatility'].replace(0, np.nan)
                df_filtered['sort_metric'] = df_filtered['sort_metric'].fillna(0)
            else:
                df_filtered['sort_metric'] = df_filtered['momentum']
            if len(df_filtered) >= self.n_portfolio:
                df = df_filtered
        
        if len(df) == 0:
            return []
        
        # 按排序指标排序（加入sec_code作为第二排序键确保稳定性）
        df = df.sort_values(['sort_metric', 'sec_code'], ascending=[False, True])
        
        selected = []
        for _, row in df.head(self.n_portfolio).iterrows():
            selected.append(row.to_dict())
        return selected
    
    def calculate_weights(self, selected_etfs):
        """权重分配（根据 weight_method）"""
        if len(selected_etfs) == 0:
            return {}

        n = len(selected_etfs)
        method = self.weight_method

        if method == 'equal':
            # 等权
            raw = [1.0] * n
        elif method == 'inv_vol':
            # 波动率倒数（风险平价简化版）
            # min_volatility地板：避免低波动资产（如国债）权重过大
            vols = [max(etf['volatility'], self.min_volatility, 1e-6) for etf in selected_etfs]
            raw = [1 / v for v in vols]
        elif method == 'momentum':
            # 动量加权：动量值归一化（动量须为正，负值截断为0）
            moms = [max(etf['momentum'], 0) for etf in selected_etfs]
            if sum(moms) <= 0:
                raw = [1.0] * n  # 全为0时退化为等权
            else:
                raw = moms
        elif method == 'inv_vol_momentum':
            # 动量×波动率倒数（综合风险与动量信号）
            vols = [max(etf['volatility'], self.min_volatility, 1e-6) for etf in selected_etfs]
            moms = [max(etf['momentum'], 0) for etf in selected_etfs]
            raw = [m / v for m, v in zip(moms, vols)]
            if sum(raw) <= 0:
                raw = [1 / v for v in vols]  # 退化为波动率倒数
        else:
            # 默认波动率倒数
            vols = [max(etf['volatility'], self.min_volatility, 1e-6) for etf in selected_etfs]
            raw = [1 / v for v in vols]

        total = sum(raw)
        if total <= 0:
            weights = [1.0 / n] * n
        else:
            weights = [w / total for w in raw]

        # 单标的权重上限（归一化后截断+重分配）
        if self.max_weight < 1.0:
            for _ in range(10):  # 迭代截断重分配，最多10次收敛
                over = [i for i, w in enumerate(weights) if w > self.max_weight]
                if not over:
                    break
                excess = sum(weights[i] - self.max_weight for i in over)
                for i in over:
                    weights[i] = self.max_weight
                under = [i for i in range(n) if weights[i] < self.max_weight]
                if under:
                    total_under = sum(weights[i] for i in under)
                    if total_under > 0:
                        for i in under:
                            weights[i] += excess * weights[i] / total_under
        return {selected_etfs[i]['sec_code']: weights[i] for i in range(n)}
    
    def generate_signals(self, price_data_dict):
        """生成交易信号"""
        factor_df = self.calculate_factors(price_data_dict)
        selected = self.select_etfs(factor_df)
        weights = self.calculate_weights(selected)
        return weights
    
    def record_training_data(self, price_data_dict, rebalance_date):
        """简单策略不需要训练数据"""
        pass
    
    def train(self):
        """简单策略无需训练"""
        print("简单动量策略无需训练")
    
    def get_training_summary(self):
        """获取策略摘要"""
        skip_info = f"\n动量跳过: {self.momentum_skip}日" if self.momentum_skip > 0 else ""
        return f"简单动量策略（优化版）\n动量模式: {self.momentum_mode}\n动量回看: {self.momentum_lookback}日{skip_info}\n动量阈值: {self.momentum_threshold*100:.1f}%\n趋势均线: {self.trend_ma}日\n最大波动率: {self.max_volatility*100:.1f}%\n持仓数量: {self.n_portfolio}\n权重方法: {self.weight_method}"
