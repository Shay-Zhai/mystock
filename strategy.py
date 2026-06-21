"""
策略核心模块
- 多因子选股
- 因子IC分析
- 训练/测试分离
- 权重计算
"""

import numpy as np
import pandas as pd
from factors import calc_all_factors


def _spearmanr(x, y):
    """计算Spearman相关系数（纯numpy实现，无需scipy）"""
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(df) < 3:
        return 0, 1
    
    # 手动计算排名
    def rankdata(a):
        """计算排名（平均排名处理并列）"""
        sorter = np.argsort(a)
        inv = np.empty(len(a), dtype=int)
        inv[sorter] = np.arange(len(a))
        # 处理并列
        sorted_a = a[sorter]
        # 找到并列的位置
        obs = np.concatenate(([True], sorted_a[1:] != sorted_a[:-1], [True]))
        dense = np.cumsum(obs[:-1])
        # 平均排名
        count = np.concatenate((np.nonzero(obs)[0], [len(a)]))
        ranks = (count[dense] + count[dense + 1] + 1) / 2.0
        # 恢复原始顺序
        result = np.empty(len(a))
        result[inv] = ranks
        return result
    
    x_rank = rankdata(df['x'].values)
    y_rank = rankdata(df['y'].values)
    
    # 计算Pearson相关系数（排名后的）
    x_mean = np.mean(x_rank)
    y_mean = np.mean(y_rank)
    numerator = np.sum((x_rank - x_mean) * (y_rank - y_mean))
    denominator = np.sqrt(np.sum((x_rank - x_mean)**2)) * np.sqrt(np.sum((y_rank - y_mean)**2))
    
    if denominator == 0:
        return 0, 1
    
    corr = numerator / denominator
    
    # 简化的p值估计
    n = len(df)
    t = corr * np.sqrt((n - 2) / max(1e-10, (1 - corr**2)))
    # 使用正态近似
    from math import erf
    pvalue = max(0, min(1, 1 - erf(abs(t) / np.sqrt(2))))
    
    return corr, pvalue


class MultiFactorETFStrategy:
    """
    多因子ETF策略
    支持训练期因子IC分析和测试期选股
    """
    
    def __init__(self,
                 n_portfolio=5,
                 momentum_lookback=12,
                 vol_lookback=12,
                 trend_ma=20,
                 train_end_date='2024-12-31',
                 ic_threshold=0.05,
                 use_equal_weight=False):
        """
        Parameters:
            n_portfolio: 持仓ETF数量
            momentum_lookback: 动量回看周数
            vol_lookback: 波动率回看周数
            trend_ma: 趋势均线周期
            train_end_date: 训练期结束日期
            ic_threshold: IC绝对值阈值，低于此值的因子被忽略
            use_equal_weight: 是否使用等权（True）或IC加权（False）
        """
        self.n_portfolio = n_portfolio
        self.momentum_lookback = momentum_lookback
        self.vol_lookback = vol_lookback
        self.trend_ma = trend_ma
        self.train_end_date = pd.to_datetime(train_end_date)
        self.ic_threshold = ic_threshold
        self.use_equal_weight = use_equal_weight
        
        # 训练结果
        self.factor_weights = None  # 因子权重
        self.factor_directions = None  # 因子方向（1或-1）
        self.factor_ics = None  # 因子IC值
        self.is_trained = False
        
        # 训练期数据记录
        self.train_factor_records = []  # 因子值记录
        self.train_return_records = []  # 未来收益记录
    
    def _calculate_single_etf_factors(self, df):
        """计算单个ETF的所有因子"""
        if len(df) < max(self.momentum_lookback * 2, self.trend_ma):
            return None
        return calc_all_factors(df, self.momentum_lookback, self.trend_ma, self.vol_lookback)
    
    def record_training_data(self, price_data_dict, rebalance_date):
        """
        记录训练期数据
        在每个调仓日调用，记录因子值和下一周收益率
        """
        if pd.to_datetime(rebalance_date) > self.train_end_date:
            return
        
        # 计算各ETF的因子值（使用调仓日之前的数据）
        for sec_code, full_df in price_data_dict.items():
            # 过滤：只使用调仓日之前的数据计算因子
            df = full_df[full_df.index < rebalance_date]
            if len(df) < self.momentum_lookback + 2:
                continue
            
            factors = self._calculate_single_etf_factors(df)
            if factors is None:
                continue
            
            # 计算下一周收益率（作为标签）
            current_price = df['close'].iloc[-1]
            # 从完整数据中找到调仓日之后的价格
            future_df = full_df[full_df.index > rebalance_date]
            if len(future_df) >= 5:  # 假设一周5个交易日
                future_price = future_df['close'].iloc[4]  # 一周后价格
            elif len(future_df) > 0:
                future_price = future_df['close'].iloc[-1]  # 最后一个可用价格
            else:
                continue
            
            weekly_return = (future_price / current_price) - 1
            
            self.train_factor_records.append({
                'date': rebalance_date,
                'sec_code': sec_code,
                **factors,
                'future_return': weekly_return
            })
    
    def train(self):
        """
        训练：计算各因子的IC并确定权重
        在训练期结束后调用
        """
        if len(self.train_factor_records) == 0:
            print("警告：没有训练数据，使用默认等权")
            self._set_default_weights()
            return
        
        df = pd.DataFrame(self.train_factor_records)
        
        # 获取所有因子列（排除date, sec_code, future_return）
        factor_cols = [col for col in df.columns 
                      if col not in ['date', 'sec_code', 'future_return']]
        
        print(f"\n{'='*50}")
        print("因子IC分析（训练期）")
        print(f"{'='*50}")
        print(f"训练样本数: {len(df)}")
        print(f"调仓次数: {df['date'].nunique()}")
        print(f"ETF数量: {df['sec_code'].nunique()}")
        
        # 计算每个因子的IC
        ic_results = {}
        for factor in factor_cols:
            factor_values = df[factor].dropna()
            returns = df.loc[factor_values.index, 'future_return']
            
            if len(factor_values) < 10:
                ic_results[factor] = {'ic': 0, 'pvalue': 1}
                continue
            
            # 计算Spearman相关系数（IC）
            ic, pvalue = _spearmanr(factor_values, returns)
            if np.isnan(ic):
                ic = 0
                pvalue = 1
            
            ic_results[factor] = {'ic': ic, 'pvalue': pvalue}
        
        # 打印IC结果
        print(f"\n{'因子名称':<25} {'IC值':>8} {'P值':>8} {'状态':>8}")
        print("-" * 55)
        
        valid_factors = []
        for factor, result in sorted(ic_results.items(), key=lambda x: abs(x[1]['ic']), reverse=True):
            status = "有效" if abs(result['ic']) > self.ic_threshold else "忽略"
            if abs(result['ic']) > self.ic_threshold and result['pvalue'] < 0.1:
                valid_factors.append(factor)
            print(f"{factor:<25} {result['ic']:>8.4f} {result['pvalue']:>8.4f} {status:>8}")
        
        # 确定因子权重和方向
        self.factor_ics = ic_results
        self.factor_directions = {}
        self.factor_weights = {}
        
        if len(valid_factors) == 0:
            print("\n警告：没有有效因子，使用默认等权")
            self._set_default_weights()
            return
        
        print(f"\n有效因子数量: {len(valid_factors)}")
        
        if self.use_equal_weight:
            # 等权
            for factor in valid_factors:
                self.factor_directions[factor] = 1 if ic_results[factor]['ic'] > 0 else -1
                self.factor_weights[factor] = 1.0 / len(valid_factors)
        else:
            # IC加权：|IC|越大权重越高
            total_ic = sum(abs(ic_results[f]['ic']) for f in valid_factors)
            if total_ic > 0:
                for factor in valid_factors:
                    self.factor_directions[factor] = 1 if ic_results[factor]['ic'] > 0 else -1
                    self.factor_weights[factor] = abs(ic_results[factor]['ic']) / total_ic
        
        print(f"\n{'因子名称':<25} {'方向':>6} {'权重':>8}")
        print("-" * 45)
        for factor in sorted(self.factor_weights.keys(), key=lambda x: self.factor_weights[x], reverse=True):
            direction = "正向" if self.factor_directions[factor] > 0 else "反向"
            print(f"{factor:<25} {direction:>6} {self.factor_weights[factor]:>8.4f}")
        
        self.is_trained = True
    
    def _set_default_weights(self):
        """设置默认权重（当训练失败时使用）"""
        default_factors = ['momentum', 'trend', 'sharpe_ratio']
        self.factor_directions = {f: 1 for f in default_factors}
        self.factor_weights = {f: 1.0/len(default_factors) for f in default_factors}
        self.is_trained = True
    
    def calculate_factors(self, price_data_dict):
        """
        计算所有ETF的因子值（用于选股）
        
        Returns:
            DataFrame: 包含sec_code和各因子值的DataFrame
        """
        results = []
        
        for sec_code, df in price_data_dict.items():
            factors = self._calculate_single_etf_factors(df)
            if factors is None:
                continue
            
            results.append({
                'sec_code': sec_code,
                'close': df['close'].iloc[-1],
                **factors
            })
        
        return pd.DataFrame(results)
    
    def calculate_composite_score(self, factor_df):
        """
        计算综合因子得分
        
        Parameters:
            factor_df: 包含各因子值的DataFrame
        
        Returns:
            DataFrame: 添加composite_score列
        """
        if not self.is_trained:
            raise ValueError("策略未训练，请先调用train()")
        
        df = factor_df.copy()
        
        # 初始化综合得分
        df['composite_score'] = 0
        
        for factor, weight in self.factor_weights.items():
            if factor not in df.columns:
                continue
            
            # 因子标准化（Z-score）
            factor_mean = df[factor].mean()
            factor_std = df[factor].std()
            if factor_std > 0 and not np.isnan(factor_std):
                normalized = (df[factor] - factor_mean) / factor_std
            else:
                normalized = 0
            
            # 应用因子方向并累加
            df['composite_score'] += normalized * weight * self.factor_directions[factor]
        
        return df
    
    def select_etfs(self, factor_df):
        """
        选股逻辑：综合因子得分排序 + 趋势过滤
        
        Parameters:
            factor_df: 因子值表
        
        Returns:
            list: 选中的ETF列表
        """
        if len(factor_df) == 0:
            return []
        
        # 计算综合得分
        df = self.calculate_composite_score(factor_df)
        
        # 趋势过滤 - 只选择上涨趋势的ETF
        uptrend = df[df['trend'] > -0.05].copy()  # 放宽趋势条件
        if len(uptrend) < self.n_portfolio:
            uptrend = df.copy()
        
        # 按综合得分排序（高分优先）
        uptrend = uptrend.sort_values('composite_score', ascending=False)
        
        # 选择前N个
        selected = []
        for _, row in uptrend.iterrows():
            if len(selected) >= self.n_portfolio:
                break
            selected.append(row.to_dict())
        
        return selected
    
    def calculate_weights(self, selected_etfs):
        """
        计算权重: 风险平价
        
        Parameters:
            selected_etfs: 选中的ETF列表
        
        Returns:
            dict: {sec_code: weight}
        """
        if len(selected_etfs) == 0:
            return {}
        
        df = pd.DataFrame(selected_etfs)
        
        # 波动率倒数权重
        vols = df['volatility'].values
        inv_vol = 1 / np.maximum(vols, 1e-6)
        
        # 如果有无效的波动率，使用等权
        if np.any(np.isnan(inv_vol)) or np.any(inv_vol <= 0):
            weights = np.ones(len(selected_etfs)) / len(selected_etfs)
        else:
            weights = inv_vol / inv_vol.sum()
        
        # 确保权重和为1
        weights = weights / weights.sum()
        
        return {selected_etfs[i]['sec_code']: weights[i] for i in range(len(selected_etfs))}
    
    def generate_signals(self, price_data_dict):
        """生成交易信号"""
        factor_df = self.calculate_factors(price_data_dict)
        selected = self.select_etfs(factor_df)
        weights = self.calculate_weights(selected)
        return weights
    
    def get_training_summary(self):
        """获取训练摘要"""
        if not self.is_trained:
            return "策略未训练"
        
        summary = []
        summary.append(f"训练状态: 已完成")
        summary.append(f"有效因子数: {len(self.factor_weights)}")
        summary.append(f"IC阈值: {self.ic_threshold}")
        summary.append(f"权重方式: {'等权' if self.use_equal_weight else 'IC加权'}")
        
        if self.factor_ics:
            summary.append(f"\n因子IC详情:")
            for factor, result in sorted(self.factor_ics.items(), key=lambda x: abs(x[1]['ic']), reverse=True):
                summary.append(f"  {factor}: IC={result['ic']:.4f}, P={result['pvalue']:.4f}")
        
        return "\n".join(summary)


class SimpleMomentumStrategy:
    """
    简单动量策略（优化版）
    基于过去N周收益率选股，添加趋势过滤和动量阈值
    支持训练期设置（可选）
    """
    
    def __init__(self, n_portfolio=5, momentum_lookback=12, volatility_lookback=12,
                 momentum_threshold=0.0, trend_ma=20, max_volatility=0.05,
                 train_end_date=None):
        """
        Parameters:
            n_portfolio: 持仓ETF数量
            momentum_lookback: 动量回看周数
            volatility_lookback: 波动率回看周数（用于权重计算）
            momentum_threshold: 动量阈值，只选择动量大于此值的ETF（默认0，即只选上涨的）
            trend_ma: 趋势均线周期（用于趋势过滤）
            max_volatility: 最大波动率阈值，超过此值的ETF不选
            train_end_date: 训练期结束日期（None表示全程交易）
        """
        self.n_portfolio = n_portfolio
        self.momentum_lookback = momentum_lookback
        self.volatility_lookback = volatility_lookback
        self.momentum_threshold = momentum_threshold
        self.trend_ma = trend_ma
        self.max_volatility = max_volatility
        self.train_end_date = pd.to_datetime(train_end_date) if train_end_date else None
        self.is_trained = True
    
    def _calculate_momentum(self, prices):
        """计算动量（过去N周收益率）"""
        if len(prices) < self.momentum_lookback:
            return np.nan
        return (prices.iloc[-1] / prices.iloc[-self.momentum_lookback]) - 1
    
    def _calculate_volatility(self, prices):
        """计算波动率"""
        if len(prices) < self.volatility_lookback + 1:
            return np.nan
        returns = prices.pct_change().dropna().tail(self.volatility_lookback)
        return returns.std()
    
    def _calculate_trend(self, prices):
        """计算趋势强度（价格相对于均线）"""
        if len(prices) < self.trend_ma:
            return np.nan
        ma = prices.tail(self.trend_ma).mean()
        return (prices.iloc[-1] / ma) - 1
    
    def calculate_factors(self, price_data_dict):
        """计算各ETF的动量、波动率和趋势"""
        results = []
        for sec_code, df in price_data_dict.items():
            if len(df) < max(self.momentum_lookback, self.trend_ma):
                continue
            
            momentum = self._calculate_momentum(df['close'])
            volatility = self._calculate_volatility(df['close'])
            trend = self._calculate_trend(df['close'])
            
            if np.isnan(momentum):
                continue
            
            results.append({
                'sec_code': sec_code,
                'close': df['close'].iloc[-1],
                'momentum': momentum,
                'volatility': volatility if not np.isnan(volatility) else 0.02,
                'trend': trend if not np.isnan(trend) else 0
            })
        return pd.DataFrame(results)
    
    def select_etfs(self, factor_df):
        """选股：动量排序 + 趋势过滤 + 动量阈值（温和版）"""
        if len(factor_df) == 0:
            return []
        
        df = factor_df.copy()
        
        # 1. 动量阈值过滤：只选择动量大于阈值的ETF
        df_positive = df[df['momentum'] > self.momentum_threshold]
        
        # 2. 趋势过滤：只选择趋势向上的ETF（价格高于均线）
        df_trend = df_positive[df_positive['trend'] > 0]
        
        # 3. 波动率过滤：避免波动率过高的ETF
        df_final = df_trend[df_trend['volatility'] <= self.max_volatility]
        
        # 如果严格过滤后ETF数量足够，直接选择
        if len(df_final) >= self.n_portfolio:
            df_final = df_final.sort_values('momentum', ascending=False)
            selected = []
            for _, row in df_final.head(self.n_portfolio).iterrows():
                selected.append(row.to_dict())
            return selected
        
        # 如果不足，放宽波动率限制（保留趋势和动量过滤）
        if len(df_trend) >= self.n_portfolio:
            df_trend = df_trend.sort_values('momentum', ascending=False)
            selected = []
            for _, row in df_trend.head(self.n_portfolio).iterrows():
                selected.append(row.to_dict())
            return selected
        
        # 如果仍然不足，放宽趋势限制（只保留动量过滤）
        if len(df_positive) >= self.n_portfolio:
            df_positive = df_positive.sort_values('momentum', ascending=False)
            selected = []
            for _, row in df_positive.head(self.n_portfolio).iterrows():
                selected.append(row.to_dict())
            return selected
        
        # 如果动量正值的ETF不足，选择动量最高的几个（即使动量为负）
        df = df.sort_values('momentum', ascending=False)
        selected = []
        for _, row in df.head(self.n_portfolio).iterrows():
            selected.append(row.to_dict())
        return selected
    
    def calculate_weights(self, selected_etfs):
        """权重：波动率倒数（风险平价）"""
        if len(selected_etfs) == 0:
            return {}
        vols = [etf['volatility'] for etf in selected_etfs]
        inv_vol = [1 / max(v, 1e-6) for v in vols]
        weights = [w / sum(inv_vol) for w in inv_vol]
        return {selected_etfs[i]['sec_code']: weights[i] for i in range(len(selected_etfs))}
    
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
        return f"简单动量策略（优化版）\n动量回看: {self.momentum_lookback}周\n动量阈值: {self.momentum_threshold*100:.1f}%\n趋势均线: {self.trend_ma}周\n最大波动率: {self.max_volatility*100:.1f}%\n持仓数量: {self.n_portfolio}"