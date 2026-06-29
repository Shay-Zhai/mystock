"""
周级别ETF量化策略 - 主程序入口
- 简单动量策略
- 回测运行与结果展示
"""

import pandas as pd
from data import get_etf_hist, get_multiple_etf_hist
from strategy import SimpleMomentumStrategy
from backtest import Backtest, calculate_benchmark_return
from visualization import plot_backtest_results, print_metrics


# ========== 策略配置 ==========

# ETF池子配置
ETF_POOL = {
    # 宽基ETF
    '宽基': [
        '510300',  # 沪深300ETF
        # '510500',  # 中证500ETF
        '512100',  # 中证1000ETF
        '159915',  # 创业板ETF
        # '588000',  # 科创50ETF
    ],
    # 行业ETF
    '行业': [
        '159928',  # 消费ETF
        '512660',  # 军工ETF
        '512880',  # 证券ETF
        '512010',  # 医药ETF
        '512760',  # 半导体ETF
        # '515790',  # 光伏ETF
        '512400',  # 有色金属ETF
        '512800',  # 银行ETF
        '515050',  # 通信ETF
    ],
    # 跨境ETF
    '跨境': [
        '513500',  # 标普500ETF
        '513100',  # 纳斯达克ETF
    ],
    # 商品ETF
    '商品': [
        '518880',  # 黄金ETF
    ],
}

# ETF代码到缩写映射
ETF_NAMES = {
    '510300': '沪深300', 
    '510500': '中证500', 
    '512100': '中证1000',
    '159915': '创业板', 
    '588000': '科创50',
    '159928': '消费', 
    '512660': '军工', 
    '512880': '证券', 
    '512010': '医药',
    '512760': '半导体', 
    '515790': '光伏', 
    '512400': '有色',
    '512800': '银行', 
    '515050': '通信',
    '513500': '标普500',
    '513100': '纳斯达克',
    '518880': '黄金',
}

# 回测参数配置
BACKTEST_CONFIG = {
    'start_date': '2018-01-01',
    'end_date': '2025-06-30',
    'initial_capital': 1000000,
    'rebalance_cost': 0.0005,  # 0.05%手续费
    # 滑点：按ETF设置（收盘价估计+收盘成交模拟实盘，假设资金量100-1000万）
    # 宽基/黄金流动性极好(0.05%)；头部行业流动性好(0.10%)；尾部行业中等(0.15%)；跨境有溢价问题(0.20%)
    'slippage': {
        '_default': 0.001,      # 默认0.1%
        # 宽基ETF（流动性极好，日均成交10-20亿）
        '510300': 0.0005,       # 沪深300
        '512100': 0.0005,       # 中证1000
        '159915': 0.0005,       # 创业板
        # 头部行业ETF（流动性好，日均成交2-5亿）
        '512880': 0.001,        # 证券
        '512760': 0.001,        # 半导体
        '512660': 0.001,        # 军工
        # 尾部行业ETF（流动性中等，日均成交0.5-2亿）
        '159928': 0.0015,       # 消费
        '512010': 0.0015,       # 医药
        '512400': 0.0015,       # 有色
        '512800': 0.0015,       # 银行
        '515050': 0.0015,       # 通信
        # 跨境ETF（有溢价/折价问题，流动性较差）
        '513500': 0.002,        # 标普500
        '513100': 0.002,        # 纳斯达克
        # 商品ETF（流动性极好，日均5亿+）
        '518880': 0.0005,       # 黄金
    },
    'rebalance_period': 20,  # 每20个交易日调仓
    # 调仓策略
    # method: 'direct' 直接调仓 | 'threshold' 阈值调仓 | 'band' 带宽调仓
    'rebalance_method': 'band',       # 调仓方法（band带宽调仓：逐标的判断，偏离>带宽才调整该标的）
    'rebalance_threshold': 0.10,      # 带宽10%（实测最优：年化+0.62pp，交易数-44.6%，费用-5pp）
    # 止损控制开关
    # mode: 'none' 不设止损 | 'individual' 仅单票止损 | 'portfolio' 仅整体止损 | 'both' 单票+整体止损
    'stop_loss': {
        'mode': 'individual',                    # 止损模式
        'individual_drawdown_pct': 0.15,         # 单票止损：从持仓最高价回撤超15%止损
        'portfolio_dd_half': 0.18,               # 整体止损：回撤超18%每个标的减半
        'portfolio_dd_clear': 0.25,              # 整体止损：回撤超25%清仓休息
    }
}

# 简单动量策略配置（优化版）
SIMPLE_STRATEGY_CONFIG = {
    'n_portfolio': 6,
    'momentum_lookback': 12,
    'volatility_lookback': 12,
    'momentum_threshold': 0.0,    # 只选择动量正值的ETF
    'trend_ma': 20,               # 趋势均线周期
    'max_volatility': 0.05,       # 最大波动率阈值（周波动率5%）
    'use_multi_momentum': True,  # 是否启用多期限动量合成
    'momentum_mode': 'vol_adjusted',       # 动量模式: 'raw'(原始), 'vol_adjusted'(波动率调整), 'downside_adjusted'(下行偏差调整)
}

# 6月动量策略配置（126日动量，跳10日，20日均线过滤）—— bug修复后重新验证最优组合
SKIP_MOMENTUM_STRATEGY_CONFIG = {
    'n_portfolio': 6,
    'momentum_lookback': 126,     # 6个月（126交易日）
    'momentum_skip': 10,          # 跳过最近2周（10交易日），规避短期反转
    'volatility_lookback': 21,    # 波动率回看周期
    'momentum_threshold': 0.0,
    'trend_ma': 20,               # 20日均线趋势过滤
    'max_volatility': 0.05,
    'use_multi_momentum': False,  # 单期限动量
    'momentum_mode': 'vol_adjusted',
    'weight_method': 'inv_vol_momentum',  # 动量×波动率倒数（bug修复后实测：夏普0.54最高，回撤-20.97%最低）
    'max_weight': 1.0,            # 不设上限（bug修复后上限40%无回撤控制效果，反而损失年化1.66pp）
}

# 6-1动量策略配置（126日动量，跳21日/1个月，20日均线过滤）—— 经典跳月动量
SKIP_MOMENTUM_21D_STRATEGY_CONFIG = {
    'n_portfolio': 6,
    'momentum_lookback': 126,     # 6个月（126交易日）
    'momentum_skip': 21,          # 跳过最近1个月（21交易日），经典跳月动量
    'volatility_lookback': 21,    # 波动率回看周期
    'momentum_threshold': 0.0,
    'trend_ma': 20,               # 20日均线趋势过滤
    'max_volatility': 0.05,
    'use_multi_momentum': False,  # 单期限动量
    'momentum_mode': 'vol_adjusted',
}


# ========== 主程序 ==========

def run_backtest(etf_codes=None, strategy_config=None, 
                 backtest_config=None, use_market_timing=False, market_timing_config=None,
                 summary_period='QE'):
    """
    运行回测
    
    Parameters:
        etf_codes: ETF代码列表（可选）
        strategy_config: 策略参数（可选）
        backtest_config: 回测参数（可选）
        use_market_timing: 是否使用市场择时（根据上证指数调整仓位）
        market_timing_config: 市场择时参数（可选）
        summary_period: 周期总结频率，'ME'为月，'QE'为季，'YE'为年（默认月）
    
    Returns:
        dict: 回测结果
    """
    # 使用默认配置
    if etf_codes is None:
        etf_codes = []
        for category in ETF_POOL.values():
            etf_codes.extend(category)
    
    if backtest_config is None:
        backtest_config = BACKTEST_CONFIG.copy()
    
    # 使用简单动量策略
    if strategy_config is None:
        strategy_config = SKIP_MOMENTUM_STRATEGY_CONFIG.copy()
    strategy = SimpleMomentumStrategy(**strategy_config)
    strategy_name = "简单动量策略"
    
    # 市场择时
    # market_timing = None
    # if use_market_timing:
    #     from strategy import MarketTiming
    #     if market_timing_config is None:
    #         market_timing_config = {}
    #     market_timing = MarketTiming(**market_timing_config)
    #     strategy_name += "（市场择时）"
    
    print("="*50)
    print(f"周级别ETF量化策略 - {strategy_name}")
    print("="*50)
    
    
    print(f"\n获取ETF数据...")
    print(f"ETF数量: {len(etf_codes)}")
    
    # 获取数据
    price_data_dict = get_multiple_etf_hist(
        etf_codes,
        backtest_config['start_date'],
        backtest_config['end_date'],
        min_length=50
    )
    
    print(f"\n成功获取 {len(price_data_dict)} 个ETF数据")
    
    if len(price_data_dict) < 2:
        print("数据不足，无法运行回测")
        return None
    
    # 运行回测
    print("\n运行回测...")
    backtest = Backtest(
        initial_capital=backtest_config['initial_capital'],
        rebalance_cost=backtest_config['rebalance_cost'],
        slippage=backtest_config.get('slippage', 0.001),
        stop_loss_config=backtest_config.get('stop_loss', {'mode': 'none'}),
        rebalance_method=backtest_config.get('rebalance_method', 'direct'),
        rebalance_threshold=backtest_config.get('rebalance_threshold', 0.05)
    )
    metrics = backtest.run(
        price_data_dict,
        strategy,
        pd.to_datetime(backtest_config['start_date']),
        pd.to_datetime(backtest_config['end_date']),
        rebalance_period=backtest_config.get('rebalance_period', 10),
        etf_names=ETF_NAMES
    )
    
    # 计算基准收益
    print("\n计算基准收益...")
    
    # 沪深300ETF基准（大盘代表）
    hs300_df = get_etf_hist('510300', backtest_config['start_date'], backtest_config['end_date'])
    hs300_return = calculate_benchmark_return(
        hs300_df,
        pd.to_datetime(backtest_config['start_date']),
        pd.to_datetime(backtest_config['end_date'])
    )
    print(f"基准（沪深300ETF）总收益: {hs300_return:.2f}%")
    
    # 中证500ETF基准（中盘代表）
    zz500_df = get_etf_hist('510500', backtest_config['start_date'], backtest_config['end_date'])
    zz500_return = calculate_benchmark_return(
        zz500_df,
        pd.to_datetime(backtest_config['start_date']),
        pd.to_datetime(backtest_config['end_date'])
    )
    print(f"基准（中证500ETF）总收益: {zz500_return:.2f}%")
    
    # 计算周期总结
    period_summary = calculate_period_summary(metrics, hs300_df, zz500_df, summary_period)
    
    # 保存调仓记录
    backtest.save_rebalance_records()
    
    # 绘制结果
    if len(metrics) > 0:
        plot_backtest_results(metrics, benchmark_data={'hs300': hs300_df, 'zz500': zz500_df},
                             period_summary=period_summary)
        print_metrics(metrics, hs300_return, zz500_return, period_summary)
        
        # 打印策略摘要
        print("\n" + "="*50)
        print("策略摘要")
        print("="*50)
        print(strategy.get_training_summary())
    else:
        print("回测失败")
    
    return metrics


def calculate_period_summary(metrics, hs300_df, zz500_df, period='M'):
    """
    计算周期总结（策略收益、基准收益、超额收益、调仓金额、费用率）

    Parameters:
        metrics: 回测指标
        hs300_df: 沪深300ETF数据（大盘基准）
        zz500_df: 中证500ETF数据（中盘基准）
        period: 周期频率，'ME'月, 'QE'季, 'YE'年

    Returns:
        DataFrame: 周期总结数据
    """
    portfolio_df = metrics.get('portfolio_df')
    if portfolio_df is None or len(portfolio_df) == 0:
        return pd.DataFrame()

    # 策略周期收益：按周期取末尾净值，计算收益率
    strategy_period = portfolio_df['value'].resample(period).last()
    strategy_return = strategy_period.pct_change() * 100

    # 基准周期收益
    hs300_period = hs300_df['close'].resample(period).last() if hs300_df is not None and len(hs300_df) > 0 else None
    zz500_period = zz500_df['close'].resample(period).last() if zz500_df is not None and len(zz500_df) > 0 else None
    hs300_return = hs300_period.pct_change() * 100 if hs300_period is not None else None
    zz500_return = zz500_period.pct_change() * 100 if zz500_period is not None else None

    summary = pd.DataFrame(index=strategy_return.index)
    summary['strategy_return'] = strategy_return
    if hs300_return is not None:
        summary['hs300_return'] = hs300_return
        summary['excess_hs300'] = summary['strategy_return'] - summary['hs300_return']
    if zz500_return is not None:
        summary['zz500_return'] = zz500_return
        summary['excess_zz500'] = summary['strategy_return'] - summary['zz500_return']
    
    # 周期调仓金额和费用（基于交易记录）
    trades_history = metrics.get('trades_history', [])
    if len(trades_history) > 0:
        trades_df = pd.DataFrame(trades_history)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        trades_df['turnover'] = trades_df['turnover'].astype(float)
        trades_df['cost'] = trades_df['cost'].astype(float)
        
        # 按周期汇总：卖出额和买入额
        trades_df['period'] = trades_df['date'].dt.to_period(period.replace('ME','M').replace('QE','Q').replace('YE','Y'))
        # 分别汇总卖出和买入
        sells = trades_df[trades_df['action']=='sell'].groupby('period')['turnover'].sum()
        buys = trades_df[trades_df['action']=='buy'].groupby('period')['turnover'].sum()
        costs = trades_df.groupby('period')['cost'].sum()
        
        # 调仓金额 = (卖出额 + 买入额) / 2
        turnover_by_period = ((sells.reindex(costs.index, fill_value=0) + 
                               buys.reindex(costs.index, fill_value=0)) / 2)
        
        # 对齐到 summary 索引
        turnover_series = pd.Series(index=summary.index, dtype=float)
        cost_series = pd.Series(index=summary.index, dtype=float)
        for p, tv in turnover_by_period.items():
            # 将 period 转为时间戳（周期末）
            ts = p.to_timestamp(how='end')
            # 找到 summary 中匹配的索引
            mask = summary.index == ts
            if mask.any():
                turnover_series.loc[ts] = tv
                cost_series.loc[ts] = costs.get(p, 0)
            else:
                # 找最近的
                diff = abs((summary.index - ts).days)
                idx_pos = diff.argmin()
                turnover_series.iloc[idx_pos] = tv
                cost_series.iloc[idx_pos] = costs.get(p, 0)
        
        summary['turnover'] = turnover_series.fillna(0)
        summary['total_cost'] = cost_series.fillna(0)
        # 费用率 = 周期总费用 / 周期初组合价值
        period_start_value = portfolio_df['value'].resample(period).first()
        cost_rate = (summary['total_cost'] / period_start_value.reindex(summary.index) * 100).fillna(0)
        summary['cost_rate'] = cost_rate
    
    summary = summary.dropna(subset=['strategy_return'])
    return summary


def main():
    return run_backtest()


if __name__ == "__main__":
    main()
