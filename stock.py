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
        '510500',  # 中证500ETF
        '512100',  # 中证1000ETF
        '159915',  # 创业板ETF
        '588000',  # 科创50ETF
    ],
    # 行业ETF
    '行业': [
        '512660',  # 军工ETF
        '512880',  # 证券ETF
        '512010',  # 医药ETF
        '512760',  # 半导体ETF
        '515790',  # 光伏ETF
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
    '510300': '沪深300', '510500': '中证500', '512100': '中证1000',
    '159915': '创业板', '588000': '科创50',
    '512660': '军工', '512880': '证券', '512010': '医药',
    '512760': '半导体', '515790': '光伏', '512400': '有色',
    '512800': '银行', '515050': '通信',
    '513500': '标普500', '513100': '纳斯达克',
    '518880': '黄金',
}

# 回测参数配置
BACKTEST_CONFIG = {
    'start_date': '2018-01-01',
    'end_date': '2025-06-30',
    'initial_capital': 1000000,
    'rebalance_cost': 0.0005,  # 0.05%手续费
    'slippage': 0.001,         # 0.1%滑点
    'rebalance_period': 20,  # 每10个交易日调仓
}

# 简单动量策略配置（优化版）
SIMPLE_STRATEGY_CONFIG = {
    'n_portfolio': 5,
    'momentum_lookback': 12,
    'volatility_lookback': 12,
    'momentum_threshold': 0.0,    # 只选择动量正值的ETF
    'trend_ma': 20,               # 趋势均线周期
    'max_volatility': 0.05,       # 最大波动率阈值（周波动率5%）
    'use_multi_momentum': True,  # 是否启用多期限动量合成
    'momentum_mode': 'vol_adjusted',       # 动量模式: 'raw'(原始), 'vol_adjusted'(波动率调整), 'downside_adjusted'(下行偏差调整)
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
        strategy_config = SIMPLE_STRATEGY_CONFIG.copy()
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
        slippage=backtest_config.get('slippage', 0.001)
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
    
    # 沪深300ETF基准
    hs300_df = get_etf_hist('510300', backtest_config['start_date'], backtest_config['end_date'])
    hs300_return = calculate_benchmark_return(
        hs300_df,
        pd.to_datetime(backtest_config['start_date']),
        pd.to_datetime(backtest_config['end_date'])
    )
    print(f"基准（沪深300ETF）总收益: {hs300_return:.2f}%")
    
    # 上证50ETF基准
    sh_df = get_etf_hist('510050', backtest_config['start_date'], backtest_config['end_date'])
    sh_return = calculate_benchmark_return(
        sh_df,
        pd.to_datetime(backtest_config['start_date']),
        pd.to_datetime(backtest_config['end_date'])
    )
    print(f"基准（上证50ETF）总收益: {sh_return:.2f}%")
    
    # 计算周期总结
    period_summary = calculate_period_summary(metrics, hs300_df, sh_df, summary_period)
    
    # 保存调仓记录
    backtest.save_rebalance_records()
    
    # 绘制结果
    if len(metrics) > 0:
        plot_backtest_results(metrics, benchmark_data={'hs300': hs300_df, 'sh': sh_df},
                             period_summary=period_summary)
        print_metrics(metrics, hs300_return, sh_return, period_summary)
        
        # 打印策略摘要
        print("\n" + "="*50)
        print("策略摘要")
        print("="*50)
        print(strategy.get_training_summary())
    else:
        print("回测失败")
    
    return metrics


def calculate_period_summary(metrics, hs300_df, sh_df, period='M'):
    """
    计算周期总结（策略收益、基准收益、超额收益）
    
    Parameters:
        metrics: 回测指标
        hs300_df: 沪深300ETF数据
        sh_df: 上证50ETF数据
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
    sh_period = sh_df['close'].resample(period).last() if sh_df is not None and len(sh_df) > 0 else None    
    hs300_return = hs300_period.pct_change() * 100 if hs300_period is not None else None
    sh_return = sh_period.pct_change() * 100 if sh_period is not None else None
    
    summary = pd.DataFrame(index=strategy_return.index)
    summary['strategy_return'] = strategy_return
    if hs300_return is not None:
        summary['hs300_return'] = hs300_return
        summary['excess_hs300'] = summary['strategy_return'] - summary['hs300_return']
    if sh_return is not None:
        summary['sh_return'] = sh_return
        summary['excess_sh'] = summary['strategy_return'] - summary['sh_return']
    
    summary = summary.dropna()
    return summary


def main():
    return run_backtest()


if __name__ == "__main__":
    main()
