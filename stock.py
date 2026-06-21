"""
周级别ETF量化策略 - 主程序入口
- 支持简单动量策略和多因子策略
- 训练/测试分离
- 回测运行与结果展示
"""

import pandas as pd
from data import get_etf_hist, get_multiple_etf_hist
from strategy import MultiFactorETFStrategy, SimpleMomentumStrategy
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
        '510050',  # 上证50ETF
        '588000',  # 科创50ETF
    ],
    # 行业ETF
    '行业': [
        '512660',  # 军工ETF
        '512880',  # 证券ETF
        '512010',  # 医药ETF
        '512760',  # 半导体ETF
        '515790',  # 光伏ETF
        '515030',  # 新能源车ETF
        '512400',  # 有色金属ETF
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

# 回测参数配置
BACKTEST_CONFIG = {
    'start_date': '2021-01-01',
    'end_date': '2026-06-18',
    'initial_capital': 1000000,
    'rebalance_cost': 0.0005,  # 0.05%手续费
}

# 简单动量策略配置（优化版）
SIMPLE_STRATEGY_CONFIG = {
    'n_portfolio': 5,
    'momentum_lookback': 12,
    'volatility_lookback': 12,
    'momentum_threshold': 0.0,    # 只选择动量正值的ETF
    'trend_ma': 20,               # 趋势均线周期
    'max_volatility': 0.05,       # 最大波动率阈值（周波动率5%）
    'train_end_date': '2023-12-31',  # 训练期结束日期（和多因子策略一致）
}

# 多因子策略配置
MULTIFACTOR_STRATEGY_CONFIG = {
    'n_portfolio': 5,
    'momentum_lookback': 12,
    'vol_lookback': 12,
    'trend_ma': 20,
    'train_end_date': '2023-12-31',
    'ic_threshold': 0.05,
    'use_equal_weight': False,
}


# ========== 主程序 ==========

def run_backtest(etf_codes=None, strategy_type='multifactor', strategy_config=None, backtest_config=None):
    """
    运行回测
    
    Parameters:
        etf_codes: ETF代码列表（可选）
        strategy_type: 策略类型，'simple'（简单动量）或 'multifactor'（多因子）
        strategy_config: 策略参数（可选）
        backtest_config: 回测参数（可选）
    
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
    
    # 根据策略类型选择配置和类
    if strategy_type == 'simple':
        if strategy_config is None:
            strategy_config = SIMPLE_STRATEGY_CONFIG.copy()
        strategy = SimpleMomentumStrategy(**strategy_config)
        strategy_name = "简单动量策略"
    else:
        if strategy_config is None:
            strategy_config = MULTIFACTOR_STRATEGY_CONFIG.copy()
        strategy = MultiFactorETFStrategy(**strategy_config)
        strategy_name = "多因子策略"
    
    print("="*50)
    print(f"周级别ETF量化策略 - {strategy_name}")
    print("="*50)
    
    if strategy_type == 'multifactor' and 'train_end_date' in strategy_config:
        print(f"训练期: {backtest_config['start_date']} 至 {strategy_config['train_end_date']}")
        print(f"测试期: {strategy_config['train_end_date']} 至 {backtest_config['end_date']}")
    
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
        rebalance_cost=backtest_config['rebalance_cost']
    )
    metrics = backtest.run(
        price_data_dict,
        strategy,
        pd.to_datetime(backtest_config['start_date']),
        pd.to_datetime(backtest_config['end_date'])
    )
    
    # 计算基准收益
    print("\n计算基准收益...")
    benchmark_df = get_etf_hist('510300', backtest_config['start_date'], backtest_config['end_date'])
    benchmark_return = calculate_benchmark_return(
        benchmark_df,
        pd.to_datetime(backtest_config['start_date']),
        pd.to_datetime(backtest_config['end_date'])
    )
    print(f"基准（沪深300）总收益: {benchmark_return:.2f}%")
    
    # 绘制结果
    if len(metrics) > 0:
        plot_backtest_results(metrics)
        print_metrics(metrics, benchmark_return)
        
        # 打印策略摘要
        print("\n" + "="*50)
        print("策略摘要")
        print("="*50)
        print(strategy.get_training_summary())
    else:
        print("回测失败")
    
    return metrics


def main():
    """主函数 - 默认运行多因子策略"""
    return run_backtest(strategy_type='multifactor')


if __name__ == "__main__":
    main()
