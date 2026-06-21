"""
可视化模块
- 回测结果绘图
- 训练/测试指标展示
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_backtest_results(metrics, benchmark_data=None, save_path='backtest_results.png'):
    """绘制回测结果"""
    if 'portfolio_df' not in metrics or metrics['portfolio_df'] is None:
        print("No data to plot")
        return
    
    df = metrics['portfolio_df']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weekly ETF Strategy Backtest Results', fontsize=14)
    
    # 1. 净值曲线
    ax1 = axes[0, 0]
    
    # 分离训练期和测试期
    train_df = df[~df['is_test']] if 'is_test' in df.columns else df
    test_df = df[df['is_test']] if 'is_test' in df.columns else pd.DataFrame()
    
    ax1.plot(train_df.index, train_df['value'], 'b-', label='Train Period', linewidth=1.5)
    if len(test_df) > 0:
        ax1.plot(test_df.index, test_df['value'], 'r-', label='Test Period', linewidth=1.5)
    
    # 添加训练/测试分界线
    if len(test_df) > 0:
        ax1.axvline(x=test_df.index[0], color='gray', linestyle='--', alpha=0.7, label='Train/Test Split')
    
    ax1.set_title('Portfolio Value')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 累计收益
    ax2 = axes[0, 1]
    ax2.plot(train_df.index, train_df['cum_return'] * 100, 'b-', linewidth=1.5, label='Train')
    if len(test_df) > 0:
        ax2.plot(test_df.index, test_df['cum_return'] * 100, 'r-', linewidth=1.5, label='Test')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    if len(test_df) > 0:
        ax2.axvline(x=test_df.index[0], color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Cumulative Return (%)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 回撤曲线
    ax3 = axes[1, 0]
    ax3.fill_between(train_df.index, train_df['drawdown'] * 100, 0, alpha=0.5, color='blue', label='Train')
    if len(test_df) > 0:
        ax3.fill_between(test_df.index, test_df['drawdown'] * 100, 0, alpha=0.5, color='red', label='Test')
    ax3.set_title('Drawdown (%)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 周收益率分布
    ax4 = axes[1, 1]
    returns = df['return'].dropna() * 100
    ax4.hist(returns, bins=20, edgecolor='black', alpha=0.7)
    ax4.axvline(x=returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
    ax4.set_title('Weekly Return Distribution')
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存至: {save_path}")


def print_metrics(metrics, benchmark_return=None):
    """打印回测指标（支持训练/测试分离）"""
    print("\n" + "="*50)
    print("Backtest Metrics")
    print("="*50)
    
    # 整体指标
    print("\n【整体表现】")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Annual Return: {metrics['annual_return']*100:.2f}%")
    print(f"Annual Volatility: {metrics['annual_volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    
    # 测试期指标
    if 'test_return' in metrics:
        print("\n【测试期表现（样本外）】")
        print(f"Test Return: {metrics['test_return']*100:.2f}%")
        print(f"Test Annual Return: {metrics['test_annual_return']*100:.2f}%")
        print(f"Test Annual Volatility: {metrics['test_annual_volatility']*100:.2f}%")
        print(f"Test Sharpe Ratio: {metrics['test_sharpe_ratio']:.2f}")
        print(f"Test Max Drawdown: {metrics['test_max_drawdown']*100:.2f}%")
        print(f"Test Calmar Ratio: {metrics['test_calmar_ratio']:.2f}")
        print(f"Test Win Rate: {metrics['test_win_rate']*100:.2f}%")
    
    if benchmark_return is not None:
        print(f"\nBenchmark Return: {benchmark_return:.2f}%")
        print(f"Excess Return: {metrics['total_return']*100 - benchmark_return:.2f}%")
        if 'test_return' in metrics:
            print(f"Test Excess Return: {metrics['test_return']*100 - benchmark_return:.2f}%")
    
    # 诊断信息
    df = metrics['portfolio_df']
    print(f"\n回测周期: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"总周数: {len(df)}")
    print(f"平均周收益率: {df['return'].mean()*100:.4f}%")
    print(f"周收益率标准差: {df['return'].std()*100:.4f}%")
    print(f"周收益率最大值: {df['return'].max()*100:.2f}%")
    print(f"周收益率最小值: {df['return'].min()*100:.2f}%")


def plot_trades_distribution(trades_df, save_path='trades_distribution.png'):
    """绘制交易分布图"""
    if len(trades_df) == 0:
        print("No trades to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 买入/卖出次数分布
    ax1 = axes[0]
    trade_counts = trades_df.groupby('sec_code')['action'].value_counts().unstack(fill_value=0)
    trade_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Trade Counts by ETF')
    ax1.set_xlabel('ETF Code')
    ax1.set_ylabel('Count')
    ax1.legend(['Buy', 'Sell'])
    ax1.grid(True, alpha=0.3)
    
    # 交易时间分布
    ax2 = axes[1]
    trades_df['month'] = trades_df['date'].dt.to_period('M')
    monthly_trades = trades_df.groupby('month').size()
    monthly_trades.plot(kind='bar', ax=ax2)
    ax2.set_title('Monthly Trade Distribution')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Trade Count')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"交易分布图已保存至: {save_path}")
