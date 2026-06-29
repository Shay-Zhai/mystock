"""
可视化模块
- 回测结果绘图
- 指标展示
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_backtest_results(metrics, benchmark_data=None, period_summary=None, save_path='backtest_results.png'):
    """绘制回测结果"""
    if 'portfolio_df' not in metrics or metrics['portfolio_df'] is None:
        print("No data to plot")
        return
    
    df = metrics['portfolio_df']
    
    # 根据是否有周期总结决定子图数量
    has_summary = period_summary is not None and len(period_summary) > 0
    nrows = 3 if has_summary else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows))
    fig.suptitle('ETF量化策略回测结果', fontsize=14)
    
    # 1. 净值曲线
    ax1 = axes[0, 0]
    
    ax1.plot(df.index, df['value'], 'b-', label='策略净值', linewidth=1.5)
    
    # 添加基准对比（归一化到初始资金）
    if benchmark_data is not None and isinstance(benchmark_data, dict):
        initial_value = df['value'].iloc[0]
        for name, bm_df in benchmark_data.items():
            if bm_df is not None and len(bm_df) > 0:
                bm_df = bm_df[bm_df.index >= df.index[0]]
                bm_df = bm_df[bm_df.index <= df.index[-1]]
                if len(bm_df) > 0:
                    bm_returns = bm_df['close'] / bm_df['close'].iloc[0]
                    bm_values = bm_returns * initial_value
                    label = '沪深300ETF' if name == 'hs300' else '上证50ETF' if name == 'sh' else name
                    ax1.plot(bm_df.index, bm_values, linestyle='--', alpha=0.7, label=f'{label}基准')
    
    ax1.set_title('组合净值')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('净值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 累计收益
    ax2 = axes[0, 1]
    ax2.plot(df.index, df['cum_return'] * 100, 'b-', linewidth=1.5, label='策略')
    
    # 添加基准累计收益
    if benchmark_data is not None and isinstance(benchmark_data, dict):
        for name, bm_df in benchmark_data.items():
            if bm_df is not None and len(bm_df) > 0:
                bm_df = bm_df[bm_df.index >= df.index[0]]
                bm_df = bm_df[bm_df.index <= df.index[-1]]
                if len(bm_df) > 0:
                    bm_cum_return = (bm_df['close'] / bm_df['close'].iloc[0] - 1) * 100
                    label = '沪深300ETF' if name == 'hs300' else '上证50ETF' if name == 'sh' else name
                    ax2.plot(bm_df.index, bm_cum_return, linestyle='--', alpha=0.7, label=f'{label}')
    
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_title('累计收益 (%)')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('收益 (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 回撤曲线
    ax3 = axes[1, 0]
    ax3.fill_between(df.index, df['drawdown'] * 100, 0, alpha=0.5, color='blue', label='策略')
    ax3.set_title('回撤 (%)')
    ax3.set_xlabel('日期')
    ax3.set_ylabel('回撤 (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 周收益率分布
    ax4 = axes[1, 1]
    returns = df['return'].dropna() * 100
    ax4.hist(returns, bins=20, edgecolor='black', alpha=0.7)
    ax4.axvline(x=returns.mean(), color='r', linestyle='--', label=f'均值: {returns.mean():.2f}%')
    ax4.set_title('调仓收益率分布')
    ax4.set_xlabel('收益率 (%)')
    ax4.set_ylabel('频次')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 周期收益对比折线图
    if has_summary:
        ax5 = axes[2, 0]
        ax5.plot(period_summary.index, period_summary['strategy_return'], 'b-o', 
                label='策略收益', linewidth=1.5, markersize=3)
        if 'hs300_return' in period_summary.columns:
            ax5.plot(period_summary.index, period_summary['hs300_return'], 'g--s', 
                    label='沪深300收益', linewidth=1, markersize=3, alpha=0.7)
        if 'sh_return' in period_summary.columns:
            ax5.plot(period_summary.index, period_summary['sh_return'], 'orange', 
                    linestyle='--', marker='^', label='上证50收益', linewidth=1, markersize=3, alpha=0.7)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax5.set_title('周期收益率对比')
        ax5.set_xlabel('周期')
        ax5.set_ylabel('收益率 (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        # 旋转x轴标签
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        
        # 6. 超额收益折线图
        ax6 = axes[2, 1]
        if 'excess_hs300' in period_summary.columns:
            ax6.plot(period_summary.index, period_summary['excess_hs300'], 'g-o', 
                    label='超额(沪深300)', linewidth=1.5, markersize=3)
        if 'excess_sh' in period_summary.columns:
            ax6.plot(period_summary.index, period_summary['excess_sh'], 'orange', 
                    linestyle='-', marker='^', label='超额(上证50)', linewidth=1.5, markersize=3, alpha=0.7)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax6.set_title('超额收益率')
        ax6.set_xlabel('周期')
        ax6.set_ylabel('超额收益 (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存至: {save_path}")


def print_metrics(metrics, benchmark_return=None, sh_return=None, period_summary=None):
    """打印回测指标"""
    print("\n" + "="*50)
    print("回测指标")
    print("="*50)
    
    # 整体指标
    print("\n【整体表现】")
    print(f"总收益: {metrics['total_return']*100:.2f}%")
    print(f"年化收益: {metrics['annual_return']*100:.2f}%")
    print(f"年化波动率: {metrics['annual_volatility']*100:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"卡玛比率: {metrics['calmar_ratio']:.2f}")
    print(f"胜率: {metrics['win_rate']*100:.2f}%")
    
    # 基准对比
    print("\n【基准对比】")
    if benchmark_return is not None:
        print(f"沪深300ETF基准收益: {benchmark_return:.2f}%")
        print(f"相对沪深300ETF超额收益: {metrics['total_return']*100 - benchmark_return:.2f}%")
    
    if sh_return is not None:
        print(f"上证50ETF基准收益: {sh_return:.2f}%")
        print(f"相对上证50ETF超额收益: {metrics['total_return']*100 - sh_return:.2f}%")
    
    # 周期总结
    if period_summary is not None and len(period_summary) > 0:
        print("\n【周期收益总结】")
        has_turnover = 'turnover' in period_summary.columns
        if has_turnover:
            print(f"{'周期':<10}{'策略收益':>9} {'沪深300':>9} {'超额(300)':>10} {'上证50':>9} {'超额(50)':>9} {'调仓金额':>12} {'费用率':>8}")
            print("-" * 82)
        else:
            print(f"{'周期':<12} {'策略收益':>10} {'沪深300':>10} {'超额(300)':>10} {'上证50':>10} {'超额(50)':>10}")
            print("-" * 72)
        for idx, row in period_summary.iterrows():
            period_str = idx.strftime('%Y-%m')
            strat = f"{row['strategy_return']:.2f}%"
            hs300 = f"{row.get('hs300_return', 0):.2f}%" if 'hs300_return' in row else '-'
            excess300 = f"{row.get('excess_hs300', 0):.2f}%" if 'excess_hs300' in row else '-'
            sh = f"{row.get('sh_return', 0):.2f}%" if 'sh_return' in row else '-'
            excess_sh = f"{row.get('excess_sh', 0):.2f}%" if 'excess_sh' in row else '-'
            if has_turnover:
                turnover = f"{row.get('turnover', 0):.0f}"
                cost_rate = f"{row.get('cost_rate', 0):.3f}%"
                print(f"{period_str:<10}{strat:>9} {hs300:>9} {excess300:>10} {sh:>9} {excess_sh:>9} {turnover:>12} {cost_rate:>8}")
            else:
                print(f"{period_str:<12} {strat:>10} {hs300:>10} {excess300:>10} {sh:>10} {excess_sh:>10}")
    
    # 诊断信息
    df = metrics['portfolio_df']
    print(f"\n回测周期: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"总调仓次数: {len(df)}")
    print(f"平均收益率: {df['return'].mean()*100:.4f}%")
    print(f"收益率标准差: {df['return'].std()*100:.4f}%")
    print(f"收益率最大值: {df['return'].max()*100:.2f}%")
    print(f"收益率最小值: {df['return'].min()*100:.2f}%")


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
