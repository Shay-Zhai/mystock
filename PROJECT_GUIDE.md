# ETF动量轮动策略 - 项目说明

## 给新agent的提示词
请先阅读以下两个文档了解项目：
- PROJECT_GUIDE.md：项目结构与各模块职责
- EXPERIMENT_NOTES.md：历史实验结论，避免重复踩坑

关键约束：
1. 所有set遍历必须sorted()保证确定性
2. 止损默认用individual模式（15%单票回撤），整体止损已验证有害
3. 6月动量跳10日是当前最优策略配置（SKIP_MOMENTUM_STRATEGY_CONFIG）
4. 实验脚本以_开头，完成后删除
5. 任何策略调整都应生成多参数对比表验证

## 项目目标
基于A股+跨境+商品ETF的动量轮动策略，通过多ETF择强买入、定期调仓，在控制回撤的前提下追求稳健的年化收益。

## 项目结构

```
mystock/
├── data.py              # 数据获取与缓存模块
├── strategy.py          # 策略核心模块（SimpleMomentumStrategy）
├── backtest.py          # 回测引擎
├── stock.py             # 主程序入口与配置
├── visualization.py     # 可视化与指标展示
├── data_cache/          # ETF历史数据缓存（pickle）
├── rebalance_records.csv  # 调仓记录输出（每次回测覆盖）
└── backtest_results.png # 回测图表输出（每次回测覆盖）
```

## 各模块职责

### 1. data.py - 数据层
- `DataCache`: 本地pickle缓存管理，默认24小时有效期
- `get_etf_hist(sec_code, start, end)`: 获取单只ETF历史K线（自动识别sh/sz前缀）
- `get_multiple_etf_hist(codes, start, end, min_length)`: 批量获取
- `get_index_hist(index_code, start, end)`: 获取指数数据
- 数据源：akshare的`fund_etf_hist_sina`
- 返回DataFrame列：`open/high/low/close/volume`，索引为日期

### 2. strategy.py - 策略层
核心类：`SimpleMomentumStrategy`

**关键参数**：
- `n_portfolio`: 持仓ETF数量（默认6）
- `momentum_lookback`: 动量回看周期数
- `momentum_skip`: 跳过最近N期（避免短期反转效应，0=不跳过）
- `volatility_lookback`: 波动率回看周期
- `momentum_threshold`: 动量阈值（默认0，只选动量为正）
- `trend_ma`: 趋势均线周期（价格需高于均线）
- `max_volatility`: 最大波动率上限
- `use_multi_momentum`: 是否启用多期限动量合成（20日40%+60日40%+120日20%）
- `momentum_mode`: `raw`/`vol_adjusted`/`downside_adjusted`

**选股流程**：
1. 计算各ETF动量、波动率、下行偏差、趋势
2. 趋势过滤（价格>均线）→ 波动率过滤 → 动量阈值过滤
3. 按风险调整动量排序，取top-N
4. 权重：波动率倒数（风险平价）

**跳月动量公式**（当`momentum_skip > 0`）：
```
momentum = price[-(skip+1)] / price[-(skip+lookback+1)] - 1
```

### 3. backtest.py - 回测引擎
核心类：`Backtest`

**关键参数**：
- `initial_capital`: 初始资金（默认100万）
- `rebalance_cost`: 手续费率（默认0.0005）
- `slippage`: 滑点率（默认0.001）
- `stop_loss_config`: 止损配置

**交易规则**：
- 调仓周期：每N个交易日（`rebalance_period`参数，默认20）
- 交易价格：收盘价（含滑点）
- 持仓按100股/手取整（`LOT_SIZE=100`）
- 买卖总成本率 = 手续费 + 滑点

**主循环**（逐日遍历）：
1. 每日计算组合净值，更新持仓ETF最高收盘价、组合净值最高点
2. 每日检查止损（单票止损、整体止损）
3. 调仓日执行策略：用调仓日之前的数据生成信号，收盘价执行交易
4. 记录每次调仓的调仓金额、费用

**止损逻辑**（四种模式）：
- `mode='none'`: 不止损
- `mode='individual'`: 单票止损——从持仓最高价回撤超过`individual_drawdown_pct`（默认15%）
- `mode='portfolio'`: 整体止损——组合回撤超`portfolio_dd_half`（默认18%）每标的减半，超`portfolio_dd_clear`（默认25%）清仓并休息到下个调仓周期
- `mode='both'`: 同时启用单票+整体止损

**输出**：
- `metrics` dict：年化收益、最大回撤、夏普、卡玛、胜率、portfolio_df等
- `trades_history`: 每笔交易记录（date/sec_code/action/shares/price/turnover/cost）
- `rebalance_records`: 每次调仓记录（含前后持仓"缩写:仓位比例"格式、调仓金额、费用率）

### 4. stock.py - 主程序
**配置区**：
- `ETF_POOL`: ETF池子（宽基/行业/跨境/商品，共14只）
- `ETF_NAMES`: 代码→缩写映射
- `BACKTEST_CONFIG`: 回测参数（含止损配置）
- `SIMPLE_STRATEGY_CONFIG`: 多期限动量策略（默认）
- `SKIP_MOMENTUM_STRATEGY_CONFIG`: 6月动量跳10日策略（参数优化最优）
- `SKIP_MOMENTUM_21D_STRATEGY_CONFIG`: 6-1经典跳月动量（126日跳21日）

**主流程**：
1. `run_backtest()`: 获取数据 → 运行回测 → 计算基准收益 → 周期总结 → 保存记录 → 绘图展示
2. `calculate_period_summary()`: 按月/季/年汇总策略收益、基准收益、超额收益、调仓金额、费用率

**运行方式**：`python stock.py`

### 5. visualization.py - 可视化
- `plot_backtest_results()`: 6子图——净值曲线、累计收益、回撤、收益率分布、周期收益对比、超额收益
- `print_metrics()`: 控制台输出整体指标、基准对比、周期收益总结表

## ETF池子
| 类别 | 代码 | 缩写 |
|------|------|------|
| 宽基 | 510300/512100/159915 | 沪深300/中证1000/创业板 |
| 行业 | 159928/512660/512880/512010/512760/512400/512800/515050 | 消费/军工/证券/医药/半导体/有色/银行/通信 |
| 跨境 | 513500/513100 | 标普500/纳斯达克 |
| 商品 | 518880 | 黄金 |

## 默认回测参数
- 回测区间：2018-01-01 至 2025-06-30
- 初始资金：100万
- 调仓周期：20交易日（约1月）
- 手续费：0.05% + 滑点0.1%（双边）
- 基准：沪深300ETF(510300)、上证50ETF(510050)

## 关键实现细节
1. **确定性保证**：所有set遍历都改为`sorted()`，避免PYTHONHASHSEED随机化导致结果不可复现
2. **未来函数防范**：调仓日用`df.index < trade_date`的历史数据生成信号
3. **一手取整**：买卖份额通过`(raw_shares // LOT_SIZE) * LOT_SIZE`确保100股整数倍
4. **现金不足处理**：买入时若现金不足，按可用现金买入最大可整手数
