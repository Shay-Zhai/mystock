"""
因子计算模块
- 动量因子
- 趋势因子
- 波动率因子
- 质量因子
- 技术因子
"""

import numpy as np
import pandas as pd


def calc_momentum(prices, lookback=12):
    """
    计算动量因子 - 过去N周收益率
    """
    if len(prices) < lookback:
        return np.nan
    return (prices.iloc[-1] / prices.iloc[-lookback]) - 1


def calc_momentum_acceleration(prices, short_lookback=4, long_lookback=12):
    """
    动量加速度因子 - 短期动量与长期动量的差异
    捕捉动量加速或减速的ETF
    """
    if len(prices) < long_lookback:
        return np.nan
    short_mom = (prices.iloc[-1] / prices.iloc[-short_lookback]) - 1
    long_mom = (prices.iloc[-1] / prices.iloc[-long_lookback]) - 1
    return short_mom - long_mom / 3  # 标准化处理


def calc_trend_strength(prices, ma_period=20):
    """
    计算趋势强度 - 价格与均线的偏离度
    """
    if len(prices) < ma_period:
        return np.nan
    ma = prices.rolling(ma_period).mean().iloc[-1]
    return (prices.iloc[-1] / ma) - 1


def calc_volatility(prices, lookback=12):
    """
    计算波动率 - 周收益率标准差
    """
    if len(prices) < lookback:
        return np.nan
    returns = prices.pct_change().dropna()
    if len(returns) < lookback:
        return np.nan
    return returns.tail(lookback).std()


def calc_sharpe_ratio(prices, lookback=12, risk_free_weekly=0.0006):
    """
    历史夏普比率因子
    使用过去N周的数据计算夏普比率
    """
    if len(prices) < lookback + 1:
        return np.nan
    returns = prices.pct_change().dropna().tail(lookback)
    if len(returns) < lookback:
        return np.nan
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0 or np.isnan(std_return):
        return np.nan
    return (mean_return - risk_free_weekly) / std_return


def calc_win_rate(prices, lookback=12):
    """
    胜率因子 - 历史上涨周数占比
    """
    if len(prices) < lookback + 1:
        return np.nan
    returns = prices.pct_change().dropna().tail(lookback)
    if len(returns) == 0:
        return np.nan
    positive_count = (returns > 0).sum()
    return positive_count / len(returns)


def calc_rsi(prices, lookback=12):
    """
    RSI相对强弱指标（周级别）
    数值越低表示超卖，越高表示超买
    作为均值回归因子使用
    """
    if len(prices) < lookback + 1:
        return np.nan
    returns = prices.pct_change().dropna().tail(lookback)
    if len(returns) < lookback:
        return np.nan
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))


def calc_max_drawdown_recovery(prices, lookback=24):
    """
    最大回撤恢复因子
    衡量从最大回撤中恢复的程度
    值越大表示恢复越好
    """
    if len(prices) < lookback:
        return np.nan
    hist_prices = prices.tail(lookback)
    cummax = hist_prices.cummax()
    drawdown = (hist_prices - cummax) / cummax
    max_dd = drawdown.min()
    current_dd = drawdown.iloc[-1]
    if max_dd == 0:
        return 0
    # 恢复程度：0表示在最低点，1表示完全恢复
    recovery = 1 - (abs(current_dd) / abs(max_dd))
    return recovery


def calc_profit_loss_ratio(prices, lookback=12):
    """
    盈亏比因子
    平均盈利 / 平均亏损
    """
    if len(prices) < lookback + 1:
        return np.nan
    returns = prices.pct_change().dropna().tail(lookback)
    if len(returns) == 0:
        return np.nan
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    avg_gain = gains.mean() if len(gains) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-6
    if avg_loss == 0:
        return np.nan
    return avg_gain / avg_loss


def calc_price_position(prices, lookback=20):
    """
    价格位置因子
    当前价格在近期区间中的位置（0-1）
    作为均值回归或趋势确认使用
    """
    if len(prices) < lookback:
        return np.nan
    hist_prices = prices.tail(lookback)
    high = hist_prices.max()
    low = hist_prices.min()
    if high == low:
        return 0.5
    return (prices.iloc[-1] - low) / (high - low)


def calc_volume_trend(volume, lookback=4):
    """
    成交量趋势因子
    近期成交量与前期成交量的比率
    """
    if len(volume) < lookback * 2:
        return np.nan
    recent_vol = volume.tail(lookback).mean()
    past_vol = volume.iloc[-lookback*2:-lookback].mean()
    if past_vol == 0:
        return np.nan
    return recent_vol / past_vol - 1


def calc_all_factors(df, momentum_lookback=12, trend_ma=20, vol_lookback=12):
    """
    计算所有因子
    
    Parameters:
        df: DataFrame 包含 'close' 和可选的 'volume' 列
        momentum_lookback: 动量回看周期
        trend_ma: 趋势均线周期
        vol_lookback: 波动率回看周期
    
    Returns:
        dict: 因子值字典
    """
    close = df['close']
    
    momentum = calc_momentum(close, momentum_lookback)
    mom_acceleration = calc_momentum_acceleration(close, 4, momentum_lookback)
    trend = calc_trend_strength(close, trend_ma)
    volatility = calc_volatility(close, vol_lookback)
    sharpe = calc_sharpe_ratio(close, vol_lookback)
    win_rate = calc_win_rate(close, vol_lookback)
    rsi = calc_rsi(close, vol_lookback)
    dd_recovery = calc_max_drawdown_recovery(close, momentum_lookback * 2)
    pl_ratio = calc_profit_loss_ratio(close, vol_lookback)
    price_pos = calc_price_position(close, trend_ma)
    
    factors = {
        'momentum': momentum,
        'momentum_acceleration': mom_acceleration,
        'trend': trend,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'rsi': rsi,
        'drawdown_recovery': dd_recovery,
        'profit_loss_ratio': pl_ratio,
        'price_position': price_pos,
    }
    
    # 如果有成交量数据，计算成交量因子
    if 'volume' in df.columns:
        volume_trend = calc_volume_trend(df['volume'], 4)
        factors['volume_trend'] = volume_trend
    
    return factors
