"""
数据获取与缓存模块
- 本地缓存管理
- ETF数据获取
"""

import akshare as ak
import pandas as pd
import os
import pickle
import time


class DataCache:
    """本地数据缓存管理"""
    
    def __init__(self, cache_dir='data_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key):
        safe_key = key.replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")
    
    def _get_meta_path(self, key):
        safe_key = key.replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.cache_dir, f"{safe_key}_meta.pkl")
    
    def save(self, key, data):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        meta = {
            'save_time': time.time(),
            'data_shape': data.shape if hasattr(data, 'shape') else len(data)
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
    
    def load(self, key):
        """从缓存加载数据"""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def is_valid(self, key, max_age_hours=24):
        """检查缓存是否有效"""
        meta_path = self._get_meta_path(key)
        if not os.path.exists(meta_path):
            return False
        
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        age_hours = (time.time() - meta['save_time']) / 3600
        return age_hours <= max_age_hours
    
    def clear(self, key=None):
        """清除缓存"""
        if key is None:
            for f in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, f))
        else:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_meta_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)


# 全局缓存实例
cache = DataCache()


def get_etf_list(force_refresh=False, cache_age_hours=24):
    """
    获取所有ETF列表（带本地缓存）
    
    Parameters:
        force_refresh: 是否强制刷新缓存
        cache_age_hours: 缓存有效期（小时）
    
    Returns:
        DataFrame: ETF列表
    """
    cache_key = 'etf_list_sina'
    
    if not force_refresh and cache.is_valid(cache_key, cache_age_hours):
        cached_data = cache.load(cache_key)
        if cached_data is not None and not cached_data.empty:
            return cached_data
    
    try:
        df = ak.fund_etf_category_sina(symbol='ETF基金')
        if not df.empty:
            cache.save(cache_key, df)
        return df
    except Exception as e:
        cached_data = cache.load(cache_key)
        if cached_data is not None:
            return cached_data
        return pd.DataFrame()


def get_etf_hist(sec_code, start_date, end_date, force_refresh=True, cache_age_hours=24):
    """
    获取ETF历史数据（带本地缓存）
    
    Parameters:
        sec_code: ETF代码（如 '510300' 或 'sh510300'）
        start_date: 开始日期
        end_date: 结束日期
        force_refresh: 是否强制刷新缓存
        cache_age_hours: 缓存有效期（小时）
    
    Returns:
        DataFrame: 历史数据（date为索引）
    """
    # 自动添加市场前缀
    if not sec_code.startswith('sh') and not sec_code.startswith('sz'):
        if sec_code.startswith('6') or sec_code.startswith('5'):
            sec_code = 'sh' + sec_code
        else:
            sec_code = 'sz' + sec_code
    
    cache_key = f'etf_hist_{sec_code}'
    
    # 尝试使用缓存
    if not force_refresh and cache.is_valid(cache_key, cache_age_hours):
        cached_df = cache.load(cache_key)
        if cached_df is not None and not cached_df.empty:
            df = cached_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            if not df.empty:
                df = df.sort_values('date')
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df.set_index('date', inplace=True)
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                return df
    
    # 联网获取数据
    try:
        df = ak.fund_etf_hist_sina(symbol=sec_code)
        if df.empty:
            # 尝试使用旧缓存
            cached_df = cache.load(cache_key)
            if cached_df is not None:
                df = cached_df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                if not df.empty:
                    df = df.sort_values('date')
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                    df.set_index('date', inplace=True)
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    return df
            return pd.DataFrame()
        
        # 保存到缓存
        cache.save(cache_key, df)
        
        # 处理数据
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.sort_values('date')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('date', inplace=True)
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df
    except Exception as e:
        # 尝试使用缓存
        cached_df = cache.load(cache_key)
        if cached_df is not None:
            df = cached_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            if not df.empty:
                df = df.sort_values('date')
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df.set_index('date', inplace=True)
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                return df
        return pd.DataFrame()


def get_index_hist(index_code, start_date, end_date, force_refresh=False, cache_age_hours=24):
    """
    获取指数历史数据（带本地缓存）
    
    Parameters:
        index_code: 指数代码（如 '000001' 上证指数, '000300' 沪深300）
        start_date: 开始日期
        end_date: 结束日期
        force_refresh: 是否强制刷新缓存
        cache_age_hours: 缓存有效期（小时）
    
    Returns:
        DataFrame: 历史数据（date为索引）
    """
    cache_key = f'index_hist_{index_code}'
    
    if not force_refresh and cache.is_valid(cache_key, cache_age_hours):
        cached_df = cache.load(cache_key)
        if cached_df is not None and not cached_df.empty:
            df = cached_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            if not df.empty:
                df = df.sort_values('date')
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df.set_index('date', inplace=True)
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                return df
    
    try:
        df = ak.stock_zh_index_daily(symbol=index_code)
        if df.empty:
            cached_df = cache.load(cache_key)
            if cached_df is not None:
                df = cached_df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                if not df.empty:
                    df = df.sort_values('date')
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                    df.set_index('date', inplace=True)
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    return df
            return pd.DataFrame()
        
        cache.save(cache_key, df)
        
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.sort_values('date')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('date', inplace=True)
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df
    except Exception as e:
        cached_df = cache.load(cache_key)
        if cached_df is not None:
            df = cached_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            if not df.empty:
                df = df.sort_values('date')
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df.set_index('date', inplace=True)
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                return df
        return pd.DataFrame()


def get_multiple_etf_hist(etf_codes, start_date, end_date, min_length=50, force_refresh=True):
    """
    批量获取多个ETF的历史数据
    
    Parameters:
        etf_codes: ETF代码列表
        start_date: 开始日期
        end_date: 结束日期
        min_length: 最小数据长度要求
        force_refresh: 是否强制刷新缓存
    
    Returns:
        dict: {sec_code: DataFrame}
    """
    price_data_dict = {}
    for i, code in enumerate(etf_codes):
        print(f"获取 {code}... ({i+1}/{len(etf_codes)})")
        df = get_etf_hist(code, start_date, end_date, force_refresh=force_refresh)
        if len(df) >= min_length:
            price_data_dict[code] = df
    return price_data_dict