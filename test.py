"""
测试文件
- 数据获取测试
- 缓存功能测试
"""

# 从data模块导入功能
from data import get_etf_list, get_etf_hist, cache
import time


def test_cache():
    """测试缓存功能"""
    print("=== 测试缓存功能 ===")
    
    print("\n第一次调用（需要联网）:")
    start_time = time.time()
    df1 = get_etf_hist('510300', '2023-01-01', '2026-01-31')
    elapsed = time.time() - start_time
    print(f"耗时: {elapsed:.2f}秒")
    print(f"数据形状: {df1.shape}")
    
    print("\n第二次调用（使用缓存）:")
    start_time = time.time()
    df2 = get_etf_hist('510300', '2023-01-01', '2026-01-31')
    elapsed = time.time() - start_time
    print(f"耗时: {elapsed:.2f}秒")
    print(f"数据形状: {df2.shape}")
    
    return df1, df2


def test_etf_list():
    """测试ETF列表获取"""
    print("\n=== 测试ETF列表获取 ===")
    etf_list = get_etf_list()
    print(f"ETF数量: {len(etf_list)}")
    print(etf_list.head())
    return etf_list


if __name__ == '__main__':
    test_cache()
    test_etf_list()