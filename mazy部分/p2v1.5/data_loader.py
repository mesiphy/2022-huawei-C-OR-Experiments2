"""
数据加载与预处理模块
"""
import numpy as np
import pandas as pd


def load_data(file_path: str) -> np.ndarray:
    """
    加载 Excel 数据并进行预处理
    
    Args:
        file_path: Excel 文件路径
        
    Returns:
        预处理后的 numpy 数组，燃油/两驱已转换为 0/1
    """
    data = pd.read_excel(file_path)
    data = np.array(data)
    data[:, 2] = np.where(data[:, 2] == '燃油', 0, 1)
    data[:, 3] = np.where(data[:, 3] == '两驱', 0, 1)
    return data
