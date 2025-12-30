"""
目标函数评估模块 - 计算车辆排序的多目标得分
"""
import numpy as np


def calculate_z1_interval(data_list: list) -> int:
    """
    计算 Z1 得分：混动车间隔违规次数
    
    Args:
        data_list: 燃油类型列表 (0=燃油, 1=混动)
        
    Returns:
        违规次数（连续混动车间隔不足3的次数）
    """
    interval_violations = 0
    consecutive_violations = 0
    data_arr = np.array(data_list)
    indices = np.where((data_arr == 1))

    for i in range(1, len(indices[0])):
        if indices[0][i] - indices[0][i - 1] == 3:
            interval_violations = interval_violations + 1
        else:
            consecutive_violations = consecutive_violations + 1

    return consecutive_violations


def calculate_z2_clustering(data_list: list) -> int:
    """
    计算 Z2 得分：四驱车聚类不平衡惩罚
    
    Args:
        data_list: 驱动类型列表 (0=两驱, 1=四驱)
        
    Returns:
        不平衡惩罚分数
    """
    block_lengths = [0]
    temp_lengths = []
    current_length = 0

    # 统计块的长度
    for i in range(1, len(data_list)):
        if data_list[i] == data_list[i - 1]:
            current_length += 1
        else:
            current_length = 0
        block_lengths.append(current_length)
    block_lengths.append(0)  # 确保最后一块能够检测到截止

    # 当后一项不在大于前一项时，说明出现更换，块结束
    for i in range(1, len(block_lengths)):
        if block_lengths[i] <= block_lengths[i - 1]:
            temp_lengths.append(block_lengths[i - 1] + 1)

    pairs_score = 0
    single_block_penalty = 0
    j = 1
    # 针对每一对，每两块比较长度，然后跳到下一对
    while j < len(temp_lengths):
        if temp_lengths[j] == temp_lengths[j - 1]:
            pairs_score = pairs_score + 1
        else:
            single_block_penalty = single_block_penalty + 1
        j = j + 2

    if len(temp_lengths) % 2:  # 如果是奇数，说明最后一块落单，不平衡
        single_block_penalty = single_block_penalty + 1
    return single_block_penalty


def evaluate_objective(sequence: list, number_of_returns: int, data: np.ndarray, makespan: int) -> tuple:
    """
    计算总目标函数得分
    
    Args:
        sequence: 输出车辆序列
        number_of_returns: 使用返回道次数
        data: 原始数据（需包含燃油和驱动类型列）
        makespan: 总耗时
        
    Returns:
        (总分, Z1加权分, Z2加权分, Z3加权分, Z4加权分)
    """
    length = len(sequence)

    data = data[:, 2:]
    drive_type_list = []
    for i in range(length):
        drive_type_list.append(data[sequence[i], 1])
    item2_score = calculate_z2_clustering(drive_type_list)

    fuel_type_list = []
    for i in range(length):
        fuel_type_list.append(data[sequence[i], 0])
    item1_score = calculate_z1_interval(fuel_type_list)

    total_score = (100 - item1_score) * 0.4 + (100 - item2_score) * 0.3 + (100 - number_of_returns) * 0.2 + 0.1 * (
                100 - (makespan - 2934) * 0.01)

    return total_score, (100 - item1_score) * 0.4, (100 - item2_score) * 0.3, (
                100 - number_of_returns) * 0.2, 0.1 * (
                                100 - (makespan - 2934) * 0.01)
