"""
GWO 车辆排序优化 - P2 版本程序入口
含热启动逻辑和种群扩散
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data
from optimizer.gwo import GreyWolfOptimizer


# ---------- 颗粒度参数 ----------
TARGET_STEPS = 3000  # 目标仿真步数
VARS_PER_STEP_P2 = 4  # P2 每步变量数
DIM_P2 = TARGET_STEPS * VARS_PER_STEP_P2  # 计算总维度: 14000


def prepare_warm_start_matrix() -> np.ndarray:
    """
    准备 P2 初始解矩阵（热启动）
    
    Returns:
        初始化的 P2 决策矩阵 (TARGET_STEPS, 4)
    """
    # Col 0 (是否工作): 初始化为 1.0 (总是工作)
    # Col 1 (返回决策): 初始化为 0.0 (默认不回)
    # Col 2 (车道选择): 初始化为 0.0 (默认0道)
    # Col 3 (出车顺序): 初始化为 0.0 (默认FIFO)
    p2_init_matrix = np.zeros((TARGET_STEPS, VARS_PER_STEP_P2))
    p2_init_matrix[:, 0] = 1.0  # 强制 Input Hoist 始终尝试工作
    p2_init_matrix[:, 3] = 0.0  # 强制 Output Hoist 优先取第0辆车

    try:
        # 加载 P1 解
        p1_best_x = np.load('best_x_p1.npy')
        # p1_best_x = np.load('best_x_p1_att2.npy')  # 附件 2

        # 还原 P1 形状 (Steps_P1, 2)
        # 注意：P1只有2列，分别对应P2的 Col 2 (车道) 和 Col 1 (返回)
        p1_matrix = p1_best_x.reshape(-1, 2)

        steps_p1 = p1_matrix.shape[0]
        valid_steps = min(steps_p1, TARGET_STEPS)

        print(f"正在注入热启动数据... P1步数:{steps_p1}, P2目标:{TARGET_STEPS}")

        # 映射逻辑：
        # P1 Col 0 (车道) -> P2 Col 2
        # P1 Col 1 (返回) -> P2 Col 1
        p2_init_matrix[:valid_steps, 2] = p1_matrix[:valid_steps, 0]
        p2_init_matrix[:valid_steps, 1] = p1_matrix[:valid_steps, 1]

        print(f"成功注入前 {valid_steps} 步的决策数据。剩余 {TARGET_STEPS - valid_steps} 步保持随机。")

    except FileNotFoundError:
        print("警告：未找到 'best_x_p1.npy'，将使用完全随机初始化（可能导致无解）。")

    p2_init_matrix = np.clip(p2_init_matrix, 0, 0.99999999)
    return p2_init_matrix


def spread_to_population(model: GreyWolfOptimizer, base_wolf: np.ndarray) -> None:
    """
    将热启动数据扩散到整个种群
    
    Args:
        model: 灰狼优化器实例
        base_wolf: 基础解向量
    """
    model.wolves_position[0] = base_wolf  # 第 0 号狼完全保留纯净的热启动数据

    # 让第 1 到 39 号狼也继承这个解，但加一点点"变异"
    print("正在将热启动数据扩散至全种群...")
    for k in range(1, 40):  # 假设 size=40
        noise = np.random.normal(0, 0.01, size=len(base_wolf))
        new_wolf = base_wolf + noise
        new_wolf = np.clip(new_wolf, 0, 0.99999999)

        temp_matrix = new_wolf.reshape(-1, 4)
        # 强制修正关键列：Input Hoist Activation (Col 0)
        mask_work = temp_matrix[:, 0] < 0.6
        temp_matrix[mask_work, 0] = 0.8 + np.random.uniform(0, 0.1, size=np.sum(mask_work))

        model.wolves_position[k] = temp_matrix.flatten()

    print("种群初始化完成。开始运行...")


def main():
    """主函数"""
    # 加载数据
    data_1 = load_data('../data/附件1.xlsx')
    # data_2 = load_data('../data/附件2.xlsx')
    
    # 准备热启动矩阵
    p2_init_matrix = prepare_warm_start_matrix()
    
    # 初始化优化器
    model = GreyWolfOptimizer(max_iterations=400, dimension=DIM_P2)
    
    # 将热启动数据扩散到整个种群
    base_wolf = p2_init_matrix.flatten()
    spread_to_population(model, base_wolf)
    
    # 获取原始序列得分（可选）
    # model.calculate_original_fitness(data_1)
    # model.calculate_original_fitness(data_2)
    
    # 运行优化
    best_x, best_y = model.run(data_1)
    # best_x, best_y = model.run(data_2)
    
    # 绘制收敛曲线
    plt.plot(-model.convergence_curve_fitness)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('GWO Convergence Curve (P2)')
    plt.show()
    
    # 导出结果（可选）
    # result, outputLine, fanNumber, costTime = model.simulator.simulate(
    #     max_steps=model.dimension / 4,
    #     data=data_1,
    #     current_wolf_index=22,
    #     decision_vector=model.global_best_position,
    #     wolves_position=model.wolves_position,
    #     dimension=model.dimension,
    #     lower_bound=model.lower_bound,
    #     upper_bound=model.upper_bound
    # )
    # result = pd.DataFrame(result)
    # result.to_excel('result21.xlsx')


if __name__ == '__main__':
    main()
