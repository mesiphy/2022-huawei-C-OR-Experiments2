"""
GWO 车辆排序优化 - 程序入口
"""
import matplotlib.pyplot as plt

from data_loader import load_data
from optimizer.gwo import GreyWolfOptimizer


def main():
    """主函数"""
    # 加载数据
    data_1 = load_data('../data/附件1.xlsx')
    # data_2 = load_data('../data/附件2.xlsx')
    
    # 创建优化器并运行
    model = GreyWolfOptimizer(max_iterations=400)
    
    # 获取原始序列得分（可选）
    # model.calculate_original_fitness(data_1)
    # model.calculate_original_fitness(data_2)
    
    # 运行优化
    best_x, best_y = model.run(data_1)
    
    # 绘制收敛曲线
    plt.plot(-model.convergence_curve_fitness)  # 原问题是最大化问题，代码的求解是最小化问题，因此需要取负号绘图
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('GWO Convergence Curve')
    plt.show()
    
    # 导出结果（可选）
    # result, outputLine, fanNumber, costTime = model.simulator.simulate(
    #     max_steps=model.dimension / 2, 
    #     data=data_1, 
    #     current_wolf_index=22,
    #     decision_vector=model.global_best_position,
    #     wolves_position=model.wolves_position,
    #     dimension=model.dimension,
    #     lower_bound=model.lower_bound,
    #     upper_bound=model.upper_bound
    # )
    # import pandas as pd
    # result = pd.DataFrame(result)
    # result.to_excel('附件 1v5.0 的结果-对应图5.0.xlsx')


if __name__ == '__main__':
    main()
