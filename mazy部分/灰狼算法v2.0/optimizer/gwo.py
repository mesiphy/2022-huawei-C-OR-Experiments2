"""
灰狼优化算法 (Grey Wolf Optimizer) 核心模块
"""
import numpy as np
import time

from simulation.simulator import ProductionLineSimulator
from evaluation.objective import evaluate_objective


class GreyWolfOptimizer:
    """灰狼优化器：使用 GWO 算法优化车辆排序"""
    
    def __init__(self, dimension: int = 6000, max_iterations: int = 200, 
                 population_size: int = 40, lower_bound: float = 0, 
                 upper_bound: float = 0.999999999):
        """
        初始化灰狼优化器
        
        Args:
            dimension: 决策空间维度
            max_iterations: 最大迭代次数
            population_size: 种群大小
            lower_bound: 位置下界
            upper_bound: 位置上界
        """
        self.dimension = dimension
        self.wolves_position = np.random.uniform(low=lower_bound, high=upper_bound, size=(population_size, dimension))
        self.alpha_position = None
        self.beta_position = None
        self.delta_position = None
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # 存储迭代过程记录
        self.convergence_curve_fitness = np.zeros(shape=(self.max_iterations + 1,))
        self.convergence_curve_position = np.zeros(shape=(self.max_iterations + 1, dimension))
        self.global_best_fitness = None
        self.global_best_position = None

        # 得分记录
        self.score_z1 = None
        self.score_z2 = None
        self.score_z3 = None
        self.score_z4 = None
        self.current_fitness_scores = None
        
        # 仿真器
        self.simulator = ProductionLineSimulator()

    def calculate_fitness(self, data: np.ndarray) -> None:
        """
        遍历每个狼群个体，计算适应度值
        
        Args:
            data: 车辆数据
        """
        self.current_fitness_scores = np.zeros(shape=(self.population_size,))

        for i in range(self.population_size):
            trajectory_matrix, output_sequence, return_usage_count, makespan = self.simulator.simulate(
                max_steps=self.dimension / 2,
                data=data,
                decision_vector=self.wolves_position[i],
                wolves_position=self.wolves_position,
                current_wolf_index=i,
                dimension=self.dimension,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound
            )

            current_score, self.score_z1, self.score_z2, self.score_z3, self.score_z4 = evaluate_objective(
                output_sequence, return_usage_count, data, makespan
            )
            self.current_fitness_scores[i] = -1 * current_score

    def calculate_original_fitness(self, data: np.ndarray) -> None:
        """
        获取原始序列（FIFO）的得分
        
        Args:
            data: 车辆数据
        """
        m = len(data)
        output_sequence = np.arange(m)
        return_usage_count = 0
        makespan = 2934
        current_score, self.score_z1, self.score_z2, self.score_z3, self.score_z4 = evaluate_objective(
            output_sequence, return_usage_count, data, makespan
        )
        # 打印结果
        print("=" * 30)
        print("【原始序列(FIFO) 得分报告】")
        print(f"总分 (Total Score): {current_score}")
        print(f"Z1 (混动间隔): {self.score_z1}")
        print(f"Z2 (四驱比例): {self.score_z2}")
        print(f"Z3 (返回道数): {self.score_z3}")
        print(f"Z4 (总耗时):   {self.score_z4}")
        print("=" * 30)

    def update_alpha_beta_delta(self, t: int) -> None:
        """
        获取当前迭代中的 alpha、beta、delta 狼的位置
        
        Args:
            t: 当前迭代次数
        """
        sorted_indices = self.current_fitness_scores.argsort()
        self.alpha_position = self.wolves_position[sorted_indices[0]]
        self.beta_position = self.wolves_position[sorted_indices[1]]
        self.delta_position = self.wolves_position[sorted_indices[2]]

    def update_wolf_positions(self, t: int) -> np.ndarray:
        """
        根据 GWO 算法更新狼群位置
        
        Args:
            t: 当前迭代次数
            
        Returns:
            更新后的狼群位置矩阵
        """
        r1_a = np.random.uniform(low=0, high=1, size=(self.population_size, self.dimension))
        r1_b = np.random.uniform(low=0, high=1, size=(self.population_size, self.dimension))
        r1_d = np.random.uniform(low=0, high=1, size=(self.population_size, self.dimension))
        r2_a = np.random.uniform(low=0, high=1, size=(self.population_size, self.dimension))
        r2_b = np.random.uniform(low=0, high=1, size=(self.population_size, self.dimension))
        r2_d = np.random.uniform(low=0, high=1, size=(self.population_size, self.dimension))

        a = 2 * (1 - t / self.max_iterations)

        A_a = 2 * a * r1_a - a
        A_b = 2 * a * r1_b - a
        A_d = 2 * a * r1_d - a

        C_a = 2 * r2_a
        C_b = 2 * r2_b
        C_d = 2 * r2_d

        distance_a = np.abs(C_a * self.alpha_position - self.wolves_position)
        distance_b = np.abs(C_b * self.beta_position - self.wolves_position)
        distance_d = np.abs(C_d * self.delta_position - self.wolves_position)

        X_a = self.alpha_position - A_a * distance_a
        X_b = self.beta_position - A_b * distance_b
        X_d = self.delta_position - A_d * distance_d

        new_positions = (X_a + X_b + X_d) / 3
        new_positions = np.where(new_positions < self.lower_bound, self.lower_bound, new_positions)
        new_positions = np.where(new_positions > self.upper_bound, self.upper_bound, new_positions)
        return new_positions

    def run(self, data: np.ndarray) -> tuple:
        """
        运行灰狼优化算法
        
        Args:
            data: 车辆数据
            
        Returns:
            (最优解向量, 最优得分)
        """
        start_time = time.time()

        for i in range(self.max_iterations):
            self.calculate_fitness(data)
            self.convergence_curve_fitness[i] = np.min(self.current_fitness_scores)
            self.convergence_curve_position[i, :] = self.alpha_position
            self.update_alpha_beta_delta(i)
            self.wolves_position = self.update_wolf_positions(i)

            if i % 10 == 0:
                print(i)
            end_time = time.time()
            if end_time - start_time > 6000:
                break

        # 最后一次更新过后的方案还没有计算，计算一次
        self.calculate_fitness(data)
        self.convergence_curve_fitness[i + 1] = np.min(self.current_fitness_scores)
        self.convergence_curve_fitness = self.convergence_curve_fitness[:i + 1]
        self.convergence_curve_position[i + 1, :] = self.alpha_position
        self.convergence_curve_position = self.convergence_curve_position[:i + 1, :]

        index = np.argmin(self.current_fitness_scores)
        best_solution_vector = self.wolves_position[index]
        best_solution_score = self.current_fitness_scores[index]

        # 寻找整个过程中的最优解——全局历史最优解
        global_min_score_index = self.convergence_curve_fitness.argmin()
        self.global_best_fitness = self.convergence_curve_fitness[global_min_score_index]
        self.global_best_position = self.convergence_curve_position[global_min_score_index]
        return best_solution_vector, best_solution_score
