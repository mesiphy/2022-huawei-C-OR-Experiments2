import math
import random
import copy
import matplotlib.pyplot as plt
import time
from data_preparation import load_data  # 导入你之前写好的读取函数


class PBS_SimulatedAnnealing:
    def __init__(self, cars_data, lanes=6):
        self.cars = cars_data
        self.n = len(cars_data)
        self.k_lanes = lanes

        # 提取属性数组，加速计算
        # H: 1=混动, 0=非混动
        self.H_list = [c['H'] for c in cars_data]
        # D: 1=四驱, 0=两驱
        self.D_list = [c['D'] for c in cars_data]
        # Input Order: 记录原始进车顺序 (0 to N-1)
        self.input_ids = [c['id'] for c in cars_data]

        # [cite_start]权重 [cite: 63-67]
        self.w1, self.w2, self.w3, self.w4 = 0.4, 0.3, 0.2, 0.1

    # ==========================
    # 1. 快速评估函数 (核心)
    # ==========================
    def calculate_score(self, sequence_indices):
        """
        计算给定序列的总分。
        sequence_indices: 车辆ID的排列列表，代表出车顺序
        """
        # --- Z1: 混动间隔 (O(N)) ---
        # [cite_start]规则: 混动(1)之间要有2个非混动(0) [cite: 71]
        z1_penalties = 0
        h_indices = []
        for rank, car_id in enumerate(sequence_indices):
            if self.H_list[car_id] == 1:
                h_indices.append(rank)

        for i in range(len(h_indices) - 1):
            gap = h_indices[i + 1] - h_indices[i] - 1
            if gap != 2:
                z1_penalties += 1

        z1_score = 100 - z1_penalties

        # --- Z2: 驱动比例 (O(N)) ---
        # [cite_start]规则: 动态分块，块内1:1 [cite: 72]
        # 简化逻辑：直接遍历序列，检测属性变化点作为分块
        z2_penalties = 0
        current_block_counts = {0: 0, 1: 0}  # 0:两驱, 1:四驱

        # 获取该序列的驱动属性列表
        seq_d = [self.D_list[i] for i in sequence_indices]

        if not seq_d:
            z2_score = 100
        else:
            current_type = seq_d[0]  # 起始类型

            for d_val in seq_d:
                # 检查是否切换了类型 (作为分块依据)
                # 题目逻辑：以4开头则变4分块，以2开头则变2分块
                # 这里简化：只要类型发生"特定翻转"（例如由主导变成稀缺），通常意味着一块结束
                # 为了速度，我们使用一种鲁棒的近似：
                # 每当遇到与当前块首元素不同的类型，且之前已经积累了同类型，可能意味着新块？
                # 严格按照题目： "如果序列以4开头，则根据从2变为4将序列进行分块"

                # 实际上 SA 跑全量数据时，Z2 很难完美，我们用一个更平滑的惩罚：
                # 滑动窗口或绝对差值（近似 MILP 中的逻辑）
                pass

                # --- Z2 快速近似算法 (Robust for SA) ---
            # 统计整个序列中，四驱和两驱是否交替出现。
            # 只要任意滑窗(size=4)内比例不失衡太严重即可。
            # 这里为了性能，暂时使用"全局差异"作为惩罚的代替，或者严格实现题目的分块
            # 下面实现严格分块逻辑：
            blocks = []
            if len(seq_d) > 0:
                block_start = 0
                first_type = seq_d[0]  # 1(四驱) 或 0(两驱)

                # 定义切割触发器
                # 如果以四驱(1)开头，当遇到 (0 -> 1) 切换时切割
                # 如果以两驱(0)开头，当遇到 (1 -> 0) 切换时切割

                for i in range(1, len(seq_d)):
                    curr = seq_d[i]
                    prev = seq_d[i - 1]

                    cut = False
                    if first_type == 1:  # 4开头
                        if prev == 0 and curr == 1: cut = True
                    else:  # 2开头
                        if prev == 1 and curr == 0: cut = True

                    if cut:
                        blocks.append(seq_d[block_start:i])
                        block_start = i

                blocks.append(seq_d[block_start:])  # 最后一个块

            for b in blocks:
                c1 = sum(b)  # 四驱数
                c0 = len(b) - c1  # 两驱数
                if c1 != c0:
                    z2_penalties += 1

        z2_score = 100 - z2_penalties

        # --- Z3 & Z4: 物理仿真 (Heuristic Simulator) ---
        # 我们需要判断：将 input_ids 转换为 sequence_indices 需要多少代价？
        # 使用贪心算法模拟进车道过程

        # 映射：Car ID -> 出车序列中的目标位置 (Rank)
        target_rank = {car_id: r for r, car_id in enumerate(sequence_indices)}

        lanes = [[] for _ in range(self.k_lanes)]  # 6个车道
        return_lane_usage = 0

        # 模拟进车过程 (Input Order 0 -> N-1)
        for car_id in self.input_ids:
            my_target = target_rank[car_id]
            best_lane = -1
            min_gap = float('inf')

            # 策略：寻找一个车道，使得我在该车道的前车(pre_car)之后出库
            # 即: tail_car_rank < my_target
            # 如果有多个满足，选 target_rank 最大的那个（紧凑堆叠）

            valid_lanes = []
            for l_idx, lane in enumerate(lanes):
                if len(lane) == 0:
                    # 空车道总是合法的，gap视为无穷大(或者是特定的值)
                    valid_lanes.append((l_idx, -1))
                else:
                    tail_car = lane[-1]
                    tail_rank = target_rank[tail_car]
                    if tail_rank < my_target:
                        # 合法：我比前车晚出，符合FIFO
                        valid_lanes.append((l_idx, my_target - tail_rank))

            if valid_lanes:
                # 贪心选择：选择 gap 最小的（最紧凑），防止浪费空间
                # 也就是找 tail_rank 最大的那个
                valid_lanes.sort(key=lambda x: x[1])  # gap 小的在前
                best_lane = valid_lanes[0][0]
                lanes[best_lane].append(car_id)
            else:
                # 所有车道的末尾车都比我晚出 -> 我被堵住了
                # 必须使用返回道 (Penalty)
                # 实际上返回道操作很复杂，这里做简化惩罚：
                # 随便塞进一个车道，但记录一次违规
                return_lane_usage += 1
                # 强行塞入最短的车道以保持平衡
                shortest_lane = min(range(self.k_lanes), key=lambda x: len(lanes[x]))
                lanes[shortest_lane].append(car_id)

        # [cite_start]
        z3_score = 100 - return_lane_usage  # [cite: 73]

        # Z4 (时间) 近似：
        # 如果返回道使用少，且序列接近原始顺序，时间分高。
        # SA中很难精确算秒，用 Z3 的结果作为强相关代理，或者简单扣分
        z4_score = 100 - (return_lane_usage * 0.5)  # 近似惩罚

        # 加权总分
        total = (self.w1 * z1_score +
                 self.w2 * z2_score +
                 self.w3 * z3_score +
                 self.w4 * z4_score)

        return total, (z1_score, z2_score, z3_score, z4_score)

    # ==========================
    # 2. 模拟退火主循环
    # ==========================
    def solve(self, initial_temp=1000, cooling_rate=0.995, max_iter=20000):
        # 1. 初始化解
        # 初始解：直接用原始顺序，或者简单的随机打乱
        current_sol = self.input_ids.copy()

        # 简单的启发式初始化：按 Z1 规则微调一下？
        # 不，SA 足够强大，从原始顺序开始即可

        current_score, _ = self.calculate_score(current_sol)
        best_sol = list(current_sol)
        best_score = current_score

        scores_history = []
        temp = initial_temp

        print(f"SA 开始... 初始分: {current_score:.2f}, 车辆数: {self.n}")

        for i in range(max_iter):
            # 2. 产生新解 (邻域变换)
            new_sol = list(current_sol)
            op = random.random()

            idx1 = random.randint(0, self.n - 1)
            idx2 = random.randint(0, self.n - 1)

            if op < 0.5:
                # Swap: 交换两车
                new_sol[idx1], new_sol[idx2] = new_sol[idx2], new_sol[idx1]
            elif op < 0.8:
                # Insert: 把 idx1 插到 idx2 后面
                val = new_sol.pop(idx1)
                new_sol.insert(idx2, val)
            else:
                # Reverse: 翻转一段区间
                s, e = min(idx1, idx2), max(idx1, idx2)
                new_sol[s:e + 1] = new_sol[s:e + 1][::-1]

            # 3. 计算新分
            new_score, _ = self.calculate_score(new_sol)

            # 4. 接受准则 (Metropolis)
            delta = new_score - current_score

            if delta > 0:
                accept = True
            else:
                # 避免除以0
                prob = math.exp(delta / max(temp, 1e-5))
                accept = random.random() < prob

            if accept:
                current_sol = new_sol
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_sol = list(current_sol)
                    # print(f"Iter {i}: 新高分 {best_score:.2f}")

            scores_history.append(current_score)
            temp *= cooling_rate

            if i % 2000 == 0:
                print(f"Iter {i}/{max_iter}, Temp={temp:.2f}, Cur={current_score:.2f}, Best={best_score:.2f}")

        return best_sol, best_score, scores_history


# ==========================
# 3. 运行脚本
# ==========================
if __name__ == "__main__":
    # 读取全量数据
    # 请确保 data_preparation.py 在同一目录
    full_data = load_data('data/附件1.xlsx')

    # 限制 N (可选，如果想先测 N=100)
    # data_to_use = full_data[:100]
    data_to_use = full_data  # 跑全量

    # 初始化优化器
    sa = PBS_SimulatedAnnealing(data_to_use)

    # 运行
    # 建议 max_iter 设大一点 (如 50000 或 100000) 以获得更好结果
    best_seq, best_val, history = sa.solve(max_iter=50000, initial_temp=500)

    # 最终评估
    final_total, details = sa.calculate_score(best_seq)
    print("\n" + "=" * 40)
    print(f"最终最优分: {final_total:.4f}")
    print(f"分项得分: Z1={details[0]}, Z2={details[1]}, Z3={details[2]}, Z4={details[3]}")
    print("=" * 40)

    # 简单的可视化
    plt.plot(history)
    plt.title("SA Optimization Process")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.show()

    # 输出结果到 Excel (Result11.xlsx)
    # 这里只打印 ID 序列，你可以自行将其填入模板
    print("Best Sequence (Car IDs):", best_seq)

    # 实例化详细仿真器
    from save_results import PBS_DetailedSimulator
    # 注意：这里传入的是 SA 算出来的 best_seq
    simulator = PBS_DetailedSimulator(data_to_use, best_seq)

    # 保存结果
    simulator.save_to_excel("result11.xlsx")

    print("全部完成！请查看目录下的 result11.xlsx 文件。")