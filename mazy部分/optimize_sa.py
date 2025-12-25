import math
import random
import copy
import time

# 尝试导入数据读取模块
try:
    from data_preparation import load_data
except ImportError:
    pass


class PBS_SimulatedAnnealing:
    def __init__(self, cars_data, lanes=6):
        self.cars = cars_data
        self.n = len(cars_data)
        self.k_lanes = lanes

        # 预处理属性
        self.H_dict = {c['id']: c['H'] for c in cars_data}
        self.D_dict = {c['id']: c['D'] for c in cars_data}
        self.input_ids = [c['id'] for c in cars_data]

        # 权重 (题目给定)
        self.w1, self.w2, self.w3, self.w4 = 0.4, 0.3, 0.2, 0.1

    def calculate_score(self, sequence_indices):
        """
        [V4.0 动态流动版]
        修复了之前只进不出导致车道假满的问题。
        加入了出车逻辑，正确模拟车道空位的释放。
        """
        # --- Z1 & Z2 计算保持不变 ---
        z1_penalties = 0
        h_ranks = [r for r, uid in enumerate(sequence_indices) if self.H_dict[uid] == 1]
        for i in range(len(h_ranks) - 1):
            if h_ranks[i + 1] - h_ranks[i] - 1 != 2:
                z1_penalties += 1
        z1_score = 100 - z1_penalties

        z2_penalties = 0
        seq_d = [self.D_dict[uid] for uid in sequence_indices]
        if seq_d:
            blocks = []
            block_start = 0
            first_type = seq_d[0]
            for i in range(1, len(seq_d)):
                cut = False
                curr, prev = seq_d[i], seq_d[i - 1]
                if first_type == 1:
                    if prev == 0 and curr == 1: cut = True
                else:
                    if prev == 1 and curr == 0: cut = True
                if cut:
                    blocks.append(seq_d[block_start:i])
                    block_start = i
                    first_type = curr
            blocks.append(seq_d[block_start:])
            for b in blocks:
                if sum(b) != len(b) - sum(b):
                    z2_penalties += 1
        z2_score = 100 - z2_penalties

        # --- Z3: 动态仿真 (Dynamic Heuristic) ---

        # 建立目标映射：车ID -> 目标出库顺序(0, 1, 2...)
        # 这让我们能快速知道某辆车是不是当前需要的
        car_target_rank = {uid: r for r, uid in enumerate(sequence_indices)}

        # 车道状态：使用列表模拟队列
        lanes = [[] for _ in range(self.k_lanes)]

        return_lane_usage = 0

        # 记录那些被迫进入返回道的车 (作为虚拟缓存)
        # 这样它们就不会阻塞进车道，且能在需要时被虚拟"移出"
        cars_in_return = set()

        # 指针：当前总装需要的下一辆车的 Target Rank (0..N-1)
        next_needed_rank = 0

        # 虚拟容量：留1个缓冲位
        VIRTUAL_CAPACITY = 9

        for car_id in self.input_ids:
            my_rank = car_target_rank[car_id]

            # --- 1. 尝试出车 (Before Placement) ---
            # 检查是否有车道头部的车是当前需要的，或者是返回道里的车
            while next_needed_rank < self.n:
                found = False

                # A. 检查返回道虚拟缓存
                # 实际上我们需要通过 ID 找车，这里简化逻辑：
                # 如果当前需要的车 sequence_indices[next_needed_rank] 在 return 集合里
                needed_id = sequence_indices[next_needed_rank]
                if needed_id in cars_in_return:
                    cars_in_return.remove(needed_id)
                    next_needed_rank += 1
                    found = True

                # B. 检查进车道头部
                if not found:
                    for k in range(self.k_lanes):
                        if lanes[k]:
                            head_rank = lanes[k][0]  # 存储的是 rank
                            if head_rank == next_needed_rank:
                                lanes[k].pop(0)  # 出车！腾出空位
                                next_needed_rank += 1
                                found = True
                                break

                if not found:
                    break  # 当前需要的车还没进站，或者被堵在后面了，停止出车

            # --- 2. 进车决策 (Placement) ---
            best_lane = -1
            min_gap = float('inf')

            # 寻找最佳车道
            for k in range(self.k_lanes):
                # 检查容量 (此时已经有车离开了，所以容量可能是新的)
                if len(lanes[k]) >= VIRTUAL_CAPACITY:
                    continue

                # 检查 FIFO
                if not lanes[k]:
                    # 空车道
                    if min_gap == float('inf'):
                        best_lane = k
                        min_gap = 99999
                    continue

                tail_rank = lanes[k][-1]
                if tail_rank < my_rank:  # 满足单调性
                    gap = my_rank - tail_rank
                    if gap < min_gap:
                        min_gap = gap
                        best_lane = k

            # 执行操作
            if best_lane != -1:
                lanes[best_lane].append(my_rank)
            else:
                # 必须进返回道
                return_lane_usage += 1
                # 标记该车进入返回道虚拟缓存，不放入普通车道，避免阻塞
                cars_in_return.add(car_id)

        # --- 3. 扫尾出车 ---
        # 输入结束后，继续处理剩余车辆
        while next_needed_rank < self.n:
            found = False
            needed_id = sequence_indices[next_needed_rank]

            if needed_id in cars_in_return:
                cars_in_return.remove(needed_id)
                next_needed_rank += 1
                found = True

            if not found:
                for k in range(self.k_lanes):
                    if lanes[k] and lanes[k][0] == next_needed_rank:
                        lanes[k].pop(0)
                        next_needed_rank += 1
                        found = True
                        break

            if not found:
                # 死锁：需要的车被堵在某个车道的后面了
                # 这种情况在使用了返回道逻辑(Item 2 else分支)后应该很少见
                # 但如果发生了，说明顺序严重违规
                remaining = self.n - next_needed_rank
                return_lane_usage += remaining * 2  # 剩余所有车都算违规
                break

        # --- Z3 评分 (分段惩罚) ---
        base_z3 = 100
        if return_lane_usage <= 10:
            penalty = return_lane_usage * 2
        elif return_lane_usage <= 20:
            penalty = 20 + ((return_lane_usage - 10) * 20)
        else:
            penalty = 20 + 200 + ((return_lane_usage - 20) * 100)

        z3_score = base_z3 - penalty
        z4_score = 100 - (return_lane_usage * 0.5)

        total_score = (self.w1 * z1_score +
                       self.w2 * z2_score +
                       self.w3 * z3_score +
                       self.w4 * z4_score)

        return total_score, (z1_score, z2_score, z3_score, z4_score)

    def get_neighbor(self, sequence):
        new_seq = sequence[:]
        n = len(new_seq)

        r = random.random()

        # 80% 概率做插入 (Insertion) - 对修补序列更有效
        if r < 0.8:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            val = new_seq.pop(i)
            new_seq.insert(j, val)

        # 10% 概率做小范围交换 (Local Swap) - 微调
        elif r < 0.9:
            i = random.randint(0, n - 2)
            # 只和后面几个邻居换，保持大结构不变
            offset = random.randint(1, min(10, n - 1 - i))
            j = i + offset
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

        # 10% 概率做远距离交换 (Global Swap) - 跳出局部最优
        else:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

        return new_seq

    def solve(self, max_iter=20000, initial_temp=500, cooling_rate=0.9995):
        print("正在生成滑动窗口贪婪初始解...")
        current_sol = self.generate_greedy_initial_solution()

        current_score, details = self.calculate_score(current_sol)
        print(f"初始贪婪分: {current_score:.2f} (Z3={details[2]})")

        # [关键修改] 如果贪婪解还是很烂(说明窗口策略导致了拥堵)，回退到 FIFO
        if current_score < -100:
            print("贪婪解风险过高，回退到原始 FIFO 顺序作为起点...")
            current_sol = self.input_ids[:]  # 回退到原始顺序
            current_score, _ = self.calculate_score(current_sol)
            print(f"FIFO 初始分: {current_score:.2f}")

        best_sol = current_sol[:]
        best_score = current_score

        # ... (后续代码不变)

        # ... (后续循环逻辑不变)

        temp = initial_temp
        scores_history = []

        print(f"SA 开始... 初始分: {current_score:.2f}, 车辆数: {self.n}")

        for i in range(max_iter):
            neighbor_sol = self.get_neighbor(current_sol)
            neighbor_score, _ = self.calculate_score(neighbor_sol)

            diff = neighbor_score - current_score

            if diff > 0 or random.random() < math.exp(diff / max(temp, 1e-5)):
                current_sol = neighbor_sol
                current_score = neighbor_score

                if current_score > best_score:
                    best_score = current_score
                    best_sol = current_sol[:]

            scores_history.append(current_score)
            temp *= cooling_rate

            if i % 5000 == 0:
                print(f"Iter {i}/{max_iter}, Temp={temp:.2f}, Cur={current_score:.2f}, Best={best_score:.2f}")

        self.best_solution = best_sol
        return best_sol, best_score, scores_history

    def run(self):
        # 接口适配
        if hasattr(self, 'solve'):
            res = self.solve()
            if isinstance(res, tuple) and len(res) >= 1:
                final_seq = res[0]
            elif isinstance(res, list):
                final_seq = res
            else:
                return []

        if final_seq and len(final_seq) > 0:
            if isinstance(final_seq[0], dict):
                return [x['id'] for x in final_seq]
            else:
                return final_seq
        return []

    def generate_greedy_initial_solution(self):
        """
        [V2.0 滑动窗口贪婪]
        只在有限的视窗内寻找最优车，保证物理可行性。
        """
        # 复制一份待排车辆ID列表 (保持原始顺序)
        remaining_ids = self.input_ids[:]
        greedy_seq = []

        # 状态追踪：上一次加入的是否为混动
        # 我们希望达成模式：H, N, N, H, N, N...
        # 记录距离上一次混动过去了多久
        gap_counter = 999  # 初始设大，允许第一辆是混动

        # 窗口大小：建议设为 12-18 (对应物理缓存能力)
        WINDOW_SIZE = 15

        while remaining_ids:
            # 1. 确定当前窗口 (只能从这些车里选)
            current_window = remaining_ids[:WINDOW_SIZE]

            best_idx = -1

            # 2. 决策逻辑
            # 我们想要：如果 gap < 2，必须选非混动(0)；如果 gap >= 2，优先选混动(1)

            # 寻找混动候选 和 非混动候选
            candidate_h = -1
            candidate_n = -1

            for i, uid in enumerate(current_window):
                h_val = self.H_dict[uid]
                if h_val == 1 and candidate_h == -1: candidate_h = i
                if h_val == 0 and candidate_n == -1: candidate_n = i
                if candidate_h != -1 and candidate_n != -1: break

            # 择优
            if gap_counter < 2:
                # 必须选非混动，攒间隔
                if candidate_n != -1:
                    best_idx = candidate_n
                else:
                    # 窗口里全是混动，没辙，只能硬选第一个
                    best_idx = 0
            else:
                # 间隔够了，可以选混动了 (优先选混动以消化库存)
                if candidate_h != -1:
                    best_idx = candidate_h
                elif candidate_n != -1:
                    best_idx = candidate_n
                else:
                    best_idx = 0

            # 3. 执行选择
            selected_uid = remaining_ids.pop(best_idx)
            greedy_seq.append(selected_uid)

            # 4. 更新状态
            if self.H_dict[selected_uid] == 1:
                gap_counter = 0
            else:
                gap_counter += 1

        return greedy_seq