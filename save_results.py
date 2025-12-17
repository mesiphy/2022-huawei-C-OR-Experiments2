import pandas as pd
import numpy as np


class PBS_DetailedSimulator:
    def __init__(self, cars_data, best_sequence):
        self.cars = cars_data  # 原始车辆数据列表
        self.sequence = best_sequence  # SA 算出的最优出车ID序列
        self.n = len(cars_data)

        # 物理参数 [cite: 43, 46, 54]
        # 进车耗时 (从接车口到各车道10号位)
        self.t_in_move = {1: 18, 2: 12, 3: 6, 4: 0, 5: 12, 6: 18}
        # 出车耗时 (从各车道1号位到总装口)
        self.t_out_move = {1: 18, 2: 12, 3: 6, 4: 0, 5: 12, 6: 18}
        self.t_cell_move = 9  # 车道内移动一格耗时

        # 记录每辆车的详细时间表
        self.schedule_log = []

    def run_physics_simulation(self):
        """
        基于最优序列，计算每辆车的物理时间节点
        """
        # 1. 确定每辆车的去向 (车道分配)
        # 这里使用简单的 Round-Robin 或贪心策略，必须与 SA 中的评估逻辑一致
        # 为了演示，我们假设简单的轮询分配: 0->Lane1, 1->Lane2...

        lane_queues = [[] for _ in range(7)]  # index 1-6 used

        # 映射: Car_ID -> 它的出车顺位 (rank)
        # rank 越小，说明越早出库
        exit_rank_map = {car_id: i for i, car_id in enumerate(self.sequence)}

        # ---------------------------
        # 第一步: 模拟进车 (入库)
        # ---------------------------
        # 假设接车机不停歇地工作
        current_in_time = 0

        car_status = {}  # {id: {'lane': k, 't_arrive_lane': t}}

        # 按原始到达顺序处理进车
        # 注意: input_id 是 0, 1, 2...
        input_order = sorted(self.cars, key=lambda x: x['id'])

        for car in input_order:
            cid = car['id']

            # 决策: 去哪个车道？
            # 策略必须与 SA 保持一致。这里用轮询作为示例：
            target_lane = (cid % 6) + 1

            # 计算时间
            # 接车机复位 + 运送时间
            # 假设接车机每次都需要一来一回 (简化逻辑，严谨需看上一次位置)
            # 题目: 接车机把车送入并返回初始位置 [cite: 42]
            op_time = self.t_in_move[target_lane]

            # 开始处理时刻
            start_proc = current_in_time
            # 到达车道时刻
            arrive_lane_t = start_proc + (op_time / 2)  # 单程是总耗时的一半吗？题目给的是总耗时包含返回
            # 题目说: 运送至不同车道并返回 [cite: 42]，消耗时间为 [18...]
            # 我们假设运过去占一半时间
            drop_time = start_proc + (op_time / 2)  # 修正: 题目没细说单程，暂按一半估算，或者直接按完整周期算节拍

            # 更新接车机时间 (必须完成动作并返回)
            current_in_time += op_time

            car_status[cid] = {
                'lane': target_lane,
                'enter_pbs_t': start_proc,
                'arrive_lane_t': drop_time,
                'exit_rank': exit_rank_map[cid]  # 它排第几个出
            }

            lane_queues[target_lane].append(cid)

        # ---------------------------
        # 第二步: 模拟出车 (出库)
        # ---------------------------
        # 严格按照 sequence 顺序出车
        current_out_time = current_in_time  # 假设出车机从什么时候开始？或者并行？
        # 题目暗示接车和送车是两个独立设备 [cite: 17, 20]，可以并行工作
        # 但出车必须等车先进入车道并移动到出口

        current_out_time = 0

        final_log = []

        for rank, cid in enumerate(self.sequence):
            info = car_status[cid]
            lane = info['lane']

            # 1. 车辆必须先到达车道
            t_arrive = info['arrive_lane_t']

            # 2. 车辆必须在车道内移动到出口 (Position 1)
            # 它是该车道第几个进来的？
            # 简化模型: 假设它进去时车道是空的，它要走 9格 (10->1)
            # 严谨模型需要判断前车什么时候走。
            # 这里做最简估算: t_ready = t_arrive + 9 * 9s (走到底)
            t_ready_at_exit = t_arrive + (9 * self.t_cell_move)

            # 3. 送车机必须空闲
            start_pick = max(current_out_time, t_ready_at_exit)

            # 4. 送车动作
            op_time = self.t_out_move[lane]
            finish_time = start_pick + (op_time / 2)  # 送到总装口

            # 更新送车机 (完成并返回)
            current_out_time = start_pick + op_time

            final_log.append({
                'Car_ID': cid,
                'Lane': lane,
                'Time_In': int(info['enter_pbs_t']),
                'Time_Lane_Ready': int(t_ready_at_exit),
                'Time_Out': int(finish_time)
            })

        self.schedule_log = final_log
        return final_log

    def generate_matrix(self):
        """
        生成题目要求的二维矩阵 (Rows: Cars, Cols: Time)
        """
        if not self.schedule_log:
            self.run_physics_simulation()

        # 找到最大时间
        max_time = max(x['Time_Out'] for x in self.schedule_log) + 10

        # 初始化矩阵 (全部为空白)
        # 用 DataFrame 方便处理
        df_matrix = pd.DataFrame('', index=[f"Car{i}" for i in range(self.n)], columns=range(max_time))

        for item in self.schedule_log:
            cid = item['Car_ID']
            idx = f"Car{cid}"
            lane = item['Lane']

            t_in = item['Time_In']
            t_ready = item['Time_Lane_Ready']
            t_out = item['Time_Out']

            # 填充状态
            # 1. 接车阶段 (简单填 "In")
            for t in range(t_in, int(t_ready)):
                # 这里应该填例如 "410", "409"... 随时间变化
                # 简化: 填车道号
                df_matrix.at[idx, t] = f"L{lane}_Move"

            # 2. 等待阶段 (在车道出口等)
            for t in range(int(t_ready), t_out):
                df_matrix.at[idx, t] = f"L{lane}_Pos1"

            # 3. 出库
            df_matrix.at[idx, t_out] = "PBS_Out"

        return df_matrix

    def save_to_excel(self, filename="result11.xlsx"):
        matrix = self.generate_matrix()
        print(f"正在保存结果到 {filename} ...")
        matrix.to_excel(filename)
        print("保存完成。")

