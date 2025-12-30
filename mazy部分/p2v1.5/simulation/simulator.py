"""
生产线仿真模块 - P2 版本（4变量决策空间）
模拟车辆通过 PBS 缓冲区的过程
"""
import numpy as np
from .trajectory_recorder import TrajectoryRecorder


class ProductionLineSimulator:
    """生产线仿真器：模拟车辆在车道和返回道中的流动（P2 版本）"""
    
    def __init__(self):
        """初始化物理参数"""
        # 横移机进出时间参数
        self.shuttle_in_out_times = np.array([18, 12, 6, 0, 12, 18]) / 3
        self.return_lane_transit_times = np.array([24, 18, 12, 6, 12, 18]) / 3
        self.time_lane_to_shuttle = self.shuttle_in_out_times / 2
        self.time_return_to_lane = np.array([4, 3, 2, 1, 1, 2])
        self.base_time = 3
        
        # 轨迹记录器
        self.recorder = TrajectoryRecorder()
    
    def simulate(self, decision_vector: np.ndarray, max_steps: int, data: np.ndarray,
                 wolves_position: np.ndarray, current_wolf_index: int, 
                 dimension: int, lower_bound: float, upper_bound: float) -> tuple:
        """
        执行仿真过程（P2 版本：4变量决策空间）
        
        Args:
            decision_vector: 决策向量
            max_steps: 最大仿真步数
            data: 车辆数据
            wolves_position: 狼群位置矩阵（用于异常重启）
            current_wolf_index: 当前狼的索引
            dimension: 决策向量维度
            lower_bound: 位置下界
            upper_bound: 位置上界
            
        Returns:
            (轨迹矩阵, 输出序列, 返回道使用次数, 完成时间)
        """
        # P2 修改: reshape 为 4 列
        # Col 0: 是否启用输入横移机 (Input Hoist Activation)
        # Col 1: 输出横移机决策 (Output Hoist Decision: Return vs Out)
        # Col 2: 车道选择 (Lane Selection)
        # Col 3: 选车策略 (Car Selection Strategy for Output)
        decision_vector = decision_vector.reshape(-1, 4)

        m, n = data.shape  ###数据长度
        output_sequence = []  ###输出队列
        inbound_shuttle_to_lane_buffer = []  ###刚刚从输入横移机放入车道
        outbound_shuttle_to_return_buffer = []  ###刚刚从输出横移机放入反车道

        trajectory_matrix = np.zeros(shape=(m, int(max_steps * 3)))

        lane_vehicle_ids = np.zeros(shape=(6, 28)) - 1  ###车道上车的编号，没车就是-1
        return_lane_vehicle_ids = np.zeros(28) - 1  ###返回道上车位的编号，没车为-1

        lane_occupancy = np.zeros(shape=(6, 28))  ###车道上有无车，有车 1，没车 0
        return_lane_occupancy = np.zeros(28)  ###反车道上的状态，有车 1，没车 0

        return_usage_count = 0  ###使用返回道次数

        vehicle_status_tracker = np.zeros(shape=(m, 5))  ###0 还没出， 1 在横移机上， 2 在车道上， 3 在输出横移机， 4 在终

        vehicle_status_tracker[:, 0] = 1

        is_return_lane_10_occupied = 0  ###返回车道的第 10 车位状态， 0 为空， 1 为满
        is_inbound_shuttle_busy = 0  ###入口横移机是否在工作， 0 为空， 1 为满
        is_outbound_shuttle_busy = 0  ###出口横移机是否在工作， 0 为空， 1 为满

        time_return_lane_arrival = 0  ###返回道到达时间
        inbound_source = 0  ###输入横移机上的车的来源， 0 表示 PBS， 1 表示返回道
        lane_exit_priority_queue = []  ###到达最后一个车位的车道顺序

        outbound_action_flag = 0  ###横移机准备把车送出去还是送回返回道， 0 表示去返回道， 1 表示送出去

        car_on_inbound_shuttle = -1  ###初始化输入横移机上的车的编号， -1 表示没有车
        car_on_outbound_shuttle = -1  ###初始化输出横移机上的车编号，没车表示-1
        pbs_vehicle_cursor = 0  ###初始化 PBS->输入横移机上的编号，第一辆为 0

        arrival_count_at_return = np.zeros(m)

        target_lane = 0  # 【P2 新增】用于记忆横移机当前的工作目标车道

        current_step = 0
        while current_step < max_steps:
            can_lane_move = np.zeros(shape=(6, 27))  ###可移动的车道,1 表示可移动， 0 表示不可移动,这得对约束的理解
            can_return_move = np.zeros(27)  ###返回道是否可移动,1 表示可移动， 0 表示不可移动

            # --- 输入横移机逻辑 (Inbound Shuttle Logic) ---
            if is_inbound_shuttle_busy == 0:  ###如果输入横移机空闲
                lane_entrance_status = lane_occupancy[:, :3]  ###10 车位上的车道状态
                lane_entrance_occupied_indices = np.where(lane_entrance_status == 1)  ###10 车位上有车的索引

                lanes_needing_check = []
                available_lanes_indices = [3, 2, 4, 1, 5, 0]  ###可选择车道

                for j in range(6):
                    if j in lane_entrance_occupied_indices[0]:
                        available_lanes_indices.remove(j)  ###剩下一定能走的车道
                        lanes_needing_check.append(j)  ###需要计算时间确认的车道

                selected_lane_index = '等待'

                if pbs_vehicle_cursor < m:
                    if is_return_lane_10_occupied == 0:
                        if len(available_lanes_indices) > 0:
                            # 使用 Col 2 进行车道选择
                            selected_lane_val = decision_vector[current_step, 2] * len(available_lanes_indices)
                            idx = int(selected_lane_val)
                            if idx >= len(available_lanes_indices): idx = len(available_lanes_indices) - 1
                            selected_lane_index = available_lanes_indices[idx]
                        else:
                            selected_lane_index = '等待'

                if is_return_lane_10_occupied == 1:
                    if len(available_lanes_indices) > 0:
                        selected_lane_val = decision_vector[current_step, 2] * len(available_lanes_indices)
                        idx = int(selected_lane_val)
                        if idx >= len(available_lanes_indices): idx = len(available_lanes_indices) - 1
                        selected_lane_index = available_lanes_indices[idx]
                    else:
                        selected_lane_index = '等待'

                if selected_lane_index != '等待':
                    time_shuttle_need = self.shuttle_in_out_times[int(selected_lane_index)]
                    time_inbound_start = current_step

                    if pbs_vehicle_cursor < m:
                        # 【P2 修改】使用 Col 0 控制输入横移机是否工作
                        # 优先给返回道让路 (decision > 0.5 且返回道空)
                        if decision_vector[current_step, 0] > 0.5 and is_return_lane_10_occupied == 0:
                            is_inbound_shuttle_busy = 1
                            target_lane = selected_lane_index  # 【P2 新增】记录目标
                            inbound_source = 0
                            trajectory_matrix[pbs_vehicle_cursor, current_step * 3:(current_step + 1) * 3] = 1
                            car_on_inbound_shuttle = pbs_vehicle_cursor
                            pbs_vehicle_cursor += 1

                        # 强制优先接返回道 (decision < 0.5 或 返回道满)
                        if decision_vector[current_step, 0] < 0.5 or is_return_lane_10_occupied == 1:
                            inbound_source = 1
                            is_inbound_shuttle_busy = 1

            if current_step == time_return_lane_arrival + 1:  # 输入横移机到达反车道 10 车位
                if selected_lane_index != '等待':
                    car_on_inbound_shuttle = return_lane_vehicle_ids[0]
                    trajectory_matrix[int(car_on_inbound_shuttle), current_step * 3:(current_step + 1) * 3] = 1
                    return_lane_vehicle_ids[0] = -1
                    return_lane_occupancy[0] = 0
                    is_return_lane_10_occupied = 0
                    time_inbound_start = current_step
                    inbound_source = 1
                    is_inbound_shuttle_busy = 1
                    target_lane = selected_lane_index  # 【P2 新增】
                else:
                    time_return_lane_arrival = current_step

            if is_inbound_shuttle_busy == 1:
                if inbound_source == 0:
                    if car_on_inbound_shuttle != -1:
                        trajectory_matrix[int(car_on_inbound_shuttle), current_step * 3:(current_step + 1) * 3] = 1
                    if current_step == (time_inbound_start + time_shuttle_need / 2):
                        inbound_shuttle_to_lane_buffer.append(target_lane)  # 使用 target_lane
                        lane_occupancy[int(target_lane), 0] = 1
                        lane_vehicle_ids[int(target_lane), 0] = car_on_inbound_shuttle
                        trajectory_matrix[car_on_inbound_shuttle, current_step * 3:(current_step + 1) * 3] = (
                                                                                                                         1 + target_lane) * 100 + 10
                        car_on_inbound_shuttle = -1
                    if current_step >= time_shuttle_need + time_inbound_start:
                        is_inbound_shuttle_busy = 0
                        time_return_lane_arrival = current_step

                if inbound_source == 1:
                    if current_step == time_inbound_start + self.time_return_to_lane[
                        int(target_lane)]:  # 使用 target_lane
                        lane_occupancy[int(target_lane), 0] = 1
                        lane_vehicle_ids[int(target_lane), 0] = car_on_inbound_shuttle
                        inbound_shuttle_to_lane_buffer.append(target_lane)
                        trajectory_matrix[
                            int(car_on_inbound_shuttle), current_step * 3:current_step * 3 + 3] = target_lane * 100 + 10
                        car_on_inbound_shuttle = -1
                    if current_step >= self.return_lane_transit_times[int(target_lane)] - 1 + time_inbound_start:
                        is_inbound_shuttle_busy = 0
                        time_return_lane_arrival = current_step

            # --- 输出横移机逻辑 (Outbound Shuttle Logic) ---
            if is_outbound_shuttle_busy == 0 and len(lane_exit_priority_queue) != 0:
                time_outbound_start = current_step
                # 使用 Col 3 决定出车顺序 (虽然代码里好像只用了 FIFO/Priority Queue, 这里用了 decision_vector[3] 来选 fan?)
                # 原代码: choosefan = renwu[i, 3] * len(arriveIndex)
                choose_fan_idx = decision_vector[current_step, 3] * len(lane_exit_priority_queue)
                choose_fan_idx = int(choose_fan_idx)
                # 注意：原代码此处可能有逻辑风险，这里仅做变量重命名，保持原逻辑
                if choose_fan_idx >= len(lane_exit_priority_queue):
                    choose_fan_idx = len(lane_exit_priority_queue) - 1

                # 这里的逻辑在原代码中似乎是用 choosefan 来索引 time_in_out，但这应该取决于车道？
                # 原代码: timeOutputHyjNeed = self.time_in_out[choosefan] / 2
                # 这里 choosefan 看起来像是从 priority_queue 里选一个车道的索引?
                # 假设 renwu[3] 是用来在 queue 里挑车的
                selected_exit_lane = lane_exit_priority_queue[0]  # 默认 FIFO，这里没有完全照搬 renwu[3] 的随机挑选逻辑，因为原代码 pop(0) 是固定的

                time_outbound_need = self.shuttle_in_out_times[selected_exit_lane] / 2
                time_outbound_lane_to_return = self.time_return_to_lane[selected_exit_lane]
                is_outbound_shuttle_busy = 1

            if is_outbound_shuttle_busy == 1:
                if current_step == time_outbound_start + time_outbound_need:
                    if return_lane_occupancy[25:].any() == 1:
                        outbound_action_flag = 1
                    if return_lane_occupancy[25:].any() != 1:
                        # 使用 Col 1 决定是送出还是回返回道
                        if decision_vector[current_step, 1] <= 0.5:
                            outbound_action_flag = 1
                        if decision_vector[current_step, 1] > 0.5:
                            outbound_action_flag = 0

                    trajectory_matrix[int(lane_vehicle_ids[int(
                        lane_exit_priority_queue[0]), -1]), current_step * 3:current_step * 3 + 3] = 2
                    car_on_outbound_shuttle = lane_vehicle_ids[int(lane_exit_priority_queue[0]), -1]
                    lane_occupancy[int(lane_exit_priority_queue[0]), -1] = 0
                    lane_vehicle_ids[int(lane_exit_priority_queue[0]), -1] = -1
                    lane_exit_priority_queue.pop(0)

                if current_step >= time_outbound_start + time_outbound_need:
                    if outbound_action_flag == 0:  # 送到返回道
                        if current_step < time_outbound_start + time_outbound_need + time_outbound_lane_to_return:
                            trajectory_matrix[int(car_on_outbound_shuttle), current_step * 3:current_step * 3 + 3] = 2

                        if current_step == time_outbound_start + time_outbound_need + time_outbound_lane_to_return:
                            arrival_count_at_return[int(car_on_outbound_shuttle)] += 1
                            trajectory_matrix[int(car_on_outbound_shuttle), current_step * 3:current_step * 3 + 3] = 71
                            outbound_shuttle_to_return_buffer.append(27)
                            return_lane_occupancy[-1] = 1
                            return_lane_vehicle_ids[-1] = car_on_outbound_shuttle
                            car_on_outbound_shuttle = -1
                            return_usage_count += 1
                        if current_step == time_outbound_start + time_outbound_need + time_outbound_lane_to_return + 1:
                            is_outbound_shuttle_busy = 0

                    if outbound_action_flag == 1:  # 送出
                        if current_step < time_outbound_start + 2 * time_outbound_need:
                            trajectory_matrix[int(car_on_outbound_shuttle), current_step * 3:current_step * 3 + 3] = 2

                        if current_step == time_outbound_start + 2 * time_outbound_need:
                            trajectory_matrix[int(car_on_outbound_shuttle), current_step * 3:current_step * 3 + 3] = 3
                            output_sequence.append(int(car_on_outbound_shuttle))
                            is_outbound_shuttle_busy = 0
                            car_on_outbound_shuttle = -1

            # --- 车道移动判定 (Lane Movement Check) ---
            for j in range(6):
                for j_index in range(9):
                    if j_index == 0:
                        if lane_occupancy[j, -1] == 0:
                            can_lane_move[j] = 1
                            break
                    if j_index > 0:
                        if lane_occupancy[j, 27 - j_index * 3:30 - j_index * 3].any() == 0:
                            can_lane_move[j, :27 - j_index * 3] = 1
                            break

            can_lane_move = np.append(can_lane_move, np.zeros(shape=(6, 1)), axis=1)

            # --- 返回道移动判定 (Return Lane Movement Check) ---
            for j in range(9):
                if j == 0:
                    if return_lane_occupancy[0] == 0:
                        can_return_move[:] = 1
                        break
                if j > 0:
                    if return_lane_occupancy[j * 3 - 2: j * 3 + 1].any() == 0:
                        can_return_move[j * 3:] = 1
                        break

            can_return_move = np.append(np.zeros(1), can_return_move)

            # --- 执行车道移动 (Execute Lane Move) ---
            active_vehicles_can_go = can_lane_move * lane_occupancy
            if inbound_shuttle_to_lane_buffer:
                active_vehicles_can_go[int(target_lane), 0] = 0  # 使用 target_lane
                inbound_shuttle_to_lane_buffer.pop()

            movable_indices_lane = np.where(active_vehicles_can_go == 1)[0]
            movable_indices_chewei = np.where(active_vehicles_can_go == 1)[1]
            target_indices_chewei = movable_indices_chewei + 1

            lane_occupancy[movable_indices_lane, target_indices_chewei] = 1
            lane_occupancy[movable_indices_lane, movable_indices_chewei] = 0

            lane_vehicle_ids[movable_indices_lane, target_indices_chewei], lane_vehicle_ids[
                movable_indices_lane, movable_indices_chewei] = \
                lane_vehicle_ids[movable_indices_lane, movable_indices_chewei], lane_vehicle_ids[
                    movable_indices_lane, target_indices_chewei]

            trajectory_matrix = self.recorder.map_lane_vehicles(lane_vehicle_ids, trajectory_matrix, current_step)

            last_chewei_status = lane_occupancy[:, -1]
            occupied_exit_lanes = np.where(last_chewei_status == 1)[0]

            for j in occupied_exit_lanes:
                if j not in lane_exit_priority_queue:
                    lane_exit_priority_queue.append(j)

            # --- 执行返回道移动 (Execute Return Lane Move) ---
            active_return_vehicles_can_go = can_return_move * return_lane_occupancy
            if outbound_shuttle_to_return_buffer:
                active_return_vehicles_can_go[-1] = 0
                outbound_shuttle_to_return_buffer.pop()

            movable_indices_return = np.where(active_return_vehicles_can_go == 1)[0]
            target_indices_return = movable_indices_return - 1

            return_lane_occupancy[movable_indices_return] = 0
            return_lane_occupancy[target_indices_return] = 1

            if return_lane_occupancy[0] == 1:
                is_return_lane_10_occupied = 1

            return_lane_vehicle_ids[movable_indices_return], return_lane_vehicle_ids[target_indices_return] = \
                return_lane_vehicle_ids[target_indices_return], return_lane_vehicle_ids[movable_indices_return]

            if return_lane_occupancy[0] == 1:
                is_return_lane_10_occupied = 1
            if return_lane_occupancy[0] == 0:
                is_return_lane_10_occupied = 0

            if car_on_inbound_shuttle != -1:
                trajectory_matrix[int(car_on_inbound_shuttle), current_step * 3:current_step * 3 + 3] = 1
            if car_on_outbound_shuttle != -1:
                trajectory_matrix[int(car_on_outbound_shuttle), current_step * 3:current_step * 3 + 3] = 2

            trajectory_matrix = self.recorder.map_return_lane_vehicles(return_lane_vehicle_ids, trajectory_matrix,
                                                              current_step)
            trajectory_matrix[output_sequence, current_step * 3:current_step * 3 + 3] = 3

            if current_step > 750:
                if len(np.where(trajectory_matrix[:, current_step * 3] != 3)[0]) == 0:
                    trajectory_matrix = trajectory_matrix[:, :(current_step + 1) * 3]
                    print('运行成功')
                    break

            # 异常重启机制
            if current_step == max_steps - 2:
                # print(0)
                # print(current_step)
                print("运行失败")
                wolves_position[current_wolf_index] = np.random.uniform(low=lower_bound,
                                                                             high=upper_bound,
                                                                             size=(dimension,))
                decision_vector = wolves_position[current_wolf_index]
                decision_vector = decision_vector.reshape(-1, 4)  # P2 修改：reshape 为 4 列
                # i = 0
                break

                # 重置逻辑（这里的代码在原版中位于break之后，实际上不可达，但为了保留结构）
                output_sequence = []  ###输出队列
                inbound_shuttle_to_lane_buffer = []  ###刚刚从输入横移机放入车道
                outbound_shuttle_to_return_buffer = []  ###刚刚从输出横移机放入反车道
                trajectory_matrix = np.zeros(shape=(m, int(max_steps * 3)))
                lane_vehicle_ids = np.zeros(shape=(6, 28)) - 1  ###车道上车的编号，没车就是-1
                return_lane_vehicle_ids = np.zeros(28) - 1  ###返回道上车位的编号，没车为-1
                lane_occupancy = np.zeros(shape=(6, 28))  ###车道上有无车，有车 1，没车 0
                return_lane_occupancy = np.zeros(28)  ###反车道上的状态，有车 1，没车 0
                return_usage_count = 0  ###使用返回道次数
                vehicle_status_tracker = np.zeros(shape=(m, 5))  ###0 还没出， 1 在横移机上， 2 在车道上， 3 在输出横移机， 4在终点
                vehicle_status_tracker[:, 0] = 1
                is_return_lane_10_occupied = 0  ###返回车道的第 10 车位状态， 0 为空， 1 为满
                is_inbound_shuttle_busy = 0  ###入口横移机是否在工作， 0 为空， 1 为满
                is_outbound_shuttle_busy = 0  ###出口横移机是否在工作， 0 为空， 1 为满
                time_return_lane_arrival = 0  ###返回道到达时间
                inbound_source = 0  ###输入横移机上的车的来源， 0 表示 PBS， 1 表示返回道
                lane_exit_priority_queue = []  ###到达最后一个车位的车道顺序
                outbound_action_flag = 0  ###横移机准备把车送出去还是送回返回道， 0 表示去返回道， 1 表示送出去
                car_on_inbound_shuttle = -1  ###初始化输入横移机上的车的编号， -1 表示没有车
                car_on_outbound_shuttle = -1  ###初始化输出横移机上的车编号，没车表示-1
                pbs_vehicle_cursor = 0  ###初始化 PBS->输入横移机上的编号，第一辆为 0
                arrival_count_at_return = np.zeros(m)
                target_lane = 0  # 【P2 新增】重置 target_lane

            current_step += 1

        return trajectory_matrix, output_sequence, return_usage_count, trajectory_matrix.shape[1]
