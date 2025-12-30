"""
生产线仿真模块 - 模拟车辆通过 PBS 缓冲区的过程
"""
import numpy as np
from .trajectory_recorder import TrajectoryRecorder


class ProductionLineSimulator:
    """生产线仿真器：模拟车辆在车道和返回道中的流动"""
    
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
        执行仿真过程
        
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
        # 重塑决策向量
        decision_vector = decision_vector.reshape(-1, 2)

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
        is_start = np.zeros(m)
        is_start[0] = 1  ###

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

                ###这里根据来源不同进行不同的计算可选择车道
                ###计算可选择车道这里有问题

                selected_lane_index = '等待'

                if pbs_vehicle_cursor < m:
                    if is_return_lane_10_occupied == 0:  ###当返回道的 10 车位为空时，
                        available_lanes_indices.append('等待')
                        selected_lane_val = decision_vector[current_step, 0] * len(
                            available_lanes_indices)  ###根据既定任务选择车道
                        selected_lane_index = available_lanes_indices[int(selected_lane_val)]  ###选择进哪个车道

                if is_return_lane_10_occupied == 1:  ###返回车道不为空
                    selected_lane_val = decision_vector[current_step, 0] * len(available_lanes_indices)  ###根据既定任务选择车道
                    selected_lane_index = available_lanes_indices[int(selected_lane_val)]

                if selected_lane_index != '等待':  ###如果不选择等待
                    time_shuttle_need = self.shuttle_in_out_times[selected_lane_index]  ###所选择的车道需要花的时间
                    time_inbound_start = current_step  ###输入横移机开始工作的时间

                    if pbs_vehicle_cursor < m:  ###仓库内还有车
                        if is_return_lane_10_occupied == 0:  ###如果返回道 10 车位没车
                            is_inbound_shuttle_busy = 1  ###立刻可以拿车，将横移机置为工作状态
                            inbound_source = 0  ###记录数据来源，从仓库来的数据
                            trajectory_matrix[pbs_vehicle_cursor, current_step * 3:(current_step + 1) * 3] = 1  ###记录数据
                            car_on_inbound_shuttle = pbs_vehicle_cursor  ###输入横移机从仓库拿车，更新输入横移机上车的序号
                            pbs_vehicle_cursor += 1  ###下一辆仓库的出车序号

                if is_return_lane_10_occupied == 1:
                    inbound_source = 1  ######记录数据来源，从反车道来的数据
                    is_inbound_shuttle_busy = 0  ###此时还没装车，为了方便处理所以将输入横移机的工作状态置为 0，实际此时为 1

                    if current_step >= time_return_lane_arrival + 1:  ###输入横移机到达反车道 10 车位
                        car_on_inbound_shuttle = return_lane_vehicle_ids[0]  ###输入横移机此时携带的车的序号
                        trajectory_matrix[
                            int(car_on_inbound_shuttle), current_step * 3:(current_step + 1) * 3] = 1  ###记录结果
                        return_lane_vehicle_ids[0] = -1  ###车被搬走，所以返回道 10 车位没有车，用-1 表示
                        return_lane_occupancy[0] = 0  ###车被搬走，返回道 10 车位没车， 0 表示没车
                        is_return_lane_10_occupied = 0  ###车被搬走，返回道 10 车位没车， 0 表示没车
                        time_inbound_start = current_step  ###输入横移机从返回道接受车的时间点
                        is_inbound_shuttle_busy = 1  ###

            if is_inbound_shuttle_busy == 1:  ###如果输入横移机在工作
                if inbound_source == 0:  ###如果从仓库来的
                    if car_on_inbound_shuttle != -1:
                        trajectory_matrix[
                            int(car_on_inbound_shuttle), current_step * 3:(current_step + 1) * 3] = 1  ###记录结果
                    if current_step == (time_inbound_start + time_shuttle_need / 2):  ###如果输入横移机送到目标车道
                        inbound_shuttle_to_lane_buffer.append(selected_lane_index)  ###输入横移机刚刚把车放在指定车道
                        lane_occupancy[selected_lane_index, 0] = 1  ###将所选择车道的 10 车位置为 1
                        lane_vehicle_ids[selected_lane_index, 0] = car_on_inbound_shuttle  ###更新车道上的车的序号
                        trajectory_matrix[car_on_inbound_shuttle, current_step * 3:(current_step + 1) * 3] = (
                                                                                                                         1 + selected_lane_index) * 100 + 10  ###记录结果
                        car_on_inbound_shuttle = -1  ###输入横移机此时不再有车

                    if current_step >= time_shuttle_need + time_inbound_start:  ###输入横移机回到原来位置时
                        is_inbound_shuttle_busy = 0  ###输入横移机工作转态置为 0
                        time_return_lane_arrival = current_step  ###修改的地方

                if inbound_source == 1:
                    if current_step == time_inbound_start + self.time_return_to_lane[int(selected_lane_index)]:
                        lane_occupancy[selected_lane_index, 0] = 1  ###从返回道的 10 车位来的车，由输入横移机放到选择的车道中
                        lane_vehicle_ids[selected_lane_index, 0] = car_on_inbound_shuttle  ###更新被选择的车道上车的编号
                        inbound_shuttle_to_lane_buffer.append(selected_lane_index)  ###输入横移机刚刚把车放在指定车道
                        trajectory_matrix[
                            int(car_on_inbound_shuttle), current_step * 3:current_step * 3 + 3] = selected_lane_index * 100 + 10  ###更新结果
                        car_on_inbound_shuttle = -1  ###输入横移机上没有车

                    if current_step >= self.return_lane_transit_times[
                        selected_lane_index] - 1 + time_inbound_start:  ###输入横移车回到开始的位置时
                        is_inbound_shuttle_busy = 0
                        time_return_lane_arrival = current_step  ###修改的地方

            # --- 输出横移机逻辑 (Outbound Shuttle Logic) ---
            if is_outbound_shuttle_busy == 0 and len(lane_exit_priority_queue) != 0:  ###如果输出横移机空闲，且 1 车位内有车
                time_outbound_start = current_step  ###输出横移机开始工作的时间
                time_outbound_need = self.shuttle_in_out_times[lane_exit_priority_queue[0]] / 2  ###输出横移机需要到达指定车道所花的时间
                time_outbound_lane_to_return = self.time_return_to_lane[
                    lane_exit_priority_queue[0]]  ###输出横移机从指定车道送到返回道的时间
                is_outbound_shuttle_busy = 1  ###输出横移机工作状态置为 1

            if is_outbound_shuttle_busy == 1:
                if current_step == time_outbound_start + time_outbound_need:  ###输出横移机到达 1 车位
                    ###到达 1 车位后对输出还是回返回道进行判断
                    if return_lane_occupancy[25:].any() == 1:
                        outbound_action_flag = 1
                    if return_lane_occupancy[25:].any() != 1:
                        if decision_vector[current_step, 1] <= 0.5:
                            outbound_action_flag = 1
                        if decision_vector[current_step, 1] > 0.5:
                            outbound_action_flag = 0

                    trajectory_matrix[int(lane_vehicle_ids[int(lane_exit_priority_queue[
                                                                   0]), -1]), current_step * 3:current_step * 3 + 3] = 2  ###车到了输出横移机身上，记录结果
                    car_on_outbound_shuttle = lane_vehicle_ids[int(lane_exit_priority_queue[0]), -1]  ###输出横移机上面车的序号
                    lane_occupancy[int(lane_exit_priority_queue[0]), -1] = 0  ###车道状态改变
                    lane_vehicle_ids[int(lane_exit_priority_queue[0]), -1] = -1  ###车道上对应车位的车的序号置为-1（-1 表示没有车）
                    lane_exit_priority_queue.pop(0)  ###到达最后一个车位的车道顺序里去除被接走的车

                if current_step >= time_outbound_start + time_outbound_need:
                    ###outOrInput 这个规则及约束还没写
                    if outbound_action_flag == 0:  ###把车送到返回道
                        if current_step < time_outbound_start + time_outbound_need + time_outbound_lane_to_return:  ###车在输出横移机上
                            trajectory_matrix[int(car_on_outbound_shuttle), current_step * 3:current_step * 3 + 3] = 2
                        if current_step == time_outbound_start + time_outbound_need + time_outbound_lane_to_return:  ###把车从输出横移机放到返回道
                            arrival_count_at_return[int(car_on_outbound_shuttle)] += 1  ###车到达返回道，这表明车已用掉一次返回的机会
                            trajectory_matrix[int(car_on_outbound_shuttle), current_step * 3:current_step * 3 + 3] = 71
                            outbound_shuttle_to_return_buffer.append(27)  ###刚刚把车从输出横移机放到反车道
                            return_lane_occupancy[-1] = 1  ###返回车道的 1 车位放车
                            return_lane_vehicle_ids[-1] = car_on_outbound_shuttle  ###返回车道上的 1 车位上车的序号
                            car_on_outbound_shuttle = -1  ###输出横移机不再有车
                            return_usage_count += 1
                        if current_step == time_outbound_start + time_outbound_need + time_outbound_lane_to_return + 1:  ###输出横移机结束工作
                            is_outbound_shuttle_busy = 0

                    if outbound_action_flag == 1:
                        if current_step < time_outbound_start + 2 * time_outbound_need:  ###输出横移机还未到达指定车道，车还在横移机上
                            trajectory_matrix[int(car_on_outbound_shuttle), current_step * 3:current_step * 3 + 3] = 2
                        if current_step == time_outbound_start + 2 * time_outbound_need:  ###输出横移机成功把车送出去
                            trajectory_matrix[
                                int(car_on_outbound_shuttle), current_step * 3:current_step * 3 + 3] = 3  ###更新记录
                            output_sequence.append(int(car_on_outbound_shuttle))  ###输出队列添加上刚刚送出去的车
                            is_outbound_shuttle_busy = 0  ###输出横移机结束工作
                            car_on_outbound_shuttle = -1  ###输出横移机上没有车，置为-

            # --- 车道移动判定 (Lane Movement Check) ---
            for j in range(6):
                for j_index in range(9):
                    if j_index == 0:
                        if lane_occupancy[j, -1] == 0:  ###1 车位没车的话，这条道一定可以走
                            can_lane_move[j] = 1
                            break
                    if j_index > 0:
                        if lane_occupancy[j, 27 - j_index * 3:30 - j_index * 3].any() == 0:  ###当前车位没车的话，后面的车一定可以往前走
                            can_lane_move[j, :27 - j_index * 3] = 1
                            break

            can_lane_move = np.append(can_lane_move, np.zeros(shape=(6, 1)), axis=1)  ###1 车位不能通过车道往前走

            # --- 返回道移动判定 (Return Lane Movement Check) ---
            for j in range(9):
                if j == 0:
                    if return_lane_occupancy[0] == 0:  ###返回道 10 车位没车时，返回道都可移动
                        can_return_move[:] = 1
                        break
                if j > 0:
                    if return_lane_occupancy[j * 3 - 2: j * 3 + 1].any() == 0:  ###返回道当前车位没车时，小于该车位的车位可移动
                        can_return_move[j * 3:] = 1
                        break

            can_return_move = np.append(np.zeros(1), can_return_move)  ###返回道 10 车位的车不可通过返回道移动

            # --- 执行车道移动 (Execute Lane Move) ---
            active_vehicles_can_go = can_lane_move * lane_occupancy  ###有车且能走的车
            if inbound_shuttle_to_lane_buffer:
                active_vehicles_can_go[int(selected_lane_index), 0] = 0  ###刚刚到车道的车不能走
                inbound_shuttle_to_lane_buffer.pop()  ###取消刚刚到车道这个'刚刚'的状态

            movable_indices_lane = np.where(active_vehicles_can_go == 1)[0]
            movable_indices_chewei = np.where(active_vehicles_can_go == 1)[1]
            target_indices_chewei = movable_indices_chewei + 1

            lane_occupancy[movable_indices_lane, target_indices_chewei] = 1
            lane_occupancy[movable_indices_lane, movable_indices_chewei] = 0

            lane_vehicle_ids[movable_indices_lane, target_indices_chewei], lane_vehicle_ids[
                movable_indices_lane, movable_indices_chewei] = \
                lane_vehicle_ids[movable_indices_lane, movable_indices_chewei], lane_vehicle_ids[
                    movable_indices_lane, target_indices_chewei]

            trajectory_matrix = self.recorder.map_lane_vehicles(lane_vehicle_ids, trajectory_matrix, current_step)  ###记录车道上车的位置

            last_chewei_status = lane_occupancy[:, -1]  ###1 车位上的状态
            occupied_exit_lanes = np.where(last_chewei_status == 1)[0]  ###1 车位上有车的车道

            for j in occupied_exit_lanes:
                if j not in lane_exit_priority_queue:  ###已经在到达 1 车位的队列则不再添加
                    lane_exit_priority_queue.append(j)  ###到达 1 车位的队列添加新到达的

            # --- 执行返回道移动 (Execute Return Lane Move) ---
            active_return_vehicles_can_go = can_return_move * return_lane_occupancy  ###反车道上有车且能走的车
            if outbound_shuttle_to_return_buffer:
                active_return_vehicles_can_go[-1] = 0  ###刚刚到反车道的车不能动
                outbound_shuttle_to_return_buffer.pop()  ###取消刚刚到反车道的这个'刚刚的'状态

            movable_indices_return = np.where(active_return_vehicles_can_go == 1)[0]  ###反车道上有车的车位
            target_indices_return = movable_indices_return - 1  ###反车道上的车移动后的车位

            return_lane_occupancy[movable_indices_return] = 0
            return_lane_occupancy[target_indices_return] = 1

            if return_lane_occupancy[0] == 1:  ###返回道 10 车位有车时
                is_return_lane_10_occupied = 1  ###返回道 10 车位修改为有车

            return_lane_vehicle_ids[movable_indices_return], return_lane_vehicle_ids[target_indices_return] = \
                return_lane_vehicle_ids[target_indices_return], return_lane_vehicle_ids[
                    movable_indices_return]  ###调整反车道上车的序号

            if return_lane_occupancy[0] == 1:
                is_return_lane_10_occupied = 1
            if return_lane_occupancy[0] == 0:
                is_return_lane_10_occupied = 0

            if car_on_inbound_shuttle != -1:
                trajectory_matrix[int(car_on_inbound_shuttle), current_step * 3:current_step * 3 + 3] = 1
            if car_on_outbound_shuttle != -1:
                trajectory_matrix[int(car_on_outbound_shuttle), current_step * 3:current_step * 3 + 3] = 2

            trajectory_matrix = self.recorder.map_return_lane_vehicles(return_lane_vehicle_ids, trajectory_matrix,
                                                              current_step)  ###记录返回道上车的位置
            trajectory_matrix[output_sequence, current_step * 3:current_step * 3 + 3] = 3  ###记录车出去后，车所在的位置

            if current_step > 750:
                if len(np.where(trajectory_matrix[:, current_step * 3] != 3)[0]) == 0:
                    trajectory_matrix = trajectory_matrix[:, :(current_step + 1) * 3]
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
                decision_vector = decision_vector.reshape(-1, 2)
                # i = 0
                break

                # 重置逻辑（这里的代码在原版中位于break之后，实际上不可达，但为了"逐行修改"的原则保留结构并重命名变量）
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
                is_start = np.zeros(m)
                is_start[0] = 1  ###
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
            current_step += 1
        return trajectory_matrix, output_sequence, return_usage_count, trajectory_matrix.shape[1]
