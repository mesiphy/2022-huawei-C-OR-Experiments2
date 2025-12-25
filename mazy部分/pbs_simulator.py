import pandas as pd
import math

# ==========================================
# 1. 全局配置
# ==========================================
TIME_PARAMS = {
    'move': 9,
    't_in': [18, 12, 6, 0, 12, 18],
    't_from_return': [24, 18, 12, 6, 12, 18],
    't_out': [18, 12, 6, 0, 12, 18],
    't_to_return': [24, 18, 12, 6, 12, 18]
}


class Car:
    def __init__(self, car_id, original_rank, type_h, type_d):
        self.id = car_id
        self.original_rank = original_rank
        self.type_h = type_h
        self.type_d = type_d

        # location 格式示例:
        # ('PAINT',) -> 长度1
        # ('TRANS_IN',) -> 长度1
        # ('RETURN', pos) -> 长度2
        # ('LANE', lane_idx, pos) -> 长度3
        self.current_location = ('PAINT',)
        self.is_moving = False
        self.move_end_time = 0
        self.arrival_at_lane_head_time = float('inf')
        self.trace = []

    def log_position(self, code):
        self.trace.append(str(code))


class PBSSimulator:
    def __init__(self, input_cars_data, target_output_sequence, area_code_file='附件3.xlsx'):
        self.cars = {d['id']: Car(d['id'], d['original_id'], d['H'], d['D']) for d in input_cars_data}
        self.total_cars = len(input_cars_data)
        self.target_order = target_output_sequence
        self.next_target_idx = 0

        # Lanes: list[0]=Pos1(出口), list[-1]=Pos10(入口)
        self.lanes = [[] for _ in range(6)]
        # Return Lane: list[0]=Pos1(入口), list[-1]=Pos10(出口)
        self.return_lane = []

        self.paint_queue = [d['id'] for d in input_cars_data]
        self.assembly_queue = []

        self.transporter_in = {'state': 'idle', 'finish_time': 0, 'car': None, 'dest': None}
        self.transporter_out = {'state': 'idle', 'finish_time': 0, 'car': None, 'dest': None}

        self.time = 0
        self.deadlock_flag = False
        self.return_usage_count = 0

        self.area_codes = self._load_area_codes(area_code_file)

    @staticmethod
    def _load_area_codes(filepath):
        """加载区域代码映射（静态方法修复）"""
        # 如果需要实际读取Excel，请在此处实现 pd.read_excel 逻辑
        return {}

    def get_area_code(self, car):
        """根据位置生成代码，增加防御性长度检查"""
        loc = car.current_location
        if not loc: return ""
        tag = loc[0]

        if tag == 'PAINT': return '1'
        if tag == 'TRANS_IN': return '2'
        if tag == 'TRANS_OUT': return '4'
        if tag == 'ASSEMBLY': return '3'

        if tag == 'LANE':
            # 确保有足够的元素
            if len(loc) >= 3:
                lane_idx = loc[1]  # 0-5
                pos = loc[2]  # 1-10
                return f"{lane_idx + 1}{pos}"
            return "ERR_LANE"

        if tag == 'RETURN':
            if len(loc) >= 2:
                pos = loc[1]
                return f"5{pos}"  # 假设返回道前缀5
            return "ERR_RET"

        return ""

    def run(self, max_time=20000):
        print(f"开始仿真... 目标车辆数: {self.total_cars}")
        while len(self.assembly_queue) < self.total_cars:
            if self.time > max_time:
                print("超时！")
                return False, 9999, 9999
            if self.deadlock_flag:
                print(f"死锁发生于 T={self.time}")
                return False, 9999, 9999

            self.step()

            for cid, car in self.cars.items():
                code = self.get_area_code(car)
                car.log_position(code)

            self.time += 1

        z3 = self.return_usage_count
        z4 = self.time
        print(f"仿真完成! Z3={z3}, Z4={z4}")
        return True, z3, z4

    def step(self):
        self._update_transporters()
        self._update_lane_creep()

        # Input Transporter Logic
        if self.transporter_in['state'] == 'idle':
            # 优先处理返回道 (Pos 10 是出口)
            if self.return_lane and self._is_at_pos(self.return_lane[-1], 'RETURN', 10):
                car_id = self.return_lane[-1]
                target_lane = self._decide_entry_lane(car_id)
                if target_lane != -1:
                    self._action_return_to_lane(target_lane)

            elif self.paint_queue:
                car_id = self.paint_queue[0]
                target_lane = self._decide_entry_lane(car_id)
                if target_lane != -1:
                    self._action_paint_to_lane(target_lane)

        # Output Transporter Logic
        if self.transporter_out['state'] == 'idle':
            priority_lane = self._get_fcfs_lane()
            if priority_lane is not None:
                lane_idx = priority_lane
                car_id = self.lanes[lane_idx][0]

                wanted_id = self.target_order[self.next_target_idx]
                if car_id == wanted_id:
                    self._action_lane_to_assembly(lane_idx)
                else:
                    if len(self.return_lane) < 10:
                        # 检查返回道入口 (Pos 1) 是否通畅
                        if not self.return_lane or not self._is_at_pos(self.return_lane[0], 'RETURN', 1):
                            self._action_lane_to_return(lane_idx)
                    else:
                        self.deadlock_flag = True

    def _decide_entry_lane(self, car_id):
        current_car = self.cars[car_id]
        my_rank = current_car.original_rank
        best_lane = -1
        min_gap = float('inf')
        empty_lanes = []

        for k in range(6):
            if len(self.lanes[k]) >= 10: continue

            # 检查入口拥堵
            if self.lanes[k]:
                tail_id = self.lanes[k][-1]
                if self._is_at_pos(tail_id, 'LANE', 10, lane_idx=k):
                    continue

            if not self.lanes[k]:
                empty_lanes.append(k)
                continue

            tail_id = self.lanes[k][-1]
            tail_car = self.cars[tail_id]
            tail_rank = tail_car.original_rank
            gap = my_rank - tail_rank

            if gap > 0 and gap < min_gap:
                min_gap = gap
                best_lane = k

        if best_lane != -1:
            return best_lane
        elif empty_lanes:
            return empty_lanes[0]
        else:
            # Fallback
            max_space = -1
            fallback_lane = -1
            for k in range(6):
                if len(self.lanes[k]) < 10:
                    if self.lanes[k] and self._is_at_pos(self.lanes[k][-1], 'LANE', 10, lane_idx=k):
                        continue
                    space = 10 - len(self.lanes[k])
                    if space > max_space:
                        max_space = space
                        fallback_lane = k
            return fallback_lane

    def _is_at_pos(self, car_id, loc_type, pos, lane_idx=None):
        """核心修复：增加长度检查，防止IndexError"""
        car = self.cars[car_id]
        if car.is_moving: return False

        curr = car.current_location
        if not curr or curr[0] != loc_type: return False

        if loc_type == 'RETURN':
            # 确保元组至少有2个元素 ('RETURN', pos)
            return len(curr) >= 2 and curr[1] == pos
        elif loc_type == 'LANE':
            # 确保元组至少有3个元素 ('LANE', k, pos)
            return len(curr) >= 3 and curr[1] == lane_idx and curr[2] == pos

        return False

    def _get_fcfs_lane(self):
        candidates = []
        for k in range(6):
            if self.lanes[k]:
                head_car = self.cars[self.lanes[k][0]]
                # 检查位置时增加长度检查的安全性
                if len(head_car.current_location) >= 3 and \
                        head_car.current_location[2] == 1 and \
                        not head_car.is_moving:
                    candidates.append((k, head_car.arrival_at_lane_head_time))

        if not candidates: return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _update_lane_creep(self):
        # 1. 进车道 (10 -> 1)
        for k in range(6):
            lane = self.lanes[k]
            for i, cid in enumerate(lane):
                car = self.cars[cid]
                # 必须确保当前位置是有效的LANE位置
                if len(car.current_location) < 3: continue

                curr_pos = car.current_location[2]

                if car.is_moving:
                    if self.time >= car.move_end_time:
                        car.is_moving = False
                        new_pos = curr_pos - 1
                        car.current_location = ('LANE', k, new_pos)
                        if new_pos == 1:
                            car.arrival_at_lane_head_time = self.time
                else:
                    if curr_pos > 1:
                        can_move = False
                        if i == 0:
                            can_move = True
                        else:
                            prev_car = self.cars[lane[i - 1]]
                            # 确保前车位置有效
                            if len(prev_car.current_location) >= 3 and \
                                    prev_car.current_location[2] < curr_pos - 1:
                                can_move = True
                        if can_move:
                            car.is_moving = True
                            car.move_end_time = self.time + TIME_PARAMS['move']

        # 2. 返回道 (1 -> 10)
        # return_lane[-1]是出口(Pos10), return_lane[0]是入口(Pos1)
        for i in range(len(self.return_lane) - 1, -1, -1):
            cid = self.return_lane[i]
            car = self.cars[cid]
            if len(car.current_location) < 2: continue

            curr_pos = car.current_location[1]

            if car.is_moving:
                if self.time >= car.move_end_time:
                    car.is_moving = False
                    new_pos = curr_pos + 1
                    car.current_location = ('RETURN', new_pos)
            else:
                if curr_pos < 10:
                    can_move = False
                    if i == len(self.return_lane) - 1:
                        can_move = True
                    else:
                        prev_car = self.cars[self.return_lane[i + 1]]
                        if len(prev_car.current_location) >= 2 and \
                                prev_car.current_location[1] > curr_pos + 1:
                            can_move = True
                    if can_move:
                        car.is_moving = True
                        car.move_end_time = self.time + TIME_PARAMS['move']

    def _update_transporters(self):
        if self.transporter_in['state'] == 'busy':
            if self.time >= self.transporter_in['finish_time']:
                cid = self.transporter_in['car']
                lane_k = self.transporter_in['dest']
                self.lanes[lane_k].append(cid)
                self.cars[cid].current_location = ('LANE', lane_k, 10)
                self.transporter_in['state'] = 'idle'
                self.transporter_in['car'] = None

        if self.transporter_out['state'] == 'busy':
            if self.time >= self.transporter_out['finish_time']:
                cid = self.transporter_out['car']
                dest_type = self.transporter_out['dest']
                if dest_type == 'ASSEMBLY':
                    self.assembly_queue.append(cid)
                    self.cars[cid].current_location = ('ASSEMBLY',)
                    self.next_target_idx += 1
                elif dest_type == 'RETURN':
                    self.return_lane.insert(0, cid)
                    self.cars[cid].current_location = ('RETURN', 1)
                    self.return_usage_count += 1
                self.transporter_out['state'] = 'idle'
                self.transporter_out['car'] = None

    def _action_paint_to_lane(self, k):
        cid = self.paint_queue.pop(0)
        self.transporter_in['state'] = 'busy'
        self.transporter_in['finish_time'] = self.time + TIME_PARAMS['t_in'][k]
        self.transporter_in['car'] = cid
        self.transporter_in['dest'] = k
        self.cars[cid].current_location = ('TRANS_IN',)

    def _action_return_to_lane(self, k):
        cid = self.return_lane.pop()
        self.transporter_in['state'] = 'busy'
        self.transporter_in['finish_time'] = self.time + TIME_PARAMS['t_from_return'][k]
        self.transporter_in['car'] = cid
        self.transporter_in['dest'] = k
        self.cars[cid].current_location = ('TRANS_IN',)

    def _action_lane_to_assembly(self, k):
        cid = self.lanes[k].pop(0)
        self.transporter_out['state'] = 'busy'
        self.transporter_out['finish_time'] = self.time + TIME_PARAMS['t_out'][k]
        self.transporter_out['car'] = cid
        self.transporter_out['dest'] = 'ASSEMBLY'
        self.cars[cid].current_location = ('TRANS_OUT',)

    def _action_lane_to_return(self, k):
        cid = self.lanes[k].pop(0)
        self.transporter_out['state'] = 'busy'
        self.transporter_out['finish_time'] = self.time + TIME_PARAMS['t_to_return'][k]
        self.transporter_out['car'] = cid
        self.transporter_out['dest'] = 'RETURN'
        self.cars[cid].current_location = ('TRANS_OUT',)

    def export_result(self, filename='result11.xlsx'):
        data = {}
        # 防止空trace报错
        if not self.target_order:
            print("无数据可导出")
            return

        sample_trace = self.cars[self.target_order[0]].trace
        max_len = len(sample_trace) if sample_trace else 0

        for cid, car in self.cars.items():
            current_len = len(car.trace)
            # 填充缺失部分
            padding = [''] * (max_len - current_len)
            row = car.trace + padding
            data[cid] = row

        df = pd.DataFrame.from_dict(data, orient='index')
        df.to_excel(filename)
        print(f"已导出: {filename}")