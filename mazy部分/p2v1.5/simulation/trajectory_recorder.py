"""
轨迹记录工具模块 - 记录车辆在各车道的位置轨迹
"""
import numpy as np


class TrajectoryRecorder:
    """轨迹记录器：将车辆位置映射到轨迹矩阵"""
    
    @staticmethod
    def map_lane_vehicles(lane_vehicle_ids: np.ndarray, trajectory_matrix: np.ndarray, 
                          current_step: int) -> np.ndarray:
        """
        将车道上的车的位置记录到轨迹矩阵中
        
        Args:
            lane_vehicle_ids: 6条车道上各车位的车辆编号，-1表示空
            trajectory_matrix: 轨迹记录矩阵
            current_step: 当前时间步
            
        Returns:
            更新后的轨迹矩阵
        """
        # 中间车位的车道映射
        center_lanes = lane_vehicle_ids[:, 3:-1]  # 去除入口区的三个时间戳，出口区的一个
        occupied_indices = np.where(center_lanes != -1)

        # 映射公式逻辑
        current_vars = (occupied_indices[0].reshape(-1, 1) + 1) * 10 + 10 - (
                    occupied_indices[1].reshape(-1, 1) + 3) // 3
        current_vars = current_vars.reshape(-1, 1)

        # 车位信息记录到全局轨迹矩阵中
        vehicle_ids = center_lanes[occupied_indices[0], occupied_indices[1]].astype(int)
        trajectory_matrix[vehicle_ids, current_step * 3: current_step * 3 + 3] = current_vars

        # 入口区的车道编号映射公式
        entrance_indices = np.where(lane_vehicle_ids[:, :3] != -1)
        vehicle_ids_entrance = lane_vehicle_ids[entrance_indices[0], entrance_indices[1]].astype(int)
        trajectory_matrix[vehicle_ids_entrance, current_step * 3: current_step * 3 + 3] = (entrance_indices[0].reshape(
            -1, 1) + 1) * 100 + 10

        # 出口区的车道编号映射公式
        exit_indices = np.where(lane_vehicle_ids[:, -1] != -1)
        vehicle_ids_exit = lane_vehicle_ids[exit_indices[0], -1].astype(int)
        trajectory_matrix[vehicle_ids_exit, current_step * 3: current_step * 3 + 3] = (exit_indices[0].reshape(-1,
                                                                                                               1) + 1) * 10 + 1

        return trajectory_matrix

    @staticmethod
    def map_return_lane_vehicles(return_lane_vehicle_ids: np.ndarray, trajectory_matrix: np.ndarray,
                                  current_step: int) -> np.ndarray:
        """
        将返回道上的车的位置记录到轨迹矩阵中
        
        Args:
            return_lane_vehicle_ids: 返回道各车位的车辆编号，-1表示空
            trajectory_matrix: 轨迹记录矩阵
            current_step: 当前时间步
            
        Returns:
            更新后的轨迹矩阵
        """
        # 处理返回道第10车位
        vehicle_at_10 = return_lane_vehicle_ids[0]
        if vehicle_at_10 != -1:
            trajectory_matrix[int(vehicle_at_10), current_step * 3: current_step * 3 + 3] = 710

        # 处理返回道1-9车位
        vehicles_at_others = return_lane_vehicle_ids[1:]
        occupied_indices = np.where(vehicles_at_others != -1)[0]
        target_vehicles = vehicles_at_others[occupied_indices].astype(int)

        trajectory_matrix[target_vehicles, current_step * 3: current_step * 3 + 3] = 70 + 9 - occupied_indices.reshape(
            -1, 1) // 3
        return trajectory_matrix

    @staticmethod
    def trim_result(trajectory_matrix: np.ndarray) -> np.ndarray:
        """
        裁剪轨迹矩阵中多余的时间戳
        
        Args:
            trajectory_matrix: 原始轨迹矩阵
            
        Returns:
            裁剪后的轨迹矩阵
        """
        max_valid_idx = np.where(trajectory_matrix != 3)[1].max()
        return trajectory_matrix[:, :max_valid_idx + 3]
