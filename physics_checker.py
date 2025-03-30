import numpy as np

class PhysicsValidator:
    """物理合理性验证器"""
    
    def __init__(self, robot_params):
        self.max_velocity = robot_params['max_velocity']
        self.gripper_range = robot_params['gripper_range']
        
    def check_trajectory(self, positions):
        """轨迹连续性检查"""
        velocities = np.diff(positions, axis=0)
        if np.any(np.linalg.norm(velocities, axis=1) > self.max_velocity):
            return False
        return True
    
    def check_grasp_pose(self, obj_cloud, grasp_pose):
        """抓取位姿可行性检查"""
        # 转换到物体坐标系
        local_pts = obj_cloud - grasp_pose[:3]
        rotated_pts = np.dot(local_pts, grasp_pose[3:].T)
        
        # 检查夹持器闭合区域
        in_x = (rotated_pts[:,0] > -0.05) & (rotated_pts[:,0] < 0.05)
        in_y = (rotated_pts[:,1] > -0.02) & (rotated_pts[:,1] < 0.02)
        in_z = rotated_pts[:,2] < self.gripper_range
        
        return np.any(in_x & in_y & in_z)
    
    def check_force_balance(self, force_vector):
        """力平衡检查"""
        return np.linalg.norm(force_vector) < 5.0  # 假设最大5N
