import torch
import numpy as np
from scipy.spatial.transform import Rotation

class ActionDiscretizer:
    """6D动作空间离散化工具"""
    def __init__(self):
        self.bin_ranges = {
            'x': (-0.2, 0.2),    # meters
            'y': (-0.2, 0.2),
            'z': (-0.1, 0.3),
            'roll': (-np.pi, np.pi),
            'pitch': (-np.pi/2, np.pi/2),
            'yaw': (-np.pi, np.pi)
        }
        self.bins = 256

    def _slerp(self, start, end, t):
        """球面线性插值离散化"""
        start_rot = Rotation.from_euler('xyz', start)
        end_rot = Rotation.from_euler('xyz', end)
        return (start_rot * Rotation.slerp(start_rot, end_rot, t)).as_euler('xyz')

    def discretize(self, pose):
        """将连续动作离散化为256 bins"""
        discretized = []
        
        # 位置离散化（均匀分布）
        for axis in ['x', 'y', 'z']:
            min_val, max_val = self.bin_ranges[axis]
            scaled = (pose[axis] - min_val) / (max_val - min_val)
            discretized.append(torch.bucketize(scaled, torch.linspace(0, 1, self.bins)))
        
        # 姿态离散化（Slerp）
        base_angles = np.array([0, 0, 0])
        target_angles = np.array(pose[3:])
        steps = np.linspace(0, 1, self.bins)
        sampled_angles = np.array([self._slerp(base_angles, target_angles, t) for t in steps])
        
        # 选择最近邻
        diffs = np.abs(sampled_angles - pose[3:])
        discretized.extend(torch.argmin(torch.tensor(diffs), dim=0))
        
        return torch.tensor(discretized, dtype=torch.long)
