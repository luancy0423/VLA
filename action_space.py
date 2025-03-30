import numpy as np
import torch

class ActionDiscretizer:
    """动作空间离散化策略"""
    def __init__(self):
        # 位移范围 (米)
        self.pos_bins = {
            'x': (-0.2, 0.2, 256),
            'y': (-0.2, 0.2, 256),
            'z': (-0.1, 0.3, 256)
        }
        
        # 欧拉角范围 (弧度)
        self.rot_bins = {
            'roll': (-np.pi, np.pi, 256),
            'pitch': (-np.pi/2, np.pi/2, 256),
            'yaw': (-np.pi, np.pi, 256)
        }
    
    def slerp(self, start, end, t):
        """球面线性插值"""
        start = start / torch.norm(start)
        end = end / torch.norm(end)
        dot = torch.dot(start, end)
        theta = torch.acos(dot) * t
        relative = end - dot * start
        return start * torch.cos(theta) + (relative / torch.norm(relative)) * torch.sin(theta)
    
    def discretize_rotation(self, euler_angles):
        """姿态离散化"""
        quats = []
        for angle in euler_angles:
            quat = torch.tensor(Rotation.from_euler('xyz', angle).as_quat())
            quats.append(quat)
        
        # 生成插值路径
        steps = torch.linspace(0, 1, 256)
        discretized = []
        for t in steps:
            q = self.slerp(quats[0], quats[-1], t)
            discretized.append(Rotation.from_quat(q).as_euler('xyz'))
        return torch.stack(discretized)
