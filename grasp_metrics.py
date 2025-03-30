import numpy as np

class GraspMetrics:
    """抓取性能评估指标"""
    def __init__(self, num_classes=10):
        self.confusion_matrix = np.zeros((num_classes, 2, 2))  # [class, pred, true]
    
    def update(self, pred_actions, true_actions, obj_classes):
        # 离散动作匹配度
        pos_errors = np.abs(pred_actions[:, :3] - true_actions[:, :3])
        rot_errors = np.abs(pred_actions[:, 3:6] - true_actions[:, 3:6])
        
        # 抓取成功判断
        pos_success = (pos_errors < 0.01).all(axis=1)  # 1cm误差
        rot_success = (rot_errors < np.deg2rad(5)).all(axis=1)
        successes = pos_success & rot_success
        
        # 更新混淆矩阵
        for cls, succ in zip(obj_classes, successes):
            self.confusion_matrix[cls, int(succ), int(succ)] += 1
    
    def compute(self):
        precision = self.confusion_matrix[:, 1, 1] / (self.confusion_matrix[:, 1].sum(1) + 1e-9)
        recall = self.confusion_matrix[:, 1, 1] / (self.confusion_matrix[:, :, 1].sum(1) + 1e-9)
        return {
            'class_avg_accuracy': np.diag(self.confusion_matrix.sum(2)) / self.confusion_matrix.sum(),
            'overall_success': self.confusion_matrix[:, 1, 1].sum() / self.confusion_matrix.sum()
        }
