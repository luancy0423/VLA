import torch

class RelationAccuracy:
    """关系推理评估器"""
    def __init__(self):
        self.correct = 0
        self.total = 0
    
    def update(self, pred_adj, true_adj):
        """比较预测与真实的邻接矩阵"""
        # 二值化预测
        pred_binary = (torch.sigmoid(pred_adj) > 0.5).int()
        self.correct += (pred_binary == true_adj).sum().item()
        self.total += true_adj.numel()
    
    def compute(self):
        return {'relation_acc': self.correct / self.total}
