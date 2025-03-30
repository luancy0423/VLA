import torch
import torch.nn as nn
from .visual_encoder import EnhancedViT
from .text_encoder import YiVLTextEncoder
from .core.semantic_decomp import SemanticDecomposition
from .modules.dynamic_vocab import DynamicVocabulary

class UniversalGraspingVLA(nn.Module):
    """机器人通用抓取主模型"""
    def __init__(self):
        super().__init__()
        # 视觉编码
        self.visual_encoder = EnhancedViT()
        
        # 文本编码
        self.text_encoder = YiVLTextEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-VL-34B")
        
        # 语义分解
        self.sem_decomp = SemanticDecomposition()
        
        # 动态投影
        self.fusion_proj = nn.Linear(1280 + 4096, 2048)
        
        # 动作生成
        self.action_decoder = nn.Sequential(
            nn.LSTM(2048, 1024, bidirectional=True),
            nn.Linear(2048, 7 * 256)  # 7DOF × 256bins
        )
        self.dynamic_vocab = DynamicVocabulary()
    
    def forward(self, images, instructions):
        # 视觉特征
        vis_feat = self.visual_encoder(images)  # [B, 197, 1280]
        cls_vis = vis_feat[:, 0]  # [B, 1280]
        
        # 文本特征
        text_input = self.tokenizer(instructions, return_tensors='pt', padding=True)
        text_feat = self.text_encoder(**text_input).last_hidden_state  # [B, Seq, 4096]
        cls_text = text_feat[:, 0]  # [B, 4096]
        
        # 语义分解
        aligned_feat, scene_graph = self.sem_decomp(
            vis_feat[:, 1:].permute(0, 2, 1).view(-1, 1280, 16, 16),
            text_feat
        )
        
        # 多模态融合
        fused_feat = self.fusion_proj(torch.cat([cls_vis, cls_text], dim=1))
        
        # 动作生成
        action_logits = self.action_decoder(fused_feat.unsqueeze(1))
        action_logits = action_logits.view(-1, 7, 256)
        
        # 应用动态词汇表
        return self.dynamic_vocab.apply_mask(action_logits)
