import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet18Block(nn.Module):
    """轻量化ResNet模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)

class MultiHeadAttention(nn.Module):
    """属性解析分支的自注意力机制"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        orig_shape = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        attn_out, _ = self.attn(x, x, x)
        return self.layer_norm(attn_out.permute(1, 2, 0).view(orig_shape))

class GraphAttentionLayer(nn.Module):
    """图注意力网络层"""
    def __init__(self, node_dim, edge_dim, num_heads):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, node_dim*2)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, node_dim),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(node_dim, num_heads)
        
    def forward(self, nodes, edges):
        # nodes: [N, D], edges: [N, N, E]
        edge_feats = self.edge_encoder(edges.mean(2))  # [N, N, D]
        q = self.node_proj(nodes).unsqueeze(0)  # [1, N, 2D]
        attn_out, _ = self.attn(q, nodes.unsqueeze(0), edge_feats)
        return attn_out.squeeze(0)

class SemanticDecomposition(nn.Module):
    """语义信息分解主模块"""
    def __init__(self, visual_dim=1280, text_dim=4096):
        super().__init__()
        # 属性解析分支
        self.attr_branch = nn.Sequential(
            nn.Conv2d(visual_dim, 256, 3, padding=1),
            ResNet18Block(256),
            MultiHeadAttention(256, 4)
        )
        
        # 关系推理分支
        self.relation_net = nn.ModuleList([
            GraphAttentionLayer(256, 64, 4) for _ in range(3)
        ])
        
        # 跨模态对齐
        self.align_proj = nn.Linear(text_dim, 256)
        self.cross_attn = nn.MultiheadAttention(256, 8)
    
    def build_scene_graph(self, visual_feat):
        """构建场景图节点特征"""
        # visual_feat: [B, C, H, W]
        B, C, H, W = visual_feat.shape
        nodes = visual_feat.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        return nodes
    
    def forward(self, visual_feat, text_feat):
        # 属性特征提取
        attr_feat = self.attr_branch(visual_feat)  # [B,256,16,16]
        
        # 场景图推理
        nodes = self.build_scene_graph(attr_feat)
        for layer in self.relation_net:
            nodes = layer(nodes, edges=None)  # 简化边特征
        
        # 文本特征对齐
        text_proj = self.align_proj(text_feat).permute(1, 0, 2)  # [Seq, B, 256]
        attn_out, _ = self.cross_attn(
            query=nodes.permute(1, 0, 2),
            key=text_proj,
            value=text_proj
        )
        return attn_out.permute(1, 0, 2), nodes
