class GraphAttentionNetwork(nn.Module):
    """多跳图注意力网络"""
    def __init__(self, node_dim=256, edge_dim=64, num_heads=4, hops=3):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphAttentionLayer(node_dim, edge_dim, num_heads)
            for _ in range(hops)
        ])
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, edge_dim),  # 假设边特征为3D坐标差
            nn.ReLU()
        )
    
    def build_edges(self, nodes):
        """构建边特征"""
        # nodes: [B, N, D]
        B, N, D = nodes.shape
        coord_diff = nodes.unsqueeze(2) - nodes.unsqueeze(1)  # [B, N, N, D]
        return self.edge_encoder(coord_diff[..., :3])  # 取前3维作为空间差
    
    def forward(self, nodes):
        edges = self.build_edges(nodes)
        for layer in self.layers:
            nodes = layer(nodes, edges)
        return nodes
