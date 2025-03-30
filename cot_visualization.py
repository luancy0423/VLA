import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
from model.core.semantic_decomp import SemanticDecomposition

class CoTVisualizer:
    def __init__(self, model):
        self.model = model
        self.attention_maps = None
        
    def _hook_attention(self, module, input, output):
        # 注册注意力钩子
        self.attention_maps = output[1].detach().cpu().numpy()

    def visualize_pipeline(self, image_path, instruction):
        # 前向传播获取中间结果
        img = Image.open(image_path)
        with torch.no_grad():
            handles = self._register_hooks()
            outputs = self.model([img], [instruction])
            self._remove_hooks(handles)
        
        # 创建可视化面板
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4)
        
        # 绘制原始图像
        ax_img = fig.add_subplot(gs[:, 0])
        ax_img.imshow(img)
        ax_img.set_title("Input Image")
        
        # 显示属性解析
        self._draw_attribute_heatmap(fig, gs[0, 1])
        
        # 显示关系推理图
        self._draw_relation_graph(fig, gs[1:3, 1])
        
        # 显示CoT推理链
        self._draw_cot_text(fig, gs[:, 2:], outputs['cot_text'])
        
        plt.tight_layout()
        plt.show()

    def _draw_attribute_heatmap(self, fig, pos):
        ax = fig.add_subplot(pos)
        attn = self.attention_maps[0].mean(axis=0)
        ax.matshow(attn, cmap='viridis')
        ax.set_title("Attribute Attention Heatmap")
        ax.axis('off')

    def _draw_relation_graph(self, fig, pos):
        ax = fig.add_subplot(pos)
        G = nx.DiGraph()
        
        # 从关系分支获取节点和边
        nodes = self.model.sem_decomp.relation_net.nodes
        edges = self.model.sem_decomp.relation_net.adj_matrix
        
        # 添加节点和边
        G.add_nodes_from(nodes)
        for i, j in zip(*np.where(edges > 0.5)):
            G.add_edge(i, j, weight=edges[i,j])
            
        # 绘制网络图
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500)
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.0)
        nx.draw_networkx_labels(G, pos, ax=ax)
        ax.set_title("Spatial Relation Graph")

    def _draw_cot_text(self, fig, pos, cot_text):
        ax = fig.add_subplot(pos)
        ax.axis('off')
        
        # 解析CoT文本
        steps = cot_text.split('\n')
        text_box = "\n".join([
            f"Step {i+1}: {s}" 
            for i, s in enumerate(steps[:4])
        ])
        
        ax.text(0.1, 0.5, text_box, 
               ha='left', va='center', 
               fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        ax.set_title("Chain-of-Thought Reasoning")

if __name__ == "__main__":
    model = UniversalGraspingVLA.load_from_checkpoint("checkpoints/stage2.ckpt")
    visualizer = CoTVisualizer(model)
    
    visualizer.visualize_pipeline(
        image_path="examples/scene1.jpg",
        instruction="请抓取桌子右侧的马克杯"
    )
