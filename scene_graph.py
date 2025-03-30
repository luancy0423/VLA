import networkx as nx
import matplotlib.pyplot as plt

def visualize_scene_graph(nodes, edges):
    """
    可视化场景关系图
    参数:
        nodes: 字典 {node_id: {'label': str, 'color': str}} 
        edges: 列表 [(src, dst, {'label': str, 'weight': float})]
    """
    G = nx.DiGraph()
    
    # 添加节点
    for n_id, attrs in nodes.items():
        G.add_node(n_id, ​**attrs)
    
    # 添加边
    for src, dst, attrs in edges:
        G.add_edge(src, dst, ​**attrs)
    
    # 可视化布局
    pos = nx.spring_layout(G, k=0.5)
    
    # 绘制节点
    node_colors = [attrs['color'] for _, attrs in nodes.items()]
    labels = {n: attrs['label'] for n, attrs in nodes.items()}
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
    # 绘制边
    edge_labels = {(u,v): d['label'] for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.axis('off')
    return plt
