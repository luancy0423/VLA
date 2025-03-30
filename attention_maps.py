import matplotlib.pyplot as plt
import numpy as np

def plot_cross_attention(attn_weights, img, texts):
    """
    可视化跨模态注意力机制
    参数:
        attn_weights: (H, W, L) 注意力权重矩阵
        img: 原始输入图像 (H, W, 3)
        texts: 对应的文本token列表
    """
    plt.figure(figsize=(20, 10))
    
    # 显示原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    
    # 文本注意力聚合
    text_agg = attn_weights.mean(axis=(0,1))
    plt.subplot(1, 3, 2)
    plt.barh(range(len(texts)), text_agg)
    plt.yticks(range(len(texts)), texts)
    plt.title('Text-side Attention')
    
    # 视觉注意力热力图
    plt.subplot(1, 3, 3)
    vis_agg = attn_weights.mean(axis=2)
    plt.imshow(vis_agg, cmap='viridis')
    plt.colorbar()
    plt.title('Visual Attention Heatmap')
    
    plt.tight_layout()
    return plt
