from timm.models.vision_transformer import VisionTransformer

class EnhancedViT(VisionTransformer):
    """改进的ViT-H/14视觉编码器"""
    def __init__(self, img_size=448, ​**kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=14,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            ​**kwargs
        )
        
        # 适配CLIP预训练权重
        self.patch_embed.proj = nn.Conv2d(3, 1280, kernel_size=14, stride=14)
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x
