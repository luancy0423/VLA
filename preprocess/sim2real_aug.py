import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random

class Sim2RealAugmentor:
    """Sim2Real域随机化增强器"""
    def __init__(self, apply_prob=0.8):
        self.base_transform = A.Compose([
            A.RandomResizedCrop(448, 448, scale=(0.8, 1.0)),
            A.ColorJitter(hue=0.1, saturation=0.3, brightness=0.3, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.CoarseDropout(
                max_holes=8, 
                max_height=32, 
                max_width=32,
                fill_value=random.choice([0, 127, 255]),
                p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.texture_transform = A.RandomGamma(
            gamma_limit=(80, 120), 
            eps=None, 
            p=0.4
        )

    def __call__(self, image, force_apply=False):
        # 纹理随机化
        if random.random() < 0.7:
            image = self.texture_transform(image=image)['image']
        
        # 基础增强
        return self.base_transform(image=image)['image']
