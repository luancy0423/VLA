import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from .preprocess import Sim2RealAugmentor, ActionDiscretizer

class GraspDataset(Dataset):
    """多模态抓取数据集"""
    def __init__(self, data_root, stage='train'):
        self.data_root = Path(data_root)
        self.stage = stage
        self.augmentor = Sim2RealAugmentor()
        self.discretizer = ActionDiscretizer()
        
        # 加载元数据
        with open(self.data_root / 'metadata.json') as f:
            self.metadata = json.load(f)
        
        # 双语指令增强
        self.translator = Translator()  # 需实现回译工具

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # 图像处理
        img_path = self.data_root / item['image_path']
        image = Image.open(img_path).convert('RGB')
        image = self.augmentor(np.array(image))
        
        # 指令增强
        instruction = item['instruction']
        if self.stage == 'train' and random.random() < 0.5:
            instruction = self.translator.back_translate(instruction)
        
        # 动作离散化
        pose = torch.tensor([
            item['pose']['x'],
            item['pose']['y'],
            item['pose']['z'],
            item['pose']['roll'],
            item['pose']['pitch'],
            item['pose']['yaw'],
            item['gripper']
        ])
        discrete_pose = self.discretizer.discretize(pose)
        
        return {
            'image': image,
            'instruction': instruction,
            'cot_labels': item['cot_chain'],
            'action_labels': discrete_pose
        }
