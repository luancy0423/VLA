import torch
from PIL import Image
from model.universal_grasp_vla import UniversalGraspingVLA
from data.preprocess.sim2real_aug import Sim2RealAugmentor

class GraspInferencer:
    """单次抓取推理示例"""
    def __init__(self, checkpoint_path):
        self.model = UniversalGraspingVLA()
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        
        self.preprocessor = Sim2RealAugmentor()
        self.tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-VL-34B")

    def predict(self, image_path, instruction):
        # 图像预处理
        raw_image = Image.open(image_path).convert("RGB")
        image = self.preprocessor(np.array(raw_image)).unsqueeze(0)
        
        # 指令编码
        text_input = self.tokenizer(
            instruction, 
            return_tensors="pt",
            padding=True
        )
        
        # 推理
        with torch.no_grad():
            action_logits = self.model(image, [instruction])
        
        # 解码动作
        action_bins = action_logits.argmax(dim=-1)[0]
        return self._decode_action(action_bins)

    def _decode_action(self, bins):
        """离散动作转连续值"""
        decode_rules = {
            'x': (-0.2, 0.2, 256),
            'y': (-0.2, 0.2, 256),
            'z': (-0.1, 0.3, 256)
        }
        action = {}
        for i, (axis, (min_v, max_v, _)) in enumerate(decode_rules.items()):
            action[axis] = min_v + (bins[i].item() / 255) * (max_v - min_v)
        
        # 姿态解码
        quat = Rotation.from_euler('xyz', [
            np.pi * (bins[3].item()/127.5 - 1),
            (np.pi/2) * (bins[4].item()/127.5 - 1),
            np.pi * (bins[5].item()/127.5 - 1)
        ]).as_quat()
        
        return {
            'position': [action['x'], action['y'], action['z']],
            'rotation': quat.tolist(),
            'gripper': bins[6].item() > 128
        }

# 使用示例
if __name__ == "__main__":
    inferencer = GraspInferencer("checkpoints/model_final.pth")
    result = inferencer.predict(
        image_path="examples/data/cup_on_table.jpg",
        instruction="抓取玻璃杯"
    )
    print("预测动作:", result)
