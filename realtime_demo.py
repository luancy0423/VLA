import cv2
import torch
import numpy as np
from model.universal_grasp_vla import UniversalGraspingVLA
from utils.deploy_utils import TensorRTWrapper

class RealtimeGrasping:
    def __init__(self, config):
        # 初始化模型
        self.device = torch.device(config['device'])
        self.model = self._load_model(config['model_path'])
        
        # 硬件加速设置
        self.trt_engine = TensorRTWrapper(
            model=self.model,
            input_shapes={'images': (1,3,448,448), 'text': (1,128)},
            precision='fp16'
        )
        
        # 机器人接口
        self.robot_arm = RobotController(config['robot_ip'])
        self.cap = cv2.VideoCapture(config['camera_id'])
        
        # 状态缓存
        self.history_actions = []
        self.current_pose = None

    def _load_model(self, ckpt_path):
        model = UniversalGraspingVLA()
        model.load_state_dict(torch.load(ckpt_path))
        return model.eval().to(self.device)

    def _preprocess_frame(self, frame):
        # Sim2Real域适应处理
        frame = cv2.resize(frame, (448,448))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = (frame / 127.5) - 1.0  # 归一化到[-1,1]
        return np.transpose(frame, (2,0,1))[np.newaxis,...]

    def _execute_action(self, action):
        # 动作后处理（包含物理验证）
        if self._check_collision(action):
            print("碰撞风险! 调整路径...")
            action = self._adjust_trajectory(action)
        
        # 转换为机器人坐标系
        target_pose = self._transform_to_robot_frame(action)
        
        # 发送指令
        self.robot_arm.move_to(
            position=target_pose[:3],
            rotation=target_pose[3:],
            speed=0.5
        )
        self.current_pose = target_pose

    def _generate_instruction(self):
        # 基于场景自动生成指令
        return "安全抓取距离最近的稳固物体"

    def run_loop(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 预处理与推理
            processed = self._preprocess_frame(frame)
            instruction = self._generate_instruction()
            
            # TensorRT加速推理
            cot, actions = self.trt_engine.infer({
                'images': processed,
                'instructions': [instruction]
            })
            
            # 选择最优动作
            best_action = actions[0].argmax(-1)
            self._execute_action(best_action)
            
            # 可视化反馈
            self._display_overlay(frame, cot, best_action)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.robot_arm.disconnect()

if __name__ == "__main__":
    config = {
        'model_path': 'checkpoints/stage3_final.pth',
        'camera_id': 0,
        'robot_ip': '192.168.1.10',
        'device': 'cuda'
    }
    demo = RealtimeGrasping(config)
    demo.run_loop()
