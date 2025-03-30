import os
import torch
import tensorrt as trt
from model.universal_grasp_vla import UniversalGraspingVLA

class TRTConverter:
    def __init__(self, config):
        self.config = config
        self.logger = trt.Logger(trt.Logger.INFO)
        self.model = self._load_pytorch_model()
        
    def _load_pytorch_model(self):
        model = UniversalGraspingVLA()
        ckpt = torch.load(self.config['ckpt_path'])
        model.load_state_dict(ckpt['state_dict'])
        return model.eval().half()
    
    def _export_onnx(self):
        dummy_image = torch.randn(
            1, 3, 448, 448, 
            device='cuda', 
            dtype=torch.float16
        )
        dummy_text = torch.randint(
            0, 32000, 
            (1, 128), 
            device='cuda'
        )
        
        torch.onnx.export(
            self.model,
            (dummy_image, dummy_text),
            self.config['onnx_path'],
            input_names=['images', 'instructions'],
            output_names=['cot_logits', 'action_logits'],
            dynamic_axes={
                'images': {0: 'batch'},
                'instructions': {0: 'batch'},
                'cot_logits': {0: 'batch'},
                'action_logits': {0: 'batch'}
            },
            opset_version=17
        )
        print(f"ONNX model saved to {self.config['onnx_path']}")
    
    def _build_engine(self):
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        # 配置优化参数
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.max_workspace_size = 4 << 30
        
        # 解析ONNX
        with open(self.config['onnx_path'], 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise ValueError("ONNX parsing failed")
        
        # 设置动态形状
        profile = builder.create_optimization_profile()
        for input_name in ['images', 'instructions']:
            input = network.get_input(0)
            profile.set_shape(
                input.name, 
                (1, *input.shape[1:]),   # min
                (4, *input.shape[1:]),    # opt
                (8, *input.shape[1:])     # max
            )
        config.add_optimization_profile(profile)
        
        # 构建引擎
        engine = builder.build_serialized_network(network, config)
        with open(self.config['trt_path'], 'wb') as f:
            f.write(engine)
        print(f"TensorRT engine saved to {self.config['trt_path']}")

    def convert(self):
        self._export_onnx()
        self._build_engine()

if __name__ == "__main__":
    config = {
        'ckpt_path': 'checkpoints/stage3_final.pth',
        'onnx_path': 'deploy/model.onnx',
        'trt_path': 'deploy/model_fp16.engine'
    }
    converter = TRTConverter(config)
    converter.convert()
