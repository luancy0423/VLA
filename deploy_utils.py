import torch
import tensorrt as trt

class ModelConverter:
    """模型部署优化工具类"""
    
    @staticmethod
    def quantize_model(model, example_input):
        """
        动态量化模型
        参数:
            model: 待量化模型
            example_input: 示例输入张量
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    
    @staticmethod
    def export_onnx(model, input_shape, save_path):
        """
        导出ONNX模型
        参数:
            model: 要导出的PyTorch模型
            input_shape: 输入张量形状 (C, H, W)
            save_path: 保存路径
        """
        dummy_input = torch.randn(1, *input_shape)
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    @staticmethod
    def build_tensorrt_engine(onnx_path, precision='fp16'):
        """
        构建TensorRT引擎
        参数:
            onnx_path: 输入ONNX路径
            precision: 精度模式 (fp32/fp16/int8)
        """
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # 解析ONNX
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())
        
        # 配置构建参数
        config = builder.create_builder_config()
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
        
        # 构建引擎
        engine = builder.build_engine(network, config)
        return engine
