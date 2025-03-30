import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import cv2
from utils.deploy_utils import preprocess_image

class JetsonInference:
    def __init__(self, engine_path):
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 分配内存
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()
        
    def _load_engine(self, engine_path):
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            return runtime.deserialize_cuda_engine(f.read())
    
    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) 
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配显存
            mem = cuda.mem_alloc(size * dtype.itemsize)
            bindings.append(int(mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'mem': mem, 'shape': self.engine.get_binding_shape(binding)})
            else:
                outputs.append({'mem': mem, 'shape': self.engine.get_binding_shape(binding)})
        
        return inputs, outputs, bindings
    
    def _infer(self, image, instruction):
        # 预处理输入
        np_image = preprocess_image(image).astype(np.float16)
        np_text = self._process_instruction(instruction)
        
        # 异步传输数据
        cuda.memcpy_htod_async(self.inputs[0]['mem'], np_image, self.stream)
        cuda.memcpy_htod_async(self.inputs[1]['mem'], np_text, self.stream)
        
        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # 取回输出
        cot_output = np.empty(self.outputs[0]['shape'], dtype=np.float32)
        action_output = np.empty(self.outputs[1]['shape'], dtype=np.float32)
        
        cuda.memcpy_dtoh_async(cot_output, self.outputs[0]['mem'], self.stream)
        cuda.memcpy_dtoh_async(action_output, self.outputs[1]['mem'], self.stream)
        self.stream.synchronize()
        
        return cot_output, action_output
    
    def _process_instruction(self, text):
        # 简化的文本处理（实际应使用与训练一致的tokenizer）
        return np.array([[ord(c) for c in text.ljust(128)]], dtype=np.int32)
    
    def process_frame(self, frame, instruction):
        # 执行推理
        cot, actions = self._infer(frame, instruction)
        
        # 后处理
        best_action = self._postprocess_actions(actions)
        return best_action
    
    def _postprocess_actions(self, actions):
        # 转换离散token到连续动作
        action_bins = actions.argmax(axis=-1)
        return self._bins_to_continuous(action_bins)
    
    def _bins_to_continuous(self, bins):
        # 与训练时一致的离散化参数
        x_range = (-0.2, 0.2)  # 单位：米
        angle_range = (-np.pi, np.pi)
        
        continuous = np.zeros(7)
        for i in range(6):
            if i < 3:  # 位移
                continuous[i] = np.interp(
                    bins[i], 
                    [0, 255], 
                    x_range
                )
            else:       # 角度
                continuous[i] = np.interp(
                    bins[i],
                    [0, 255],
                    angle_range
                )
        continuous[6] = 1 if bins[6] > 128 else 0  # 夹爪
        return continuous

if __name__ == "__main__":
    # 示例用法
    inferencer = JetsonInference("deploy/model_fp16.engine")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        action = inferencer.process_frame(frame, "抓取最近的物体")
        print(f"执行动作：{action}")
