import pytest
import torch
from model import UniversalGraspingVLA

@pytest.fixture
def vla_model():
    return UniversalGraspingVLA()

@pytest.fixture
def visual_input():
    return torch.randn(1, 3, 448, 448)

@pytest.fixture
def text_input():
    return ["抓取右侧的透明物体"]

def test_model_initialization(vla_model):
    """验证模型组件初始化正确性"""
    assert isinstance(vla_model.visual_encoder, VisionTransformer)
    assert vla_model.text_encoder.config.model_type == "yi-vl-34b"
    assert hasattr(vla_model, 'sem_decomp')
    assert isinstance(vla_model.action_decoder, ActionDecoder)

def test_forward_pass_shapes(vla_model, visual_input, text_input):
    """验证前向传播输出形状"""
    cot_logits, action_logits = vla_model(visual_input, text_input)
    assert cot_logits.shape == (1, 1024, 32000)
    assert action_logits.shape == (1, 7, 256)

def test_multi_batch_processing(vla_model, text_input):
    """验证多批次输入处理能力"""
    batch_images = torch.randn(3, 3, 448, 448)
    cot_logits, action_logits = vla_model(batch_images, text_input*3)
    assert cot_logits.shape[0] == 3
    assert action_logits.shape[0] == 3

def test_multi_modal_inputs(vla_model):
    """验证多模态输入异常处理"""
    with pytest.raises(RuntimeError):
        vla_model(None, ["test instruction"])
    with pytest.raises(TypeError):
        vla_model(torch.randn(1,3,448,448), None)

def test_inference_speed(vla_model, visual_input, text_input):
    """验证单帧推理时延"""
    import time
    start = time.time()
    _ = vla_model(visual_input, text_input)
    assert time.time() - start < 1.0
