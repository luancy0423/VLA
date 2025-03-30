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

def test_semantic_decomposition(vla_model, visual_input, text_input):
    """验证语义分解模块特征形状"""
    visual_feat = vla_model.visual_encoder(visual_input)
    text_feat = vla_model.text_encoder(
        vla_model.tokenizer(text_input, return_tensors='pt').input_ids
    ).last_hidden_state
    
    aligned_feat, adj_matrix = vla_model.sem_decomp(
        visual_feat[:,1:].permute(0,2,1).view(1,1280,16,16),
        text_feat
    )
    assert aligned_feat.shape[1:] == (256, 16, 16)
    assert adj_matrix.shape == (16, 16)

def test_training_mode(vla_model, visual_input, text_input):
    """验证训练模式梯度计算"""
    vla_model.train()
    cot_logits, action_logits = vla_model(visual_input, text_input)
    loss = cot_logits.mean() + action_logits.mean()
    loss.backward()
    assert vla_model.sem_decomp.attr_net[0].weight.grad is not None
