import pytest
import torch
from model import UniversalGraspingVLA

@pytest.fixture
def vla_model():
    return UniversalGraspingVLA()

def test_dynamic_vocab_constraint(vla_model):
    """验证动态词汇表约束"""
    _, action_logits = vla_model(torch.randn(1,3,448,448), ["test"])
    assert torch.all(action_logits[:, :, 255] == -1e9)

def test_action_space_config(vla_model):
    """验证动作空间参数"""
    assert vla_model.action_decoder.action_head[-1].out_features == 1792
    assert vla_model.action_vocab.shape == (7, 256)
