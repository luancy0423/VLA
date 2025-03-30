from transformers import AutoModel, AutoConfig

class YiVLTextEncoder(nn.Module):
    """Yi-VL-34B文本编码适配层"""
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained("01-ai/Yi-VL-34B")
        self.model = AutoModel.from_config(config)
        
        # 固定底层参数
        for param in self.model.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state
