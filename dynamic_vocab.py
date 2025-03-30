class DynamicVocabulary:
    """运行时动作词汇表约束"""
    def __init__(self, num_dof=7, bins=256, keep_ratio=0.8):
        self.keep_tokens = int(bins * keep_ratio)
        self.mask_value = -1e9
        
        # 初始化保留token索引
        self.register_buffer('vocab_mask', 
            torch.zeros(num_dof, bins))
        for i in range(num_dof):
            self.vocab_mask[i, :self.keep_tokens] = 1
    
    def apply_mask(self, logits):
        """应用词汇表约束"""
        # logits: [B, num_dof, bins]
        masked_logits = logits * self.vocab_mask.unsqueeze(0)
        masked_logits += (1 - self.vocab_mask.unsqueeze(0)) * self.mask_value
        return masked_logits
