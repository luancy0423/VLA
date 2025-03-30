import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from collections import OrderedDict

class ProgressiveTrainer:
    """三阶段渐进式训练器"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.scaler = GradScaler()
        self.optimizer = self._build_optimizer()
        self.current_stage = 1
        
        # 参数分组
        self.param_groups = OrderedDict([
            ('visual', model.visual_encoder.parameters()),
            ('text', model.text_encoder.parameters()),
            ('semantic', model.sem_decomp.parameters()),
            ('action', model.action_decoder.parameters())
        ])

    def _build_optimizer(self):
        return AdamW(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['lr'],
            weight_decay=1e-5
        )

    def _freeze_except(self, module_names):
        """冻结除指定模块外的所有参数"""
        for name, params in self.param_groups.items():
            for p in params:
                p.requires_grad_(name in module_names)

    def stage_transition(self, new_stage):
        """阶段过渡参数解冻策略"""
        if new_stage == 1:
            self._freeze_except(['visual'])
        elif new_stage == 2:
            self._freeze_except(['visual', 'semantic'])
        elif new_stage == 3:
            self._freeze_except(['visual', 'semantic', 'action'])
        
        self.current_stage = new_stage
        self.optimizer = self._build_optimizer()

    def train_step(self, batch):
        """混合精度训练步骤"""
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            # 前向传播
            action_logits = self.model(batch['images'], batch['instructions'])
            
            # 损失计算
            action_loss = F.cross_entropy(
                action_logits.view(-1, 256),
                batch['action_labels'].view(-1)
            )
            
            # 阶段相关损失
            if self.current_stage >= 2:
                sem_loss = self._semantic_loss(batch)
                action_loss += 0.3 * sem_loss
            
            if self.current_stage == 3:
                adv_loss = self._adversarial_loss(action_logits)
                action_loss += 0.2 * adv_loss
        
        # 梯度回传
        self.scaler.scale(action_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {'loss': action_loss.item()}

    def _semantic_loss(self, batch):
        """语义分解辅助损失"""
        attr_pred = self.model.sem_decomp.attr_branch(batch['images'])
        return F.mse_loss(attr_pred, batch['attr_labels'])

    def _adversarial_loss(self, action_logits):
        """对抗训练损失"""
        real_actions = batch['action_labels']
        fake_actions = action_logits.argmax(-1)
        return F.mse_loss(
            self.discriminator(real_actions),
            self.discriminator(fake_actions.detach())
        )
