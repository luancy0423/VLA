import torch.nn as nn

class ActionDiscriminator(nn.Module):
    """动作序列判别器"""
    def __init__(self, input_dim=7, hidden_dim=128):
        super().__init__()
        self.temporal_net = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, bidirectional=True),
            nn.Linear(2*hidden_dim, 64),
            nn.ReLU()
        )
        
        self.disc_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, actions):
        # actions: [B, T, 7]
        temporal_feat, _ = self.temporal_net(actions.permute(1,0,2))
        return self.disc_head(temporal_feat[-1])

class AdversarialTrainer:
    """对抗训练协调器"""
    def __init__(self, generator, discriminator, config):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optim = AdamW(generator.parameters(), lr=config['gen_lr'])
        self.dis_optim = AdamW(discriminator.parameters(), lr=config['dis_lr'])
        self.scaler = GradScaler()

    def train_step(self, real_actions, fake_actions):
        # 训练判别器
        self.dis_optim.zero_grad()
        with torch.cuda.amp.autocast():
            real_pred = self.discriminator(real_actions)
            fake_pred = self.discriminator(fake_actions.detach())
            dis_loss = F.binary_cross_entropy(
                torch.cat([real_pred, fake_pred]),
                torch.cat([torch.ones_like(real_pred), 
                          torch.zeros_like(fake_pred)])
            )
        
        self.scaler.scale(dis_loss).backward()
        self.scaler.step(self.dis_optim)
        
        # 训练生成器
        self.gen_optim.zero_grad()
        with torch.cuda.amp.autocast():
            fake_pred = self.discriminator(fake_actions)
            gen_loss = F.binary_cross_entropy(
                fake_pred, 
                torch.ones_like(fake_pred)
            )
        
        self.scaler.scale(gen_loss).backward()
        self.scaler.step(self.gen_optim)
        self.scaler.update()
        
        return {'dis_loss': dis_loss.item(), 'gen_loss': gen_loss.item()}
