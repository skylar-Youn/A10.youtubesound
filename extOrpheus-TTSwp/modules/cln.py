import torch, torch.nn as nn

class CLN(nn.Module):
    def __init__(self, hidden, emo_dim):
        super().__init__()
        self.gamma = nn.Linear(emo_dim, hidden)
        self.beta  = nn.Linear(emo_dim, hidden)

    def forward(self, h, e):
        # h: [B,T,H], e: [B,emo_dim]
        g = self.gamma(e).unsqueeze(1)
        b = self.beta(e).unsqueeze(1)
        return g * h + b
