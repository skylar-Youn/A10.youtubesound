\
import torch, torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, ch, k=5, d=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=(k//2)*d, dilation=d),
            nn.ReLU(),
            nn.Conv1d(ch, ch, k, padding=(k//2)*d, dilation=d),
            nn.ReLU()
        )
    def forward(self, x):  # [B, C, T]
        return self.net(x)

class ProsodyNet(nn.Module):
    """
    Input: [mel_neu, f0_neu, energy_neu] -> predict [Δmel]
    Optional emotion_id embedding for multi-emotion training.
    """
    def __init__(self, n_mels=80, emo_classes=1, model_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        in_ch = n_mels + 2  # mel + f0 + energy
        self.emo_emb = nn.Embedding(emo_classes, 32)
        self.inp = nn.Conv1d(in_ch + 32, model_dim, 1)
        self.convs = nn.ModuleList([ConvBlock(model_dim) for _ in range(3)])
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, batch_first=True)
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out = nn.Conv1d(model_dim, n_mels, 1)

    def forward(self, mel_neu, f0_neu, ene_neu, emo_id=0):
        """
        mel_neu: [B, T, n_mels]
        f0_neu:  [B, T]
        ene_neu: [B, T]
        """
        B, T, M = mel_neu.shape
        emo = self.emo_emb(torch.full((B,), emo_id, dtype=torch.long, device=mel_neu.device))  # [B, 32]
        emo = emo.unsqueeze(-1).expand(B, 32, T)  # [B, 32, T]

        x_mel = mel_neu.transpose(1,2)           # [B, M, T]
        x_f0  = f0_neu.unsqueeze(1)              # [B, 1, T]
        x_en  = ene_neu.unsqueeze(1)             # [B, 1, T]
        x = torch.cat([x_mel, x_f0, x_en, emo], dim=1)  # [B, M+2+32, T]
        x = self.inp(x)                          # [B, D, T]
        for c in self.convs: x = x + c(x)
        x_tr = x.transpose(1,2)                  # [B, T, D]
        x_tr = self.tr(x_tr)                     # [B, T, D]
        x = x_tr.transpose(1,2)                  # [B, D, T]
        dmel = self.out(x).transpose(1,2)        # [B, T, M]
        return dmel
