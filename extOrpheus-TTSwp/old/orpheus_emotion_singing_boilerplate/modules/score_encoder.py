import torch, torch.nn as nn

class ScoreEncoder(nn.Module):
    def __init__(self, pitch_embed=256, dur_embed=128, beat_pos_embed=64):
        super().__init__()
        self.pitch_emb = nn.Embedding(128, pitch_embed)     # MIDI 0..127
        self.dur_mlp   = nn.Sequential(nn.Linear(1, dur_embed), nn.ReLU(), nn.Linear(dur_embed, dur_embed))
        self.beat_emb  = nn.Embedding(16, beat_pos_embed)   # 16 sub-beats per bar (example)
        self.proj = nn.Linear(pitch_embed + dur_embed + beat_pos_embed, 256)

    def forward(self, midi_pitch, dur, beat_pos):
        # midi_pitch: [B,T], dur: [B,T,1] (seconds or beats), beat_pos: [B,T] (0..15)
        p = self.pitch_emb(midi_pitch)
        d = self.dur_mlp(dur)
        b = self.beat_emb(beat_pos)
        x = torch.cat([p, d, b], dim=-1)
        return self.proj(x)
