\
import os, glob, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from prosodynet import ProsodyNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_MELS = 80
BATCH = 8
EPOCHS = 50
LR = 2e-4
FEAT_DIR = "feats"
CKPT = "ckpt/prosodynet.pt"

os.makedirs("ckpt", exist_ok=True)

class ProsodyDataset(Dataset):
    def __init__(self, feat_dir):
        self.ids = sorted(set([os.path.basename(p).split("_")[0] for p in glob.glob(f"{feat_dir}/*_nmel.npy")]))
        self.fd = feat_dir
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        k = self.ids[i]
        nmel = np.load(f"{self.fd}/{k}_nmel.npy")   # [T, M]
        dmel = np.load(f"{self.fd}/{k}_dmel.npy")   # [T, M]
        nf0  = np.load(f"{self.fd}/{k}_nf0.npy")    # [T]
        nene = np.load(f"{self.fd}/{k}_nene.npy")   # [T]

        T = min(len(nmel), len(dmel), len(nf0), len(nene))
        nmel = nmel[:T]; dmel = dmel[:T]
        nf0  = nf0[:T];  nene = nene[:T]
        return (
            torch.from_numpy(nmel).float(),  # [T, M]
            torch.from_numpy(nf0).float(),   # [T]
            torch.from_numpy(nene).float(),  # [T]
            torch.from_numpy(dmel).float(),  # [T, M]
        )

def collate(batch):
    minT = min([b[0].shape[0] for b in batch])
    out = []
    for nmel, nf0, nene, dmel in batch:
        out.append((nmel[:minT], nf0[:minT], nene[:minT], dmel[:minT]))
    nmel = torch.stack([o[0] for o in out])  # [B, T, M]
    nf0  = torch.stack([o[1] for o in out])  # [B, T]
    nene = torch.stack([o[2] for o in out])  # [B, T]
    dmel = torch.stack([o[3] for o in out])  # [B, T, M]
    return nmel, nf0, nene, dmel

def stft_loss(pred_mel, tgt_mel):
    return torch.mean(torch.abs(pred_mel - tgt_mel))

def main():
    ds = ProsodyDataset(FEAT_DIR)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate, drop_last=True)

    net = ProsodyNet(n_mels=N_MELS, emo_classes=1).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    l1 = nn.L1Loss()

    for ep in range(EPOCHS):
        net.train()
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{EPOCHS}")
        total = 0.0
        for nmel, nf0, nene, dmel in pbar:
            nmel, nf0, nene, dmel = nmel.to(DEVICE), nf0.to(DEVICE), nene.to(DEVICE), dmel.to(DEVICE)
            pred = net(nmel, nf0, nene, emo_id=0)   # extend for multi-emotions
            loss = l1(pred, dmel) + 0.5 * stft_loss(nmel+pred, nmel+dmel)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        torch.save(net.state_dict(), CKPT)
        print(f"[ep{ep+1}] avg_loss={total/len(dl):.4f} saved → {CKPT}")

if __name__ == "__main__":
    main()
