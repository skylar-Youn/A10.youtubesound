\
import os, glob, numpy as np, tqdm
from utils_audio import load_wav, wav_to_mel, extract_f0_pw, extract_energy, dtw_align

SR = 22050
HOP = 256
N_MELS = 80

NEU_DIR = "data/neutral"
EMO_DIR = "data/emotional_happy"   # example: happy
OUT_DIR = "feats"

os.makedirs(OUT_DIR, exist_ok=True)

def list_pairs(neu_dir, emo_dir):
    neu_files = sorted(glob.glob(os.path.join(neu_dir, "*.wav")))
    pairs = []
    for nf in neu_files:
        base = os.path.basename(nf)
        ef = os.path.join(emo_dir, base)
        if os.path.exists(ef):
            pairs.append((nf, ef))
    return pairs

def process_pair(nf, ef, idx):
    nwav, _ = load_wav(nf, SR)
    ewav, _ = load_wav(ef, SR)

    nmel = wav_to_mel(nwav, SR, N_MELS, 1024, HOP, 1024)
    emel = wav_to_mel(ewav, SR, N_MELS, 1024, HOP, 1024)

    # DTW alignment on mel
    nmel_aln, emel_aln = dtw_align(nmel, emel, metric='cosine')

    nf0 = extract_f0_pw(nwav, SR, HOP)
    ef0 = extract_f0_pw(ewav, SR, HOP)

    # naive resample f0 to aligned length
    T = nmel_aln.shape[0]
    def resample_1d(x, target_len):
        import numpy as np
        idx = np.linspace(0, len(x)-1, target_len)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

    nf0_aln = resample_1d(nf0, T)
    ef0_aln = resample_1d(ef0, T)

    nene = extract_energy(nmel_aln)
    eene = extract_energy(emel_aln)

    dmel = emel_aln - nmel_aln
    df0  = ef0_aln  - nf0_aln
    dene = eene     - nene

    np.save(os.path.join(OUT_DIR, f"{idx:05d}_nmel.npy"), nmel_aln)
    np.save(os.path.join(OUT_DIR, f"{idx:05d}_dmel.npy"), dmel)
    np.save(os.path.join(OUT_DIR, f"{idx:05d}_nf0.npy"),  nf0_aln)
    np.save(os.path.join(OUT_DIR, f"{idx:05d}_df0.npy"),  df0)
    np.save(os.path.join(OUT_DIR, f"{idx:05d}_nene.npy"), nene)
    np.save(os.path.join(OUT_DIR, f"{idx:05d}_dene.npy"), dene)

def main():
    pairs = list_pairs(NEU_DIR, EMO_DIR)
    print(f"found pairs: {len(pairs)}")
    for i, (nf, ef) in enumerate(tqdm.tqdm(pairs)):
        process_pair(nf, ef, i)

if __name__ == "__main__":
    main()
