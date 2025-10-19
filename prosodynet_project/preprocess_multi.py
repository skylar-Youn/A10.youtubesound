#!/usr/bin/env python3
"""
병렬 처리 멀티 감정 전처리 스크립트
multiprocessing.Pool을 사용하여 CPU 코어를 모두 활용
"""
import os
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils_audio import load_wav, wav_to_mel, extract_f0_pw, extract_energy, dtw_align

SR = 22050
HOP = 256
N_MELS = 80

NEU_DIR = "data/neutral"
EMO_ROOT = "data/emotions"   # subdirs per emotion
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

def resample_1d(x, target_len):
    import numpy as np
    if len(x) == target_len:
        return x.astype(np.float32)
    idx = np.linspace(0, len(x)-1, target_len)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

def process_pair(args):
    """단일 페어 처리 (병렬 처리용 래퍼)"""
    nf, ef, idx, emoid = args
    try:
        nwav, _ = load_wav(nf, SR)
        ewav, _ = load_wav(ef, SR)

        nmel = wav_to_mel(nwav, SR, N_MELS, 1024, HOP, 1024)
        emel = wav_to_mel(ewav, SR, N_MELS, 1024, HOP, 1024)

        # DTW alignment on mel
        nmel_aln, emel_aln = dtw_align(nmel, emel, metric='cosine')

        nf0 = extract_f0_pw(nwav, SR, HOP)
        ef0 = extract_f0_pw(ewav, SR, HOP)

        T = nmel_aln.shape[0]
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
        np.save(os.path.join(OUT_DIR, f"{idx:05d}_emoid.npy"), np.array([emoid], dtype=np.int64))

        return True
    except Exception as e:
        print(f"❌ Error processing {nf} <-> {ef}: {e}")
        return False

def main():
    # map emotion name -> id
    emo_dirs = sorted([d for d in glob.glob(os.path.join(EMO_ROOT, "*")) if os.path.isdir(d)])
    emo_map = {os.path.basename(d): i for i, d in enumerate(emo_dirs)}
    print("✅ Emotion map:", emo_map)

    # 모든 페어 수집
    all_tasks = []
    idx = 0
    for emo_name, emoid in emo_map.items():
        emo_dir = os.path.join(EMO_ROOT, emo_name)
        pairs = list_pairs(NEU_DIR, emo_dir)
        print(f"  [{emo_name}] pairs: {len(pairs)}")

        for nf, ef in pairs:
            all_tasks.append((nf, ef, idx, emoid))
            idx += 1

    total_pairs = len(all_tasks)
    n_workers = cpu_count()
    print(f"\n🚀 Total pairs: {total_pairs}")
    print(f"⚙️  Using {n_workers} CPU cores for parallel processing\n")

    # 병렬 처리
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_pair, all_tasks),
            total=total_pairs,
            desc="Processing",
            unit="pair"
        ))

    success_count = sum(results)
    print(f"\n✅ Preprocessing complete: {success_count}/{total_pairs} successful")
    print(f"📁 Features saved to: {os.path.abspath(OUT_DIR)}/")

if __name__ == "__main__":
    main()
