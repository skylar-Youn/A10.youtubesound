\
import numpy as np
import librosa, pyworld as pw
import soundfile as sf

def load_wav(path, sr=22050):
    wav, s = sf.read(path)
    if s != sr:
        wav = librosa.resample(wav, orig_sr=s, target_sr=sr)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav.astype(np.float32), sr

def wav_to_mel(wav, sr=22050, n_mels=80, n_fft=1024, hop=256, win=1024, fmin=0, fmax=8000):
    S = librosa.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win, window='hann', center=True)
    mag = np.abs(S)
    mel_fb = librosa.filters.mel(sr, n_fft, n_mels, fmin=fmin, fmax=fmax)
    mel = np.dot(mel_fb, mag)
    mel = np.log(np.clip(mel, 1e-5, None))
    return mel.T  # [T, n_mels]

def extract_f0_pw(wav, sr=22050, hop=256):
    _f0, t = pw.harvest(wav.astype(np.float64), sr, frame_period=hop/sr*1000)
    f0 = pw.stonemask(wav.astype(np.float64), _f0, t, sr)
    return f0.astype(np.float32)  # [T]

def extract_energy(mel):  # mel [T, n_mels]
    spec = np.exp(mel)
    energy = np.sqrt((spec**2).mean(axis=1)).astype(np.float32)  # [T]
    return energy

def dtw_align(src_feat, tgt_feat, metric='cosine'):
    """
    Align tgt_feat to src_feat with DTW.
    src_feat: [T1, D]
    tgt_feat: [T2, D]
    Returns aligned_src, aligned_tgt with same length.
    """
    # distance matrix via cosine distance
    D = librosa.sequence.dtw(X=src_feat.T, Y=tgt_feat.T, metric=metric)[0]  # cost matrix
    # path
    wp = librosa.sequence.dtw(X=src_feat.T, Y=tgt_feat.T, metric=metric)[1]
    path = list(zip(*wp))  # (i_idx, j_idx)
    path = path[::-1]  # forward

    src_aligned = []
    tgt_aligned = []
    for i, j in path:
        src_aligned.append(src_feat[i])
        tgt_aligned.append(tgt_feat[j])
    return np.stack(src_aligned, 0), np.stack(tgt_aligned, 0)
