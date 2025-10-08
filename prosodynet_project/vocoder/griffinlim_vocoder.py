\
import argparse, numpy as np, librosa, soundfile as sf

def mel_to_linear(mel_db, sr=22050, n_fft=1024, fmin=0, fmax=8000):
    # mel_db: [T, n_mels] in log scale
    mel = np.exp(mel_db.T)  # [n_mels, T] in magnitude-like
    mel_filter = librosa.filters.mel(sr, n_fft, mel.shape[0], fmin=fmin, fmax=fmax)
    # Pseudo-inverse to approximate linear spectrogram
    inv_mel = np.linalg.pinv(mel_filter)
    mag = np.clip(inv_mel @ mel, 1e-6, None)  # [n_fft//2+1, T]
    return mag

def griffinlim(mag, n_iter=60, n_fft=1024, hop=256, win_length=1024):
    # mag: [n_fft//2+1, T]
    return librosa.griffinlim(mag, n_iter=n_iter, hop_length=hop, win_length=win_length)

def main(args):
    mel = np.load(args.mel)        # [T, n_mels] (log-mel)
    mag = mel_to_linear(mel, sr=args.sr, n_fft=args.n_fft, fmin=0, fmax=8000)
    wav = griffinlim(mag, n_iter=args.iters, n_fft=args.n_fft, hop=args.hop, win_length=args.win)
    sf.write(args.out, wav, args.sr)
    print(f"saved wav → {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()
    main(args)
