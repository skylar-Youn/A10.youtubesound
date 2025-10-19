\
import argparse
import importlib
import json
import librosa
import numpy as np
import soundfile as sf
import torch

def griffinlim_from_mel(mel, sr=22050, n_fft=1024, hop=256, win=1024, fmin=0, fmax=8000, iters=60):
    mel = np.exp(mel.T)  # [n_mels, T]
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=mel.shape[0], fmin=fmin, fmax=fmax)
    inv_mel = np.linalg.pinv(mel_filter)
    mag = np.clip(inv_mel @ mel, 1e-6, None)
    wav = librosa.griffinlim(mag, n_iter=iters, hop_length=hop, win_length=win)
    return wav

def main(args):
    mel = np.load(args.mel)  # [T, n_mels]

    if args.mode == "griffinlim":
        wav = griffinlim_from_mel(mel, sr=args.sr, n_fft=args.n_fft, hop=args.hop, win=args.win, iters=args.iters)
        sf.write(args.out, wav, args.sr)
        print(f"[Griffin-Lim] saved wav → {args.out}")
        return

    # HiFi-GAN mode - requires user to provide generator module path and ckpt
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # dynamic import: module must expose create_generator(config_dict)->nn.Module
    mod = importlib.import_module(args.generator_module)
    gen = mod.create_generator(cfg).eval()
    sd = torch.load(args.generator_ckpt, map_location="cpu")
    gen.load_state_dict(sd, strict=False)

    mel_t = torch.from_numpy(mel.T).unsqueeze(0).float()
    with torch.no_grad():
        wav = gen(mel_t).squeeze().cpu().numpy()
    sf.write(args.out, wav, cfg.get("sample_rate", args.sr))
    print(f"[HiFi-GAN] saved wav → {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--mode", choices=["griffinlim", "hifigan"], default="griffinlim")
    # Griffin-Lim params
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=60)
    # HiFi-GAN params
    ap.add_argument("--generator_module", help="python module path exposing create_generator(cfg)")
    ap.add_argument("--generator_ckpt", help="path to HiFi-GAN generator .pth")
    ap.add_argument("--config", help="HiFi-GAN config json")
    args = ap.parse_args()
    main(args)
