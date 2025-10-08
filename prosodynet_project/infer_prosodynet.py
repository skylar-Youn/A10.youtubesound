\
import os, argparse, numpy as np, torch
from utils_audio import load_wav, wav_to_mel, extract_f0_pw, extract_energy
from prosodynet import ProsodyNet

SR=22050; HOP=256; N_MELS=80

def main(args):
    wav, _ = load_wav(args.input, SR)
    nmel = wav_to_mel(wav, SR, N_MELS, 1024, HOP, 1024)        # [T, M]
    nf0  = extract_f0_pw(wav, SR, HOP)                         # [T]
    nene = extract_energy(nmel)                                # [T]

    net = ProsodyNet(n_mels=N_MELS, emo_classes=1)
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    net.eval()

    with torch.no_grad():
        nmel_t = torch.from_numpy(nmel).unsqueeze(0).float()   # [1, T, M]
        nf0_t  = torch.from_numpy(nf0).unsqueeze(0).float()    # [1, T]
        nene_t = torch.from_numpy(nene).unsqueeze(0).float()   # [1, T]
        dmel_t = net(nmel_t, nf0_t, nene_t, emo_id=args.emotion_id)  # [1, T, M]
        emel_t = nmel_t + dmel_t
    emel = emel_t.squeeze(0).cpu().numpy()
    np.save(args.output_mel, emel)
    print(f"saved emotional mel → {args.output_mel}")
    print("Next: use your vocoder to convert mel to wav.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="neutral wav (from TTS/RVC)")
    ap.add_argument("--ckpt", default="ckpt/prosodynet.pt")
    ap.add_argument("--output_mel", default="emotional_mel.npy")
    ap.add_argument("--emotion_id", type=int, default=0)
    args = ap.parse_args()
    main(args)
