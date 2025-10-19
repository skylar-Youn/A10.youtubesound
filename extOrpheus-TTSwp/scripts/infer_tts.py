import argparse, yaml, soundfile as sf
import numpy as np

def synth_dummy(text=None, lyrics=None, score=None, mode="tts"):
    # Dummy audio for scaffold (1s silence). Replace with real inference.
    return np.zeros(int(24000*1.0), dtype=np.float32), 24000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--text', default=None)
    ap.add_argument('--lyrics', default=None)
    ap.add_argument('--score', default=None)
    ap.add_argument('--mode', default="tts", choices=["tts","singing"])
    ap.add_argument('--out', required=True)
    ap.add_argument('--vibrato', type=float, default=0.6)
    ap.add_argument('--vib_rate', type=float, default=6.2)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    audio, sr = synth_dummy(args.text, args.lyrics, args.score, args.mode)
    sf.write(args.out, audio, sr)
    print(f"[infer] wrote {args.out} ({len(audio)/sr:.2f}s)")

if __name__ == "__main__":
    main()
