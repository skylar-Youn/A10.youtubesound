import argparse, yaml, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("[train_backbone] loaded config keys:", list(cfg.keys()))
    # TODO:
    # 1) build dataset (modules/dataset.py)
    # 2) load Orpheus backbone and insert CLN layers (modules/cln.py)
    # 3) add emotion encoder, alignment loss
    # 4) train loop (fp16), save checkpoints
    # This is a scaffold; wire with your Orpheus codebase.

if __name__ == "__main__":
    main()
