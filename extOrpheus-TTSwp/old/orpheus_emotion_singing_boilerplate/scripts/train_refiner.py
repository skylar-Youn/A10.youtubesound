import argparse, yaml, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("[train_refiner] loaded config keys:", list(cfg.keys()))
    # TODO:
    # 1) Build mel-rough -> mel-target pairs
    # 2) Train UNet1D diffusion refiner (steps in cfg['refiner']['steps'])
    # 3) Save refiner checkpoint

if __name__ == "__main__":
    main()
