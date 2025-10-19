# Orpheus++: Emotion-Enhanced TTS with Singing Mode

This is a research boilerplate to implement the proposed improvements:
- Continuous **Emotion Embeddings** (8–16D)
- **CLN** (Emotion-Conditional LayerNorm) fusion
- **Cross-Modal Emotion Alignment** loss
- **Diffusion Refiner** for prosody and singing expressivity
- **SVS** (Score-based Singing) and **Free-Form Singing**

## Quickstart (Linux + CUDA)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train backbone with CLN & alignment
python scripts/train_backbone.py --config configs/base_tts.yaml

# Train diffusion refiner
python scripts/train_refiner.py --config configs/base_tts.yaml

# Inference (TTS)
python scripts/infer_tts.py --text "달빛에 노래하는 로봇개" --config configs/base_tts.yaml --out out_tts.wav

# Inference (Singing with score)
python scripts/infer_tts.py --mode singing --lyrics data/examples/lyrics.txt --score data/examples/score.musicxml   --config configs/singing.yaml --out out_song.wav --vibrato 0.7 --vib_rate 6.2
```

## Folders
- `configs/` YAML configs (TTS / Singing)
- `modules/` core model pieces (CLN, score encoder, F0 generator, dataset)
- `scripts/` training and inference entrypoints
- `data/examples/` toy MusicXML + lyrics for sanity checks
- `metrics/` objective metrics + MOS protocol

> NOTE: This is a skeleton. Replace stubs with your implementations, wire your dataset, and adapt the model APIs to your Orpheus codebase.
