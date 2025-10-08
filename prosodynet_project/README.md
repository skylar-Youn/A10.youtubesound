# ProsodyNet Emotion Post-Processor (DTW + HiFi-GAN hook + FastAPI)

This project gives you an end-to-end **emotion injection** pipeline you can place **after TTS and/or RVC**:

```
TXT --> (TTS Neutral) --> neutral.wav
                      --> [ProsodyNet: DTW-aligned Δ prosody] --> emotional_mel.npy
                      --> (Vocoder: HiFi-GAN/UnivNet) --> emotional.wav
(Option) --> RVC voice conversion on the final wav
```

## Features
- **DTW alignment** between neutral and emotional pairs during preprocessing.
- **ProsodyNet** (small Conv + Transformer) learns **Δmel** (optionally ΔF0/ΔEnergy) from neutral to target emotion.
- **HiFi-GAN hook** (`vocoder/hifigan_infer.py`) to synthesize waveform from mel.
- **FastAPI server** (`server.py`) to run the pipeline via REST:
  - `/synthesize` : text -> (Coqui TTS neutral) -> ProsodyNet -> Vocoder -> wav
  - Optional RVC post-process using your RVC CLI (stub via subprocess).

## Quickstart

1) Install packages (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```

2) Prepare paired data (same filenames preferred):

```
data/
  neutral/
    0001.wav, 0002.wav, ...
  emotional_happy/
    0001.wav, 0002.wav, ...
```

3) Preprocess (DTW alignment + feature extraction):
```bash
python preprocess.py
```

4) Train ProsodyNet:
```bash
python train_prosodynet.py
```

5) Inference (offline):
```bash
# Convert a neutral wav (from any TTS/RVC) into emotional mel
python infer_prosodynet.py --input neutral_line.wav --output_mel emotional_mel.npy

# Vocoder (HiFi-GAN) to convert mel -> wav
python vocoder/hifigan_infer.py --mel emotional_mel.npy --generator_ckpt path/to/generator.pth --config vocoder/hifigan_config_example.json --out emotional.wav
```

6) Run the API server:
```bash
python server.py
# POST /synthesize with JSON:
# {
#   "text": "감정을 넣을 문장입니다.",
#   "emotion_id": 0,
#   "tts_model": "tts_models/ko/kss/vits",   # or any Coqui model id
#   "use_rvc": false,
#   "rvc": {
#       "pth": "rvc/G_10000.pth",
#       "config": "rvc/config.json",
#       "cli": "python infer-web.py"  # your RVC CLI
#   }
# }
```

### FastAPI UI & 환경
- `server.py`가 시작되면 프로젝트 루트의 `.venv`를 자동 활성화하여 의존성 충돌 없이 실행됩니다.
- 루트 페이지(`/`)에서 감정 선택, TTS 모델, Vocoder, RVC 옵션을 설정할 수 있는 웹 UI(`server_static/`)가 제공됩니다.
- 응답으로 전달되는 경로는 `/static/...` 형태이므로 브라우저에서 바로 재생·다운로드가 가능합니다.
- 기본 TTS 모델은 다국어 지원 `tts_models/multilingual/multi-dataset/xtts_v2`이며, 다른 모델을 사용하려면 Coqui TTS 패키지가 해당 ID를 지원해야 합니다.
- RVC 경로가 `/home/sk/ws/SD/RVC_TRAIN/Mangio-RVC-v23.7.0/weights`에 있다면 RVC 섹션에서 **목록 불러오기** 버튼으로 `.pth`/설정 파일 이름 목록을 바로 불러와 선택할 수 있습니다. (`PROSODYNET_RVC_DIR` 환경변수로 기본 경로를 재정의할 수 있음)
- UI는 마지막으로 입력한 TTS/RVC 옵션과 텍스트를 브라우저에 저장하므로 새로고침 후에도 동일한 설정으로 이어서 작업할 수 있습니다.

> **Note:** You must provide your own **HiFi-GAN** generator checkpoint (and config) and optional **RVC** CLI.  
> For a minimal demo without a vocoder, you can implement Griffin-Lim or hook any vocoder you already use.

## Tips
- Better alignment -> Better Δ estimation. If you have time, curate more neutral/emotional pairs.
- If you have multiple emotions, set `emo_classes > 1` in `ProsodyNet` and pass `emotion_id` during training/inference.
- For production, export vocoder & ProsodyNet to TorchScript/ONNX if needed.



## Multi-Emotion Data Layout
For multiple emotions, organize like this:
```
data/
  neutral/
    0001.wav, 0002.wav, ...
  emotions/
    happy/
      0001.wav, 0002.wav, ...
    sad/
      0001.wav, 0002.wav, ...
    angry/
      0001.wav, 0002.wav, ...
```
`preprocess_multi.py` will scan `data/emotions/*` and create DTW-aligned features and `*_emoid.npy` tags.
Training script will automatically set `emo_classes` based on discovered IDs.


uvicorn server:app --reload --port 7000
lsof -ti :7000 | xargs kill -9
source .venv/bin/activate
