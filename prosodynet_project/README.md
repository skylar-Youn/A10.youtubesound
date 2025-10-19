# ProsodyNet Emotion Post-Processor (DTW + HiFi-GAN hook + FastAPI)

This project gives you an end-to-end **emotion injection** pipeline you can place **after TTS and/or RVC**:

```
TXT --> (TTS Neutral) --> neutral.wav
                      --> [ProsodyNet: DTW-aligned Δ prosody] --> emotional_mel.npy
                      --> (Vocoder: HiFi-GAN/UnivNet) --> emotional.wav
(Option) --> RVC voice conversion on the final wav
```

### Requirements
- Place trained ProsodyNet checkpoints at `ckpt/prosodynet.pt` (and optionally `ckpt/prosodynet_multi.pt`).
- Multi-speaker/multi-lingual Coqui TTS models need a `speaker` (and optionally `language`). The server will auto-pick defaults, but you can override them via the UI or API.

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

### RVC Post-Processing
- 웹 UI에서 **RVC 후처리 사용**을 체크하면 추가 입력란이 열립니다. RVC CLI는 `python infer-web.py`처럼 직접 실행 가능한 명령어를 입력합니다.
- 서버는 해당 CLI 호출 시 `--input_path`, `--output_path`, `--pth_path`, `--config_path`, `--f0method`, `--pitch_shift` 인자를 자동으로 덧붙입니다. 사용하는 RVC 스크립트가 같은 인자를 지원하는지 확인하세요.
- RVC 기본 경로를 지정한 뒤 **목록 불러오기** 버튼을 누르면 하위의 `.pth`와 설정 파일 목록을 자동으로 채웁니다. 목록에서 선택하면 상대경로가 저장되고, 요청 시 기본 경로와 합쳐집니다.
- 기본값으로 `/home/sk/ws/SD/RVC_TRAIN/Mangio-RVC-v23.7.0`과 `logs/mi-test7/G_2333333.pth`, `logs/mi-test7/config.json`, `python infer-web.py`가 세팅되어 있으니 바로 테스트할 수 있습니다. 필요하면 원하는 모델·CLI로 교체하세요.
- REST API를 직접 호출할 때는 `use_rvc: true`와 함께 `rvc` 객체를 전달하세요. 예시는 위 JSON 블록을 참고하면 됩니다.

### Building Accurate Emotional Voices
1. **데이터 정비:** 중립/감정 음성이 같은 대사를 말한 페어 형태로 정렬돼 있는지 확인하세요. 길이 차이가 크거나 침묵이 오래 남아 있으면 DTW가 비정상적으로 늘어나므로, 필요한 경우 수동으로 트리밍하거나 문장 단위로 잘라 둡니다.
2. **DTW/피처 점검:** `python preprocess.py` 실행 후 생성된 정렬 파형과 멜을 시각화하거나 몇 샘플을 들어보며 잘 맞는지 확인합니다. 엇갈린 구간이 보이면 다시 데이터 손질을 진행하세요.
3. **F0·에너지 Δ 활용:** ProsodyNet은 ΔF0/ΔEnergy까지 학습할 수 있습니다. `extract_f0_pw`, `extract_energy` 결과를 Plot 또는 Praat로 검증하고, 감정 음성에서 기대하는 억양/강세 패턴이 추출되는지 살펴보세요.
4. **감정 ID 일치:** `emo_classes`와 `emotion_id`가 학습 데이터의 감정 라벨과 정확히 대응해야 합니다. 잘못된 매핑은 Δ를 다른 감정으로 학습시키므로, preprocessing 단계에서 생성한 `*_emoid.npy`를 확인하세요.
5. **Vocoder 설정 일치:** HiFi-GAN을 사용할 때는 학습에 쓰인 config와 checkpoint 조합을 그대로 가져와야 샘플링레이트·hop length 등이 맞습니다. ProsodyNet 저장소의 `vocoder/hifigan_config_example.json`을 참조해 각 항목을 체크하세요.
6. **RVC 후처리 검증:** RVC CLI가 ProsodyNet에서 전달하는 인자(`--input_path`, `--output_path`, `--pth_path`, `--config_path`, `--f0method`, `--pitch_shift`)를 모두 지원하는지 확인하고, 변환 전/후 음성을 비교試청해 감정 표현이 유지되는지 판단합니다.
7. **청취 및 반복 보정:** 감정 음성이 충분히 분별되는지 A/B 청취하면서 모델 파라미터나 데이터 구성을 조정하세요. 필요하면 추가 감정 데이터를 모으거나 훈련 epoch, 학습률을 재조정합니다.

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



좋아요. IndexTTS-2에서 한국어·일본어로 테스트/사용할 수 있는 바로가기를 정리했어요.

공식 GitHub (설치/추론 가이드)
https://github.com/index-tts/index-tts
 — 최신 커밋과 설치 방법, 감정 제어(emo_text/벡터) 설명. 
GitHub

웹 데모 (Hugging Face Space) – 텍스트·레퍼런스 오디오 업로드 후 감정 조절
https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo
 
Hugging Face

(KO/JA 선택 지원) ComfyUI 노드 문서 – language 파라미터에 "ja"(일본어), "ko"(한국어) 옵션 명시
https://www.runcomfy.com/comfyui-nodes/ComfyUI-Index-TTS/index-tts-node
 
RunComfy

일본어 데모 케이스(공식 페이지) – 일본 애니/드라마 장면 더빙 사례 영상 수록
https://index-tts.github.io/index-tts2.github.io/
 
index-tts.github.io

일본어 학습량 언급(모델 호스팅 문서) – JA 데이터 10만+ 시간으로 학습했다고 명시
https://www.siliconflow.com/models/indexteam-indextts-2
 
siliconflow.com

메모: 공식 이슈에서 EN/ZH 최적화가 중심이고, 다른 언어(예: KO) 파인튜닝/적응 요청이 진행 중입니다. 바로 쓸 때 품질 편차가 있을 수 있어요. 
GitHub