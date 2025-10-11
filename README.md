# A10.youtubesound
uvicorn server:app --reload --port 7000

source /home/sk/ws/youtubesound/.venv/bin/activate
주기적으로 다음 순서를 실행하면 최신 상태를 받을 수 있어요:

  cd /home/sk/ws/youtubesound
  git submodule update --remote Orpheus-TTS

  source /home/sk/ws/youtubesound/.venv/bin/activate
  export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.11/site-packages/
     nvidia/nvjitlink/lib:$VIRTUAL_ENV/lib/python3.11/site-packages/
     nvidia/cusparse/lib:${LD_LIBRARY_PATH}

# 공개 데이터셋 목록(키) 보기
./aihubshell -mode l

# datasetkey로 파일 목록 시도
./aihubshell -mode l -datasetkey 507715

# datapckagekey로 파일 목록 시도  ← 맞다면 여기가 성공
./aihubshell -mode pl -datapckagekey 507715
(.venv) (base) sk@sk-System-Product-Name:~/ws/youtubesound$ ./aihubshell -mode l -datasetkey 71631
==========================================
aihubshell version 25.09.19 v0.6
==========================================
Fetching file tree structure...
The contents are encoded in UTF-8 including Korean characters. 
If the following contents are not output normally, 
Please modify the character information of the OS. 
=================공지사항=================== 
========================================== 

    └─134-1.감정이 태깅된 자유대화 (성인)
        └─01-1.정식개방데이터
            ├─Training
            │  ├─01.원천데이터
            │  │  ├─TS_01.실내_1.zip | 47 GB | 507715
            │  │  ├─TS_01.실내_2.zip | 51 GB | 507716
            │  │  ├─TS_01.실내_3.zip | 51 GB | 507717
            │  │  ├─TS_01.실내_4.zip | 50 GB | 507718
            │  │  ├─TS_01.실내_5.zip | 22 GB | 507719
            │  │  └─TS_02.실외.zip | 37 GB | 507720
            │  └─02.라벨링데이터
            │      ├─TL_01.실내.zip | 97 MB | 507721
            │      └─TL_02.실외.zip | 17 MB | 507722
            └─Validation
                ├─01.원천데이터
                │  ├─VS_01.실내.zip | 38 GB | 507723
                │  └─VS_02.실외.zip | 6 GB | 507724
                └─02.라벨링데이터
                    ├─VL_01.실내.zip | 17 MB | 507725
                    └─VL_02..실외.zip | 3 MB | 507726


(.venv) (base) sk@sk-System-Product-Name:~/ws/youtubesound$ ./aihubshell -mode d -aihubapikey DBE9CC02-5E69-4E86-A7DD-754A41EF4DD9 -datasetkey 71631

## Orpheus TTS 참고 메모

- Orpheus 엔진은 `git submodule update --init --recursive` 명령으로 서브모듈을 내려받은 뒤 사용할 수 있습니다.
- 빠른 데모 실행: `pip install orpheus-speech` 설치 후 `python run_orpheus_tts.py`를 실행하면 `output.wav`가 생성됩니다.
- 자주 사용할 모델은 `huggingface-cli download`로 미리 받아 두면 네트워크 없이도 빠르게 로딩할 수 있습니다.


  ✅ 완료된 작업

  백엔드 (server.py)

  1. use_prosodynet 필드 추가 (기본값: True)
  2. ProsodyNet 체크 해제 시:
    - 감정 변환 건너뛰기
    - TTS 결과를 바로 최종 출력으로 사용
    - 체크포인트 없어도 500 에러 발생 안 함
  3. ProsodyNet 체크 시: 기존과 동일하게 감정 변환 수행

  프론트엔드 (index.html, app.js)

  1. "ProsodyNet 감정 변환 사용" 체크박스 추가
  2. 체크 해제 시:
    - 감정 선택 UI 숨김
    - Vocoder 설정 UI 숨김 (ProsodyNet 없으면 불필요)
  3. 선택 상태 localStorage에 자동 저장

  🚀 사용 방법

  옵션 1: ProsodyNet 없이 TTS만 사용 (체크포인트 불필요)

  1. 브라우저에서 http://localhost:7000 접속
  2. "ProsodyNet 감정 변환 사용" 체크 해제
  3. TTS 엔진 선택 (Coqui 또는 Orpheus)
  4. 문장 입력 후 합성 실행
  → 깔끔한 음성 즉시 생성! ✅

  옵션 2: ProsodyNet으로 감정 변환 (체크포인트 필요)

  1. "ProsodyNet 감정 변환 사용" 체크 ✅
  2. 감정 선택 (Happy/Sad/Angry 등)
  3. TTS 엔진 및 Vocoder 설정
  4. 합성 실행
  → 감정이 입힌 음성 생성! 🎭

  📝 구조 요약

  TTS 생성 (Coqui/Orpheus)
      ↓
  ProsodyNet 사용?
      ├─ Yes → 감정 변환 → Vocoder → 감정 음성 ✨
      └─ No  → 원본 음성 그대로 출력 ✅
      ↓
  (선택) RVC 후처리


    사용 방법

  브라우저에서 TTS 엔진을 Coqui로 선택하세요!

  API 예시:

  curl -X POST http://127.0.0.1:7000/synthesize \
    -H "Content-Type: application/json" \
    -d '{
      "text": "안녕하세요",
      "tts_engine": "coqui",
      "use_prosodynet": true,
      "emotion_id": 1
    }'

  감정 ID:

  - 0: Angry (화남)
  - 1: Happy (기쁨)
  - 2: Sad (슬픔)
  - 3: Surprise (놀람)

  ---
  Orpheus TTS 사용하려면 (선택사항)

  HuggingFace 접근 권한이 필요합니다:

  1. https://huggingface.co/canopylabs/orpheus-tts-0.1-finetu
  ne-prod 방문
  2. "Request Access" 클릭
  3. 승인 후 HuggingFace 토큰 설정:
  huggingface-cli login

  지금은 Coqui TTS를 사용하시면 됩니다! 🎉


 Orpheus를 꼭 사용하려면?

  별도 가상환경을 만들어야 합니다:

  # Orpheus 전용 환경
  python -m venv .venv-orpheus
  source .venv-orpheus/bin/activate
  pip install orpheus-speech numpy>=2.0

  현재 환경에서는 Coqui TTS + ProsodyNet만 사용 가능합니다!


  환경 전환 방법

  Coqui TTS + ProsodyNet 사용:
  source .venv/bin/activate
  uvicorn server:app --reload --port 7000

  Orpheus TTS 사용:
  source .venv-orpheus/bin/activate
  uvicorn server:app --reload --port 7000

  브라우저를 새로고침하고 TTS Engine을 Orpheus로 선택하면 됩니다!


lsof -ti :8000 | xargs kill -9


  해결 방법

  옵션 1: Orpheus 설정 조정 (복잡함)

  prosodynet_project/server.py에서 OrpheusModel 초기화 수정 필요:

● Read(file_path: "/home/sk/ws/youtubesound/prosodynet_project/server.py", offset: 154, limit:
      20)
  ⎿  Read 20 lines

● OrpheusModel은 vllm 설정을 내부적으로 관리하므로 max_model_len을 쉽게 조정할 수 없습니다.

  옵션 2: Coqui TTS 사용 (권장) ✅

  12GB GPU에서는 Coqui TTS + ProsodyNet이 가장 안정적입니다:

  # 서버 종료
  pkill -f "uvicorn.*7000"

  # Coqui TTS 환경으로 전환
  source .venv/bin/activate
  uvicorn server:app --reload --port 7000

  브라우저 새로고침 후:
  - TTS Engine: Coqui
  - ProsodyNet: 체크
  - Emotion ID: 0~3

  ---
  요약

  Orpheus TTS는 24GB+ GPU가 필요합니다 (A6000, RTX 4090 등).

  현재 12GB GPU로는:
  - ✅ Coqui TTS + ProsodyNet 작동
  - ❌ Orpheus TTS 메모리 부족

  Coqui TTS를 사용하시는 것을 강력히 권장합니다! 🎯