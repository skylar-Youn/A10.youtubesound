# GPU 서버 이전 가이드

## 1. 파일 복사
```bash
# 현재 시스템에서
cd /home/sk/ws
tar -czf youtubesound.tar.gz youtubesound/ --exclude='.venv*' --exclude='*.pyc' --exclude='__pycache__'

# 새 서버로 전송
scp youtubesound.tar.gz user@new-server:/path/to/destination/
```

## 2. 새 서버에서 설치

```bash
# 압축 해제
tar -xzf youtubesound.tar.gz
cd youtubesound

# Python 가상환경 생성
python3 -m venv .venv-orpheus
source .venv-orpheus/bin/activate

# 의존성 설치
pip install -r requirements-orpheus.txt
pip install -e ./prosodynet_project
```

## 3. HuggingFace 로그인

```bash
huggingface-cli login
# 또는
hf auth login
```

토큰 입력: https://huggingface.co/settings/tokens

## 4. 서버 실행

```bash
source .venv-orpheus/bin/activate
uvicorn server:app --reload --port 7000 --host 0.0.0.0
```

## 필수 파일 체크리스트

- ✅ 프로젝트 코드 (prosodynet_project/, server.py 등)
- ✅ requirements-orpheus.txt
- ✅ prosodynet_project/ckpt/prosodynet_multi.pt (체크포인트)
- ✅ .gitignore, README.md
- ❌ .venv* (재생성 필요)
- ❌ __pycache__, *.pyc (불필요)

## GPU 확인

```bash
nvidia-smi  # 24GB+ 확인
```

## 주의사항

1. **체크포인트 파일** 포함 확인 (prosodynet_project/ckpt/)
2. **HuggingFace 토큰** 재설정 필요
3. **포트 7000** 방화벽 설정
4. **CUDA 버전** 호환성 확인 (CUDA 12.x 권장)


  빠른 이전 방법:

  1) 현재 시스템에서:
  cd /home/sk/ws
  tar -czf youtubesound.tar.gz youtubesound/ \
    --exclude='.venv*' --exclude='*.pyc'
  --exclude='__pycache__'

  2) 새 서버로 전송:
  scp youtubesound.tar.gz
  user@new-gpu-server:/path/

  3) 새 서버에서:
  tar -xzf youtubesound.tar.gz
  cd youtubesound
  python3 -m venv .venv-orpheus
  source .venv-orpheus/bin/activate
  pip install -r requirements-orpheus.txt
  pip install -e ./prosodynet_project
  huggingface-cli login  # 토큰 입력
  uvicorn server:app --reload --port 7000 --host
  0.0.0.0

  ---
  ✅ 준비 완료

  - requirements-orpheus.txt 생성됨
  - MIGRATION.md 가이드 생성됨

  새 GPU 서버 요구사항:
  - GPU: 24GB+ (RTX 4090, A6000, A100 등)
  - CUDA: 12.x
  - Python: 3.10+

  파일들을 옮기시면 됩니다! 🚀