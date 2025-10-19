# GPU 서버 이전 가이드

## 서버 정보

**현재 시스템:**
- 경로: `/home/sk/ws/youtubesound`

**목적지 서버:**
- SSH: `tripleyoung@192.168.219.112`
- 코드 경로: `/mnt/sdc1/ws/ws-sky/youtubesound/`
- 데이터 경로: `/mnt/sdb1/ws-sky-data/youtubesound-data/`

## 전략 선택: 전체 이전 vs 데이터/코드 분리

### 방법 A: 전체 이전 (간단, 시간 오래 걸림)
모든 데이터와 코드를 한 번에 옮김. 로컬 네트워크라서 빠를 수 있음.

### 방법 B: 데이터/코드 분리 (권장, 빠르고 유연함)
- 코드만 먼저 옮기고 (수십 MB, 수 초)
- 큰 데이터는 다른 디스크(`/mnt/sdb1`)에 별도 저장
- 빠르게 테스트 가능

---

## 방법 A: 전체 이전

### 1. 전체 파일 압축 및 전송
```bash
# 현재 시스템에서
cd /home/sk/ws
tar -czf youtubesound.tar.gz youtubesound/ \
  --exclude='.venv*' \
  --exclude='*.pyc' \
  --exclude='__pycache__'

# 새 서버로 전송
scp youtubesound.tar.gz tripleyoung@192.168.219.112:/mnt/sdc1/ws/ws-sky/
```

### 2. 새 서버에서 설치
```bash
# SSH로 새 서버 접속
ssh tripleyoung@192.168.219.112

# 압축 해제
cd /mnt/sdc1/ws/ws-sky/
tar -xzf youtubesound.tar.gz

# Python 가상환경 생성
cd youtubesound
python3 -m venv .venv-orpheus
source .venv-orpheus/bin/activate

# 의존성 설치
pip install -r requirements-orpheus.txt
pip install -e ./prosodynet_project
```

### 3. HuggingFace 로그인 (Orpheus 사용 시)
```bash
huggingface-cli login
# 또는
hf auth login
```

토큰 발급: https://huggingface.co/settings/tokens

### 4. 서버 실행
```bash
source .venv-orpheus/bin/activate
cd prosodynet_project
uvicorn server:app --reload --port 7000 --host 0.0.0.0
```

---

## 방법 B: 데이터/코드 분리 (권장)

### 1-A. rsync로 코드 직접 전송 (가장 빠르고 확실함, 로컬 네트워크)
```bash
# 현재 시스템에서 직접 rsync로 전송
rsync -avz --progress \
  --exclude='.venv/' \
  --exclude='.venv-orpheus/' \
  --exclude='*.pyc' \
  --exclude='__pycache__/' \
  --exclude='prosodynet_project/ckpt/' \
  --exclude='prosodynet_project/server_static/*.wav' \
  --exclude='prosodynet_project/server_static/*.npy' \
  --exclude='prosodynet_project/data/' \
  --exclude='data/' \
  --exclude='esd_data/' \
  --exclude='datasets/' \
  /home/sk/ws/youtubesound/ \
  tripleyoung@192.168.219.112:/mnt/sdc1/ws/ws-sky/youtubesound/

# 데이터는 별도 디스크로 전송
rsync -avz --progress \
  /home/sk/ws/youtubesound/prosodynet_project/ckpt/ \
  tripleyoung@192.168.219.112:/mnt/sdb1/ws-sky-data/youtubesound-data/prosodynet_project/ckpt/
```

### 1-B. 코드만 압축 및 전송 (tar 사용)
```bash
# 현재 시스템에서
cd /home/sk/ws
tar -czf youtubesound-code.tar.gz \
  --exclude='youtubesound/.venv' \
  --exclude='youtubesound/.venv-orpheus' \
  --exclude='youtubesound/*.pyc' \
  --exclude='youtubesound/__pycache__' \
  --exclude='youtubesound/prosodynet_project/ckpt' \
  --exclude='youtubesound/prosodynet_project/server_static/*.wav' \
  --exclude='youtubesound/prosodynet_project/server_static/*.npy' \
  --exclude='youtubesound/prosodynet_project/data' \
  --exclude='youtubesound/data' \
  --exclude='youtubesound/esd_data' \
  --exclude='youtubesound/datasets' \
  youtubesound/

# 파일 크기 확인 (50MB 내외여야 함)
ls -lh youtubesound-code.tar.gz

# 새 서버로 전송 (수 MB, 빠름)
scp youtubesound-code.tar.gz tripleyoung@192.168.219.112:/mnt/sdc1/ws/ws-sky/
```

### 2. 데이터 별도 전송 (느림, 수 GB)

#### 옵션 1: 체크포인트 직접 전송 (데이터 디스크에 저장)
```bash
# 현재 시스템에서 ProsodyNet 체크포인트만 압축
cd /home/sk/ws/youtubesound
tar -czf prosodynet-ckpt.tar.gz prosodynet_project/ckpt/*.pt

# 새 서버의 데이터 디스크로 전송
scp prosodynet-ckpt.tar.gz tripleyoung@192.168.219.112:/mnt/sdb1/ws-sky-data/youtubesound-data/
```

#### 옵션 2: 새 서버에서 직접 다운로드 (권장)
```bash
# 새 서버 접속
ssh tripleyoung@192.168.219.112

# 데이터 디렉토리 생성
mkdir -p /mnt/sdb1/ws-sky-data/youtubesound-data/ckpt

# 구글 드라이브에서 다운로드
cd /mnt/sdb1/ws-sky-data/youtubesound-data/ckpt
gdown <google-drive-file-id>

# 또는 HuggingFace에서 다운로드
huggingface-cli download <repo-id> --local-dir ./
```

### 3. 새 서버에서 설치

```bash
# SSH로 새 서버 접속
ssh tripleyoung@192.168.219.112

# 코드 압축 해제
cd /mnt/sdc1/ws/ws-sky/
tar -xzf youtubesound-code.tar.gz
cd youtubesound

# 체크포인트 압축 해제 (옵션 1 선택 시)
mkdir -p /mnt/sdb1/ws-sky-data/youtubesound-data
cd /mnt/sdb1/ws-sky-data/youtubesound-data/
tar -xzf prosodynet-ckpt.tar.gz  # prosodynet_project/ckpt/ 폴더가 생성됨

# 심볼릭 링크 생성 (코드에서 데이터로 연결)
cd /mnt/sdc1/ws/ws-sky/youtubesound/prosodynet_project
ln -s /mnt/sdb1/ws-sky-data/youtubesound-data/prosodynet_project/ckpt ./ckpt

# 디렉토리 구조 확인
ls -l /mnt/sdc1/ws/ws-sky/youtubesound/prosodynet_project/ckpt/  # prosodynet_multi.pt 있어야 함

# Python 가상환경 생성
cd /mnt/sdc1/ws/ws-sky/youtubesound
python3 -m venv .venv-orpheus
source .venv-orpheus/bin/activate

# 의존성 설치
pip install -r requirements-orpheus.txt
pip install -e ./prosodynet_project
```

### 4. HuggingFace 로그인 (Orpheus 사용 시)
```bash
huggingface-cli login
```

### 5. 환경 변수 설정 (데이터 경로)

코드와 데이터가 다른 디스크에 있으므로 환경 변수 설정:

```bash
# ~/.bashrc 또는 ~/.zshrc에 추가
export PROSODYNET_CKPT_DIR="/mnt/sdb1/ws-sky-data/youtubesound-data/ckpt"
export PROSODYNET_RVC_DIR="/mnt/sdb1/ws-sky-data/youtubesound-data/rvc"  # 필요시

# 설정 적용
source ~/.bashrc
```

또는 심볼릭 링크 사용 (더 간단, 권장):
```bash
# 코드 디렉토리에서 데이터 디렉토리로 링크 생성
cd /mnt/sdc1/ws/ws-sky/youtubesound/prosodynet_project
ln -s /mnt/sdb1/ws-sky-data/youtubesound-data/prosodynet_project/ckpt ./ckpt

# 확인
ls -l ckpt/  # 심볼릭 링크 확인
```

### 6. 서버 실행
```bash
cd /mnt/sdc1/ws/ws-sky/youtubesound
source .venv-orpheus/bin/activate
cd prosodynet_project
uvicorn server:app --reload --port 7000 --host 0.0.0.0
```

---

## 필수 파일 체크리스트

### 반드시 필요한 파일
- ✅ 프로젝트 코드 (prosodynet_project/, server.py 등)
- ✅ requirements-orpheus.txt
- ✅ prosodynet_project/ckpt/prosodynet_multi.pt (체크포인트)
- ✅ .gitignore, README.md

### 재생성 필요 (옮기지 않음)
- ❌ .venv*, .venv-orpheus (새 서버에서 재생성)
- ❌ __pycache__, *.pyc (자동 생성됨)
- ❌ prosodynet_project.egg-info (pip install -e 시 자동 생성)

### 선택 사항
- ⚠️ prosodynet_project/server_static/*.wav (생성된 오디오 파일, 새로 생성 가능)
- ⚠️ esd_data/, data/ (학습 데이터, 재학습 시에만 필요)

---

## 데이터 크기 참고

| 항목 | 크기 | 전송 시간 (예상) |
|------|------|-----------------|
| 코드만 | ~50 MB | 수 초 |
| 코드 + 체크포인트 | ~500 MB | 1-2분 |
| 전체 (학습 데이터 포함) | ~5-10 GB | 10-30분 |

---

## GPU 및 시스템 요구사항

### Coqui TTS + ProsodyNet 사용 시
- GPU: 12GB+ (RTX 3060, RTX 3080 등)
- CUDA: 11.x 이상
- Python: 3.10+

### Orpheus TTS 사용 시 (고급)
- GPU: **24GB+** (RTX 4090, A6000, A100 등)
- CUDA: 12.x 권장
- Python: 3.10+

GPU 확인:
```bash
nvidia-smi  # GPU 메모리 확인
```

---

## 주의사항

1. **체크포인트 파일** 포함 확인 (prosodynet_project/ckpt/prosodynet_multi.pt)
2. **HuggingFace 토큰** 재설정 필요 (Orpheus 사용 시)
3. **포트 7000** 방화벽 설정
4. **CUDA 버전** 호환성 확인 (nvidia-smi로 확인)
5. **가상환경** 반드시 재생성 (기존 .venv 옮기지 말 것)

---

## 빠른 요약

### rsync로 옮기기 (방법 B - 가장 권장, 로컬 네트워크)
```bash
# 1) 현재 시스템에서 rsync로 코드 전송
rsync -avz --progress \
  --exclude='.venv/' --exclude='.venv-orpheus/' \
  --exclude='*.pyc' --exclude='__pycache__/' \
  --exclude='prosodynet_project/ckpt/' \
  --exclude='prosodynet_project/data/' \
  --exclude='data/' --exclude='esd_data/' --exclude='datasets/' \
  /home/sk/ws/youtubesound/ \
  tripleyoung@192.168.219.112:/mnt/sdc1/ws/ws-sky/youtubesound/

# 2) 체크포인트 데이터 디스크로 전송
rsync -avz --progress \
  /home/sk/ws/youtubesound/prosodynet_project/ckpt/ \
  tripleyoung@192.168.219.112:/mnt/sdb1/ws-sky-data/youtubesound-data/prosodynet_project/ckpt/

# 3) 새 서버에서 설치
ssh tripleyoung@192.168.219.112
cd /mnt/sdc1/ws/ws-sky/youtubesound
python3 -m venv .venv-orpheus
source .venv-orpheus/bin/activate
pip install -r requirements-orpheus.txt
pip install -e ./prosodynet_project

# 4) 심볼릭 링크 생성 (코드에서 데이터로 연결)
cd /mnt/sdc1/ws/ws-sky/youtubesound/prosodynet_project
ln -s /mnt/sdb1/ws-sky-data/youtubesound-data/prosodynet_project/ckpt ./ckpt

# 5) 서버 실행
cd /mnt/sdc1/ws/ws-sky/youtubesound/prosodynet_project
source ../.venv-orpheus/bin/activate
uvicorn server:app --reload --port 7000 --host 0.0.0.0
```

### tar로 옮기기 (방법 B - 대안)
```bash
# 1) 현재 시스템에서 코드 압축 및 전송
cd /home/sk/ws
tar -czf youtubesound-code.tar.gz youtubesound/ \
  --exclude='youtubesound/.venv' --exclude='youtubesound/.venv-orpheus' \
  --exclude='youtubesound/prosodynet_project/ckpt' \
  --exclude='youtubesound/prosodynet_project/data' \
  --exclude='youtubesound/data' --exclude='youtubesound/esd_data'
scp youtubesound-code.tar.gz tripleyoung@192.168.219.112:/mnt/sdc1/ws/ws-sky/

# 2) 체크포인트 압축 및 전송 (데이터 디스크)
cd /home/sk/ws/youtubesound
tar -czf prosodynet-ckpt.tar.gz prosodynet_project/ckpt/*.pt
scp prosodynet-ckpt.tar.gz tripleyoung@192.168.219.112:/mnt/sdb1/ws-sky-data/youtubesound-data/

# 3) 새 서버에서 설치
ssh tripleyoung@192.168.219.112
cd /mnt/sdc1/ws/ws-sky/
tar -xzf youtubesound-code.tar.gz
cd youtubesound
python3 -m venv .venv-orpheus
source .venv-orpheus/bin/activate
pip install -r requirements-orpheus.txt
pip install -e ./prosodynet_project

# 4) 데이터 디스크에 체크포인트 압축 해제 및 심볼릭 링크
mkdir -p /mnt/sdb1/ws-sky-data/youtubesound-data
cd /mnt/sdb1/ws-sky-data/youtubesound-data/
tar -xzf prosodynet-ckpt.tar.gz
# 심볼릭 링크 생성
cd /mnt/sdc1/ws/ws-sky/youtubesound/prosodynet_project
ln -s /mnt/sdb1/ws-sky-data/youtubesound-data/prosodynet_project/ckpt ./ckpt

# 5) 서버 실행
cd /mnt/sdc1/ws/ws-sky/youtubesound/prosodynet_project
source ../.venv-orpheus/bin/activate
uvicorn server:app --reload --port 7000 --host 0.0.0.0
```

### 전체 옮기기 (방법 A - 간단)
```bash
# 1) 현재 시스템
cd /home/sk/ws
tar -czf youtubesound.tar.gz youtubesound/ \
  --exclude='.venv*' --exclude='*.pyc' --exclude='__pycache__'
scp youtubesound.tar.gz tripleyoung@192.168.219.112:/mnt/sdc1/ws/ws-sky/

# 2) 새 서버
ssh tripleyoung@192.168.219.112
cd /mnt/sdc1/ws/ws-sky/
tar -xzf youtubesound.tar.gz
cd youtubesound
python3 -m venv .venv-orpheus
source .venv-orpheus/bin/activate
pip install -r requirements-orpheus.txt
pip install -e ./prosodynet_project

# 3) 서버 실행
cd prosodynet_project
uvicorn server:app --reload --port 7000 --host 0.0.0.0
```

---

✅ 준비 완료!

이제 위의 명령어를 복사해서 실행하시면 됩니다.
