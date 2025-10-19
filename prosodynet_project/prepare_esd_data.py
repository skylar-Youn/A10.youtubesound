#!/usr/bin/env python3
"""
ESD 데이터셋을 ProsodyNet multi-emotion 학습 형식으로 변환하는 스크립트
ESD는 오프셋 기반 파일명을 사용하므로 통일된 인덱스로 변환
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# 경로 설정
ESD_ROOT = Path("data_raw/Emotion Speech Dataset")
OUTPUT_ROOT = Path("data")

# 감정 매핑 및 파일 번호 오프셋
EMOTIONS = {
    "Neutral": ("neutral", 0),
    "Angry": ("angry", 350),
    "Happy": ("happy", 700),
    "Sad": ("sad", 1050),
    "Surprise": ("surprise", 1400)
}

def prepare_data():
    """ESD 데이터를 ProsodyNet 형식으로 변환"""

    # 출력 디렉토리 초기화
    neutral_dir = OUTPUT_ROOT / "neutral"
    emotions_dir = OUTPUT_ROOT / "emotions"

    # 기존 데이터 삭제 후 재생성
    import shutil as sh
    if neutral_dir.exists():
        sh.rmtree(neutral_dir)
    if emotions_dir.exists():
        sh.rmtree(emotions_dir)

    neutral_dir.mkdir(parents=True, exist_ok=True)
    for emo_name, _ in EMOTIONS.values():
        if emo_name != "neutral":
            (emotions_dir / emo_name).mkdir(parents=True, exist_ok=True)

    # 화자 목록
    speakers = sorted([d for d in ESD_ROOT.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"발견된 화자 수: {len(speakers)}")

    # 각 화자의 데이터 처리
    for speaker_dir in tqdm(speakers, desc="화자 처리 중"):
        speaker_id = speaker_dir.name

        # 각 감정 처리
        for esd_emo, (emo_name, offset) in EMOTIONS.items():
            emo_src = speaker_dir / esd_emo
            if not emo_src.exists():
                continue

            for wav_file in emo_src.glob("*.wav"):
                # 파일명에서 번호 추출 (예: 0001_000701.wav -> 701)
                file_num = int(wav_file.name.split('_')[1].split('.')[0])

                # 정규화된 인덱스 계산 (오프셋 제거)
                normalized_idx = file_num - offset

                # 새 파일명 생성 (예: spk0001_00001.wav)
                new_name = f"spk{speaker_id}_{normalized_idx:05d}.wav"

                # 목적지 결정
                if emo_name == "neutral":
                    dst = neutral_dir / new_name
                else:
                    dst = emotions_dir / emo_name / new_name

                # 복사
                if not dst.exists():
                    shutil.copy2(wav_file, dst)

    # 통계 출력
    print("\n=== 데이터 준비 완료 ===")
    neutral_count = len(list(neutral_dir.glob('*.wav')))
    print(f"중립 음성: {neutral_count}개")

    for emo_name, _ in EMOTIONS.values():
        if emo_name != "neutral":
            count = len(list((emotions_dir / emo_name).glob('*.wav')))
            print(f"{emo_name.capitalize()} 음성: {count}개")

    print(f"\n✅ 모든 감정이 동일한 인덱스를 사용하여 페어링 가능합니다.")

if __name__ == "__main__":
    prepare_data()
