# A10.youtubesound
uvicorn server:app --reload --port 7000

source /home/sk/ws/youtubesound/.venv/bin/activate

## Orpheus TTS 참고 메모

- 자주 사용할 모델은 `huggingface-cli download`로 미리 받아 두면 네트워크 없이도 빠르게 로딩할 수 있습니다.
- `Orpheus-TTS` 디렉터리에서 `python run_orpheus_tts.py --help`를 실행하면 제공되는 CLI 옵션을 확인할 수 있습니다.
