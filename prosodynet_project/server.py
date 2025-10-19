import os
import site
import sys
from pathlib import Path


def _activate_local_venv():
    """Ensure the project-local virtualenv is on PATH for subprocesses and imports."""
    if os.environ.get("VIRTUAL_ENV"):
        return

    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".venv"
        if not candidate.exists():
            continue

        bin_dir = candidate / ("Scripts" if os.name == "nt" else "bin")
        if bin_dir.exists():
            current_path = os.environ.get("PATH", "")
            path_parts = current_path.split(os.pathsep) if current_path else []
            if str(bin_dir) not in path_parts:
                os.environ["PATH"] = os.pathsep.join([str(bin_dir), *path_parts]) if path_parts else str(bin_dir)
        os.environ.setdefault("VIRTUAL_ENV", str(candidate))

        if os.name == "nt":
            site_dir = candidate / "Lib" / "site-packages"
        else:
            version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            site_dir = candidate / "lib" / version / "site-packages"
        if site_dir.exists():
            site.addsitedir(str(site_dir))
        break


_activate_local_venv()

def _prepend_env_path(var: str, new_path: Path) -> None:
    """Prepend a path to an environment variable list if it is not already present."""
    new_path = new_path.resolve()
    if not new_path.exists():
        return
    current = os.environ.get(var, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if str(new_path) in parts:
        return
    os.environ[var] = os.pathsep.join([str(new_path), *parts]) if parts else str(new_path)


def _ensure_nv_libraries_resolvable():
    """Make bundled NVIDIA CUDA libs visible to the dynamic loader before importing torch."""
    site_roots = [Path(p) for p in sys.path if "site-packages" in p]
    for root in site_roots:
        nv_root = root / "nvidia"
        if not nv_root.exists():
            continue
        for sub in ("nvjitlink/lib", "cusparse/lib", "cuda_runtime/lib"):
            candidate = nv_root / sub
            _prepend_env_path("LD_LIBRARY_PATH", candidate)


_ensure_nv_libraries_resolvable()

import json
import subprocess
import uuid
import wave
import platform
from multiprocessing import cpu_count
from functools import lru_cache

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    TTS = None

try:
    from orpheus_tts import OrpheusModel
    ORPHEUS_AVAILABLE = True
except ImportError:
    ORPHEUS_AVAILABLE = False
    OrpheusModel = None

try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    KPipeline = None

try:
    import edge_tts
    import asyncio
    EDGE_AVAILABLE = True
except ImportError:
    EDGE_AVAILABLE = False
    edge_tts = None

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    gTTS = None

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None

try:
    from bark import SAMPLE_RATE as BARK_SAMPLE_RATE, generate_audio
    from scipy.io import wavfile
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    generate_audio = None
    BARK_SAMPLE_RATE = None

try:
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
    from boson_multimodal.data_types import ChatMLSample, Message
    import torchaudio
    HIGGS_AVAILABLE = True
    # Higgs Audio V2 모델 경로 (HuggingFace repo ID 사용 - 자동으로 캐시에서 로드)
    HIGGS_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
    HIGGS_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
except ImportError:
    HIGGS_AVAILABLE = False
    HiggsAudioServeEngine = None
    HiggsAudioResponse = None
    ChatMLSample = None
    Message = None
    torchaudio = None
    HIGGS_MODEL_PATH = None
    HIGGS_TOKENIZER_PATH = None

# Fish Speech - HTTP API 방식 (별도 서버 필요)
FISH_SPEECH_API_URL = "http://localhost:8001"  # Fish Speech API 서버 주소
FISH_SPEECH_AVAILABLE = True  # Fish Speech API 서버가 실행 중이면 True

try:  # allow running both as package and as a script
    from .prosodynet import ProsodyNet
    from .utils_audio import load_wav, wav_to_mel, extract_f0_pw, extract_energy
except ImportError:  # pragma: no cover - fallback for direct execution
    from prosodynet import ProsodyNet
    from utils_audio import load_wav, wav_to_mel, extract_f0_pw, extract_energy

SR = 22050; HOP = 256; N_MELS = 80
PROJECT_DIR = Path(__file__).resolve().parent
CKPT_SINGLE = PROJECT_DIR / "ckpt" / "prosodynet.pt"
CKPT_MULTI  = PROJECT_DIR / "ckpt" / "prosodynet_multi.pt"  # if trained with multi-emotions

STATIC_DIR = PROJECT_DIR / "server_static"
STATIC_DIR.mkdir(exist_ok=True)
DEFAULT_RVC_DIR = Path(os.environ.get("PROSODYNET_RVC_DIR", "/home/sk/ws/SD/RVC_TRAIN/Mangio-RVC-v23.7.0/weights")).expanduser()

app = FastAPI(title="ProsodyNet Emotion Server (GL/HiFi-GAN)")

@app.get("/static/list")
def list_static_files():
    """서버 static 폴더의 파일 목록 반환"""
    try:
        files = []
        if STATIC_DIR.exists():
            for file_path in STATIC_DIR.iterdir():
                if file_path.is_file() and not file_path.name.startswith('.'):
                    files.append(file_path.name)
        files.sort(reverse=True)  # 최신 파일 먼저
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 목록 조회 실패: {e}") from e

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class RVCConfig(BaseModel):
    pth: str | None = None
    config: str | None = None
    cli: str | None = None  # e.g., "python infer-web.py"

class VocoderConfig(BaseModel):
    mode: str = "griffinlim"  # "griffinlim" or "hifigan"
    generator_module: str | None = None
    generator_ckpt: str | None = None
    config: str | None = None

class SInput(BaseModel):
    text: str
    emotion_id: int = 0
    tts_engine: str = "coqui"  # "coqui", "orpheus", "kokoro", "edge", "gtts", "pyttsx3", "bark", "higgs", or "fish"
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    use_prosodynet: bool = True  # Apply emotion conversion with ProsodyNet
    use_rvc: bool = False
    language: str | None = None
    speaker: str | None = None
    orpheus_voice: str = "tara"  # for Orpheus: tara, leah, jess, leo, dan, mia, zac, zoe
    orpheus_model: str = "canopylabs/orpheus-tts-0.1-finetune-prod"  # or "canopylabs/3b-ko-ft-research_release" for Korean
    kokoro_lang: str = "j"  # for Kokoro: a=American English, b=British English, j=Japanese, z=Chinese, e=Spanish, f=French, h=Hindi, i=Italian, p=Portuguese
    kokoro_voice: str = "jf_alpha"  # for Kokoro Japanese: jf_alpha, jm_kumo (more voices available for other languages)
    edge_voice: str = "ko-KR-SunHiNeural"  # for Edge TTS: ko-KR-SunHiNeural, en-US-JennyNeural, ja-JP-NanamiNeural, etc.
    edge_rate: str = "+0%"  # Speech rate: -50% to +100%
    edge_pitch: str = "+0Hz"  # Pitch: -50Hz to +50Hz
    gtts_lang: str = "ko"  # for gTTS: ko, en, ja, etc. (100+ languages)
    gtts_tld: str = "com"  # for gTTS: com, co.uk, com.au, co.in, ca, etc.
    pyttsx3_voice_id: str | None = None  # for pyttsx3: voice ID (platform-specific)
    bark_voice: str = "v2/en_speaker_6"  # for Bark: voice preset (v2/en_speaker_0-9, v2/zh_speaker_0-9, etc.)
    higgs_voice: str | None = None  # for Higgs Audio V2: reference audio file path (optional, auto voice if None)
    higgs_temperature: float = 0.3  # for Higgs Audio V2: temperature (0.1-1.0, lower=more consistent)
    higgs_top_p: float = 0.95  # for Higgs Audio V2: top_p sampling (0.0-1.0)
    higgs_max_tokens: int = 1024  # for Higgs Audio V2: max audio tokens to generate
    fish_temperature: float = 0.7  # for Fish Speech: temperature (0.1-1.0)
    fish_top_p: float = 0.7  # for Fish Speech: top_p sampling (0.0-1.0)
    fish_max_tokens: int = 1024  # for Fish Speech: max tokens to generate
    fish_repetition_penalty: float = 1.2  # for Fish Speech: repetition penalty (1.0-2.0)
    rvc: RVCConfig | None = None
    vocoder: VocoderConfig | None = VocoderConfig()


@lru_cache(maxsize=4)
def load_tts_model(model_name: str) -> TTS:
    if not COQUI_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Coqui TTS is not available. There may be a library conflict. Try using Orpheus TTS instead."
        )
    try:
        return TTS(model_name)
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"TTS model '{model_name}' is not available in the current Coqui TTS package.",
        ) from exc
    except Exception as exc:  # pragma: no cover - surface cause to client
        raise HTTPException(status_code=500, detail=f"Failed to load TTS model '{model_name}': {exc}") from exc


@lru_cache(maxsize=2)
def load_orpheus_model(model_name: str = "canopylabs/orpheus-tts-0.1-finetune-prod") -> OrpheusModel:
    if not ORPHEUS_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Orpheus TTS is not installed. Run: pip install orpheus-speech"
        )
    try:
        return OrpheusModel(model_name=model_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load Orpheus model '{model_name}': {exc}") from exc


@lru_cache(maxsize=4)
def load_kokoro_pipeline(lang_code: str = 'j') -> KPipeline:
    """Load Kokoro TTS pipeline for specified language.

    Supported language codes:
    - 'a': American English
    - 'b': British English
    - 'j': Japanese
    - 'z': Mandarin Chinese
    - 'e': Spanish
    - 'f': French
    - 'h': Hindi
    - 'i': Italian
    - 'p': Portuguese
    """
    if not KOKORO_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Kokoro TTS is not installed. Run: pip install kokoro pyopenjtalk fugashi[unidic-lite] jaconv mojimoji cutlet"
        )
    try:
        return KPipeline(lang_code=lang_code)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load Kokoro pipeline for '{lang_code}': {exc}") from exc


def _resolve_rvc_base(base_path: str | None) -> Path:
    if base_path:
        candidate = Path(base_path).expanduser()
    else:
        candidate = DEFAULT_RVC_DIR
    if not candidate:
        raise HTTPException(status_code=400, detail="RVC base path is not configured.")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"RVC base path '{candidate}' does not exist.")
    if not candidate.is_dir():
        raise HTTPException(status_code=400, detail=f"RVC base path '{candidate}' is not a directory.")
    return candidate.resolve()


def _collect_files(base: Path, patterns: list[str]) -> list[str]:
    seen: set[str] = set()
    for pattern in patterns:
        for file in base.rglob(pattern):
            if file.is_file():
                rel = file.relative_to(base).as_posix()
                seen.add(rel)
    return sorted(seen)


@app.get("/rvc/files")
def list_rvc_files(base_path: str | None = Query(None, alias="basePath")):
    base = _resolve_rvc_base(base_path)
    return {
        "base": str(base),
        "pth": _collect_files(base, ["*.pth"]),
        "config": _collect_files(base, ["*.json", "*.yaml", "*.yml"]),
    }


def _flatten_iterable(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _pick_language(tts: TTS, requested: str | None) -> str | None:
    if requested:
        return requested
    if not getattr(tts, "is_multi_lingual", False):
        return None
    candidates: list[str] = []
    try:
        candidates = _flatten_iterable(tts.languages)  # type: ignore[arg-type]
    except Exception:
        candidates = []
    if not candidates:
        synth = getattr(tts, "synthesizer", None)
        if synth is not None:
            try:
                candidates = _flatten_iterable(getattr(synth, "tts_languages", None))
            except Exception:
                candidates = []
    if candidates:
        return candidates[0]
    return "en"


def _load_cached_speakers(tts: TTS) -> list[str]:
    base_dir = getattr(tts.manager, "output_prefix", None)
    if not base_dir:
        return []
    cache_dir = Path(base_dir) / tts.model_name.replace("/", "--")
    if not cache_dir.exists():
        return []
    for name in ("speakers.pth", "speakers_xtts.pth", "speakers.json"):
        candidate = cache_dir / name
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".json":
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            else:
                payload = torch.load(candidate, map_location="cpu")
        except Exception:
            continue
        speakers = _flatten_iterable(payload)
        if speakers:
            return speakers
    return []


def _pick_speaker(tts: TTS, requested: str | None) -> str | None:
    if requested:
        return requested
    if not getattr(tts, "is_multi_speaker", False):
        return None
    candidates: list[str] = []
    try:
        candidates = _flatten_iterable(tts.speakers)  # type: ignore[attr-defined]
    except Exception:
        candidates = []
    if not candidates:
        synth = getattr(tts, "synthesizer", None)
        if synth is not None:
            speaker_manager = getattr(synth, "speaker_manager", None)
            if speaker_manager is not None and hasattr(speaker_manager, "speakers"):
                try:
                    candidates = _flatten_iterable(speaker_manager.speakers)
                except Exception:
                    candidates = []
            if not candidates:
                candidates = _flatten_iterable(getattr(synth, "tts_speakers", None))
    if not candidates:
        candidates = _load_cached_speakers(tts)
    if candidates:
        return candidates[0]
    raise HTTPException(
        status_code=400,
        detail=f"TTS model '{tts.model_name}' requires a speaker id; include `speaker` in the request.",
    )

def load_net(emo_classes):
    net = ProsodyNet(n_mels=N_MELS, emo_classes=emo_classes)
    ckpt = CKPT_MULTI if emo_classes > 1 and CKPT_MULTI.exists() else CKPT_SINGLE
    if not ckpt.exists():
        raise HTTPException(
            status_code=500,
            detail=f"ProsodyNet checkpoint not found at '{ckpt}'. Please train the model first."
        )
    net.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    net.eval()
    return net, str(ckpt)

@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "UI not found. Upload text via POST /synthesize"}

@app.post("/synthesize")
def synth(in_: SInput):
    # 1) TTS neutral - choose engine
    neutral_name = f"neutral_{uuid.uuid4().hex}.wav"
    neutral_path = STATIC_DIR / neutral_name
    speaker = None
    language = None

    if in_.tts_engine == "orpheus":
        # Use Orpheus TTS
        model = load_orpheus_model(in_.orpheus_model)
        try:
            with wave.open(str(neutral_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                # Use voice parameter only if provided (for non-English models, voice might not be needed)
                kwargs = {"prompt": in_.text}
                if in_.orpheus_voice:
                    kwargs["voice"] = in_.orpheus_voice
                for audio_chunk in model.generate_speech(**kwargs):
                    wf.writeframes(audio_chunk)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Orpheus TTS synthesis failed: {exc}") from exc

    elif in_.tts_engine == "kokoro":
        # Use Kokoro TTS
        pipeline = load_kokoro_pipeline(in_.kokoro_lang)
        try:
            # Generate audio using Kokoro pipeline
            # The pipeline returns a generator of (gs, ps, audio) tuples
            generator = pipeline(in_.text, voice=in_.kokoro_voice)
            audio_chunks = []
            for gs, ps, audio in generator:
                # audio is a torch tensor, convert to numpy
                if hasattr(audio, 'cpu'):
                    audio_np = audio.cpu().numpy()
                else:
                    audio_np = np.array(audio)
                audio_chunks.append(audio_np)

            # Concatenate all audio chunks
            full_audio = np.concatenate(audio_chunks)

            # Kokoro outputs 24kHz float32 audio, convert to int16 for WAV
            audio_int16 = (full_audio * 32767).astype(np.int16)

            # Write to WAV file
            with wave.open(str(neutral_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_int16.tobytes())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Kokoro TTS synthesis failed: {exc}") from exc

    elif in_.tts_engine == "edge":
        # Use Edge TTS
        if not EDGE_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="Edge TTS is not installed. Run: pip install edge-tts"
            )
        try:
            # Edge TTS requires async, so we use asyncio.run
            async def generate_edge_tts():
                communicate = edge_tts.Communicate(
                    text=in_.text,
                    voice=in_.edge_voice,
                    rate=in_.edge_rate,
                    pitch=in_.edge_pitch
                )
                await communicate.save(str(neutral_path))

            asyncio.run(generate_edge_tts())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Edge TTS synthesis failed: {exc}") from exc

    elif in_.tts_engine == "gtts":
        # Use gTTS (Google Text-to-Speech)
        if not GTTS_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="gTTS is not installed. Run: pip install gtts"
            )
        try:
            tts = gTTS(text=in_.text, lang=in_.gtts_lang, tld=in_.gtts_tld)
            tts.save(str(neutral_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"gTTS synthesis failed: {exc}") from exc

    elif in_.tts_engine == "pyttsx3":
        # Use pyttsx3 (offline TTS)
        if not PYTTSX3_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="pyttsx3 is not installed. Run: pip install pyttsx3"
            )
        try:
            engine = pyttsx3.init()
            if in_.pyttsx3_voice_id:
                engine.setProperty('voice', in_.pyttsx3_voice_id)
            engine.save_to_file(in_.text, str(neutral_path))
            engine.runAndWait()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"pyttsx3 synthesis failed: {exc}") from exc

    elif in_.tts_engine == "bark":
        # Use Bark TTS
        if not BARK_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="Bark TTS is not installed. Run: pip install git+https://github.com/suno-ai/bark.git"
            )
        try:
            # Generate audio with Bark
            audio_array = generate_audio(in_.text, history_prompt=in_.bark_voice)
            # Bark outputs at 24kHz, need to convert to 22050Hz for ProsodyNet compatibility
            import librosa
            audio_resampled = librosa.resample(audio_array, orig_sr=BARK_SAMPLE_RATE, target_sr=SR)
            # Save as wav file
            wavfile.write(str(neutral_path), SR, (audio_resampled * 32767).astype(np.int16))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Bark TTS synthesis failed: {exc}") from exc

    elif in_.tts_engine == "higgs":
        # Use Higgs Audio V2
        if not HIGGS_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="Higgs Audio V2 is not installed. Run: pip install -e /mnt/sdb1/ws-sky-data/youtubesound-data/higgs_audio_v2/higgs-audio"
            )
        try:
            # Initialize Higgs Audio engine
            device = "cuda" if torch.cuda.is_available() else "cpu"
            serve_engine = HiggsAudioServeEngine(HIGGS_MODEL_PATH, HIGGS_TOKENIZER_PATH, device=device)

            # Prepare system prompt
            system_prompt = (
                "Generate audio following instruction.\n\n<|scene_desc_start|>\n"
                "Audio is recorded from a quiet room.\n<|scene_desc_end|>"
            )

            # Prepare messages
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=in_.text),
            ]

            # Generate audio
            output: HiggsAudioResponse = serve_engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=in_.higgs_max_tokens,
                temperature=in_.higgs_temperature,
                top_p=in_.higgs_top_p,
                top_k=50,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )

            # Higgs outputs at 24kHz, need to convert to 22050Hz for ProsodyNet compatibility
            import librosa
            audio_resampled = librosa.resample(output.audio, orig_sr=output.sampling_rate, target_sr=SR)
            # Save as wav file
            wavfile.write(str(neutral_path), SR, (audio_resampled * 32767).astype(np.int16))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Higgs Audio V2 synthesis failed: {exc}") from exc

    elif in_.tts_engine == "fish":
        # Use Fish Speech (HTTP API)
        if not FISH_SPEECH_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="Fish Speech API is not available. Please start the Fish Speech API server on port 8001."
            )
        try:
            import requests
            # Fish Speech API 호출
            api_url = f"{FISH_SPEECH_API_URL}/v1/tts"
            payload = {
                "text": in_.text,
                "format": "wav",
                "streaming": False,
                "max_new_tokens": in_.fish_max_tokens,
                "chunk_length": 200,
                "top_p": in_.fish_top_p,
                "repetition_penalty": in_.fish_repetition_penalty,
                "temperature": in_.fish_temperature,
            }

            response = requests.post(api_url, json=payload, timeout=60)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Fish Speech API error: {response.status_code} - {response.text}"
                )

            # WAV 데이터를 파일로 저장
            with open(neutral_path, "wb") as f:
                f.write(response.content)

            # Fish Speech는 일반적으로 44.1kHz로 출력, 22050Hz로 변환
            import librosa
            audio, sr_orig = librosa.load(str(neutral_path), sr=None)
            if sr_orig != SR:
                audio_resampled = librosa.resample(audio, orig_sr=sr_orig, target_sr=SR)
                wavfile.write(str(neutral_path), SR, (audio_resampled * 32767).astype(np.int16))

        except requests.exceptions.RequestException as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Fish Speech API connection failed: {exc}. Make sure Fish Speech API server is running on {FISH_SPEECH_API_URL}"
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Fish Speech synthesis failed: {exc}") from exc

    else:
        # Use Coqui TTS (default)
        tts = load_tts_model(in_.tts_model)
        try:
            speaker = _pick_speaker(tts, in_.speaker)
            language = _pick_language(tts, in_.language)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - unexpected TTS metadata failure
            raise HTTPException(status_code=500, detail=f"Failed to prepare TTS arguments: {exc}") from exc

        tts_kwargs: dict[str, str] = {}
        if speaker is not None:
            tts_kwargs["speaker"] = speaker
        if language is not None:
            tts_kwargs["language"] = language

        try:
            tts.tts_to_file(text=in_.text, file_path=str(neutral_path), **tts_kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

    # 2) ProsodyNet -> emotional mel (optional)
    if in_.use_prosodynet:
        wav, _ = load_wav(str(neutral_path), SR)
        nmel = wav_to_mel(wav, SR, N_MELS, 1024, HOP, 1024)
        nf0  = extract_f0_pw(wav, SR, HOP)
        nene = extract_energy(nmel)

        emo_classes = 4  # set your trained number (e.g., happy/sad/angry/neutral); adjust as needed
        net, ckpt = load_net(emo_classes)

        with torch.no_grad():
            nmel_t = torch.from_numpy(nmel).unsqueeze(0).float()
            nf0_t  = torch.from_numpy(nf0).unsqueeze(0).float()
            nene_t = torch.from_numpy(nene).unsqueeze(0).float()
            dmel_t = net(nmel_t, nf0_t, nene_t, emo_id=in_.emotion_id)
            emel_t = nmel_t + dmel_t
        emel = emel_t.squeeze(0).cpu().numpy()

        mel_name = f"emel_{uuid.uuid4().hex}.npy"
        mel_path = STATIC_DIR / mel_name
        np.save(str(mel_path), emel)

        # 3) Vocoder
        out_name = f"emotional_{uuid.uuid4().hex}.wav"
        out_path = STATIC_DIR / out_name
        voc = in_.vocoder or VocoderConfig()
        vocoder_script = PROJECT_DIR / "vocoder" / "hifigan_infer.py"
        if voc.mode == "hifigan" and voc.generator_module and voc.generator_ckpt and voc.config:
            subprocess.run([
                sys.executable, str(vocoder_script),
                "--mel", str(mel_path),
                "--out", str(out_path),
                "--mode", "hifigan",
                "--generator_module", voc.generator_module,
                "--generator_ckpt", voc.generator_ckpt,
                "--config", voc.config
            ], check=False)
        else:
            subprocess.run([
                sys.executable, str(vocoder_script),
                "--mel", str(mel_path),
                "--out", str(out_path),
                "--mode", "griffinlim"
            ], check=False)

        result_exists = out_path.exists()
    else:
        # Skip ProsodyNet - use neutral audio as final output
        mel_name = None
        mel_path = None
        ckpt = "N/A (ProsodyNet disabled)"
        out_name = neutral_name  # Use neutral audio directly
        out_path = neutral_path
        result_exists = True

    # 4) Optional RVC
    if in_.use_rvc and in_.rvc and in_.rvc.cli and result_exists:
        rvc_name = f"rvc_{uuid.uuid4().hex}.wav"
        rvc_path = STATIC_DIR / rvc_name
        subprocess.run([
            *in_.rvc.cli.split(),
            "--input_path", str(out_path),
            "--output_path", str(rvc_path),
            "--pth_path", in_.rvc.pth,
            "--config_path", in_.rvc.config,
            "--f0method", "harvest",
            "--pitch_shift", "0"
        ], check=False)
        if rvc_path.exists():
            return {
                "neutral_wav": f"/static/{neutral_name}",
                "mel": f"/static/{mel_name}",
                "emotional_wav": f"/static/{rvc_name}",
                "ckpt_used": ckpt,
                "tts_used": {
                    "engine": in_.tts_engine,
                    "model": (
                        in_.tts_model if in_.tts_engine == "coqui" else
                        in_.orpheus_model if in_.tts_engine == "orpheus" else
                        "hexgrad/Kokoro-82M" if in_.tts_engine == "kokoro" else
                        "Microsoft Edge TTS" if in_.tts_engine == "edge" else
                        f"Google TTS ({in_.gtts_lang})" if in_.tts_engine == "gtts" else
                        "pyttsx3 (Offline TTS)" if in_.tts_engine == "pyttsx3" else
                        f"Bark TTS ({in_.bark_voice})" if in_.tts_engine == "bark" else
                        "Higgs Audio V2 (BosonAI 3B)" if in_.tts_engine == "higgs" else
                        "Fish Speech 1.5"
                    ),
                    "language": (
                        language if in_.tts_engine == "coqui" else
                        in_.kokoro_lang if in_.tts_engine == "kokoro" else
                        in_.gtts_lang if in_.tts_engine == "gtts" else
                        None
                    ),
                    "speaker": speaker,
                    "orpheus_voice": in_.orpheus_voice if in_.tts_engine == "orpheus" else None,
                    "kokoro_voice": in_.kokoro_voice if in_.tts_engine == "kokoro" else None,
                    "edge_voice": in_.edge_voice if in_.tts_engine == "edge" else None,
                    "gtts_lang": in_.gtts_lang if in_.tts_engine == "gtts" else None,
                    "gtts_tld": in_.gtts_tld if in_.tts_engine == "gtts" else None,
                    "pyttsx3_voice_id": in_.pyttsx3_voice_id if in_.tts_engine == "pyttsx3" else None,
                    "bark_voice": in_.bark_voice if in_.tts_engine == "bark" else None,
                    "higgs_temperature": in_.higgs_temperature if in_.tts_engine == "higgs" else None,
                    "higgs_top_p": in_.higgs_top_p if in_.tts_engine == "higgs" else None,
                    "fish_temperature": in_.fish_temperature if in_.tts_engine == "fish" else None,
                    "fish_top_p": in_.fish_top_p if in_.tts_engine == "fish" else None,
                    "fish_max_tokens": in_.fish_max_tokens if in_.tts_engine == "fish" else None,
                    "fish_repetition_penalty": in_.fish_repetition_penalty if in_.tts_engine == "fish" else None,
                },
            }

    return {
        "neutral_wav": f"/static/{neutral_name}",
        "mel": f"/static/{mel_name}" if mel_name else None,
        "emotional_wav": f"/static/{out_name}" if result_exists else None,
        "ckpt_used": ckpt,
        "prosodynet_enabled": in_.use_prosodynet,
        "tts_used": {
            "engine": in_.tts_engine,
            "model": (
                in_.tts_model if in_.tts_engine == "coqui" else
                in_.orpheus_model if in_.tts_engine == "orpheus" else
                "hexgrad/Kokoro-82M" if in_.tts_engine == "kokoro" else
                "Microsoft Edge TTS" if in_.tts_engine == "edge" else
                f"Google TTS ({in_.gtts_lang})" if in_.tts_engine == "gtts" else
                "pyttsx3 (Offline TTS)" if in_.tts_engine == "pyttsx3" else
                f"Bark TTS ({in_.bark_voice})" if in_.tts_engine == "bark" else
                "Higgs Audio V2 (BosonAI 3B)" if in_.tts_engine == "higgs" else
                "Fish Speech 1.5"
            ),
            "language": (
                language if in_.tts_engine == "coqui" else
                in_.kokoro_lang if in_.tts_engine == "kokoro" else
                in_.gtts_lang if in_.tts_engine == "gtts" else
                None
            ),
            "speaker": speaker,
            "orpheus_voice": in_.orpheus_voice if in_.tts_engine == "orpheus" else None,
            "kokoro_voice": in_.kokoro_voice if in_.tts_engine == "kokoro" else None,
            "edge_voice": in_.edge_voice if in_.tts_engine == "edge" else None,
            "gtts_lang": in_.gtts_lang if in_.tts_engine == "gtts" else None,
            "gtts_tld": in_.gtts_tld if in_.tts_engine == "gtts" else None,
            "pyttsx3_voice_id": in_.pyttsx3_voice_id if in_.tts_engine == "pyttsx3" else None,
            "bark_voice": in_.bark_voice if in_.tts_engine == "bark" else None,
            "higgs_temperature": in_.higgs_temperature if in_.tts_engine == "higgs" else None,
            "higgs_top_p": in_.higgs_top_p if in_.tts_engine == "higgs" else None,
            "fish_temperature": in_.fish_temperature if in_.tts_engine == "fish" else None,
            "fish_top_p": in_.fish_top_p if in_.tts_engine == "fish" else None,
            "fish_max_tokens": in_.fish_max_tokens if in_.tts_engine == "fish" else None,
            "fish_repetition_penalty": in_.fish_repetition_penalty if in_.tts_engine == "fish" else None,
        },
    }

@app.get("/system/info")
def system_info():
    """시스템 정보 반환 (CPU 코어 수, 플랫폼 등)"""
    try:
        cpu_cores = cpu_count()
        cpu_model = None

        # Linux: /proc/cpuinfo에서 CPU 모델 읽기
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_model = line.split(":", 1)[1].strip()
                            break
            except:
                pass

        if not cpu_model:
            cpu_model = platform.processor() or "Unknown"

        return {
            "cpu": {
                "cores": cpu_cores,
                "model": cpu_model
            },
            "platform": platform.system(),
            "python_version": platform.python_version()
        }
    except Exception as e:
        return {
            "cpu": {
                "cores": cpu_count(),
                "model": "Unknown"
            },
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "error": str(e)
        }
