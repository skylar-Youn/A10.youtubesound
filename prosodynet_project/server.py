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
from functools import lru_cache

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from TTS.api import TTS

try:  # allow running both as package and as a script
    from .prosodynet import ProsodyNet
    from .utils_audio import load_wav, wav_to_mel, extract_f0_pw, extract_energy
except ImportError:  # pragma: no cover - fallback for direct execution
    from prosodynet import ProsodyNet
    from utils_audio import load_wav, wav_to_mel, extract_f0_pw, extract_energy

SR = 22050; HOP = 256; N_MELS = 80
CKPT_SINGLE = "ckpt/prosodynet.pt"
CKPT_MULTI  = "ckpt/prosodynet_multi.pt"  # if trained with multi-emotions

STATIC_DIR = Path(__file__).resolve().parent / "server_static"
STATIC_DIR.mkdir(exist_ok=True)
DEFAULT_RVC_DIR = Path(os.environ.get("PROSODYNET_RVC_DIR", "/home/sk/ws/SD/RVC_TRAIN/Mangio-RVC-v23.7.0/weights")).expanduser()

app = FastAPI(title="ProsodyNet Emotion Server (GL/HiFi-GAN)")
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
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    use_rvc: bool = False
    language: str | None = None
    speaker: str | None = None
    rvc: RVCConfig | None = None
    vocoder: VocoderConfig | None = VocoderConfig()


@lru_cache(maxsize=4)
def load_tts_model(model_name: str) -> TTS:
    try:
        return TTS(model_name)
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"TTS model '{model_name}' is not available in the current Coqui TTS package.",
        ) from exc
    except Exception as exc:  # pragma: no cover - surface cause to client
        raise HTTPException(status_code=500, detail=f"Failed to load TTS model '{model_name}': {exc}") from exc


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
    ckpt = CKPT_MULTI if emo_classes > 1 and os.path.exists(CKPT_MULTI) else CKPT_SINGLE
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    return net, ckpt

@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "UI not found. Upload text via POST /synthesize"}

@app.post("/synthesize")
def synth(in_: SInput):
    # 1) TTS neutral
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

    neutral_name = f"neutral_{uuid.uuid4().hex}.wav"
    neutral_path = STATIC_DIR / neutral_name
    try:
        tts.tts_to_file(text=in_.text, file_path=str(neutral_path), **tts_kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

    # 2) ProsodyNet -> emotional mel
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
    if voc.mode == "hifigan" and voc.generator_module and voc.generator_ckpt and voc.config:
        subprocess.run([
            "python", "vocoder/hifigan_infer.py",
            "--mel", str(mel_path),
            "--out", str(out_path),
            "--mode", "hifigan",
            "--generator_module", voc.generator_module,
            "--generator_ckpt", voc.generator_ckpt,
            "--config", voc.config
        ], check=False)
    else:
        subprocess.run([
            "python", "vocoder/hifigan_infer.py",
            "--mel", str(mel_path),
            "--out", str(out_path),
            "--mode", "griffinlim"
        ], check=False)

    result_exists = out_path.exists()

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
                    "model": in_.tts_model,
                    "language": language,
                    "speaker": speaker,
                },
            }

    return {
        "neutral_wav": f"/static/{neutral_name}",
        "mel": f"/static/{mel_name}",
        "emotional_wav": f"/static/{out_name}" if result_exists else None,
        "ckpt_used": ckpt,
        "tts_used": {
            "model": in_.tts_model,
            "language": language,
            "speaker": speaker,
        },
    }
