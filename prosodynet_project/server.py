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
    neutral_name = f"neutral_{uuid.uuid4().hex}.wav"
    neutral_path = STATIC_DIR / neutral_name
    tts.tts_to_file(text=in_.text, file_path=str(neutral_path))

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
                "ckpt_used": ckpt
            }

    return {
        "neutral_wav": f"/static/{neutral_name}",
        "mel": f"/static/{mel_name}",
        "emotional_wav": f"/static/{out_name}" if result_exists else None,
        "ckpt_used": ckpt
    }
