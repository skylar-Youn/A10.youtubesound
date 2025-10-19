"""
Utility script to synthesize speech with Orpheus-TTS and save it to `output.wav`.

Usage:
    python run_orpheus_tts.py

Requirements:
    pip install orpheus-speech
"""

from __future__ import annotations

import time
import wave

from orpheus_tts import OrpheusModel


def main() -> None:
    prompt = (
        "Man, the way social media has, um, completely changed how we interact is just wild, right? "
        "Like, we're all connected 24/7 but somehow people feel more alone than ever. "
        "And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot."
    )

    model = OrpheusModel(
        model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
        max_model_len=2048,
    )

    start_time = time.monotonic()

    # Stream chunks as they are generated and write to a WAV container.
    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        for chunk_index, audio_chunk in enumerate(
            model.generate_speech(prompt=prompt, voice="tara"),
            start=1,
        ):
            frames = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frames
            wf.writeframes(audio_chunk)
            print(f"[chunk {chunk_index}] wrote {frames} frames")

    duration = total_frames / 24000
    elapsed = time.monotonic() - start_time
    print(f"It took {elapsed:.2f}s to generate {duration:.2f}s of audio → output.wav")


if __name__ == "__main__":
    main()
