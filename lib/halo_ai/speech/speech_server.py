#!/usr/bin/env python3
"""Loopback-oriented Seamless M4T v2 speech translation service."""

from __future__ import annotations

import io
import os
import threading
import time
from typing import Annotated

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from transformers import AutoProcessor, SeamlessM4Tv2Model

from speech_languages import speech_input_output_languages


MODEL_PATH = os.environ.get("SPEECH_MODEL_PATH", "/models/seamless-m4t-v2-large")
MODEL_REVISION = os.environ.get("SPEECH_MODEL_REVISION", "unknown")
PORT = int(os.environ.get("SPEECH_PORT", "7860"))
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 16_000
if not torch.cuda.is_available():
    raise RuntimeError("ROCm device is unavailable through torch.cuda")

DEVICE = torch.device("cuda")
DTYPE = torch.float16
PROCESSOR = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)
MODEL = SeamlessM4Tv2Model.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    use_safetensors=True,
    dtype=DTYPE,
).to(DEVICE)
MODEL.eval()
LANGUAGES = speech_input_output_languages(MODEL.generation_config)
INFERENCE_LOCK = threading.Lock()


def normalize_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim == 2:
        value = value.mean(axis=1)
    value = np.squeeze(value)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("audio must contain at least one mono or stereo sample")
    if sample_rate != INPUT_SAMPLE_RATE:
        tensor = torch.from_numpy(value).unsqueeze(0)
        value = torchaudio.functional.resample(
            tensor,
            orig_freq=sample_rate,
            new_freq=INPUT_SAMPLE_RATE,
        ).squeeze(0).numpy()
    return value


def translate(audio: np.ndarray, sample_rate: int, target_lang: str) -> tuple[np.ndarray, float]:
    if target_lang not in LANGUAGES.values():
        raise ValueError(f"unsupported target language: {target_lang}")
    value = normalize_audio(audio, sample_rate)
    inputs = PROCESSOR(
        audio=value,
        sampling_rate=INPUT_SAMPLE_RATE,
        return_tensors="pt",
    )
    inputs = {
        key: item.to(DEVICE) if isinstance(item, torch.Tensor) else item
        for key, item in inputs.items()
    }
    started = time.monotonic()
    with INFERENCE_LOCK, torch.inference_mode():
        generated = MODEL.generate(**inputs, tgt_lang=target_lang)[0]
    elapsed = time.monotonic() - started
    output = generated.float().cpu().numpy().squeeze()
    if output.ndim != 1 or output.size == 0:
        raise RuntimeError("model returned no audio samples")
    peak = float(np.max(np.abs(output)))
    if peak > 1.0:
        output = output / peak
    return output.astype(np.float32, copy=False), elapsed


app = FastAPI(title="halo-ai Seamless M4T v2", version="1")


@app.get("/healthz")
def health() -> JSONResponse:
    properties = torch.cuda.get_device_properties(0)
    return JSONResponse({
        "status": "ready",
        "model_path": MODEL_PATH,
        "model_revision": MODEL_REVISION,
        "device": properties.name,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "dtype": str(DTYPE),
        "input_sample_rate": INPUT_SAMPLE_RATE,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "target_languages": LANGUAGES,
    })


@app.post("/api/v1/translate")
async def translate_api(
    audio: Annotated[UploadFile, File()],
    target_lang: Annotated[str, Form()] = "eng",
) -> Response:
    try:
        data = await audio.read()
        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        output, elapsed = translate(samples, int(sample_rate), target_lang)
        stream = io.BytesIO()
        sf.write(stream, output, OUTPUT_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    except (RuntimeError, ValueError, sf.LibsndfileError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=stream.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Halo-AI-Inference-Seconds": f"{elapsed:.3f}",
            "X-Halo-AI-Target-Language": target_lang,
        },
    )


def translate_gradio(audio: tuple[int, np.ndarray] | None, target_name: str):
    if audio is None:
        raise gr.Error("Record or upload an audio clip first.")
    sample_rate, samples = audio
    output, elapsed = translate(samples, int(sample_rate), LANGUAGES[target_name])
    return (OUTPUT_SAMPLE_RATE, output), f"Inference: {elapsed:.2f} seconds"


demo = gr.Interface(
    fn=translate_gradio,
    inputs=[
        gr.Audio(type="numpy", label="Source speech"),
        gr.Dropdown(list(LANGUAGES), value="English", label="Target language"),
    ],
    outputs=[
        gr.Audio(type="numpy", label="Translated speech"),
        gr.Textbox(label="Runtime"),
    ],
    title="halo-ai Seamless M4T v2",
    description="Local speech-to-speech translation on AMD ROCm. No public sharing is enabled.",
    api_name=False,
)
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=True)
