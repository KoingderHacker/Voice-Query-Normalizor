"""Optional speech-to-text front end for the text-normalization pipeline.

Audio is converted to a raw transcript with faster-whisper. The transcript is then
passed unchanged to the existing Problem 1 pipeline, so every downstream stage
(cleaning, tokenization, stemming, lemmatization) behaves exactly as for typed text.

faster-whisper is imported lazily: the text-only workflow does not need it, and a
missing package or model produces a clear error instead of an import failure.
"""

from __future__ import annotations

import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".aac")
DEFAULT_MODEL_SIZE = "base"
MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")

_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


class TranscriptionUnavailable(RuntimeError):
    """Raised when faster-whisper or the requested speech model cannot be used."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    audio_duration_seconds: float
    transcription_time_seconds: float
    model_size: str


def validate_audio_path(path: str | Path) -> Path:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{audio_path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )
    return audio_path


def _read_wav_samples(path: Path):
    """Decode PCM WAV with the standard library into 16 kHz mono float32 samples.

    This avoids depending on PyAV for the most common case. Returns None for anything
    it cannot decode (non-WAV, float or 24-bit WAV), so the caller falls back to
    faster-whisper's own decoder.
    """
    if path.suffix.lower() != ".wav":
        return None
    import numpy as np

    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError):
        return None
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None
    if channels > 1:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    if rate != 16000 and data.size > 1:
        target = max(1, int(round(data.size * 16000 / rate)))
        data = np.interp(np.linspace(0, data.size - 1, target), np.arange(data.size), data)
    return data.astype(np.float32)


def _load_model(model_size: str, device: str, compute_type: str):
    key = (model_size, device, compute_type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise TranscriptionUnavailable(
            "Audio input needs faster-whisper. Install it with: python -m pip install faster-whisper"
        ) from error
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as error:  # download/network/device errors differ by platform
        raise TranscriptionUnavailable(
            f"Could not load the '{model_size}' speech model ({error}). "
            "The model is downloaded on first use, so an internet connection is required once."
        ) from error
    _MODEL_CACHE[key] = model
    return model


def transcribe_audio(
    path: str | Path,
    *,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str | None = "en",
    device: str = "auto",
    compute_type: str = "int8",
) -> TranscriptionResult:
    """Transcribe one audio file. Pass language=None to auto-detect the language."""
    audio_path = validate_audio_path(path)
    model = _load_model(model_size, device, compute_type)
    samples = _read_wav_samples(audio_path)
    audio_input = samples if samples is not None else str(audio_path)
    started = time.perf_counter()
    try:
        segments, info = model.transcribe(
            audio_input,
            language=language,
            beam_size=5,
            vad_filter=True,  # skip silence so quiet gaps are not hallucinated into words
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except TypeError as error:
        if "metadata_errors" in str(error):
            raise TranscriptionUnavailable(
                "The installed PyAV ('av') package is too old to decode this file format. "
                "Run: python -m pip install --upgrade --force-reinstall \"av>=12\" "
                "(WAV files do not need PyAV)."
            ) from error
        raise
    elapsed = time.perf_counter() - started
    return TranscriptionResult(
        text=text,
        language=info.language,
        language_probability=float(info.language_probability),
        audio_duration_seconds=float(info.duration),
        transcription_time_seconds=elapsed,
        model_size=model_size,
    )


def transcribe_audio_bytes(
    data: bytes | BinaryIO,
    filename: str,
    **kwargs,
) -> TranscriptionResult:
    """Transcribe uploaded or recorded audio held in memory (used by the Streamlit app)."""
    suffix = Path(filename).suffix.lower() or ".wav"
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{suffix}'. Supported: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )
    payload = data if isinstance(data, bytes) else data.read()
    if not payload:
        raise ValueError("Audio input is empty.")
    with tempfile.TemporaryDirectory() as directory:
        temp_path = Path(directory) / f"input{suffix}"
        temp_path.write_bytes(payload)
        return transcribe_audio(temp_path, **kwargs)