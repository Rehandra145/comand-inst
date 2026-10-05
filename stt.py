import wave
from pathlib import Path
from typing import Union
import numpy as np
import sherpa_onnx

from config import MODEL, STT_RATE, NUM_THREADS


# Load model SEKALI saat program mulai
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=f"{MODEL}/tokens.txt",
    encoder=f"{MODEL}/encoder.onnx",
    decoder=f"{MODEL}/decoder.onnx",
    joiner=f"{MODEL}/joiner.onnx",
    num_threads=NUM_THREADS,
    sample_rate=STT_RATE,
    feature_dim=80,
    decoding_method="greedy_search",
    provider="cpu",
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=2.0,
    rule2_min_trailing_silence=0.8,
    rule3_min_utterance_length=20.0,
)


def create_stream():
    """Membuat stream baru untuk real-time streaming audio."""
    return recognizer.create_stream()


def speech_to_text_samples(samples: np.ndarray, sample_rate: int = STT_RATE) -> str:
    """Proses audio langsung dari buffer memory (numpy float32) tanpa write/read file."""
    if samples.ndim > 1:
        samples = samples.squeeze()
    samples = np.ascontiguousarray(samples, dtype=np.float32)

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    return recognizer.get_result(stream).strip()


def speech_to_text(audio_file: Union[str, Path]) -> str:
    """Proses audio dari file WAV."""
    with wave.open(str(audio_file), "rb") as f:
        sample_rate = f.getframerate()
        frames = f.readframes(f.getnframes())

    samples = np.frombuffer(
        frames,
        dtype=np.int16
    ).astype(np.float32) / 32768.0

    return speech_to_text_samples(samples, sample_rate=sample_rate)