"""
STT Module — Sherpa-ONNX streaming Zipformer RNN-T (Bahasa Indonesia).
"""

import logging
import os
import threading
import time
import wave
from pathlib import Path
from typing import Optional, Union
import numpy as np
import sherpa_onnx

from config import MODEL, STT_RATE, NUM_THREADS

logger = logging.getLogger(__name__)

# Base root directory resolver
_ROOT_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "audio" else Path(__file__).resolve().parent
_MODEL_DIR = _ROOT_DIR / MODEL if not Path(MODEL).is_absolute() else Path(MODEL)


def _pick(name: str, int8_name: str) -> str:
    """Pilih file INT8 jika tersedia, fallback ke file biasa."""
    int8_path = _MODEL_DIR / int8_name
    normal_path = _MODEL_DIR / name
    if int8_path.exists():
        logger.info(f"[STT] Menggunakan model quantized: {int8_name}")
        return str(int8_path)
    return str(normal_path)


# Otomatis pilih encoder/joiner INT8 jika ada (hasil dari quantize_model.py)
_encoder = _pick("encoder.onnx", "encoder.int8.onnx")
_joiner = _pick("joiner.onnx", "joiner.int8.onnx")
_decoder = str(_MODEL_DIR / "decoder.onnx")
_tokens = str(_MODEL_DIR / "tokens.txt")


# =========================
# PRE-OPTIMIZED ENCODER CACHE
# =========================
def _optimized_path(src: Path) -> Path:
    return src.with_name(src.stem + ".opt.onnx")


def _ensure_optimized_encoder(src: Path) -> Optional[Path]:
    """Kembalikan path encoder teroptimasi (buat jika belum ada / kadaluarsa)."""
    # Model INT8 sudah terkuantisasi dan memakai ConvInteger yang tidak didukung
    # graph optimization ONNX Runtime CPU. Langsung gunakan model INT8 apa adanya.
    if ".int8." in src.name:
        return src

    opt = _optimized_path(src)
    try:
        if opt.exists() and opt.stat().st_mtime >= src.stat().st_mtime:
            return opt
    except OSError:
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        return None

    tmp = opt.with_name(opt.stem + ".tmp.onnx")
    try:
        t0 = time.time()
        logger.info(f"[STT] Membuat cache encoder teroptimasi (sekali saja): {opt.name} ...")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = str(tmp)
        ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
        os.replace(tmp, opt)
        logger.info(f"[STT] Cache encoder siap dalam {time.time() - t0:.1f}s")
        return opt
    except Exception as e:
        logger.warning(f"[STT] Gagal membuat encoder teroptimasi, pakai model asli: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return None


def _build_recognizer(encoder: str):
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=_tokens,
        encoder=encoder,
        decoder=_decoder,
        joiner=_joiner,
        num_threads=NUM_THREADS,
        sample_rate=STT_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cuda",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.8,
        rule2_min_trailing_silence=2.2,
        rule3_min_utterance_length=30.0,
    )


# =========================
# LAZY + BACKGROUND LOADING
# =========================
_recognizer = None
_load_lock = threading.Lock()
_preload_started = False


def get_recognizer():
    """Ambil recognizer; load sekarang (blocking) jika belum siap."""
    global _recognizer
    if _recognizer is not None:
        return _recognizer
    with _load_lock:
        if _recognizer is None:
            from audio.scheduler import model_init_lock
            with model_init_lock:
                t0 = time.time()
                src = Path(_encoder)
                opt = _ensure_optimized_encoder(src)
                rec = None
                if opt is not None:
                    try:
                        rec = _build_recognizer(str(opt))
                    except Exception as e:
                        logger.warning(f"[STT] Encoder teroptimasi gagal dimuat ({e}), pakai model asli")
                if rec is None:
                    rec = _build_recognizer(str(src))
                _recognizer = rec
                logger.info(f"[STT] Model siap dalam {time.time() - t0:.1f}s")
    return _recognizer


def is_ready() -> bool:
    """True jika model STT sudah selesai dimuat."""
    return _recognizer is not None


def preload_async():
    """Mulai load model STT di background thread (aman dipanggil berkali-kali)."""
    global _preload_started
    if _recognizer is not None or _preload_started:
        return
    _preload_started = True

    def _run():
        try:
            get_recognizer()
        except Exception as e:
            logger.error(f"[STT] Preload model gagal: {e}")

    threading.Thread(target=_run, name="STT-Preload", daemon=True).start()


class _LazyRecognizer:
    """Proxy agar `from stt import recognizer` tetap bekerja tanpa memblokir saat import."""

    def __getattr__(self, name):
        return getattr(get_recognizer(), name)


recognizer = _LazyRecognizer()


def create_stream():
    """Membuat stream baru untuk real-time streaming audio."""
    return get_recognizer().create_stream()


def speech_to_text_samples(samples: np.ndarray, sample_rate: int = STT_RATE) -> str:
    """Proses audio langsung dari buffer memory (numpy float32) tanpa write/read file."""
    if samples.ndim > 1:
        samples = samples.squeeze()
    samples = np.ascontiguousarray(samples, dtype=np.float32)

    rec = get_recognizer()
    stream = rec.create_stream()
    stream.accept_waveform(sample_rate, samples)

    while rec.is_ready(stream):
        rec.decode_stream(stream)

    return rec.get_result(stream).strip()


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
