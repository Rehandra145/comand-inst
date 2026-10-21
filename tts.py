"""
TTS Module — Sherpa-ONNX Offline Text-to-Speech (Bahasa Indonesia).

Arsitektur sequential & resource-optimized:
  - Hanya satu audio output pada satu waktu (mutual exclusion).
  - Audio Cache (in-memory LRU-like): kalimat umum disimpan di RAM (0 ms latency inference).
  - Background Warmup: engine dimuat di thread latar belakang agar interaksi pertama instan.
  - Terintegrasi dengan ResourceScheduler (settling delay, thread priority, mutual exclusion).

Playback:
  - Windows / dev: sounddevice (cross-platform, blocking)
  - Linux / Raspi : aplay via subprocess (low-latency, direct ALSA)
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np

try:
    from scheduler import scheduler, set_thread_high_priority
except ImportError:
    scheduler = None
    set_thread_high_priority = lambda: None

logger = logging.getLogger(__name__)


# =========================
# CONFIG
# =========================

# Folder model TTS Sherpa-ONNX (Piper VITS Indonesian)
TTS_MODEL_DIR = Path("model") / "tts"

# Speaker ID (0 untuk single-speaker model)
TTS_SPEAKER_ID = 0

# Kecepatan bicara (1.0 = normal)
TTS_SPEED = 1.0

# Jumlah CPU thread untuk TTS inference (2 optimal untuk Pi 4 quad-core)
TTS_NUM_THREADS = 2

# Perangkat output audio Raspberry Pi (aplay):
# "plughw:2,0" = jack audio 3.5mm,  "plughw:0,0" = HDMI
SPEAKER_DEVICE_ALSA = "plughw:2,0"

# Kalimat umum yang di-cache saat warmup agar respons instan (0 ms inference)
PREWARM_PHRASES = (
    "Halo, ada yang bisa saya bantu hari ini?",
    "Ke ruangan berapa?",
    "Ke arah mana?",
    "Siapa yang harus saya ikuti?",
    "Baik, saya berhenti.",
    "Baik, saya kembali.",
    "Baik, saya akan menunggu.",
    "Maaf, saya tidak mendengar perintah. Silakan coba lagi.",
    "Perintah diterima.",
)


# =========================
# LAZY TTS ENGINE & AUDIO CACHE
# =========================

_tts_engine = None
_tts_lock = threading.Lock()
_playback_lock = threading.Lock()

# Cache hasil sintesis: (text_lower, speed, speaker_id) -> (samples, sample_rate)
_audio_cache: Dict[Tuple[str, float, int], Tuple[np.ndarray, int]] = {}
_cache_lock = threading.Lock()
_is_warming_up = False


def _get_tts_engine():
    """Lazy-load TTS engine sekali saat pertama dipakai (thread-safe singleton)."""
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine

    with _tts_lock:
        if _tts_engine is not None:
            return _tts_engine

        try:
            import sherpa_onnx
        except ImportError:
            logger.error("[TTS] sherpa-onnx belum terinstal. Jalankan: pip install sherpa-onnx")
            return None

        model_path = TTS_MODEL_DIR / "id_ID-news_tts-medium.onnx"
        tokens_path = TTS_MODEL_DIR / "tokens.txt"
        data_dir = TTS_MODEL_DIR / "espeak-ng-data"

        if not model_path.exists():
            logger.error(
                f"[TTS] Model tidak ditemukan di '{TTS_MODEL_DIR}'. "
                "Jalankan: python download_tts_model.py"
            )
            return None

        logger.info(f"[TTS] Loading model dari '{TTS_MODEL_DIR}'...")

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_path),
                    tokens=str(tokens_path),
                    data_dir=str(data_dir),
                ),
                num_threads=TTS_NUM_THREADS,
                provider="cpu",
            ),
        )

        if not tts_config.validate():
            logger.error("[TTS] Konfigurasi model TTS tidak valid! Periksa path model.")
            return None

        _tts_engine = sherpa_onnx.OfflineTts(tts_config)
        logger.info("[TTS] Engine berhasil diinisialisasi (offline Piper VITS id_ID)")
        return _tts_engine


# =========================
# WARMUP & PRE-CACHING
# =========================

def warmup() -> bool:
    """
    Inisialisasi model dan pra-sintesis kalimat umum ke memori RAM.
    Menghilangkan latensi awal dan menghemat siklus CPU.
    """
    global _is_warming_up
    if _is_warming_up:
        return True

    _is_warming_up = True
    try:
        engine = _get_tts_engine()
        if engine is None:
            return False

        t0 = time.time()
        cached_count = 0
        for phrase in PREWARM_PHRASES:
            key = (phrase.strip().lower(), round(TTS_SPEED, 2), TTS_SPEAKER_ID)
            with _cache_lock:
                if key in _audio_cache:
                    continue

            audio = engine.generate(phrase.strip(), sid=TTS_SPEAKER_ID, speed=TTS_SPEED)
            samples = np.array(audio.samples, dtype=np.float32)
            with _cache_lock:
                _audio_cache[key] = (samples, audio.sample_rate)
            cached_count += 1

        elapsed = time.time() - t0
        logger.info(f"[TTS] Warmup selesai: {cached_count} kalimat di-cache dalam {elapsed:.2f}s")
        return True
    finally:
        _is_warming_up = False


def warmup_async():
    """Jalankan warmup di background daemon thread."""
    t = threading.Thread(target=warmup, name="TTS-Warmup", daemon=True)
    t.start()


# =========================
# CORE TTS SYNTHESIS
# =========================

def synthesize(text: str, speed: float = TTS_SPEED) -> Optional[Tuple[np.ndarray, int]]:
    """
    Synthesize teks ke audio samples (numpy float32).
    Menggunakan in-memory cache jika teks sudah pernah disintesis sebelumnya (0 ms latency).
    """
    if not text or not text.strip():
        return None

    clean_text = text.strip()
    cache_key = (clean_text.lower(), round(speed, 2), TTS_SPEAKER_ID)

    # Cek in-memory cache
    with _cache_lock:
        if cache_key in _audio_cache:
            samples, sample_rate = _audio_cache[cache_key]
            logger.debug(f"[TTS Cache HIT] \"{clean_text[:40]}\" (0ms)")
            return samples.copy(), sample_rate

    # Generate baru jika belum di cache
    engine = _get_tts_engine()
    if engine is None:
        return None

    t0 = time.time()
    audio = engine.generate(clean_text, sid=TTS_SPEAKER_ID, speed=speed)
    elapsed = time.time() - t0

    samples = np.array(audio.samples, dtype=np.float32)
    sample_rate = audio.sample_rate

    # Simpan ke cache jika cache belum terlalu besar (maks 100 kalimat)
    with _cache_lock:
        if len(_audio_cache) < 100:
            _audio_cache[cache_key] = (samples, sample_rate)

    logger.info(
        f"[TTS] Synthesized \"{clean_text[:45]}...\" → "
        f"{len(samples)/sample_rate:.2f}s audio "
        f"({elapsed*1000:.0f}ms inference)"
    )

    return samples, sample_rate


# =========================
# AUDIO PLAYBACK (SEQUENTIAL)
# =========================

_current_sd_stream = None


def _play_sounddevice(samples: np.ndarray, sample_rate: int):
    """Playback via sounddevice (cross-platform, blocking)."""
    import sounddevice as sd
    sd.play(samples, samplerate=sample_rate)
    sd.wait()


def _play_aplay(samples: np.ndarray, sample_rate: int, device: str = SPEAKER_DEVICE_ALSA):
    """Playback via aplay (Linux ALSA, direct low-latency, blocking)."""
    import subprocess
    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        sf.write(tmp_path, samples, sample_rate, subtype="PCM_16")
        subprocess.run(
            ["aplay", "-D", device, "-q", tmp_path],
            check=False,
            timeout=30,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _is_aplay_available() -> bool:
    """Cek apakah aplay tersedia di sistem."""
    import shutil
    return shutil.which("aplay") is not None


def stop_playback():
    """Hentikan pemutaran audio segera jika sedang berjalan."""
    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass


def play_audio(samples: np.ndarray, sample_rate: int):
    """
    Playback audio sekuensial dengan integrasi ResourceScheduler:
    - Memberi sinyal scheduler.enter_tts()
    - Memutar audio
    - Memberi sinyal scheduler.exit_tts() (termasuk jeda settling 150ms agar tidak ada gema)
    """
    if scheduler:
        scheduler.enter_tts()

    try:
        with _playback_lock:
            if sys.platform == "win32" or not _is_aplay_available():
                _play_sounddevice(samples, sample_rate)
            else:
                _play_aplay(samples, sample_rate)
    finally:
        if scheduler:
            scheduler.exit_tts()


def speak(text: str, speed: float = TTS_SPEED, blocking: bool = True) -> bool:
    """
    Synthesize teks dan putar audio secara sequential.
    
    Args:
        text: Teks yang ingin di-ucapkan.
        speed: Kecepatan bicara (1.0 = normal).
        blocking: True = tunggu hingga audio selesai diputar.
    """
    result = synthesize(text, speed=speed)
    if result is None:
        logger.warning(f"[TTS] Gagal synthesize: \"{text}\"")
        return False

    samples, sample_rate = result

    if blocking:
        play_audio(samples, sample_rate)
    else:
        t = threading.Thread(
            target=play_audio,
            args=(samples, sample_rate),
            name="TTS-Playback",
            daemon=True,
        )
        t.start()

    return True


def speak_and_wait(text: str, speed: float = TTS_SPEED) -> bool:
    """Shortcut: speak dengan blocking (menunggu sampai selesai)."""
    return speak(text, speed=speed, blocking=True)


def is_tts_available() -> bool:
    """Cek apakah TTS engine tersedia dan model terdownload."""
    return _get_tts_engine() is not None
