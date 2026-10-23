"""
TTS Module — Sherpa-ONNX Offline Text-to-Speech (Bahasa Indonesia).

Arsitektur sequential & resource-optimized:
  - Hanya satu audio output pada satu waktu (mutual exclusion).
  - Audio Cache 2 level: RAM (instan) + disk `.tts_cache/` (persisten antar restart).
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
    from audio.scheduler import scheduler, set_thread_high_priority
except ImportError:
    scheduler = None
    set_thread_high_priority = lambda: None

logger = logging.getLogger(__name__)

# Base root directory resolver
_ROOT_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "audio" else Path(__file__).resolve().parent

# =========================
# CONFIG
# =========================

# Folder model TTS Sherpa-ONNX (Piper VITS Indonesian)
TTS_MODEL_DIR = _ROOT_DIR / "model" / "tts"

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
try:
    from responses import (
        GREETINGS as _GREETINGS,
        ASK_LOCATION as _ASK_LOCATION,
        UNKNOWN_INTENT as _UNKNOWN_INTENT,
    )
except ImportError:
    _GREETINGS, _ASK_LOCATION, _UNKNOWN_INTENT = (), (), ()

PREWARM_PHRASES = (
    *_GREETINGS,
    *_ASK_LOCATION,
    *_UNKNOWN_INTENT,
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

# Batas jumlah entri di cache RAM (cache disk tidak dibatasi)
MEMORY_CACHE_MAX = 200


# =========================
# PERSISTENT DISK CACHE
# =========================
# Hasil sintesis disimpan sebagai file WAV di disk, sehingga setelah restart
# program kalimat yang pernah diucapkan tidak perlu disintesis ulang.

TTS_MODEL_FILE = "id_ID-news_tts-medium.onnx"
TTS_DISK_CACHE_DIR = _ROOT_DIR / ".tts_cache"


def _disk_cache_path(key: Tuple[str, float, int]) -> Path:
    """Path file WAV cache untuk sebuah key (hash dari teks + speed + speaker + model)."""
    import hashlib
    raw = f"{TTS_MODEL_FILE}|{key[0]}|{key[1]}|{key[2]}".encode("utf-8")
    return TTS_DISK_CACHE_DIR / f"{hashlib.sha1(raw).hexdigest()}.wav"


def _disk_cache_load(key: Tuple[str, float, int]) -> Optional[Tuple[np.ndarray, int]]:
    """Baca audio dari cache disk. Return None jika tidak ada / rusak."""
    path = _disk_cache_path(key)
    if not path.exists():
        return None
    try:
        import soundfile as sf
        samples, sample_rate = sf.read(str(path), dtype="float32")
        return samples, sample_rate
    except Exception as e:
        logger.warning(f"[TTS] Cache disk rusak, dihapus: {path.name} ({e})")
        try:
            path.unlink()
        except OSError:
            pass
        return None


def _disk_cache_save(key: Tuple[str, float, int], samples: np.ndarray, sample_rate: int):
    """Simpan audio ke cache disk (atomic write via file sementara)."""
    try:
        import soundfile as sf
        TTS_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _disk_cache_path(key)
        tmp = path.with_suffix(".tmp")
        sf.write(str(tmp), samples, sample_rate, subtype="PCM_16", format="WAV")
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"[TTS] Gagal menyimpan cache disk: {e}")


def _memory_cache_put(key: Tuple[str, float, int], samples: np.ndarray, sample_rate: int):
    with _cache_lock:
        if key in _audio_cache or len(_audio_cache) < MEMORY_CACHE_MAX:
            _audio_cache[key] = (samples, sample_rate)


def clear_cache():
    """Hapus seluruh cache (RAM + disk), misal setelah ganti model / speed."""
    import shutil
    with _cache_lock:
        _audio_cache.clear()
    shutil.rmtree(TTS_DISK_CACHE_DIR, ignore_errors=True)


def _get_tts_engine():
    """Lazy-load TTS engine sekali saat pertama dipakai (thread-safe singleton)."""
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine

    with _tts_lock:
        if _tts_engine is not None:
            return _tts_engine

        from audio.scheduler import model_init_lock
        with model_init_lock:
            try:
                import sherpa_onnx
            except ImportError:
                logger.error("[TTS] sherpa-onnx belum terinstal. Jalankan: pip install sherpa-onnx")
                return None

            model_path = TTS_MODEL_DIR / TTS_MODEL_FILE
            tokens_path = TTS_MODEL_DIR / "tokens.txt"
            data_dir = TTS_MODEL_DIR / "espeak-ng-data"

            if not model_path.exists():
                logger.error(
                    f"[TTS] Model tidak ditemukan di '{TTS_MODEL_DIR}'. "
                    "Jalankan: python download_model.py"
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
        t0 = time.time()

        # 1) Muat dulu dari cache disk (cepat, tanpa perlu load model)
        missing = []
        disk_count = 0
        for phrase in PREWARM_PHRASES:
            key = (phrase.strip().lower(), round(TTS_SPEED, 2), TTS_SPEAKER_ID)
            with _cache_lock:
                if key in _audio_cache:
                    continue
            loaded = _disk_cache_load(key)
            if loaded is not None:
                _memory_cache_put(key, *loaded)
                disk_count += 1
            else:
                missing.append((phrase.strip(), key))

        # 2) Load engine (agar kalimat dinamis yang belum di-cache tetap cepat)
        engine = _get_tts_engine()
        if engine is None:
            return disk_count > 0

        # 3) Sintesis kalimat yang belum ada di cache, simpan ke RAM + disk
        synth_count = 0
        for phrase, key in missing:
            with _generate_lock:
                audio = engine.generate(phrase, sid=TTS_SPEAKER_ID, speed=TTS_SPEED)
            samples = np.array(audio.samples, dtype=np.float32)
            _memory_cache_put(key, samples, audio.sample_rate)
            _disk_cache_save(key, samples, audio.sample_rate)
            synth_count += 1

        elapsed = time.time() - t0
        logger.info(
            f"[TTS] Warmup selesai dalam {elapsed:.2f}s: "
            f"{disk_count} dari cache disk, {synth_count} disintesis baru"
        )
        return True
    finally:
        _is_warming_up = False


def warmup_async(precache_rooms: bool = True):
    """
    Jalankan warmup di background daemon thread.
    Setelah warmup selesai, lanjut pre-cache kalimat konfirmasi semua ruangan
    (hanya yang belum ada di cache disk).
    """
    def _run():
        warmup()
        if precache_rooms:
            precache_room_phrases()

    t = threading.Thread(target=_run, name="TTS-Warmup", daemon=True)
    t.start()


# =========================
# BACKGROUND PRE-CACHE (KONFIRMASI RUANGAN)
# =========================

# Serialisasi akses ke engine.generate (engine dipakai bersama oleh banyak thread)
_generate_lock = threading.Lock()
# Counter request sintesis foreground; pre-cache background mengalah jika > 0
_foreground_requests = 0
_foreground_lock = threading.Lock()
_precache_running = False

VALID_ROOMS_JSON = _ROOT_DIR / "nlu" / "valid_rooms.json"


def _load_room_names() -> list:
    import json
    try:
        with open(VALID_ROOMS_JSON, "r", encoding="utf-8") as f:
            return list(json.load(f).get("rooms", []))
    except Exception as e:
        logger.warning(f"[TTS] Gagal membaca daftar ruangan: {e}")
        return []


def _lower_current_thread_priority():
    """Turunkan prioritas thread ini (Linux: nice 19) agar tidak mengganggu STT/TTS."""
    if sys.platform.startswith("linux"):
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        except Exception:
            pass


def _system_busy() -> bool:
    """True jika ada sintesis foreground atau mic sedang mendengarkan."""
    if _foreground_requests > 0:
        return True
    if scheduler is not None:
        stage = getattr(scheduler.current_stage, "value", "")
        if stage == "STT_LISTENING":
            return True
    return False


def precache_room_phrases():
    """
    Pre-sintesis semua variasi kalimat konfirmasi navigasi untuk setiap ruangan
    dan simpan HANYA ke cache disk (tidak ke RAM, agar hemat memori).
    Berjalan dengan prioritas rendah dan otomatis mengalah saat sistem sibuk.
    Kalimat yang sudah ada di disk dilewati, jadi run berikutnya sangat cepat.
    """
    global _precache_running
    if _precache_running:
        return
    _precache_running = True
    try:
        try:
            from responses import NAVIGATE_CONFIRMS
        except ImportError:
            return

        rooms = _load_room_names()
        if not rooms:
            return

        _lower_current_thread_priority()

        todo = []
        for room in rooms:
            for tpl in NAVIGATE_CONFIRMS:
                text = tpl.format(loc=room).strip()
                key = (text.lower(), round(TTS_SPEED, 2), TTS_SPEAKER_ID)
                if not _disk_cache_path(key).exists():
                    todo.append((text, key))

        total = len(rooms) * len(NAVIGATE_CONFIRMS)
        if not todo:
            logger.info(f"[TTS] Pre-cache ruangan: semua {total} kalimat sudah ada di cache")
            return

        engine = _get_tts_engine()
        if engine is None:
            return

        logger.info(f"[TTS] Pre-cache ruangan dimulai: {len(todo)}/{total} kalimat (background)")
        t0 = time.time()
        done = 0
        for text, key in todo:
            # Mengalah selama sistem sedang dipakai interaksi
            while _system_busy():
                time.sleep(0.3)
            if _disk_cache_path(key).exists():
                continue
            with _generate_lock:
                audio = engine.generate(text, sid=TTS_SPEAKER_ID, speed=TTS_SPEED)
            samples = np.array(audio.samples, dtype=np.float32)
            _disk_cache_save(key, samples, audio.sample_rate)
            done += 1
            if done % 25 == 0:
                logger.info(f"[TTS] Pre-cache ruangan: {done}/{len(todo)}")

        logger.info(
            f"[TTS] Pre-cache ruangan selesai: {done} kalimat dalam {time.time()-t0:.0f}s"
        )
    except Exception as e:
        logger.warning(f"[TTS] Pre-cache ruangan gagal: {e}")
    finally:
        _precache_running = False


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

    # Cek cache disk (persisten antar restart)
    loaded = _disk_cache_load(cache_key)
    if loaded is not None:
        _memory_cache_put(cache_key, *loaded)
        logger.debug(f"[TTS Disk Cache HIT] \"{clean_text[:40]}\"")
        return loaded[0].copy(), loaded[1]

    # Generate baru jika belum di cache
    engine = _get_tts_engine()
    if engine is None:
        return None

    global _foreground_requests
    with _foreground_lock:
        _foreground_requests += 1
    try:
        t0 = time.time()
        with _generate_lock:
            audio = engine.generate(clean_text, sid=TTS_SPEAKER_ID, speed=speed)
        elapsed = time.time() - t0
    finally:
        with _foreground_lock:
            _foreground_requests -= 1

    samples = np.array(audio.samples, dtype=np.float32)
    sample_rate = audio.sample_rate

    # Simpan ke cache RAM (terbatas) dan disk (persisten)
    _memory_cache_put(cache_key, samples, sample_rate)
    _disk_cache_save(cache_key, samples, sample_rate)

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
    import time
    # Tambahkan padding hening 0.25 detik di akhir agar driver audio Windows
    # tidak memotong suku kata terakhir saat sd.wait() selesai.
    pad_len = int(sample_rate * 0.25)
    padded_samples = np.pad(samples, (0, pad_len), mode="constant")
    sd.play(padded_samples, samplerate=sample_rate)
    sd.wait()
    time.sleep(0.35)  # Beri waktu buffer soundcard dan reverb ruangan reda tuntas


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


def play_mic_cue():
    """
    Mainkan nada singkat dan lembut (chime / beep) penanda mikrofon siap mendengarkan.
    Memberikan isyarat audio instan bagi pengguna bahwa sekarang saatnya berbicara.
    Dilengkapi jeda settling akustik agar speaker tuntas dan tidak membocorkan audio ke mikrofon.
    """
    try:
        import sounddevice as sd
        sr = 16000
        # Nada dua-nada lembut D5 (587Hz, 45ms) -> A5 (880Hz, 60ms) dengan amplitudo terukur (rendah distorsi)
        t1 = np.linspace(0, 0.045, int(sr * 0.045), False)
        t2 = np.linspace(0, 0.060, int(sr * 0.060), False)
        tone1 = 0.06 * np.sin(2 * np.pi * 587.33 * t1)
        tone2 = 0.07 * np.sin(2 * np.pi * 880.00 * t2)
        # Smoothing window to prevent clicks and sharp transients
        w1 = np.hanning(len(tone1) * 2)[len(tone1):]
        w2 = np.hanning(len(tone2) * 2)[len(tone2):]
        cue = np.concatenate([tone1 * w1, tone2 * w2]).astype(np.float32)
        pad = np.zeros(int(sr * 0.08), dtype=np.float32)
        cue = np.concatenate([cue, pad])
        sd.play(cue, samplerate=sr)
        sd.wait()
        try:
            sd.stop()
        except Exception:
            pass
        # Jeda settling akustik (200ms) agar buffer soundcard tuntas
        # dan gema ruangan hilang sepenuhnya sebelum mikrofon STT dibuka.
        time.sleep(0.20)
    except Exception as e:
        logger.debug(f"[AUDIO] Mic cue error: {e}")


if __name__ == "__main__":
    # Jalankan manual: python tts.py --precache   (isi cache disk untuk semua kalimat)
    #                  python tts.py --clear-cache
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--clear-cache" in sys.argv:
        clear_cache()
        print("Cache TTS dihapus.")
    if "--precache" in sys.argv:
        warmup()
        precache_room_phrases()
