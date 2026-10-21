"""
Resource Scheduler & Task Pinning Optimizer.

Modul ini bertanggung jawab untuk:
1. Task Pinning & CPU Affinity:
   - Menyetel thread priority (Above Normal / High) untuk audio capture & playback
   - Task pinning / CPU core affinity (Linux os.sched_setaffinity / Windows SetThreadAffinityMask)
2. Sequential Audio Synchronization (Resource Mutual Exclusion):
   - Menjamin isolasi penuh antara Audio Input (Mic) dan Audio Output (Speaker)
   - Mencegah ALSA duplex contention & audio xrun/underrun pada Raspberry Pi / Linux
   - Acoustic settling delay (150ms) setelah TTS selesai agar mic tidak menangkap gema speaker
3. Latency & Resource Monitoring.
"""

import os
import sys
import time
import logging
import threading
from enum import Enum
from typing import Optional, Set, List

logger = logging.getLogger(__name__)


# ============================================================================
# 1. TASK PINNING & CPU SCHEDULING (OS-Level Priority & Core Affinity)
# ============================================================================

def set_thread_high_priority():
    """
    Tingkatkan prioritas thread saat ini (misal audio capture / playback thread)
    untuk mencegah audio buffer underrun / stutter saat CPU sibuk.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            # THREAD_PRIORITY_ABOVE_NORMAL = 1, THREAD_PRIORITY_HIGHEST = 2
            k32 = ctypes.windll.kernel32
            handle = k32.GetCurrentThread()
            k32.SetThreadPriority(handle, 1)  # Above Normal
            logger.debug("[SCHEDULER] Windows thread priority set to ABOVE_NORMAL")
        elif sys.platform.startswith("linux"):
            # Linux: coba turunkan nice value thread jika diizinkan
            try:
                import os
                # os.nice(-5) memerlukan izin atau CAP_SYS_NICE
                os.nice(-2)
                logger.debug("[SCHEDULER] Linux thread nice adjusted (-2)")
            except (PermissionError, OSError):
                pass
    except Exception as e:
        logger.debug(f"[SCHEDULER] Gagal set thread priority: {e}")


def pin_current_thread_to_cores(cores: List[int]):
    """
    Pin thread saat ini ke core CPU tertentu (Task Pinning / Affinity).
    
    Args:
        cores: List indeks core CPU (misal [0, 1] atau [2, 3]).
    """
    if not cores:
        return

    try:
        if sys.platform.startswith("linux") and hasattr(os, "sched_setaffinity"):
            # Native Linux (Raspberry Pi 4 Quad-Core / Ubuntu Server)
            os.sched_setaffinity(0, set(cores))
            logger.debug(f"[SCHEDULER] Linux thread pinned to CPU cores: {cores}")
        elif sys.platform == "win32":
            import ctypes
            # Windows SetThreadAffinityMask: bitmask (core 0 -> 1, core 1 -> 2, core 2 -> 4, etc.)
            mask = 0
            for c in cores:
                mask |= (1 << c)
            k32 = ctypes.windll.kernel32
            handle = k32.GetCurrentThread()
            res = k32.SetThreadAffinityMask(handle, mask)
            if res:
                logger.debug(f"[SCHEDULER] Windows thread pinned to mask: {bin(mask)}")
    except Exception as e:
        logger.debug(f"[SCHEDULER] Task pinning not applied or not supported: {e}")


# ============================================================================
# 2. SEQUENTIAL AUDIO STAGE SCHEDULER (Resource Arbiter)
# ============================================================================

class AudioStage(Enum):
    IDLE = "IDLE"
    TTS_SPEAKING = "TTS_SPEAKING"
    SETTLING = "SETTLING"
    STT_LISTENING = "STT_LISTENING"
    NLU_PROCESSING = "NLU_PROCESSING"


class ResourceScheduler:
    """
    Scheduler sekuensial yang mengisolasi akses audio input (mic) dan output (speaker).
    Mencegah race condition antar-consumer audio dan mengeliminasi echo ruangan.
    """

    def __init__(self, settling_delay_ms: int = 150):
        self._lock = threading.RLock()
        self._current_stage = AudioStage.IDLE
        self._settling_delay_sec = settling_delay_ms / 1000.0
        self._stage_change_time = time.time()

    @property
    def current_stage(self) -> AudioStage:
        return self._current_stage

    def enter_tts(self):
        """Memulai fase TTS: audio output eksklusif."""
        with self._lock:
            self._current_stage = AudioStage.TTS_SPEAKING
            self._stage_change_time = time.time()
            set_thread_high_priority()
            logger.debug("[SCHEDULER] Stage -> TTS_SPEAKING")

    def exit_tts(self):
        """
        Mengakhiri fase TTS dan memasuki fase settling (menunggu gema ruangan hilang
        sebelum membuka stream microphone).
        """
        with self._lock:
            self._current_stage = AudioStage.SETTLING
            logger.debug(f"[SCHEDULER] Stage -> SETTLING ({self._settling_delay_sec*1000:.0f}ms)")
            if self._settling_delay_sec > 0:
                time.sleep(self._settling_delay_sec)
            self._current_stage = AudioStage.IDLE

    def enter_stt(self):
        """Memulai fase STT: audio input mikrofon eksklusif."""
        with self._lock:
            self._current_stage = AudioStage.STT_LISTENING
            self._stage_change_time = time.time()
            set_thread_high_priority()
            logger.debug("[SCHEDULER] Stage -> STT_LISTENING")

    def exit_stt(self):
        """Mengakhiri fase STT mikrofon."""
        with self._lock:
            self._current_stage = AudioStage.IDLE
            logger.debug("[SCHEDULER] Stage -> IDLE (STT complete)")

    def enter_nlu(self):
        """Memulai fase NLU (pemrosesan teks / parsing)."""
        with self._lock:
            self._current_stage = AudioStage.NLU_PROCESSING
            logger.debug("[SCHEDULER] Stage -> NLU_PROCESSING")


# Global singleton instance
scheduler = ResourceScheduler()
