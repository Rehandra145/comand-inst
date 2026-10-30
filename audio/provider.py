"""
Provider resolver — Deteksi akselerasi GPU (CUDA) vs fallback otomatis ke CPU,
serta manajemen preferensi pengguna (Mode GPU vs Mode CPU) dengan persistensi konfigurasi.
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("compute_config.json")

_cuda_available_cache: Optional[bool] = None
_stt_provider: Optional[str] = None
_tts_provider: Optional[str] = None
_manual_override_mode: Optional[str] = None


def is_cuda_available() -> bool:
    """
    Cek ketersediaan hardware GPU NVIDIA dan pustaka CUDA sherpa-onnx.
    """
    global _cuda_available_cache
    if _cuda_available_cache is not None:
        return _cuda_available_cache

    try:
        # 1. Deteksi hardware/driver GPU NVIDIA
        is_tegra = os.path.exists("/sys/devices/platform/gpu.0") or os.path.exists("/sys/devices/gpu.0")
        has_smi = shutil.which("nvidia-smi") is not None
        has_gpu_hw = is_tegra or has_smi

        if not has_gpu_hw:
            _cuda_available_cache = False
            return False

        # 2. Deteksi apakah paket sherpa_onnx memiliki shared library CUDA
        import sherpa_onnx
        sherpa_dir = Path(sherpa_onnx.__file__).parent
        lib_dir = sherpa_dir / "lib"
        has_cuda_lib = False

        if lib_dir.exists():
            for f in lib_dir.iterdir():
                if "cuda" in f.name.lower():
                    has_cuda_lib = True
                    break

        _cuda_available_cache = has_gpu_hw and has_cuda_lib
        return _cuda_available_cache
    except Exception as e:
        logger.debug(f"[PROVIDER] Gagal mengecek ketersediaan CUDA: {e}")
        _cuda_available_cache = False
        return False


def load_compute_mode() -> str:
    """
    Baca mode akselerasi yang dipilih pengguna ('gpu', 'cpu', atau 'auto') dari compute_config.json.
    Default: 'gpu' (jika CUDA didukung hardware/library) atau 'cpu' (jika tidak ada CUDA).
    """
    global _manual_override_mode
    if _manual_override_mode is not None:
        return _manual_override_mode

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                mode = str(data.get("mode", "auto")).lower()
                if mode in ("gpu", "cuda"):
                    return "gpu"
                elif mode == "cpu":
                    return "cpu"
                elif mode == "auto":
                    return "auto"
        except Exception as e:
            logger.warning(f"[PROVIDER] Gagal membaca {CONFIG_PATH}: {e}")

    return "gpu" if is_cuda_available() else "cpu"


def save_compute_mode(mode: str) -> bool:
    """
    Simpan mode akselerasi yang dipilih pengguna ('gpu' atau 'cpu') ke compute_config.json.
    """
    global _manual_override_mode
    mode = mode.lower()
    if mode in ("cuda", "gpu"):
        target_mode = "gpu"
    elif mode == "cpu":
        target_mode = "cpu"
    else:
        target_mode = "auto"

    _manual_override_mode = target_mode

    try:
        data = {"mode": target_mode}
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[PROVIDER] Preferensi komputasi berhasil disimpan ke {CONFIG_PATH}: {target_mode.upper()}")
        return True
    except Exception as e:
        logger.error(f"[PROVIDER] Gagal menyimpan preferensi komputasi: {e}")
        return False


def get_preferred_provider() -> str:
    """
    Menentukan provider ('cuda' atau 'cpu') berdasarkan pilihan pengguna dan ketersediaan hardware.
    - Jika pengguna memilih 'cpu' -> paksa 'cpu'.
    - Jika pengguna memilih 'gpu' atau 'auto' -> gunakan 'cuda' jika hardware & library mendukung,
      jika tidak mendukung, otomatis fallback ke 'cpu'.
    """
    mode = load_compute_mode()
    if mode == "cpu":
        return "cpu"

    # Default adalah GPU (CUDA) jika hardware & pustaka tersedia
    if is_cuda_available():
        return "cuda"

    return "cpu"


def set_stt_provider(provider: str) -> None:
    global _stt_provider
    _stt_provider = provider.lower()


def get_stt_provider() -> str:
    global _stt_provider
    if _stt_provider is not None:
        return _stt_provider
    return get_preferred_provider()


def set_tts_provider(provider: str) -> None:
    global _tts_provider
    _tts_provider = provider.lower()


def get_tts_provider() -> str:
    global _tts_provider
    if _tts_provider is not None:
        return _tts_provider
    return get_preferred_provider()


def get_active_provider_name() -> str:
    """Nama deskriptif akselerator aktif untuk dashboard dan CLI."""
    stt_p = get_stt_provider()
    tts_p = get_tts_provider()
    if stt_p == "cuda" or tts_p == "cuda":
        return "GPU (CUDA)"
    return "CPU"


def restart_current_process(root_widget=None):
    """
    Restart aplikasi secara bersih dengan argumen yang sama persis.
    """
    logger.info("[PROVIDER] Memulai restart aplikasi...")
    try:
        if root_widget is not None and hasattr(root_widget, "destroy"):
            root_widget.destroy()
    except Exception:
        pass

    python = sys.executable
    os.execv(python, [python] + sys.argv)
