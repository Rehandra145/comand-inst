"""
Audio Utilities — Deteksi hardware mikrofon, sample rate, dan resampling DSP.
"""

import logging
from typing import Optional
import numpy as np

from config import DEFAULT_MIC_DEVICE

logger = logging.getLogger(__name__)


def get_effective_input_device(device: Optional[int] = None) -> Optional[int]:
    """
    Menentukan index perangkat mikrofon yang aktif:
    1. Jika ditentukan via argumen/flag (--device <ID>), prioritaskan itu.
    2. Jika diset di config.py (DEFAULT_MIC_DEVICE), gunakan itu.
    3. Jika default OS memiliki channel input > 0, gunakan itu.
    4. Jika default OS tidak punya input channel (kasus umum di Ubuntu Server / Raspi),
       otomatis cari dan gunakan perangkat input pertama yang aktif (misal USB Mic).
    """
    if device is not None:
        return device

    try:
        import sounddevice as sd
        devices = sd.query_devices()

        if DEFAULT_MIC_DEVICE is not None:
            if 0 <= DEFAULT_MIC_DEVICE < len(devices) and devices[DEFAULT_MIC_DEVICE].get("max_input_channels", 0) > 0:
                return DEFAULT_MIC_DEVICE
            logger.warning(
                f"[AUDIO] DEFAULT_MIC_DEVICE={DEFAULT_MIC_DEVICE} tidak valid atau tidak memiliki channel input. "
                "Beralih ke deteksi otomatis."
            )

        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else -1
        if 0 <= default_in < len(devices) and devices[default_in].get("max_input_channels", 0) > 0:
            return default_in

        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                logger.info(f"[AUDIO] Otomatis memilih input device ID {i}: {dev['name']}")
                return i
    except Exception:
        pass

    return None


def get_supported_sample_rate(device: Optional[int] = None) -> int:
    """
    Deteksi sample rate yang didukung langsung oleh hardware mikrofon.
    Prioritaskan 16000 Hz (native STT). Jika hardware menolak 16000 Hz
    (misal USB mic yang hanya mendukung 48000 Hz / 44100 Hz), otomatis
    gunakan sample rate asli hardware tersebut.
    """
    import sounddevice as sd
    candidates = [16000, 48000, 44100, 32000, 22050, 8000]
    for rate in candidates:
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="float32")
            return rate
        except Exception:
            continue

    try:
        dev_info = sd.query_devices(device, "input")
        return int(dev_info.get("default_samplerate", 16000))
    except Exception:
        return 16000


def get_optimal_channels(device: Optional[int] = None, rate: int = 16000) -> int:
    """Deteksi jumlah channel minimum yang didukung hardware (1 atau max_channels)."""
    import sounddevice as sd
    try:
        sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="float32")
        return 1
    except Exception:
        try:
            dev_info = sd.query_devices(device, "input")
            max_ch = int(dev_info.get("max_input_channels", 2))
            return max_ch if max_ch > 0 else 1
        except Exception:
            return 1


def resample_to_16k(samples, orig_sr: int):
    """Resample array audio float32 dari orig_sr ke 16000 Hz untuk Sherpa-ONNX."""
    if orig_sr == 16000 or len(samples) == 0:
        return np.ascontiguousarray(samples, dtype=np.float32)

    if orig_sr == 48000:
        # Decimation 3:1 (48000 Hz / 3 = 16000 Hz) — C-contiguous array wajib untuk C++ pybind11!
        return np.ascontiguousarray(samples[::3], dtype=np.float32)

    if orig_sr == 32000:
        return np.ascontiguousarray(samples[::2], dtype=np.float32)

    if orig_sr == 44100:
        try:
            from scipy.signal import resample_poly
            return np.ascontiguousarray(resample_poly(samples, 160, 441), dtype=np.float32)
        except ImportError:
            pass

    # Interpolasi cepat untuk sample rate sembarang (misal 44100 Hz tanpa scipy)
    target_length = int(len(samples) * 16000 / orig_sr)
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0, 1, len(samples), endpoint=False)
    x_new = np.linspace(0, 1, target_length, endpoint=False)
    return np.ascontiguousarray(np.interp(x_new, x_old, samples), dtype=np.float32)


def list_audio_devices():
    """Tampilkan daftar perangkat input audio (microphone) yang tersedia dalam kartu styling."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"\n\033[1;36m╭─── 🎧 DAFTAR PERANGKAT INPUT AUDIO (MIKROFON) ─────────────────────╮\033[0m")
        found = False
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else -1
        effective_in = get_effective_input_device()
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                found = True
                is_active = (i == effective_in)
                marker = "\033[1;32m● AKTIF\033[0m" if is_active else "\033[38;5;244m○      \033[0m"
                native_sr = get_supported_sample_rate(i)
                ch = dev.get("max_input_channels", 1)
                print(f"  [{marker}] \033[1mID {i:2d}\033[0m: \033[1;37m{dev['name']}\033[0m \033[38;5;244m({ch} Ch, {native_sr} Hz)\033[0m")
        if not found:
            print("  \033[1;31m(Tidak ada perangkat mikrofon yang terdeteksi)\033[0m")
        else:
            print(f"\n  \033[38;5;244mKeterangan: [\033[1;32m● AKTIF\033[38;5;244m] = Perangkat aktif yang digunakan (ID: {effective_in}).\033[0m")
            print(f"  \033[38;5;244mGunakan flag --device <ID> untuk menentukan ID mikrofon tertentu.\033[0m")
        print(f"\033[1;36m╰────────────────────────────────────────────────────────────────────╯\033[0m\n")
    except Exception as e:
        print(f"Gagal mendeteksi perangkat audio: {e}\n")


def test_mic_hardware(device: Optional[int] = None, duration: int = 3):
    """Diagnosa dan uji coba mikrofon dengan visual VU-meter dan analisis sensitivitas."""
    device = get_effective_input_device(device)
    print(f"\n\033[1;36m╭─── ⚡ DIAGNOSA PERANGKAT & LEVEL AUDIO MIKROFON ────────────────────╮\033[0m")
    try:
        import sounddevice as sd
    except ImportError as e:
        print(f"  \033[1;31m[ERROR] Library sounddevice belum terinstal: {e}\033[0m")
        print("  Jalankan: uv pip install sounddevice numpy\n")
        print(f"\033[1;36m╰────────────────────────────────────────────────────────────────────╯\033[0m\n")
        return

    try:
        dev_info = sd.query_devices(device, "input")
    except Exception as e:
        print(f"  \033[1;31m[ERROR] Gagal mengakses perangkat audio input (ID: {device}): {e}\033[0m")
        print("  Gunakan flag --list-devices untuk melihat ID yang tersedia.\n")
        print(f"\033[1;36m╰────────────────────────────────────────────────────────────────────╯\033[0m\n")
        return

    dev_name = dev_info.get("name", "Unknown")
    channels = dev_info.get("max_input_channels", 0)
    capture_sr = get_supported_sample_rate(device)
    optimal_channels = get_optimal_channels(device, capture_sr)

    print(f"  \033[1m1. Informasi Hardware:\033[0m")
    print(f"     - Perangkat Terpilih   : \033[1;32m[{device if device is not None else 'Default'}] {dev_name}\033[0m")
    print(f"     - Channel Hardware     : {channels} Ch (Menggunakan: {optimal_channels} Ch Optimal)")
    sr_note = "(\033[38;5;244mauto-resample 16000 Hz\033[0m)" if capture_sr != 16000 else "(\033[1;32mNative STT\033[0m)"
    print(f"     - Hardware Sample Rate : {capture_sr} Hz {sr_note}")

    print(f"\n  \033[1m2. Merekam sampel audio ({duration} detik)...\033[0m")
    print("     \033[1;33m>> SILAKAN BERBICARA ATAU BUAT SUARA SEKARANG...\033[0m")

    try:
        recording = sd.rec(
            int(duration * capture_sr),
            samplerate=capture_sr,
            channels=optimal_channels,
            dtype="float32",
            device=device,
        )
        sd.wait()
    except Exception as e:
        print(f"\n  \033[1;31m[ERROR] Gagal merekam audio dari driver: {e}\033[0m")
        print(f"\033[1;36m╰────────────────────────────────────────────────────────────────────╯\033[0m\n")
        return

    # Hitung RMS & Peak Amplitude
    samples_flat = recording.flatten()
    rms = float(np.sqrt(np.mean(samples_flat ** 2)))
    peak = float(np.max(np.abs(samples_flat)))

    # Visual VU-Meter bar (24 blocks)
    vu_blocks = int(min(1.0, peak * 2.5) * 24)
    vu_bar = "█" * vu_blocks + "░" * (24 - vu_blocks)
    db_level = 20 * np.log10(peak) if peak > 1e-5 else -99.9

    print(f"\n  \033[1m3. Hasil Pengukuran Level Sinyal Audio:\033[0m")
    print(f"     - Level VU Meter  : [\033[1;32m{vu_bar[:16]}\033[1;33m{vu_bar[16:21]}\033[1;31m{vu_bar[21:]}\033[0m] {db_level:.1f} dBFS")
    print(f"     - RMS Sinyal      : {rms:.5f}")
    print(f"     - Peak Amplitude  : {peak:.5f} / 1.000")

    if peak < 0.01:
        print("\n  \033[1;31m[PERINGATAN] Sinyal suara SANGAT LEMAH atau HENING (MUTE)!\033[0m")
        print("  Driver hardware terdeteksi, tetapi mikrofon tidak menangkap suara.")
        print("  Tips: Periksa slider volume microphone di pengaturan audio / alsamixer.")
    elif peak < 0.05:
        print("\n  \033[1;33m[CATATAN] Sinyal suara terdeteksi namun agak pelan.\033[0m")
        print("  Disarankan sedikit menaikkan gain/volume mikrofon.")
    else:
        print("\n  \033[1;32m✓ [BERHASIL] Driver & Mikrofon berfungsi NORMAL! Sinyal audio masuk dengan jelas.\033[0m")

    try:
        import soundfile as sf
        sf.write("test_mic.wav", recording, capture_sr)
        print("  \033[38;5;244m(Sampel rekaman disimpan ke 'test_mic.wav' untuk verifikasi)\033[0m")
    except Exception:
        pass

    print(f"\033[1;36m╰────────────────────────────────────────────────────────────────────╯\033[0m\n")
