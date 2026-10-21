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

    if DEFAULT_MIC_DEVICE is not None:
        return DEFAULT_MIC_DEVICE

    try:
        import sounddevice as sd
        devices = sd.query_devices()
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
    """Tampilkan daftar perangkat input audio (microphone) yang tersedia."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("\n=== DAFTAR PERANGKAT INPUT AUDIO ===")
        found = False
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else -1
        effective_in = get_effective_input_device()
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                found = True
                marker = "*" if i == effective_in else " "
                native_sr = get_supported_sample_rate(i)
                print(f" [{marker}] Device ID {i:2d}: {dev['name']} (Channels: {dev['max_input_channels']}, Rate: {native_sr} Hz)")
        if not found:
            print("  (Tidak ada perangkat microphone yang terdeteksi)")
        else:
            print(f"\n  Keterangan: [*] = Perangkat input aktif yang akan digunakan (ID: {effective_in}).")
            print("  Gunakan flag --device <ID> untuk memilih mikrofon tertentu.\n")
    except Exception as e:
        print(f"Gagal mendeteksi perangkat audio: {e}\n")


def test_mic_hardware(device: Optional[int] = None, duration: int = 3):
    """Diagnosa dan uji coba mikrofon, periksa level sinyal audio & volume."""
    device = get_effective_input_device(device)
    print("\n" + "=" * 60)
    print("  [DIAGNOSA] Driver Audio & Perangkat Mikrofon")
    print("=" * 60)
    try:
        import sounddevice as sd
    except ImportError as e:
        print(f"[ERROR] Library belum lengkap: {e}")
        print("   Jalankan: sudo apt install -y portaudio19-dev libasound2-dev")
        print("   Lalu: uv pip install sounddevice numpy\n")
        return

    try:
        dev_info = sd.query_devices(device, "input")
    except Exception as e:
        print(f"[ERROR] Gagal mengakses perangkat audio input (ID: {device}): {e}")
        print("   Gunakan flag --list-devices untuk melihat ID perangkat yang tersedia.\n")
        return

    dev_name = dev_info.get("name", "Unknown")
    channels = dev_info.get("max_input_channels", 0)
    capture_sr = get_supported_sample_rate(device)
    optimal_channels = get_optimal_channels(device, capture_sr)

    print(f"\n1. Perangkat Terpilih : [{device if device is not None else 'Default'}] {dev_name}")
    print(f"   Jumlah Channel     : {channels} (Menggunakan: {optimal_channels} channel)")
    print(f"   Hardware Sample Rate: {capture_sr} Hz {'(akan di-resample otomatis ke 16000 Hz untuk STT)' if capture_sr != 16000 else ''}")

    print(f"\n2. Merekam suara sampel selama {duration} detik...")
    print("   >> SILAKAN BERBICARA ATAU BUAT SUARA SEKARANG...")

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
        print(f"\n[ERROR] Gagal merekam audio dari driver: {e}")
        print("   Tips Ubuntu Server:")
        print("   1. Pastikan user masuk ke grup audio: 'sudo usermod -aG audio $USER'")
        print("   2. Cek apakah ALSA mengenali mic USB: 'arecord -l'\n")
        return

    # Hitung RMS & Peak Amplitude
    samples_flat = recording.flatten()
    rms = float(np.sqrt(np.mean(samples_flat ** 2)))
    peak = float(np.max(np.abs(samples_flat)))

    print(f"\n3. Hasil Pengukuran Level Sinyal Audio:")
    print(f"   - Root Mean Square (RMS) Level : {rms:.5f}")
    print(f"   - Peak Amplitude (0.0 s/d 1.0) : {peak:.5f}")

    if peak < 0.01:
        print("\n[PERINGATAN] Sinyal suara SANGAT LEMAH atau HENING (MUTE)!")
        print("   Driver hardware terdeteksi, tetapi mikrofon tidak menangkap suara.")
        print("   Penyebab & Solusi Umum di Ubuntu Server:")
        print("   - Volume Capture ALSA di-mute atau bernilai 0%.")
        print("   - Jalankan 'alsamixer' di terminal -> Tekan F6 (pilih USB Mic) -> Tekan F4 (Capture) -> Naikkan volume tombol Panah Atas.")
    elif peak < 0.05:
        print("\n[CATATAN] Sinyal suara terdeteksi tetapi volumenya agak pelan.")
        print("   Disarankan menaikkan gain mikrofon lewat 'alsamixer'.")
    else:
        print("\n[BERHASIL] Driver & Mikrofon berfungsi NORMAL! Sinyal audio masuk dengan jelas.")

    # Simpan sampel ke file test_mic.wav
    try:
        import soundfile as sf
        sf.write("test_mic.wav", recording, capture_sr)
        print("   (File rekaman sampel berhasil disimpan ke 'test_mic.wav' untuk verifikasi)")
    except Exception:
        pass

    print("=" * 60 + "\n")
