"""
Download model TTS Sherpa-ONNX (Piper VITS Indonesian).

Model: vits-piper-id_ID-news_tts-medium
  - Ukuran: ~60 MB (compressed tar.bz2)
  - Format: ONNX + espeak-ng-data
  - Bahasa: Indonesia (id-ID)
  - Speaker: Single speaker

Usage:
    python download_tts_model.py
"""

import os
import sys
import shutil
import tarfile
import urllib.request
from pathlib import Path


# =========================
# CONFIG
# =========================

MODEL_NAME = "vits-piper-id_ID-news_tts-medium"
DOWNLOAD_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{MODEL_NAME}.tar.bz2"
TARGET_DIR = Path("model") / "tts"
ARCHIVE_PATH = Path(f"{MODEL_NAME}.tar.bz2")


# =========================
# DOWNLOAD HELPERS
# =========================

def _progress_hook(block_num, block_size, total_size):
    """Progress bar callback untuk urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {percent:5.1f}%  ({mb_done:.1f} / {mb_total:.1f} MB)")
        sys.stdout.flush()
    else:
        mb_done = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded: {mb_done:.1f} MB...")
        sys.stdout.flush()


def download_tts_model():
    """Download dan extract model TTS Indonesian ke folder model/tts/."""

    print("=" * 60)
    print("  DOWNLOAD MODEL TTS SHERPA-ONNX (INDONESIAN)")
    print("=" * 60)
    print(f"\n  Model  : {MODEL_NAME}")
    print(f"  URL    : {DOWNLOAD_URL}")
    print(f"  Target : {TARGET_DIR.resolve()}")

    # Cek apakah sudah ada
    if (TARGET_DIR / "id_ID-news_tts-medium.onnx").exists():
        print(f"\n  [OK] Model sudah ada di '{TARGET_DIR}'. Tidak perlu download ulang.")
        print("      Hapus folder tersebut jika ingin download ulang.\n")
        return True

    # Download
    print(f"\n  [1/3] Downloading {MODEL_NAME}.tar.bz2 ...")
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, str(ARCHIVE_PATH), reporthook=_progress_hook)
        print()  # newline setelah progress bar
    except Exception as e:
        print(f"\n\n  [ERROR] Gagal download: {e}")
        print("  Tips: Pastikan koneksi internet tersedia.")
        print("  Alternatif: Download manual dari URL di atas.\n")
        return False

    # Extract
    print(f"  [2/3] Extracting archive ...")
    try:
        with tarfile.open(str(ARCHIVE_PATH), "r:bz2") as tar:
            tar.extractall(".")
    except Exception as e:
        print(f"\n  [ERROR] Gagal extract archive: {e}\n")
        return False

    # Move ke target directory
    print(f"  [3/3] Moving ke {TARGET_DIR} ...")
    extracted_dir = Path(MODEL_NAME)
    if not extracted_dir.exists():
        print(f"\n  [ERROR] Folder hasil extract '{MODEL_NAME}' tidak ditemukan!\n")
        return False

    # Buat parent dir jika belum ada
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Hapus target jika sudah ada (re-download)
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    shutil.move(str(extracted_dir), str(TARGET_DIR))

    # Cleanup archive
    try:
        ARCHIVE_PATH.unlink()
    except OSError:
        pass

    # Verify
    required_files = ["id_ID-news_tts-medium.onnx", "tokens.txt"]
    for fname in required_files:
        if not (TARGET_DIR / fname).exists():
            print(f"\n  [ERROR] File '{fname}' tidak ditemukan di model directory!\n")
            return False

    print(f"\n  [OK] Model TTS berhasil diinstall di '{TARGET_DIR}'")
    print(f"      - id_ID-news_tts-medium.onnx")
    print(f"      - tokens.txt")
    print(f"      - espeak-ng-data/")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    success = download_tts_model()
    sys.exit(0 if success else 1)
