"""
Downloader Model STT & TTS Sherpa-ONNX (Bahasa Indonesia).

1. Model STT (Zipformer RNN-T Streaming):
   - encoder.onnx
   - decoder.onnx
   - joiner.onnx
   - tokens.txt
   - Target: model/

2. Model TTS (Piper VITS Offline Indonesian):
   - vits-piper-id_ID-news_tts-medium.tar.bz2
   - id_ID-news_tts-medium.onnx, tokens.txt, espeak-ng-data/
   - Target: model/tts/

Usage:
    python download_model.py          # Download STT + TTS (otomatis skip jika sudah ada)
    python download_model.py --stt    # Hanya download model STT
    python download_model.py --tts    # Hanya download model TTS
"""

import os
import sys
import shutil
import tarfile
import urllib.request
import argparse
from pathlib import Path

# =========================
# CONFIG
# =========================

MODEL_DIR = Path("model")

# STT Model Config (spacewave/sherpa-onnx-streaming-zipformer2-id)
STT_REPO_URL = "https://huggingface.co/spacewave/sherpa-onnx-streaming-zipformer2-id/resolve/main"
STT_FILES = {
    "encoder.onnx": f"{STT_REPO_URL}/encoder-iter-100000-avg-15-chunk-32-left-256.onnx",
    "decoder.onnx": f"{STT_REPO_URL}/decoder-iter-100000-avg-15-chunk-32-left-256.onnx",
    "joiner.onnx": f"{STT_REPO_URL}/joiner-iter-100000-avg-15-chunk-32-left-256.onnx",
    "tokens.txt": f"{STT_REPO_URL}/tokens.txt",
}

# TTS Model Config (k2-fsa/sherpa-onnx Piper VITS Indonesian)
TTS_MODEL_NAME = "vits-piper-id_ID-news_tts-medium"
TTS_DOWNLOAD_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{TTS_MODEL_NAME}.tar.bz2"
TTS_TARGET_DIR = MODEL_DIR / "tts"
TTS_ARCHIVE_PATH = Path(f"{TTS_MODEL_NAME}.tar.bz2")


# =========================
# PROGRESS BAR HELPER
# =========================

def _progress_hook(block_num, block_size, total_size):
    """Progress bar callback untuk urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar_len = 35
        filled = int(bar_len * percent / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {percent:5.1f}%  ({mb_done:.1f} / {mb_total:.1f} MB)")
        sys.stdout.flush()
    else:
        mb_done = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded: {mb_done:.1f} MB...")
        sys.stdout.flush()


def _download_file(url: str, dest_path: Path):
    """Download single file dengan progress bar dan User-Agent."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
    urllib.request.install_opener(opener)

    try:
        urllib.request.urlretrieve(url, str(temp_path), reporthook=_progress_hook)
        print()  # newline setelah progress bar
        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
        return True
    except Exception as e:
        print(f"\n  [ERROR] Gagal mengunduh {dest_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


# =========================
# 1. DOWNLOAD STT MODEL
# =========================

def download_stt_model() -> bool:
    """Download model STT (encoder, decoder, joiner, tokens.txt) ke folder model/."""
    print("=" * 64)
    print("  DOWNLOAD MODEL STT SHERPA-ONNX (INDONESIAN ZIPFORMER)")
    print("=" * 64)
    print(f"  Target : {MODEL_DIR.resolve()}\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Cek apakah semua file STT sudah ada
    required_stt = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
    all_exist = all((MODEL_DIR / f).exists() for f in required_stt)

    if all_exist:
        print("  [OK] Model STT sudah lengkap di folder 'model/'. Tidak perlu download ulang.")
        for f in required_stt:
            size_mb = (MODEL_DIR / f).stat().st_size / (1024 * 1024)
            print(f"       - {f:<15} ({size_mb:.2f} MB)")
        print()
        return True

    success = True
    for fname, url in STT_FILES.items():
        target_file = MODEL_DIR / fname
        if target_file.exists():
            print(f"  [OK] {fname} sudah ada, lewati.")
            continue

        print(f"  Downloading {fname} ...")
        ok = _download_file(url, target_file)
        if not ok:
            success = False
            break

    if success:
        print("\n  [OK] Model STT berhasil diinstall di 'model/'")
    print("=" * 64 + "\n")
    return success


# =========================
# 2. DOWNLOAD TTS MODEL
# =========================

def download_tts_model() -> bool:
    """Download dan extract model TTS Indonesian ke folder model/tts/."""
    print("=" * 64)
    print("  DOWNLOAD MODEL TTS SHERPA-ONNX (PIPER VITS INDONESIAN)")
    print("=" * 64)
    print(f"  Model  : {TTS_MODEL_NAME}")
    print(f"  Target : {TTS_TARGET_DIR.resolve()}\n")

    # Cek apakah model TTS sudah ada
    if (TTS_TARGET_DIR / "id_ID-news_tts-medium.onnx").exists():
        print(f"  [OK] Model TTS sudah ada di '{TTS_TARGET_DIR}'. Tidak perlu download ulang.")
        print("       - id_ID-news_tts-medium.onnx")
        print("       - tokens.txt")
        print("       - espeak-ng-data/\n")
        return True

    # Download archive tar.bz2
    print(f"  [1/3] Downloading {TTS_MODEL_NAME}.tar.bz2 (~60 MB) ...")
    try:
        urllib.request.urlretrieve(TTS_DOWNLOAD_URL, str(TTS_ARCHIVE_PATH), reporthook=_progress_hook)
        print()
    except Exception as e:
        print(f"\n  [ERROR] Gagal download TTS archive: {e}")
        return False

    # Extract
    print(f"  [2/3] Extracting archive ...")
    try:
        with tarfile.open(str(TTS_ARCHIVE_PATH), "r:bz2") as tar:
            tar.extractall(".")
    except Exception as e:
        print(f"\n  [ERROR] Gagal extract archive: {e}\n")
        return False

    # Move ke target directory
    print(f"  [3/3] Moving ke {TTS_TARGET_DIR} ...")
    extracted_dir = Path(TTS_MODEL_NAME)
    if not extracted_dir.exists():
        print(f"\n  [ERROR] Folder extract '{TTS_MODEL_NAME}' tidak ditemukan!\n")
        return False

    TTS_TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    if TTS_TARGET_DIR.exists():
        shutil.rmtree(TTS_TARGET_DIR)

    shutil.move(str(extracted_dir), str(TTS_TARGET_DIR))

    # Cleanup archive
    try:
        TTS_ARCHIVE_PATH.unlink()
    except OSError:
        pass

    # Verify
    required_files = ["id_ID-news_tts-medium.onnx", "tokens.txt"]
    for fname in required_files:
        if not (TTS_TARGET_DIR / fname).exists():
            print(f"\n  [ERROR] File '{fname}' tidak ditemukan di model directory!\n")
            return False

    print(f"\n  [OK] Model TTS berhasil diinstall di '{TTS_TARGET_DIR}'")
    print("=" * 64 + "\n")
    return True


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Downloader Model STT & TTS Offline Sherpa-ONNX (Bahasa Indonesia)"
    )
    parser.add_argument("--stt", action="store_true", help="Hanya unduh model STT (Zipformer)")
    parser.add_argument("--tts", action="store_true", help="Hanya unduh model TTS (Piper VITS)")

    args = parser.parse_args()

    # Jika tidak ada flag spesifik, unduh keduanya
    download_all = not args.stt and not args.tts

    stt_ok = True
    tts_ok = True

    if download_all or args.stt:
        stt_ok = download_stt_model()

    if download_all or args.tts:
        tts_ok = download_tts_model()

    if stt_ok and tts_ok:
        print("Semua model telah siap digunakan!")
        sys.exit(0)
    else:
        print("Terdapat kegagalan saat mempersiapkan model.")
        sys.exit(1)


if __name__ == "__main__":
    main()
