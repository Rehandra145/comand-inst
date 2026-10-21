"""
Quantize model Sherpa-ONNX ke INT8 untuk performa lebih cepat di Raspberry Pi.

Penggunaan:
    python quantize_model.py

Output:
    model/encoder.int8.onnx  (dari model/encoder.onnx)
    model/joiner.int8.onnx   (dari model/joiner.onnx)

Catatan:
    - decoder.onnx TIDAK perlu di-quantize (ukurannya sudah kecil ~500 KB)
    - Setelah quantize, stt.py akan otomatis mendeteksi file .int8.onnx
"""

import sys
from pathlib import Path

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("[ERROR] onnxruntime tidak ditemukan.")
    print("   Install dulu: pip install onnxruntime")
    sys.exit(1)


MODEL_DIR = Path("model")

# File yang akan di-quantize (encoder & joiner — yang paling berat untuk inferensi)
# decoder.onnx (~500 KB) tidak perlu di-quantize
TARGETS = [
    ("encoder.onnx", "encoder.int8.onnx"),
    ("joiner.onnx", "joiner.int8.onnx"),
]


def quantize():
    for src_name, dst_name in TARGETS:
        src = MODEL_DIR / src_name
        dst = MODEL_DIR / dst_name

        if not src.exists():
            print(f"[SKIP] {src} tidak ditemukan, lewati.")
            continue

        if dst.exists():
            print(f"[SKIP] {dst} sudah ada. Hapus file tersebut jika ingin quantize ulang.")
            continue

        src_size_mb = src.stat().st_size / (1024 * 1024)
        print(f"[QUANTIZE] {src_name} ({src_size_mb:.1f} MB) -> {dst_name} ...")

        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
        )

        dst_size_mb = dst.stat().st_size / (1024 * 1024)
        ratio = (1 - dst_size_mb / src_size_mb) * 100
        print(f"   [OK] Selesai: {dst_size_mb:.1f} MB (hemat {ratio:.0f}%)")

    print("\n[DONE] Quantization selesai.")
    print("   stt.py akan otomatis mendeteksi file .int8.onnx saat program dijalankan.")


if __name__ == "__main__":
    quantize()
