from pathlib import Path


# Folder model Sherpa-ONNX
MODEL = Path("model")


# Sample rate audio
STT_RATE = 16000


# Jumlah CPU thread
NUM_THREADS = 4


# Index default perangkat mikrofon (None = otomatis deteksi perangkat input aktif)
# Bisa diisi angka ID spesifik (misal: DEFAULT_MIC_DEVICE = 2 untuk USB Mic)
DEFAULT_MIC_DEVICE = None