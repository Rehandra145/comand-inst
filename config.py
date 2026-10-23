from pathlib import Path


# Folder model Sherpa-ONNX
MODEL = Path("model")


# Sample rate audio
STT_RATE = 16000


# Jumlah CPU thread
NUM_THREADS = 3

# Index default perangkat mikrofon (None = otomatis deteksi perangkat input aktif)
# Bisa diisi angka ID spesifik (misal: DEFAULT_MIC_DEVICE = 0 atau 2)
DEFAULT_MIC_DEVICE = 24

# Default ROS 2 TCP Socket Node Target
DEFAULT_ROS2_HOST = "10.107.245.221"
DEFAULT_ROS2_PORT = 5000
