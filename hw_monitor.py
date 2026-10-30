"""
Hardware & Compute Accelerator Monitor — Deteksi & Telemetri Real-Time GPU vs CPU.
Khusus dioptimasi untuk arsitektur NVIDIA Jetson (Orin/Xavier/Nano) dan Linux/CUDA.
"""

import os
import platform
import sys
import time
from typing import Dict, Any, Optional, Tuple


class HardwareMonitor:
    """Monitor telemetri real-time untuk utilisasi GPU (Tegra/CUDA), CPU, RAM, dan suhu."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HardwareMonitor, cls).__new__(cls)
            cls._instance._init_monitor()
        return cls._instance

    def _init_monitor(self):
        self._prev_idle = None
        self._prev_total = None
        self._cached_cpu = 0.0
        self._last_cpu_time = 0.0

        # Identifikasi chip & arsitektur perangkat
        self.device_model = self._detect_device_model()
        self.cuda_info = self._detect_cuda_info()
        self.providers = self._detect_configured_providers()

    def _detect_device_model(self) -> str:
        """Deteksi model spesifik board/komputer (misal: NVIDIA Jetson Orin Nano)."""
        dt_model_path = "/proc/device-tree/model"
        if os.path.exists(dt_model_path):
            try:
                with open(dt_model_path, "r", encoding="utf-8", errors="ignore") as f:
                    model = f.read().strip().replace("\x00", "")
                    if model:
                        return model
            except Exception:
                pass
        return f"{platform.system()} {platform.machine()} ({platform.processor() or 'CPU'})"

    def _detect_cuda_info(self) -> Dict[str, Any]:
        """Periksa ketersediaan driver dan engine CUDA di sistem."""
        info = {
            "available": False,
            "version": None,
            "driver": None,
            "device_name": "CPU",
            "is_jetson": False,
        }

        # Cek apakah platform Jetson Tegra
        if os.path.exists("/sys/devices/platform/gpu.0") or os.path.exists("/sys/devices/gpu.0"):
            info["is_jetson"] = True
            info["available"] = True
            info["device_name"] = "NVIDIA Orin GPU (Tegra)"

        # Cek via nvidia-smi / CUDA runtime jika ada
        try:
            import subprocess
            res = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2,
            )
            if res.returncode == 0:
                info["available"] = True
                out = res.stdout
                if "CUDA Version:" in out:
                    parts = out.split("CUDA Version:")
                    info["version"] = parts[1].split()[0].strip()
                if "Driver Version:" in out:
                    parts = out.split("Driver Version:")
                    info["driver"] = parts[1].split()[0].strip()
        except Exception:
            pass

        return info

    def _detect_configured_providers(self) -> Dict[str, str]:
        """Cek konfigurasi execution provider pada modul audio STT dan TTS."""
        providers = {
            "stt": "cpu",
            "tts": "cpu",
            "active": "CPU",
        }
        try:
            from audio.provider import get_stt_provider, get_tts_provider
            stt_p = get_stt_provider()
            tts_p = get_tts_provider()
            providers["stt"] = stt_p
            providers["tts"] = tts_p
            if stt_p == "cuda" or tts_p == "cuda":
                providers["active"] = "GPU (CUDA)"
            else:
                providers["active"] = "CPU"
        except Exception:
            pass

        return providers

    def get_gpu_load(self) -> float:
        """
        Ambil utilisasi engine 3D/compute GPU (persentase 0.0 - 100.0).
        Pada NVIDIA Jetson (Orin/Xavier), load bernilai 0..1000 pada sysfs.
        """
        for p in ["/sys/devices/platform/gpu.0/load", "/sys/devices/gpu.0/load"]:
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        val = float(f.read().strip())
                        # 0 - 1000 -> 0.0% - 100.0%
                        return max(0.0, min(100.0, val / 10.0))
                except Exception:
                    pass
        return 0.0

    def get_cpu_load(self) -> float:
        """
        Ambil utilisasi CPU (persentase 0.0 - 100.0) secara non-blocking
        menggunakan delta dari /proc/stat.
        """
        now = time.time()
        # Batasi sampling minimal 0.2 detik agar delta signifikan
        if now - self._last_cpu_time < 0.2 and self._prev_total is not None:
            return self._cached_cpu

        try:
            with open("/proc/stat", "r") as f:
                fields = [float(x) for x in f.readline().strip().split()[1:8]]
            idle = fields[3]
            total = sum(fields)

            if self._prev_idle is not None and self._prev_total is not None:
                d_idle = idle - self._prev_idle
                d_total = total - self._prev_total
                if d_total > 0:
                    pct = 100.0 * (1.0 - (d_idle / d_total))
                    self._cached_cpu = max(0.0, min(100.0, pct))

            self._prev_idle = idle
            self._prev_total = total
            self._last_cpu_time = now
        except Exception:
            pass

        return self._cached_cpu

    def get_temperatures(self) -> Dict[str, Optional[float]]:
        """Baca suhu CPU dan GPU dari /sys/class/thermal/ (dalam Celsius)."""
        temps = {"cpu": None, "gpu": None}
        base = "/sys/class/thermal"
        if not os.path.exists(base):
            return temps

        try:
            for item in sorted(os.listdir(base)):
                if not item.startswith("thermal_zone"):
                    continue
                type_path = os.path.join(base, item, "type")
                temp_path = os.path.join(base, item, "temp")
                try:
                    if os.path.exists(type_path) and os.path.exists(temp_path):
                        with open(type_path, "r") as f:
                            ztype = f.read().strip().lower()
                        with open(temp_path, "r") as f:
                            t_val = f.read().strip()
                            if t_val.isdigit():
                                deg = float(t_val) / 1000.0
                                if "gpu" in ztype and temps["gpu"] is None:
                                    temps["gpu"] = deg
                                elif "cpu" in ztype and temps["cpu"] is None:
                                    temps["cpu"] = deg
                except Exception:
                    continue
        except Exception:
            pass
        return temps

    def get_memory_info(self) -> Dict[str, Any]:
        """Ambil kapasitas RAM terpakai dan total dalam Megabytes."""
        mem = {"total_mb": 0, "used_mb": 0, "pct": 0.0}
        try:
            with open("/proc/meminfo", "r") as f:
                info = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v = parts[1].strip().split()[0]
                        if v.isdigit():
                            info[k] = int(v)
                total = info.get("MemTotal", 0) // 1024
                avail = info.get("MemAvailable", 0) // 1024
                used = max(0, total - avail)
                pct = (used / total * 100.0) if total > 0 else 0.0
                mem = {"total_mb": total, "used_mb": used, "pct": pct}
        except Exception:
            pass
        return mem

    def get_snapshot(self) -> Dict[str, Any]:
        """Ambil ringkasan seluruh telemetri hardware saat ini."""
        gpu_pct = self.get_gpu_load()
        cpu_pct = self.get_cpu_load()
        temps = self.get_temperatures()
        mem = self.get_memory_info()

        # Update dynamic providers
        self.providers = self._detect_configured_providers()
        is_cuda = self.providers["active"] == "GPU (CUDA)"

        return {
            "device_model": self.device_model,
            "cuda_available": self.cuda_info["available"],
            "cuda_version": self.cuda_info["version"],
            "stt_provider": self.providers["stt"],
            "tts_provider": self.providers["tts"],
            "active_accelerator": self.providers["active"],
            "is_cuda": is_cuda,
            "gpu_load": gpu_pct,
            "cpu_load": cpu_pct,
            "gpu_temp": temps["gpu"],
            "cpu_temp": temps["cpu"],
            "ram_used_mb": mem["used_mb"],
            "ram_total_mb": mem["total_mb"],
            "ram_pct": mem["pct"],
            "is_gpu_active": gpu_pct > 1.0,
        }


# Global singleton instance
hw_monitor = HardwareMonitor()


def format_ascii_bar(pct: float, width: int = 24, fill_char: str = "█", empty_char: str = "░") -> str:
    """Format persentase menjadi visual bar teks yang rapi."""
    pct = max(0.0, min(100.0, pct))
    filled_len = int(round(width * pct / 100.0))
    bar = fill_char * filled_len + empty_char * (width - filled_len)
    return bar


def run_live_monitor_cli(interval: float = 0.8):
    """
    Dashboard interaktif di terminal untuk memantau aktivitas GPU vs CPU secara langsung.
    Tekan Ctrl+C untuk berhenti.
    """
    CLR_CYAN = "\033[1;36m"
    CLR_BLUE = "\033[1;34m"
    CLR_GREEN = "\033[1;32m"
    CLR_YELLOW = "\033[1;33m"
    CLR_MAGENTA = "\033[1;35m"
    CLR_RED = "\033[1;31m"
    CLR_WHITE = "\033[1;37m"
    CLR_MUTED = "\033[38;5;244m"
    CLR_BOLD = "\033[1m"
    CLR_RESET = "\033[0m"

    print("\n" + CLR_CYAN + "⏳ Memulai Hardware Compute Monitor..." + CLR_RESET)
    time.sleep(0.3)

    try:
        while True:
            snap = hw_monitor.get_snapshot()

            gpu_bar = format_ascii_bar(snap["gpu_load"], width=26)
            cpu_bar = format_ascii_bar(snap["cpu_load"], width=26)
            ram_bar = format_ascii_bar(snap["ram_pct"], width=26)

            gpu_color = CLR_GREEN if snap["gpu_load"] > 0.5 else CLR_MUTED
            gpu_status = f"{CLR_GREEN}ACTIVE (CUDA INFERENCE){CLR_RESET}" if snap["gpu_load"] > 2.0 else f"{CLR_MUTED}IDLE / STANDBY{CLR_RESET}"

            gpu_temp_str = f"({snap['gpu_temp']:.1f}°C)" if snap["gpu_temp"] else ""
            cpu_temp_str = f"({snap['cpu_temp']:.1f}°C)" if snap["cpu_temp"] else ""

            # Bersihkan terminal dan tampilkan kartu dashboard
            sys.stdout.write("\033[H\033[J")
            accel_label = f"● {snap['active_accelerator']} (CUDA {snap['cuda_version'] or '12.6'})" if snap['is_cuda'] else f"● {snap['active_accelerator']} (Standard)"
            sys.stdout.write(f"""{CLR_CYAN}╭────────────────────────────────────────────────────────────────────────╮
│   {CLR_WHITE}⚡  L I V E   H A R D W A R E   C O M P U T E   M O N I T O R{CLR_CYAN}        │
│   {CLR_MUTED}Real-time Accelerator Telemetry (GPU vs CPU Profile){CLR_CYAN}                 │
├────────────────────────────────────────────────────────────────────────┤
│  {CLR_WHITE}Board / Model :{CLR_RESET} {snap['device_model'][:50]:<53}{CLR_CYAN}│
│  {CLR_WHITE}Akselerator   :{CLR_RESET} {CLR_GREEN}{accel_label:<53}{CLR_CYAN}│
│  {CLR_WHITE}Engine STT    :{CLR_RESET} Sherpa-ONNX Zipformer [{CLR_GREEN}{snap['stt_provider'].upper()}{CLR_RESET}]{'':<26}{CLR_CYAN}│
│  {CLR_WHITE}Engine TTS    :{CLR_RESET} Sherpa-ONNX Piper VITS [{CLR_GREEN}{snap['tts_provider'].upper()}{CLR_RESET}]{'':<26}{CLR_CYAN}│
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  {CLR_WHITE}GPU (GR3D Core):{CLR_RESET}                                                        │
│  {gpu_color}[{gpu_bar}]{CLR_RESET}  {CLR_BOLD}{snap['gpu_load']:5.1f}%{CLR_RESET} {gpu_temp_str:<9} {gpu_status:<22}   {CLR_CYAN}│
│                                                                        │
│  {CLR_WHITE}CPU Multi-Core :{CLR_RESET}                                                        │
│  {CLR_BLUE}[{cpu_bar}]{CLR_RESET}  {CLR_BOLD}{snap['cpu_load']:5.1f}%{CLR_RESET} {cpu_temp_str:<9}                            {CLR_CYAN}│
│                                                                        │
│  {CLR_WHITE}RAM Sistem     :{CLR_RESET}                                                        │
│  {CLR_YELLOW}[{ram_bar}]{CLR_RESET}  {CLR_BOLD}{snap['ram_pct']:5.1f}%{CLR_RESET} ({snap['ram_used_mb']} MB / {snap['ram_total_mb']} MB)              {CLR_CYAN}│
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│  {CLR_MUTED}Petunjuk: Uji dengan berbicara di mikrofon atau putar audio TTS untuk{CLR_CYAN} │
│  {CLR_MUTED}melihat lonjakan utilisasi GPU (GR3D) secara real-time.{CLR_CYAN}                 │
│  {CLR_WHITE}[ Tekan Ctrl+C untuk kembali ke menu sebelumnya ]{CLR_CYAN}                     │
╰────────────────────────────────────────────────────────────────────────╯{CLR_RESET}\n""")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n{CLR_GREEN}✓ Monitoring dihentikan.{CLR_RESET}\n")
