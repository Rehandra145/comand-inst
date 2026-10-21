"""
ROS 2 Adapter — Interface + Mock Implementation.

NLU tidak bergantung langsung pada ROS 2 implementation detail.
Mock adapter digunakan selama ROS 2 belum tersedia.
"""

import json
import logging
import os
import socket
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple


logger = logging.getLogger(__name__)

CONFIG_PATH = Path("ros2_config.json")
DEFAULT_HOST = "10.107.245.221"
DEFAULT_PORT = 5000


def load_ros2_config() -> Tuple[str, int]:
    """Baca host dan port ROS 2 dari environment variable / ros2_config.json / default."""
    host = os.environ.get("ROS2_HOST")
    port = os.environ.get("ROS2_PORT")

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not host and "host" in data:
                    host = str(data["host"]).strip()
                if not port and "port" in data:
                    port = data["port"]
        except Exception as e:
            logger.warning(f"Gagal membaca ros2_config.json: {e}")

    host = host or DEFAULT_HOST
    try:
        port = int(port) if port else DEFAULT_PORT
    except (ValueError, TypeError):
        port = DEFAULT_PORT

    return host, port


def save_ros2_config(host: str, port: int) -> bool:
    """Simpan host dan port ROS 2 ke file ros2_config.json."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"host": str(host).strip(), "port": int(port)}, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Gagal menyimpan ros2_config.json: {e}")
        return False


# =========================
# INTERFACE (ABC)
# =========================

class BaseROS2Adapter(ABC):
    """
    Abstract interface untuk ROS 2 adapter.
    Implementasi bisa mock atau real ROS 2 publisher.
    """

    @abstractmethod
    def send_command(self, command: Dict[str, Any]) -> bool:
        """
        Kirim structured command ke ROS 2.
        Returns True jika berhasil, False jika gagal.
        """
        pass

    def ping(self, timeout: float = 1.5) -> Tuple[bool, str]:
        """Uji konektivitas ke Node ROS 2."""
        return True, "Adapter siap"

    def set_target(self, host: str, port: int):
        """Update host dan port tujuan."""
        pass


# =========================
# MOCK IMPLEMENTATION
# =========================

class MockROS2Adapter(BaseROS2Adapter):
    """
    Mock ROS 2 adapter untuk development/testing.
    Mencetak command ke log.
    """

    def ping(self, timeout: float = 1.5) -> Tuple[bool, str]:
        return True, "Mock ROS 2 Adapter siap (Simulasi)"

    def send_command(self, command: Dict[str, Any]) -> bool:
        intent = command.get("intent", "UNKNOWN")
        status = command.get("status", "UNKNOWN")
        slots = command.get("slots", {})
        command["ros2_target"] = "MOCK_ROS2"
        command["ros2_payload"] = json.dumps(command, ensure_ascii=False)

        if status != "VALID":
            logger.warning(
                f"[ROS2] Command tidak dikirim (status={status})"
            )
            command["ros2_sent"] = False
            command["ros2_error"] = f"Perintah belum valid (status={status})"
            return False

        command["ros2_sent"] = True
        command["ros2_error"] = None

        logger.info("[ROS2] Mengirim command ke robot:")
        logger.info(f"[ROS2]   intent = {intent}")

        for key, value in slots.items():
            logger.info(f"[ROS2]   {key} = {value}")

        # Robot response based on intent (clean terminal text)
        if intent == "NAVIGATE":
            location = slots.get("location", "?")
            logger.info(f"[ROS2] [ROBOT] menuju {location}")

        elif intent == "MOVE":
            direction = slots.get("direction", "?")
            distance = slots.get("distance", "")
            unit = slots.get("unit", "")
            dist_str = f" {distance} {unit}" if distance else ""
            logger.info(f"[ROS2] [ROBOT] bergerak {direction}{dist_str}")

        elif intent == "STOP":
            logger.info("[ROS2] [ROBOT] berhenti")

        elif intent == "GO_BACK":
            logger.info("[ROS2] [ROBOT] kembali")

        elif intent == "WAIT":
            logger.info("[ROS2] [ROBOT] menunggu")

        elif intent == "FOLLOW":
            person = slots.get("person", "user")
            logger.info(f"[ROS2] [ROBOT] mengikuti {person}")

        elif intent == "EMERGENCY":
            loc = slots.get("location", "posisi saat ini")
            logger.info(f"[ROS2] [ROBOT] DARURAT: prioritaskan menuju {loc}")

        elif intent == "STATUS":
            logger.info("[ROS2] [ROBOT] cek status / telemetri")

        elif intent == "GREETING":
            logger.info("[ROS2] [ROBOT] membalas sapaan")

        elif intent == "THANK":
            logger.info("[ROS2] [ROBOT] membalas ucapan terima kasih")

        return True

# =========================
# SOCKET (TCP) IMPLEMENTATION
# =========================

class SocketROS2Adapter(BaseROS2Adapter):
    """
    Adapter yang mengirim perintah dalam bentuk JSON string 
    melalui TCP Socket ke Node ROS 2 di robot.
    Cocok untuk setup lintas OS (Windows -> Ubuntu).
    """

    def __init__(self, host: str = None, port: int = None):
        cfg_host, cfg_port = load_ros2_config()
        self.host = str(host or cfg_host).strip()
        try:
            self.port = int(port if port is not None else cfg_port)
        except (ValueError, TypeError):
            self.port = DEFAULT_PORT
        logger.info(f"Initialized SocketROS2Adapter targeting {self.host}:{self.port}")

    def set_target(self, host: str, port: int):
        """Update host dan port TCP tujuan saat runtime."""
        self.host = str(host).strip()
        try:
            self.port = int(port)
        except (ValueError, TypeError):
            self.port = DEFAULT_PORT
        logger.info(f"SocketROS2Adapter target diperbarui ke {self.host}:{self.port}")

    def ping(self, timeout: float = 1.5) -> Tuple[bool, str]:
        """Uji konektivitas cepat ke host & port Node ROS 2."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((self.host, self.port))
            return True, f"Terhubung ke Node ROS 2 ({self.host}:{self.port})"
        except Exception as e:
            return False, f"Gagal terhubung ({e})"

    def send_command(self, command: Dict[str, Any]) -> bool:
        status = command.get("status", "UNKNOWN")
        command["ros2_target"] = f"{self.host}:{self.port}"
        
        if status != "VALID":
            logger.warning(f"[Socket] Command tidak dikirim (status={status})")
            command["ros2_sent"] = False
            command["ros2_error"] = f"Perintah belum valid (status={status})"
            return False

        # Membuat koneksi socket baru setiap kali mengirim
        # Pendekatan ini aman untuk menjaga koneksi tidak hang jika terputus
        payload = json.dumps(command, ensure_ascii=False)
        command["ros2_payload"] = payload
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)  # timeout 2 detik
                s.connect((self.host, self.port))
                s.sendall(payload.encode('utf-8'))
            logger.info(f"[Socket] Berhasil mengirim: {payload}")
            command["ros2_sent"] = True
            command["ros2_error"] = None
            return True
        except Exception as e:
            err_msg = str(e)
            logger.error(f"[Socket] Gagal mengirim command ke {self.host}:{self.port} - Error: {err_msg}")
            command["ros2_sent"] = False
            command["ros2_error"] = err_msg
            return False
