"""
ROS 2 Adapter — Interface + Mock Implementation.

NLU tidak bergantung langsung pada ROS 2 implementation detail.
Mock adapter digunakan selama ROS 2 belum tersedia.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any


logger = logging.getLogger(__name__)


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


# =========================
# MOCK IMPLEMENTATION
# =========================

class MockROS2Adapter(BaseROS2Adapter):
    """
    Mock ROS 2 adapter untuk development/testing.
    Mencetak command ke log.
    """

    def send_command(self, command: Dict[str, Any]) -> bool:
        intent = command.get("intent", "UNKNOWN")
        status = command.get("status", "UNKNOWN")
        slots = command.get("slots", {})

        if status != "VALID":
            logger.warning(
                f"[ROS2] Command tidak dikirim (status={status})"
            )
            return False

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

        return True

import socket
import json
import threading

# =========================
# SOCKET (TCP) IMPLEMENTATION
# =========================

class SocketROS2Adapter(BaseROS2Adapter):
    """
    Adapter yang mengirim perintah dalam bentuk JSON string 
    melalui TCP Socket ke Node ROS 2 di robot.
    Cocok untuk setup lintas OS (Windows -> Ubuntu).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.host = host
        self.port = port
        logger.info(f"Initialized SocketROS2Adapter targeting {self.host}:{self.port}")

    def send_command(self, command: Dict[str, Any]) -> bool:
        status = command.get("status", "UNKNOWN")
        if status != "VALID":
            logger.warning(f"[Socket] Command tidak dikirim (status={status})")
            return False

        # Membuat koneksi socket baru setiap kali mengirim
        # Pendekatan ini aman untuk menjaga koneksi tidak hang jika terputus
        payload = json.dumps(command)
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)  # timeout 2 detik
                s.connect((self.host, self.port))
                s.sendall(payload.encode('utf-8'))
            logger.info(f"[Socket] Berhasil mengirim: {payload}")
            return True
        except Exception as e:
            logger.error(f"[Socket] Gagal mengirim command ke {self.host}:{self.port} - Error: {e}")
            return False
