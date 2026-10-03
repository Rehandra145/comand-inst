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
