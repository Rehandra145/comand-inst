"""
Room Validator — Verifikasi apakah lokasi yang di-parse ada di daftar ruangan valid.

Memuat daftar ruangan dari valid_rooms.json.
Untuk menambah/hapus ruangan valid, cukup edit file JSON tersebut.
"""

import json
import logging
import os
from typing import Optional


logger = logging.getLogger(__name__)

# Path ke JSON file
_JSON_PATH = os.path.join(os.path.dirname(__file__), "valid_rooms.json")


def _load_valid_rooms() -> set:
    """Load daftar ruangan valid dari JSON file."""
    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Normalize ke lowercase untuk matching
        rooms = {room.lower() for room in data.get("rooms", [])}
        logger.info(f"[ROOM_VALIDATOR] Loaded {len(rooms)} valid rooms")
        return rooms
    except FileNotFoundError:
        logger.warning(
            f"[ROOM_VALIDATOR] {_JSON_PATH} not found, "
            "semua lokasi akan dianggap valid"
        )
        return set()
    except json.JSONDecodeError as e:
        logger.error(f"[ROOM_VALIDATOR] JSON parse error: {e}")
        return set()


# Load sekali saat module di-import
VALID_ROOMS = _load_valid_rooms()


def is_valid_room(location: str) -> bool:
    """
    Cek apakah lokasi ada di daftar ruangan valid.

    Args:
        location: nama lokasi hasil parsing (e.g. "ruang 318")

    Returns:
        True jika valid, False jika tidak ditemukan
    """
    if not VALID_ROOMS:
        # Jika JSON kosong/tidak ada, anggap semua valid
        # (supaya tidak blocking saat belum dikonfigurasi)
        return True

    return location.lower() in VALID_ROOMS


def get_valid_rooms() -> list:
    """Return sorted list of valid rooms (untuk debugging/display)."""
    return sorted(VALID_ROOMS)
