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


import re

# Load sekali saat module di-import
VALID_ROOMS = _load_valid_rooms()

_ANGKA_WORDS = {
    "nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4,
    "lima": 5, "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9,
}


def normalize_room_name(location: str) -> str:
    """
    Normalisasi variasi nama ruangan (e.g. 'ruang satu' -> 'ruang 1', 'ruangan dua' -> 'ruang 2',
    'ruas 1' -> 'ruang 1').
    """
    if not location:
        return location

    from nlu.entities import LOCATION_ALIASES
    loc_clean = location.strip().lower()

    if loc_clean in LOCATION_ALIASES:
        return LOCATION_ALIASES[loc_clean]

    # Ganti prefix 'ruangan' atau 'ruas' menjadi 'ruang'
    norm = re.sub(r'^(ruangan|ruas|ruan)\b', 'ruang', loc_clean).strip()
    if norm in LOCATION_ALIASES:
        return LOCATION_ALIASES[norm]

    # Cek apakah 'ruang <kata angka>' (misal: 'ruang satu', 'ruang dua nol lima')
    m = re.match(r'^ruang\s+(.+)$', norm)
    if m:
        words = m.group(1).split()
        if all(w in _ANGKA_WORDS for w in words):
            val = 0
            for w in words:
                val = val * 10 + _ANGKA_WORDS[w]
            return f"ruang {val}"

    return location


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

    loc_lower = location.lower().strip()
    if loc_lower in VALID_ROOMS:
        return True

    from nlu.entities import LOCATION_ALIASES
    if loc_lower in LOCATION_ALIASES and LOCATION_ALIASES[loc_lower].lower() in VALID_ROOMS:
        return True

    norm = normalize_room_name(loc_lower)
    if norm.lower() in VALID_ROOMS:
        return True

    return False


def get_valid_rooms() -> list:
    """Return sorted list of valid rooms (untuk debugging/display)."""
    return sorted(VALID_ROOMS)

