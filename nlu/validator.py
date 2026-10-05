"""
Command Validator — Validasi apakah structured command sudah lengkap
dan lokasi valid.
"""

import logging
from typing import Dict, Any, Tuple

from nlu.room_validator import is_valid_room


logger = logging.getLogger(__name__)


# =========================
# SLOT REQUIREMENTS
# =========================

_EMPTY_TUPLE: Tuple[str, ...] = ()

# Required slots per intent (tuple for immutable zero-copy access)
REQUIRED_SLOTS: Dict[str, Tuple[str, ...]] = {
    "NAVIGATE": ("location",),
    "MOVE": ("direction",),
    "STOP": _EMPTY_TUPLE,
    "GO_BACK": _EMPTY_TUPLE,
    "WAIT": _EMPTY_TUPLE,
    "FOLLOW": _EMPTY_TUPLE,
    "UNKNOWN": _EMPTY_TUPLE,
}

# Optional slots per intent (for reference)
OPTIONAL_SLOTS: Dict[str, Tuple[str, ...]] = {
    "NAVIGATE": _EMPTY_TUPLE,
    "MOVE": ("distance", "unit"),
    "STOP": _EMPTY_TUPLE,
    "GO_BACK": ("location",),
    "WAIT": ("duration",),
    "FOLLOW": ("person",),
    "UNKNOWN": _EMPTY_TUPLE,
}


# =========================
# VALIDATOR
# =========================

def validate_command(command: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validasi command apakah sudah lengkap dan lokasi valid.

    Returns command dengan tambahan:
        - "status": "VALID", "INCOMPLETE", "INVALID_LOCATION", atau "UNKNOWN_INTENT"
        - "missing_slots": list slot yang belum terisi (jika INCOMPLETE)
    """

    intent = command.get("intent", "UNKNOWN")
    slots = command.get("slots", {})

    # UNKNOWN intent — cannot validate
    if intent == "UNKNOWN":
        return {
            **command,
            "status": "UNKNOWN_INTENT",
            "missing_slots": [],
        }

    # Check required slots
    required = REQUIRED_SLOTS.get(intent, _EMPTY_TUPLE)
    missing = [
        slot for slot in required
        if slot not in slots or slots[slot] is None
    ]

    if missing:
        return {
            **command,
            "status": "INCOMPLETE",
            "missing_slots": missing,
        }

    # Room validation: cek apakah lokasi ada di daftar valid
    if intent == "NAVIGATE":
        location = slots.get("location", "")
        # Normalisasi otomatis jika berupa alias atau kata angka (e.g. "ruang satu" -> "ruang 1")
        from nlu.room_validator import normalize_room_name
        norm_location = normalize_room_name(location)
        if norm_location:
            location = norm_location
            slots["location"] = location

        if location and not is_valid_room(location):
            logger.warning(
                f"[VALIDATION] Lokasi tidak valid: \"{location}\""
            )
            return {
                **command,
                "status": "INVALID_LOCATION",
                "missing_slots": [],
            }

    return {
        **command,
        "status": "VALID",
        "missing_slots": [],
    }
