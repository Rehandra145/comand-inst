"""
Slot Extractor — Interface + Rule-Based Implementation.

Interface (BaseSlotExtractor) dipisahkan dari implementasi
agar nantinya mudah diganti dengan ML slot tagging model.

Optimized: semua sorting, regex compile, dan lookup table
dilakukan SEKALI di __init__, bukan per-call.
"""

import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from typing import Dict, Any, Tuple, List

from nlu.entities import (
    LOCATIONS,
    LOCATION_ALIASES,
    PERSONS,
    DIRECTION_MAP,
    NUMBER_WORDS,
)
from nlu.room_validator import VALID_ROOMS


# =========================
# INTERFACE (ABC)
# =========================

class BaseSlotExtractor(ABC):
    """
    Abstract interface untuk slot extraction.
    Implementasi bisa rule-based atau ML-based slot tagging.
    """

    @abstractmethod
    def extract(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract slots dari text berdasarkan intent."""
        pass


# =========================
# CONSTANTS (computed once at module load)
# =========================

# Number words (0-9) for room number parsing
_ANGKA = {
    "nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4,
    "lima": 5, "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9,
}

# Compound words — "se-" prefix
_COMPOUND_WORDS = {"sepuluh": 10, "sebelas": 11, "seratus": 100}

# Modifier keywords
_MODIFIERS = frozenset({"ratus", "puluh", "belas"})

# All number-related words (for fast lookup)
_ALL_NUM_WORDS = frozenset(
    set(_ANGKA.keys()) | _MODIFIERS | set(_COMPOUND_WORDS.keys())
)

# Build location originals preserving uppercase abbreviations from LOCATIONS
_LOCATION_ORIGINALS = {}
for loc in LOCATIONS:
    _LOCATION_ORIGINALS[loc.lower()] = loc

for r in VALID_ROOMS:
    if r.lower() not in _LOCATION_ORIGINALS:
        _LOCATION_ORIGINALS[r.lower()] = r

_ALL_LOCATIONS = list(_LOCATION_ORIGINALS.values())

# Pre-sorted locations (longest first) — computed once
_SORTED_LOCATIONS: Tuple[str, ...] = tuple(
    sorted(_LOCATION_ORIGINALS.keys(), key=len, reverse=True)
)

# Pre-sorted aliases (longest first) — computed once
_SORTED_ALIASES: Tuple[Tuple[str, str], ...] = tuple(
    sorted(((k.lower(), v) for k, v in LOCATION_ALIASES.items()), key=lambda x: len(x[0]), reverse=True)
)

# Non-numeric locations for fuzzy matching (prevents "ruang" from matching "ruang 1")
_NAMED_LOCATIONS: Tuple[str, ...] = tuple(
    loc for loc in _SORTED_LOCATIONS
    if not re.search(r'\d+', loc)
)

# Base names without prefix (e.g. "radiologi" -> "ruang radiologi", "melati" -> "ruang melati")
_NAMED_BASE_NAMES: Tuple[Tuple[str, str], ...] = tuple(
    (loc.lower().replace("ruang ", "").replace("ruangan ", "").replace("poli ", "").strip(), loc)
    for loc in _ALL_LOCATIONS
    if loc.lower().startswith(("ruang ", "ruangan ", "poli "))
    and not re.search(r'\d+', loc)
    and loc.lower().replace("ruang ", "").replace("ruangan ", "").replace("poli ", "").strip() not in _ALL_NUM_WORDS
)

# Words to never consider as standalone location candidates in fuzzy matching
_GENERIC_PREFIX_WORDS = frozenset({
    "ruang", "ruangan", "ruas", "poli", "kamar", "tempat", "bagian", "pos", "gedung", "di", "ke", "saya", "mau"
})

# Pre-sorted directions (longest first) — computed once
_SORTED_DIRECTIONS: Tuple[Tuple[str, str], ...] = tuple(
    sorted(DIRECTION_MAP.items(), key=lambda x: len(x[0]), reverse=True)
)

# Pre-sorted persons (longest first) — computed once
_SORTED_PERSONS: Tuple[str, ...] = tuple(
    sorted((p.lower() for p in PERSONS), key=len, reverse=True)
)

# Original person names for output (maintain casing)
_PERSON_ORIGINALS = {p.lower(): p for p in PERSONS}

# Pre-compiled regex patterns (support phonetic STT errors like "ruas" / "ruan" for "ruang")
_RE_ROOM_PREFIX = re.compile(r'\b(ruang(?:an)?|ruas|ruan)\s+(.+)')
_RE_ROOM_DIGITS = re.compile(r'^(\d+)\b')
_RE_DISTANCE_NUM = re.compile(r'(\d+(?:\.\d+)?)\s*(meter|langkah|m)\b')

# Pre-compiled distance patterns for word numbers
_RE_DISTANCE_WORDS: Tuple[Tuple[str, int | float, re.Pattern], ...] = tuple(
    (word, number, re.compile(rf'\b{word}\s+(meter|langkah|m)\b'))
    for word, number in NUMBER_WORDS.items()
)


# =========================
# RULE-BASED IMPLEMENTATION
# =========================

class RuleBasedSlotExtractor(BaseSlotExtractor):
    """
    Rule-based slot extractor menggunakan pattern matching.
    Sederhana, deterministic, mudah di-debug.

    Semua data yang perlu sorting/compile sudah di-precompute
    di module level — tidak ada overhead per-call.
    """

    def extract(self, text: str, intent: str) -> Dict[str, Any]:
        text_lower = text.lower().strip()
        slots: Dict[str, Any] = {}

        if intent == "NAVIGATE":
            slots.update(self._extract_location(text_lower))

        elif intent == "MOVE":
            slots.update(self._extract_direction(text_lower))
            slots.update(self._extract_distance(text_lower))

        elif intent == "FOLLOW":
            slots.update(self._extract_person(text_lower))

        elif intent == "GO_BACK":
            slots.update(self._extract_location(text_lower))

        # STOP, WAIT, UNKNOWN — no slots needed

        return slots

    def _extract_location(self, text: str) -> Dict[str, str]:
        """Extract lokasi dari text."""

        # 1. Try dynamic room number extraction first
        room_result = self._extract_room_number(text)
        if room_result:
            return room_result

        # 2. Fall back to pre-sorted static location list
        for location in _SORTED_LOCATIONS:
            if location in text:
                original = _LOCATION_ORIGINALS[location]
                normalized = LOCATION_ALIASES.get(location, original)
                return {"location": normalized}

        # 3. Try pre-sorted alias keys directly
        for alias_lower, standard_name in _SORTED_ALIASES:
            if alias_lower in text:
                return {"location": standard_name}

        # 4. Fallback: Fuzzy matching against known locations and aliases
        clean_text = text
        for prefix in (
            "antar saya ke ", "antar ke ", "bawa saya ke ", "bawa ke ",
            "bahwa saya ke ", "bahwa ke ", "tolong ke ", "ke ruang ",
            "ke ruangan ", "ke ", "menuju "
        ):
            if prefix in clean_text:
                clean_text = clean_text.split(prefix, 1)[1].strip()
                break

        candidates = [clean_text] + clean_text.split()
        best_match = None
        best_ratio = 0.0

        for cand in candidates:
            cand_clean = cand.strip().lower()
            if len(cand_clean) < 3 or cand_clean in _GENERIC_PREFIX_WORDS:
                continue

            for alias_lower, standard_name in _SORTED_ALIASES:
                ratio = SequenceMatcher(None, cand_clean, alias_lower).ratio()
                if ratio >= 0.78 and ratio > best_ratio:
                    best_ratio = ratio
                    best_match = standard_name

            for location in _NAMED_LOCATIONS:
                ratio = SequenceMatcher(None, cand_clean, location).ratio()
                if ratio >= 0.78 and ratio > best_ratio:
                    best_ratio = ratio
                    original = _LOCATION_ORIGINALS[location]
                    best_match = LOCATION_ALIASES.get(location, original)

            for base_name, original_loc in _NAMED_BASE_NAMES:
                ratio = SequenceMatcher(None, cand_clean, base_name).ratio()
                if ratio >= 0.78 and ratio > best_ratio:
                    best_ratio = ratio
                    best_match = LOCATION_ALIASES.get(base_name, original_loc)

        if best_match and best_ratio >= 0.78:
            return {"location": best_match}

        return {}

    def _extract_room_number(self, text: str) -> Dict[str, str]:
        """
        Extract nomor ruangan dari kata angka berurutan.
        Berdasarkan logic parser.cpp — diperluas:

        Digit mode:  "satu nol empat" → "ruang 104"
        Compound:    "dua ratus tiga" → "ruang 203"
        """

        match = _RE_ROOM_PREFIX.search(text)
        if not match:
            return {}

        rest = match.group(2).strip()

        # Case 1: Already numeric
        num_match = _RE_ROOM_DIGITS.match(rest)
        if num_match:
            return {"location": f"ruang {num_match.group(1)}"}

        # Case 2: Parse number words
        words = rest.split()

        # Collect number-related words (fast frozenset lookup)
        num_words: List[str] = []
        for word in words:
            if word in _ALL_NUM_WORDS:
                num_words.append(word)
            else:
                break

        if not num_words:
            return {}

        # Check mode
        has_compound = False
        all_digits = True
        for w in num_words:
            if w not in _ANGKA:
                all_digits = False
                has_compound = True
                break

        if not has_compound:
            # Quick check for compound words if all_digits is True
            # but we still need len >= 2 for digit mode
            pass

        if all_digits and len(num_words) >= 1:
            # Digit mode: "satu" → 1, "dua nol tiga" → 203
            result = 0
            for w in num_words:
                result = result * 10 + _ANGKA[w]
            return {"location": f"ruang {result}"}

        elif has_compound:
            # Compound mode
            result = 0
            current = 0

            for w in num_words:
                if w in _COMPOUND_WORDS:
                    result += _COMPOUND_WORDS[w]
                    current = 0
                elif w in _ANGKA:
                    current = _ANGKA[w]
                elif w == "ratus":
                    if current == 0:
                        current = 1
                    result += current * 100
                    current = 0
                elif w == "puluh":
                    result += current * 10
                    current = 0
                elif w == "belas":
                    result += current + 10
                    current = 0

            result += current
            if result > 0:
                return {"location": f"ruang {result}"}

        return {}

    def _extract_direction(self, text: str) -> Dict[str, str]:
        """Extract arah dari text."""

        for direction_id, direction_en in _SORTED_DIRECTIONS:
            if direction_id in text:
                return {"direction": direction_en}

        return {}

    def _extract_distance(self, text: str) -> Dict[str, Any]:
        """Extract jarak (angka + satuan) dari text."""

        # Try numeric digits first (pre-compiled regex)
        match = _RE_DISTANCE_NUM.search(text)
        if match:
            distance = float(match.group(1))
            unit = match.group(2)
            if unit == "m":
                unit = "meter"
            return {
                "distance": int(distance) if distance == int(distance) else distance,
                "unit": unit,
            }

        # Try word numbers (pre-compiled patterns)
        for word, number, pattern in _RE_DISTANCE_WORDS:
            match = pattern.search(text)
            if match:
                unit = match.group(1)
                if unit == "m":
                    unit = "meter"
                return {
                    "distance": number,
                    "unit": unit,
                }

        return {}

    def _extract_person(self, text: str) -> Dict[str, str]:
        """Extract person/orang dari text."""

        for person_lower in _SORTED_PERSONS:
            if person_lower in text:
                return {"person": _PERSON_ORIGINALS[person_lower]}

        return {}
