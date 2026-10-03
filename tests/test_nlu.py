"""
Test suite untuk NLU Pipeline.

Menguji:
1. Intent classification (berbagai variasi bahasa Indonesia)
2. Slot extraction (location, direction, distance, person)
3. Command building
4. Command validation
5. End-to-end pipeline
"""

import sys
import os
import json

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlu.intent_classifier import KeywordIntentClassifier, Intent
from nlu.slot_extractor import RuleBasedSlotExtractor
from nlu.command_builder import build_command
from nlu.validator import validate_command
from nlu.pipeline import NLUPipeline


# =========================
# HELPERS
# =========================

passed = 0
failed = 0


def assert_equal(test_name, expected, actual):
    global passed, failed
    if expected == actual:
        passed += 1
        print(f"  [PASS] {test_name}")
    else:
        failed += 1
        print(f"  [FAIL] {test_name}")
        print(f"     Expected: {expected}")
        print(f"     Actual:   {actual}")


def assert_in(test_name, expected_key, actual_dict):
    global passed, failed
    if expected_key in actual_dict:
        passed += 1
        print(f"  [PASS] {test_name}")
    else:
        failed += 1
        print(f"  [FAIL] {test_name}")
        print(f"     Expected key '{expected_key}' in {actual_dict}")


# =========================
# 1. INTENT CLASSIFICATION
# =========================

def test_intent_classification():
    print("\n" + "=" * 50)
    print("TEST: Intent Classification")
    print("=" * 50)

    classifier = KeywordIntentClassifier()

    # --- NAVIGATE ---
    navigate_cases = [
        "antar saya ke ruang ICU",
        "tolong bawa saya ke ruang ICU",
        "saya mau pergi ke ruang ICU",
        "bisa antar saya menuju ruang ICU?",
        "tolong pergi ke ICU",
        "pergi ke ruang radiologi",
        "antar ke poli anak",
        "saya mau ke lobi",
        "bawa saya ke ruang operasi",
        "tolong antar ke ruang 1",
        "saya ingin pergi ke ruangan dua",
        "menuju ruang IGD",
        "ke ruang farmasi",
    ]

    print("\n  NAVIGATE:")
    for text in navigate_cases:
        result = classifier.classify(text)
        assert_equal(
            f'"{text}" → {result.intent}',
            Intent.NAVIGATE,
            result.intent,
        )

    # --- MOVE ---
    move_cases = [
        "maju dua meter",
        "mundur satu meter",
        "belok kiri",
        "belok kanan",
        "maju",
        "ke depan",
        "ke belakang",
    ]

    print("\n  MOVE:")
    for text in move_cases:
        result = classifier.classify(text)
        assert_equal(
            f'"{text}" → {result.intent}',
            Intent.MOVE,
            result.intent,
        )

    # --- STOP ---
    stop_cases = [
        "berhenti",
        "stop",
        "berhenti sekarang",
        "hentikan",
    ]

    print("\n  STOP:")
    for text in stop_cases:
        result = classifier.classify(text)
        assert_equal(
            f'"{text}" → {result.intent}',
            Intent.STOP,
            result.intent,
        )

    # --- GO_BACK ---
    go_back_cases = [
        "kembali",
        "balik",
        "putar balik",
    ]

    print("\n  GO_BACK:")
    for text in go_back_cases:
        result = classifier.classify(text)
        assert_equal(
            f'"{text}" → {result.intent}',
            Intent.GO_BACK,
            result.intent,
        )

    # --- WAIT ---
    wait_cases = [
        "tunggu di sini",
        "tunggu sebentar",
        "tunggu dulu",
    ]

    print("\n  WAIT:")
    for text in wait_cases:
        result = classifier.classify(text)
        assert_equal(
            f'"{text}" → {result.intent}',
            Intent.WAIT,
            result.intent,
        )

    # --- FOLLOW ---
    follow_cases = [
        "ikuti saya",
        "ikut saya",
        "ikuti dokter",
    ]

    print("\n  FOLLOW:")
    for text in follow_cases:
        result = classifier.classify(text)
        assert_equal(
            f'"{text}" → {result.intent}',
            Intent.FOLLOW,
            result.intent,
        )

    # --- UNKNOWN ---
    unknown_cases = [
        "berapa jam sekarang",
        "bagaimana cuaca hari ini",
        "tolong nyalakan lampu",
        "siapa nama kamu",
        "saya lapar",
        "",
    ]

    print("\n  UNKNOWN:")
    for text in unknown_cases:
        result = classifier.classify(text)
        assert_equal(
            f'"{text}" → {result.intent}',
            Intent.UNKNOWN,
            result.intent,
        )


# =========================
# 2. SLOT EXTRACTION
# =========================

def test_slot_extraction():
    print("\n" + "=" * 50)
    print("TEST: Slot Extraction")
    print("=" * 50)

    extractor = RuleBasedSlotExtractor()

    # --- NAVIGATE: location ---
    print("\n  NAVIGATE locations:")

    result = extractor.extract("tolong antar saya ke ruang ICU", "NAVIGATE")
    assert_equal(
        '"ruang ICU" -> location',
        "ruang ICU",
        result.get("location"),
    )

    result = extractor.extract("pergi ke ruang radiologi", "NAVIGATE")
    assert_equal(
        '"ruang radiologi" -> location',
        "ruang radiologi",
        result.get("location"),
    )

    result = extractor.extract("antar ke poli anak", "NAVIGATE")
    assert_equal(
        '"poli anak" -> location',
        "poli anak",
        result.get("location"),
    )

    result = extractor.extract("bawa saya ke lobi", "NAVIGATE")
    assert_equal(
        '"lobi" -> location',
        "lobi",
        result.get("location"),
    )

    result = extractor.extract("pergi ke ruang 1", "NAVIGATE")
    assert_equal(
        '"ruang 1" -> location',
        "ruang 1",
        result.get("location"),
    )

    result = extractor.extract("menuju ruang dua", "NAVIGATE")
    assert_equal(
        '"ruang dua" -> location (alias -> ruang 2)',
        "ruang 2",
        result.get("location"),
    )

    # --- MOVE: direction + distance ---
    print("\n  MOVE direction/distance:")

    result = extractor.extract("maju dua meter", "MOVE")
    assert_equal(
        '"maju" -> direction = forward',
        "forward",
        result.get("direction"),
    )
    assert_equal(
        '"dua meter" -> distance = 2',
        2,
        result.get("distance"),
    )
    assert_equal(
        '"dua meter" -> unit = meter',
        "meter",
        result.get("unit"),
    )

    result = extractor.extract("mundur satu meter", "MOVE")
    assert_equal(
        '"mundur" -> direction = backward',
        "backward",
        result.get("direction"),
    )
    assert_equal(
        '"satu meter" -> distance = 1',
        1,
        result.get("distance"),
    )

    result = extractor.extract("belok kiri", "MOVE")
    assert_equal(
        '"belok kiri" -> direction = left',
        "left",
        result.get("direction"),
    )

    result = extractor.extract("belok kanan", "MOVE")
    assert_equal(
        '"belok kanan" -> direction = right',
        "right",
        result.get("direction"),
    )

    # --- FOLLOW: person ---
    print("\n  FOLLOW person:")

    result = extractor.extract("ikuti dokter", "FOLLOW")
    assert_equal(
        '"ikuti dokter" -> person = dokter',
        "dokter",
        result.get("person"),
    )

    result = extractor.extract("ikuti perawat", "FOLLOW")
    assert_equal(
        '"ikuti perawat" -> person = perawat',
        "perawat",
        result.get("person"),
    )


# =========================
# 3. COMMAND VALIDATION
# =========================

def test_validation():
    print("\n" + "=" * 50)
    print("TEST: Command Validation")
    print("=" * 50)

    # VALID NAVIGATE
    cmd = {
        "intent": "NAVIGATE",
        "slots": {"location": "ruang ICU"},
        "confidence": 1.0,
    }
    result = validate_command(cmd)
    assert_equal(
        "NAVIGATE + location -> VALID",
        "VALID",
        result["status"],
    )

    # INCOMPLETE NAVIGATE (no location)
    cmd = {
        "intent": "NAVIGATE",
        "slots": {},
        "confidence": 1.0,
    }
    result = validate_command(cmd)
    assert_equal(
        "NAVIGATE + no location -> INCOMPLETE",
        "INCOMPLETE",
        result["status"],
    )
    assert_equal(
        "missing_slots = ['location']",
        ["location"],
        result["missing_slots"],
    )

    # VALID MOVE
    cmd = {
        "intent": "MOVE",
        "slots": {"direction": "forward", "distance": 2, "unit": "meter"},
        "confidence": 1.0,
    }
    result = validate_command(cmd)
    assert_equal(
        "MOVE + direction -> VALID",
        "VALID",
        result["status"],
    )

    # INCOMPLETE MOVE (no direction)
    cmd = {
        "intent": "MOVE",
        "slots": {"distance": 2},
        "confidence": 1.0,
    }
    result = validate_command(cmd)
    assert_equal(
        "MOVE + no direction -> INCOMPLETE",
        "INCOMPLETE",
        result["status"],
    )

    # VALID STOP (no slots needed)
    cmd = {
        "intent": "STOP",
        "slots": {},
        "confidence": 1.0,
    }
    result = validate_command(cmd)
    assert_equal(
        "STOP -> VALID",
        "VALID",
        result["status"],
    )

    # UNKNOWN
    cmd = {
        "intent": "UNKNOWN",
        "slots": {},
        "confidence": 0.0,
    }
    result = validate_command(cmd)
    assert_equal(
        "UNKNOWN -> UNKNOWN_INTENT",
        "UNKNOWN_INTENT",
        result["status"],
    )


# =========================
# 4. END-TO-END PIPELINE
# =========================

def test_pipeline_e2e():
    print("\n" + "=" * 50)
    print("TEST: End-to-End Pipeline")
    print("=" * 50)

    pipeline = NLUPipeline()

    # --- NAVIGATE ---
    print("\n  E2E NAVIGATE:")

    result = pipeline.process("tolong antar saya ke ruang ICU")
    assert_equal(
        '"tolong antar saya ke ruang ICU" -> NAVIGATE',
        "NAVIGATE",
        result["intent"],
    )
    assert_equal(
        "location = ruang ICU",
        "ruang ICU",
        result["slots"].get("location"),
    )
    assert_equal(
        "status = VALID",
        "VALID",
        result["status"],
    )

    result = pipeline.process("bawa saya ke poli jantung")
    assert_equal(
        '"bawa saya ke poli jantung" -> NAVIGATE',
        "NAVIGATE",
        result["intent"],
    )
    assert_equal(
        "location = poli jantung",
        "poli jantung",
        result["slots"].get("location"),
    )

    # --- MOVE ---
    print("\n  E2E MOVE:")

    result = pipeline.process("maju dua meter")
    assert_equal(
        '"maju dua meter" -> MOVE',
        "MOVE",
        result["intent"],
    )
    assert_equal(
        "direction = forward",
        "forward",
        result["slots"].get("direction"),
    )
    assert_equal(
        "distance = 2",
        2,
        result["slots"].get("distance"),
    )
    assert_equal(
        "status = VALID",
        "VALID",
        result["status"],
    )

    # --- STOP ---
    print("\n  E2E STOP:")

    result = pipeline.process("berhenti sekarang")
    assert_equal(
        '"berhenti sekarang" -> STOP',
        "STOP",
        result["intent"],
    )
    assert_equal(
        "status = VALID",
        "VALID",
        result["status"],
    )

    # --- FOLLOW ---
    print("\n  E2E FOLLOW:")

    result = pipeline.process("ikuti dokter")
    assert_equal(
        '"ikuti dokter" -> FOLLOW',
        "FOLLOW",
        result["intent"],
    )
    assert_equal(
        "person = dokter",
        "dokter",
        result["slots"].get("person"),
    )

    # --- WAIT ---
    print("\n  E2E WAIT:")

    result = pipeline.process("tunggu di sini")
    assert_equal(
        '"tunggu di sini" -> WAIT',
        "WAIT",
        result["intent"],
    )
    assert_equal(
        "status = VALID",
        "VALID",
        result["status"],
    )

    # --- GO_BACK ---
    print("\n  E2E GO_BACK:")

    result = pipeline.process("kembali")
    assert_equal(
        '"kembali" -> GO_BACK',
        "GO_BACK",
        result["intent"],
    )

    # --- UNKNOWN ---
    print("\n  E2E UNKNOWN:")

    result = pipeline.process("berapa jam sekarang")
    assert_equal(
        '"berapa jam sekarang" -> UNKNOWN',
        "UNKNOWN",
        result["intent"],
    )
    assert_equal(
        "status = UNKNOWN_INTENT",
        "UNKNOWN_INTENT",
        result["status"],
    )

    # --- INCOMPLETE (navigate tanpa lokasi yang dikenali) ---
    print("\n  E2E INCOMPLETE:")

    result = pipeline.process("antar saya ke sana")
    # Should be NAVIGATE but no known location
    assert_equal(
        '"antar saya ke sana" -> NAVIGATE',
        "NAVIGATE",
        result["intent"],
    )
    assert_equal(
        "status = INCOMPLETE (no known location)",
        "INCOMPLETE",
        result["status"],
    )

    # --- INVALID_LOCATION (lokasi tidak terdaftar di JSON) ---
    print("\n  E2E INVALID_LOCATION:")
    result = pipeline.process("antar saya ke ruang 999")
    assert_equal(
        '"antar saya ke ruang 999" -> NAVIGATE',
        "NAVIGATE",
        result["intent"],
    )
    assert_equal(
        "status = INVALID_LOCATION",
        "INVALID_LOCATION",
        result["status"],
    )

    # --- Dynamic Room Number (parser.cpp logic) ---
    print("\n  E2E DYNAMIC ROOM NUMBER:")
    result_digit = pipeline.process("antar ke ruang satu nol empat")
    assert_equal(
        '"ruang satu nol empat" -> ruang 104',
        "ruang 104",
        result_digit["slots"].get("location"),
    )
    assert_equal("status = VALID", "VALID", result_digit["status"])

    result_compound = pipeline.process("antar ke ruang tiga ratus satu")
    assert_equal(
        '"ruang tiga ratus satu" -> ruang 301',
        "ruang 301",
        result_compound["slots"].get("location"),
    )
    assert_equal("status = VALID", "VALID", result_compound["status"])

    # --- Fuzzy Matching (STT Typo / Misheard) ---
    print("\n  E2E FUZZY MATCHING (STT Typos):")
    result_fuzzy1 = pipeline.process("BAHWA KE UGD")
    assert_equal('"BAHWA KE UGD" -> NAVIGATE', "NAVIGATE", result_fuzzy1["intent"])
    assert_equal('location = ruang UGD', "ruang UGD", result_fuzzy1["slots"].get("location"))
    assert_equal("status = VALID", "VALID", result_fuzzy1["status"])

    result_fuzzy2 = pipeline.process("bawak ke ruang icuu")
    assert_equal('"bawak ke ruang icuu" -> NAVIGATE', "NAVIGATE", result_fuzzy2["intent"])
    assert_equal('location = ruang ICU', "ruang ICU", result_fuzzy2["slots"].get("location"))
    assert_equal("status = VALID", "VALID", result_fuzzy2["status"])

    result_fuzzy3 = pipeline.process("tolong anterin ke radiolgi")
    assert_equal('"tolong anterin ke radiolgi" -> NAVIGATE', "NAVIGATE", result_fuzzy3["intent"])
    assert_equal('location = ruang radiologi', "ruang radiologi", result_fuzzy3["slots"].get("location"))
    assert_equal("status = VALID", "VALID", result_fuzzy3["status"])


# =========================
# RUN ALL TESTS
# =========================

if __name__ == "__main__":
    print("=" * 50)
    print("NLU PIPELINE TEST SUITE")
    print("=" * 50)

    test_intent_classification()
    test_slot_extraction()
    test_validation()
    test_pipeline_e2e()

    print("\n" + "=" * 50)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)
