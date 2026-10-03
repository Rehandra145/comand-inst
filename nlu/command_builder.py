"""
Command Builder — Menggabungkan intent + slots menjadi structured command.
"""

from typing import Dict, Any

from nlu.intent_classifier import IntentResult


def build_command(intent_result: IntentResult, slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build structured command dari intent dan slots.

    Returns:
        {
            "intent": "NAVIGATE",
            "slots": {"location": "ruang ICU"},
            "confidence": 1.0,
        }
    """

    return {
        "intent": intent_result.intent,
        "slots": slots,
        "confidence": intent_result.confidence,
    }
