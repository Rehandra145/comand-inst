"""
Intent Classifier — Interface + Keyword & Fuzzy Implementation.

Interface (BaseIntentClassifier) dipisahkan dari implementasi
agar nantinya mudah diganti dengan model ML (IndoBERT, dll).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Tuple


# =========================
# INTENT ENUM
# =========================

class Intent:
    NAVIGATE = "NAVIGATE"
    MOVE = "MOVE"
    STOP = "STOP"
    GO_BACK = "GO_BACK"
    WAIT = "WAIT"
    FOLLOW = "FOLLOW"
    GREETING = "GREETING"
    THANK = "THANK"
    EMERGENCY = "EMERGENCY"
    STATUS = "STATUS"
    UNKNOWN = "UNKNOWN"


# =========================
# RESULT
# =========================

@dataclass(slots=True)
class IntentResult:
    intent: str
    confidence: float


# Pre-allocated singletons untuk hasil yang sering dipakai
_UNKNOWN_RESULT = IntentResult(intent=Intent.UNKNOWN, confidence=0.0)
_VALID_RESULTS = {
    Intent.NAVIGATE: IntentResult(intent=Intent.NAVIGATE, confidence=1.0),
    Intent.MOVE: IntentResult(intent=Intent.MOVE, confidence=1.0),
    Intent.STOP: IntentResult(intent=Intent.STOP, confidence=1.0),
    Intent.GO_BACK: IntentResult(intent=Intent.GO_BACK, confidence=1.0),
    Intent.WAIT: IntentResult(intent=Intent.WAIT, confidence=1.0),
    Intent.FOLLOW: IntentResult(intent=Intent.FOLLOW, confidence=1.0),
    Intent.GREETING: IntentResult(intent=Intent.GREETING, confidence=1.0),
    Intent.THANK: IntentResult(intent=Intent.THANK, confidence=1.0),
    Intent.EMERGENCY: IntentResult(intent=Intent.EMERGENCY, confidence=1.0),
    Intent.STATUS: IntentResult(intent=Intent.STATUS, confidence=1.0),
}


# =========================
# FUZZY TRIGGERS (CORE ACTION WORDS)
# =========================

_FUZZY_CORE_TRIGGERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (Intent.EMERGENCY, ("darurat", "emergency")),
    (Intent.STOP, ("berhenti", "hentikan")),
    (Intent.WAIT, ("tunggu", "standby")),
    (Intent.FOLLOW, ("ikuti", "follow")),
    (Intent.GO_BACK, ("kembali", "putar balik")),
    (Intent.NAVIGATE, ("antar", "antarkan", "bawakan", "menuju")),
    (Intent.MOVE, ("maju", "mundur", "belok")),
    (Intent.STATUS, ("baterai", "status")),
    (Intent.GREETING, ("halo", "assalamualaikum")),
    (Intent.THANK, ("makasih", "terimakasih")),
)

# Ambiguous words to ignore in fuzzy matching
_FUZZY_IGNORE_WORDS = frozenset({"dan", "di", "ke", "ini", "itu", "yang", "dari", "saya", "kamu", "bisa", "tolong", "mau"})


# =========================
# INTERFACE (ABC)
# =========================

class BaseIntentClassifier(ABC):
    """
    Abstract interface untuk intent classifier.
    Implementasi bisa keyword-based, fuzzy, SVM, IndoBERT, dll.
    """

    @abstractmethod
    def classify(self, text: str) -> IntentResult:
        """Classify text menjadi intent + confidence."""
        pass


# =========================
# KEYWORD & FUZZY IMPLEMENTATION
# =========================

class KeywordIntentClassifier(BaseIntentClassifier):
    """
    Intent classifier dengan keyword matching cepat (fast-path)
    dan fuzzy similarity fallback untuk mentoleransi typo/misheard dari STT.
    """

    def __init__(self, fuzzy_threshold: float = 0.75):
        self.fuzzy_threshold = fuzzy_threshold

        # Keyword patterns ordered by priority (first match wins)
        self._patterns: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
            # EMERGENCY — highest priority
            (Intent.EMERGENCY, (
                "keadaan darurat",
                "kondisi darurat",
                "kode merah",
                "darurat",
                "emergency",
                "butuh bantuan segera",
                "tolong darurat",
            )),

            # STOP — high priority
            (Intent.STOP, (
                "berhenti sekarang",
                "berhenti",
                "stop",
                "diam",
                "hentikan",
            )),

            # WAIT
            (Intent.WAIT, (
                "tunggu sebentar",
                "tunggu di sini",
                "tunggu dulu",
                "tunggu",
                "standby",
            )),

            # FOLLOW
            (Intent.FOLLOW, (
                "ikuti saya",
                "ikuti dokter",
                "ikuti perawat",
                "ikuti",
                "ikut saya",
                "ikut",
                "follow",
            )),

            # GO_BACK
            (Intent.GO_BACK, (
                "putar balik",
                "kembali ke",
                "kembali",
                "balik",
                "go back",
            )),

            # NAVIGATE — destination keywords & colloquial/STT variants
            (Intent.NAVIGATE, (
                "antar saya ke",
                "antar ke",
                "antarkan saya ke",
                "antarkan ke",
                "antarkan",
                "anter saya ke",
                "anter ke",
                "anterin ke",
                "anterin",
                "antar",
                "bawa saya ke",
                "bawa ke",
                "bawakan ke",
                "bawakan",
                "bahwa saya ke",
                "bahwa ke",
                "bawa",
                "bahwa",
                "pergi ke",
                "menuju ke",
                "menuju",
                "saya mau ke",
                "saya ingin ke",
                "saya ingin pergi ke",
                "saya mau pergi ke",
                "tolong ke",
                "tolong antar",
                "tolong antarkan",
                "tolong bawa",
                "tolong bawakan",
                "tolong pergi ke",
                "ke ruang",
                "ke ruangan",
                "ke poli",
                "ke lobi",
                "ke lobby",
                "ke kasir",
                "ke kantin",
                "ke farmasi",
                "ke laboratorium",
                "ke lab",
                "ke radiologi",
                "ke igd",
                "ke ugd",
                "ke icu",
                "ke picu",
                "ke nicu",
            )),

            # MOVE — directional movement
            (Intent.MOVE, (
                "belok kiri",
                "belok kanan",
                "ke depan",
                "ke belakang",
                "ke kiri",
                "ke kanan",
                "maju",
                "mundur",
                "gerak",
                "jalan",
            )),

            # STATUS — cek status / baterai
            (Intent.STATUS, (
                "status baterai",
                "cek baterai",
                "berapa baterai",
                "kondisi baterai",
                "status robot",
                "cek status",
                "kamu di mana",
                "posisi kamu",
            )),

            # GREETING — sapaan
            (Intent.GREETING, (
                "selamat pagi",
                "selamat siang",
                "selamat sore",
                "selamat malam",
                "halo robot",
                "hai robot",
                "halo",
                "hai",
                "assalamualaikum",
            )),

            # THANK — ucapan terima kasih
            (Intent.THANK, (
                "terima kasih banyak",
                "terima kasih",
                "makasih banyak",
                "makasih ya",
                "makasih",
                "thanks",
            )),
        )

    def classify(self, text: str) -> IntentResult:
        text_lower = text.lower().strip()

        if not text_lower:
            return _UNKNOWN_RESULT

        # 1. Fast-path: Keyword matching
        words_in_text = set(text_lower.split())
        for intent, keywords in self._patterns:
            for keyword in keywords:
                # Untuk keyword sangat pendek (<= 3 karakter seperti "hai"), periksa sebagai kata utuh
                if len(keyword) <= 3:
                    if keyword in words_in_text:
                        return _VALID_RESULTS[intent]
                elif keyword in text_lower:
                    return _VALID_RESULTS[intent]

        # 2. Fallback: Fuzzy word similarity matching (menangani typo STT seperti "bawak", "bahwa", "berenti")
        words = [w for w in text_lower.split() if w not in _FUZZY_IGNORE_WORDS]
        best_intent = None
        best_ratio = 0.0

        for word in words:
            if len(word) < 3:
                continue
            for intent, triggers in _FUZZY_CORE_TRIGGERS:
                for trigger in triggers:
                    ratio = SequenceMatcher(None, word, trigger).ratio()
                    if ratio >= self.fuzzy_threshold and ratio > best_ratio:
                        best_ratio = ratio
                        best_intent = intent

        if best_intent and best_ratio >= self.fuzzy_threshold:
            return IntentResult(intent=best_intent, confidence=round(best_ratio, 2))

        # 3. Contextual fallback: jika ada preposisi tujuan "ke <kata>"
        if " ke " in text_lower or text_lower.startswith("ke "):
            return IntentResult(intent=Intent.NAVIGATE, confidence=0.80)

        return _UNKNOWN_RESULT
