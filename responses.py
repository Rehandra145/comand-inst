"""
Responses — Variasi kalimat respons robot agar tidak monoton.

Setiap kategori berisi beberapa variasi kalimat. Fungsi `pick()` memilih secara
acak, dan menghindari kalimat yang sama dengan yang terakhir diucapkan pada
kategori tersebut.

Untuk menambah variasi, cukup tambahkan kalimat baru ke list di bawah.
Placeholder `{loc}` akan diganti dengan nama lokasi tujuan.
"""

import random
import threading

# Sapaan awal mode interaktif
GREETINGS = (
    "Halo, ada yang bisa saya bantu hari ini?",
    "Apa yang bisa saya lakukan untuk Anda?",
    "Saya akan dengan senang hati membantu Anda. Silakan sampaikan perintahnya.",
    "Halo, silakan katakan apa yang Anda butuhkan.",
    "Selamat datang, ada yang bisa saya bantu?",
    "Saya siap membantu. Mau ke mana hari ini?",
)

# Konfirmasi ketika menuju ruangan / lokasi
NAVIGATE_CONFIRMS = (
    "Baik, saya akan menuju {loc}.",
    "Siap, mari kita ke {loc}.",
    "Baik, saya antar Anda ke {loc}. Silakan ikuti saya.",
    "Oke, kita berangkat ke {loc} sekarang.",
    "Dengan senang hati, saya akan mengantar Anda ke {loc}.",
    "Siap, menuju {loc}. Mohon ikuti saya.",
)

# Pertanyaan klarifikasi ketika lokasi belum disebutkan
ASK_LOCATION = (
    "Ke ruangan berapa?",
    "Anda ingin ke ruangan mana?",
    "Ruangan mana yang ingin Anda tuju?",
    "Boleh sebutkan ruangan tujuannya?",
)

# Respons ketika perintah tidak dikenali (UNKNOWN_INTENT)
UNKNOWN_INTENT = (
    "Perintah tidak dapat di proses",
    "Perintah anda tidak dapat saya pahami",
)

_last_choice = {}
_lock = threading.Lock()


def pick(options, **fmt) -> str:
    """Pilih satu variasi secara acak (tidak mengulang pilihan terakhir)."""
    key = id(options)
    with _lock:
        last = _last_choice.get(key)
        candidates = [o for o in options if o != last] or list(options)
        choice = random.choice(candidates)
        _last_choice[key] = choice
    return choice.format(**fmt) if fmt else choice


def greeting() -> str:
    return pick(GREETINGS)


def navigate_confirm(loc: str) -> str:
    return pick(NAVIGATE_CONFIRMS, loc=loc)


def ask_location() -> str:
    return pick(ASK_LOCATION)


def unknown_intent() -> str:
    return pick(UNKNOWN_INTENT)
