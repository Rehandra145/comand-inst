"""
Konfigurasi entitas dan lokasi rumah sakit.

Tambahkan lokasi/entitas baru di sini tanpa mengubah logic utama.
"""


# =========================
# LOKASI RUMAH SAKIT
# =========================

LOCATIONS = [
    # Ruang rawat inap
    "ruang ICU",
    "ruang IGD",
    "ruang UGD",
    "ruang NICU",
    "ruang PICU",
    "ruang VIP",
    "ruang isolasi",
    "ruang rawat inap",
    "ruang bersalin",

    # Ruang tindakan
    "ruang operasi",
    "ruang bedah",
    "ruang radiologi",
    "ruang rontgen",
    "ruang CT scan",
    "ruang MRI",
    "ruang laboratorium",
    "ruang lab",
    "ruang farmasi",

    # Ruang umum
    "ruang tunggu",
    "lobi",
    "lobby",
    "resepsionis",
    "kasir",
    "kantin",
    "mushola",
    "parkir",

    # Ruang bernomor
    "ruang 1",
    "ruang 2",
    "ruang 3",
    "ruang 4",
    "ruang 5",

    # Poli
    "poli umum",
    "poli anak",
    "poli gigi",
    "poli mata",
    "poli jantung",
    "poli paru",
    "poli bedah",
    "poli saraf",
    "poli kulit",
    "poli THT",
    "poli kandungan",
    "poli dalam",
    "poli ortopedi",
]

# Alias: variasi nama → nama standar
LOCATION_ALIASES = {
    "icu": "ruang ICU",
    "igd": "ruang IGD",
    "ugd": "ruang UGD",
    "nicu": "ruang NICU",
    "picu": "ruang PICU",
    "vip": "ruang VIP",
    "lobby": "lobi",
    "lab": "ruang laboratorium",
    "ruang lab": "ruang laboratorium",
    "rontgen": "ruang rontgen",
    # Ruang kata angka -> digit standar
    "ruang satu": "ruang 1",
    "ruang dua": "ruang 2",
    "ruang tiga": "ruang 3",
    "ruang empat": "ruang 4",
    "ruang lima": "ruang 5",
    # Ruangan variasi
    "ruangan 1": "ruang 1",
    "ruangan satu": "ruang 1",
    "ruangan 2": "ruang 2",
    "ruangan dua": "ruang 2",
    "ruangan 3": "ruang 3",
    "ruangan tiga": "ruang 3",
    "ruangan 4": "ruang 4",
    "ruangan empat": "ruang 4",
    "ruangan 5": "ruang 5",
    "ruangan lima": "ruang 5",
    # Variasi typo fonetik STT: ruas -> ruang
    "ruas 1": "ruang 1",
    "ruas satu": "ruang 1",
    "ruas 2": "ruang 2",
    "ruas dua": "ruang 2",
    "ruas 3": "ruang 3",
    "ruas tiga": "ruang 3",
    "ruas 4": "ruang 4",
    "ruas empat": "ruang 4",
    "ruas 5": "ruang 5",
    "ruas lima": "ruang 5",
}


# =========================
# ORANG / PERSON
# =========================

PERSONS = [
    "dokter",
    "perawat",
    "suster",
    "pasien",
    "bidan",
    "apoteker",
    "satpam",
    "cleaning service",
]


# =========================
# ARAH / DIRECTION
# =========================

DIRECTION_MAP = {
    "maju": "forward",
    "ke depan": "forward",
    "mundur": "backward",
    "ke belakang": "backward",
    "belok kiri": "left",
    "ke kiri": "left",
    "kiri": "left",
    "belok kanan": "right",
    "ke kanan": "right",
    "kanan": "right",
}


# =========================
# NUMBER WORDS (Indonesia)
# =========================

NUMBER_WORDS = {
    "nol": 0,
    "satu": 1,
    "dua": 2,
    "tiga": 3,
    "empat": 4,
    "lima": 5,
    "enam": 6,
    "tujuh": 7,
    "delapan": 8,
    "sembilan": 9,
    "sepuluh": 10,
    "setengah": 0.5,
}
