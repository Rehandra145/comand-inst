"""
Dialog & Context Utilities — Penggabungan konteks multi-turn dan format pesan konfirmasi robot.
"""

import re
import responses

# Mapping nama slot internal → label Indonesia untuk TTS/display
SLOT_DISPLAY_NAMES = {
    "location": "lokasi",
    "direction": "arah",
    "person": "orang",
    "distance": "jarak",
    "unit": "satuan",
    "object": "objek",
    "duration": "durasi",
}


def translate_slot_names(slot_names: list) -> str:
    """Terjemahkan daftar nama slot ke bahasa Indonesia untuk TTS."""
    return ", ".join(SLOT_DISPLAY_NAMES.get(s, s) for s in slot_names)


def merge_dialog_context(command_text: str, answer_text: str, missing_slots: list) -> str:
    """
    Gabungkan perintah awal dan jawaban klarifikasi secara cerdas berdasarkan slot yang kurang.

    Contoh:
      - cmd: "antarkan obat ini", ans: "204"           -> "antarkan obat ini ke ruang 204"
      - cmd: "antarkan obat ini", ans: "ke ruang 204"  -> "antarkan obat ini ke ruang 204" (bukan double "ke")
      - cmd: "antarkan obat ini", ans: "laboratorium"  -> "antarkan obat ini ke laboratorium"
      - cmd: "tolong gerak",      ans: "kanan"         -> "tolong gerak ke kanan"
      - cmd: "tolong ikuti",      ans: "dokter budi"   -> "tolong ikuti dokter budi"
    """
    cmd = command_text.strip()
    ans = answer_text.strip().rstrip('.?!')
    if not ans:
        return cmd

    if "location" in missing_slots:
        # Jika user hanya menjawab angka (misal "204"), tambahkan prefix "ruang"
        if re.match(r'^\d+$', ans):
            ans = f"ruang {ans}"
        if ans.lower().startswith("ke "):
            return f"{cmd} {ans}"
        elif ans.lower().startswith("ruang "):
            return f"{cmd} ke {ans}"
        else:
            return f"{cmd} ke {ans}"

    elif "direction" in missing_slots:
        if ans.lower().startswith("ke "):
            return f"{cmd} {ans}"
        else:
            return f"{cmd} ke {ans}"

    elif "person" in missing_slots:
        return f"{cmd} {ans}"

    else:
        return f"{cmd} {ans}"


def get_confirmation_message(result: dict) -> tuple:
    """
    Mengembalikan tuple (robot_label_text, tts_speech_message) dari hasil validasi NLU.
    """
    vstatus = result["status"]
    intent = result.get("intent", "UNKNOWN")
    slots = result.get("slots", {})

    if vstatus == "VALID":
        if intent == "NAVIGATE":
            loc = slots.get('location', '')
            obj = slots.get('object', '')
            if obj and loc:
                return f"$ robot: mengantarkan {obj} ke {loc}", f"Baik, akan saya antarkan {obj} ke {loc}."
            elif obj:
                return f"$ robot: mengantarkan {obj}", f"Baik, akan saya antarkan {obj}."
            elif loc:
                return f"$ robot: menuju {loc}", responses.navigate_confirm(loc)
            else:
                return "$ robot: menuju lokasi", "Baik, saya akan menuju ke lokasi."
        elif intent == "MOVE":
            direction = slots.get("direction", "?")
            distance = slots.get("distance", "")
            unit = slots.get("unit", "")
            dist_str = f" {distance} {unit}" if distance else ""
            return f"$ robot: bergerak {direction}{dist_str}", f"Baik, saya bergerak ke {direction}{dist_str}."
        elif intent == "STOP":
            return "$ robot: berhenti", "Baik, saya berhenti."
        elif intent == "GO_BACK":
            return "$ robot: kembali", "Baik, saya kembali."
        elif intent == "WAIT":
            return "$ robot: menunggu", "Baik, saya akan menunggu."
        elif intent == "FOLLOW":
            person = slots.get('person', 'user')
            return f"$ robot: mengikuti {person}", f"Baik, saya akan mengikuti {person}."
        elif intent == "EMERGENCY":
            loc = slots.get('location', '')
            if loc:
                return f"$ robot: [DARURAT] menuju {loc}", f"Peringatan kondisi darurat. Segera menuju {loc} sekarang juga."
            return "$ robot: [DARURAT] mode darurat aktif", "Peringatan kondisi darurat. Mode prioritas darurat diaktifkan."
        elif intent == "STATUS":
            return "$ robot: status normal (baterai 85%)", "Status robot normal. Daya baterai delapan puluh lima persen dan siap beroperasi."
        elif intent == "GREETING":
            return "$ robot: menyapa", "Halo, ada yang bisa saya bantu?"
        elif intent == "THANK":
            return "$ robot: sama-sama", "Sama-sama, senang bisa membantu."
        else:
            return "$ robot: stand by", "Perintah diterima."
    elif vstatus == "INCOMPLETE":
        missing_raw = ", ".join(result.get("missing_slots", []))
        missing_id = translate_slot_names(result.get("missing_slots", []))
        return f"$ robot: perintah tidak lengkap (kurang: {missing_raw})", f"Maaf, perintah belum lengkap. Informasi {missing_id} belum ada."
    elif vstatus == "INVALID_LOCATION":
        loc = slots.get('location', '?')
        return f'$ robot: lokasi "{loc}" tidak ditemukan', f"Maaf, lokasi {loc} tidak terdaftar di sistem."
    elif vstatus == "UNKNOWN_INTENT":
        msg = responses.unknown_intent()
        return f"$ robot: {msg.lower()}", msg
    else:
        return "$ robot: stand by", "Perintah diproses."
