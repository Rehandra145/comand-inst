"""
Backward-compatibility shim for tts module.
All implementations have been modularized into the `audio.tts` package.
"""

from audio.tts import *  # noqa: F401, F403

if __name__ == "__main__":
    import sys
    import logging
    from audio.tts import clear_cache, warmup, precache_room_phrases

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--clear-cache" in sys.argv:
        clear_cache()
        print("Cache TTS dihapus.")
    if "--precache" in sys.argv:
        warmup()
        precache_room_phrases()
