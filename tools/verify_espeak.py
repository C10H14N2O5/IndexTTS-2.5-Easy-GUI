from __future__ import annotations

import ctypes
import sys


def main() -> int:
    try:
        import espeakng  # noqa: F401 - verify the mandatory Python binding too.
        from espeakng_loader import get_data_path, get_library_path, make_library_available

        make_library_available()
        library = ctypes.CDLL(get_library_path())
        library.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        library.espeak_Initialize.restype = ctypes.c_int
        sample_rate = library.espeak_Initialize(0, 0, get_data_path().encode("utf-8"), 0)
        if sample_rate <= 0:
            raise RuntimeError(f"eSpeak NG initialization returned {sample_rate}")

        library.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        library.espeak_SetVoiceByName.restype = ctypes.c_int
        if library.espeak_SetVoiceByName(b"es") != 0:
            raise RuntimeError("Spanish eSpeak NG voice could not be selected")

        library.espeak_TextToPhonemes.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int,
        ]
        library.espeak_TextToPhonemes.restype = ctypes.c_char_p
        buffer = ctypes.create_string_buffer("prueba de instalación".encode("utf-8"))
        text_ptr = ctypes.c_void_p(ctypes.addressof(buffer))
        ipa = library.espeak_TextToPhonemes(ctypes.byref(text_ptr), 0, 2)
        if not ipa:
            raise RuntimeError("eSpeak NG returned no IPA output")
        print(f"eSpeak NG native runtime: OK ({sample_rate} Hz) | {ipa.decode('utf-8', errors='ignore')}")
        library.espeak_Terminate()
        return 0
    except Exception as exc:
        print(f"eSpeak NG verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
