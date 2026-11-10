"""Static choices shared by the API and the browser (moved from easy_gui.py).

Values are stable English identifiers; the browser translates labels.
"""
from __future__ import annotations

from indextts.utils.text_chunking import CHUNK_CHOICES

LANGUAGES = {
    "English": "EN",
    "Spanish": "ES",
    "Japanese": "JA",
    "Chinese": "ZH",
    "Arabic": "AR",
}
LANGUAGE_LABELS = {code: label for label, code in LANGUAGES.items()}
DEFAULT_LANGUAGE = "ZH"
EMOTIONS = ["Happy", "Angry", "Sad", "Afraid", "Disgusted", "Melancholic", "Surprised", "Calm"]
EMOTION_MODES = ["Same as voice reference", "Emotion reference audio", "Emotion vector", "Emotion description"]
DIALOGUE_EMOTION_SOURCES = ["Emotion Vector Preset", "Emotion Description"]
DIALOGUE_MAX_SEGMENTS = 12
DEFAULT_PRESET = "Neutral / balanced"
VRAM_PRESETS = ["24 GB", "32 GB+"]
LORA_RANKS = [4, 8, 16, 32, 64, 128, 256]

EMOTION_PRESETS = {
    "Neutral / balanced": [0, 0, 0, 0, 0, 0, 0, 0.55],
    "Warm / friendly": [0.30, 0, 0, 0, 0, 0, 0.05, 0.40],
    "Joyful": [0.65, 0, 0, 0, 0, 0, 0.10, 0.05],
    "Excited": [0.50, 0, 0, 0, 0, 0, 0.25, 0.05],
    "Playful": [0.40, 0, 0, 0, 0, 0, 0.25, 0.10],
    "Confident / proud": [0.20, 0.10, 0, 0, 0, 0, 0.05, 0.40],
    "Tender": [0.25, 0, 0.10, 0, 0, 0.10, 0, 0.30],
    "Romantic": [0.25, 0, 0.05, 0, 0, 0.20, 0, 0.25],
    "Hopeful": [0.30, 0, 0.05, 0.05, 0, 0.05, 0.10, 0.20],
    "Relieved": [0.20, 0, 0.05, 0, 0, 0, 0.05, 0.45],
    "Nostalgic": [0.10, 0, 0.15, 0, 0, 0.35, 0, 0.15],
    "Melancholic": [0, 0, 0.15, 0, 0, 0.55, 0, 0.10],
    "Sad": [0, 0, 0.65, 0, 0, 0.10, 0, 0.05],
    "Grieving": [0, 0, 0.55, 0.15, 0, 0.10, 0, 0],
    "Lonely": [0, 0, 0.35, 0.10, 0, 0.25, 0, 0.05],
    "Anxious": [0, 0, 0.05, 0.50, 0, 0.05, 0.10, 0.05],
    "Fearful": [0, 0, 0, 0.65, 0, 0, 0.10, 0.05],
    "Tense": [0, 0.15, 0, 0.35, 0, 0, 0.10, 0.15],
    "Angry": [0, 0.65, 0, 0.05, 0, 0, 0.05, 0.05],
    "Furious": [0, 0.75, 0, 0.05, 0, 0, 0, 0],
    "Irritated": [0, 0.40, 0, 0, 0.15, 0, 0, 0.20],
    "Disgusted": [0, 0.10, 0, 0, 0.60, 0, 0, 0.10],
    "Sarcastic": [0.10, 0.15, 0, 0, 0.25, 0, 0.10, 0.15],
    "Suspicious": [0, 0.10, 0, 0.20, 0.15, 0.10, 0.05, 0.15],
    "Surprised": [0.10, 0, 0, 0.05, 0, 0, 0.60, 0.05],
    "Amazed": [0.20, 0, 0, 0, 0, 0, 0.50, 0.05],
    "Shocked": [0, 0, 0, 0.20, 0, 0, 0.60, 0],
    "Urgent": [0, 0.25, 0, 0.25, 0, 0, 0.20, 0.10],
    "Determined": [0.10, 0.30, 0, 0, 0, 0, 0, 0.35],
    "Serious": [0, 0.10, 0.05, 0, 0, 0.05, 0, 0.55],
    "Documentary / neutral narrator": [0, 0, 0, 0, 0, 0, 0.05, 0.65],
    "News presenter": [0, 0, 0, 0, 0, 0.05, 0.05, 0.60],
    "Advertising / upbeat": [0.45, 0, 0, 0, 0, 0, 0.20, 0.10],
    "Advertising / luxury": [0.10, 0, 0, 0, 0, 0.15, 0, 0.50],
    "Storytelling / suspense": [0, 0, 0, 0.25, 0, 0.20, 0.10, 0.20],
    "Storytelling / intimate": [0.15, 0, 0.05, 0, 0, 0.15, 0, 0.40],
    "Cold / villainous": [0, 0.20, 0, 0.10, 0.20, 0.15, 0, 0.10],
}

# Inference defaults and ranges (identical to the 1.x widgets).
SYNTHESIS_PARAMS = {
    "duration_factor": {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05},
    "interval_silence": {"default": 200, "min": 0, "max": 1000, "step": 10},
    "temperature": {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05},
    "top_p": {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
    "top_k": {"default": 30, "min": 0, "max": 200, "step": 1},
    "num_beams": {"default": 3, "min": 1, "max": 10, "step": 1},
    "repetition_penalty": {"default": 10.0, "min": 0.1, "max": 30.0, "step": 0.1},
    "length_penalty": {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1},
    "max_mel_tokens": {"default": 1500, "min": 50, "max": 3000, "step": 10},
    "max_text_tokens": {"default": 120, "min": 20, "max": 300, "step": 5},
    "emotion_alpha": {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05},
    "chunk_silence": {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1},
    "dialogue_silence": {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1},
    "emotion_value": {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
}
SYNTHESIS_FLAGS = {"do_sample": True, "text_normalization": True, "random_emotion": False, "save_outputs": True}


def bootstrap_catalog() -> dict:
    return {
        "languages": [{"label": label, "code": code} for label, code in LANGUAGES.items()],
        "default_language": DEFAULT_LANGUAGE,
        "emotions": EMOTIONS,
        "emotion_modes": EMOTION_MODES,
        "emotion_presets": EMOTION_PRESETS,
        "default_preset": DEFAULT_PRESET,
        "dialogue_emotion_sources": DIALOGUE_EMOTION_SOURCES,
        "dialogue_max_turns": DIALOGUE_MAX_SEGMENTS,
        "chunk_modes": list(CHUNK_CHOICES),
        "vram_presets": VRAM_PRESETS,
        "lora_ranks": LORA_RANKS,
        "synthesis_params": SYNTHESIS_PARAMS,
        "synthesis_flags": SYNTHESIS_FLAGS,
    }
