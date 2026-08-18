"""Language discovery for the pinned Seamless M4T v2 speech runtime."""

from __future__ import annotations

from typing import Any


# Friendly labels for the speech-input/speech-output languages published for
# Seamless M4T v2 Large. Runtime acceptance is still derived from the loaded
# checkpoint configuration so this table cannot accidentally enable a code the
# model's text-to-unit decoder or vocoder cannot synthesize.
LANGUAGE_NAMES_BY_CODE = {
    "arb": "Arabic (Modern Standard)",
    "ben": "Bengali",
    "cat": "Catalan",
    "ces": "Czech",
    "cmn": "Chinese (Mandarin)",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "eng": "English",
    "est": "Estonian",
    "fin": "Finnish",
    "fra": "French",
    "hin": "Hindi",
    "ind": "Indonesian",
    "ita": "Italian",
    "jpn": "Japanese",
    "kor": "Korean",
    "mlt": "Maltese",
    "nld": "Dutch",
    "pes": "Persian (Western)",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "spa": "Spanish",
    "swe": "Swedish",
    "swh": "Swahili",
    "tel": "Telugu",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Uzbek (Northern)",
    "vie": "Vietnamese",
}


def speech_input_output_languages(generation_config: Any) -> dict[str, str]:
    """Return friendly-name -> code for languages the checkpoint can speak.

    Seamless uses a language-independent speech encoder. A speech result still
    requires the code in the text decoder, text-to-unit decoder, and vocoder;
    intersecting those checkpoint maps prevents the UI/API allowlist from
    advertising a language that cannot complete speech-to-speech generation.
    """

    stage_names = (
        "text_decoder_lang_to_code_id",
        "t2u_lang_code_to_id",
        "vocoder_lang_code_to_id",
    )
    stages: list[set[str]] = []
    for name in stage_names:
        value = getattr(generation_config, name, None)
        if not isinstance(value, dict) or not value:
            raise RuntimeError(f"model generation configuration is missing {name}")
        stages.append(set(value))

    supported = set.intersection(*stages)
    if not supported:
        raise RuntimeError("model generation configuration has no speech-output languages")

    pairs = (
        (LANGUAGE_NAMES_BY_CODE.get(code, f"{code} (unlabeled)"), code)
        for code in supported
    )
    return dict(sorted(pairs, key=lambda item: (item[0].casefold(), item[1])))
