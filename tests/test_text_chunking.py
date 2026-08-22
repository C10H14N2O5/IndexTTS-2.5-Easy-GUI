import re

from indextts.utils.text_chunking import paragraph_sentence_split, split_long_text


def test_automatic_chunking_preserves_orthographic_units_and_source_text():
    source = "La versión 2.5 funciona. ¿Qué tal? ¡Muy bien! Después continúa el texto."
    chunks = paragraph_sentence_split(source, max_chars=30)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert " ".join(chunks) == re.sub(r"\s+", " ", source).strip()
    assert not any(chunk == "La versión 2." for chunk in chunks)
    assert chunks[0].endswith(".")
    assert any("¿Qué tal?" in chunk for chunk in chunks)
    assert any("¡Muy bien!" in chunk for chunk in chunks)


def test_automatic_chunking_supports_cjk_terminators():
    chunks = paragraph_sentence_split("これは最初です。これは次です！最後です？", max_chars=12)
    assert chunks == ["これは最初です。", "これは次です！", "最後です？"]


def test_reference_modes_keep_expected_hierarchy():
    text = "[SPEAKER0] Hola.\n[SPEAKER1] Adiós."
    assert split_long_text(text, "None") == [text]
    assert split_long_text(text, "Lines") == ["[SPEAKER0] Hola.", "[SPEAKER1] Adiós."]
    assert split_long_text(text, "Speaker turns") == ["[SPEAKER0] Hola.", "[SPEAKER1] Adiós."]
