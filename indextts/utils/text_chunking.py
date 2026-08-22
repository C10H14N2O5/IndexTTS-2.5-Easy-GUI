"""Text chunking rules shared by the IndexTTS Easy GUI inference workflow.

The public modes intentionally match the MOSS and FireRed Easy GUIs.  The
automatic mode is punctuation-aware instead of being a blind character
split: it keeps sentence punctuation with the sentence, understands Spanish
and CJK terminators, and only falls back to clause/word wrapping when one
orthographic unit is longer than the target size.
"""

from __future__ import annotations

import re


CHUNK_CHOICES = [
    "None",
    "Paragraph/Sentence Auto",
    "Periods",
    "Paragraphs",
    "Lines",
    "Speaker turns",
]

_SENTENCE_TERMINATORS = frozenset(".!?…。！？")
_CLAUSE_TERMINATORS = frozenset(",;:，；：、")
_CLOSING_PUNCTUATION = frozenset('"\'»”’)]}）〕〉》」』】')
_COMMON_ABBREVIATIONS = frozenset(
    {
        "a.m",
        "p.m",
        "art",
        "av",
        "aprox",
        "dr",
        "dra",
        "etc",
        "e.g",
        "fig",
        "ing",
        "núm",
        "nº",
        "p.ej",
        "prof",
        "sr",
        "sra",
        "srta",
        "st",
        "vs",
    }
)


def _period_is_orthographic_boundary(text: str, index: int) -> bool:
    """Return whether a period is sentence punctuation, not a decimal/abbr."""

    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return False

    token_match = re.search(r"([^\s]+)$", text[:index])
    if token_match:
        token = token_match.group(1).strip("([{\"'«“‘")
        if token.lower() in _COMMON_ABBREVIATIONS:
            return False
    return True


def _sentence_units(text: str, terminators: frozenset[str] = _SENTENCE_TERMINATORS) -> list[str]:
    """Split at sentence punctuation while preserving all source characters."""

    text = text.strip()
    if not text:
        return []

    units: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char not in terminators:
            index += 1
            continue
        if char == "." and not _period_is_orthographic_boundary(text, index):
            index += 1
            continue

        end = index + 1
        while end < len(text) and text[end] in terminators:
            end += 1
        while end < len(text) and text[end] in _CLOSING_PUNCTUATION:
            end += 1
        # CJK prose commonly omits the space after 。！？; Latin prose still
        # requires whitespace so an inline exclamation mark is not treated as
        # a sentence boundary accidentally.
        if end < len(text) and not text[end].isspace() and char not in "。！？":
            index += 1
            continue

        unit = text[start:end].strip()
        if unit:
            units.append(unit)
        start = end
        index = end

    tail = text[start:].strip()
    if tail:
        units.append(tail)
    return units or [text]


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """Wrap a single overlong clause on word boundaries without losing text."""

    words = re.findall(r"\S+(?:\s+|$)", text.strip())
    if not words:
        return []
    chunks: list[str] = []
    current = ""
    for raw_word in words:
        word = raw_word.strip()
        if not word:
            continue
        candidate = f"{current} {word}".strip() if current else word
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_overlong_unit(text: str, max_chars: int) -> list[str]:
    """Prefer clause punctuation before falling back to word boundaries."""

    clauses: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char not in _CLAUSE_TERMINATORS:
            continue
        end = index + 1
        while end < len(text) and text[end].isspace():
            end += 1
        clause = text[start:end].strip()
        if clause:
            clauses.append(clause)
        start = end
    tail = text[start:].strip()
    if tail:
        clauses.append(tail)
    if len(clauses) <= 1:
        return _hard_wrap(text, max_chars)

    chunks: list[str] = []
    current = ""
    for clause in clauses:
        if len(clause) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_wrap(clause, max_chars))
            continue
        candidate = f"{current} {clause}".strip() if current else clause
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = clause
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def paragraph_sentence_split(text: str, max_chars: int = 120) -> list[str]:
    """Split long prose by paragraphs, sentences, clauses, then words."""

    text = (text or "").strip()
    if not text:
        return []
    max_chars = max(1, int(max_chars))
    chunks: list[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    for paragraph in paragraphs or [text]:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        units = _sentence_units(paragraph)
        current = ""
        for unit in units:
            if len(unit) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(_split_overlong_unit(unit, max_chars))
                continue
            candidate = f"{current} {unit}".strip() if current else unit
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = unit
            else:
                current = candidate
        if current:
            chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def split_by_periods(text: str) -> list[str]:
    """Match the reference GUIs' explicit period-only mode."""

    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return _sentence_units(text, frozenset("."))


def split_long_text(text: str, mode: str) -> list[str]:
    """Split text using the MOSS/FireRed-compatible mode names."""

    text = (text or "").strip()
    if not text:
        return []
    if mode == "None":
        return [text]
    if mode == "Paragraph/Sentence Auto":
        return paragraph_sentence_split(text)
    if mode == "Periods":
        return split_by_periods(text)
    if mode == "Paragraphs":
        return [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if mode == "Lines":
        return [part.strip() for part in text.splitlines() if part.strip()]
    if mode == "Speaker turns":
        chunks: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            if re.match(r"^\s*\[?SPEAKER\d+\]?", line, flags=re.IGNORECASE) and current:
                chunks.append("\n".join(current).strip())
                current = [line.strip()]
            elif line.strip():
                current.append(line.strip())
        if current:
            chunks.append("\n".join(current).strip())
        return chunks or [text]
    return [text]
