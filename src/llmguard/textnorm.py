"""Unicode normalisation that keeps a map back to the original offsets.

An injection payload does not have to be typed in plain ASCII. ``ignore previous
instructions`` survives a zero-width space between every letter, a fullwidth
codepoint here and there, or a combining accent, and still reads perfectly to the
model. Matching on the raw text misses all of it; matching on a normalised copy
and then reporting offsets into that copy points at the wrong characters.

So the normaliser returns both: the folded text, and an index map where
``index[i]`` is the offset in the original string of the character that produced
``folded[i]``. Every heuristic match on the folded copy is translated back before
it becomes a :class:`~llmguard.types.Finding`, and redaction always cuts the
original string.

The folding is deliberately aggressive -- NFKC, casefold, drop invisibles, drop
combining marks -- because it is only ever used for heuristic *matching*. PII
detection and every byte that gets redacted work on the original text.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata

# Codepoints with no visible width, plus the bidirectional overrides. Scattered
# through a trigger phrase they defeat a substring check while changing nothing
# about how the text reads. They are written as escapes on purpose: spelled
# literally they would be invisible in this file too, which is the whole problem.
_INVISIBLE = frozenset(
    [
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\u2060",  # word joiner
        "\ufeff",  # zero width no-break space
        "\u00ad",  # soft hyphen
        "\u180e",  # mongolian vowel separator
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u202a",  # left-to-right embedding
        "\u202b",  # right-to-left embedding
        "\u202c",  # pop directional formatting
        "\u202d",  # left-to-right override
        "\u202e",  # right-to-left override
    ]
)


def normalise(text: str) -> tuple[str, list[int]]:
    """Fold ``text`` for matching and return ``(folded, index_map)``.

    ``index_map`` has one entry per character of ``folded``. Folding is done one
    character at a time precisely so the map stays exact even where a single
    codepoint expands: a capital I with a dot above casefolds to two characters,
    and the fi ligature to two letters. Both halves then point at the same
    original offset, so a match spanning them still maps back correctly.
    """
    folded: list[str] = []
    index: list[int] = []
    for position, char in enumerate(text):
        if char in _INVISIBLE:
            continue
        expanded = unicodedata.normalize("NFKC", char).casefold()
        for out_char in expanded:
            if unicodedata.combining(out_char):
                continue
            folded.append(out_char)
            index.append(position)
    return "".join(folded), index


def to_original(index: list[int], start: int, end: int, fallback_end: int = 0) -> tuple[int, int]:
    """Translate a ``[start, end)`` span on the folded text back to the original.

    ``fallback_end`` is used when the span is empty or runs off the end of the
    map, which happens when every character in the range was dropped by folding.
    """
    if not index:
        return 0, fallback_end
    start = max(0, min(start, len(index) - 1))
    if end <= start:
        return index[start], index[start] + 1
    end = min(end, len(index))
    return index[start], index[end - 1] + 1


_B64_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,512}={0,2}(?![A-Za-z0-9+/=])")


def base64_candidates(text: str) -> list[tuple[int, int, str]]:
    """Find base64 runs that decode to readable text.

    Returns ``(start, end, decoded)`` for each run whose decoded bytes are mostly
    printable ASCII. Anything that decodes to binary is not a hidden instruction,
    it is a thumbnail, so it is dropped here rather than downstream.
    """
    out: list[tuple[int, int, str]] = []
    for match in _B64_RE.finditer(text):
        blob = match.group(0)
        padded = blob + "=" * (-len(blob) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            continue
        if len(raw) < 12:
            continue
        printable = sum(1 for byte in raw if 0x20 <= byte < 0x7F or byte in (0x09, 0x0A, 0x0D))
        if printable / len(raw) < 0.9:
            continue
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        out.append((match.start(), match.end(), decoded))
    return out
