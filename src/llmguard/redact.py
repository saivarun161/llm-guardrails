"""Turning findings into redacted text.

Two things here are easy to get wrong and are worth doing once, centrally.

**Overlaps.** Detectors overlap: an API key assignment and a JWT can cover the
same characters, a card number and a phone pattern can both fire on the same
digits. Redacting overlapping spans naively either double-redacts or produces
mangled text where one replacement lands inside another. :func:`merge` resolves
overlaps by picking the more severe finding, then the more confident, then the
longer one, and drops the losers.

**Direction.** Replacements change length, so they are applied right to left.
Left to right, every offset after the first replacement is wrong.

The strategies exist because "redacted" means different things to different
teams. A support tool wants the last four digits kept so an agent can confirm a
card with a caller. An analytics pipeline wants a stable pseudonym so the same
customer joins across rows without the address ever being stored. Both are here;
neither is the default, because the default should be the one that reveals
nothing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .types import Finding

#: Strategy names a policy may use.
STRATEGIES = ("label", "mask", "partial", "hash", "remove")


def merge(findings: Iterable[Finding]) -> list[Finding]:
    """Drop overlapping findings, keeping the most important one per region."""
    ordered = sorted(
        findings,
        key=lambda f: (-int(f.severity), -f.confidence, -f.length, f.start),
    )
    kept: list[Finding] = []
    for finding in ordered:
        if any(finding.start < other.end and other.start < finding.end for other in kept):
            continue
        kept.append(finding)
    kept.sort(key=lambda f: f.start)
    return kept


def placeholder(finding: Finding, strategy: str, original: str, *, salt: str = "") -> str:
    """Render the replacement text for one finding."""
    matched = original[finding.start : finding.end]
    token = finding.kind.upper()

    if strategy == "label":
        return f"[{token}]"
    if strategy == "remove":
        return ""
    if strategy == "mask":
        # Keep the shape (separators, length) but none of the content, which is
        # what makes a masked log still useful for spotting a formatting bug.
        return "".join("*" if char.isalnum() else char for char in matched)
    if strategy == "partial":
        visible = matched[-4:] if len(matched) > 4 else ""
        hidden = "".join("*" if char.isalnum() else char for char in matched[: len(matched) - 4])
        return f"{hidden}{visible}"
    if strategy == "hash":
        digest = hashlib.sha256((salt + matched).encode("utf-8")).hexdigest()[:12]
        return f"[{token}:{digest}]"
    raise ValueError(f"unknown redaction strategy {strategy!r}; expected one of {STRATEGIES}")


def apply(
    text: str,
    findings: Iterable[tuple[Finding, str]],
    *,
    salt: str = "",
) -> str:
    """Redact ``text``, given ``(finding, strategy)`` pairs.

    The pairs need not be sorted and may overlap; overlaps are resolved by
    :func:`merge` before anything is replaced.
    """
    by_span = {(f.start, f.end, f.label): (f, strategy) for f, strategy in findings}
    surviving = merge(finding for finding, _ in by_span.values())

    out = text
    for finding in sorted(surviving, key=lambda f: f.start, reverse=True):
        _, strategy = by_span[(finding.start, finding.end, finding.label)]
        replacement = placeholder(finding, strategy, text, salt=salt)
        out = out[: finding.start] + replacement + out[finding.end :]
    return out
