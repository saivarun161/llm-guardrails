"""Prompt-injection heuristics, scored from named signals.

Be clear about what this is. There is no sound detector for prompt injection --
the attack is natural language, and natural language has no grammar that
separates "instructions from the operator" from "instructions from a document
the operator retrieved". What is here is a weighted signal model: a set of
patterns that empirically show up in injection attempts, each with a weight, and
a total that maps onto a severity. It catches the lazy attempts, which is most of
them, and it will not catch a careful one.

Two things make it more useful than a substring blocklist:

**It matches on a normalised copy.** Zero-width characters between letters,
fullwidth codepoints and combining marks all defeat a plain ``in`` check while
reading identically to the model. :mod:`llmguard.textnorm` folds them away and
keeps an index map, so a finding still points at the right characters in the
original text.

**It looks inside base64.** A retrieved document that carries a base64 blob
decoding to "ignore previous instructions" is a real pattern, and a scanner that
only reads the surface text sees nothing.

Every finding lists the signals that fired, so an operator tuning a threshold can
see *why* something scored 3.5 rather than being handed a number.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..textnorm import base64_candidates, normalise, to_original
from ..types import Finding, Severity


@dataclass(frozen=True, slots=True)
class Signal:
    name: str
    weight: float
    regex: re.Pattern[str]
    description: str


def _sig(name: str, weight: float, pattern: str, description: str) -> Signal:
    return Signal(name, weight, re.compile(pattern), description)


#: Patterns are written against normalised text: already casefolded, with
#: invisible characters and combining marks removed.
SIGNALS: tuple[Signal, ...] = (
    _sig(
        "instruction_override",
        2.5,
        r"\b(?:ignore|disregard|forget|override|discard)\b[^.\n]{0,40}?\b"
        r"(?:previous|prior|above|earlier|preceding|all)\b[^.\n]{0,30}?\b"
        r"(?:instruction|prompt|rule|direction|command|guideline|context)s?\b",
        "asks the model to drop the instructions it was given",
    ),
    _sig(
        "system_prompt_exfiltration",
        2.5,
        r"\b(?:reveal|print|repeat|show|output|display|disclose|dump|recite)\b[^.\n]{0,40}?\b"
        r"(?:system\s*prompt|initial\s*(?:prompt|instruction)|your\s*(?:instructions|rules|prompt)"
        r"|the\s*prompt\s*above)\b",
        "asks the model to disclose its own instructions",
    ),
    _sig(
        "role_hijack",
        2.0,
        r"(?:^|\n)\s*(?:system|assistant|developer)\s*[:>]|"
        r"\byou\s+are\s+now\s+(?:a|an|in)\b|"
        r"\bnew\s+(?:system\s+)?(?:instruction|persona|role)s?\b|"
        r"\benter\s+(?:developer|debug|god)\s+mode\b",
        "tries to reassign the model's role or open a new system turn",
    ),
    _sig(
        "chat_template_forgery",
        2.5,
        r"<\|(?:im_start|im_end|system|endoftext|start_header_id|eot_id)\|>|"
        r"\[/?INST\]|<<SYS>>|\bbegin\s+system\s+message\b",
        "forges the delimiters a chat template uses to separate turns",
    ),
    _sig(
        "safety_bypass",
        1.5,
        r"\b(?:without|bypass|ignore|skip|disable|turn\s+off)\b[^.\n]{0,30}?\b"
        r"(?:restriction|filter|guardrail|safety|censor|limitation|policy|policies)\b|"
        r"\bdo\s+anything\s+now\b|\bjailbreak\b|\bunfiltered\s+mode\b",
        "asks for safety behaviour to be switched off",
    ),
    _sig(
        "exfiltration_channel",
        1.5,
        r"\b(?:send|post|email|upload|exfiltrate|forward|transmit)\b[^.\n]{0,40}?\b"
        r"(?:to\s+https?://|to\s+[\w.-]+@[\w.-]+|webhook|attacker|external\s+server)\b|"
        r"!\[[^\]]*\]\(https?://[^)]*\{|"
        r"\bcurl\s+https?://",
        "names a channel for getting data out",
    ),
    _sig(
        "tool_coercion",
        1.5,
        r"\b(?:call|invoke|execute|run|use)\s+(?:the\s+)?(?:tool|function|command|shell|api)\b"
        r"[^.\n]{0,40}?\b(?:delete|drop|rm\s+-rf|transfer|payout|refund|grant|escalate)\b|"
        r"\bwithout\s+(?:asking|confirmation|approval)\b",
        "pushes the model towards a side-effecting tool call unprompted",
    ),
    _sig(
        "authority_claim",
        1.0,
        r"\b(?:i\s+am|this\s+is)\s+(?:the\s+)?(?:developer|administrator|admin|operator|owner)\b|"
        r"\bauthorized\s+by\s+(?:the\s+)?(?:developer|admin|operator|vendor|provider|platform)\b|"
        r"\burgent(?:ly)?\b[^.\n]{0,30}?\boverride\b",
        "claims an authority the text cannot actually carry",
    ),
    _sig(
        "output_format_hijack",
        1.0,
        r"\bfrom\s+now\s+on\b[^.\n]{0,40}?\b(?:respond|reply|answer|output)\b|"
        r"\balways\s+(?:respond|reply|answer|end)\s+with\b|"
        r"\bdo\s+not\s+(?:mention|reveal|tell)\b[^.\n]{0,30}?\b(?:this|these\s+instructions)\b",
        "tries to install a standing rule on future replies",
    ),
)

_SIGNALS_BY_NAME = {signal.name: signal for signal in SIGNALS}

#: The widest span, in characters of the *original* text, that a finding from
#: this detector may cover. It is load-bearing rather than descriptive: the
#: streaming holdback is derived from it, so a finding wider than this could
#: straddle the emit boundary and be seen only after its text was forwarded.
#:
#: 512 is the upper bound on a base64 run in :data:`llmguard.textnorm._B64_RE`,
#: plus two padding characters, and every signal regex is bounded far below that
#: on folded text. The number is necessary rather than obvious because folding
#: does not preserve length in the direction that matters: a trigger phrase
#: spelled with a zero-width character between every letter folds to thirty
#: characters and occupies arbitrarily many original ones. A hit that needs more
#: than this many original characters is therefore *not reported at all* --
#: see :meth:`InjectionDetector.scan`.
MAX_MATCH_LEN = 514

#: Extra weight for a payload that was hidden rather than written plainly.
OBFUSCATION_WEIGHT = 1.5

#: Score thresholds, set so that the strongest single signal (2.5) lands on
#: ``medium`` and two independent signals reach ``high``. The bundled policy
#: blocks at ``high``, so this is the line between flagging a support ticket that
#: quotes an attack and blocking it -- and single-signal blocking is where the
#: false positives live. Evidence that the payload was *hidden* counts as the
#: second signal on its own, because benign text does not arrive wrapped in
#: zero-width characters.
_THRESHOLDS: tuple[tuple[float, Severity], ...] = (
    (4.5, Severity.CRITICAL),
    (3.0, Severity.HIGH),
    (1.5, Severity.MEDIUM),
    (0.5, Severity.LOW),
)


def score(signal_names: tuple[str, ...] | set[str]) -> float:
    """Total weight for a set of signal names. Unknown names contribute nothing."""
    return round(
        sum(_SIGNALS_BY_NAME[name].weight for name in set(signal_names) if name in _SIGNALS_BY_NAME)
        + (OBFUSCATION_WEIGHT if "obfuscated_payload" in signal_names else 0.0),
        3,
    )


def severity_for(total: float) -> Severity:
    for threshold, severity in _THRESHOLDS:
        if total >= threshold:
            return severity
    return Severity.INFO


def describe(signal_names: tuple[str, ...]) -> list[str]:
    """Human-readable reasons, for the CLI and for audit records."""
    out = []
    for name in signal_names:
        if name == "obfuscated_payload":
            out.append(
                "obfuscated_payload: the trigger text was hidden rather than written plainly"
            )
        elif name in _SIGNALS_BY_NAME:
            out.append(f"{name}: {_SIGNALS_BY_NAME[name].description}")
    return out


def _cluster(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping and nearby spans, never past :data:`MAX_MATCH_LEN`.

    Greedy over sorted starts: a span joins the open cluster only while the
    result still fits inside the bound, so every cluster returned is within it
    without needing a truncation step afterwards.
    """
    clusters: list[tuple[int, int]] = []
    for start, end in sorted(set(spans)):
        if clusters and end - clusters[-1][0] <= MAX_MATCH_LEN:
            clusters[-1] = (clusters[-1][0], max(clusters[-1][1], end))
        else:
            clusters.append((start, end))
    return clusters


class InjectionDetector:
    """Scores prompt-injection signals over normalised text."""

    name: str = "injection"

    #: Annotated rather than inferred: a bare ``("prompt_injection",)`` infers as
    #: ``tuple[str]``, and :class:`~llmguard.detectors.base.Detector` declares
    #: ``tuple[str, ...]``. Protocol attributes are invariant, so the narrower
    #: type makes this class fail a static conformance check while behaving
    #: correctly at runtime -- the kind of mismatch nobody notices until a user
    #: types their own detector against the protocol.
    kinds: tuple[str, ...] = ("prompt_injection",)

    #: See :data:`MAX_MATCH_LEN`. Every finding :meth:`scan` returns is no wider
    #: than this, by construction rather than by inspection of the patterns.
    max_match_len: int = MAX_MATCH_LEN

    def scan(self, text: str) -> list[Finding]:
        """Findings for every cluster of signals, none wider than the bound.

        Two things here exist only to keep :data:`MAX_MATCH_LEN` honest, and
        both cost a little detection to do it.

        A hit whose *original* span is wider than the bound is discarded. That
        is the zero-width padding case: the folded text still reads as a trigger
        phrase, but following it would need a window this detector has promised
        not to need. Reporting it would make the batch result depend on how much
        text happened to be buffered, which is precisely the divergence the
        streaming guarantee rules out. Dropping it makes batch and stream agree
        on missing it, which is worse detection and honest arithmetic.

        Hits far apart become separate findings rather than one finding spanning
        the gap between them. The score is still computed over every signal seen
        anywhere in the text -- an attack does not become weaker for being spread
        out -- so what changes is the span each finding covers, not the verdict.
        """
        folded, index = normalise(text)
        hits: dict[str, tuple[int, int]] = {}

        for signal in SIGNALS:
            match = signal.regex.search(folded)
            if match is None:
                continue
            span = to_original(index, match.start(), match.end(), len(text))
            if span[1] - span[0] <= MAX_MATCH_LEN:
                hits[signal.name] = span

        obfuscated = self._scan_encoded(text, hits)

        if not hits:
            return []

        names = tuple(sorted(hits))
        if obfuscated:
            names = (*names, "obfuscated_payload")
        elif folded != text.casefold() and any(
            self._span_was_folded(text, start, end) for start, end in hits.values()
        ):
            # Something was folded away inside a matched span: invisible
            # characters or lookalike codepoints wrapped around a trigger phrase.
            names = (*names, "obfuscated_payload")

        total = score(names)
        severity = severity_for(total)
        confidence = min(0.99, round(total / 5.0, 3))

        return [
            Finding(
                detector=self.name,
                kind="prompt_injection",
                start=start,
                end=min(end, len(text)),
                severity=severity,
                confidence=confidence,
                detail=f"score {total} from {len(names)} signal(s)",
                signals=names,
            )
            for start, end in _cluster(hits.values())
        ]

    def _scan_encoded(self, text: str, hits: dict[str, tuple[int, int]]) -> bool:
        """Re-run the signals over anything base64 hides, attributing to the blob."""
        found = False
        for start, end, decoded in base64_candidates(text):
            folded, _ = normalise(decoded)
            for signal in SIGNALS:
                if signal.regex.search(folded):
                    hits.setdefault(signal.name, (start, end))
                    found = True
        return found

    @staticmethod
    def _span_was_folded(text: str, start: int, end: int) -> bool:
        raw = text[start:end]
        folded, _ = normalise(raw)
        return folded != raw.casefold()
