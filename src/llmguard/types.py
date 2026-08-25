"""The vocabulary every other module speaks: severities, actions, findings, results.

Two small decisions here shape the rest of the codebase.

``Severity`` and ``Action`` are ordered enums. Severity ordering lets a policy say
"anything at or above ``high``" without enumerating levels, and action ordering
turns "several rules fired, what happens" into ``max(actions)`` -- a block always
beats a redaction, a redaction always beats a flag -- instead of a precedence
table that drifts out of sync with the enum.

``Finding`` carries character offsets into the *original* text and never carries
the matched substring. A finding is meant to be logged, and the whole point of
this layer is that the matched substring is the thing you must not log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Severity(IntEnum):
    """How bad a finding is, ordered so policies can express thresholds."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, value: str | Severity) -> Severity:
        if isinstance(value, Severity):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError:
            raise ValueError(
                f"unknown severity {value!r}; expected one of {', '.join(s.label for s in cls)}"
            ) from None

    @property
    def label(self) -> str:
        return self.name.lower()


class Action(IntEnum):
    """What a policy rule does about a finding, ordered weakest to strongest."""

    ALLOW = 0
    FLAG = 1
    REDACT = 2
    BLOCK = 3

    @classmethod
    def parse(cls, value: str | Action) -> Action:
        if isinstance(value, Action):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError:
            raise ValueError(
                f"unknown action {value!r}; expected one of {', '.join(a.label for a in cls)}"
            ) from None

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing a detector noticed, located by offsets into the original text.

    ``confidence`` is not decoration. Detectors that validate their matches --
    Luhn for card numbers, mod-97 for IBANs, a decodable header for JWTs -- report
    high confidence, and detectors matching on shape alone report less. Overlap
    resolution and policy thresholds both read it.
    """

    detector: str
    kind: str
    start: int
    end: int
    severity: Severity
    confidence: float
    detail: str = ""
    signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")

    @property
    def label(self) -> str:
        """The dotted name a policy rule matches against, e.g. ``pii.email``."""
        return f"{self.detector}.{self.kind}"

    @property
    def length(self) -> int:
        return self.end - self.start

    def shifted(self, offset: int) -> Finding:
        """Move the span, for translating window offsets back to stream offsets."""
        return Finding(
            detector=self.detector,
            kind=self.kind,
            start=self.start + offset,
            end=self.end + offset,
            severity=self.severity,
            confidence=self.confidence,
            detail=self.detail,
            signals=self.signals,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "detector": self.detector,
            "kind": self.kind,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "severity": self.severity.label,
            "confidence": round(self.confidence, 4),
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.signals:
            payload["signals"] = list(self.signals)
        return payload


@dataclass(frozen=True, slots=True)
class RuleDecision:
    """Which rule fired on which finding, and what it decided.

    The audit trail people actually want is not "the request was blocked" but
    "rule ``block-injection`` blocked it because of a critical injection finding
    at offset 412". This is that record.
    """

    rule_id: str
    action: Action
    finding_index: int
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "rule": self.rule_id,
            "action": self.action.label,
            "finding": self.finding_index,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class GuardResult:
    """The outcome of checking one piece of text at one stage."""

    stage: str
    policy: str
    verdict: Action
    text: str
    original_length: int
    findings: tuple[Finding, ...] = ()
    decisions: tuple[RuleDecision, ...] = ()
    elapsed_ms: float = 0.0
    budget_exceeded: bool = False
    skipped_detectors: tuple[str, ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.verdict is Action.BLOCK

    @property
    def modified(self) -> bool:
        return self.verdict is Action.REDACT

    @property
    def block_reasons(self) -> tuple[str, ...]:
        return tuple(
            d.reason or f"{self.findings[d.finding_index].label} ({d.rule_id})"
            for d in self.decisions
            if d.action is Action.BLOCK
        )

    def counts(self) -> dict[str, int]:
        """Findings per dotted label, for a one-line summary."""
        out: dict[str, int] = {}
        for finding in self.findings:
            out[finding.label] = out.get(finding.label, 0) + 1
        return dict(sorted(out.items()))

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "policy": self.policy,
            "verdict": self.verdict.label,
            "original_length": self.original_length,
            "findings": [f.to_dict() for f in self.findings],
            "decisions": [d.to_dict() for d in self.decisions],
            "elapsed_ms": round(self.elapsed_ms, 3),
            "budget_exceeded": self.budget_exceeded,
        }
        if self.skipped_detectors:
            payload["skipped_detectors"] = list(self.skipped_detectors)
        if include_text:
            # Safe by construction: on a redact verdict this is the redacted text,
            # and on a block verdict the engine never puts the payload here.
            payload["text"] = self.text
        return payload


class PolicyError(ValueError):
    """A policy file is malformed, or names something that does not exist."""


class StreamBlocked(RuntimeError):
    """Raised by the streaming guard when a block-level finding lands mid-stream.

    Carries the findings so the caller can log why, and ``emitted`` so it can be
    honest about how much text already reached the client.
    """

    def __init__(self, findings: tuple[Finding, ...], emitted: int, reasons: tuple[str, ...] = ()):
        self.findings = findings
        self.emitted = emitted
        self.reasons = reasons
        detail = ", ".join(reasons) if reasons else ", ".join(sorted({f.label for f in findings}))
        super().__init__(f"stream blocked after {emitted} characters: {detail}")


_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?$|^\*$")


def valid_label(label: str) -> bool:
    """``pii``, ``pii.email`` and ``*`` are addressable; anything else is a typo."""
    return bool(_LABEL_RE.match(label))
