"""Deterministic PII and secret detection.

Every pattern here is paired with a validator, and that is the entire point. A
regex for card numbers matches any sixteen digits, so a scanner built on regexes
alone reports the order number in "your order 4532015112830367 shipped" and the
tracking number in the next sentence with equal confidence, and a team that gets
two false positives a day turns the guardrail off. The validators cut the shape
matches that cannot be the thing they look like:

* card numbers must pass Luhn and carry a known issuer prefix;
* IBANs must pass the mod-97 check;
* social security numbers must not use a reserved area, group or serial;
* JWTs must have a header that base64-decodes to JSON containing ``alg``;
* IPv4 octets must be in range.

What is deliberately missing is name and address detection. Those need a model,
a model makes this layer non-deterministic, and a non-deterministic redactor
cannot be tested with the exact-output assertions the rest of this project relies
on. The README says so rather than implying coverage that is not here.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from ..types import Finding, Severity


def luhn_ok(digits: str) -> bool:
    """The check digit algorithm every card issuer uses."""
    if not digits.isdigit() or len(digits) < 12:
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


#: Issuer prefixes, kept coarse on purpose -- the goal is to reject arbitrary
#: Luhn-valid digit strings, not to identify the network.
_CARD_PREFIXES = (
    "4",
    "34",
    "37",
    "300",
    "301",
    "302",
    "303",
    "304",
    "305",
    "36",
    "38",
    "6011",
    "62",
    "65",
    "51",
    "52",
    "53",
    "54",
    "55",
    "2221",
    "2720",
    "3528",
    "3589",
)


def card_brand_ok(digits: str) -> bool:
    if len(digits) not in range(13, 20):
        return False
    return any(digits.startswith(prefix) for prefix in _CARD_PREFIXES)


def iban_ok(candidate: str) -> bool:
    """ISO 13616 mod-97: move the first four characters to the end, then check."""
    body = candidate[4:] + candidate[:4]
    digits = []
    for char in body:
        if char.isdigit():
            digits.append(char)
        elif char.isalpha():
            digits.append(str(ord(char.upper()) - 55))
        else:
            return False
    try:
        return int("".join(digits)) % 97 == 1
    except ValueError:
        return False


def ssn_ok(area: str, group: str, serial: str) -> bool:
    if area in {"000", "666"} or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


def jwt_ok(token: str) -> bool:
    header = token.split(".", 1)[0]
    padded = header + "=" * (-len(header) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded)
        payload = json.loads(decoded)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return False
    return isinstance(payload, dict) and "alg" in payload


@dataclass(frozen=True, slots=True)
class _Pattern:
    kind: str
    regex: re.Pattern[str]
    severity: Severity
    max_match_len: int
    #: Returns the confidence to report, or ``None`` to reject the match.
    validate: Callable[[re.Match[str]], float | None]
    detail: str = ""


def _always(confidence: float) -> Callable[[re.Match[str]], float | None]:
    def check(_: re.Match[str]) -> float | None:
        return confidence

    return check


def _check_email(match: re.Match[str]) -> float | None:
    local, _, domain = match.group(0).partition("@")
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return None
    tld = domain.rsplit(".", 1)[-1]
    return None if tld.isdigit() else 0.97


def _check_card(match: re.Match[str]) -> float | None:
    digits = re.sub(r"[ -]", "", match.group(0))
    if not luhn_ok(digits) or not card_brand_ok(digits):
        return None
    return 0.98


def _check_ssn(match: re.Match[str]) -> float | None:
    area, separator, group, serial = match.group(1), match.group(2), match.group(3), match.group(4)
    if not ssn_ok(area, group, serial):
        return None
    # Nine bare digits are far more often an order number than a social security
    # number. A separator is what turns a shape match into a confident one.
    return 0.95 if separator else 0.55


def _check_ipv4(match: re.Match[str]) -> float | None:
    octets = match.group(0).split(".")
    if any(not 0 <= int(part) <= 255 or (len(part) > 1 and part[0] == "0") for part in octets):
        return None
    return 0.9


def _check_iban(match: re.Match[str]) -> float | None:
    return 0.99 if iban_ok(match.group(0)) else None


def _check_jwt(match: re.Match[str]) -> float | None:
    return 0.99 if jwt_ok(match.group(0)) else None


_PATTERNS: tuple[_Pattern, ...] = (
    _Pattern(
        kind="email",
        regex=re.compile(
            r"(?<![\w.+-])[A-Za-z0-9._%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,24}(?![\w-])"
        ),
        severity=Severity.MEDIUM,
        max_match_len=96,
        validate=_check_email,
    ),
    _Pattern(
        kind="credit_card",
        regex=re.compile(r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![\d\-])"),
        severity=Severity.CRITICAL,
        max_match_len=25,
        validate=_check_card,
        detail="passes Luhn and matches a known issuer prefix",
    ),
    _Pattern(
        kind="ssn",
        regex=re.compile(r"(?<!\d)(\d{3})([\- ]?)(\d{2})\2(\d{4})(?!\d)"),
        severity=Severity.HIGH,
        max_match_len=11,
        validate=_check_ssn,
    ),
    _Pattern(
        kind="phone",
        regex=re.compile(
            r"(?<![\d\-])(?:\+1[ .\-]?)?\(?[2-9]\d{2}\)?[ .\-]?[2-9]\d{2}[ .\-]?\d{4}(?![\d\-])"
        ),
        severity=Severity.MEDIUM,
        max_match_len=20,
        validate=_always(0.85),
    ),
    _Pattern(
        kind="ipv4",
        regex=re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"),
        severity=Severity.LOW,
        max_match_len=15,
        validate=_check_ipv4,
    ),
    _Pattern(
        kind="iban",
        regex=re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-Z0-9]{11,26}(?![A-Z0-9])"),
        severity=Severity.HIGH,
        max_match_len=34,
        validate=_check_iban,
        detail="passes the ISO 13616 mod-97 check",
    ),
    _Pattern(
        kind="aws_key",
        regex=re.compile(
            r"(?<![A-Z0-9])(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ABIA|ACCA)[A-Z0-9]{16}"
            r"(?![A-Z0-9])"
        ),
        severity=Severity.CRITICAL,
        max_match_len=20,
        validate=_always(0.99),
    ),
    _Pattern(
        kind="jwt",
        regex=re.compile(
            r"(?<![\w.\-])eyJ[A-Za-z0-9_\-]{6,120}\.[A-Za-z0-9_\-]{6,180}\.[A-Za-z0-9_\-]{0,86}"
            r"(?![\w.\-])"
        ),
        severity=Severity.CRITICAL,
        max_match_len=3 + 120 + 1 + 180 + 1 + 86,
        validate=_check_jwt,
        detail="header decodes to JSON carrying an 'alg' claim",
    ),
    _Pattern(
        kind="api_key",
        regex=re.compile(
            r"(?i)(?:api[_\- ]?key|secret[_\- ]?key|access[_\- ]?token|auth[_\- ]?token|password)"
            r"\s*[:=]\s*[\"']?([A-Za-z0-9_\-]{16,64})[\"']?"
        ),
        severity=Severity.CRITICAL,
        max_match_len=100,
        validate=_always(0.9),
        detail="anchored on a nearby key-like label, not on entropy alone",
    ),
    _Pattern(
        kind="private_key",
        regex=re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
        severity=Severity.CRITICAL,
        max_match_len=40,
        validate=_always(1.0),
    ),
)


class PiiDetector:
    """Scans for personal data and credentials with validated patterns."""

    name: str = "pii"
    kinds: tuple[str, ...] = tuple(pattern.kind for pattern in _PATTERNS)
    max_match_len: int = max(pattern.max_match_len for pattern in _PATTERNS)

    def __init__(self, kinds: tuple[str, ...] | None = None) -> None:
        """Restrict to ``kinds`` to shrink the streaming holdback window.

        The JWT pattern alone accounts for most of the default holdback, so a
        service that never handles tokens can cut its time-to-first-token by
        naming the kinds it actually needs.
        """
        if kinds is None:
            self._patterns = _PATTERNS
        else:
            unknown = sorted(set(kinds) - set(self.kinds))
            if unknown:
                raise ValueError(f"unknown pii kinds: {', '.join(unknown)}")
            self._patterns = tuple(p for p in _PATTERNS if p.kind in set(kinds))
        self.kinds = tuple(p.kind for p in self._patterns)
        self.max_match_len = max((p.max_match_len for p in self._patterns), default=1)

    def scan(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        for pattern in self._patterns:
            for match in pattern.regex.finditer(text):
                confidence = pattern.validate(match)
                if confidence is None:
                    continue
                # api_key reports the secret's span, not the label's, so that
                # redaction leaves "api_key=" readable in the audit trail.
                start, end = match.span(1) if pattern.kind == "api_key" else match.span(0)
                findings.append(
                    Finding(
                        detector=self.name,
                        kind=pattern.kind,
                        start=start,
                        end=end,
                        severity=pattern.severity,
                        confidence=confidence,
                        detail=pattern.detail,
                    )
                )
        findings.sort(key=lambda f: (f.start, -f.length))
        return findings
