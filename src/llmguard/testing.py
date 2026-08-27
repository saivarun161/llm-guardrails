"""A conformance harness for detectors, meant to be run from your test suite.

The streaming guarantee is not a property of the streaming code. It is a
property of the streaming code *and every detector it runs*. ``StreamGuard``
withholds the last ``max_match_len`` characters of the buffer, so the arithmetic
that makes a boundary leak impossible holds exactly as long as no detector
returns a span wider than the number it declared. A detector that understates
its bound does not produce slightly different output -- it leaks, in the one
place this library exists to be trustworthy.

The builtin detectors are held to that in this repository's own suite. A
detector written anywhere else needs the same treatment, and asking every author
to reimplement the fuzz is asking for it not to happen. So the harness is part
of the library:

    from llmguard.testing import assert_detector_contract

    def test_my_detector_keeps_its_promises():
        assert_detector_contract(MyDetector())

That runs a corpus chosen to break the assumptions detectors usually make --
invisible characters, combining marks, characters that change length under NFKC,
text outside the basic plane -- and, for each case, drives the real streaming
guard at every chunk size and checks the output is identical to the batch one.
Pass your own corpus as well; the built-in cases are adversarial, not
representative, and they know nothing about what you are detecting.

Violations are returned as data by :func:`check_detector_contract` and raised as
one exception by :func:`assert_detector_contract`. Neither ever puts matched
text in the report: a corpus is usually built from real examples of the thing
being detected, and a test runner's output is not a place to print those. A
violation carries the case name and the offsets, which is enough to reproduce it.

No pytest dependency: this is plain functions and one exception, usable from any
runner.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .budget import Budget
from .detectors import validate_shape
from .engine import Guard
from .metrics import GuardMetrics
from .policy import Policy, Rule
from .streaming import StreamGuard
from .types import Action, DetectorContractError, Finding

#: Cases chosen because they break something. Each is a name and a text.
#:
#: A detector that matches on a normalised copy of the text has to map offsets
#: back, and every folding rule that changes length is a chance to get that
#: wrong: ``ﬁ`` is one character that becomes two under NFKC, full-width
#: digits fold to ASCII ones, and an emoji outside the basic plane is one
#: character in Python and two UTF-16 code units in the language the pattern was
#: probably ported from. The long runs are here because a bound is easiest to
#: exceed when a match is allowed to grow, and the repeated pairs because a
#: chunk boundary lands inside a match far more often when every second position
#: could start one.
CORPUS: tuple[tuple[str, str], ...] = (
    ("empty", ""),
    ("one-char", "x"),
    ("prose", "The quarterly report is attached; let me know if anything looks off."),
    # Written as escapes rather than as the characters themselves. A corpus of
    # invisible and confusable characters is unreadable in source, and a reviewer
    # cannot tell a deliberate zero-width space from one pasted in by accident.
    ("zero-width", "se\u200bcr\u200bet va\u200blue in\u200bside"),
    ("combining-marks", "cre\u0301dit ca\u0301rd nu\u0301mber"),
    ("full-width-digits", "card \uff14\uff12\uff14\uff12\uff14\uff12\uff14\uff12 here"),
    ("nfkc-expands", "the \ufb01le is \ufb01ne"),
    ("astral", "ok \U0001f600 fine \U0001f4a9 done"),
    ("crlf", "line one\r\nline two\r\n\r\nline three"),
    ("control-chars", "a\tb\x0bc\x0cd e"),
    ("long-run", "a" * 800),
    ("boundary-repeats", "ab" * 400),
    ("digit-run", "1234567890" * 60),
    ("base64ish", "payload: " + "aGVsbG8gd29ybGQ" * 40 + " end"),
    ("mixed", "\u200bab" * 100 + "\ufb01" * 50 + "\U0001f600" * 20),
)

#: Chunk sizes always tried, on top of the ones derived from the bound.
_BASE_CHUNK_SIZES = (1, 2, 3, 5, 8, 13, 21, 64)


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken promise, located well enough to reproduce without the text."""

    rule: str
    detail: str
    case: str = ""

    def __str__(self) -> str:
        where = f" [case {self.case!r}]" if self.case else ""
        return f"{self.rule}: {self.detail}{where}"


def check_detector_contract(
    detector: object,
    corpus: Iterable[tuple[str, str]] | Mapping[str, str] | Iterable[str] | None = None,
    *,
    chunk_sizes: Sequence[int] | None = None,
    include_default_corpus: bool = True,
) -> list[Violation]:
    """Run every contract check and return the violations, worst first.

    ``corpus`` accepts ``{name: text}``, ``[(name, text), ...]`` or a plain list
    of strings (named by position). It is added to :data:`CORPUS` unless
    ``include_default_corpus`` is false.

    An empty list means the detector kept its promises on this corpus. It does
    not mean the bound is provably right -- that would need every input -- but an
    understated bound survives surprisingly few adversarial cases.
    """
    violations: list[Violation] = []
    try:
        validate_shape(detector)
    except DetectorContractError as exc:
        # Everything below reads name, kinds or max_match_len, so there is
        # nothing meaningful left to check once the shape is wrong.
        return [Violation("shape", str(exc))]

    cases = list(CORPUS) if include_default_corpus else []
    cases.extend(_normalise_corpus(corpus))

    results: dict[str, list[Finding]] = {}
    for name, text in cases:
        try:
            findings = list(detector.scan(text))  # type: ignore[attr-defined]
        except Exception as exc:
            violations.append(
                Violation("raises", f"scan() raised {type(exc).__name__}: {exc}", name)
            )
            continue
        results[name] = findings
        violations.extend(_check_findings(detector, name, text, findings))

    violations.extend(_check_repeatable(detector, cases, results))
    for name, text in cases:
        if name in results:
            violations.extend(_check_streaming(detector, name, text, chunk_sizes))

    order = {
        "shape": 0,
        "raises": 1,
        "bound": 2,
        "stream": 3,
        "leak": 4,
        "span": 5,
        "label": 6,
        "secrecy": 7,
        "repeatable": 8,
        "stateless": 9,
    }
    violations.sort(key=lambda v: (order.get(v.rule, 99), v.case, v.detail))
    return violations


def assert_detector_contract(
    detector: object,
    corpus: Iterable[tuple[str, str]] | Mapping[str, str] | Iterable[str] | None = None,
    **kwargs: object,
) -> None:
    """:func:`check_detector_contract`, raising instead of returning."""
    violations = check_detector_contract(detector, corpus, **kwargs)  # type: ignore[arg-type]
    if not violations:
        return
    name = getattr(detector, "name", type(detector).__name__)
    lines = "\n".join(f"  - {violation}" for violation in violations)
    raise DetectorContractError(
        f"detector {name!r} broke {len(violations)} contract check(s):\n{lines}"
    )


# -- the individual checks -------------------------------------------------


def _check_findings(
    detector: object, case: str, text: str, findings: list[Finding]
) -> list[Violation]:
    """Everything decidable from one scan of one text."""
    out: list[Violation] = []
    name = detector.name  # type: ignore[attr-defined]
    kinds = set(detector.kinds)  # type: ignore[attr-defined]
    bound = detector.max_match_len  # type: ignore[attr-defined]

    for finding in findings:
        if not isinstance(finding, Finding):
            out.append(Violation("span", f"scan() returned a {type(finding).__name__}", case))
            continue
        where = f"[{finding.start}:{finding.end}]"
        if finding.end > len(text):
            out.append(
                Violation(
                    "span", f"{finding.label} {where} runs past the {len(text)}-char text", case
                )
            )
            continue
        if finding.detector != name:
            out.append(
                Violation(
                    "label",
                    f"finding {where} says detector={finding.detector!r}, "
                    f"but the detector is named {name!r}, so no policy rule can address it",
                    case,
                )
            )
        if finding.kind not in kinds:
            out.append(
                Violation(
                    "label",
                    f"finding {where} has kind {finding.kind!r}, which is not in kinds; "
                    "a policy naming it would be rejected at load time",
                    case,
                )
            )
        if finding.length > bound:
            out.append(
                Violation(
                    "bound",
                    f"{finding.label} {where} is {finding.length} characters, over the declared "
                    f"max_match_len of {bound}. The streaming holdback is derived from that "
                    "number, so a wider match can straddle the emit boundary and leak.",
                    case,
                )
            )
        matched = text[finding.start : finding.end]
        if len(matched) >= 8 and matched in _rendered(finding):
            out.append(
                Violation(
                    "secrecy",
                    f"{finding.label} {where} carries its own matched text into to_dict(); "
                    "findings are written to audit logs, and the matched text is the thing "
                    "the guard exists to keep out of them",
                    case,
                )
            )
    return out


def _check_repeatable(
    detector: object, cases: list[tuple[str, str]], first: dict[str, list[Finding]]
) -> list[Violation]:
    """Same text, same findings -- before and after scanning everything else.

    Detectors are documented as stateless, and the streaming guard leans on it:
    it rescans an overlapping window on every chunk, so a detector that carried
    state between calls would give a different answer to the same characters
    depending on how the stream happened to be split.
    """
    out: list[Violation] = []
    for name, text in cases:
        if name not in first:
            continue
        again, failure = _rescan(detector, text)
        if failure is not None:
            out.append(Violation("raises", f"scanning the same text again {failure}", name))
        elif again != first[name]:
            out.append(
                Violation(
                    "repeatable", "scanning the same text twice gave different findings", name
                )
            )

    for name, text in reversed(cases):
        if name not in first:
            continue
        again, failure = _rescan(detector, text)
        if failure is not None:
            continue  # already reported by the pass above
        if again != first[name]:
            out.append(
                Violation(
                    "stateless",
                    "findings changed after other texts were scanned, so the detector "
                    "carries state between calls",
                    name,
                )
            )
    return out


def _rescan(detector: object, text: str) -> tuple[list[Finding], str | None]:
    """Scan again, turning a raise into a reportable string.

    A detector that blows up on a repeat scan is a violation to report, not an
    exception to propagate: the harness runs inside someone's test suite, and a
    traceback out of the harness itself reads as a bug in the harness.
    """
    try:
        return list(detector.scan(text)), None  # type: ignore[attr-defined]
    except Exception as exc:
        return [], f"raised {type(exc).__name__}: {exc}"


def _check_streaming(
    detector: object, case: str, text: str, chunk_sizes: Sequence[int] | None
) -> list[Violation]:
    """Drive the real streaming guard and compare against the batch result.

    This is the check the others exist to support. A bound can be understated in
    ways no single scan reveals -- the span is only too wide for text the corpus
    happens to include -- and this is where that shows up, as streamed output
    that differs from what the same guard produces on the whole string.
    """
    out: list[Violation] = []
    bound = detector.max_match_len  # type: ignore[attr-defined]
    guard = _contract_guard(detector)

    try:
        batch = guard.check_output(text)
    except Exception as exc:
        return [Violation("stream", f"the batch check raised {type(exc).__name__}: {exc}", case)]

    for size in _chunk_sizes(text, bound, chunk_sizes):
        stream = StreamGuard(guard, "output", raise_on_block=False)
        try:
            pieces = [stream.feed(text[i : i + size]) for i in range(0, len(text), size)]
            pieces.append(stream.close())
        except Exception as exc:
            out.append(
                Violation("stream", f"chunk size {size} raised {type(exc).__name__}: {exc}", case)
            )
            break
        if "".join(pieces) != batch.text:
            out.append(
                Violation(
                    "stream",
                    f"chunk size {size} produced different text from the non-streaming check; "
                    f"a match is escaping the {bound}-character holdback window",
                    case,
                )
            )
            break

    leaks = guard.metrics.stream_leaks.value()
    if leaks:
        out.append(
            Violation(
                "leak",
                f"{leaks:g} finding(s) were detected only after their text had been emitted, "
                "which is what an understated max_match_len looks like at runtime",
                case,
            )
        )
    return out


# -- helpers ---------------------------------------------------------------


def _contract_guard(detector: object) -> Guard:
    """A guard running only this detector, redacting everything it reports.

    Redact rather than flag, because a flag would leave the text untouched and
    the streaming comparison would pass on any detector at all. No budget: an
    overrun makes the stream defer, which is correct behaviour and would show up
    here as a spurious difference.
    """
    name = detector.name  # type: ignore[attr-defined]
    policy = Policy(
        name="detector-contract",
        output_rules=(Rule(id="redact-everything", detect=name, action=Action.REDACT),),
        budget=Budget(),
    )
    return Guard(policy, metrics=GuardMetrics(), detectors={name: detector})  # type: ignore[dict-item]


def _chunk_sizes(text: str, bound: int, explicit: Sequence[int] | None) -> list[int]:
    """Small sizes, the sizes either side of the bound, and the whole text.

    Every size from one upwards is what this repository's own suite does, and it
    is affordable there because the fixtures are short. A harness other people
    run on their own corpus cannot assume that, so it takes the sizes where
    boundary bugs actually live: the small ones, where a match is split many
    times, and the ones around the holdback, where the window arithmetic changes.
    """
    if explicit is not None:
        return [size for size in explicit if size > 0]
    length = max(len(text), 1)
    sizes = {size for size in _BASE_CHUNK_SIZES if size <= length}
    sizes.update(size for size in (bound - 1, bound, bound + 1) if 0 < size <= length)
    sizes.add(length)
    return sorted(sizes)


def _rendered(finding: Finding) -> str:
    payload = finding.to_dict()
    return " ".join(str(value) for value in payload.values())


def _normalise_corpus(
    corpus: Iterable[tuple[str, str]] | Mapping[str, str] | Iterable[str] | None,
) -> list[tuple[str, str]]:
    if corpus is None:
        return []
    if isinstance(corpus, Mapping):
        return [(str(name), text) for name, text in corpus.items()]
    out: list[tuple[str, str]] = []
    for position, entry in enumerate(corpus):
        if isinstance(entry, str):
            out.append((f"custom[{position}]", entry))
        else:
            name, text = entry
            out.append((str(name), text))
    return out


def normalised_length_changes(text: str) -> bool:
    """Whether NFKC folding changes this text's length.

    Exposed because it is the single most common reason a detector's offsets
    drift: the pattern matches on a folded copy and the span is mapped back with
    arithmetic that assumed the two were the same length.
    """
    return len(unicodedata.normalize("NFKC", text)) != len(text)


__all__ = [
    "CORPUS",
    "DetectorContractError",
    "Violation",
    "assert_detector_contract",
    "check_detector_contract",
    "normalised_length_changes",
]
