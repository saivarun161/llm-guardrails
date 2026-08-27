"""The conformance harness, tested against detectors built to break it.

A check that never fires is decoration, so every rule the harness can report has
a detector here that triggers it. The first two tests are the ones that matter
in the other direction: the detectors that ship with the library must pass their
own harness, or it is measuring nothing.
"""

from __future__ import annotations

import re

import pytest

from llmguard.detectors.injection import InjectionDetector
from llmguard.detectors.pii import PiiDetector
from llmguard.testing import (
    CORPUS,
    Violation,
    assert_detector_contract,
    check_detector_contract,
    normalised_length_changes,
)
from llmguard.types import DetectorContractError, Finding, Severity

SECRET_RE = re.compile(r"secret\d\d")
CORPUS_WITH_SECRET = {"planted": "a secret42 in the middle of some text", "clean": "nothing here"}


class Honest:
    """The shape every broken detector below is a mutation of."""

    name = "house"
    kinds = ("secret",)
    max_match_len = 8

    def scan(self, text: str) -> list[Finding]:
        return [
            Finding(
                detector=self.name,
                kind="secret",
                start=match.start(),
                end=match.end(),
                severity=Severity.HIGH,
                confidence=0.9,
            )
            for match in SECRET_RE.finditer(text)
        ]


def rules(violations: list[Violation]) -> set[str]:
    return {violation.rule for violation in violations}


# -- the detectors that ship must pass -------------------------------------


@pytest.mark.parametrize(
    "detector",
    [
        pytest.param(PiiDetector(), id="pii"),
        pytest.param(PiiDetector(kinds=("email", "credit_card")), id="pii-narrowed"),
        pytest.param(InjectionDetector(), id="injection"),
    ],
)
def test_the_builtin_detectors_keep_their_own_contract(detector):
    assert check_detector_contract(detector) == []


def test_an_honest_custom_detector_passes():
    assert_detector_contract(Honest(), CORPUS_WITH_SECRET)


# -- every rule the harness can report -------------------------------------


def test_an_understated_bound_is_caught_as_both_a_bound_and_a_stream_violation():
    """The failure the whole harness exists for.

    Declaring 4 while returning 8 means the holdback is half what the match
    needs, so the front of the match is emitted before anything identifies it --
    which is exactly what the streaming comparison sees.
    """
    understated = type("Understated", (Honest,), {"max_match_len": 4})()
    violations = check_detector_contract(understated, CORPUS_WITH_SECRET)

    assert "bound" in rules(violations)
    assert "stream" in rules(violations)
    bound = next(v for v in violations if v.rule == "bound")
    assert "over the declared max_match_len of 4" in bound.detail
    assert bound.case == "planted"


def test_a_span_running_past_the_end_of_the_text_is_caught():
    class Overrunning(Honest):
        def scan(self, text: str) -> list[Finding]:
            return [
                Finding(
                    detector=self.name,
                    kind="secret",
                    start=f.start,
                    end=f.end + 40,
                    severity=f.severity,
                    confidence=f.confidence,
                )
                for f in super().scan(text)
            ]

    violations = check_detector_contract(Overrunning(), CORPUS_WITH_SECRET)
    assert "span" in rules(violations)
    assert "runs past the 37-char text" in next(v for v in violations if v.rule == "span").detail
    # An unusable span stops the per-finding checks there: slicing the text to
    # test anything else about it would just report the harness's own confusion.
    assert "secrecy" not in rules(violations)


def test_an_undeclared_kind_is_caught():
    class Undeclared(Honest):
        def scan(self, text: str) -> list[Finding]:
            return [
                Finding(
                    detector=self.name,
                    kind="token",
                    start=f.start,
                    end=f.end,
                    severity=f.severity,
                    confidence=f.confidence,
                )
                for f in super().scan(text)
            ]

    violations = check_detector_contract(Undeclared(), CORPUS_WITH_SECRET)
    assert "label" in rules(violations)
    assert "rejected at load time" in next(v for v in violations if v.rule == "label").detail


def test_a_finding_attributed_to_another_detector_is_caught():
    class Misattributed(Honest):
        def scan(self, text: str) -> list[Finding]:
            return [
                Finding(
                    detector="somebody_else",
                    kind="secret",
                    start=f.start,
                    end=f.end,
                    severity=f.severity,
                    confidence=f.confidence,
                )
                for f in super().scan(text)
            ]

    violations = check_detector_contract(Misattributed(), CORPUS_WITH_SECRET)
    assert "label" in rules(violations)
    label = next(v for v in violations if v.rule == "label")
    assert "no policy rule can address it" in label.detail


def test_a_finding_carrying_its_own_matched_text_is_caught():
    """``Finding`` is designed to be logged, so it must not contain the secret."""

    class Chatty(Honest):
        def scan(self, text: str) -> list[Finding]:
            return [
                Finding(
                    detector=self.name,
                    kind="secret",
                    start=f.start,
                    end=f.end,
                    severity=f.severity,
                    confidence=f.confidence,
                    detail=f"matched {text[f.start : f.end]}",
                )
                for f in super().scan(text)
            ]

    violations = check_detector_contract(Chatty(), CORPUS_WITH_SECRET)
    assert "secrecy" in rules(violations)


def test_a_detector_that_carries_state_between_calls_is_caught():
    """Deduplicating across calls is the tempting version of this mistake.

    It looks like a sensible optimisation and it breaks streaming outright: the
    guard rescans an overlapping window on every chunk, so a detector that
    reports each span only once would find a secret in one window and stay quiet
    about it in the next.
    """

    class Deduplicating(Honest):
        def __init__(self) -> None:
            self.reported: set[tuple[int, int]] = set()

        def scan(self, text: str) -> list[Finding]:
            out = []
            for finding in super().scan(text):
                if (finding.start, finding.end) in self.reported:
                    continue
                self.reported.add((finding.start, finding.end))
                out.append(finding)
            return out

    violations = check_detector_contract(Deduplicating(), CORPUS_WITH_SECRET)
    assert "repeatable" in rules(violations)


def test_a_detector_that_raises_is_reported_rather_than_crashing_the_harness():
    class Explodes(Honest):
        def scan(self, text: str) -> list[Finding]:
            if "​" in text:
                raise ValueError("zero-width characters are not handled")
            return super().scan(text)

    violations = check_detector_contract(Explodes(), CORPUS_WITH_SECRET)
    assert "raises" in rules(violations)
    assert any("zero-width characters are not handled" in v.detail for v in violations)


def test_something_that_is_not_a_finding_is_reported_rather_than_crashing():
    class Loose(Honest):
        def scan(self, text: str) -> list[Finding]:
            return [*super().scan(text), {"kind": "secret", "start": 0, "end": 8}]

    violations = check_detector_contract(Loose(), CORPUS_WITH_SECRET)
    assert any(v.rule == "span" and "returned a dict" in v.detail for v in violations)


def test_a_detector_that_only_fails_on_short_windows_is_caught_by_the_stream_pass():
    """Scanning the whole text is not evidence of scanning a window of it.

    A detector indexing without bounds checks passes every batch scan and then
    raises the first time the streaming guard hands it a partial buffer, which
    is why the harness drives the real stream rather than trusting one scan.
    """

    class Brittle(Honest):
        def scan(self, text: str) -> list[Finding]:
            if len(text) < 5:
                raise IndexError("assumed at least five characters")
            return super().scan(text)

    violations = check_detector_contract(Brittle(), CORPUS_WITH_SECRET)
    assert any(v.rule == "stream" and "IndexError" in v.detail for v in violations)


def test_a_detector_that_fails_on_a_repeat_scan_is_reported_not_propagated():
    class Fragile(Honest):
        def __init__(self) -> None:
            self.seen: set[str] = set()

        def scan(self, text: str) -> list[Finding]:
            if text in self.seen:
                raise RuntimeError("scanned twice")
            self.seen.add(text)
            return super().scan(text)

    violations = check_detector_contract(Fragile(), CORPUS_WITH_SECRET)
    assert any("scanned twice" in v.detail for v in violations)


def test_a_wrong_shape_short_circuits_to_a_single_violation():
    """Every other check reads name, kinds or max_match_len, so there is nothing
    left to say once those are wrong -- and a page of downstream noise would bury
    the one line that matters."""
    shapeless = type("Shapeless", (Honest,), {"max_match_len": 0})()
    violations = check_detector_contract(shapeless, CORPUS_WITH_SECRET)
    assert [v.rule for v in violations] == ["shape"]
    assert "positive integer" in violations[0].detail


# -- the harness API -------------------------------------------------------


def test_assert_raises_with_every_violation_listed():
    understated = type("Understated", (Honest,), {"max_match_len": 4})()
    with pytest.raises(DetectorContractError) as caught:
        assert_detector_contract(understated, CORPUS_WITH_SECRET)

    message = str(caught.value)
    assert "'house' broke" in message
    assert "max_match_len" in message
    assert message.count("\n  - ") >= 2


def test_the_report_never_contains_the_matched_text():
    """A corpus is usually built from real examples of the thing being detected."""
    understated = type("Understated", (Honest,), {"max_match_len": 4})()
    with pytest.raises(DetectorContractError) as caught:
        assert_detector_contract(understated, {"planted": "a secret42 in some text"})
    assert "secret42" not in str(caught.value)


@pytest.mark.parametrize(
    ("corpus", "expected_case"),
    [
        pytest.param({"named": "a secret42 here"}, "named", id="mapping"),
        pytest.param([("pair", "a secret42 here")], "pair", id="pairs"),
        pytest.param(["a secret42 here"], "custom[0]", id="bare-strings"),
    ],
)
def test_a_corpus_can_be_given_in_three_shapes(corpus, expected_case):
    understated = type("Understated", (Honest,), {"max_match_len": 4})()
    violations = check_detector_contract(understated, corpus)
    assert any(v.case == expected_case for v in violations)


def test_the_default_corpus_can_be_dropped():
    calls: list[str] = []

    class Recording(Honest):
        def scan(self, text: str) -> list[Finding]:
            calls.append(text)
            return super().scan(text)

    check_detector_contract(Recording(), ["only this"], include_default_corpus=False)
    # The streaming pass scans windows rather than the whole string, so the
    # assertion is that nothing outside the given corpus was ever looked at.
    assert calls
    assert all(scanned in "only this" for scanned in calls)


def test_explicit_chunk_sizes_are_honoured():
    sizes: list[int] = []

    class Recording(Honest):
        def scan(self, text: str) -> list[Finding]:
            sizes.append(len(text))
            return super().scan(text)

    check_detector_contract(
        Recording(), ["abcdefghij"], include_default_corpus=False, chunk_sizes=[10]
    )
    # One batch scan of the whole string, then one stream that never has to
    # split it. A sweep would have scanned many shorter windows.
    assert max(sizes) == 10


def test_the_default_corpus_exercises_the_cases_it_claims_to():
    names = {name for name, _ in CORPUS}
    assert {"empty", "zero-width", "astral", "long-run", "boundary-repeats"} <= names
    assert any(normalised_length_changes(text) for _, text in CORPUS)
    assert any(len(text) > 500 for _, text in CORPUS)


def test_a_violation_renders_its_case():
    assert str(Violation("bound", "too wide", "planted")) == "bound: too wide [case 'planted']"
    assert str(Violation("shape", "no kinds")) == "shape: no kinds"
