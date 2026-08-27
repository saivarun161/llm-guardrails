"""Registering a detector written outside the package.

The point of these tests is the round trip: a custom detector should be nameable
in a YAML policy, resolvable to a rule, and counted in the streaming holdback,
without touching a line of the library. Before the registry existed the strict
policy loader rejected the label and the extension point was unreachable.
"""

from __future__ import annotations

import re

import pytest

from llmguard import Guard, StreamGuard, detectors
from llmguard import policy as policy_module
from llmguard.metrics import GuardMetrics
from llmguard.types import DetectorContractError, Finding, PolicyError, Severity

TICKET_RE = re.compile(r"\bINC-\d{6}\b")


class TicketDetector:
    """A house pattern: internal incident ids that must not reach a model."""

    name = "ticket"
    kinds = ("incident",)
    max_match_len = 10

    def scan(self, text: str) -> list[Finding]:
        return [
            Finding(
                detector="ticket",
                kind="incident",
                start=match.start(),
                end=match.end(),
                severity=Severity.MEDIUM,
                confidence=0.99,
            )
            for match in TICKET_RE.finditer(text)
        ]


POLICY = """
version: 1
name: with-a-custom-detector
input:
  rules:
    - id: redact-incident
      detect: ticket.incident
      action: redact
output:
  rules:
    - id: redact-incident-out
      detect: ticket
      action: redact
"""


# -- the round trip --------------------------------------------------------


def test_a_registered_detector_is_addressable_from_a_policy():
    detectors.register(TicketDetector)
    guard = Guard(policy_module.loads(POLICY), metrics=GuardMetrics())

    result = guard.check_input("please look at INC-004212 before the call")

    assert result.text == "please look at [INCIDENT] before the call"
    assert [d.rule_id for d in result.decisions] == ["redact-incident"]


def test_the_same_policy_is_rejected_before_the_detector_is_registered():
    with pytest.raises(PolicyError, match=re.escape("no detector produces 'ticket.incident'")):
        policy_module.loads(POLICY)


def test_a_typo_against_a_custom_detector_gets_the_same_suggestion_builtins_do():
    detectors.register(TicketDetector)
    with pytest.raises(PolicyError, match=re.escape("did you mean 'ticket.incident'")):
        policy_module.loads(POLICY.replace("ticket.incident", "ticket.incidnet"))


def test_a_custom_detector_widens_the_holdback_it_needs():
    detectors.register(TicketDetector)
    guard = Guard(policy_module.loads(POLICY), metrics=GuardMetrics())
    stream = StreamGuard(guard, "output", raise_on_block=False)

    assert stream.holdback == TicketDetector.max_match_len

    text = "incident INC-004212 is the one"
    batch = guard.check_output(text)
    for size in range(1, len(text) + 2):
        streamed = StreamGuard(guard, "output", raise_on_block=False)
        pieces = [streamed.feed(text[i : i + size]) for i in range(0, len(text), size)]
        pieces.append(streamed.close())
        assert "".join(pieces) == batch.text, f"chunk size {size} diverged"


def test_a_wildcard_policy_picks_up_whatever_is_registered():
    wildcard = policy_module.loads(
        "version: 1\nname: everything\ninput:\n  rules:\n"
        "    - id: all\n      detect: '*'\n      action: flag\n"
    )
    assert "ticket" not in wildcard.detectors_for("input")

    detectors.register(TicketDetector)
    assert "ticket" in wildcard.detectors_for("input")


def test_known_labels_and_build_both_see_the_registration():
    detectors.register(TicketDetector)
    assert "ticket" in detectors.known_labels()
    assert "ticket.incident" in detectors.known_labels()
    assert isinstance(detectors.build(["ticket"])["ticket"], TicketDetector)
    assert "ticket" in detectors.build()


# -- what registration refuses --------------------------------------------


def test_registering_twice_needs_an_explicit_replace():
    detectors.register(TicketDetector)
    with pytest.raises(DetectorContractError, match="already registered"):
        detectors.register(TicketDetector)
    assert detectors.register(TicketDetector, replace=True) == "ticket"


def test_a_builtin_is_not_shadowed_by_accident():
    class Impostor:
        name = "pii"
        kinds = ("email",)
        max_match_len = 4

        def scan(self, text: str) -> list[Finding]:
            return []

    with pytest.raises(DetectorContractError, match="already registered"):
        detectors.register(Impostor)


def test_registering_under_a_name_the_detector_does_not_answer_to_is_refused():
    """Findings carry ``detector=self.name``, so the rule would never match."""
    with pytest.raises(DetectorContractError, match="findings carry the detector's own name"):
        detectors.register(TicketDetector, name="incidents")


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("name", "Ticket", "must be a lowercase identifier"),
        ("name", "house.ticket", "without a dot"),
        ("name", "", "must be a lowercase identifier"),
        ("kinds", (), "non-empty tuple"),
        ("kinds", ["incident"], "non-empty tuple"),
        ("kinds", ("in.cident",), "not a valid label component"),
        ("kinds", ("incident", "incident"), "duplicates"),
        ("max_match_len", 0, "positive integer"),
        ("max_match_len", 10.0, "positive integer"),
        ("max_match_len", True, "positive integer"),
    ],
)
def test_a_malformed_detector_is_rejected_at_registration(attribute, value, message):
    broken = type("Broken", (TicketDetector,), {attribute: value})
    with pytest.raises(DetectorContractError, match=message):
        detectors.register(broken)


def test_a_detector_without_a_scan_method_is_rejected():
    broken = type("Broken", (), {"name": "x", "kinds": ("y",), "max_match_len": 1})
    with pytest.raises(DetectorContractError, match="no callable scan"):
        detectors.register(broken)


def test_a_factory_that_needs_arguments_says_so():
    class NeedsArgs(TicketDetector):
        def __init__(self, required):
            self.required = required

    with pytest.raises(DetectorContractError, match="callable with no arguments"):
        detectors.register(NeedsArgs)


# -- lifecycle -------------------------------------------------------------


def test_unregister_removes_it_and_unknown_names_raise():
    detectors.register(TicketDetector)
    detectors.unregister("ticket")
    assert "ticket" not in detectors.registered()
    with pytest.raises(KeyError):
        detectors.unregister("ticket")


def test_reset_restores_exactly_the_builtins():
    detectors.register(TicketDetector)
    detectors.unregister("pii")
    detectors.reset()
    assert detectors.registered() == detectors.BUILTIN


def test_registered_returns_a_copy_rather_than_the_live_mapping():
    snapshot = detectors.registered()
    snapshot["ticket"] = TicketDetector
    assert "ticket" not in detectors.registered()
