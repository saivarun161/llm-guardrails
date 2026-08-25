from __future__ import annotations

import pytest

from llmguard import policy as policy_module
from llmguard.budget import FAIL_CLOSED
from llmguard.types import Action, Finding, PolicyError, Severity, valid_label

MINIMAL = """
version: 1
name: minimal
input:
  rules:
    - id: redact-everything
      detect: "*"
      action: redact
"""


def finding(label="pii.email", severity=Severity.MEDIUM, confidence=0.9):
    detector, _, kind = label.partition(".")
    return Finding(
        detector=detector,
        kind=kind,
        start=0,
        end=1,
        severity=severity,
        confidence=confidence,
    )


def test_default_policy_loads():
    policy = policy_module.default()
    assert policy.name == "default"
    assert policy.input_rules and policy.output_rules
    assert policy.budget.total_ms == 50


def test_minimal_policy_round_trips():
    policy = policy_module.loads(MINIMAL)
    reloaded = policy_module.loads(policy.dump())
    assert reloaded.to_dict() == policy.to_dict()


@pytest.mark.parametrize(
    "document,message",
    [
        ("", "empty"),
        ("version: 2\nname: x\n", "expected 1"),
        ("name: x\n", "expected 1"),
        ("version: 1\nname: x\nnope: 1\n", "unknown key"),
        ("version: 1\nname: x\ninput:\n  rules: {}\n", "expected a list"),
        ("version: 1\nname: x\ninput:\n  rules:\n    - {}\n", "missing required key"),
        (
            "version: 1\nname: x\ninput:\n  rules:\n    - id: a\n      detect: pii\n"
            "      action: shred\n",
            "unknown action",
        ),
        (
            "version: 1\nname: x\ninput:\n  rules:\n    - id: a\n      detect: pii\n"
            "      action: redact\n      min_severity: catastrophic\n",
            "unknown severity",
        ),
        (
            "version: 1\nname: x\ninput:\n  rules:\n    - id: a\n      detect: pii\n"
            "      action: redact\n      redaction: vanish\n",
            "unknown strategy",
        ),
        (
            "version: 1\nname: x\ninput:\n  rules:\n    - id: a\n      detect: pii\n"
            "      action: flag\n      redaction: label\n",
            "only meaningful with action 'redact'",
        ),
        (
            "version: 1\nname: x\ninput:\n  rules:\n    - id: a\n      detect: pii\n"
            "      action: redact\n    - id: a\n      detect: injection\n      action: flag\n",
            "duplicate rule id",
        ),
        (
            "version: 1\nname: x\ninput:\n  rules:\n    - id: a\n      detect: pii\n"
            "      action: redact\n      min_confidence: 2\n",
            r"in \[0, 1\]",
        ),
        ("version: 1\nname: x\nbudget:\n  total_ms: -1\n", "positive number"),
        ("version: 1\nname: x\nbudget:\n  total_ms: 10\n", "on_exceeded"),
        (
            "version: 1\nname: x\nbudget:\n  total_ms: 10\n  on_exceeded: shrug\n",
            "unknown value",
        ),
        ("version: 1\nname: x\ndefaults:\n  redaction: vanish\n", "unknown strategy"),
        ("version: 1\nname: x\ninput: []\n", "expected a mapping"),
        ("[1, 2, 3]\n", "expected a mapping"),
        ("version: 1\nname: x\ninput:\n  rules: [1]\n", "expected a mapping"),
        ("version: 1\nname: [unclosed\n", "not valid YAML"),
    ],
)
def test_malformed_policies_are_rejected_with_a_useful_message(document, message):
    with pytest.raises(PolicyError, match=message):
        policy_module.loads(document)


def test_unknown_detector_suggests_the_near_miss():
    document = (
        "version: 1\nname: x\ninput:\n  rules:\n"
        "    - id: a\n      detect: pii.emial\n      action: redact\n"
    )
    with pytest.raises(PolicyError, match=r"did you mean 'pii\.email'"):
        policy_module.loads(document)


def test_a_detector_that_does_not_exist_is_named_in_the_error():
    document = (
        "version: 1\nname: x\ninput:\n  rules:\n"
        "    - id: a\n      detect: telepathy\n      action: redact\n"
    )
    with pytest.raises(PolicyError, match="no detector produces 'telepathy'"):
        policy_module.loads(document)


@pytest.mark.parametrize(
    "label,ok",
    [
        ("pii", True),
        ("pii.email", True),
        ("*", True),
        ("Pii", False),
        ("a.b.c", False),
        ("", False),
    ],
)
def test_valid_label(label, ok):
    assert valid_label(label) is ok


def test_more_specific_rules_win():
    document = (
        "version: 1\nname: x\ninput:\n  rules:\n"
        "    - id: broad\n      detect: '*'\n      action: flag\n"
        "    - id: mid\n      detect: pii\n      action: redact\n"
        "    - id: narrow\n      detect: pii.email\n      action: block\n"
    )
    policy = policy_module.loads(document)
    assert policy.resolve("input", finding("pii.email")).id == "narrow"
    assert policy.resolve("input", finding("pii.ssn")).id == "mid"
    assert policy.resolve("input", finding("injection.prompt_injection")).id == "broad"


def test_severity_and_confidence_thresholds_filter():
    document = (
        "version: 1\nname: x\ninput:\n  rules:\n"
        "    - id: strict\n      detect: pii\n      action: block\n"
        "      min_severity: critical\n      min_confidence: 0.95\n"
    )
    policy = policy_module.loads(document)
    assert policy.resolve("input", finding(severity=Severity.MEDIUM)) is None
    assert policy.resolve("input", finding(severity=Severity.CRITICAL, confidence=0.5)) is None
    assert policy.resolve("input", finding(severity=Severity.CRITICAL, confidence=0.99)) is not None


def test_detectors_for_stage_only_lists_what_the_rules_use():
    policy = policy_module.loads(MINIMAL)
    assert set(policy.detectors_for("input")) == {"pii", "injection"}

    narrow = policy_module.loads(
        "version: 1\nname: x\noutput:\n  rules:\n"
        "    - id: a\n      detect: pii.email\n      action: redact\n"
    )
    assert narrow.detectors_for("output") == ("pii",)
    assert narrow.detectors_for("input") == ()


def test_unknown_stage_is_an_error():
    with pytest.raises(PolicyError, match="unknown stage"):
        policy_module.default().rules_for("middle")


def test_load_from_disk(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    assert policy_module.load(path).name == "minimal"


def test_missing_file_is_a_policy_error(tmp_path):
    with pytest.raises(PolicyError, match="cannot read policy"):
        policy_module.load(tmp_path / "absent.yaml")


def test_name_defaults_to_the_filename(tmp_path):
    path = tmp_path / "house-rules.yaml"
    path.write_text("version: 1\ninput:\n  rules: []\n", encoding="utf-8")
    assert policy_module.load(path).name == "house-rules"


def test_fail_closed_budget_parses():
    policy = policy_module.loads(
        "version: 1\nname: x\nbudget:\n  total_ms: 5\n  on_exceeded: fail_closed\n"
    )
    assert policy.budget.on_exceeded == FAIL_CLOSED


def test_default_policy_blocks_injection_and_redacts_pii():
    policy = policy_module.default()
    injection = finding("injection.prompt_injection", severity=Severity.HIGH, confidence=0.8)
    assert policy.resolve("input", injection).action is Action.BLOCK

    email = finding("pii.email")
    assert policy.resolve("input", email).action is Action.REDACT

    key = finding("pii.aws_key", severity=Severity.CRITICAL, confidence=0.99)
    assert policy.resolve("output", key).action is Action.BLOCK
