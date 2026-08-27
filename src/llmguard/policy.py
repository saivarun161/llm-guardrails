"""The declarative policy: what gets checked, and what happens when it fires.

Policies are YAML because the people who decide whether card numbers may reach a
model are frequently not the people who deploy the service, and a config file is
reviewable by both. That only works if the file is *strict*. A policy engine that
shrugs at ``pii.emial`` and quietly never matches is worse than no policy engine,
because it reports success.

So loading validates everything: unknown keys, unknown detectors (with a "did you
mean"), unknown actions and severities, duplicate rule ids, redaction strategies
that only make sense with a redact action, and a budget that names an overrun
behaviour that does not exist. Errors carry the path into the document --
``input.rules[2].action`` -- so the message points at the line rather than at the
file.

Rule matching is longest-prefix: a rule for ``pii.credit_card`` wins over a rule
for ``pii``, which wins over ``*``. Within the same specificity the first rule in
the file wins, so the reading order matches the precedence order.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import detectors as detector_registry
from .budget import ON_EXCEEDED, Budget
from .redact import STRATEGIES
from .types import Action, Finding, PolicyError, Severity, valid_label

STAGES = ("input", "output")

_RULE_KEYS = {"id", "detect", "action", "min_severity", "min_confidence", "redaction", "reason"}
_STAGE_KEYS = {"rules"}
_ROOT_KEYS = {"version", "name", "description", "defaults", "budget", "input", "output"}
_DEFAULTS_KEYS = {"redaction", "hash_salt"}
_BUDGET_KEYS = {"total_ms", "on_exceeded"}

#: Bumped only for a breaking change to the schema. A policy written for v1 must
#: keep loading, so new keys are optional and this stays 1.
SCHEMA_VERSION = 1

DEFAULT_POLICY_PATH = Path(__file__).parent / "policies" / "default.yaml"


@dataclass(frozen=True, slots=True)
class Rule:
    """One line of policy: match some findings, do something about them."""

    id: str
    detect: str
    action: Action
    min_severity: Severity = Severity.INFO
    min_confidence: float = 0.0
    redaction: str | None = None
    reason: str = ""

    @property
    def specificity(self) -> int:
        if self.detect == "*":
            return 0
        return 2 if "." in self.detect else 1

    def matches(self, finding: Finding) -> bool:
        if finding.severity < self.min_severity or finding.confidence < self.min_confidence:
            return False
        if self.detect == "*":
            return True
        if "." in self.detect:
            return finding.label == self.detect
        return finding.detector == self.detect

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "detect": self.detect,
            "action": self.action.label,
        }
        if self.min_severity is not Severity.INFO:
            payload["min_severity"] = self.min_severity.label
        if self.min_confidence:
            payload["min_confidence"] = self.min_confidence
        if self.redaction:
            payload["redaction"] = self.redaction
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class Policy:
    name: str
    version: int = SCHEMA_VERSION
    description: str = ""
    input_rules: tuple[Rule, ...] = ()
    output_rules: tuple[Rule, ...] = ()
    budget: Budget = field(default_factory=Budget)
    default_redaction: str = "label"
    hash_salt: str = ""

    def rules_for(self, stage: str) -> tuple[Rule, ...]:
        if stage not in STAGES:
            raise PolicyError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
        return self.input_rules if stage == "input" else self.output_rules

    def detectors_for(self, stage: str) -> tuple[str, ...]:
        """Only the detectors some rule in this stage can actually use.

        A policy that never mentions injection does not pay for injection
        scanning, and -- because the streaming holdback is derived from the active
        detector set -- also gets a shorter holdback.

        ``*`` means everything currently registered, custom detectors included.
        The corollary is that registering a detector widens the holdback of every
        ``*`` policy, which is the honest answer: the window has to cover what is
        actually being detected.
        """
        names: list[str] = []
        for rule in self.rules_for(stage):
            if rule.detect == "*":
                return tuple(detector_registry.registered())
            head = rule.detect.split(".", 1)[0]
            if head not in names:
                names.append(head)
        return tuple(names)

    def resolve(self, stage: str, finding: Finding) -> Rule | None:
        """The one rule that governs this finding, or ``None`` if none match."""
        best: Rule | None = None
        for rule in self.rules_for(stage):
            if not rule.matches(finding):
                continue
            if best is None or rule.specificity > best.specificity:
                best = rule
        return best

    def redaction_for(self, rule: Rule) -> str:
        return rule.redaction or self.default_redaction

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "name": self.name,
        }
        if self.description:
            payload["description"] = self.description
        payload["defaults"] = {"redaction": self.default_redaction}
        if not self.budget.unlimited:
            payload["budget"] = {
                "total_ms": self.budget.total_ms,
                "on_exceeded": self.budget.on_exceeded,
            }
        payload["input"] = {"rules": [rule.to_dict() for rule in self.input_rules]}
        payload["output"] = {"rules": [rule.to_dict() for rule in self.output_rules]}
        return payload

    def dump(self) -> str:
        stream = io.StringIO()
        yaml.safe_dump(self.to_dict(), stream, sort_keys=False, default_flow_style=False)
        return stream.getvalue()


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError(f"{path}: expected a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise PolicyError(
            f"{path}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
        )


def _parse_rule(raw: Any, path: str, seen: set[str]) -> Rule:
    mapping = _require_mapping(raw, path)
    _reject_unknown(mapping, _RULE_KEYS, path)

    for required in ("id", "detect", "action"):
        if required not in mapping:
            raise PolicyError(f"{path}: missing required key {required!r}")

    rule_id = str(mapping["id"])
    if rule_id in seen:
        raise PolicyError(f"{path}.id: duplicate rule id {rule_id!r}")
    seen.add(rule_id)

    detect = str(mapping["detect"])
    if not valid_label(detect):
        raise PolicyError(
            f"{path}.detect: {detect!r} is not a valid label; expected 'detector', "
            "'detector.kind' or '*'"
        )
    if detect != "*" and detect not in detector_registry.known_labels():
        hint = detector_registry.suggest(detect)
        suffix = f"; did you mean {hint!r}?" if hint else ""
        raise PolicyError(f"{path}.detect: no detector produces {detect!r}{suffix}")

    try:
        action = Action.parse(mapping["action"])
    except ValueError as exc:
        raise PolicyError(f"{path}.action: {exc}") from None

    try:
        min_severity = Severity.parse(mapping.get("min_severity", "info"))
    except ValueError as exc:
        raise PolicyError(f"{path}.min_severity: {exc}") from None

    min_confidence = mapping.get("min_confidence", 0.0)
    if not isinstance(min_confidence, int | float) or not 0.0 <= float(min_confidence) <= 1.0:
        raise PolicyError(f"{path}.min_confidence: expected a number in [0, 1]")

    redaction = mapping.get("redaction")
    if redaction is not None:
        redaction = str(redaction)
        if redaction not in STRATEGIES:
            raise PolicyError(
                f"{path}.redaction: unknown strategy {redaction!r}; "
                f"expected one of {', '.join(STRATEGIES)}"
            )
        if action is not Action.REDACT:
            raise PolicyError(
                f"{path}.redaction: only meaningful with action 'redact', "
                f"but this rule's action is {action.label!r}"
            )

    return Rule(
        id=rule_id,
        detect=detect,
        action=action,
        min_severity=min_severity,
        min_confidence=float(min_confidence),
        redaction=redaction,
        reason=str(mapping.get("reason", "")),
    )


def _parse_stage(raw: Any, stage: str) -> tuple[Rule, ...]:
    if raw is None:
        return ()
    mapping = _require_mapping(raw, stage)
    _reject_unknown(mapping, _STAGE_KEYS, stage)
    rules_raw = mapping.get("rules")
    if rules_raw is None:
        rules_raw = []
    if not isinstance(rules_raw, list):
        raise PolicyError(f"{stage}.rules: expected a list, got {type(rules_raw).__name__}")
    seen: set[str] = set()
    return tuple(
        _parse_rule(entry, f"{stage}.rules[{position}]", seen)
        for position, entry in enumerate(rules_raw)
    )


def _parse_budget(raw: Any) -> Budget:
    if raw is None:
        return Budget()
    mapping = _require_mapping(raw, "budget")
    _reject_unknown(mapping, _BUDGET_KEYS, "budget")
    total = mapping.get("total_ms")
    if total is None:
        return Budget()
    if not isinstance(total, int | float) or total <= 0:
        raise PolicyError("budget.total_ms: expected a positive number of milliseconds")
    if "on_exceeded" not in mapping:
        raise PolicyError(
            "budget.on_exceeded: required whenever a budget is set; "
            f"expected one of {', '.join(ON_EXCEEDED)}. There is no safe default -- "
            "failing open and failing closed are opposite bets."
        )
    on_exceeded = str(mapping["on_exceeded"])
    if on_exceeded not in ON_EXCEEDED:
        raise PolicyError(
            f"budget.on_exceeded: unknown value {on_exceeded!r}; "
            f"expected one of {', '.join(ON_EXCEEDED)}"
        )
    return Budget(total_ms=float(total), on_exceeded=on_exceeded)


def loads(text: str, *, name: str | None = None) -> Policy:
    """Parse and validate a policy document."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyError(f"policy is not valid YAML: {exc}") from None

    if raw is None:
        raise PolicyError("policy is empty")
    mapping = _require_mapping(raw, "policy")
    _reject_unknown(mapping, _ROOT_KEYS, "policy")

    version = mapping.get("version")
    if version != SCHEMA_VERSION:
        raise PolicyError(
            f"policy.version: expected {SCHEMA_VERSION}, got {version!r}. "
            "The version is required so an older loader refuses a newer file "
            "instead of ignoring the parts it does not understand."
        )

    defaults = mapping.get("defaults") or {}
    defaults = _require_mapping(defaults, "defaults")
    _reject_unknown(defaults, _DEFAULTS_KEYS, "defaults")
    default_redaction = str(defaults.get("redaction", "label"))
    if default_redaction not in STRATEGIES:
        raise PolicyError(
            f"defaults.redaction: unknown strategy {default_redaction!r}; "
            f"expected one of {', '.join(STRATEGIES)}"
        )

    policy_name = str(mapping.get("name") or name or "unnamed")
    return Policy(
        name=policy_name,
        version=SCHEMA_VERSION,
        description=str(mapping.get("description", "")),
        input_rules=_parse_stage(mapping.get("input"), "input"),
        output_rules=_parse_stage(mapping.get("output"), "output"),
        budget=_parse_budget(mapping.get("budget")),
        default_redaction=default_redaction,
        hash_salt=str(defaults.get("hash_salt", "")),
    )


def load(path: str | Path) -> Policy:
    """Load a policy from disk."""
    resolved = Path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyError(f"cannot read policy {resolved}: {exc}") from None
    return loads(text, name=resolved.stem)


def default() -> Policy:
    """The bundled policy: redact personal data, block confident injection."""
    return load(DEFAULT_POLICY_PATH)
