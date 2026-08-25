"""The guard: run detectors under a deadline, apply the policy, record the result.

The order of operations matters and is worth stating.

1. Only the detectors the stage's rules can use are run, most severe rules first,
   each against the shared deadline.
2. Every finding is resolved to exactly one rule -- longest prefix wins -- so an
   audit record never has to explain which of three overlapping rules "really"
   applied.
3. The verdict is the strongest action any rule returned.
4. Redaction happens once, over all redact-marked findings together, so
   overlapping spans are resolved rather than replaced twice.
5. On a block verdict the text is *not* carried on the result. A result object
   ends up in logs, and the whole job of a block is to stop that payload from
   travelling.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from . import detectors as detector_registry
from . import policy as policy_module
from . import redact
from .budget import FAIL_CLOSED, Deadline
from .detectors.base import Detector
from .metrics import GuardMetrics
from .policy import Policy, Rule
from .types import Action, Finding, GuardResult, RuleDecision, Severity

#: Detectors run in this order so that, when the budget runs out, what got
#: skipped is the cheaper-to-lose check rather than whatever happened to be last
#: in a dict.
_PRIORITY = ("injection", "pii")


class Guard:
    """Checks text against a policy at the input or output stage."""

    def __init__(
        self,
        policy: Policy | None = None,
        *,
        metrics: GuardMetrics | None = None,
        clock: Callable[[], float] | None = None,
        detectors: dict[str, Detector] | None = None,
    ) -> None:
        self.policy = policy or policy_module.default()
        self.metrics = metrics or GuardMetrics()
        self._clock = clock or time.perf_counter
        needed = sorted(
            set(self.policy.detectors_for("input")) | set(self.policy.detectors_for("output"))
        )
        self.detectors = detectors if detectors is not None else detector_registry.build(needed)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: object) -> Guard:
        return cls(policy_module.load(path), **kwargs)  # type: ignore[arg-type]

    def check_input(self, text: str) -> GuardResult:
        return self._check("input", text)

    def check_output(self, text: str) -> GuardResult:
        return self._check("output", text)

    def detectors_for(self, stage: str) -> list[Detector]:
        """Active detectors for a stage, in run order."""
        names = self.policy.detectors_for(stage)
        ordered = [name for name in _PRIORITY if name in names]
        ordered += [name for name in names if name not in _PRIORITY]
        return [self.detectors[name] for name in ordered if name in self.detectors]

    def holdback_for(self, stage: str) -> int:
        """Characters the streaming guard must withhold for this stage.

        Derived, not configured: it is the longest span any active detector can
        return. Understating it would let a match straddle the boundary between
        emitted and pending text, which is the one failure mode a streaming
        filter must not have.
        """
        return max((detector.max_match_len for detector in self.detectors_for(stage)), default=1)

    def scan(
        self, stage: str, text: str, deadline: Deadline | None = None
    ) -> tuple[list[Finding], list[str], dict[str, float]]:
        """Run the stage's detectors, stopping when the budget is gone.

        Returns findings, the names of detectors that never ran, and per-detector
        timings.
        """
        deadline = deadline or self.policy.budget.start(self._clock)
        findings: list[Finding] = []
        skipped: list[str] = []
        timings: dict[str, float] = {}

        for detector in self.detectors_for(stage):
            if deadline.expired:
                skipped.append(detector.name)
                continue
            started = self._clock()
            findings.extend(detector.scan(text))
            elapsed = (self._clock() - started) * 1000.0
            timings[detector.name] = elapsed
            self.metrics.detector_duration.observe(elapsed / 1000.0, detector.name)

        findings.sort(key=lambda f: (f.start, f.detector, f.kind))
        return findings, skipped, timings

    def _check(self, stage: str, text: str) -> GuardResult:
        deadline = self.policy.budget.start(self._clock)
        findings, skipped, timings = self.scan(stage, text, deadline)

        decisions: list[RuleDecision] = []
        to_redact: list[tuple[Finding, str]] = []
        verdict = Action.ALLOW

        for index, finding in enumerate(findings):
            rule = self.policy.resolve(stage, finding)
            if rule is None:
                continue
            decisions.append(
                RuleDecision(
                    rule_id=rule.id,
                    action=rule.action,
                    finding_index=index,
                    reason=rule.reason,
                )
            )
            verdict = max(verdict, rule.action)
            if rule.action is Action.REDACT:
                to_redact.append((finding, self.policy.redaction_for(rule)))

        budget_exceeded = bool(skipped)
        if budget_exceeded:
            findings, decisions, verdict = self._apply_budget_overrun(
                stage, findings, decisions, verdict, skipped, deadline
            )

        if verdict is Action.BLOCK:
            # Never carry a blocked payload on an object destined for a log line.
            output_text = ""
        elif to_redact:
            output_text = redact.apply(text, to_redact, salt=self.policy.hash_salt)
            if output_text == text:
                # The replacement happened to be identical to what it replaced,
                # so do not report a redact verdict for text nobody changed.
                verdict = Action.FLAG
        else:
            output_text = text

        elapsed_ms = deadline.elapsed_ms()
        self._record(stage, verdict, findings, elapsed_ms, budget_exceeded)

        return GuardResult(
            stage=stage,
            policy=self.policy.name,
            verdict=verdict,
            text=output_text,
            original_length=len(text),
            findings=tuple(findings),
            decisions=tuple(decisions),
            elapsed_ms=elapsed_ms,
            budget_exceeded=budget_exceeded,
            skipped_detectors=tuple(skipped),
            timings_ms={name: round(value, 4) for name, value in timings.items()},
        )

    def _apply_budget_overrun(
        self,
        stage: str,
        findings: list[Finding],
        decisions: list[RuleDecision],
        verdict: Action,
        skipped: list[str],
        deadline: Deadline,
    ) -> tuple[list[Finding], list[RuleDecision], Action]:
        """Record the incomplete check as a finding, and honour fail_closed."""
        marker = Finding(
            detector="budget",
            kind="exceeded",
            start=0,
            end=0,
            severity=Severity.HIGH,
            confidence=1.0,
            detail=(
                f"{self.policy.budget.total_ms:g}ms budget exhausted after "
                f"{deadline.elapsed_ms():.2f}ms; skipped {', '.join(skipped)}"
            ),
        )
        findings = [*findings, marker]
        fail_closed = self.policy.budget.on_exceeded == FAIL_CLOSED
        action = Action.BLOCK if fail_closed else Action.FLAG
        decisions = [
            *decisions,
            RuleDecision(
                rule_id="__budget__",
                action=action,
                finding_index=len(findings) - 1,
                reason=(
                    "latency budget exceeded before every detector ran "
                    f"({self.policy.budget.on_exceeded})"
                ),
            ),
        ]
        return findings, decisions, max(verdict, action)

    def _record(
        self,
        stage: str,
        verdict: Action,
        findings: list[Finding],
        elapsed_ms: float,
        budget_exceeded: bool,
    ) -> None:
        self.metrics.checks.inc(stage, verdict.label)
        self.metrics.duration.observe(elapsed_ms / 1000.0, stage)
        if budget_exceeded:
            self.metrics.budget_exceeded.inc(stage)
        for finding in findings:
            self.metrics.findings.inc(finding.detector, finding.kind, finding.severity.label)


def block_rules(policy: Policy, stage: str) -> tuple[Rule, ...]:
    """Rules in a stage whose action is ``block``.

    The streaming guard needs these separately: it has to decide whether to stop
    a stream using findings that are still inside the holdback window, before the
    text they cover is eligible to be emitted.
    """
    return tuple(rule for rule in policy.rules_for(stage) if rule.action is Action.BLOCK)
