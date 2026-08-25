"""The latency budget, and the question it forces you to answer.

A guardrail runs on every request, so it competes with the model for the user's
patience. Give it an unbounded time slice and one pathological input -- a 400KB
pasted log, a regex that backtracks -- turns a safety feature into an outage.

So checks run against a deadline, and detectors are attempted in priority order
until it expires. What happens then is not a detail the library gets to decide,
because the two answers are opposites:

``fail_open``
    Let the text through with a finding recording that the check was incomplete.
    Right for a consumer chat feature where a missed redaction is bad and a total
    outage is worse.

``fail_closed``
    Block. Right for anything where unchecked text reaching the model is the
    incident you are trying to avoid.

The policy has to say which. There is no sensible default, so the loader requires
it whenever a budget is configured.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

FAIL_OPEN = "fail_open"
FAIL_CLOSED = "fail_closed"
ON_EXCEEDED = (FAIL_OPEN, FAIL_CLOSED)


@dataclass(frozen=True, slots=True)
class Budget:
    """How long a check may take, and what to do when it does not fit."""

    total_ms: float | None = None
    on_exceeded: str = FAIL_OPEN

    def __post_init__(self) -> None:
        if self.total_ms is not None and self.total_ms <= 0:
            raise ValueError("budget.total_ms must be positive")
        if self.on_exceeded not in ON_EXCEEDED:
            raise ValueError(
                f"budget.on_exceeded must be one of {', '.join(ON_EXCEEDED)}, "
                f"got {self.on_exceeded!r}"
            )

    @property
    def unlimited(self) -> bool:
        return self.total_ms is None

    def start(self, clock: Callable[[], float] | None = None) -> Deadline:
        ticker = clock or time.perf_counter
        return Deadline(self, ticker(), ticker)


@dataclass(slots=True)
class Deadline:
    """A running clock against one budget.

    The clock is injectable so the budget path is tested with a fake clock rather
    than with ``sleep`` calls, which keeps the suite fast and, more importantly,
    keeps it from going flaky on a loaded CI runner.
    """

    budget: Budget
    started_at: float
    _clock: Callable[[], float] = time.perf_counter

    def elapsed_ms(self) -> float:
        return (self._clock() - self.started_at) * 1000.0

    def remaining_ms(self) -> float:
        if self.budget.total_ms is None:
            return float("inf")
        return self.budget.total_ms - self.elapsed_ms()

    @property
    def expired(self) -> bool:
        return self.remaining_ms() <= 0.0

    def with_clock(self, clock: Callable[[], float]) -> Deadline:
        return Deadline(self.budget, self.started_at, clock)
