from __future__ import annotations

import dataclasses

import pytest

from llmguard import Guard, detectors
from llmguard.budget import Budget
from llmguard.metrics import GuardMetrics
from llmguard.policy import default as default_policy


@pytest.fixture(autouse=True)
def _clean_registry():
    """The detector registry is process-wide, so a test that adds to it must not
    leak into the next one -- a stray custom detector would widen the holdback of
    every ``*`` policy and quietly change what the streaming tests are asserting.
    """
    yield
    detectors.reset()


@pytest.fixture
def policy():
    return default_policy()


@pytest.fixture
def unbudgeted_guard(policy):
    """The default policy with the latency budget removed.

    The streaming invariant is about the holdback window, not about scheduling.
    Leaving a 50ms budget in place would make those tests depend on how loaded
    the machine is: a window that overruns is deliberately deferred, and a window
    that overruns at close() is deliberately allowed through under fail_open. The
    budget path has its own tests, with a clock the test drives by hand.
    """
    return Guard(dataclasses.replace(policy, budget=Budget()), metrics=GuardMetrics())


@pytest.fixture
def guard(policy):
    # A fresh metrics registry per test: counters are cumulative, and a shared
    # one would make assertions depend on test ordering.
    return Guard(policy, metrics=GuardMetrics())


class FakeClock:
    """A monotonic clock the test drives by hand.

    The budget tests need elapsed time to be exact, and ``sleep`` on a loaded CI
    runner is neither exact nor fast.
    """

    def __init__(self, step: float = 0.0):
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock():
    return FakeClock
