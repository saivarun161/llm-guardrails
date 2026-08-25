"""llm-guardrails: input/output safety middleware for LLM applications.

    >>> from llmguard import Guard
    >>> guard = Guard()
    >>> guard.check_input("mail me at ada@example.com").text
    'mail me at [EMAIL]'

Four pieces, usable together or separately:

``Guard``
    Runs detectors under a latency budget and applies a YAML policy.
``StreamGuard``
    The same policy over a token stream, with a holdback window that makes a
    boundary leak impossible rather than unlikely.
``schema``
    Schema-constrained output with a deterministic repair tier before any
    regeneration round trip.
``GuardMetrics``
    Prometheus timeseries for all of it, with no second metrics dependency.
"""

from __future__ import annotations

from .budget import Budget
from .engine import Guard
from .metrics import GuardMetrics, Registry
from .policy import Policy, Rule, load, loads
from .streaming import StreamGuard, filter_stream
from .types import (
    Action,
    Finding,
    GuardResult,
    PolicyError,
    RuleDecision,
    Severity,
    StreamBlocked,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Budget",
    "Finding",
    "Guard",
    "GuardMetrics",
    "GuardResult",
    "Policy",
    "PolicyError",
    "Registry",
    "Rule",
    "RuleDecision",
    "Severity",
    "StreamBlocked",
    "StreamGuard",
    "__version__",
    "filter_stream",
    "load",
    "loads",
]
