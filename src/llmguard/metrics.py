"""A minimal Prometheus-compatible metrics registry.

This is deliberately not ``prometheus_client``. A guardrail sits in front of a
model in someone else's service, and that service usually already has a metrics
client with its own registry, its own multiprocess mode and its own opinion about
collector registration. Pulling in a second one is how you get duplicate
timeseries and a hard-to-diagnose import order bug.

So this module owns roughly a hundred lines: counters, gauges, histograms with
labels, and a renderer that emits the text exposition format. If you already have
a metrics stack, ignore all of it and read ``GuardResult`` directly -- every
number rendered here comes off that object.

Output is sorted, so a rendered scrape is byte-for-byte reproducible and can be
asserted on in a test rather than eyeballed.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar, cast

LabelValues = tuple[str, ...]

#: Bucket boundaries in seconds. Chosen around the latency budget this library
#: exists to protect: sub-millisecond is the common case, and anything past 50ms
#: is a budget breach worth seeing as its own bucket.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    1.0,
)


def _escape_help(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(names: Sequence[str], values: LabelValues) -> str:
    if not names:
        return ""
    pairs = ",".join(
        f'{name}="{_escape_label(value)}"' for name, value in zip(names, values, strict=True)
    )
    return "{" + pairs + "}"


@dataclass
class _Metric:
    name: str
    help: str
    labelnames: tuple[str, ...] = ()

    def _key(self, labels: Sequence[str]) -> LabelValues:
        if len(labels) != len(self.labelnames):
            raise ValueError(f"{self.name} expects labels {self.labelnames}, got {tuple(labels)}")
        return tuple(str(value) for value in labels)


@dataclass
class Counter(_Metric):
    values: dict[LabelValues, float] = field(default_factory=dict)

    def inc(self, *labels: str, amount: float = 1.0) -> None:
        key = self._key(labels)
        self.values[key] = self.values.get(key, 0.0) + amount

    def value(self, *labels: str) -> float:
        return self.values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} counter"]
        if not self.values and not self.labelnames:
            lines.append(f"{self.name} 0")
        for key in sorted(self.values):
            lines.append(f"{self.name}{_render_labels(self.labelnames, key)} {self.values[key]:g}")
        return lines


@dataclass
class Gauge(_Metric):
    values: dict[LabelValues, float] = field(default_factory=dict)

    def set(self, value: float, *labels: str) -> None:
        self.values[self._key(labels)] = value

    def value(self, *labels: str) -> float:
        return self.values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} gauge"]
        for key in sorted(self.values):
            lines.append(f"{self.name}{_render_labels(self.labelnames, key)} {self.values[key]:g}")
        return lines


@dataclass
class Histogram(_Metric):
    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    counts: dict[LabelValues, list[int]] = field(default_factory=dict)
    sums: dict[LabelValues, float] = field(default_factory=dict)
    totals: dict[LabelValues, int] = field(default_factory=dict)

    def observe(self, value: float, *labels: str) -> None:
        key = self._key(labels)
        counts = self.counts.setdefault(key, [0] * len(self.buckets))
        for position, bound in enumerate(self.buckets):
            if value <= bound:
                counts[position] += 1
        self.sums[key] = self.sums.get(key, 0.0) + value
        self.totals[key] = self.totals.get(key, 0) + 1

    def count(self, *labels: str) -> int:
        return self.totals.get(self._key(labels), 0)

    def quantile(self, q: float, *labels: str) -> float:
        """Bucket-interpolated quantile, good enough for a demo and a smoke test.

        Real percentiles come from the histogram in your metrics backend; this is
        here so the CLI can print a p95 without a second dependency.
        """
        key = self._key(labels)
        total = self.totals.get(key, 0)
        if total == 0:
            return math.nan
        target = q * total
        counts = self.counts[key]
        for position, bound in enumerate(self.buckets):
            if counts[position] >= target:
                return bound
        return math.inf

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} histogram"]
        for key in sorted(self.counts):
            counts = self.counts[key]
            for position, bound in enumerate(self.buckets):
                labels = (*key, f"{bound:g}")
                rendered = _render_labels((*self.labelnames, "le"), labels)
                lines.append(f"{self.name}_bucket{rendered} {counts[position]}")
            inf_labels = _render_labels((*self.labelnames, "le"), (*key, "+Inf"))
            lines.append(f"{self.name}_bucket{inf_labels} {self.totals[key]}")
            base = _render_labels(self.labelnames, key)
            lines.append(f"{self.name}_sum{base} {self.sums[key]:g}")
            lines.append(f"{self.name}_count{base} {self.totals[key]}")
        return lines


#: The metric classes a registry holds. Bound rather than constrained, so a
#: subclass of one of them stays that subclass through :meth:`Registry._get_or_create`.
_M = TypeVar("_M", bound="Counter | Gauge | Histogram")


class Registry:
    """Holds the metric objects and renders a scrape."""

    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help: str, labelnames: Sequence[str] = ()) -> Counter:
        return self._get_or_create(Counter(name, help, tuple(labelnames)))

    def gauge(self, name: str, help: str, labelnames: Sequence[str] = ()) -> Gauge:
        return self._get_or_create(Gauge(name, help, tuple(labelnames)))

    def histogram(
        self,
        name: str,
        help: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> Histogram:
        return self._get_or_create(Histogram(name, help, tuple(labelnames), buckets=tuple(buckets)))

    def _get_or_create(self, metric: _M) -> _M:
        """Register ``metric``, or return the equivalent one already registered.

        Generic in the metric type rather than returning the union: the three
        public accessors each promise a concrete class, and a union return type
        would have them all lying by one widening. The ``type(existing)`` guard
        below is what makes the cast at the end sound -- an ``existing`` that
        survives it is the same class as ``metric``, which is ``_M``.
        """
        with self._lock:
            existing = self._metrics.get(metric.name)
            if existing is not None:
                if type(existing) is not type(metric) or existing.labelnames != metric.labelnames:
                    raise ValueError(
                        f"metric {metric.name} already registered with a different shape"
                    )
                return cast("_M", existing)
            self._metrics[metric.name] = metric
            return metric

    def render(self) -> str:
        """The text exposition format, sorted so the output is reproducible."""
        lines: list[str] = []
        for name in sorted(self._metrics):
            lines.extend(self._metrics[name].render())
        return "\n".join(lines) + "\n"


class GuardMetrics:
    """The metric set this library records, in one place.

    Naming follows the Prometheus conventions that matter in practice: a
    ``_total`` suffix on counters, base units (seconds, not milliseconds) on the
    histogram, and label values drawn from small closed sets so cardinality
    cannot run away. In particular there is no label carrying a policy rule's
    free-text reason or any part of the text being checked.
    """

    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry or Registry()
        r = self.registry
        self.checks = r.counter(
            "llmguard_checks_total", "Guard checks performed.", ("stage", "verdict")
        )
        self.findings = r.counter(
            "llmguard_findings_total", "Findings raised.", ("detector", "kind", "severity")
        )
        self.duration = r.histogram(
            "llmguard_check_duration_seconds", "Wall time per guard check.", ("stage",)
        )
        self.detector_duration = r.histogram(
            "llmguard_detector_duration_seconds", "Wall time per detector.", ("detector",)
        )
        self.budget_exceeded = r.counter(
            "llmguard_budget_exceeded_total",
            "Checks that ran out of latency budget before every detector had run.",
            ("stage",),
        )
        self.stream_chunks = r.counter(
            "llmguard_stream_chunks_total", "Chunks fed to the streaming guard."
        )
        self.stream_blocks = r.counter(
            "llmguard_stream_blocks_total", "Streams terminated by a block-level finding."
        )
        self.stream_leaks = r.counter(
            "llmguard_stream_leaks_total",
            "Findings detected only after their text had already been emitted. "
            "Must stay at zero; a non-zero value means a detector understated max_match_len.",
        )
        self.repairs = r.counter(
            "llmguard_schema_repair_attempts_total",
            "Schema repair attempts.",
            ("outcome",),
        )
        self.holdback = r.gauge(
            "llmguard_stream_holdback_chars",
            "Characters the streaming guard withholds, derived from the active detectors.",
        )

    def render(self) -> str:
        return self.registry.render()
