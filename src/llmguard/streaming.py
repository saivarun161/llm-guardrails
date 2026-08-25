"""Streaming output filtering that cannot leak across a chunk boundary.

This is the part of a guardrail that is usually wrong.

Token streaming hands you the response a few characters at a time. The obvious
implementation redacts each chunk as it arrives, and it fails the moment a secret
lands on a boundary: ``sk-live_abc`` in one chunk and ``123def`` in the next are
each individually harmless, so both go out, and the client concatenates them back
into the credential. Buffering the whole response instead fixes correctness and
throws away the reason anyone streams.

The fix is a **holdback window**. Keep the last *H* characters of the buffer
back, where *H* is the longest span any active detector can produce. Then:

* a match starting before ``len(buffer) - H`` must end before ``len(buffer)``,
  so it is entirely inside what has already been received -- nothing is missed;
* a match is only finalised if it *ends* at or before the emit boundary, so its
  trailing lookahead is evaluated against real text rather than against the end
  of a truncated buffer -- nothing spurious is emitted;
* a match that straddles the boundary pulls the boundary back to its start, so
  no part of it is ever emitted;
* the previously emitted tail is kept as scan context so lookbehind assertions
  still see the characters in front of the buffer.

Together those give an exact property, and it is the one the test suite asserts
by fuzzing every chunking of every fixture: **for any chunk sizes, the
concatenated stream output equals what the non-streaming guard produces on the
whole text.**

*H* is derived from the active detectors rather than configured, because a
configured value is a value someone will lower. :attr:`StreamGuard.holdback`
exposes it, and narrowing the policy or the detector kinds is the supported way
to shrink it.

Blocking works on a wider set than redaction: a block-level finding stops the
stream even while it is still inside the holdback window, which is precisely the
"do not leak a blocked span across chunk boundaries" requirement. Injection
signals additionally accumulate across the whole stream, so a payload spread over
more text than one window can hold still escalates.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from . import redact as redact_module
from .budget import FAIL_CLOSED
from .detectors.injection import score, severity_for
from .engine import Guard, block_rules
from .types import Action, Finding, GuardResult, RuleDecision, Severity, StreamBlocked


class StreamGuard:
    """Incremental filtering over a token stream.

    Feed it chunks with :meth:`feed` and finish with :meth:`close`. Both return
    text that is safe to forward. :meth:`close` must be called: the final
    holdback window is only released there.
    """

    def __init__(self, guard: Guard, stage: str = "output", *, raise_on_block: bool = True):
        self.guard = guard
        self.stage = stage
        self.raise_on_block = raise_on_block
        self.holdback = guard.holdback_for(stage)
        self._block_rules = block_rules(guard.policy, stage)

        self._pending = ""
        self._context = ""
        self._emitted = 0
        self._consumed = 0
        # Absolute offset, in the original text, of the first character of
        # ``_pending``. Findings are reported against the original stream, not
        # against whatever window happened to be in the buffer.
        self._origin = 0
        self._closed = False
        self._blocked = False
        self._findings: list[Finding] = []
        self._decisions: list[RuleDecision] = []
        self._injection_signals: set[str] = set()

        guard.metrics.holdback.set(float(self.holdback))

    # -- public API ------------------------------------------------------

    @property
    def blocked(self) -> bool:
        return self._blocked

    @property
    def emitted(self) -> int:
        """Characters forwarded to the client so far, after redaction."""
        return self._emitted

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(self._findings)

    def feed(self, chunk: str) -> str:
        """Absorb a chunk and return whatever is now safe to forward."""
        if self._closed:
            raise RuntimeError("StreamGuard is closed")
        self.guard.metrics.stream_chunks.inc()
        self._pending += chunk
        self._consumed += len(chunk)
        return self._drain(final=False)

    def close(self) -> str:
        """Flush the holdback window and return the tail."""
        if self._closed:
            return ""
        out = self._drain(final=True)
        self._closed = True
        return out

    def run(self, chunks: Iterable[str]) -> Iterator[str]:
        """Filter an iterable of chunks, yielding only non-empty output."""
        for chunk in chunks:
            out = self.feed(chunk)
            if out:
                yield out
        tail = self.close()
        if tail:
            yield tail

    def result(self) -> GuardResult:
        """An audit record for the stream as a whole."""
        if self._blocked:
            verdict = Action.BLOCK
        elif self._decisions:
            verdict = max(d.action for d in self._decisions)
        else:
            verdict = Action.ALLOW
        return GuardResult(
            stage=self.stage,
            policy=self.guard.policy.name,
            verdict=verdict,
            text="",
            original_length=self._consumed,
            findings=tuple(self._findings),
            decisions=tuple(self._decisions),
        )

    # -- internals -------------------------------------------------------

    def _drain(self, *, final: bool) -> str:
        if self._blocked:
            return ""

        scan_text = self._context + self._pending
        base = len(self._context)
        findings, skipped = self._scan(scan_text)

        if skipped and not self._handle_overrun(skipped, final=final):
            # Deferred: nothing is emitted this round. The text stays pending and
            # is rescanned on the next chunk against a fresh budget.
            return ""

        block_hit = self._check_blocks(findings, base)
        if block_hit:
            self._blocked = True
            self.guard.metrics.stream_blocks.inc()
            if self.raise_on_block:
                raise StreamBlocked(
                    tuple(self._findings),
                    self._emitted,
                    tuple(d.reason for d in self._decisions if d.action is Action.BLOCK),
                )
            return ""

        limit = len(scan_text) if final else base + max(0, len(self._pending) - self.holdback)

        # A finding that straddles the emit boundary drags it back to the
        # finding's start, so no prefix of a match is ever forwarded.
        for finding in findings:
            if finding.start < limit < finding.end:
                limit = finding.start

        emitted_chunk = scan_text[base:limit]
        final_findings = [f for f in findings if base <= f.start and f.end <= limit]

        # Provably empty while every detector's max_match_len is honest, and
        # counted rather than asserted so a violation shows up as a metric in
        # production instead of a crash. tests/test_streaming.py asserts zero.
        late = [f for f in findings if f.start < base < f.end]
        if late:
            self.guard.metrics.stream_leaks.inc(amount=len(late))

        out = self._redact(emitted_chunk, base, final_findings)

        self._pending = scan_text[limit:]
        self._context = (self._context + emitted_chunk)[-self.holdback :]
        self._origin += limit - base
        self._emitted += len(out)
        return out

    def _scan(self, scan_text: str) -> tuple[list[Finding], list[str]]:
        findings, skipped, _ = self.guard.scan(self.stage, scan_text)
        return self._escalate_injection(findings), skipped

    def _handle_overrun(self, skipped: list[str], *, final: bool) -> bool:
        """A window that ran out of budget. Returns whether to emit anyway.

        The batch guard can afford to answer this once per check. A stream cannot
        emit text that was never scanned and then take it back, so the default is
        to *defer*: hold the window, emit nothing, and rescan on the next chunk
        with a fresh budget. Transient load costs latency instead of leaking.

        Deferring stops working at ``close()``, where there is no next chunk. At
        that point the policy's own answer applies -- ``fail_closed`` blocks the
        stream, ``fail_open`` releases text that was only partly scanned -- and
        either way a finding records that the check was incomplete.
        """
        self.guard.metrics.budget_exceeded.inc(self.stage)
        if not final:
            return False

        marker = Finding(
            detector="budget",
            kind="exceeded",
            start=0,
            end=0,
            severity=Severity.HIGH,
            confidence=1.0,
            detail=f"stream closed with {', '.join(skipped)} unscanned",
        )
        self._findings.append(marker)
        fail_closed = self.guard.policy.budget.on_exceeded == FAIL_CLOSED
        self._decisions.append(
            RuleDecision(
                rule_id="__budget__",
                action=Action.BLOCK if fail_closed else Action.FLAG,
                finding_index=len(self._findings) - 1,
                reason=(
                    "latency budget exhausted before the final window was fully scanned "
                    f"({self.guard.policy.budget.on_exceeded})"
                ),
            )
        )
        if fail_closed:
            self._blocked = True
            self.guard.metrics.stream_blocks.inc()
            if self.raise_on_block:
                raise StreamBlocked(tuple(self._findings), self._emitted, (marker.detail,))
            return False
        return True

    def _escalate_injection(self, findings: list[Finding]) -> list[Finding]:
        """Re-score injection findings against every signal seen in the stream.

        A window only sees ``holdback + pending`` characters. An attacker who
        spreads three weak signals across a long response would otherwise be
        scored three times as one weak signal. Accumulating the signal *names*
        for the whole stream fixes that, and makes blocking monotone: once a
        stream has enough signal to block, more text cannot take it back below
        the threshold.
        """
        out: list[Finding] = []
        for finding in findings:
            if finding.detector != "injection":
                out.append(finding)
                continue
            self._injection_signals.update(finding.signals)
            names = tuple(sorted(self._injection_signals))
            total = score(names)
            out.append(
                Finding(
                    detector=finding.detector,
                    kind=finding.kind,
                    start=finding.start,
                    end=finding.end,
                    severity=max(finding.severity, severity_for(total)),
                    confidence=min(0.99, round(total / 5.0, 3)),
                    detail=f"score {total} from {len(names)} signal(s) seen so far in the stream",
                    signals=names,
                )
            )
        return out

    def _check_blocks(self, findings: list[Finding], base: int) -> bool:
        """Does anything in the window -- held back or not -- demand a block?"""
        hit = False
        for finding in findings:
            if finding.end <= base:
                continue
            for rule in self._block_rules:
                if rule.matches(finding):
                    self._findings.append(finding.shifted(self._origin - base))
                    self._decisions.append(
                        RuleDecision(
                            rule_id=rule.id,
                            action=Action.BLOCK,
                            finding_index=len(self._findings) - 1,
                            reason=rule.reason
                            or f"{finding.label} at offset {self._origin + finding.start - base}",
                        )
                    )
                    hit = True
                    break
        return hit

    def _redact(self, emitted_chunk: str, base: int, findings: list[Finding]) -> str:
        pairs: list[tuple[Finding, str]] = []
        for finding in findings:
            rule = self.guard.policy.resolve(self.stage, finding)
            if rule is None:
                continue
            local = finding.shifted(-base)
            self._findings.append(local.shifted(self._origin))
            self._decisions.append(
                RuleDecision(
                    rule_id=rule.id,
                    action=rule.action,
                    finding_index=len(self._findings) - 1,
                    reason=rule.reason,
                )
            )
            if rule.action is Action.REDACT:
                pairs.append((local, self.guard.policy.redaction_for(rule)))

        if not pairs:
            return emitted_chunk
        return redact_module.apply(emitted_chunk, pairs, salt=self.guard.policy.hash_salt)


def filter_stream(
    guard: Guard,
    chunks: Iterable[str],
    stage: str = "output",
    *,
    raise_on_block: bool = True,
) -> Iterator[str]:
    """Convenience wrapper: filter ``chunks`` through a fresh :class:`StreamGuard`."""
    yield from StreamGuard(guard, stage, raise_on_block=raise_on_block).run(chunks)


def holdback_report(guard: Guard, stage: str = "output") -> list[tuple[str, int]]:
    """Per-detector contribution to the holdback window, for the CLI and README."""
    rows = [(detector.name, detector.max_match_len) for detector in guard.detectors_for(stage)]
    rows.sort(key=lambda row: -row[1])
    return rows
