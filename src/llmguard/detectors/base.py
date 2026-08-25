"""What a detector is, and the contract the streaming guard leans on.

The interesting member is ``max_match_len``. A detector must declare the longest
span it can ever return, because the streaming filter derives its holdback window
from that number: it refuses to emit the last ``max_match_len`` characters of the
buffer, which is exactly the condition under which a match cannot straddle the
boundary between what has been emitted and what has not.

That makes the bound load-bearing rather than documentation. A detector that
understates it does not merely make the streaming output slightly different from
the batch output -- it leaks. ``tests/test_streaming.py`` fuzzes every chunking of
every fixture and asserts the two agree, which is how an understated bound gets
caught.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Finding


@runtime_checkable
class Detector(Protocol):
    """A stateless scanner over text."""

    #: Namespace for the findings it produces, e.g. ``pii``.
    name: str

    #: Every ``kind`` it can emit, so a policy referring to ``pii.emial`` fails
    #: at load time instead of silently never matching.
    kinds: tuple[str, ...]

    #: The longest span this detector can ever return, in characters.
    max_match_len: int

    def scan(self, text: str) -> list[Finding]:
        """Return findings with offsets into ``text``. Must not mutate anything."""
        ...
