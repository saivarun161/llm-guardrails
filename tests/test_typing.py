"""The package's type information, checked rather than asserted.

Two separate claims live here.

The first is that the annotations ship at all. A library without a ``py.typed``
marker is invisible to a downstream type checker no matter how carefully it is
annotated -- mypy reports it as missing stubs and moves on, so every guard
result, finding and verdict arrives in the caller's code as ``Any``. For a
library whose whole pitch is a property you can rely on, silently switching off
type checking at the boundary is the wrong default.

The second is that :class:`~llmguard.detectors.base.Detector` is a protocol a
person can actually satisfy. It is the library's extension point and it is
documented in the README, so "does the shape in the README type-check" is a real
question with a checkable answer -- and it was ``no`` until the bundled
detectors' own ``kinds`` were annotated. These tests run mypy over the documented
shape so that answer stays ``yes``.

Every snippet below is written at its final indentation and joined verbatim, so
what mypy sees is what the test reads like.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from llmguard.detectors import BUILTIN, build
from llmguard.detectors.base import Detector

pytest.importorskip("mypy", reason="type-level checks need the dev extra")


PREAMBLE = """\
import re

from llmguard.detectors.base import Detector
from llmguard.types import Finding, Severity

TICKET = re.compile(r"\\bINC-\\d{6}\\b")


"""

#: A class body, so it starts one level in.
SCAN = """\

    def scan(self, text: str) -> list[Finding]:
        return [
            Finding("ticket", "incident", m.start(), m.end(), Severity.MEDIUM, 0.99)
            for m in TICKET.finditer(text)
        ]
"""

ASSIGN = """\


detector: Detector = TicketDetector()
"""


def _mypy(tmp_path: Path, *parts: str) -> str:
    """Run mypy over one snippet and return its combined output."""
    module = tmp_path / "snippet.py"
    module.write_text("".join(parts), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / ".mypy_cache"),
            str(module),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    return completed.stdout + completed.stderr


def test_py_typed_marker_sits_inside_the_package() -> None:
    """Without this file the annotations are dead weight to a consumer.

    This checks the marker is in the package directory, which under the usual
    editable install is the checkout. Whether it survives into a built wheel is
    a packaging question the same file cannot answer from in here -- CI opens
    the wheel and looks.
    """
    import llmguard

    marker = Path(llmguard.__file__).parent / "py.typed"
    assert marker.is_file(), (
        "py.typed is missing, so a downstream mypy run treats llmguard as untyped "
        "and every value crossing the boundary becomes Any"
    )


def test_the_documented_detector_shape_satisfies_the_protocol(tmp_path: Path) -> None:
    """The README's extension-point example, checked as written.

    This is the regression test for the reason ``kinds`` carries an annotation:
    the protocol declares ``tuple[str, ...]``, protocol attributes are
    invariant, and so a detector assigning a literal tuple has to say so.
    """
    output = _mypy(
        tmp_path,
        PREAMBLE,
        "class TicketDetector:\n"
        '    name: str = "ticket"\n'
        '    kinds: tuple[str, ...] = ("incident",)\n'
        "    max_match_len: int = 10\n",
        SCAN,
        ASSIGN,
    )
    assert "Success" in output, output


def test_an_unannotated_kinds_tuple_does_not_satisfy_the_protocol(tmp_path: Path) -> None:
    """Why the README tells people to annotate, pinned as a fact.

    ``kinds = ("incident",)`` infers as ``tuple[str]``, which is not
    ``tuple[str, ...]``. The detector works perfectly at runtime, which is what
    makes this worth a test rather than a sentence: the failure is silent until
    someone type-checks, and advice to annotate is only worth giving while it is
    still needed. If a future mypy narrows this gap, this test fails and the
    README paragraph can go.
    """
    output = _mypy(
        tmp_path,
        PREAMBLE,
        "class TicketDetector:\n"
        '    name = "ticket"\n'
        '    kinds = ("incident",)\n'
        "    max_match_len = 10\n",
        SCAN,
        ASSIGN,
    )
    assert "Success" not in output
    assert "kinds" in output, output


def test_a_detector_with_the_wrong_scan_signature_is_rejected(tmp_path: Path) -> None:
    """The protocol earns its keep on the shapes, not just the attribute names."""
    output = _mypy(
        tmp_path,
        PREAMBLE,
        "class TicketDetector:\n"
        '    name: str = "ticket"\n'
        '    kinds: tuple[str, ...] = ("incident",)\n'
        "    max_match_len: int = 10\n"
        "\n"
        "    def scan(self, text: str) -> list[str]:\n"
        "        return []\n",
        ASSIGN,
    )
    assert "Success" not in output
    assert "scan" in output, output


def test_the_bundled_detectors_satisfy_the_protocol_statically(tmp_path: Path) -> None:
    """The library holds itself to the shape it asks other people to implement.

    ``InjectionDetector`` did not, and nothing noticed: it declared
    ``kinds = ("prompt_injection",)``, which infers as ``tuple[str]`` and fails
    the protocol on a static check while running perfectly. ``PiiDetector``
    happened to pass only because its ``kinds`` is built by a generator
    expression, which infers as ``tuple[str, ...]`` -- an accident of how it was
    written rather than a decision. Both are annotated now, and this is the
    check that keeps the next one honest.
    """
    output = _mypy(
        tmp_path,
        "from llmguard.detectors.base import Detector\n"
        "from llmguard.detectors.injection import InjectionDetector\n"
        "from llmguard.detectors.pii import PiiDetector\n"
        "\n"
        "injection: Detector = InjectionDetector()\n"
        "pii: Detector = PiiDetector()\n"
        'narrowed: Detector = PiiDetector(kinds=("email",))\n',
    )
    assert "Success" in output, output


@pytest.mark.parametrize("name", sorted(BUILTIN))
def test_builtin_detectors_satisfy_the_protocol_at_runtime(name: str) -> None:
    """The runtime half of the same claim, for every detector that ships."""
    assert isinstance(build([name])[name], Detector)
