"""The detector registry.

Policies name detectors by string, so the registry is what turns a typo in a YAML
file into a load-time error with a suggestion instead of a rule that silently
never fires.
"""

from __future__ import annotations

import difflib

from .base import Detector
from .injection import InjectionDetector
from .pii import PiiDetector

#: Constructors rather than instances: a caller narrowing ``PiiDetector`` to a
#: few kinds gets its own object and its own (smaller) holdback window.
BUILTIN: dict[str, type] = {
    "pii": PiiDetector,
    "injection": InjectionDetector,
}


def build(names: list[str] | tuple[str, ...] | None = None) -> dict[str, Detector]:
    """Instantiate detectors by name."""
    selected = list(BUILTIN) if names is None else list(names)
    out: dict[str, Detector] = {}
    for name in selected:
        if name not in BUILTIN:
            raise KeyError(name)
        out[name] = BUILTIN[name]()
    return out


def known_labels() -> list[str]:
    """Every ``detector`` and ``detector.kind`` string a policy may reference."""
    labels: list[str] = ["*"]
    for name, factory in BUILTIN.items():
        detector = factory()
        labels.append(name)
        labels.extend(f"{name}.{kind}" for kind in detector.kinds)
    return sorted(labels)


def suggest(label: str) -> str:
    """The 'did you mean' half of a policy error message."""
    matches = difflib.get_close_matches(label, known_labels(), n=1, cutoff=0.6)
    return matches[0] if matches else ""


__all__ = [
    "BUILTIN",
    "Detector",
    "InjectionDetector",
    "PiiDetector",
    "build",
    "known_labels",
    "suggest",
]
