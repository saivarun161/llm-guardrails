"""The detector registry.

Policies name detectors by string, so the registry is what turns a typo in a YAML
file into a load-time error with a suggestion instead of a rule that silently
never fires.

That only works if the registry knows about every detector in play. A detector
written outside this package -- a house pattern for internal ticket ids, an
allowlist for a domain the PII rules over-match -- used to be invisible to it:
you could pass it to ``Guard(detectors=...)`` and it would run, but no policy
could name it, because the strict loader rejected any ``detect:`` label the
builtin set did not produce. The guardrail's own strictness locked out the
extension point.

:func:`register` fixes that, and validates the detector's shape while it is
there. The checks are deliberately at registration rather than at first use:
``name``, ``kinds`` and ``max_match_len`` are what policy validation and the
streaming holdback are computed from, so a detector that gets them wrong should
fail where the traceback names the detector rather than three layers down inside
a stream. Whether the detector *keeps* its ``max_match_len`` promise is a
property of its behaviour on text and cannot be checked here at all --
:mod:`llmguard.testing` is the harness for that, and any detector worth
registering should be run through it in its own test suite.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable

from ..types import DetectorContractError, valid_label
from .base import Detector
from .injection import InjectionDetector
from .pii import PiiDetector

#: A zero-argument callable returning a fresh detector. Constructors rather than
#: instances: a caller narrowing ``PiiDetector`` to a few kinds gets its own
#: object and its own (smaller) holdback window, and two guards never share
#: mutable state through the registry.
Factory = Callable[[], Detector]

#: The detectors that ship with the library. Frozen: :func:`reset` restores the
#: live registry to exactly this, which is what lets a test register something
#: broken without leaking it into every test that runs afterwards.
BUILTIN: dict[str, Factory] = {
    "pii": PiiDetector,
    "injection": InjectionDetector,
}

_REGISTRY: dict[str, Factory] = dict(BUILTIN)


def register(factory: Factory, *, name: str | None = None, replace: bool = False) -> str:
    """Make a detector addressable by policies, and return the name it took.

    ``factory`` is called once here to inspect the detector it produces, and
    again for every guard that needs one.

    Registering under a name the detector does not answer to is rejected rather
    than accommodated: findings carry ``detector=self.name``, so a mismatch
    produces a rule that loads cleanly and then never matches anything -- the
    exact failure the strict policy loader exists to prevent.
    """
    try:
        probe = factory()
    except Exception as exc:
        raise DetectorContractError(
            f"detector factory {factory!r} must be callable with no arguments: {exc}"
        ) from exc

    validate_shape(probe)
    resolved = name or probe.name
    if resolved != probe.name:
        raise DetectorContractError(
            f"cannot register {probe.name!r} under {resolved!r}: findings carry the "
            "detector's own name, so a rule naming the registry key would never match"
        )
    if resolved in _REGISTRY and not replace:
        raise DetectorContractError(
            f"{resolved!r} is already registered; pass replace=True to shadow it. "
            "Silently replacing a detector would change what every existing policy "
            "means without changing a line of it."
        )

    _REGISTRY[resolved] = factory
    return resolved


def unregister(name: str) -> None:
    """Remove a detector from the registry. Unknown names raise ``KeyError``."""
    del _REGISTRY[name]


def reset() -> None:
    """Restore the registry to the detectors that ship with the library."""
    _REGISTRY.clear()
    _REGISTRY.update(BUILTIN)


def registered() -> dict[str, Factory]:
    """Every currently addressable detector, as a copy."""
    return dict(_REGISTRY)


def build(names: list[str] | tuple[str, ...] | None = None) -> dict[str, Detector]:
    """Instantiate detectors by name."""
    selected = list(_REGISTRY) if names is None else list(names)
    out: dict[str, Detector] = {}
    for name in selected:
        if name not in _REGISTRY:
            raise KeyError(name)
        out[name] = _REGISTRY[name]()
    return out


def known_labels() -> list[str]:
    """Every ``detector`` and ``detector.kind`` string a policy may reference."""
    labels: list[str] = ["*"]
    for name, factory in _REGISTRY.items():
        detector = factory()
        labels.append(name)
        labels.extend(f"{name}.{kind}" for kind in detector.kinds)
    return sorted(labels)


def suggest(label: str) -> str:
    """The 'did you mean' half of a policy error message."""
    matches = difflib.get_close_matches(label, known_labels(), n=1, cutoff=0.6)
    return matches[0] if matches else ""


def validate_shape(detector: object) -> None:
    """Check the attributes a policy and the holdback are computed from.

    Shape only. Whether the detector *keeps* the ``max_match_len`` it declares is
    a property of its behaviour on text; :mod:`llmguard.testing` checks that.
    """
    name = getattr(detector, "name", None)
    if not isinstance(name, str) or not name or not valid_label(name) or "." in name:
        raise DetectorContractError(
            f"detector name {name!r} must be a lowercase identifier without a dot; "
            "the dot separates it from the kind in a policy label"
        )

    kinds = getattr(detector, "kinds", None)
    if not isinstance(kinds, tuple) or not kinds:
        raise DetectorContractError(f"{name}.kinds must be a non-empty tuple, got {kinds!r}")
    for kind in kinds:
        if not isinstance(kind, str) or not kind or not valid_label(kind) or "." in kind:
            raise DetectorContractError(
                f"{name}.kinds contains {kind!r}, which is not a valid label component"
            )
    if len(set(kinds)) != len(kinds):
        raise DetectorContractError(f"{name}.kinds contains duplicates: {kinds!r}")

    bound = getattr(detector, "max_match_len", None)
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 1:
        raise DetectorContractError(
            f"{name}.max_match_len must be a positive integer, got {bound!r}. "
            "The streaming holdback window is derived from it."
        )

    if not callable(getattr(detector, "scan", None)):
        raise DetectorContractError(f"{name} has no callable scan(text) method")


__all__ = [
    "BUILTIN",
    "Detector",
    "Factory",
    "InjectionDetector",
    "PiiDetector",
    "build",
    "known_labels",
    "register",
    "registered",
    "reset",
    "suggest",
    "unregister",
    "validate_shape",
]
