"""Schema-constrained output, and a repair loop that tries the cheap fixes first.

A model asked for JSON returns JSON *nearly* always. The failures are boring and
repetitive: a markdown fence around it, a sentence of preamble, a trailing comma,
``True`` instead of ``true``, a number sent as a string, an extra field nobody
asked for. Sending all of those back to the model is the expensive answer -- a
whole extra round trip, more latency, more spend, and a fresh chance to return
something different.

So repair runs in two tiers, and the ordering is the point:

**Tier one, deterministic and free.** Unwrap fences, isolate the JSON value,
normalise the syntax a model gets wrong, then coerce values that the schema
uniquely determines -- ``"42"`` where an integer is required, ``"true"`` where a
boolean is, a bare object where a one-element array is. Every transform is named
and recorded, so a run reports *what* it had to fix rather than just that it did.

**Tier two, a regeneration prompt.** Only if tier one still leaves errors. The
prompt names the exact validation failures by JSON pointer, because "your output
was invalid" produces another invalid output.

The validator is a deliberate subset of JSON Schema -- types, ``required``,
``enum``, ``properties``, ``items``, bounds, ``pattern``, ``additionalProperties``
-- covering what structured-output prompts actually use. It is not a conformant
implementation and does not pretend to be: no ``$ref``, no ``allOf``, no remote
schema resolution. Reach for ``jsonschema`` when you need those; this exists so
that constraining a model's output does not require a dependency.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

JsonValue = Any
Schema = dict[str, Any]

_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


@dataclass(frozen=True, slots=True)
class SchemaError:
    """One validation failure, located by JSON pointer."""

    pointer: str
    message: str

    def __str__(self) -> str:
        return f"{self.pointer or '/'}: {self.message}"


def _type_name(value: JsonValue) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def _matches_type(value: JsonValue, expected: str) -> bool:
    if expected not in _TYPES:
        return True
    # JSON has no separate boolean-as-integer, but Python does, and `True` where
    # an integer is required is almost always a mistake worth reporting.
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    if expected == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, _TYPES[expected])


def validate(value: JsonValue, schema: Schema, pointer: str = "") -> list[SchemaError]:
    """Collect every validation error, rather than stopping at the first.

    All of them go into the repair prompt: fixing one error per round trip is how
    a repair loop turns into four.
    """
    errors: list[SchemaError] = []

    expected = schema.get("type")
    if expected is not None:
        allowed = [expected] if isinstance(expected, str) else list(expected)
        if not any(_matches_type(value, name) for name in allowed):
            errors.append(
                SchemaError(
                    pointer,
                    f"expected type {' or '.join(allowed)}, got {_type_name(value)}",
                )
            )
            return errors

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(json.dumps(option) for option in schema["enum"])
        errors.append(SchemaError(pointer, f"value must be one of [{allowed}]"))

    if isinstance(value, str):
        errors.extend(_validate_string(value, schema, pointer))
    elif isinstance(value, int | float) and not isinstance(value, bool):
        errors.extend(_validate_number(value, schema, pointer))
    elif isinstance(value, list):
        errors.extend(_validate_array(value, schema, pointer))
    elif isinstance(value, dict):
        errors.extend(_validate_object(value, schema, pointer))

    return errors


def _validate_string(value: str, schema: Schema, pointer: str) -> list[SchemaError]:
    errors: list[SchemaError] = []
    if "minLength" in schema and len(value) < schema["minLength"]:
        errors.append(SchemaError(pointer, f"shorter than minLength {schema['minLength']}"))
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(SchemaError(pointer, f"longer than maxLength {schema['maxLength']}"))
    pattern = schema.get("pattern")
    if pattern and not re.search(pattern, value):
        errors.append(SchemaError(pointer, f"does not match pattern {pattern!r}"))
    return errors


def _validate_number(value: float, schema: Schema, pointer: str) -> list[SchemaError]:
    errors: list[SchemaError] = []
    if "minimum" in schema and value < schema["minimum"]:
        errors.append(SchemaError(pointer, f"below minimum {schema['minimum']}"))
    if "maximum" in schema and value > schema["maximum"]:
        errors.append(SchemaError(pointer, f"above maximum {schema['maximum']}"))
    return errors


def _validate_array(value: list[Any], schema: Schema, pointer: str) -> list[SchemaError]:
    errors: list[SchemaError] = []
    if "minItems" in schema and len(value) < schema["minItems"]:
        errors.append(SchemaError(pointer, f"fewer than minItems {schema['minItems']}"))
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        errors.append(SchemaError(pointer, f"more than maxItems {schema['maxItems']}"))
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for position, item in enumerate(value):
            errors.extend(validate(item, item_schema, f"{pointer}/{position}"))
    return errors


def _validate_object(value: dict[str, Any], schema: Schema, pointer: str) -> list[SchemaError]:
    errors: list[SchemaError] = []
    properties: dict[str, Any] = schema.get("properties", {})

    for name in schema.get("required", []):
        if name not in value:
            errors.append(SchemaError(pointer, f"missing required property {name!r}"))

    if schema.get("additionalProperties") is False:
        for name in sorted(set(value) - set(properties)):
            errors.append(SchemaError(f"{pointer}/{name}", "property not allowed by the schema"))

    for name, sub_schema in properties.items():
        if name in value and isinstance(sub_schema, dict):
            errors.extend(validate(value[name], sub_schema, f"{pointer}/{name}"))

    return errors


# -- extraction and syntax repair ----------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")
_PY_LITERAL_RE = re.compile(r"(?<![\"\w])(True|False|None)(?![\"\w])")


def strip_fence(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def isolate_json(text: str) -> str:
    """Return the first balanced ``{...}`` or ``[...]``, ignoring braces in strings.

    Models like to introduce their JSON ("Here is the result:") and to follow it
    with a note. A depth counter that understands string literals and escapes
    finds the value in both cases; a regex for the outermost braces does not.
    """
    start = None
    for position, char in enumerate(text):
        if char in "{[":
            start = position
            break
    if start is None:
        return text

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    return text[start:]


def normalise_syntax(text: str) -> str:
    """The syntax mistakes models actually make, in the order they compound."""
    out = text.replace("\u201c", '"').replace("\u201d", '"')
    out = out.replace("\u2018", "'").replace("\u2019", "'")
    literals = {"True": "true", "False": "false", "None": "null"}
    out = _PY_LITERAL_RE.sub(lambda m: literals[m[1]], out)
    out = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', out)
    out = _TRAILING_COMMA_RE.sub(r"\1", out)
    return out


def _single_to_double_quotes(text: str) -> str:
    """A last resort for output that is Python's ``repr`` rather than JSON.

    Only applied when the text contains no double quotes at all, because
    otherwise a legitimate apostrophe inside a string would be rewritten into a
    delimiter and turn recoverable output into garbage.
    """
    if '"' in text:
        return text
    return text.replace("'", '"')


def coerce(value: JsonValue, schema: Schema) -> tuple[JsonValue, list[str]]:
    """Coerce values where the schema leaves exactly one sensible reading."""
    steps: list[str] = []
    expected = schema.get("type")
    names = [expected] if isinstance(expected, str) else list(expected or [])

    if isinstance(value, str) and names and not any(_matches_type(value, n) for n in names):
        stripped = value.strip()
        if "integer" in names:
            try:
                return int(stripped, 10), ["coerce:string->integer"]
            except ValueError:
                pass
        if "number" in names:
            try:
                return float(stripped), ["coerce:string->number"]
            except ValueError:
                pass
        if "boolean" in names and stripped.lower() in ("true", "false", "yes", "no"):
            return stripped.lower() in ("true", "yes"), ["coerce:string->boolean"]
        if "null" in names and stripped.lower() in ("null", "none", ""):
            return None, ["coerce:string->null"]

    if "array" in names and not isinstance(value, list) and value is not None:
        item_schema = schema.get("items")
        wrapped, sub_steps = (
            coerce(value, item_schema) if isinstance(item_schema, dict) else (value, [])
        )
        return [wrapped], ["coerce:scalar->array", *sub_steps]

    if isinstance(value, list) and "array" in names:
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            out_list = []
            for item in value:
                coerced, sub_steps = coerce(item, item_schema)
                out_list.append(coerced)
                steps.extend(sub_steps)
            return out_list, steps

    if isinstance(value, dict) and (not names or "object" in names):
        properties: dict[str, Any] = schema.get("properties", {})
        out_dict = dict(value)

        if schema.get("additionalProperties") is False:
            for name in sorted(set(out_dict) - set(properties)):
                del out_dict[name]
                steps.append(f"drop:{name}")

        for name, sub_schema in properties.items():
            if not isinstance(sub_schema, dict):
                continue
            if name in out_dict:
                coerced, sub_steps = coerce(out_dict[name], sub_schema)
                out_dict[name] = coerced
                steps.extend(f"{step}@{name}" for step in sub_steps)
            elif "default" in sub_schema and name in schema.get("required", []):
                out_dict[name] = sub_schema["default"]
                steps.append(f"default:{name}")
        return out_dict, steps

    return value, steps


@dataclass(slots=True)
class Attempt:
    """One pass through the repair pipeline."""

    source: str
    steps: list[str] = field(default_factory=list)
    errors: list[SchemaError] = field(default_factory=list)
    parsed: bool = False
    regenerated: bool = False
    #: The coerced value, set whenever ``parsed`` is true. Carrying it here is
    #: what keeps the caller from having to redo the parse to get at it.
    value: JsonValue = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "regenerated": self.regenerated,
            "steps": list(self.steps),
            "errors": [str(error) for error in self.errors],
        }


@dataclass(slots=True)
class RepairResult:
    ok: bool
    value: JsonValue
    attempts: list[Attempt]
    prompt: str = ""

    @property
    def steps(self) -> list[str]:
        return [step for attempt in self.attempts for step in attempt.steps]

    @property
    def errors(self) -> list[SchemaError]:
        return self.attempts[-1].errors if self.attempts else []

    @property
    def regenerations(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.regenerated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "value": self.value,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "regenerations": self.regenerations,
            "steps": self.steps,
            "errors": [str(error) for error in self.errors],
        }


def parse_once(text: str, schema: Schema) -> Attempt:
    """Run the deterministic tier over one candidate string."""
    attempt = Attempt(source=text)
    candidate = text

    fenced = strip_fence(candidate)
    if fenced != candidate:
        attempt.steps.append("unwrap:code_fence")
        candidate = fenced

    isolated = isolate_json(candidate)
    if isolated != candidate.strip():
        attempt.steps.append("isolate:json_value")
    candidate = isolated

    value: JsonValue = None
    try:
        value = json.loads(candidate)
        attempt.parsed = True
    except json.JSONDecodeError:
        repaired = normalise_syntax(candidate)
        if repaired != candidate:
            attempt.steps.append("normalise:syntax")
        try:
            value = json.loads(repaired)
            attempt.parsed = True
        except json.JSONDecodeError:
            quoted = _single_to_double_quotes(repaired)
            if quoted != repaired:
                attempt.steps.append("normalise:quotes")
                try:
                    value = json.loads(quoted)
                    attempt.parsed = True
                except json.JSONDecodeError:
                    pass

    if not attempt.parsed:
        attempt.errors = [SchemaError("", "output is not parseable as JSON")]
        return attempt

    coerced, coerce_steps = coerce(value, schema)
    attempt.steps.extend(coerce_steps)
    attempt.errors = validate(coerced, schema)
    attempt.value = coerced
    return attempt


def repair_prompt(schema: Schema, errors: list[SchemaError], previous: str) -> str:
    """Ask for a correction that names the failures, rather than just rejecting.

    Truncating the previous output matters: echoing a long invalid response back
    costs tokens and, worse, gives the model more of its own mistake to imitate.
    """
    listed = "\n".join(f"- {error}" for error in errors) or "- output was not valid JSON"
    excerpt = previous.strip()
    if len(excerpt) > 400:
        excerpt = excerpt[:400] + " ...[truncated]"
    return (
        "Your previous response did not satisfy the required JSON schema.\n\n"
        f"Schema:\n{json.dumps(schema, indent=2, sort_keys=True)}\n\n"
        f"Your response:\n{excerpt}\n\n"
        f"Problems:\n{listed}\n\n"
        "Reply with corrected JSON only. No explanation, no code fence."
    )


def repair(
    text: str,
    schema: Schema,
    *,
    max_attempts: int = 3,
    regenerate: Callable[[str, int], str] | None = None,
    metrics: Any | None = None,
) -> RepairResult:
    """Coerce ``text`` into a value satisfying ``schema``.

    Without ``regenerate`` this is one deterministic pass. With it, the callable
    is invoked as ``regenerate(prompt, attempt_number)`` for each further attempt,
    which is where a real model call goes -- and where a test passes a scripted
    one instead.
    """
    attempts: list[Attempt] = []
    candidate = text

    for number in range(1, max(1, max_attempts) + 1):
        attempt = parse_once(candidate, schema)
        attempt.regenerated = number > 1
        attempts.append(attempt)

        if not attempt.errors:
            if metrics is not None:
                metrics.repairs.inc("repaired" if attempt.steps or number > 1 else "clean")
            return RepairResult(ok=True, value=attempt.value, attempts=attempts)

        if regenerate is None or number == max_attempts:
            break
        candidate = regenerate(repair_prompt(schema, attempt.errors, candidate), number)

    if metrics is not None:
        metrics.repairs.inc("failed")
    return RepairResult(
        ok=False,
        value=None,
        attempts=attempts,
        prompt=repair_prompt(schema, attempts[-1].errors, candidate),
    )
