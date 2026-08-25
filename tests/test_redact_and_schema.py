from __future__ import annotations

import json

import pytest

from llmguard import demo, redact, schema
from llmguard.types import Finding, Severity


def f(start, end, *, kind="email", severity=Severity.MEDIUM, confidence=0.9):
    return Finding("pii", kind, start, end, severity, confidence)


# -- redaction -------------------------------------------------------------


def test_overlaps_resolve_by_severity_then_confidence():
    low = f(0, 10, kind="ipv4", severity=Severity.LOW)
    high = f(5, 15, kind="ssn", severity=Severity.HIGH)
    assert [finding.kind for finding in redact.merge([low, high])] == ["ssn"]

    same_a = f(0, 10, kind="email", confidence=0.5)
    same_b = f(0, 10, kind="phone", confidence=0.99)
    assert [finding.kind for finding in redact.merge([same_a, same_b])] == ["phone"]


def test_non_overlapping_findings_all_survive():
    kept = redact.merge([f(0, 4), f(10, 14), f(20, 24)])
    assert [finding.start for finding in kept] == [0, 10, 20]


@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("label", "call [PHONE] now"),
        ("mask", "call (***) ***-**** now"),
        ("partial", "call (***) ***-0142 now"),
        ("remove", "call  now"),
    ],
)
def test_strategies(strategy, expected):
    text = "call (415) 555-0142 now"
    finding = f(5, 19, kind="phone")
    assert redact.apply(text, [(finding, strategy)]) == expected


def test_hash_is_stable_and_salted():
    text = "ada@example.com"
    finding = f(0, len(text))
    first = redact.apply(text, [(finding, "hash")])
    assert first == redact.apply(text, [(finding, "hash")])
    assert first != redact.apply(text, [(finding, "hash")], salt="pepper")
    assert "ada@example.com" not in first


def test_replacements_are_applied_right_to_left():
    """Left to right, every offset after the first replacement would be wrong."""
    text = "a@b.co and c@d.co"
    findings = [(f(0, 6), "label"), (f(11, 17), "label")]
    assert redact.apply(text, findings) == "[EMAIL] and [EMAIL]"


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown redaction strategy"):
        redact.apply("text", [(f(0, 4), "evaporate")])


# -- schema validation -----------------------------------------------------


def test_valid_document_passes():
    value = json.loads(demo.CORRECTED_JSON)
    assert schema.validate(value, demo.INVOICE_SCHEMA) == []


@pytest.mark.parametrize(
    "mutation,pointer,message",
    [
        ({"currency": "ZZZ"}, "/currency", "must be one of"),
        ({"total": -5}, "/total", "below minimum"),
        ({"invoice_id": "nope"}, "/invoice_id", "pattern"),
        ({"line_items": []}, "/line_items", "minItems"),
        ({"paid": "yes"}, "/paid", "expected type boolean"),
        ({"extra": 1}, "/extra", "not allowed"),
    ],
)
def test_each_constraint_is_enforced(mutation, pointer, message):
    value = json.loads(demo.CORRECTED_JSON) | mutation
    errors = schema.validate(value, demo.INVOICE_SCHEMA)
    assert any(error.pointer == pointer and message in error.message for error in errors)


def test_missing_required_property_is_reported():
    value = json.loads(demo.CORRECTED_JSON)
    del value["total"]
    errors = schema.validate(value, demo.INVOICE_SCHEMA)
    assert any("missing required property 'total'" in error.message for error in errors)


def test_all_errors_are_collected_not_just_the_first():
    value = {"invoice_id": "x", "total": -1, "currency": "ZZZ", "paid": 1, "line_items": []}
    assert len(schema.validate(value, demo.INVOICE_SCHEMA)) >= 4


def test_a_bool_is_not_an_integer():
    assert schema.validate(True, {"type": "integer"})
    assert schema.validate(1, {"type": "integer"}) == []


def test_an_integral_float_satisfies_integer():
    assert schema.validate(3.0, {"type": "integer"}) == []
    assert schema.validate(3.5, {"type": "integer"})


def test_union_types():
    both = {"type": ["string", "null"]}
    assert schema.validate("x", both) == []
    assert schema.validate(None, both) == []
    assert schema.validate(3, both)


def test_string_and_array_bounds():
    assert schema.validate("ab", {"type": "string", "minLength": 3})
    assert schema.validate("abcd", {"type": "string", "maxLength": 3})
    assert schema.validate([1, 2, 3], {"type": "array", "maxItems": 2})


# -- extraction ------------------------------------------------------------


def test_isolate_json_ignores_braces_inside_strings():
    text = 'preamble {"note": "a } here", "n": 1} trailing'
    assert json.loads(schema.isolate_json(text)) == {"note": "a } here", "n": 1}


def test_isolate_json_handles_escaped_quotes():
    text = r'{"note": "a \" and a } ", "n": 1}'
    assert json.loads(schema.isolate_json(text))["n"] == 1


def test_isolate_json_returns_input_when_there_is_no_value():
    assert schema.isolate_json("no json at all") == "no json at all"


def test_isolate_json_handles_arrays():
    assert json.loads(schema.isolate_json("here: [1, 2, 3] done")) == [1, 2, 3]


def test_strip_fence():
    assert schema.strip_fence('```json\n{"a": 1}\n```').strip() == '{"a": 1}'
    assert schema.strip_fence('{"a": 1}') == '{"a": 1}'


def test_normalise_syntax_fixes_the_usual_mistakes():
    messy = "{key: 'v', 'flag': True, 'n': 1,}"
    assert "true" in schema.normalise_syntax(messy)
    assert '"key"' in schema.normalise_syntax(messy)
    assert not schema.normalise_syntax(messy).rstrip().endswith(",}")


def test_single_quotes_are_only_swapped_when_unambiguous():
    """A repair that turns an apostrophe into a delimiter makes things worse."""
    risky = '{"note": "it\'s fine"}'
    assert schema._single_to_double_quotes(risky) == risky


# -- the repair loop -------------------------------------------------------


def test_deterministic_tier_fixes_the_messy_document():
    result = schema.repair(demo.MESSY_JSON, demo.INVOICE_SCHEMA)
    assert result.ok
    assert result.regenerations == 0
    assert result.value["total"] == 128.5
    assert result.value["paid"] is False
    assert result.value["line_items"] == [{"sku": "WIDGET-1", "qty": 3}]
    assert "confidence" not in result.value
    assert "unwrap:code_fence" in result.steps


def test_clean_input_needs_no_steps():
    result = schema.repair(demo.CORRECTED_JSON, demo.INVOICE_SCHEMA)
    assert result.ok
    assert result.steps == []


def test_regeneration_is_only_used_when_local_repair_cannot_help():
    result = schema.repair(
        demo.UNFIXABLE_JSON,
        demo.INVOICE_SCHEMA,
        regenerate=demo.scripted_model,
    )
    assert result.ok
    assert result.regenerations == 1


def test_failure_reports_the_prompt_it_would_have_sent():
    result = schema.repair("not json at all", demo.INVOICE_SCHEMA)
    assert not result.ok
    assert result.value is None
    assert "Reply with corrected JSON only" in result.prompt


def test_the_repair_prompt_names_every_error():
    errors = schema.validate({"currency": "ZZZ"}, demo.INVOICE_SCHEMA)
    prompt = schema.repair_prompt(demo.INVOICE_SCHEMA, errors, '{"currency": "ZZZ"}')
    for error in errors:
        assert str(error) in prompt


def test_the_repair_prompt_truncates_a_long_previous_response():
    prompt = schema.repair_prompt(demo.INVOICE_SCHEMA, [], "x" * 5000)
    assert "[truncated]" in prompt
    assert len(prompt) < 2000


def test_repair_stops_after_max_attempts():
    calls = []

    def never_right(prompt, attempt):
        calls.append(attempt)
        return "{}"

    result = schema.repair("{}", demo.INVOICE_SCHEMA, max_attempts=3, regenerate=never_right)
    assert not result.ok
    assert len(result.attempts) == 3
    assert calls == [1, 2]


def test_repair_records_metrics():
    from llmguard.metrics import GuardMetrics

    metrics = GuardMetrics()
    schema.repair(demo.MESSY_JSON, demo.INVOICE_SCHEMA, metrics=metrics)
    schema.repair("garbage", demo.INVOICE_SCHEMA, metrics=metrics)
    assert metrics.repairs.value("repaired") == 1.0
    assert metrics.repairs.value("failed") == 1.0


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"42"', 42),
        ('"3.5"', 3.5),
        ('"true"', True),
        ("42", 42),
    ],
)
def test_scalar_coercions(raw, expected):
    types = {42: "integer", 3.5: "number", True: "boolean"}
    result = schema.repair(raw, {"type": types[expected]})
    assert result.ok
    assert result.value == expected


def test_a_string_that_cannot_be_coerced_is_reported_not_mangled():
    result = schema.repair('"not a number"', {"type": "integer"})
    assert not result.ok


def test_defaults_fill_missing_required_properties():
    spec = {
        "type": "object",
        "required": ["mode"],
        "properties": {"mode": {"type": "string", "default": "safe"}},
    }
    result = schema.repair("{}", spec)
    assert result.ok
    assert result.value == {"mode": "safe"}


def test_result_to_dict_is_serialisable():
    payload = schema.repair(demo.MESSY_JSON, demo.INVOICE_SCHEMA).to_dict()
    assert json.loads(json.dumps(payload))["ok"] is True
