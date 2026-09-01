"""Command line interface.

Exit codes are part of the contract, because the main non-interactive use of this
tool is a pre-commit hook or a CI step:

``0``
    Allowed, possibly after redaction.
``1``
    Blocked by policy, or a schema repair that did not converge.
``2``
    The tool could not do its job: a bad policy, unreadable input. Distinguishing
    this from ``1`` is what stops a typo in a policy file from reading as a clean
    scan.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

from . import demo as fixtures
from . import policy as policy_module
from . import schema as schema_module
from .detectors import known_labels
from .detectors.injection import describe
from .detectors.pii import PiiDetector
from .engine import Guard
from .streaming import StreamGuard, holdback_report
from .types import GuardResult, PolicyError, StreamBlocked

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_USAGE = 2

_RULE = "=" * 78


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def _print_result(result: GuardResult, *, show_text: bool) -> None:
    print(f"  verdict   {result.verdict.label.upper()}  ({result.elapsed_ms:.2f} ms)")
    if result.findings:
        print("  findings")
        for index, finding in enumerate(result.findings):
            rule = next((d for d in result.decisions if d.finding_index == index), None)
            action = rule.action.label if rule else "-"
            print(
                f"    {finding.label:<22} {finding.severity.label:<8} "
                f"conf {finding.confidence:.2f}  [{finding.start}:{finding.end}]  -> {action}"
            )
            for reason in describe(finding.signals):
                print(f"        {reason}")
    else:
        print("  findings  none")
    if result.blocked:
        for reason in result.block_reasons:
            print(f"  blocked   {reason}")
    elif show_text:
        print("  text")
        for line in result.text.splitlines():
            print(f"    {line}")


def _load_guard(args: argparse.Namespace) -> Guard:
    policy = policy_module.load(args.policy) if args.policy else policy_module.default()
    return Guard(policy)


def _read_input(args: argparse.Namespace) -> str:
    """Text from the argument, a file, or stdin, in that order.

    The local annotations are load-bearing: ``argparse.Namespace`` types every
    attribute as ``Any``, so without them the value flows out of here unchecked
    and takes the rest of the CLI's type safety with it.
    """
    text: str | None = args.text
    if text is not None:
        return text
    path: str | None = args.file
    if path:
        return Path(path).read_text(encoding="utf-8")
    return sys.stdin.read()


# -- commands -------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    guard = _load_guard(args)
    text = _read_input(args)
    result = guard.check_input(text) if args.stage == "input" else guard.check_output(text)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"stage {result.stage}, policy {result.policy}")
        _print_result(result, show_text=True)
    return EXIT_BLOCKED if result.blocked else EXIT_OK


def cmd_stream(args: argparse.Namespace) -> int:
    guard = _load_guard(args)
    text = _read_input(args)
    stream = StreamGuard(guard, args.stage, raise_on_block=False)

    pieces = [stream.feed(part) for part in fixtures.chunk(text, args.chunk_size)]
    pieces.append(stream.close())
    streamed = "".join(pieces)

    batch = (guard.check_input if args.stage == "input" else guard.check_output)(text)
    agrees = streamed == batch.text and not stream.blocked

    if args.json:
        print(
            json.dumps(
                {
                    "chunk_size": args.chunk_size,
                    "holdback": stream.holdback,
                    "blocked": stream.blocked,
                    "matches_batch": agrees,
                    "streamed": streamed,
                    "batch": "" if batch.blocked else batch.text,
                },
                indent=2,
            )
        )
    else:
        print(f"holdback   {stream.holdback} characters (derived from the active detectors)")
        print(f"chunk size {args.chunk_size}")
        print(f"blocked    {stream.blocked}")
        print("streamed")
        print(f"    {streamed}")
        print(f"matches non-streaming output: {agrees}")
    return EXIT_BLOCKED if stream.blocked else EXIT_OK


def cmd_policy(args: argparse.Namespace) -> int:
    if args.policy_command == "labels":
        for label in known_labels():
            print(label)
        return EXIT_OK

    policy = policy_module.load(args.path) if args.path else policy_module.default()
    if args.policy_command == "show":
        print(policy.dump(), end="")
        return EXIT_OK

    print(f"{policy.name}: valid")
    print(f"  input rules   {len(policy.input_rules)}")
    print(f"  output rules  {len(policy.output_rules)}")
    budget = policy.budget
    if budget.unlimited:
        print("  budget        unlimited")
    else:
        print(f"  budget        {budget.total_ms:g} ms, {budget.on_exceeded} on overrun")
    guard = Guard(policy)
    for stage in ("input", "output"):
        contributions = ", ".join(
            f"{name} {length}" for name, length in holdback_report(guard, stage)
        )
        print(f"  {stage:<6} holdback {guard.holdback_for(stage)} chars  ({contributions})")
    return EXIT_OK


def cmd_repair(args: argparse.Namespace) -> int:
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    text = _read_input(args)
    result = schema_module.repair(text, schema, max_attempts=args.max_attempts)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"ok       {result.ok}")
        print(f"steps    {', '.join(result.steps) or 'none'}")
        if result.ok:
            print(json.dumps(result.value, indent=2))
        else:
            for error in result.errors:
                print(f"  error  {error}")
    return EXIT_OK if result.ok else EXIT_BLOCKED


def cmd_metrics(args: argparse.Namespace) -> int:
    guard = _load_guard(args)
    for _ in range(args.iterations):
        guard.check_input(fixtures.TICKET)
        guard.check_output(fixtures.BENIGN)
    print(guard.metrics.render(), end="")
    return EXIT_OK


def cmd_demo(_args: argparse.Namespace) -> int:
    guard = Guard()

    print(_RULE)
    print("1. Validated detection: shape alone is not evidence")
    print(_RULE)
    print(
        "A support ticket with six real identifiers, plus two decoys: a sixteen-digit\n"
        "order number that fails Luhn, and a nine-digit reference in a reserved SSN\n"
        "range. Neither is reported."
    )
    print()
    result = guard.check_input(fixtures.TICKET)
    _print_result(result, show_text=True)
    print()
    for decoy, why in (
        ("4532015112830360", "sixteen digits, fails the Luhn check"),
        ("666451234", "nine digits, reserved SSN area"),
    ):
        offset = fixtures.TICKET.index(decoy)
        reported = any(f.start < offset + len(decoy) and offset < f.end for f in result.findings)
        print(f"  decoy {decoy:<18} reported: {'YES' if reported else 'no':<4} ({why})")
    print()
    print("  The card that does pass Luhn keeps its last four digits, on purpose: a")
    print("  support agent needs them to confirm the card with the caller.")

    print()
    print(_RULE)
    print("2. Injection, at three levels of effort")
    print(_RULE)
    print(
        "The same document carries a plain-text override, one hidden behind zero-width\n"
        "characters, and one base64-encoded inside an HTML comment."
    )
    print()
    retrieved = guard.check_input(fixtures.RETRIEVED)
    _print_result(retrieved, show_text=False)

    print()
    print("And the case that matters more -- a support ticket that describes an attack:")
    print()
    benign = guard.check_input(fixtures.BENIGN)
    _print_result(benign, show_text=False)
    print(
        "\n  Blocking needs two independent signals. One phrase in a sentence about\n"
        "  phishing is not an attack, and a guardrail that says it is gets turned off."
    )

    print()
    print(_RULE)
    print("3. Streaming: the same answer at every chunk size")
    print(_RULE)
    batch = guard.check_output(fixtures.STREAM_TEXT)
    holdback = StreamGuard(guard).holdback
    print(f"  holdback {holdback} characters, derived from the active detectors:")
    for name, length in holdback_report(guard, "output"):
        print(f"    {name:<12} longest possible match {length}")
    print()
    print("  A 636-character response, redacted while streaming.")
    print()
    print("  chunk size   identical to non-streaming   card digits leaked   emits")
    print("  " + "-" * 70)
    for size in (1, 3, 8, 64, 4096):
        stream = StreamGuard(guard, "output", raise_on_block=False)
        pieces = [stream.feed(part) for part in fixtures.chunk(fixtures.STREAM_TEXT, size)]
        emits = sum(1 for piece in pieces if piece)
        pieces.append(stream.close())
        streamed = "".join(pieces)
        identical = streamed == batch.text
        leaked = "4242424242424242" in streamed
        print(
            f"  {size:<12} {'yes' if identical else 'NO':<28} "
            f"{'YES' if leaked else 'no':<20} {emits}"
        )
    print()
    print("  output (identical in every row above)")
    for line in _wrap(batch.text, 72):
        print(f"    {line}")

    print()
    print("  And the case a per-chunk filter gets wrong. An access key in a response")
    print("  is a block, not a redaction, and no prefix of it may go out:")
    print()
    print("  chunk size   blocked   characters emitted before the block   key leaked")
    print("  " + "-" * 74)
    for size in (1, 3, 8, 64):
        stream = StreamGuard(guard, "output", raise_on_block=False)
        pieces = [stream.feed(part) for part in fixtures.chunk(fixtures.STREAM_BLOCK_TEXT, size)]
        pieces.append(stream.close())
        streamed = "".join(pieces)
        leaked = any(
            fixtures.SECRET[:length] in streamed for length in range(6, len(fixtures.SECRET) + 1)
        )
        print(
            f"  {size:<12} {stream.blocked!s:<9} {stream.emitted:<36} {'YES' if leaked else 'no'}"
        )
    print()
    print("  The key becomes recognisable only at its fourth character. A filter that")
    print("  redacts each chunk as it arrives has already forwarded the first three.")
    print()
    narrow = Guard(
        guard.policy,
        detectors={"pii": PiiDetector(kinds=("email", "credit_card", "ipv4", "phone", "ssn"))},
    )
    print(f"  The window is a cost: {holdback} characters of the response are always")
    print("  one step behind. It is derived, not configured, so the way to shrink it is")
    print("  to narrow what you detect -- dropping token detection from the output stage")
    print(f"  takes it to {StreamGuard(narrow).holdback}.")

    print()
    print(_RULE)
    print("4. Schema repair: fix locally, regenerate only when you must")
    print(_RULE)
    repaired = schema_module.repair(fixtures.MESSY_JSON, fixtures.INVOICE_SCHEMA)
    print("  Fenced JSON with an unquoted key, a Python literal, two stringified")
    print("  numbers, a scalar where an array belongs and an extra field:")
    print(f"    ok            {repaired.ok}")
    print(f"    regenerations {repaired.regenerations}")
    print(f"    steps         {', '.join(repaired.steps)}")
    print()
    retried = schema_module.repair(
        fixtures.UNFIXABLE_JSON,
        fixtures.INVOICE_SCHEMA,
        regenerate=fixtures.scripted_model,
    )
    print("  And one no local transform can fix -- a currency outside the enum:")
    print(f"    ok            {retried.ok}")
    print(f"    regenerations {retried.regenerations}")
    print(f"    first errors  {'; '.join(str(e) for e in retried.attempts[0].errors)}")

    print()
    print(_RULE)
    print("5. Budget and metrics")
    print(_RULE)
    iterations = 200
    started = time.perf_counter()
    for _ in range(iterations):
        guard.check_input(fixtures.TICKET)
    wall = (time.perf_counter() - started) * 1000.0
    p95 = guard.metrics.duration.quantile(0.95, "input") * 1000.0
    print(f"  {iterations} checks of a {len(fixtures.TICKET)}-character ticket in {wall:.0f} ms")
    print(f"  p95 per check   <= {p95:.2f} ms   (budget {guard.policy.budget.total_ms:g} ms)")
    print()
    for line in guard.metrics.render().splitlines():
        if line.startswith(("llmguard_checks_total", "llmguard_stream_leaks")):
            print(f"  {line}")
    print()
    print("  llmguard_stream_leaks_total is the honest one: it counts findings that")
    print("  were detected only after their text had already gone out. It must be 0.")
    return EXIT_OK


# -- wiring ---------------------------------------------------------------


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("text", nargs="?", default=None, help="text to check; omit to read stdin")
    parser.add_argument("-f", "--file", help="read the text from a file")
    parser.add_argument("-p", "--policy", help="path to a policy YAML file")
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmguard",
        description="Input/output guardrails for LLM applications.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="check one piece of text against a policy")
    _add_input_args(scan)
    scan.add_argument("--stage", choices=("input", "output"), default="input")
    scan.set_defaults(func=cmd_scan)

    stream = subparsers.add_parser(
        "stream", help="filter text in chunks and compare with the non-streaming result"
    )
    _add_input_args(stream)
    stream.add_argument("--stage", choices=("input", "output"), default="output")
    stream.add_argument("--chunk-size", type=int, default=4)
    stream.set_defaults(func=cmd_stream)

    policy = subparsers.add_parser("policy", help="validate, inspect or list policy vocabulary")
    policy.add_argument("policy_command", choices=("validate", "show", "labels"), help="what to do")
    policy.add_argument("path", nargs="?", default=None, help="policy file; omit for the default")
    policy.set_defaults(func=cmd_policy)

    repair = subparsers.add_parser("repair", help="coerce model output into a JSON schema")
    _add_input_args(repair)
    repair.add_argument("-s", "--schema", required=True, help="path to a JSON schema file")
    repair.add_argument("--max-attempts", type=int, default=1)
    repair.set_defaults(func=cmd_repair)

    metrics = subparsers.add_parser("metrics", help="render a Prometheus scrape after N checks")
    _add_input_args(metrics)
    metrics.add_argument("-n", "--iterations", type=int, default=50)
    metrics.set_defaults(func=cmd_metrics)

    demo = subparsers.add_parser("demo", help="the guided walkthrough; no keys, no network")
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except PolicyError as exc:
        print(f"policy error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except StreamBlocked as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
