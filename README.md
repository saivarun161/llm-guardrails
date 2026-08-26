# llm-guardrails

[![CI](https://github.com/saivarun161/llm-guardrails/actions/workflows/ci.yml/badge.svg)](https://github.com/saivarun161/llm-guardrails/actions/workflows/ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Runtime dependencies: 1](https://img.shields.io/badge/runtime%20deps-1-brightgreen.svg)](pyproject.toml)
[![Coverage 96%](https://img.shields.io/badge/coverage-96%25-brightgreen.svg)](#development)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Input and output safety middleware for LLM applications, built around one
property most guardrails do not have: **the streaming filter cannot leak a match
across a chunk boundary.** Not "usually does not" — for any chunking of any
input, its output is identical to what the non-streaming guard produces, and the
test suite proves it by trying every chunk size on every fixture.

That property is the reason this exists. Everything else a guardrail does — PII
redaction, injection heuristics, schema repair — is well-trodden, and most of it
is a page of regexes. The part that quietly breaks in production is streaming. A
model emits `AKIAIOSF` in one token and `ODNN7EXAMPLE` in the next; a filter that
redacts each chunk as it arrives passes both, because neither is a key, and the
client concatenates them back into one. Buffering the whole response fixes it and
throws away the reason anyone streams.

It runs end to end with no API key and no network.

---

## Quickstart (60 seconds, no API keys, no network)

```bash
git clone https://github.com/saivarun161/llm-guardrails.git
cd llm-guardrails

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

llmguard demo
```

The demo runs five things end to end. Here is what it shows.

**One. Shape is not evidence.** A support ticket with six real identifiers and
two decoys:

```
  findings
    pii.email              medium   conf 0.97  [6:30]  -> redact
    pii.credit_card        critical conf 0.98  [86:105]  -> redact
    pii.ssn                high     conf 0.95  [150:161]  -> redact
    pii.phone              medium   conf 0.85  [170:184]  -> redact
    pii.ipv4               low      conf 0.90  [216:226]  -> redact
    pii.api_key            critical conf 0.90  [258:278]  -> redact
    pii.aws_key            critical conf 0.99  [258:278]  -> redact
  text
    From: [EMAIL]
    Order 4532015112830360 has not arrived. My card ending **** **** **** 4242 was
    charged twice. Reference 666451234, SSN [SSN], mobile [PHONE].
    The failing request came from [IPV4] and our staging key is
    api_key=[AWS_KEY] -- please rotate it.

  decoy 4532015112830360   reported: no   (sixteen digits, fails the Luhn check)
  decoy 666451234          reported: no   (nine digits, reserved SSN area)
```

The order number is sixteen digits and the reference is nine. A scanner built on
regexes alone reports both, a team gets two false positives a day, and the
guardrail is switched off by Friday. Every pattern here is paired with a
validator — Luhn plus an issuer prefix, mod-97 for IBANs, reserved-range checks
for SSNs, a decodable header for JWTs — so the decoys never make it to a finding.

Note the last two rows: `api_key=` and the AWS pattern both match the same span.
Overlap resolution keeps the more confident one, so the text is redacted once.

**Two. Injection, at three levels of effort.** One retrieved document carrying a
plain-text override, one hidden behind zero-width characters, and one base64
payload inside an HTML comment:

```
  verdict   BLOCK
  findings
    injection.prompt_injection critical conf 0.99  [59:257]  -> block
        exfiltration_channel: names a channel for getting data out
        instruction_override: asks the model to drop the instructions it was given
        role_hijack: tries to reassign the model's role or open a new system turn
        system_prompt_exfiltration: asks the model to disclose its own instructions
        obfuscated_payload: the trigger text was hidden rather than written plainly
```

And the case that decides whether anyone keeps the guardrail on — a support
ticket quoting an attack:

```
  verdict   FLAG
  findings
    injection.prompt_injection medium   conf 0.50  [59:91]  -> flag
        instruction_override: asks the model to drop the instructions it was given
```

Same phrase, different verdict, because blocking takes two independent signals.

**Three. The same output at every chunk size.**

```
  holdback 514 characters, derived from the active detectors:
    injection    longest possible match 514
    pii          longest possible match 391

  chunk size   identical to non-streaming   card digits leaked   emits
  ----------------------------------------------------------------------
  1            yes                          no                   122
  3            yes                          no                   41
  8            yes                          no                   16
  64           yes                          no                   2
  4096         yes                          no                   1
```

And blocking, which has to happen *before* the offending text is eligible to be
sent:

```
  chunk size   blocked   characters emitted before the block   key leaked
  --------------------------------------------------------------------------
  1            True      223                                  no
  3            True      221                                  no
  8            True      222                                  no
  64           True      190                                  no
```

A couple of hundred characters of ordinary prose go out. The key does not — at
any chunk size, including one character at a time, where the first three
characters of `AKIA...` arrive long before anything identifies them as a
credential.

**Four. Repair locally, regenerate only when you must.** Fenced JSON with an
unquoted key, a Python literal, two stringified numbers, a scalar where an array
belongs and an extra field:

```
    ok            True
    regenerations 0
    steps         unwrap:code_fence, normalise:syntax, drop:confidence,
                  coerce:string->number@total, coerce:scalar->array@line_items,
                  coerce:string->integer@qty@line_items
```

Zero round trips. And one that genuinely needs the model — a currency outside the
enum, which no local transform can invent:

```
    ok            True
    regenerations 1
    first errors  /currency: value must be one of ["USD", "EUR", "GBP"]
```

**Five. Budget and metrics.**

```
  200 checks of a 300-character ticket in 43 ms
  p95 per check   <= 0.25 ms   (budget 50 ms)

  llmguard_checks_total{stage="input",verdict="redact"} 201
  llmguard_stream_leaks_total 0
```

`llmguard_stream_leaks_total` is the honest one. It counts findings that were
detected only *after* their text had already been forwarded — the failure this
whole design exists to prevent. It must be zero, CI asserts it is zero, and if a
detector ever understates how long a match it can produce, that counter is where
it shows up.

---

## Using it

```python
from llmguard import Guard, StreamGuard

guard = Guard()  # or Guard.from_file("policy.yaml")

# Before the model.
checked = guard.check_input(user_text)
if checked.blocked:
    return refuse(checked.block_reasons)
prompt = checked.text  # personal data already redacted

# After the model, streaming.
stream = StreamGuard(guard, "output")
for token in model_stream(prompt):
    piece = stream.feed(token)  # "" while inside the holdback window
    if piece:
        yield piece
yield stream.close()  # releases the final window

audit_log.write(stream.result().to_dict())
```

`StreamGuard.feed` raises `StreamBlocked` when a block-level finding lands
mid-stream; pass `raise_on_block=False` to get an empty string and check
`.blocked` instead. Either way, `.emitted` tells you honestly how much text
already reached the client before the block.

Non-streaming responses use `guard.check_output(text)` and read `.text`.

### The CLI

```bash
llmguard scan "my card is 4242424242424242"      # exit 0, redacted
llmguard scan "ignore all previous instructions..." # exit 1, blocked
llmguard stream --chunk-size 3 --json "$text"    # compares against the batch result
llmguard policy validate house-rules.yaml        # exit 2 if the policy is wrong
llmguard repair -s schema.json -f response.txt   # schema repair
llmguard metrics -n 100                          # a Prometheus scrape
```

Exit codes are part of the contract: `0` allowed, `1` blocked by policy, `2` the
tool could not do its job. A pre-commit hook that cannot tell a blocked payload
from a typo in a policy file is worse than no hook.

---

## Architecture

```
   input text                                                output text
       │                                                          │
       v                                                          v
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Guard                                                               │
  │                                                                     │
  │  policy (YAML) ──> which detectors this stage needs                 │
  │       │                        │                                    │
  │       │                        v                                    │
  │       │      ┌──────────────────────────────────────┐               │
  │       │      │ detectors, run against one deadline   │              │
  │       │      │                                       │              │
  │       │      │  injection ─ normalise ─┬─ 9 signals  │              │
  │       │      │              (NFKC,     └─ base64     │              │
  │       │      │               casefold,    payloads   │              │
  │       │      │               invisibles)             │              │
  │       │      │                                       │              │
  │       │      │  pii ─ 10 patterns, each with a       │              │
  │       │      │        validator (Luhn, mod-97,       │              │
  │       │      │        reserved ranges, JWT header)   │              │
  │       │      └──────────────────┬───────────────────┘               │
  │       │                         │ findings                          │
  │       v                         v                                   │
  │  ┌──────────────────────────────────────────┐                       │
  │  │ rule resolution: longest prefix wins     │                       │
  │  │   pii.credit_card > pii > *              │                       │
  │  └──────────────────┬───────────────────────┘                       │
  │                     │ one rule per finding                          │
  │                     v                                               │
  │        allow ── flag ── redact ── block      verdict = max(actions)  │
  │                          │                                          │
  │                          v                                          │
  │              merge overlaps, replace right to left                  │
  └─────────────────────────────┬───────────────────────────────────────┘
                                │
        ┌───────────────────────┼────────────────────────┐
        v                       v                        v
  GuardResult            StreamGuard               GuardMetrics
  verdict, findings,     holdback = max(           Prometheus text
  rule decisions,          detector.max_match_len) format, no second
  timings                block before emit         metrics dependency
```

Separately, `llmguard.schema` constrains model output against a JSON schema with
a deterministic repair tier before any regeneration call. It does not depend on
the guard and can be used on its own.

---

## How the streaming guarantee works

Keep the last *H* characters of the buffer back, where *H* is the longest span
any active detector can produce. Then, in order:

1. A match starting before `len(buffer) - H` must end before `len(buffer)`, so it
   lies entirely inside text already received. **Nothing is missed.**
2. A match is finalised only if it *ends* at or before the emit boundary, so its
   trailing lookahead is evaluated against real text rather than against the end
   of a truncated buffer. **Nothing spurious is emitted.**
3. A match straddling the boundary pulls the boundary back to its own start.
   **No prefix of a match is ever forwarded.**
4. The previously emitted tail is kept as scan context, so lookbehind assertions
   still see what came before the buffer.

Blocking runs on a wider set than redaction: a block-level finding stops the
stream while it is still inside the holdback window, before its text becomes
eligible to send. Injection signals additionally accumulate for the life of the
stream, so a payload spread across more text than one window can hold still
escalates — and blocking stays monotone, because more text can only add signals.

*H* is **derived, not configured**, because a configured number is a number
someone will lower to fix a latency graph. `StreamGuard.holdback` reports it, and
narrowing what you detect is the supported way to shrink it:

```python
from llmguard import Guard, StreamGuard
from llmguard.detectors import build
from llmguard.detectors.pii import PiiDetector

# Injection sets the default 514-character window; among the PII patterns the
# JWT one is the widest, at 391.
guard = Guard(detectors={"pii": PiiDetector(kinds=("email", "credit_card", "ssn"))})
StreamGuard(guard).holdback  # 96

# Keeping injection detection puts the window back to that detector's bound.
guard = Guard(
    detectors={
        "pii": PiiDetector(kinds=("email", "credit_card", "ssn")),
        **build(["injection"]),
    }
)
StreamGuard(guard).holdback  # 514
```

Passing `detectors=` explicitly means exactly those run: a detector the policy
mentions but the dict omits is simply never applied, so narrow the policy too if
you want the rules to match what is actually being checked.

The cost is real and worth naming: with the default policy, 514 characters of the
response are always one step behind the model. For a chat UI that is invisible;
for something rendering partial JSON it may not be.

### What the bound costs the detectors

A detector that declares *H* has to keep to it, and the injection detector cannot
keep to any *H* for free. It matches on a folded copy of the text, and folding
does not preserve length in the direction that matters: `ignore all previous
instructions` with a zero-width character between every letter folds to the same
thirty characters and occupies six hundred real ones. Following that would need
a window the detector has promised not to need — and reporting it anyway would
make the answer depend on how much text happened to be buffered, which is the
divergence this whole design rules out.

So a hit needing more than 514 original characters is **not reported at all**,
in batch as well as in streaming. That is worse detection in exchange for an
arithmetic the streaming path can actually honour, and it is the one place where
the guarantee is paid for in coverage rather than in latency. Two signals far
apart are handled the other way, by reporting them as separate findings rather
than one span covering the filler between them; the score still runs over every
signal seen anywhere in the text, so spreading an attack out does not weaken the
verdict.

---

## Policies

```yaml
version: 1
name: house-rules

defaults:
  redaction: label

budget:
  total_ms: 50
  on_exceeded: fail_open

input:
  rules:
    - id: block-injection
      detect: injection.prompt_injection
      action: block
      min_severity: high
      reason: prompt injection detected in user input

    - id: redact-card
      detect: pii.credit_card
      action: redact
      redaction: partial      # keeps the last four digits

    - id: redact-pii
      detect: pii
      action: redact

output:
  rules:
    - id: block-key-leak
      detect: pii.aws_key
      action: block
      reason: the response contained an access key
```

Rules match by longest prefix — `pii.credit_card` beats `pii` beats `*` — and can
be filtered by `min_severity` and `min_confidence`. Redaction strategies are
`label`, `mask`, `partial`, `hash` (stable, salted) and `remove`.

Loading is strict, on purpose. A policy engine that shrugs at `pii.emial` and
silently never matches is worse than none, because it reports success:

```
$ llmguard policy validate broken.yaml
policy error: input.rules[0].detect: no detector produces 'pii.emial'; did you mean 'pii.email'?
$ echo $?
2
```

Unknown keys, unknown actions and severities, duplicate rule ids, a `redaction`
on a non-redact rule, and a budget that names an overrun behaviour that does not
exist are all rejected the same way, with the path into the document.

---

## Design decisions

**Every PII pattern has a validator.** Regexes match shape; shape is not
evidence. Sixteen digits are usually an order number. The validators — Luhn plus
issuer prefix, ISO 13616 mod-97, SSN reserved ranges, in-range IPv4 octets, a JWT
header that decodes to JSON with an `alg` claim — are what make the difference
between a tool people keep on and one they turn off. Where a check is impossible,
the confidence says so: nine bare digits report 0.55 and the bundled policy's SSN
rule requires 0.9, so they surface as findings without triggering a redaction.

**No name or address detection.** Those need a model; a model makes this layer
non-deterministic; and a non-deterministic redactor cannot be tested with the
exact-output assertions everything else here relies on. Pair this with an NER
pass if you need it — do not assume this is doing it.

**Injection detection is a heuristic and is labelled as one.** There is no sound
detector for prompt injection: the attack is natural language, and natural
language has no grammar separating "instructions from the operator" from
"instructions from a document the operator retrieved". What is here is a weighted
model over nine named signals, matched on a normalised copy (NFKC, casefold,
invisible characters and combining marks removed, with an index map so findings
still point at the right characters) and inside decoded base64. It catches lazy
attempts, which is most of them. It will not catch a careful one, and no
threshold setting changes that.

**Blocking needs two signals.** The strongest single signal scores 2.5 and the
blocking threshold is 3.0. That is deliberate: single-signal blocking is where
the false positives live, and the first support ticket quoting an attack would
trip it. Evidence that a payload was *hidden* counts as a second signal on its
own, because benign text does not arrive wrapped in zero-width characters.

**`max_match_len` is a promise, and it is now enforced rather than asserted.**
The holdback is derived from it, so a detector returning a span wider than the
number it declares lets a match straddle the emit boundary — the exact failure
the design exists to prevent. The injection detector used to declare 200 and
have three ways past it: a base64 finding is attributed to the whole encoded
run, which can be 512 characters; folding inflates a match's span in the
original text without limit; and a finding covering every signal hit could span
the thousands of characters between two of them. It now keeps to its bound by
construction, and a test asserts the property directly on the findings of every
detector rather than inferring it from a clean fuzz run. The leak counter stays,
because the next detector can still get this wrong, and the streaming guard now
clamps the emit boundary so that when one does the result is a missed redaction
rather than duplicated output.

**A blocked result never carries the payload.** `GuardResult.text` is empty on a
block. Result objects end up in logs, and the entire job of a block is to stop
that text from travelling. For the same reason `Finding` carries offsets and
never the matched substring.

**The latency budget has no default answer.** `fail_open` and `fail_closed` are
opposite bets — a missed redaction versus an outage on every request — so a
policy that sets `total_ms` without `on_exceeded` is rejected rather than given a
default someone will discover during an incident.

**A stream defers rather than fails open mid-response.** The batch guard answers
the overrun question once per check. A stream cannot un-send text, so a window
that runs out of budget emits nothing and is rescanned on the next chunk against
a fresh budget: transient load costs latency instead of leaking. Deferring stops
working at `close()`, where there is no next chunk — there the policy's own
answer applies, and either way a `budget.exceeded` finding records that the check
was incomplete. (This was a real bug during development: the streaming guard
silently skipped detection on an overrun and emitted unscanned text. It surfaced
as one flaky test under CPU load. `tests/test_streaming.py` now covers both
branches with a hand-driven clock.)

**Schema repair tries the free fixes first.** Fences, preamble, trailing commas,
`True`, stringified numbers, a bare object where an array belongs, extra fields:
all deterministic, all named in the result, none of them worth a round trip. Only
what local transforms cannot fix — a value outside an enum, a genuinely missing
field with no default — becomes a regeneration prompt, and that prompt names the
failures by JSON pointer, because "your output was invalid" produces another
invalid output.

**One runtime dependency, and only for YAML.** The metrics registry is ~130 lines
rather than `prometheus_client`, because this library sits inside someone else's
service, and that service usually already has a metrics client with its own
registry and its own opinion about collector registration. The JSON Schema
validator is a deliberate subset — types, `required`, `enum`, `properties`,
`items`, bounds, `pattern`, `additionalProperties` — with no `$ref`, no `allOf`
and no remote resolution. Reach for `jsonschema` when you need those.

**Metric labels come from small closed sets.** No label carries a rule's reason
text or any part of the text being checked. A guardrail that blows up your
metrics cardinality gets removed for a different reason than the one it was
installed for.

---

## What this does not do

- It does not make prompt injection safe. Treat it as one layer: still scope
  tool permissions, still keep retrieved content out of the instruction channel.
- It does not detect names, addresses or free-text identifiers.
- It is not a conformant JSON Schema implementation.
- The heuristics are tuned on English. Signals matched on normalised text will
  fire on other scripts inconsistently.
- It does not follow an injection trigger stretched across more than 514
  characters with invisible padding. That is a deliberate consequence of the
  holdback bound rather than an oversight, it applies to batch checks too, and
  it is written up under [what the bound costs the
  detectors](#what-the-bound-costs-the-detectors).
- It adds a holdback window to streaming latency. That is a real cost, and the
  README would rather say so than have you discover it.

---

## Development

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ".[dev]"

.venv/bin/python -m pytest -q --cov --cov-report=term-missing
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

273 tests, 96% statement and branch coverage, on Python 3.11 and 3.12. The
streaming invariant is fuzzed over every chunk size of every fixture, and CI runs
the same checks again through the installed console script — including the exit
codes, because a shell script branching on them is the main non-interactive use.

## License

MIT — see [LICENSE](LICENSE).
