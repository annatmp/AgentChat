# CLAUDE.md — working notes for AI coding sessions

Context for Claude Code (and any other coding agent) working in this repo.
Humans: read this too — it doubles as the contributor onboarding doc.

## What this project is

A research repo backing a conference talk on **multi-agent turn-taking**
(`talk.md` holds the abstract and outline). The question: *can LLM agents
negotiate their own speaking order, and can we measure whether it helps?*

The playground is a **Scrum refinement meeting**. Role-based agents turn a
business request into a backlog of user stories. We implement several
turn-taking strategies, run them under controlled conditions, and score the
results with LLM-as-a-judge plus deterministic conversation metrics.

Two things follow from "research repo" and should guide every change:

1. **Runs must be reproducible and comparable.** Anything that affects a
   conversation's outcome (model, prompt, strategy, turn budget, temperature)
   belongs in a config that gets recorded with the run. Never hardcode it in a
   place the run record can't see.
2. **The code is a talk artifact.** It gets shown on slides and read by
   attendees afterwards. Prefer small, readable, obvious modules over clever
   abstraction. A strategy should be readable in one screen.

## Repo layout

```
agent_chat/
  agents.py        # Agent dataclass + YAML loaders (load_roster is the experiment path)
  config.py        # run config -> ResolvedRun: roster, prompts, file hashes, run_id
  conversation.py  # provider clients, history mapping, Conversation loop
  policies.py      # stop conditions, post-processors
  pricing.py       # token prices -> per-call cost (None when unpriced, never 0)
  records.py       # run record schema, totals, provenance (run_id, git sha, hashes)
  retry.py         # exponential backoff on 429/5xx, attempt count into the record
  sanitize.py      # strips echoed speaker tags on ingest
  strategies/      # turn selectors + name->factory registry
agents/            # one YAML per role (name, role, model, provider, knowledge)
knowledge/         # one file per role: private context only that agent sees
configs/           # run configs — the only place experimental variables live
prompts/           # system prompt, task prompts, summarizer prompt + role
runs/              # structured run records, one JSON per run — this is the data
logs/              # per-run text transcripts (stdout tee) — presentation only
tests/             # pytest over the pure functions
judge.ipynb        # LLM-as-a-judge over log files, prints a leaderboard
main.py            # entry point: takes a run config, runs one conversation, records it
role_specific_knowledge.md   # index pointing at knowledge/
talk.md            # talk abstract + outline (the contract we have to deliver on)
docs/EXPERIMENT_DESIGN.md    # how controlled experiments should be set up
TODO.md            # what still has to be built
```

## Core abstractions

Everything hangs off three callable types defined in
[conversation.py:298-300](agent_chat/conversation.py#L298-L300):

```python
TurnSelector  = Callable[[list[Message], dict[str, Agent]], str]  # -> agent name
StopCondition = Callable[[list[Message]], bool]
PostProcessor = Callable[[list[Message]], None]
```

`Conversation.run(turn_selector, stop_condition, stream, post_processors)`
loops: ask the selector who speaks, call that agent, append to shared history,
repeat until the stop condition fires, then run post-processors.

**A turn-taking strategy is just a `TurnSelector`.** That is the central design
decision of the repo — keep it that way. If a strategy needs to make its own LLM
calls (e.g. urgency bidding), it does so inside the selector via
`call_agent_recorded(..., kind=KIND_SELECTOR)` and returns a name.

To record *why* it chose someone without changing that signature, a strategy
closes over a `SelectorLog` (`records.py`) handed to it by
`strategies.build()`. `log.note(...)` sets the rationale for the next pick and
`log.add_call(record)` reports a bid's cost; `Conversation.run` drains both after
each turn into `TurnRecord.selector` and `RunRecord.selector_calls`. That keeps
bidding overhead out of the conversation token totals, which
EXPERIMENT_DESIGN §2 requires.

### Run records

`main.py` assembles one `RunRecord` per run and writes it to
`runs/<run_id>.json`. `run_id` hashes the resolved config plus a SHA-256 of every
prompt, agent and knowledge file, so it identifies an experimental cell: a
renamed config keeps the same id, an edited prompt gets a new one, and P3's
resume check is `output_path().exists()`. Read `agent_chat/records.py` before
adding a field — the schema is the contract with every downstream stage, and
`schema_version` needs bumping if you change its meaning.

### Message history

All agents share one `list[Message]`. `_build_history` maps it per-agent: the
agent's own turns become `assistant`, everyone else's become `user` with a
`[speaker]: ` prefix. This is the standard speaker-tag pattern (AutoGen, CrewAI
do the same). Consequence: an agent can see the whole meeting, which is what we
want for a meeting simulation.

### Providers

Six, selected by the `provider` field on an agent:

| provider       | client            | env vars |
| -------------- | ----------------- | -------- |
| `anthropic`    | `anthropic.Anthropic` | `ANTHROPIC_API_KEY` |
| `azure_openai` | `AzureOpenAI`     | `AZURE_OPENAI_ENDPOINT` (bare base URL), `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` |
| `azure_ai`     | `OpenAI` (compat) | `AZURE_AI_ENDPOINT` (**must** end in `/openai/v1/`), `AZURE_AI_API_KEY` |
| `google`       | `OpenAI` (compat) | `GOOGLE_API_KEY` — base URL is fixed (`agent_chat.conversation.GOOGLE_BASE_URL`), unlike Azure there's no per-resource endpoint to configure |
| `mistral`      | `OpenAI` (compat) | `MISTRAL_API_KEY` — base URL is fixed (`agent_chat.conversation.MISTRAL_BASE_URL`), Mistral's own API rather than an Azure AI Foundry deployment |
| `deepseek`     | `OpenAI` (compat) | `DEEPSEEK_API_KEY` — base URL is fixed (`agent_chat.conversation.DEEPSEEK_BASE_URL`) |

`azure_openai`, `azure_ai`, `google`, `mistral` and `deepseek` all speak the
same OpenAI-compatible `chat.completions` API — only the client (base URL +
auth) differs — so they share `_call_openai_compatible` and the
`_OPENAI_COMPATIBLE_CLIENTS` dispatch dict in `conversation.py`. Adding a
provider on that API means one entry in that dict, one branch in
`model_check.py`'s `_list_ids`, and one row in the README table. A provider
with a genuinely different wire format (like `anthropic`) needs its own
`_call_*` function and its own branch in `call_agent_recorded`.

`call_agent_recorded` is the real entry point: it returns a `CallResult`
(`.text` + `.record`) carrying tokens, cost, latency, retries and the resolved
model string. `call_agent` is a thin wrapper returning just the text, kept
stable because the judge notebook calls it — don't change its signature.

Two provider quirks worth knowing:

- **OpenAI-compatible endpoints only report usage if asked.** `stream_options={"include_usage": True}`
  is required, and the usage then arrives in a final chunk whose `choices` list is
  *empty* — a loop that skips chunks without choices silently discards it. Not
  every Azure AI Foundry deployment accepts the flag, so there's a one-shot
  fallback that marks `usage.available = False` rather than reporting zeros.
- **Anthropic has no `seed`.** It's recorded as a replicate label and used to
  seed the run's `random.Random`, but not sent.

Clients are constructed with `max_retries=0` on purpose: retrying happens in
`retry.py`, where the attempt count can reach the run record. A streaming retry
would re-print a partial turn, so a call that has already emitted tokens is not
retried — it's recorded as an errored call instead.

## Conventions

- **Python 3.13**, dependencies via `uv` (`uv sync`, `uv run main.py`).
- `from __future__ import annotations` at the top of modules with type hints.
- Dataclasses over dicts for anything with a fixed shape.
- Policies are **factories returning closures** (`max_turns(4)` returns a
  `StopCondition`). Follow that pattern for new policies — it keeps `main.py`
  declarative and readable on a slide.
- Prompts live in `prompts/*.txt`, never inline in Python. They are experimental
  variables; inlining them makes them impossible to vary or record.
- **Experimental variables live in `configs/*.yaml`, never in `main.py`.** If a
  value could change a conversation's outcome and the run record can't see it,
  it's in the wrong place.
- Fail fast at config load, not mid-run. A 140-run grid must not discover a bad
  provider name or an unsupported `temperature` on run 87 — `config.resolve()`
  validates and raises `ConfigError`, and `main.py` exits 2.
- Secrets in `.env` (gitignored). Never commit keys, never print them into logs.

## Known rough edges (don't be surprised by these)

- **`judge.ipynb` still reads `logs/*.log`, not run records.** It regexes the
  `--- CONFIG ---` / `--- SUMMARY ---` blocks and greps `Total score: X / 25` out
  of prose. `main.py` still prints those markers purely so the notebook keeps
  working; moving the judge onto run records is the top P2 item, and until then
  don't change the header format.
- **`logs/*.log` are stdout dumps, not data.** They're presentation. Anything
  measuring something reads `runs/*.json`.
- **`load_agents` still loads every YAML in `agents/`.** It's fine for the demo
  path; experiments must use `load_roster` / a run config's `roster`, or adding a
  role file silently changes what a run contains.
- **`pricing.py` has no Azure rates** — those are per-subscription and `model` is
  a deployment name there. Runs on those panels report `cost_usd: null` and
  `cost_complete: false`. Token counts are still exact.
- **"Pin model snapshot IDs" isn't a thing for current Anthropic models, but
  which form is fixed varies per model — check, don't assume.** `claude-sonnet-4-6`
  is bare and 404s with a date suffix appended. `claude-haiku-4-5` is the
  opposite: there is no bare form, only `claude-haiku-4-5-20251001` resolves.
  Run `uv run main.py <config> --check-models` (calls `models.list()` per
  provider, no generation tokens spent) before trusting a model string in a new
  config or agent YAML. Reproducibility comes from recording `model_resolved`
  (what the API says it served) per call instead.
- **Runs recorded before the P0 rework** (the three `logs/conversation_*.log`
  files) predate structured records, the six-role roster and the neutral
  summarizer. They are not comparable with anything produced now.

## When making changes

- Changing anything under `prompts/`, `agents/`, `knowledge/`, or a turn selector
  **invalidates previously collected results**. Say so in the commit message, and
  don't mix such a change with an unrelated refactor. (`run_id` will change, which
  is the mechanism that keeps old and new records from being pooled by accident —
  but a human still needs to know.)
- New turn-taking strategy → add it to `agent_chat/strategies/`, register it in
  `REGISTRY`, add a row to the README table, and record its decisions via the
  `SelectorLog`.
- Don't add a dependency without a reason that survives the talk. Current stack
  is deliberately thin: `anthropic`, `openai`, `pyyaml`, `python-dotenv`,
  `jupyterlab`, plus `pytest` as a dev group. Evaluation will justify
  `pandas`/`numpy`/`scipy`; think twice about anything beyond that.
- `uv run pytest` covers the pure functions (speaker-tag sanitisation, usage
  mapping for both provider shapes, `run_id` determinism, totals, config
  validation). Metrics and bid parsing should join them as they land — they're the
  parts where a silent bug quietly corrupts published results.

## Pointers

- `docs/EXPERIMENT_DESIGN.md` — factors, controls, judge protocol, statistics.
- `TODO.md` — prioritized backlog to get from here to the promised results.
- `talk.md` — what we publicly committed to delivering.
