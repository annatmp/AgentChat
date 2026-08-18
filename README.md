# agent-chat

A lightweight Python framework for running structured multi-agent conversations using LLMs. Agents are defined in YAML, take turns according to a configurable turn-taking strategy, and every run is recorded as structured JSON so runs can be compared.

The playground is a **Scrum refinement meeting**: six role-based agents turn a business request into a backlog of user stories. This repo backs a conference talk on multi-agent turn-taking — see [talk.md](talk.md).

> **Contributors:** start with [CLAUDE.md](CLAUDE.md) (architecture, conventions, known rough edges — also the context file for AI coding sessions), then [TODO.md](TODO.md) for the backlog and [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md) for the evaluation methodology.

## How it works

1. A **run config** (`configs/*.yaml`) names everything that shapes the run: roster, strategy, prompts, turn budget, temperature, seed.
2. `resolve()` loads that roster, hashes every input file, and derives a `run_id` from the lot.
3. The task prompt is injected as a user message, and `conv.run()` drives the conversation using:
   - a **turn selector** (e.g. `round_robin`) to decide who speaks next
   - a **stop condition** (e.g. `max_turns`) to end the loop
   - optional **post-processors** (e.g. `summarize`) that run after the loop
4. Two artifacts come out: `runs/<run_id>.json` (the data) and `logs/*.log` (the transcript as it looked in the terminal).

## Project structure

```
agent_chat/
  agents.py        # Agent dataclass + YAML loaders
  config.py        # run config -> ResolvedRun (roster, prompts, hashes, run_id)
  conversation.py  # provider clients, history mapping, Conversation loop
  policies.py      # stop conditions, post-processors
  pricing.py       # token prices -> per-call cost
  records.py       # run record schema, totals, provenance
  retry.py         # exponential backoff on 429/5xx
  sanitize.py      # strips echoed speaker tags on ingest
  strategies/      # turn selectors + name->factory registry
agents/            # one YAML per role
knowledge/         # one file per role: private context only that agent sees
configs/           # run configs
prompts/           # system, task and summarizer prompts
runs/              # structured run records — the data
logs/              # terminal transcripts
tests/             # pytest over the pure functions
judge.ipynb        # LLM-as-a-judge over log files, prints a leaderboard
main.py            # entry point
```

## Setup

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

Create a `.env` file with your API credentials:

```env
# Anthropic
ANTHROPIC_API_KEY=your_key_here

# Azure OpenAI — classic GPT deployments (AzureOpenAI client)
# Use the bare base URL — do NOT include /openai/v1/
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_API_VERSION=2024-02-01   # optional, defaults to 2024-02-01

# Azure AI Foundry — serverless models e.g. Mistral, Llama (OpenAI-compatible client)
# Must include /openai/v1/ — the URL is used as-is
AZURE_AI_ENDPOINT=https://your-resource.openai.azure.com/openai/v1/
AZURE_AI_API_KEY=your_key_here

# Google — Gemini, via its OpenAI-compatible endpoint (fixed URL, no per-resource endpoint needed)
GOOGLE_API_KEY=your_key_here

# Mistral — La Plateforme, via its OpenAI-compatible endpoint (fixed URL, not Azure AI Foundry)
MISTRAL_API_KEY=your_key_here

# DeepSeek — via its OpenAI-compatible endpoint (fixed URL)
DEEPSEEK_API_KEY=your_key_here
```

## Usage

```bash
uv run main.py                                   # configs/baseline.yaml
uv run main.py configs/baseline.yaml
uv run main.py configs/baseline.yaml --dry-run       # resolve + hash only, no API calls
uv run main.py configs/baseline.yaml --check-models  # models.list() per provider, no generation tokens
uv run main.py configs/baseline.yaml --force         # redo a run whose record exists
uv run pytest
```

`--dry-run` is the cheap way to check a config: it prints the `run_id`, the resolved roster, and the hash of every input file without spending a token. A run whose `runs/<run_id>.json` already exists is skipped, so re-running a config is idempotent.

`--check-models` calls `models.list()` once per provider in use and checks every configured agent/summarizer model against it — a typo'd or invalid model ID (e.g. a date-suffixed current Anthropic model, which 404s) is caught before any generation call. It costs no conversation tokens. Not every Azure AI Foundry deployment exposes `models.list()`; when it doesn't, that provider's models are reported `unverified` rather than failed.

### Tracing (optional)

To inspect exactly what every call — turns, bids, consensus votes, the summarizer — actually received as input, trace a run to a local [Phoenix](https://arize.com/docs/phoenix) instance:

```bash
uv sync --group tracing        # arize-phoenix + openinference instrumentors; not installed by default
uvx arize-phoenix serve        # local UI at http://localhost:6006
```

Then set `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006` in `.env` and run as usual. Purely observational — it never affects `run_id` or a run's output, and does nothing at all if the env var is unset. See the comment above the tracing block in `main.py` for two library quirks worth knowing if this stops working after an `arize-phoenix` upgrade: `endpoint` needs the `/v1/traces` path (the bare host 405s), and instrumentation is done explicitly per-provider rather than via `auto_instrument=True`, which didn't activate anything when this was last verified.

## Run configs

Everything that could change a conversation's outcome lives here, so it can be recorded and varied. See [configs/baseline.yaml](configs/baseline.yaml).

| Field                            | Description                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------ |
| `roster`                         | Explicit list of agent names. Required — nothing is picked up implicitly.       |
| `strategy`                       | `{name, params}`; `name` must be in the strategy registry                      |
| `task_prompt` / `system_prompt`  | Paths to the prompt files                                                      |
| `summarize_prompt`               | Path to the summarizer template; must contain `{transcript}`                    |
| `role_knowledge`                 | Whether agents get their private context from `knowledge/`                      |
| `turn_budget`                    | Total agent turns (the summary is excluded)                                     |
| `temperature`                    | Overrides per-agent temperature. `null` omits it from requests.                 |
| `seed`                           | Passed to OpenAI-compatible endpoints; a replicate label on Anthropic           |
| `summarizer`                     | The neutral summarizer's model/provider/temperature                             |
| `panel`                          | Per-role `{model, provider, max_tokens, temperature}` overrides                  |
| `output_dir`                     | Where run records are written (default `runs/`)                                 |

Config errors — an unknown provider, a roster name with no YAML, a temperature on a model that rejects one — fail at load time with exit code 2, rather than surfacing as a 400 partway through a grid.

## Run records

One JSON file per run, at `runs/<run_id>.json`. `run_id` is a hash of the resolved config plus the SHA-256 of every prompt, agent and knowledge file used, so two records with the same id came from the same setup, and editing a prompt gives you a new id.

| Field                     | Description                                                                     |
| ------------------------- | ------------------------------------------------------------------------------- |
| `config` / `file_hashes`  | The fully resolved config and a hash per input file                              |
| `git_sha`                 | The commit the run was executed at                                               |
| `turns[]`                 | Per turn: speaker, content, `content_raw` if sanitised, and the call record       |
| `turns[].call`            | Tokens in/out, cache tokens, cost, latency, retries, resolved model, stop reason |
| `turns[].selector`        | Why the selector picked this speaker (bids, agenda state, …)                      |
| `selector_calls[]`        | LLM calls a strategy made on its own behalf, counted as overhead not conversation |
| `summary` / `summary_call`| The judged artifact and what producing it cost                                    |
| `totals`                  | conversation / selector overhead / summary / failed, plus participation per agent  |
| `errors[]`                | Recorded, not retried away — a strategy's robustness is a finding                  |

Models with no entry in [`pricing.py`](agent_chat/pricing.py) report `cost_usd: null` and set `cost_complete: false`, so a missing price never reads as a free run.

## Defining agents

Each agent is a YAML file in `agents/`:

```yaml
name: product_owner
role: |
  You are the Product Owner. You own the business goal and the user's
  perspective, and you are the one who decides scope.
model: claude-sonnet-4-6
provider: anthropic
knowledge: knowledge/product_owner.md
```

| Field         | Description                                                                    |
| ------------- | ------------------------------------------------------------------------------ |
| `name`        | Unique identifier used in turn selectors and history; must match the filename    |
| `role`        | System prompt appended to the shared conversation system prompt                  |
| `model`       | Model name or Azure deployment name                                             |
| `provider`    | `anthropic`, `azure_openai`, `azure_ai`, `google`, `mistral` or `deepseek`      |
| `max_tokens`  | Max tokens per response (default: 4096)                                         |
| `temperature` | Per-agent default; the run config's `temperature` wins when set                  |
| `knowledge`   | Path to this role's private context, appended to its system prompt only          |

## Policies and strategies

### Turn selectors (`agent_chat/strategies/`)

Registered by name so a run config can select one. A strategy is just `(history, agents) -> agent name`.

| Strategy      | Params  | Description                                            |
| ------------- | ------- | ------------------------------------------------------ |
| `round_robin` | `order` | Cycles through agents; defaults to the roster order    |
| `bidding`     | `bid_prompt`, `bid_max_tokens`, `starting_agent` | Every agent runs a private think step scoring 0-4 how urgent it is for them to speak; highest bid wins, ties broken by the run's seeded RNG. Bid calls are recorded as selector overhead, not conversation spend. `starting_agent` skips the auction on the first turn. |
| `obligation_first` | `obligation_prompt`, `obligation_max_tokens`, plus every `bidding` param (used for its fallback) | The last speaker privately reports whether they addressed one specific agent; that agent gets the floor. Falls back to the `bidding` auction when nobody was addressed, the response is unparseable, or more than one agent was named. |
| `facilitator` | `chair` (default `scrum_master`), `facilitator_prompt`, `facilitator_max_tokens`, `facilitator_max_attempts` | The chair privately decides who speaks next each turn, including themselves. On an unparseable pick, retries with corrective feedback up to `facilitator_max_attempts`; if every attempt fails, the run aborts rather than guessing. |

### Stop conditions (`agent_chat/policies.py`)

| Policy                       | Description                                            |
| ---------------------------- | ------------------------------------------------------ |
| `max_turns(n)`               | Stops after `n` agent turns                            |
| `stop_on_keyword(*keywords)` | Stops when a keyword appears in the last agent message |

### Post-processors

| Policy                        | Description                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| `summarize(agent, outcome)`   | Sends the transcript to the neutral summarizer and captures the result       |

## Model selection

Models you can use during this session:

| Model Name                     | Deployment Type |
| ------------------------------ | --------------- |
| gpt-4o                         | azure_openai    |
| gpt-4.1-nano                   | azure_openai    |
| mistral-medium-2505            | azure_ai        |
| Llama-4-Scout-17B-16E-Instruct | azure_ai        |
| Mistral-Large-3                | azure_ai        |

Anthropic model IDs are fixed, complete strings, but *which* form that is varies per model — don't assume. `claude-sonnet-4-6` is bare and 404s with a date suffix; `claude-haiku-4-5` is the opposite, it only resolves as `claude-haiku-4-5-20251001`. Run `uv run main.py <config> --check-models` to verify against the live catalog rather than guessing. The string the API reports serving is recorded per call as `model_resolved`.
