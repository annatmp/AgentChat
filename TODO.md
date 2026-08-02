# TODO — what's left before we can run the promised evaluations

Ordered by dependency. Methodology behind the P1/P2 items is in
[docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md); the abstract we have to
deliver on is in [talk.md](talk.md).

Rough sequencing: P0 is infrastructure everything else depends on, P1 is the
four strategies from the abstract, P2 is evaluation, P3 is the experiment
harness, P4 is polish. P0 and P1 can be worked in parallel by two people;
P2 metrics only need the run-record schema from P0, not real data.

---

## P0 — Foundations (blocks everything) — **done**

- [x] **Structured run records.** One JSON per run at `runs/<run_id>.json`:
      run_id (hash of resolved config + input file hashes), full config, file
      hashes, git SHA, per-turn records (speaker, content, tokens in/out,
      latency, retries, selector rationale), final summary, totals, errors.
      Schema in [records.py](agent_chat/records.py); `schema_version: 1`.
      Terminal output is unchanged and is no longer the data.
- [x] **Token, latency and cost accounting** in `call_agent_recorded`. Both SDKs'
      usage objects are captured, including cache tokens; cost from
      [pricing.py](agent_chat/pricing.py) (`null` when a model has no price,
      never 0). `totals` splits conversation / selector overhead / summary /
      failed, so bidding cost can't leak into the quality comparison.
- [x] **Temperature + seed pass-through.** Both on `Agent` and the run config,
      omitted from the request when unset. Anthropic has no seed, so it's
      recorded as a replicate label and seeds the run's RNG.
      *Note:* several current models (Opus 5, Sonnet 5, Opus 4.8/4.7, Fable 5)
      **reject** `temperature` with a 400; `config.resolve()` catches that
      pairing at load time.
- [x] **Record the resolved model string** per call (`model_resolved`).
      *This replaces "pin model snapshot IDs":* current Anthropic IDs
      (`claude-sonnet-4-6`, `claude-haiku-4-5`) are already complete, fixed
      strings and appending a date suffix 404s, so there is no dated snapshot to
      pin. For Azure, `model` is a deployment name you control. Recording what
      the API says it served is the achievable version of that control.
- [x] **Replaced `time.sleep(3)`** with retry + exponential backoff on 429/5xx
      ([retry.py](agent_chat/retry.py)), honouring `retry-after`. SDK-internal
      retries are disabled (`max_retries=0`) so the attempt count reaches the
      record. A call that already streamed tokens is *not* retried — it's
      recorded as an errored call, since retrying would duplicate the turn.
- [x] **Run config file** ([configs/baseline.yaml](configs/baseline.yaml)):
      roster, strategy, prompts, turn budget, temperature, seed, summarizer,
      per-role panel overrides. `main.py` takes a config path, plus `--dry-run`
      (resolve + hash, no API calls) and skip-if-recorded for P3 resumability.
      Rosters are explicit via `load_roster`.
- [x] **Six Scrum role agents** — PO, Backend, Frontend, QA, Scrum Master,
      Architect. `knowledge/<role>.md` holds each one's private context, loaded
      into that agent's system prompt only, hashed into the run record, and
      switchable off via `role_knowledge: false`. Architect knowledge is new;
      the other five are the content from `role_specific_knowledge.md`.
      *Turn budget is now 24 (4 per agent for six), not the 20 in
      EXPERIMENT_DESIGN §7 which assumed five.*
- [x] **Neutral summarizer** — fixed `claude-haiku-4-5`, temperature 0, never a
      participant, outside the turn budget, with its own neutral-scribe role
      prompt. Recorded as `kind="summary"` and excluded from conversation totals.
- [x] **Sanitise speaker-tag echo** (`[critic]: [critic]:`) on ingest
      ([sanitize.py](agent_chat/sanitize.py)). Only the speaker's own leading tag
      is stripped; the raw text is kept in `content_raw` so the sanitiser stays
      auditable.

Also landed, from P4: `pytest` setup plus tests for the pure functions
(sanitiser, usage mapping for both provider shapes, `run_id` determinism,
totals, config validation).

**Not verified against a live API.** There were no credentials in the
environment, so the two provider branches in `call_agent_recorded` have only been
exercised through unit tests of their usage mapping. First real run should be a
2-agent config with `turn_budget: 2` before anything larger.

## P1 — Turn-taking strategies (the talk's core content)

All four are `TurnSelector`s. Each must write its decision rationale into the
run record — for three of them, *why* an agent was picked is the interesting
part. The plumbing for that now exists: `agent_chat/strategies/` holds the
selectors with a name→factory `REGISTRY` a run config can name, and each factory
receives a seeded `rng` plus a `SelectorLog` for rationale
(`log.note(...)`) and for bid costs (`log.add_call(...)`, which land in
`selector_calls` rather than conversation totals).

- [ ] **1. Fixed order** — `round_robin` exists and records its position in the
      cycle. Add starting-speaker rotation across repeats (use the `rng` the
      factory is handed) so position isn't confounded with role.
- [ ] **2. Importance-based self-selection** — each agent runs a think step and
      emits an urgency score; highest bid speaks. Decisions: bid prompt (private
      or shared?), score scale and anchors, tie-breaking, whether an agent sees
      others' bids, anti-monopoly damping for a chronically loud agent.
      Record every bid — "urgency realism" in the abstract depends on it.
      *This is the expensive strategy: N extra LLM calls per turn. Budget for it.*
- [ ] **3. Obligation-first** — detect direct questions in the previous turn,
      give the addressed agent the floor; fall back to the auction from (2) when
      no question was asked. Decisions: detection method (LLM extraction with
      structured output is more robust than regex/name-matching, but adds a call
      and a failure mode — measure both), what happens when several agents are
      addressed, how long an obligation stays open.
- [ ] **4. Facilitator** — a role-based chair with an agenda and explicit
      constraints on speaking dominance. Decisions: does the facilitator consume
      a turn from the budget (it should be counted, and reported), agenda
      representation, dominance rule (hard cap on consecutive turns? share-based
      throttle?), and how it decides the meeting is done.
- [ ] **Shared:** strategies must be deterministic given the same history + seed
      where they don't call an LLM. (Off-roster and empty speaker names are
      already caught by `Conversation._check_speaker`, which records the error and
      aborts rather than raising a bare `KeyError`.)

## P2 — Evaluation

- [ ] **Deterministic metrics module** (`agent_chat/metrics.py`), no LLM calls:
      participation entropy + Gini over turns *and* tokens, dominance, silent
      agents, turn-length stats, redundancy (embedding cosine + n-gram overlap
      fallback), topic drift vs. the task statement, cost/latency split, bid
      statistics. **Unit-test these on hand-built transcripts** — a silent bug
      here corrupts published results and nothing else will catch it.
- [ ] **Judge as a module, not a notebook.** `judge.ipynb` regexes `--- SUMMARY ---`
      out of log text and greps `Total score: X / 25` from prose. Move to a
      module that reads run-record JSON and requires **structured JSON output**
      from the judge. Keep a notebook as a thin presentation layer over it.
- [ ] **Anchored rubric** — describe what each score point 1–5 concretely means.
      Unanchored scales cluster at 4 and can't separate conditions.
- [ ] **Multi-judge panel** — 3 judges from different model families, temp 0,
      transcripts blinded (strip config header and model names), no judge scoring
      a run its own model participated in. Report the mean **and** inter-judge
      agreement (Krippendorff's α or pairwise Spearman).
- [ ] **Question extraction + resolution.** LLM extracts (turn, asker, addressee,
      question) as structured output; a deterministic pass checks for a response
      within k turns. Splitting it this way is far more reliable than asking a
      judge "were questions answered?".
- [ ] **Urgency realism judge** — per decision point, not per conversation.
- [ ] **Judgement cache** keyed on (transcript hash, rubric hash, judge model),
      so a rubric tweak doesn't re-judge everything.
- [ ] **Human validation set** — hand-label ~20 transcripts, report each judge's
      correlation with the human labels. This is the slide that makes the
      "reusable evaluation strategy" claim in the abstract real.
- [ ] **Pairwise judging mode** for conditions that come out within noise:
      blinded, both orders, report win rate + binomial CI and the
      order-inconsistency rate.

## P3 — Experiment harness

- [ ] **Grid runner** — takes a grid spec (strategies × panels × tasks × prompt
      variants × seeds), expands it, executes with the strategy loop *innermost*
      so paired runs share a seed/panel/task block (see EXPERIMENT_DESIGN §1).
- [ ] **Resumable + idempotent** — skip runs whose output JSON exists; a rate
      limit partway through 140 runs must not cost the whole grid.
- [ ] **Bounded concurrency** across runs (not within a conversation — turn order
      is sequential by definition), with per-provider rate limits.
- [ ] **Failure ledger** — record dropped/errored/truncated runs rather than
      silently retrying until success. Robustness differences between strategies
      are a legitimate finding.
- [ ] **Analysis notebook** — paired differences within blocks, bootstrap CIs,
      per-metric (not one leaderboard number), cost-vs-quality scatter.
- [ ] **Pilot at R=2** on Grid 1 before committing to the full budget; set R from
      the observed variance instead of guessing.
- [ ] **Publish the artifacts** — anonymised run records + notebooks in the repo
      so attendees can reproduce and adapt. Promised in the abstract.

## P4 — Repo quality

- [x] `pytest` setup + tests for the sanitiser, usage mapping, `run_id`
      determinism, totals and config validation.
- [ ] Extend those tests to metrics, `_build_history` and bid parsing as they land.
- [x] README rewritten for the run config, run records and the strategy registry.
- [ ] `.gitignore` for `logs/` and `runs/` (decide: are run records committed as
      artifacts, or is only the aggregated dataset? P0 writes `runs/` but does not
      ignore it, so this decision is still open).
- [ ] Decide licence + a `CITATION`/attribution line before making the repo
      public alongside the talk.
- [ ] Cost estimate before the big grid — 140 runs × 20 turns × 5 agents, plus
      selector overhead for strategies 2–4, plus 3 judges. Compute it, don't
      discover it.

---

## Open questions to decide before P1 lands

1. **Do bids happen in private?** If agents see each other's urgency scores, the
   bidding becomes strategic and much more interesting — and much noisier. Pick
   one for the headline result, mention the other as future work.
2. **Does the facilitator's turn count against the budget?** It must at least be
   counted and reported, or strategy 4 gets free tokens.
3. **When is a meeting done?** A fixed turn cap is the clean control, but a
   strategy that can *end* the meeting early is arguably the point of agent
   autonomy. Suggestion: fixed cap for the controlled grid, plus a separate
   uncapped demo run for the talk.
4. ~~**Five roles or six?**~~ **Decided: six** — PO, Backend, Frontend, QA,
   Scrum Master, Architect. Architect's private context is newly written; the
   other five carry over. Consequence: the turn budget is 24, not the 20 in
   EXPERIMENT_DESIGN §7, so that turns-per-agent stays at 4.
5. **Which model families for judges?** Needs to be settled early — the
   no-self-judging rule couples judge choice to participant panel choice.
