# Controlled experiment design

How to get from "we ran some conversations and an LLM liked one of them" to a
result you can defend on stage. This is the methodology doc; `TODO.md` is the
build order.

The claim we want to make is causal: **turn-taking strategy affects conversation
quality and backlog quality.** Everything below exists to stop something *other
than the strategy* from explaining the difference.

---

## 1. The factors

| # | Factor | Levels (proposed) | Type |
|---|--------|-------------------|------|
| A | **Turn-taking strategy** | fixed-order, urgency self-selection, obligation-first, facilitator | the treatment |
| B | **Participant models** | homogeneous panels (3× same model) + 1 heterogeneous panel | controlled |
| C | **Task prompt** | simple (calculator), complex (expense splitter) | controlled |
| D | **Prompt variant** | baseline / terse / verbose-rubric system prompt; role knowledge on-off | controlled |
| E | **Repeat (seed)** | r = 1..R | replication |
| F | **Judge model** | 3 judges | *post-hoc*, see §5 |

The one that matters most for cost: **F is not part of the conversation grid.**
Judges score transcripts after the fact, so adding a judge costs judging tokens
only, never a re-run. Design the conversation grid first, then judge everything
with every judge.

### Don't run the full factorial

A × B × C × D × E fully crossed is 4 × 4 × 2 × 6 × R = 192·R conversations. At
~20 turns each that is not a talk-sized budget.

Use a **baseline + one-factor-at-a-time (OFAT)** design instead. Pick a baseline
cell, cross A fully against whatever you're currently varying, and hold the rest
fixed:

- **Grid 1 (headline result).** A × B × E, task fixed to the complex prompt,
  prompts fixed at baseline. `4 × 4 × 5 = 80` runs. This is the result on the
  slide.
- **Grid 2 (task sensitivity).** A × C × E with the baseline model panel.
  `4 × 2 × 5 = 40` runs, of which 20 are already in Grid 1 → 20 new.
- **Grid 3 (prompt sensitivity).** A × D × E with the baseline panel and task.
  `4 × 3 × 5 = 60`, 20 already covered → 40 new.

≈140 conversations total. That is a defensible number and it lets you say
"strategy effect holds across models, tasks and prompt phrasings" — which is a
much stronger claim than a single grid, and it's the claim that survives the
"you just got lucky with your prompt" question from the audience.

### Blocking: reuse the same seed across strategies

The dominant noise source is run-to-run variance, not between-condition
difference. So **pair the runs**: for repeat `r`, use the same seed, the same
task, the same model panel across all four strategies. Then analyse *paired
differences* rather than group means. This typically cuts the sample size you
need by a large factor, and it's free — it's just how you iterate the loops.

```python
for r in range(R):
    for panel in PANELS:
        for strategy in STRATEGIES:      # innermost → strategies share (r, panel)
            run(strategy, panel, seed=r, task=TASK, prompts=BASELINE)
```

---

## 2. Controls that must be in place before collecting anything

These are the ones that will otherwise invalidate the results silently.

**Equal budget across strategies.** A strategy that produces more turns or more
tokens will look better on "completeness" for reasons that have nothing to do
with turn-taking. Fix **both** and report both:

- cap total agent turns (`max_turns(N)`) identically, **and**
- record total prompt+completion tokens per run, and check the strategies land
  in the same range. If a strategy systematically burns 40% more tokens
  (self-selection with a think step will), that's a *finding* — report cost per
  run as a first-class metric next to quality, don't let it leak into the
  quality comparison.
- Bidding/thinking calls made *by the selector* are part of the strategy's cost.
  Count them separately from conversation tokens so you can report both
  "conversation tokens" and "overhead tokens".

**Pinned model versions.** Use dated snapshot IDs, not floating aliases, and
record the exact string in every run record. An alias silently repointing
mid-experiment is the classic way to lose a week of results.

**Temperature and determinism.** Participants: fix temperature (e.g. 0.7) for
all agents, identical across conditions — variety comes from repeats, not from
uncontrolled sampling settings. Judges: temperature 0. Pass a seed where the
provider supports it (OpenAI-compatible endpoints do, best-effort; Anthropic
does not) and record it either way, so "seed" is at minimum a replicate label.

**Model identity vs role.** In a heterogeneous panel, if the PO is Claude and QA
is Llama, you cannot tell whether "the PO dominated" is about the role or the
model. Two defences: (a) make homogeneous panels the primary grid, so every
role runs on the same model; (b) in the heterogeneous panel, **rotate the
model→role assignment** across repeats.

**Randomise ordering artifacts.** Randomise the starting speaker (except where
fixed-order requires otherwise — then rotate the fixed order across repeats),
and randomise the order roles are listed in the shared system prompt. Both are
known to bias which agent dominates.

**Frozen prompt hashes.** Record a SHA of every prompt file and agent YAML used
in the run. When someone tweaks a role description three weeks in, you want to
be able to tell which results are still comparable.

---

## 3. What to measure

Split into two tiers. **Do not let the LLM judge measure things a deterministic
metric can measure** — it's more expensive and less trustworthy.

### Tier 1 — deterministic conversation metrics (no LLM)

Computed from the run record. These are cheap, reproducible, and they're the
backbone of the conversation-level claims in the talk.

| Metric | Definition |
|---|---|
| Participation balance | normalised Shannon entropy of turn counts per agent (1.0 = perfectly even). Also report the Gini coefficient of *tokens* per agent — turn count alone hides an agent that speaks rarely but at length. |
| Dominance | max share of turns / of tokens held by one agent |
| Silence | number of agents with zero turns |
| Turn length | mean/σ tokens per turn, per agent |
| Redundancy | mean pairwise cosine similarity between turn embeddings; plus 5-gram overlap with earlier turns as an embedding-free fallback |
| Topic drift | cosine distance between each turn's embedding and the task statement embedding, as a function of turn index; report slope and endpoint |
| Cost / latency | tokens and wall-clock, split conversation vs. selector overhead |
| Bid statistics | (self-selection only) distribution of urgency scores, how often the top bidder is chosen, bid inflation over time |

### Tier 2 — LLM-as-a-judge

Reserve for things that genuinely need judgement:

- **Backlog quality** — the existing rubric in `judge.ipynb` (clarity,
  completeness, feasibility, testability, risk coverage). Judge the final
  summary.
- **Unanswered questions** — extract every direct question and who it was
  addressed to, then check whether the addressee responded within k turns. Split
  this into two steps: an LLM *extraction* pass (structured output: turn index,
  asker, addressee, question) and a *deterministic* resolution check. Judging
  extraction is much more reliable than judging "were questions answered".
- **Urgency realism** — (self-selection only) given the transcript up to turn t
  and the bids, did the highest bidder actually have the most relevant thing to
  say? Score per decision point, not per conversation.
- **Conversation coherence / repair** — did agents address each other, build on
  each other, resolve disagreements.

Everything Tier 2 produces should be **structured JSON**, not prose with a score
buried in the last line. The current `Total score: X / 25` regex will bite you.

---

## 4. Judge protocol

LLM judges have well-documented, large biases. Each one has a specific defence,
and doing them is the difference between "we scored it with an LLM" and a
result that holds up.

**Self-preference.** A model rates text it produced higher. **Never let a model
judge a conversation it participated in** — or if you do (it's a nice slide),
report it separately as a measured bias, with the same transcript judged by all
three judges side by side.

**Position bias.** In pairwise comparisons, models favour whichever option came
first. If you do pairwise A/B judging, run **both orders** and count only
consistent verdicts; the inconsistency rate is itself a reportable number.

**Verbosity bias.** Longer answers score higher. Because of this, always report
summary length alongside quality scores, and check the correlation. If quality
and length correlate at r > 0.6 you're measuring length.

**Rubric anchoring.** Give each score point a concrete description ("3 = every
story has a title and description, but acceptance criteria are missing or
vague"). Unanchored 1–5 scales cluster at 4 and have almost no resolving power.

**Panel of 3 judges from different families.** Use one Anthropic, one OpenAI,
one open-weights model. Report the mean, and report **inter-judge agreement**
(Krippendorff's α for ordinal scales, or simply pairwise Spearman correlation
between judges). If α is low, your metric doesn't mean anything yet and the
honest move is to fix the rubric, not to average harder.

**Validate against humans.** Hand-label ~20 transcripts yourself against the
same rubric. Report each judge's correlation with your labels. This one slide —
"here is how much you can trust our judge" — is worth more than any additional
condition you could run with the same budget, and it's what makes the
*reusable evaluation strategy* promised in the abstract actually reusable.

**Blind the judge.** Strip model names and any config header from the transcript
before judging. Present conditions in randomised order. The judge must not be
able to infer which strategy produced the transcript — especially the facilitator
strategy, which is structurally obvious from the transcript. Consider judging
*only the final summary* for backlog quality (strategy-invisible) and using
deterministic metrics for conversation-level properties, precisely to sidestep
this.

**Prefer pairwise for close calls.** Absolute Likert scoring has poor resolution.
If two strategies come out within noise on the 1–5 rubric, re-judge them as
randomised, blinded pairwise comparisons of paired runs (same seed, same panel)
and report win rate with a binomial CI.

---

## 5. Analysis

- **Primary comparison:** paired differences between strategies within
  (panel, task, seed) blocks. Report mean difference + bootstrap 95% CI.
  A CI that excludes zero is your result; one that doesn't is also a result —
  "we could not distinguish these two strategies at n=5" is an honest and
  interesting talk finding.
- **Don't report a single leaderboard number.** Report per-metric, with CIs.
  Strategies will likely win on different axes (self-selection: better
  participation balance, worse cost; facilitator: less drift, more rigidity).
  That trade-off *is* the talk's payload.
- **Multiple comparisons.** You're testing 4 strategies × several metrics.
  Either pre-register one primary metric or apply a correction and say so.
- **Report every dropped run.** API errors, truncated summaries, agents emitting
  empty turns. If runs are silently retried until they succeed, the failure rate
  differences between strategies disappear — and robustness is a legitimate
  finding.
- **R = 5 to start.** Run the full Grid 1 at R=2 first, look at the spread, then
  decide R from the observed variance rather than guessing. If a metric's
  between-run σ is as large as the between-strategy difference, more repeats is
  the only fix.

---

## 6. Reliability and reproducibility

- **One JSON per run**, written atomically, containing: run_id (hash of the
  resolved config), full config (models, provider, temperature, seed, strategy,
  turn budget), prompt file hashes, git commit SHA, per-turn records (speaker,
  content, tokens in/out, latency, and *why the selector picked them* — bids,
  scores, agenda state), the final summary, aggregate token/cost totals, and any
  error.
- **Separate the three stages** — `run` (expensive, produces transcripts),
  `metrics` (cheap, deterministic, re-runnable), `judge` (moderate, cacheable).
  You will want to change the rubric after seeing results; you must be able to
  re-judge without re-running conversations.
- **Cache judgements** keyed on (transcript hash, rubric hash, judge model). A
  rubric tweak should only re-judge what changed.
- **Make experiments resumable.** A grid of 140 runs will hit a rate limit
  partway through. Skip runs whose output JSON already exists.
- **Log the failures.** Retry with backoff, but record that a retry happened.
- **Test the metrics.** Participation entropy, question resolution and bid
  parsing are pure functions — unit-test them on hand-built transcripts. A
  silent bug here corrupts a published result and nothing will catch it.

---

## 7. Suggested concrete configuration

A starting point, not a prescription:

- **Panels:** three homogeneous (a large Anthropic model, a large OpenAI model,
  a strong open-weights model) + one heterogeneous with rotated role assignment.
- **Roles:** the five in `role_specific_knowledge.md` — PO, backend, frontend,
  QA, Scrum Master. Five agents is enough for turn-taking to be a real problem
  and small enough to fit a transcript on a slide.
- **Turn budget:** 20 agent turns (4 per agent on average), summary excluded
  from the budget and always produced by a fixed neutral summarizer model.
- **Judges:** three, one per family, temperature 0, structured JSON output,
  transcripts blinded, none of them judging their own panel's runs.
- **Repeats:** R = 5, seeds 0–4, shared across all strategies.
