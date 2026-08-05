# TODO — build order for the promised evaluations

The concrete experimental plan is [docs/experiment_setup.md](docs/experiment_setup.md) —
read that first for what's actually being run, with which models, and why.
[docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md) still holds the
statistical methodology the setup doc doesn't restate: paired-difference
analysis, bootstrap CIs, the judge-bias catalogue (position/verbosity/
anchoring bias, blinding). [talk.md](talk.md) is the abstract we have to
deliver on.

Five phases, roughly in dependency order. P0 is done. P1 and P2 can be built
in parallel; both are needed before P3 or P4 can run anything for real.

---

## P0 — Foundations — done

Structured run records (`runs/<run_id>.json`), token/cost/latency accounting,
temperature+seed pass-through, resolved-model recording, retry with backoff,
the run config format, the six-role roster with private knowledge files, a
placeholder summarizer, speaker-tag sanitisation, and pytest coverage of all
of the above. **Not yet verified against a live API** — first real run should
be a 2-agent, `turn_budget: 2` smoke test before anything larger.

## P1 — Turn-taking strategies + early termination

The four strategies from the abstract, plus the consensus-based early stop
that now applies to every condition, solo included
(`docs/experiment_setup.md` → Early Stopping).

- [ ] **Fixed-order** — `round_robin` exists; add starting-position rotation
      across repeats so position isn't confounded with role.
- [ ] **Self-selection (bidding)** — urgency score per turn, highest bid
      speaks. Decide: bids private or shared, tie-breaking, anti-monopoly
      damping for a chronically loud agent.
- [ ] **Obligation-first** — detect direct questions, give the addressee the
      floor; fall back to the bidding auction when none was asked.
- [ ] **Facilitator** — chair role with an agenda and a dominance rule. Its
      own turns must count against the shared turn budget and be reported,
      not run for free (EXPERIMENT_DESIGN.md §2) — the setup doc's turn
      accounting doesn't yet spell this out explicitly.
- [ ] **Consensus early-stop.** Any agent can propose closing; the run ends
      only if every agent votes yes. Decide: is the vote an explicit check
      before every turn (as the setup doc implies) or read off ordinary
      turns; what counts as a "no"; does a failed vote cost a turn from the
      budget. Must degrade cleanly to the solo baseline (self-termination,
      no vote needed).
- [ ] Shared: strategies must be deterministic given history + seed wherever
      they don't call an LLM.

## P2 — The three panel tiers

Solo, homogeneous, mixed — per `docs/experiment_setup.md` → Setup.

- [ ] **Solo baseline.** One agent, union of all six role knowledge files
      presented as different "lenses," same total turn budget as a team,
      "REVIEW ROUND X" framing per round.
- [ ] **Mixed teams.** Pair the six roles into three fixed pairs; rotate
      which of the three team families covers which pair across exactly
      three configurations (cyclic), so every role is covered by every
      family once.
- [ ] **Summarizer.** Wire to the held-out model that never participates in
      any team (setup doc: DeepSeek) — replaces the current
      `claude-haiku-4-5` placeholder, which is in-family with the Anthropic
      panel and would bias that panel's write-up.
- [ ] **Pin concrete models** from the setup doc's Model Selection /
      Pricing tables (GPT-5.6-terra, Claude Sonnet 5, Gemini-3.6-flash for
      teams and judges; DeepSeek for the summarizer) and resolve the
      still-TBD independent judge model.

## P3 — Evaluation

- [ ] **Deterministic metrics module**, no LLM calls: participation entropy
      + Gini (turns and tokens), dominance, silent agents, turn-length
      stats, redundancy, topic drift, bid statistics. Unit-test on
      hand-built transcripts — a silent bug here corrupts a published
      result and nothing else will catch it.
- [ ] **Judge module.** Panel = the 3 team families + 1 independent family
      that never participates in any team or as the summarizer; each judge
      rates a plan 5× (setup doc), all scores stored. Report mean, SD and
      inter-judge agreement — structured JSON output, anchored rubric,
      blinded transcripts.
- [ ] **Human validation set.** Rank the plans with the largest score gaps
      (easiest to rank) plus any run with high judge SD; compare to the
      LLM panel's ranking.

## P4 — Experiment harness

- [ ] **Grid runner** for the 135-run plan (15 solo + 60 homogeneous + 60
      mixed), strategy loop innermost so paired runs share a seed/panel
      block.
- [ ] **Resumable + idempotent** — skip runs whose output JSON already
      exists.
- [ ] **Cost tracking** per the setup doc's definition: conversation +
      selector overhead + summary tokens; judge cost tracked separately and
      never mixed into a run's total.
- [ ] **Pilot at R=2** before committing to R=5 on the full grid.
- [ ] **Analysis notebook** — paired differences within blocks, bootstrap
      CIs, per-metric reporting (not one leaderboard number), cost-vs-quality.

## P5 — Repo quality / release

- [ ] Extend pytest to metrics, `_build_history`, bid parsing and the
      consensus-vote logic.
- [ ] `.gitignore` decision for `logs/` and `runs/`.
- [ ] Licence + `CITATION` line before the repo goes public with the talk.
- [ ] **Cost estimate for the full grid** before running it for real — 135
      runs × turn budget × selector overhead, plus the judge panel rating
      every plan multiple times. Compute it, don't discover it.

---

Already-landed, from the old P4: `pytest` setup plus tests for the
sanitiser, usage mapping, `run_id` determinism, totals and config
validation.
