# Decisions

Running log of design decisions made while working through the Experimentation
Guide, in the order we reach them. Each entry links back to the question it
belongs to. Items under "Proposed, not yet confirmed" are things I suggested
that haven't had an explicit yes from Anna yet — don't treat them as settled.

---

## Research question structure (consolidated)

The original guide framed Q1 (team vs. solo), Q2 (which strategy wins), and
their sub-questions (agent eagerness, model differences) as separate
questions. Working through Q1's design made clear they're mostly different
cuts of **one grid**: strategy × model panel × repeat, task and prompts held
fixed. Q1 reads it as team-vs-solo per strategy; Q2 reads the same data as
strategy-vs-strategy; "differences between models" is just asking whether the
strategy effect holds across panels, which is the reason the grid crosses
strategy with panel at all rather than fixing one model. Eagerness is a
descriptive metric that falls out of the bid data for two of the four
strategies (self-selection, and obligation-first's fallback auction), not a
standalone hypothesis needing its own design.

**Decided:** treat this as one primary grid and analysis plan rather than
three separate studies, while still naming Q1/Q2 as distinct questions in the
eventual writeup for narrative clarity. Q3 (mixed-model teams) stays a
genuinely separate grid — see below, now elevated from "footnote" to a
first-class part of the plan.

---

## Q1 — Are scrum teams better than a single agent?

### Decided

- **Turn budget parity.** The single agent gets the same total turn budget as
  the team (e.g. if a five-agent team gets 20 turns total, the single agent
  also gets 20 rounds), not a smaller per-agent-equivalent budget. Each round
  is prompted with a cue like "REVIEW ROUND X."
- **Knowledge access.** The single agent receives the union of all role
  knowledge files (PO, backend, frontend, QA, Scrum Master, Architect),
  presented as different "lenses" within its own prompt. Rationale: if the
  team collectively sees private knowledge the single agent doesn't, any
  quality gap could just reflect information access rather than team
  structure — this keeps the comparison about structure, not information.
- **Which strategy represents "the team."** The single-agent condition is
  compared against all four turn-taking strategies rather than picking one to
  stand in for "the team," so the result can be strategy-specific — e.g. "the
  team beats the single agent under facilitator and obligation-first, but not
  under fixed-order" — instead of one aggregate verdict. Concretely, this
  does *not* mean re-running the solo condition four times: a single solo run
  per (panel, seed) block is reused as the shared baseline, compared against
  each of the four strategies' results for that same block. Solo doesn't
  have a turn-taking strategy to vary, so it isn't multiplied by that factor
  the way team panels are — see the run-count table under Q3 below for how
  this plays out arithmetically.
- **Model panel parity.** Run the single agent once per model panel (a single
  Claude agent vs. a five-Claude team, a single GPT agent vs. a five-GPT team,
  etc.), rather than one fixed model for every single-agent run. Otherwise a
  quality or cost difference might reflect which model was picked for the
  solo condition rather than team structure. All Q1 panels are homogeneous at
  this stage — see sequencing note below for how the mixed panel (Q3) relates.

- **Summarizer parity.** Pass the single agent's final round through the same
  fixed neutral summarizer used for team runs, before judging, rather than
  judging its last message directly. Keeps the artifact-production step
  identical across conditions so the only variable is how many agents did the
  reasoning.
- **Summarizer must be a held-out model.** Model-family self-preference bias
  (a model favouring output from its own family) is a documented risk for
  judges in EXPERIMENT_DESIGN.md, but the same risk applies to the
  summarizer: if it shares a family with the team it's writing up, it could
  render that team's contribution more favourably in the artifact itself,
  before any judge sees it. Fix: use a model that is not used as a
  participant in *any* panel, rather than matching the summarizer to each
  team's family (which would also make the summarizer vary across
  conditions, undermining the "identical across every condition" invariant
  `policies.py`'s `summarize()` already documents).
  **Conflicts with current code:** `TODO.md` records the summarizer as
  already wired to `claude-haiku-4-5`, which is Anthropic — in-family with
  the Anthropic-homogeneous panel. Needs a concrete held-out model chosen
  before this is implemented; flagged as a TODO revisit, not yet fixed in
  code per "finish the guide before editing code."

- **Cost is descoped as a primary comparison for now.** "Are teams cheaper
  than a single agent" is not a question Anna wants to optimize the design
  around at this stage — it becomes more interesting later if comparing
  several differently-sized models from within one family, which isn't part
  of the current scope. Cost is still something to report for every run, just
  not a metric the experiment design needs to control for or split apart yet.
- **Total run cost definition.** Full token usage of the run, including the
  summarizer call, but excluding judge cost. This lines up with the existing
  three-stage separation in EXPERIMENT_DESIGN.md §6 (run / metrics / judge as
  separate stages) — judging happens after the fact over a stored transcript,
  so its cost was never part of a run's own totals to begin with. In terms of
  the totals `records.py` already splits out (conversation / selector
  overhead / summary / failed), "total run cost" means all of those added
  together, not just the conversation figure.

---

## Q3 — Mixed-model teams: does mixing families help?

Elevated from an afterthought to a genuine part of the plan — Anna wants this
tested properly, not just mentioned as future work.

### Decided

- **Sequencing.** If homogeneous teams beat the single agent in Q1, the mixed
  panel is worth building as a direct follow-on. If homogeneous teams don't
  beat the single agent, the more basic "does team structure help at all"
  question wasn't answered yes, so mixed panels become lower priority. Q3
  is conditionally downstream of Q1's result, not a parallel independent
  track — but "downstream" now means "runs right after," not "maybe
  someday."
- **Which families are available to mix.** Only the 3 team families
  (Anthropic, OpenAI, and one of Google/Llama), not the 4th family held out
  for the summarizer — that family needs to stay disjoint from every team
  panel, homogeneous or mixed, or the summarizer-neutrality decision under Q1
  breaks for this condition too.
- **Model-role confound.** EXPERIMENT_DESIGN.md already flags this: if the PO
  is Claude and QA is Llama, you can't tell whether "the PO dominated" is
  about the role or the model. Defence: rotate the model→role assignment
  across repeats rather than fixing it, so no single family gets permanently
  paired with a single role.

### Decided (continued)

- **Number and construction of heterogeneous configs: 3, via role-pairing
  and cyclic rotation.** Pair the 6 roles into 3 fixed pairs (e.g.
  PO+Backend, Frontend+QA, Scrum Master+Architect), then rotate which family
  covers which pair across exactly 3 configurations, cyclically, so every
  role is covered by every family exactly once across the 3 configs, and
  every configuration stays balanced (each family covers exactly 2 roles per
  config, so no config is family-dominant). This avoids the full
  combinatorial explosion of covering every possible role↔family assignment.
  **Known trade-off:** the two roles inside a fixed pair always share a
  family with each other in every configuration — e.g. PO and Backend are
  never split across different families. That interaction (what happens when
  *these two specific roles* are on different families) isn't observable
  with this design. Accepted as a reasonable trade for keeping this to 3
  configs instead of a much larger set.
- **Crossed against all four strategies — the full grid, not a subset.**
  Rationale: mixed teams might behave differently under different strategies
  than homogeneous teams do — e.g. a self-selection auction might play out
  differently when the loudest bidder is also a different model family than
  the rest of the team — so restricting the mixed-panel runs to only
  whichever strategy won in the homogeneous grid could hide exactly the
  interaction Q3 is trying to find. Costs more (60 runs for this tier instead
  of 15) but keeps Q3's answer independent of Q2's outcome rather than
  conditional on it.

### Run-count arithmetic (at R = 5 repeats)

The panel dimension now has 9 distinct settings across three tiers, each
multiplying differently by strategy and repeats since solo has no
turn-taking strategy to vary:

| Tier | Panel configs | × Strategy | × Repeats | = Runs |
|---|---|---|---|---|
| Solo baseline | 3 (one per family) | n/a | 5 | 15 |
| Homogeneous team | 3 (one per family) | 4 | 5 | 60 |
| Heterogeneous team | 3 (rotated mixed configs) | 4 | 5 | 60 |
| **Total** | | | | **135** |

In line with the ~140-run budget EXPERIMENT_DESIGN.md was already targeting
for the original three-grid OFAT design, so this is comparable scope, not a
blowout — it's just organized around panel structure (solo / homogeneous /
mixed) rather than the original grid-1/2/3 split by factor.

---

## Cross-cutting: judge panel

Anna's proposal: use all models for judging, average, and drop the highest
and lowest score (Olympic-style trimmed mean) — raised while thinking through
the summarizer's family-bias problem, since the same self-preference concern
applies to judges.

- **Self-preference still needs per-run exclusion if judges overlap with
  participants.** EXPERIMENT_DESIGN.md already rules out a judge scoring a
  run its own model participated in. If "all models" means every model used
  anywhere in the study, then for any given run the judge(s) sharing a family
  with that run's team must be excluded — meaning the judge panel size for a
  run *varies* depending on which panel produced it, which breaks having one
  consistent panel to compute inter-judge agreement (Krippendorff's α) over.
- **Trimmed mean needs enough judges to mean anything.** Dropping the highest
  and lowest of exactly 3 judges leaves one value — that's a median, not an
  average, even though it's calculated the same way as Olympic scoring on a
  larger panel. If a run's self-preference exclusion drops it to 2 available
  judges, trimming can't apply at all.
- **Superseded — model access is limited to four families** (Anthropic,
  OpenAI, Google, Llama), not enough to keep judges and participants fully
  disjoint and still have a real panel on each side.
- **Decided instead:** drop the no-self-judging exclusion and use all four
  families as judges on every run, including the run's own family, then
  trim the highest and lowest score and average the remaining two. This
  works because the mechanism is different from the summarizer case: with a
  4-judge panel, an in-family judge is one vote out of four that gets
  averaged against three out-of-family votes, so its bias is diluted rather
  than deciding the score outright. Applied uniformly (every condition's own
  family always sits on that condition's judging panel), the bias is
  symmetric across conditions rather than favouring one over another,
  *provided* self-preference bias is roughly similar in size across model
  families — which isn't guaranteed. This is the "report it separately as a
  measured bias" option EXPERIMENT_DESIGN.md §4 already names as a fallback
  when self-judging can't be avoided; worth doing for real: report each
  run's in-family score alongside the trimmed mean so the bias is visible,
  not just absorbed.
  **Caveat for Q3 (mixed panels):** a heterogeneous team run pulls in-family
  judges from *every* family represented in that team, not just one, so a
  mixed panel could end up with more of its judging panel "in-family" than a
  homogeneous panel gets. Worth watching when comparing Q1/Q2 (homogeneous)
  results against Q3 (mixed) — flagged, not yet resolved.
  **New conflict this creates:** the held-out-summarizer decision above
  assumed a family free of any team-participant duty. If all four available
  families are used as team panels, none is left over to be the neutral
  summarizer — dilution doesn't rescue the summarizer the way it does the
  judges, because there's only one summarizer, not a panel to average across.
  Needs a decision: hold one family out of the team panels entirely
  (three homogeneous panels instead of four, reserving the fourth for
  summarizer duty), or accept an in-family summarizer for whichever
  condition shares its family and report that as a limitation too.
- **Resolved:** three families for team panels (Anthropic, OpenAI, one of
  Google/Llama), the fourth held out exclusively for the summarizer.
- **Should the held-out summarizer family also sit on the judge panel?
  Decided: no.** The dilution argument that justifies letting each team's own
  family judge it doesn't transfer here. A team's own-family judge is only
  "in-family" for *that one condition* — different condition, different
  in-family judge, so across the four judges the bias rotates and averages
  out comparatively. The summarizer is different: it authors the literal
  wording of *every* condition's artifact, since every run — team or
  single-agent — is written up by the same fixed summarizer before judging.
  If the summarizer's family also judges, that judge isn't in-family for one
  condition out of four, it's in-family for 100% of what it scores, in the
  same direction every time. That's a constant bias, not a rotating,
  averaging-out one, and it would also quietly undermine the human-validation
  step (a stylistically-biased judge should correlate worse with human
  labels, which is exactly the check meant to catch this).
  **Resolved:** judge panel is the 3 team families, no trimming. Report the
  plain mean across the 3 judges alongside the standard deviation, so
  disagreement between judges is visible rather than smoothed over by a trim
  that wasn't really averaging anything with only 3 inputs anyway. A high SD
  on a given run is itself a signal worth reporting (low inter-judge
  agreement), not just noise to average away.

---

## Cross-cutting: early termination

Applies to every condition (team, any strategy, and the single-agent
baseline), not just Q1.

- **Mechanism (decided in principle).** Any run can end before its turn
  budget is exhausted via consensus voting: any agent may propose closing the
  meeting (e.g. "That seals it, I think — if there are no further remarks, we
  can close the meeting"), and the run ends only if every agent votes yes.
- **Implementation open.** Logged as TODO.md open question 3: how votes get
  collected (an explicit extra call per agent, similar to strategy-2 bidding,
  vs. reading agreement off each agent's next ordinary turn), what counts as
  a "no" (explicit objection vs. silence), whether a failed vote costs a turn
  from the budget, and how the mechanism degrades cleanly to a single
  participant for the Q1 baseline.
