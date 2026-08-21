"""
Entry point: run one conversation from a run config and record it.

    uv run main.py                          # configs/baseline.yaml
    uv run main.py configs/baseline.yaml
    uv run main.py configs/baseline.yaml --dry-run   # resolve only, no API calls

Two outputs, with different jobs:
  runs/<run_id>.json  the data — everything downstream reads this
  logs/*.log          the transcript as it appeared in the terminal
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agent_chat import strategies
from agent_chat.config import ConfigError, ResolvedRun, apply_overrides, load_run_config, resolve
from agent_chat.conversation import Conversation
from agent_chat.model_check import check_models
from agent_chat.policies import (
    ConsensusOutcome,
    SummaryOutcome,
    consensus_stop,
    max_turns,
    review_round_context,
    stop_when_any,
    summarize,
)
from agent_chat.records import RunRecord, SelectorLog, compute_totals, git_sha, is_successful_run

DEFAULT_CONFIG = "configs/baseline.yaml"


def _setup_tracing(experiment_id: str | None) -> None:
    """
    Optional: trace every LLM call (full prompt, response, latency, tokens) to a
    local Phoenix instance for inspection — `uvx arize-phoenix serve`, then set
    PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 in .env. Purely observational:
    never affects run_id or a run's output. Silent no-op unless both the env var is
    set and `uv sync --group tracing` has been run.

    `experiment_id`, if given (from `--experiment-id`), scopes the Phoenix
    project to `agent-chat-<experiment_id>` instead of the plain `agent-chat`
    project every ad-hoc run shares — so a harness-driven batch of runs is
    browsable as one group in the Phoenix UI, distinct from other batches.

    Two things are done explicitly here rather than relying on the library's own
    "convenient" defaults, both verified live against arize-phoenix 15.1.0 after
    each silently exported nothing instead of raising:
      - `endpoint` must include the /v1/traces path. register()'s own env-var
        pickup (i.e. leaving `endpoint` unset) resolves to an unrelated OTel SDK
        default (localhost:4317/gRPC) instead of this value; passing the bare host
        without the path 405s on every export.
      - `auto_instrument=True` did not activate the openai/anthropic instrumentors
        in this environment — instrumenting both classes explicitly is what
        actually put spans in Phoenix when this was tested against a live run.
    """
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from phoenix.otel import register

        if not endpoint.rstrip("/").endswith("/v1/traces"):
            endpoint = endpoint.rstrip("/") + "/v1/traces"
        project_name = f"agent-chat-{experiment_id}" if experiment_id else "agent-chat"
        tracer_provider = register(project_name=project_name, endpoint=endpoint)
        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    except ImportError:
        print(
            "PHOENIX_COLLECTOR_ENDPOINT is set but tracing packages aren't installed — "
            "run `uv sync --group tracing` to enable tracing, or unset it to skip.",
            file=sys.stderr,
        )


class _Tee:
    """Write to both the real stdout and a log file."""
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one refinement meeting.")
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG,
                        help=f"path to a run config YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve the config, hash inputs and print the run_id "
                             "without making any API calls")
    parser.add_argument("--check-models", action="store_true",
                        help="call each provider's models.list() to confirm every "
                             "configured model exists, then exit — no generation "
                             "tokens spent")
    parser.add_argument("--force", action="store_true",
                        help="run even if a record for this run_id already exists")
    parser.add_argument("--strategy", default=None,
                        help="override the config's strategy.name (e.g. for a grid cell)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the config's seed (e.g. for a grid replicate)")
    parser.add_argument("--experiment-id", default=None,
                        help="scope output_dir to runs/<id>, logs to logs/<id>, and the "
                             "Phoenix project to agent-chat-<id> — set by run_experiment.py, "
                             "rarely needed by hand")
    return parser.parse_args(argv)


def _print_config(run: ResolvedRun) -> None:
    """
    Config header for the terminal transcript.

    The `name: model=... provider=... max_tokens=...` lines are the format
    judge.ipynb parses out of the log files; keep them until the judge reads
    run records instead.
    """
    print("--- CONFIG ---")
    print(f"run_id={run.run_id} config={run.config.name} "
          f"strategy={run.config.strategy.name} turn_budget={run.config.turn_budget} "
          f"temperature={run.config.temperature} seed={run.config.seed} "
          f"role_knowledge={run.config.role_knowledge}")
    for agent in run.agents.values():
        print(f"{agent.name}: model={agent.model} provider={agent.provider} "
              f"max_tokens={agent.max_tokens}")
    print(f"summarizer: model={run.summarizer.model} provider={run.summarizer.provider} "
          f"max_tokens={run.summarizer.max_tokens}")
    print("--- END CONFIG ---\n")


def _print_dry_run(run: ResolvedRun) -> None:
    output = run.output_path()
    if output.exists():
        marker = "  (exists, clean — would skip)" if is_successful_run(output) else \
            "  (exists but recorded errors — would re-run)"
    else:
        marker = ""
    print(f"run_id      {run.run_id}")
    print(f"config      {run.config.name} ({run.config.strategy.name}, "
          f"{run.config.turn_budget} turns, seed {run.config.seed})")
    print(f"output      {output}{marker}")
    print(f"roster      {', '.join(run.agents)}")
    print(f"knowledge   {', '.join(run.knowledge) or '(none loaded)'}")
    print(f"summarizer  {run.summarizer.model} ({run.summarizer.provider})")
    print(f"git_sha     {git_sha()}")
    print("\nhashed inputs:")
    for path, digest in sorted(run.file_hashes.items()):
        print(f"  {digest[:12]}  {path}")
    print("\nresolved config:")
    print(json.dumps(run.config_dict(), indent=2, sort_keys=True))


def _print_model_check(run: ResolvedRun) -> bool:
    """Print models.list() results for every configured model. Returns True iff none are missing."""
    results = check_models(run.agents, run.summarizer)
    ok = True
    for r in results:
        if r.status == "ok":
            print(f"  {r.provider:12} {r.model:30} OK")
        elif r.status == "not_found":
            print(f"  {r.provider:12} {r.model:30} NOT FOUND")
            ok = False
        else:
            print(f"  {r.provider:12} {r.model:30} unverified ({r.detail})")
    return ok


def _execute(run: ResolvedRun) -> RunRecord:
    _print_config(run)

    selector_log = SelectorLog()
    conversation = Conversation(
        agents=run.agents,
        system_prompt=run.system_prompt,
        knowledge=run.knowledge,
        temperature=run.config.temperature,
        seed=run.config.seed,
    )
    conversation.user(run.task_prompt)

    turn_selector = strategies.build(
        run.config.strategy.name,
        run.config.strategy.params,
        roster=run.agents,
        rng=run.rng,
        log=selector_log,
        knowledge=run.knowledge,
        system_prompt=run.system_prompt,
        turn_budget=run.config.turn_budget,
    )
    outcome = SummaryOutcome()
    consensus_outcome = ConsensusOutcome()

    # Every condition gets a shared turn-budget ceiling; consensus_stop layers
    # an early exit on top of it when the run config enables it.
    stop_conditions = [max_turns(run.config.turn_budget)]
    if run.config.consensus_stop:
        stop_conditions.append(consensus_stop(
            run.agents,
            vote_prompt_file=run.config.consensus_prompt,
            knowledge=run.knowledge,
            system_prompt=run.system_prompt,
            log=selector_log,
            outcome=consensus_outcome,
        ))

    # Solo baseline only: "REVIEW ROUND N" framing injected before each turn.
    turn_context = (
        review_round_context(run.config.review_round_template, run.config.review_round_first_template)
        if run.config.review_rounds else None
    )

    started_at = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    aborted: str | None = None
    try:
        conversation.run(
            turn_selector=turn_selector,
            stop_condition=stop_when_any(*stop_conditions),
            stream=True,
            post_processors=[summarize(
                run.summarizer, outcome, stream=True,
                prompt_file=run.config.summarize_prompt,
            )],
            selector_log=selector_log,
            turn_context=turn_context,
        )
    except Exception as exc:
        # Recorded rather than retried to success: robustness differences
        # between strategies are a finding, not noise to hide.
        aborted = f"aborted after {len(conversation.turns)} turns: {type(exc).__name__}: {exc}"
        print(f"\n{aborted}", flush=True)
    wall_clock = time.perf_counter() - clock

    if consensus_outcome.stopped:
        print(f"\nmeeting ended early by consensus after {len(conversation.turns)} turns "
              f"(turn budget was {run.config.turn_budget})", flush=True)
    elif consensus_outcome.votes is not None:
        print(f"\nmeeting ran to its turn budget ({run.config.turn_budget} turns); "
              f"last consensus check did not reach unanimity", flush=True)
    if consensus_outcome.votes is not None:
        for vote in consensus_outcome.votes:
            marker = "STOP" if vote["stop"] else "continue"
            print(f"  - {vote['agent']}: {marker} — {vote['reason']}", flush=True)

    errors = list(conversation.errors)
    if aborted:
        errors.append(aborted)
    if outcome.error:
        errors.append(f"summary: {outcome.error}")

    return RunRecord(
        run_id=run.run_id,
        config=run.config_dict(),
        file_hashes=run.file_hashes,
        git_sha=git_sha(),
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        turns=conversation.turns,
        selector_calls=conversation.selector_calls,
        consensus={"stopped": consensus_outcome.stopped, "votes": consensus_outcome.votes}
        if consensus_outcome.votes is not None else None,
        summary=outcome.text,
        summary_call=outcome.call,
        totals=compute_totals(
            conversation.turns, conversation.selector_calls,
            outcome.call if outcome.text else None, wall_clock,
            failed_calls=conversation.failed_calls,
        ),
        errors=errors,
    )


def main(argv=None) -> int:
    args = _parse_args(argv)
    _setup_tracing(args.experiment_id)

    try:
        config = load_run_config(args.config)
        config = apply_overrides(
            config, strategy=args.strategy, seed=args.seed, experiment_id=args.experiment_id,
        )
        run = resolve(config)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.check_models:
        ok = _print_model_check(run)
        return 0 if ok else 1

    if args.dry_run:
        _print_dry_run(run)
        return 0

    output = run.output_path()
    if output.exists() and is_successful_run(output) and not args.force:
        print(f"{output} already exists and completed cleanly — skipping. Use --force to redo it.")
        return 0
    if output.exists() and not is_successful_run(output) and not args.force:
        print(f"{output} exists but recorded errors from a previous attempt — re-running.")

    log_dir = Path("logs") / args.experiment_id if args.experiment_id else Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keeps the `conversation_*` prefix judge.ipynb globs for, until the judge
    # reads run records instead (P2).
    log_path = str(log_dir / f"conversation_{stamp}_{run.config.name}_{run.run_id}.log")
    real_stdout = sys.stdout
    with open(log_path, "w") as log_file:
        sys.stdout = _Tee(log_file)
        try:
            record = _execute(run)
        finally:
            sys.stdout = real_stdout

    path = record.write(run.config.output_dir)
    totals = record.totals["all"]
    print(f"\nrun record  {path}")
    print(f"transcript  {log_path}")
    print(f"tokens      {totals['input_tokens']} in / {totals['output_tokens']} out "
          f"across {totals['calls']} calls"
          f"{'' if totals['cost_complete'] else ' (cost incomplete — see pricing.py)'}")
    if totals["cost_usd"] is not None:
        print(f"cost        ${totals['cost_usd']:.4f}")
    if record.errors:
        print("errors:")
        for message in record.errors:
            print(f"  - {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
