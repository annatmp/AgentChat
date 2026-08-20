"""
Grid runner for an experiment definition (experiments/*.yaml).

    uv run run_experiment.py experiments/pilot_r2.yaml
    uv run run_experiment.py experiments/pilot_r2.yaml --dry-run
    uv run run_experiment.py experiments/pilot_r2.yaml --force

Walks the (panel x strategy x seed) grid an experiment config expands to
(agent_chat/experiment.py), running each cell as a separate `uv run main.py`
subprocess so a crash in one conversation can't take down the batch. Every
cell is content-addressed via its own run_id exactly like a standalone
`main.py` run, so re-running this after an interruption or a crash just
skips whatever already completed cleanly (agent_chat.records.is_successful_run)
and redoes the rest — that's the whole resume mechanism, no extra state.

Two outputs, per experiment:
  runs/<experiment_id>/<run_id>.json            one per cell (main.py writes these)
  runs/<experiment_id>/experiment_summary.json  the rollup this script writes
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agent_chat.config import ConfigError, ResolvedRun, apply_overrides, load_run_config, resolve
from agent_chat.experiment import Cell, ExperimentConfig, expand_grid, load_experiment_config
from agent_chat.records import file_sha256, is_successful_run

DEFAULT_EXPERIMENT = "experiments/pilot_r2.yaml"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run every cell of an experiment's grid.")
    parser.add_argument("experiment", nargs="?", default=DEFAULT_EXPERIMENT,
                        help=f"path to an experiment config YAML (default: {DEFAULT_EXPERIMENT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate every cell and print the grid preview without running anything")
    parser.add_argument("--force", action="store_true",
                        help="re-run every cell even if it already completed cleanly")
    return parser.parse_args(argv)


def _resolve_cell(cell: Cell, experiment_id: str) -> ResolvedRun:
    """Pre-resolve one cell in-process — pure, no API calls. Raises on a bad cell."""
    config = load_run_config(cell.panel_config)
    config = apply_overrides(config, strategy=cell.strategy, seed=cell.seed, experiment_id=experiment_id)
    return resolve(config)


def _validate_grid(cells: list[Cell], experiment_id: str) -> tuple[list[tuple[Cell, ResolvedRun]], list[str]]:
    """
    Resolve every cell up front so a bad config anywhere in the grid is caught
    before any subprocess spawns — a 135-cell grid must not discover a typo'd
    provider on cell 87, the same fail-fast bar main.py holds one config to.
    """
    resolved: list[tuple[Cell, ResolvedRun]] = []
    problems: list[str] = []
    for cell in cells:
        try:
            resolved.append((cell, _resolve_cell(cell, experiment_id)))
        except (ConfigError, FileNotFoundError, ValueError) as exc:
            strategy = cell.strategy or "(default)"
            problems.append(
                f"{cell.tier}/{Path(cell.panel_config).stem} strategy={strategy} seed={cell.seed}: {exc}"
            )
    return resolved, problems


def _cell_status(run: ResolvedRun) -> str:
    output = run.output_path()
    if not output.exists():
        return "pending"
    return "done" if is_successful_run(output) else "retry"


def _print_preview(resolved: list[tuple[Cell, ResolvedRun]], problems: list[str], experiment_id: str) -> None:
    print(f"experiment  {experiment_id}")
    print(f"cells       {len(resolved) + len(problems)} total")
    counts = {"done": 0, "retry": 0, "pending": 0}
    for _, run in resolved:
        counts[_cell_status(run)] += 1
    to_run = counts["pending"] + counts["retry"]
    summary_line = f"            {counts['done']} already done, {to_run} to run"
    if counts["retry"]:
        summary_line += f" ({counts['retry']} retrying a prior failure)"
    if problems:
        summary_line += f", {len(problems)} invalid"
    print(summary_line + "\n")
    for cell, run in resolved:
        strategy = cell.strategy or "(default)"
        print(f"  [{_cell_status(run):7}] {cell.tier:12} {Path(cell.panel_config).stem:28} "
              f"{strategy:18} seed={cell.seed}  {run.run_id}")
    if problems:
        print("\ninvalid cells:")
        for problem in problems:
            print(f"  - {problem}")


def _execute_cell(cell: Cell, experiment_id: str, force: bool) -> None:
    cmd = ["uv", "run", "main.py", cell.panel_config,
           "--experiment-id", experiment_id, "--seed", str(cell.seed)]
    if cell.strategy:
        cmd += ["--strategy", cell.strategy]
    if force:
        cmd += ["--force"]
    subprocess.run(cmd, check=False)  # nonzero exit is expected/recorded data, not a harness bug


def _cell_row(cell: Cell, run: ResolvedRun, ran: bool) -> dict:
    output = run.output_path()
    if not output.exists():
        return {
            "tier": cell.tier, "panel": cell.panel_config, "strategy": cell.strategy, "seed": cell.seed,
            "run_id": run.run_id, "status": "missing", "cost_usd": None, "cost_complete": False,
            "turns_used": None, "turn_budget": run.config.turn_budget, "wall_clock_s": None,
            "errors": ["no output record was written"],
        }
    data = json.loads(output.read_text())
    errors = data.get("errors") or []
    totals = data.get("totals") or {}
    all_totals = totals.get("all") or {}
    return {
        "tier": cell.tier, "panel": cell.panel_config, "strategy": cell.strategy, "seed": cell.seed,
        "run_id": run.run_id, "status": "failed" if errors else ("done" if ran else "skipped"),
        "cost_usd": all_totals.get("cost_usd"), "cost_complete": all_totals.get("cost_complete", False),
        "turns_used": totals.get("turns"), "turn_budget": run.config.turn_budget,
        "wall_clock_s": totals.get("wall_clock_s"), "errors": errors,
    }


def _write_summary(
    experiment_path: Path, experiment_id: str, rows: list[dict], started_at: str, wall_clock_s: float,
) -> Path:
    known_costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    summary = {
        "experiment_id": experiment_id,
        "experiment_config": str(experiment_path),
        "experiment_config_sha256": file_sha256(experiment_path),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_s": round(wall_clock_s, 3),
        "cells": rows,
        "totals": {
            "cells": len(rows),
            "done": sum(1 for r in rows if r["status"] == "done"),
            "skipped": sum(1 for r in rows if r["status"] == "skipped"),
            "failed": sum(1 for r in rows if r["status"] in ("failed", "missing")),
            "cost_usd": round(sum(known_costs), 6) if known_costs else (0.0 if not rows else None),
            "cost_complete": len(known_costs) == len(rows),
        },
    }
    out_dir = Path("runs") / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "experiment_summary.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp, path)  # atomic, same pattern as RunRecord.write
    return path


def main(argv=None) -> int:
    args = _parse_args(argv)
    experiment_path = Path(args.experiment)

    try:
        experiment: ExperimentConfig = load_experiment_config(experiment_path)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"experiment config error: {exc}", file=sys.stderr)
        return 2

    experiment_id = experiment.name
    cells = expand_grid(experiment)
    resolved, problems = _validate_grid(cells, experiment_id)
    _print_preview(resolved, problems, experiment_id)

    if problems:
        print(f"\n{len(problems)} cell(s) failed to resolve — fix these before running anything.",
              file=sys.stderr)
        return 2

    if args.dry_run:
        return 0

    rows: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    try:
        for cell, run in resolved:
            output = run.output_path()
            if output.exists() and is_successful_run(output) and not args.force:
                rows.append(_cell_row(cell, run, ran=False))
                continue
            strategy = cell.strategy or "(default)"
            print(f"\n=== {cell.tier}/{Path(cell.panel_config).stem} "
                  f"strategy={strategy} seed={cell.seed} ===", flush=True)
            _execute_cell(cell, experiment_id, args.force)
            rows.append(_cell_row(cell, run, ran=True))
    finally:
        summary_path = _write_summary(experiment_path, experiment_id, rows, started_at,
                                       time.perf_counter() - clock)
        print(f"\nexperiment summary  {summary_path}")

    failed = [r for r in rows if r["status"] in ("failed", "missing")]
    print(f"{len(rows)}/{len(resolved)} cells processed this run, {len(failed)} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
