"""
Regression coverage for run_experiment.py's summary rollup — a cell's own
cost_usd can be a non-None *partial* sum (some calls priced, some not, e.g.
an unpriced model in pricing.py), so aggregate completeness must come from
each cell's own cost_complete flag, not from "cost_usd isn't None", or a
partial sum silently reads as a complete one at the experiment level.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_experiment import _write_summary  # noqa: E402


def _row(status="done", cost_usd=0.1, cost_complete=True):
    return {
        "tier": "homogeneous", "panel": "configs/homogeneous_cheap.yaml", "strategy": "bidding",
        "seed": 0, "run_id": "abc123", "status": status, "cost_usd": cost_usd,
        "cost_complete": cost_complete, "turns_used": 5, "turn_budget": 10,
        "wall_clock_s": 12.0, "errors": [],
    }


def test_summary_cost_complete_false_when_any_cell_has_an_unpriced_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text("name: t\n")

    rows = [_row(cost_complete=True), _row(cost_complete=False, cost_usd=0.05)]
    path = _write_summary(experiment_config, "t", rows, "2026-01-01T00:00:00Z", 1.0)

    summary = json.loads(path.read_text())
    assert summary["totals"]["cost_complete"] is False
    assert summary["totals"]["cost_usd"] == 0.15  # partial sum still reported, just flagged incomplete


def test_summary_cost_complete_true_when_every_cell_is_fully_priced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text("name: t\n")

    rows = [_row(cost_complete=True), _row(cost_complete=True)]
    path = _write_summary(experiment_config, "t", rows, "2026-01-01T00:00:00Z", 1.0)

    assert json.loads(path.read_text())["totals"]["cost_complete"] is True


def test_summary_cost_complete_true_for_an_empty_grid(tmp_path, monkeypatch):
    # all([]) is True — matches _sum_usage's "no calls genuinely cost nothing".
    monkeypatch.chdir(tmp_path)
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text("name: t\n")

    path = _write_summary(experiment_config, "t", [], "2026-01-01T00:00:00Z", 0.0)

    summary = json.loads(path.read_text())
    assert summary["totals"]["cost_complete"] is True
    assert summary["totals"]["cost_usd"] == 0.0
