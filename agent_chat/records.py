"""
Structured run records — the data every downstream stage reads.

One conversation produces one `RunRecord`, written as one JSON file. Metrics,
judging and grid analysis all read that file; the terminal transcript in
`logs/` is presentation only.

The invariant: anything that could change a conversation's outcome is either in
`config` or covered by `file_hashes`, and `run_id` is a hash of both. Two runs
with the same `run_id` were produced by the same resolved setup.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 2

# What a CallRecord.kind can be. The conversation/overhead split in `totals`
# depends on this: turns are the conversation, selector calls are what a
# turn-taking strategy costs on top of it.
KIND_TURN = "turn"
KIND_SELECTOR = "selector"
KIND_SUMMARY = "summary"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    available: bool = True  # False when the provider returned no usage object


@dataclass
class CallRecord:
    """One LLM call. Conversation turns, selector bids and the summary all use this."""
    agent: str
    kind: str
    provider: str
    model_requested: str
    model_resolved: str | None = None
    temperature: float | None = None
    seed: int | None = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    latency_s: float = 0.0
    retries: int = 0
    stop_reason: str | None = None
    error: str | None = None


@dataclass
class TurnRecord:
    index: int
    speaker: str
    content: str
    call: CallRecord
    # Set only when the speaker-tag sanitiser changed something, so its effect
    # stays auditable instead of silently rewriting the transcript.
    content_raw: str | None = None
    # Why the selector picked this speaker (bids, agenda state, ...). Populated
    # by strategies via SelectorLog; None for selectors that don't explain.
    selector: dict | None = None


@dataclass
class SelectorLog:
    """
    Sink a TurnSelector writes to, drained by `Conversation` after each turn.

    A `TurnSelector` is just `(history, agents) -> name` and that stays true —
    a strategy that wants to record *why* it chose someone closes over one of
    these instead of changing the signature.
    """
    _pending: dict | None = None
    _calls: list[CallRecord] = field(default_factory=list)

    def note(self, **fields) -> None:
        """
        Record the rationale for the choice about to be made.

        Merges into any pending fields rather than replacing them, so a
        consensus-stop check and a turn selector can both call this within the
        same turn without one clobbering the other's rationale.
        """
        self._pending = {**(self._pending or {}), **fields}

    def add_call(self, record: CallRecord) -> None:
        """Record an LLM call the selector made (a bid, a think step)."""
        self._calls.append(record)

    def drain(self) -> dict | None:
        pending, self._pending = self._pending, None
        return pending

    def take_calls(self) -> list[CallRecord]:
        calls, self._calls = self._calls, []
        return calls


@dataclass
class RunRecord:
    run_id: str
    config: dict
    file_hashes: dict[str, str]
    schema_version: int = SCHEMA_VERSION
    git_sha: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    selector_calls: list[CallRecord] = field(default_factory=list)
    # The final consensus vote round: {"stopped": bool, "votes": [...]}, or
    # None when consensus_stop was disabled or never reached agent turn 1.
    consensus: dict | None = None
    summary: str | None = None
    summary_call: CallRecord | None = None
    totals: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, directory: str | Path) -> Path:
        """Write to <directory>/<run_id>.json atomically."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        os.replace(tmp, path)  # atomic: a reader never sees a half-written record
        return path


# --- Totals ---

def _sum_usage(calls: list[CallRecord]) -> dict:
    known = [c.cost_usd for c in calls if c.cost_usd is not None]
    if not calls:
        cost: float | None = 0.0          # no calls genuinely cost nothing
    elif known:
        cost = round(sum(known), 6)
    else:
        cost = None                        # calls happened, but no price for them
    return {
        "calls": len(calls),
        "input_tokens": sum(c.usage.input_tokens for c in calls),
        "output_tokens": sum(c.usage.output_tokens for c in calls),
        "cache_read_input_tokens": sum(c.usage.cache_read_input_tokens for c in calls),
        "cache_creation_input_tokens": sum(c.usage.cache_creation_input_tokens for c in calls),
        "latency_s": round(sum(c.latency_s for c in calls), 3),
        "retries": sum(c.retries for c in calls),
        "cost_usd": cost,
        # False when at least one model has no price in pricing.py — so a
        # partial total is never mistaken for a complete one.
        "cost_complete": len(known) == len(calls),
    }


def compute_totals(
    turns: list[TurnRecord],
    selector_calls: list[CallRecord],
    summary_call: CallRecord | None,
    wall_clock_s: float,
    failed_calls: list[CallRecord] | None = None,
) -> dict:
    """
    Split spend into conversation / selector overhead / summary / failed.

    EXPERIMENT_DESIGN §2 requires reporting these separately: a strategy that
    burns extra tokens on bidding must not have that leak into the comparison.
    Failed calls are counted too — an aborted run still spent money.
    """
    turn_calls = [t.call for t in turns]
    summary_calls = [summary_call] if summary_call else []
    failed = failed_calls or []
    return {
        "turns": len(turns),
        "wall_clock_s": round(wall_clock_s, 3),
        "conversation": _sum_usage(turn_calls),
        "selector_overhead": _sum_usage(selector_calls),
        "summary": _sum_usage(summary_calls),
        "failed": _sum_usage(failed),
        "all": _sum_usage(turn_calls + selector_calls + summary_calls + failed),
        "participation_turns": {
            name: sum(1 for t in turns if t.speaker == name)
            for name in sorted({t.speaker for t in turns})
        },
    }


# --- Provenance ---

def canonical_json(payload) -> str:
    """Byte-stable JSON, so a hash of it is reproducible."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_run_id(payload: dict, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:length]


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None
