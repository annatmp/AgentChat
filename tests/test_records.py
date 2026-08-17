import json

from agent_chat.records import (
    CallRecord,
    SelectorLog,
    TurnRecord,
    Usage,
    canonical_json,
    compute_run_id,
    compute_totals,
)


def _call(kind, model="claude-sonnet-4-6", **usage):
    return CallRecord(
        agent="a", kind=kind, provider="anthropic", model_requested=model,
        usage=Usage(**usage), cost_usd=0.5, latency_s=1.0,
    )


def test_run_id_is_stable_across_key_order():
    a = {"seed": 0, "roster": ["x", "y"], "temperature": 0.7}
    b = {"temperature": 0.7, "roster": ["x", "y"], "seed": 0}
    assert compute_run_id(a) == compute_run_id(b)


def test_run_id_changes_when_any_input_changes():
    base = {"seed": 0, "file_hashes": {"prompts/system_prompt.txt": "abc"}}
    changed_seed = {"seed": 1, "file_hashes": {"prompts/system_prompt.txt": "abc"}}
    changed_hash = {"seed": 0, "file_hashes": {"prompts/system_prompt.txt": "abd"}}
    assert compute_run_id(base) != compute_run_id(changed_seed)
    assert compute_run_id(base) != compute_run_id(changed_hash)


def test_canonical_json_is_byte_stable():
    assert canonical_json({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'
    assert json.loads(canonical_json({"a": 1})) == {"a": 1}


def test_totals_split_conversation_from_selector_overhead():
    turns = [
        TurnRecord(index=0, speaker="a", content="hi",
                   call=_call("turn", input_tokens=100, output_tokens=10)),
        TurnRecord(index=1, speaker="b", content="ho",
                   call=_call("turn", input_tokens=200, output_tokens=20)),
    ]
    selector = [_call("selector", input_tokens=50, output_tokens=5)]
    summary = _call("summary", input_tokens=400, output_tokens=40)

    totals = compute_totals(turns, selector, summary, wall_clock_s=12.5)

    assert totals["turns"] == 2
    assert totals["conversation"]["input_tokens"] == 300
    assert totals["selector_overhead"]["input_tokens"] == 50
    assert totals["summary"]["output_tokens"] == 40
    assert totals["all"]["input_tokens"] == 750
    assert totals["all"]["calls"] == 4
    assert totals["participation_turns"] == {"a": 1, "b": 1}


def test_unpriced_models_do_not_report_a_complete_cost():
    priced = _call("turn", input_tokens=10)
    unpriced = _call("turn", input_tokens=10)
    unpriced.cost_usd = None  # no entry in pricing.py for this deployment

    turns = [TurnRecord(index=0, speaker="a", content="x", call=priced),
             TurnRecord(index=1, speaker="b", content="y", call=unpriced)]
    totals = compute_totals(turns, [], None, wall_clock_s=1.0)

    assert totals["conversation"]["cost_complete"] is False
    assert totals["conversation"]["cost_usd"] == 0.5   # the known half, not a fake zero


def test_selector_log_note_merges_across_calls_within_one_turn():
    # A consensus-stop check and a turn selector can both call .note() before
    # the same drain; neither should clobber the other's rationale.
    log = SelectorLog()
    log.note(strategy="round_robin", position=0)
    log.note(consensus_stop=False)
    assert log.drain() == {"strategy": "round_robin", "position": 0, "consensus_stop": False}
