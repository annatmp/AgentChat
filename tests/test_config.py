"""
The fail-fast paths: a config that cannot be run must be rejected at load time,
not discovered as a 400 partway through a 140-run grid.
"""

import pytest

from agent_chat.config import ConfigError, load, load_run_config, resolve

BASELINE = "configs/baseline.yaml"


def test_baseline_config_resolves():
    run = load(BASELINE)
    assert len(run.agents) == 6
    assert run.summarizer.model == "deepseek-chat"
    assert run.summarizer.name not in run.agents      # neutral by construction
    assert len(run.knowledge) == 6                    # every role has private context
    assert run.run_id and len(run.run_id) == 12


def test_run_id_is_deterministic():
    assert load(BASELINE).run_id == load(BASELINE).run_id


def test_run_id_ignores_relabelling_but_tracks_content():
    """A renamed config is the same experimental cell; a changed seed is not."""
    renamed = load_run_config(BASELINE)
    renamed.name = "something-else"
    renamed.output_dir = "elsewhere"
    assert resolve(renamed).run_id == load(BASELINE).run_id

    reseeded = load_run_config(BASELINE)
    reseeded.seed = 99
    assert resolve(reseeded).run_id != load(BASELINE).run_id


def test_role_knowledge_off_changes_the_run_id():
    config = load_run_config(BASELINE)
    config.role_knowledge = False
    run = resolve(config)
    assert run.knowledge == {}
    assert run.run_id != load(BASELINE).run_id


def test_missing_roster_member_fails_at_load():
    config = load_run_config(BASELINE)
    config.roster = ["product_owner", "does_not_exist"]
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        resolve(config)


def test_empty_roster_is_rejected():
    config = load_run_config(BASELINE)
    config.roster = []
    with pytest.raises(ConfigError, match="roster is empty"):
        resolve(config)


def test_duplicate_roster_entries_are_rejected():
    config = load_run_config(BASELINE)
    config.roster = ["product_owner", "product_owner"]
    with pytest.raises(ConfigError, match="duplicates"):
        resolve(config)


def test_temperature_on_a_model_that_rejects_it_fails_at_load():
    config = load_run_config(BASELINE)
    config.panel = {"architect": {"model": "claude-opus-5"}}
    with pytest.raises(ConfigError, match="rejects `temperature`"):
        resolve(config)


def test_same_model_is_fine_without_a_temperature():
    config = load_run_config(BASELINE)
    config.panel = {"architect": {"model": "claude-opus-5"}}
    config.temperature = None
    resolve(config)  # per-agent temperature is unset in the YAMLs, so this is valid


def test_unknown_provider_is_rejected():
    config = load_run_config(BASELINE)
    config.panel = {"qa_engineer": {"provider": "openai"}}
    with pytest.raises(ConfigError, match="unknown provider"):
        resolve(config)


def test_panel_cannot_name_an_agent_outside_the_roster():
    config = load_run_config(BASELINE)
    config.panel = {"planner": {"model": "claude-haiku-4-5"}}
    with pytest.raises(ConfigError, match="not in the roster"):
        resolve(config)


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ConfigError, match="unknown keys"):
        load_run_config("tests/fixtures/unknown_key.yaml")


def test_unknown_strategy_is_rejected_when_built():
    from agent_chat import strategies
    run = load(BASELINE)
    with pytest.raises(ValueError, match="unknown strategy"):
        strategies.build("urgency_auction", {}, roster=run.agents)


def test_round_robin_order_must_be_in_the_roster():
    from agent_chat import strategies
    run = load(BASELINE)
    with pytest.raises(ValueError, match="outside the roster"):
        strategies.build("round_robin", {"order": ["ghost"]}, roster=run.agents)
