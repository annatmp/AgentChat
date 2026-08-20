import pytest

from agent_chat.config import ConfigError
from agent_chat.experiment import Cell, ExperimentConfig, TierConfig, expand_grid, load_experiment_config

PILOT = "experiments/pilot_r2.yaml"


def test_pilot_experiment_loads():
    experiment = load_experiment_config(PILOT)
    assert experiment.name == "pilot_sm"
    assert experiment.seeds == [0]
    assert set(experiment.tiers) == {"solo", "homogeneous"}
    assert experiment.tiers["solo"].strategies == []


def test_solo_style_tier_yields_one_cell_per_panel_and_seed_with_no_strategy():
    experiment = ExperimentConfig(
        name="t", seeds=[0, 1],
        tiers={"solo": TierConfig(configs=["configs/solo_anthropic.yaml"], strategies=[])},
    )
    cells = expand_grid(experiment)
    assert cells == [
        Cell("solo", "configs/solo_anthropic.yaml", None, 0),
        Cell("solo", "configs/solo_anthropic.yaml", None, 1),
    ]


def test_strategy_crossed_tier_loops_panel_then_seed_then_strategy():
    experiment = ExperimentConfig(
        name="t", seeds=[0, 1], strategies=["round_robin", "bidding"],
        tiers={"homogeneous": TierConfig(configs=["configs/homogeneous_anthropic.yaml"])},
    )
    cells = expand_grid(experiment)
    # strategy innermost: both strategies for seed 0 before seed 1 starts —
    # TODO.md's "paired runs share a seed/panel block".
    assert cells == [
        Cell("homogeneous", "configs/homogeneous_anthropic.yaml", "round_robin", 0),
        Cell("homogeneous", "configs/homogeneous_anthropic.yaml", "bidding", 0),
        Cell("homogeneous", "configs/homogeneous_anthropic.yaml", "round_robin", 1),
        Cell("homogeneous", "configs/homogeneous_anthropic.yaml", "bidding", 1),
    ]


def test_tier_without_its_own_strategies_falls_back_to_the_experiment_default():
    experiment = ExperimentConfig(
        name="t", seeds=[0], strategies=["facilitator"],
        tiers={"mixed": TierConfig(configs=["configs/mixed_1.yaml"])},  # no tier-level override
    )
    cells = expand_grid(experiment)
    assert cells == [Cell("mixed", "configs/mixed_1.yaml", "facilitator", 0)]


def test_multiple_panels_in_one_tier_are_each_crossed_independently():
    experiment = ExperimentConfig(
        name="t", seeds=[0], strategies=["round_robin"],
        tiers={"homogeneous": TierConfig(
            configs=["configs/homogeneous_anthropic.yaml", "configs/homogeneous_google.yaml"],
        )},
    )
    cells = expand_grid(experiment)
    assert [c.panel_config for c in cells] == [
        "configs/homogeneous_anthropic.yaml", "configs/homogeneous_google.yaml",
    ]


def test_unknown_top_level_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nnonsense: true\ntiers:\n  solo:\n    configs: [configs/solo_anthropic.yaml]\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_experiment_config(bad)


def test_tier_without_configs_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ntiers:\n  solo:\n    strategies: []\n")
    with pytest.raises(ConfigError, match="non-empty 'configs'"):
        load_experiment_config(bad)


def test_experiment_with_no_tiers_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\n")
    with pytest.raises(ConfigError, match="at least one tier"):
        load_experiment_config(bad)


def test_missing_experiment_config_is_rejected():
    with pytest.raises(ConfigError, match="no experiment config"):
        load_experiment_config("experiments/does_not_exist.yaml")
