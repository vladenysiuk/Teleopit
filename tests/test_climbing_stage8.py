"""Stage 8 checks for climbing PPO smoke integration."""

from __future__ import annotations

import pytest

from train_mimic.app import SUPPORTED_TASKS
from train_mimic.tasks.climbing.config.constants import CLIMBING_TASK_ID
from train_mimic.tasks.climbing.config.rl import make_general_climbing_ppo_runner_cfg
from train_mimic.tasks.climbing.config.smoke import (
    CI_SMOKE_MAX_ITERATIONS,
    CI_SMOKE_NUM_ENVS,
    SMOKE_MAX_ITERATIONS,
    SMOKE_NUM_ENVS,
    make_climbing_smoke_env_cfg,
    make_climbing_smoke_ppo_runner_cfg,
)
from train_mimic.tasks.climbing.ppo_smoke import diagnose_reward_scale, run_ppo_smoke
from train_mimic.tasks.climbing.rl.runner import ClimbingOnPolicyRunner


def test_climbing_task_is_supported_for_training() -> None:
    assert CLIMBING_TASK_ID in SUPPORTED_TASKS


def test_smoke_env_uses_pinned_easy_ladder() -> None:
    pytest.importorskip("mjlab")
    cfg = make_climbing_smoke_env_cfg(num_envs=2, seed=7)
    assert cfg.scene.num_envs == 2
    assert cfg.episode_length_s == 10.0
    assert cfg.decimation == 4
    ladder_events = cfg.events["reset_ladder"]
    ladder_cfg = ladder_events.params["ladder_cfg"]
    assert ladder_cfg.min_active_rungs == ladder_cfg.max_active_rungs == 6
    assert ladder_cfg.ladder_yaw_range == (0.0, 0.0)
    assert ladder_cfg.ladder_tilt_range == (0.0, 0.0)


def test_smoke_ppo_runner_cfg_matches_stage8_plan() -> None:
    smoke = make_climbing_smoke_ppo_runner_cfg()
    prod = make_general_climbing_ppo_runner_cfg()
    assert smoke.max_iterations == SMOKE_MAX_ITERATIONS
    assert smoke.save_interval == 50
    assert smoke.actor.class_name == prod.actor.class_name
    assert smoke.obs_groups == prod.obs_groups


def test_climbing_registry_uses_climbing_runner() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_runner_cls

    assert load_runner_cls(CLIMBING_TASK_ID) is ClimbingOnPolicyRunner


def test_reward_scale_diagnosis_flags_invalid_latch_dominance() -> None:
    diagnosis = diagnose_reward_scale(
        {
            "upward_progress": 0.4,
            "invalid_latch": -15.0,
            "time_penalty": -0.05,
        }
    )
    assert diagnosis.dominant_term == "invalid_latch"
    assert "invalid_latch" in diagnosis.flagged_terms
    assert diagnosis.notes


def test_ppo_smoke_run_is_finite_and_checkpoints() -> None:
    pytest.importorskip("mjlab")
    report = run_ppo_smoke(
        num_envs=CI_SMOKE_NUM_ENVS,
        max_iterations=CI_SMOKE_MAX_ITERATIONS,
        save_interval=2,
        seed=42,
    )
    assert report.losses_finite
    assert report.nonfinite_reward_steps == 0
    assert report.completed_episodes > 0
    assert report.reward_variance >= 0.0
    assert report.latch_action_std >= 0.0
    assert report.checkpoint_path is not None
    assert report.playback_ok
    assert report.iterations == CI_SMOKE_MAX_ITERATIONS
    assert report.num_envs == CI_SMOKE_NUM_ENVS


def test_smoke_defaults_match_owner_plan() -> None:
    assert SMOKE_NUM_ENVS == 64
    assert SMOKE_MAX_ITERATIONS == 100
