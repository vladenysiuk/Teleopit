"""Stage 9 checks for curriculum hooks and easy learnability experiment."""

from __future__ import annotations

import pytest
import torch

from train_mimic.tasks.climbing.config.curriculum import (
    CurriculumConfig,
    curriculum_ladder_cfg,
    easy_curriculum_cfg,
    with_curriculum_axis,
)
from train_mimic.tasks.climbing.config.easy import (
    CI_EASY_MAX_ITERATIONS,
    CI_EASY_NUM_ENVS,
    EASY_EPISODE_LENGTH_S,
    EASY_NUM_ENVS,
    make_climbing_easy_env_cfg,
    make_climbing_easy_ppo_runner_cfg,
)
from train_mimic.tasks.climbing.debug_mdp import observations_finite
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.learnability import (
    compare_learnability,
    run_baseline_comparison,
    run_learnability_experiment,
)


def test_easy_curriculum_disables_randomization_axes() -> None:
    cfg = easy_curriculum_cfg()
    assert cfg.initial_hand_attach_prob == 1.0
    assert not cfg.randomize_rung_count
    assert not cfg.randomize_spacing
    assert not cfg.randomize_ladder_pose
    assert not cfg.randomize_rung_physics
    assert not cfg.randomize_initial_pose
    assert cfg.use_hold_pose_reset


def test_curriculum_ladder_cfg_fixed_when_axes_disabled() -> None:
    curriculum = easy_curriculum_cfg()
    ladder = curriculum_ladder_cfg(curriculum)
    assert ladder.min_active_rungs == ladder.max_active_rungs == 6
    assert ladder.spacing_min == ladder.spacing_max == 0.25
    assert ladder.ladder_yaw_range == (0.0, 0.0)
    assert ladder.rung_radius == pytest.approx(0.035)


def test_with_curriculum_axis_enables_single_knob() -> None:
    from dataclasses import replace

    curriculum = replace(
        with_curriculum_axis(
            easy_curriculum_cfg(),
            randomize_spacing=True,
            initial_hand_attach_prob=0.5,
        ),
        spacing_max=0.28,
    )
    assert curriculum.randomize_spacing
    assert curriculum.initial_hand_attach_prob == 0.5
    ladder = curriculum_ladder_cfg(curriculum)
    assert ladder.spacing_max > ladder.spacing_min


def test_easy_env_wires_curriculum_reset_events() -> None:
    pytest.importorskip("mjlab")
    cfg = make_climbing_easy_env_cfg(num_envs=2, seed=7)
    assert cfg.episode_length_s == EASY_EPISODE_LENGTH_S
    assert "reset_climb_curriculum_pose" in cfg.events
    assert "reset_climb_curriculum_physics" in cfg.events
    ladder_cfg = cfg.events["reset_ladder"].params["ladder_cfg"]
    assert ladder_cfg.min_active_rungs == ladder_cfg.max_active_rungs == 6


def test_easy_env_reset_applies_hold_pose_and_hand_latches() -> None:
    pytest.importorskip("mjlab")
    pytest.importorskip("mink")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg=make_climbing_easy_env_cfg(num_envs=1, seed=42), device="cpu")
    try:
        env.reset()
        assert observations_finite(env)
        latch = ClimbLatchState.get(env)
        assert bool(latch.state.attached[0].all().item())
        assert int(latch.state.rung_id[0, 0].item()) >= 0
    finally:
        env.close()


def test_easy_ppo_runner_cfg_differs_from_smoke_length() -> None:
    easy = make_climbing_easy_ppo_runner_cfg()
    assert easy.max_iterations > 100
    assert easy.save_interval == 200


def test_compare_learnability_detects_improvement() -> None:
    from train_mimic.tasks.climbing.learnability import EpisodeRolloutMetrics

    zero = EpisodeRolloutMetrics(
        agent="zero",
        steps=100,
        completed_episodes=1,
        max_head_height_l=0.5,
        max_valid_higher_attachments=0,
        max_invalid_latch=10,
        invalid_latch_per_step=0.1,
        torque_saturation_fraction=0.2,
        reward_sum=-1.0,
    )
    random = EpisodeRolloutMetrics(
        agent="random",
        steps=100,
        completed_episodes=1,
        max_head_height_l=0.55,
        max_valid_higher_attachments=0,
        max_invalid_latch=20,
        invalid_latch_per_step=0.2,
        torque_saturation_fraction=0.25,
        reward_sum=-2.0,
    )
    trained = EpisodeRolloutMetrics(
        agent="trained",
        steps=100,
        completed_episodes=1,
        max_head_height_l=0.62,
        max_valid_higher_attachments=1,
        max_invalid_latch=5,
        invalid_latch_per_step=0.05,
        torque_saturation_fraction=0.22,
        reward_sum=0.5,
    )
    cmp = compare_learnability(zero=zero, random=random, trained=trained)
    assert cmp.learning_signal
    assert cmp.head_height_gain_vs_zero > 0.0
    assert cmp.attachment_gain_vs_zero == 1


def test_baseline_comparison_runs_on_easy_env() -> None:
    pytest.importorskip("mjlab")
    pytest.importorskip("mink")
    zero, random = run_baseline_comparison(
        num_envs=CI_EASY_NUM_ENVS,
        seed=42,
        rollout_steps=24,
        device="cpu",
    )
    assert zero.ok
    assert random.ok
    assert zero.agent == "zero"
    assert random.agent == "random"


def test_learnability_experiment_completes() -> None:
    pytest.importorskip("mjlab")
    pytest.importorskip("mink")
    report = run_learnability_experiment(
        num_envs=CI_EASY_NUM_ENVS,
        max_iterations=CI_EASY_MAX_ITERATIONS,
        save_interval=3,
        seed=42,
        rollout_steps=24,
        device="cpu",
    )
    assert report.ok
    assert report.checkpoint_path is not None
    assert report.trained is not None
    assert report.comparison is not None
    assert report.train_iterations == CI_EASY_MAX_ITERATIONS
    assert report.num_envs == CI_EASY_NUM_ENVS


def test_easy_defaults_match_owner_plan() -> None:
    assert EASY_NUM_ENVS == 256


def test_play_climb_easy_and_smoke_flags_are_mutually_exclusive() -> None:
    import sys
    from unittest.mock import patch

    from train_mimic.scripts import play_climb

    with patch.object(
        sys,
        "argv",
        ["play_climb.py", "--checkpoint", "model.pt", "--easy", "--smoke-ladder"],
    ):
        with pytest.raises(SystemExit):
            play_climb.parse_args()


def test_easy_env_play_preset_keeps_curriculum_reset() -> None:
    cfg = make_climbing_easy_env_cfg(num_envs=1, seed=42, play=True)
    assert "reset_climb_curriculum_pose" in cfg.events
    curriculum = cfg.events["reset_climb_curriculum_pose"].params["curriculum_cfg"]
    assert curriculum.initial_hand_attach_prob == 1.0
    assert not cfg.observations["actor_proprio"].enable_corruption


def test_play_climb_video_length_defaults_to_full_easy_episode() -> None:
    pytest.importorskip("mjlab")
    from train_mimic.scripts.play_climb import resolve_play_video_length

    cfg = make_climbing_easy_env_cfg(num_envs=1, seed=42, play=True)
    assert resolve_play_video_length(cfg, None) == 750
    assert resolve_play_video_length(cfg, 120) == 120


def test_play_climb_video_fps_uses_requested_playback_speed() -> None:
    from train_mimic.scripts.play_climb import (
        DEFAULT_VIDEO_CLIPS,
        DEFAULT_VIDEO_SPEED,
        resolve_video_fps,
    )

    # Easy policy rate is 50 Hz; 30% speed → 15 fps.
    assert DEFAULT_VIDEO_SPEED == 0.3
    assert DEFAULT_VIDEO_CLIPS == 8
    assert resolve_video_fps(0.02, 0.3) == 15.0
    assert resolve_video_fps(0.02, 1.0) == 50.0


def test_play_climb_video_cli_defaults() -> None:
    import sys
    from unittest.mock import patch

    from train_mimic.scripts import play_climb

    with patch.object(
        sys,
        "argv",
        ["play_climb.py", "--checkpoint", "model.pt", "--easy", "--video"],
    ):
        args = play_climb.parse_args()
    assert args.video is True
    assert args.video_clips == play_climb.DEFAULT_VIDEO_CLIPS
    assert args.video_speed == play_climb.DEFAULT_VIDEO_SPEED
    assert args.video_width == play_climb.DEFAULT_VIDEO_WIDTH
    assert args.video_height == play_climb.DEFAULT_VIDEO_HEIGHT
    assert args.video_width == 1920
    assert args.video_height == 1080
    assert args.seed == 42
