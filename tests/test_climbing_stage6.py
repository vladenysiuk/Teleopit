"""Stage 6 checks for assisted contact-pose validation (assisted_hold)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from train_mimic.tasks.climbing.config.hold_pose import HoldPoseConfig
from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.ladder.hold_ik import solve_hold_pose_ik
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.ladder.state import LadderRuntime


@pytest.fixture
def hold_cfg() -> HoldPoseConfig:
    return HoldPoseConfig(hold_duration_s=2.0)


@pytest.fixture
def fixed_ladder_cfg(hold_cfg: HoldPoseConfig) -> LadderConfig:
    return LadderConfig(
        min_active_rungs=6,
        max_active_rungs=6,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
        rung_radius=hold_cfg.rung_radius_m,
        rung_friction=hold_cfg.rung_friction,
    )


def _make_hold_env(
    seed: int = 42,
    *,
    hold_cfg: HoldPoseConfig | None = None,
    ladder_cfg: LadderConfig | None = None,
):
    pytest.importorskip("mjlab")
    pytest.importorskip("mink")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import make_climbing_hold_pose_env_cfg

    cfg = make_climbing_hold_pose_env_cfg(
        num_envs=1,
        seed=seed,
        hold_cfg=hold_cfg or HoldPoseConfig(hold_duration_s=2.0),
        ladder_cfg=ladder_cfg,
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    return env


def test_hold_pose_ik_finite_and_within_limits(hold_cfg, fixed_ladder_cfg):
    env = _make_hold_env(hold_cfg=hold_cfg, ladder_cfg=fixed_ladder_cfg)
    try:
        solution = solve_hold_pose_ik(env.unwrapped, hold_cfg)
        assert solution.finite
        assert solution.within_joint_limits
        assert np.isfinite(solution.joint_pos).all()
        assert solution.joint_pos.shape == (29,)
        assert float(solution.hand_errors_m.max()) < 0.06
        assert float(solution.foot_errors_m.max()) < 0.08
    finally:
        env.close()


def test_hold_pose_deterministic_for_fixed_seed(hold_cfg, fixed_ladder_cfg):
    env_a = _make_hold_env(seed=42, hold_cfg=hold_cfg, ladder_cfg=fixed_ladder_cfg)
    env_b = _make_hold_env(seed=42, hold_cfg=hold_cfg, ladder_cfg=fixed_ladder_cfg)
    try:
        sol_a = solve_hold_pose_ik(env_a.unwrapped, hold_cfg)
        sol_b = solve_hold_pose_ik(env_b.unwrapped, hold_cfg)
        np.testing.assert_allclose(sol_a.joint_pos, sol_b.joint_pos, atol=1e-5)
        np.testing.assert_allclose(sol_a.root_pos, sol_b.root_pos, atol=1e-5)
    finally:
        env_a.close()
        env_b.close()


def test_hold_pose_init_latches_and_foot_pairs(hold_cfg, fixed_ladder_cfg):
    from train_mimic.tasks.climbing.debug_hold_pose import (
        collect_hold_sample,
        initialize_hold_pose_scene,
    )

    env = _make_hold_env(hold_cfg=hold_cfg, ladder_cfg=fixed_ladder_cfg)
    try:
        solution = initialize_hold_pose_scene(env.unwrapped, hold_cfg)
        latch = ClimbLatchState.get(env.unwrapped)
        assert bool(latch.state.attached[0, 0].item())
        assert bool(latch.state.attached[0, 1].item())
        assert int(latch.state.rung_id[0, 0].item()) == hold_cfg.left_hand_rung_id
        assert int(latch.state.rung_id[0, 1].item()) == hold_cfg.right_hand_rung_id
        assert int(latch.state.site_id[0, 0].item()) == hold_cfg.left_hand_site_id
        assert int(latch.state.site_id[0, 1].item()) == hold_cfg.right_hand_site_id

        sample = collect_hold_sample(env.unwrapped, hold_cfg)
        assert sample["foot_axis_dist_m"].shape == (2,)
        # Sole sites on/above the cylinder surface — never deep inside the rung.
        r = float(hold_cfg.rung_radius_m)
        expected = float(
            np.hypot(hold_cfg.foot_approach_offset_m, r + hold_cfg.foot_clearance_m)
        )
        assert float(sample["foot_axis_dist_m"].min()) > expected - 0.015
        assert float(sample["foot_axis_dist_m"].max()) < expected + 0.025
        assert float(sample["foot_axis_dist_m"].min()) > r
        # Flat plant: ankles must not slam into deep opposing roll.
        assert abs(float(solution.joint_pos[5])) <= hold_cfg.ankle_roll_limit_rad + 1e-6
        assert abs(float(solution.joint_pos[11])) <= hold_cfg.ankle_roll_limit_rad + 1e-6
        assert float(solution.joint_pos[4]) >= hold_cfg.ankle_pitch_min_rad - 1e-6
        assert not sample["nonfinite"]

        runtime = LadderRuntime.get(env.unwrapped)
        assert "geom_pos" in env.unwrapped.sim.expanded_fields
        assert runtime.sample is not None
        assert bool(runtime.sample.active_mask[0, hold_cfg.left_foot_rung_id].item())
    finally:
        env.close()


def test_hold_pose_spawn_clear_on_secondary_seed(hold_cfg, fixed_ladder_cfg):
    """Seed 45455 previously buried ankles; plant+realign must keep soles clear."""
    from train_mimic.tasks.climbing.debug_hold_pose import (
        collect_hold_sample,
        initialize_hold_pose_scene,
    )

    env = _make_hold_env(seed=45455, hold_cfg=hold_cfg, ladder_cfg=fixed_ladder_cfg)
    try:
        solution = initialize_hold_pose_scene(env.unwrapped, hold_cfg)
        sample = collect_hold_sample(env.unwrapped, hold_cfg)
        r = float(hold_cfg.rung_radius_m)
        assert float(sample["foot_axis_dist_m"].min()) > r
        assert float(solution.joint_pos[4]) >= hold_cfg.ankle_pitch_min_rad - 1e-6
        assert abs(float(solution.joint_pos[5])) <= 1e-6
        assert abs(float(solution.joint_pos[11])) <= 1e-6
    finally:
        env.close()


def test_hold_pose_logger_shapes_and_finite_sim(hold_cfg, fixed_ladder_cfg, tmp_path: Path):
    from train_mimic.tasks.climbing.debug_hold_pose import (
        run_headless_hold,
        save_hold_pose_npz,
    )

    # Seed-42 hold should satisfy provisional Stage 6 acceptance gates.
    short_cfg = HoldPoseConfig(hold_duration_s=2.5)
    env = _make_hold_env(seed=42, hold_cfg=short_cfg, ladder_cfg=fixed_ladder_cfg)
    try:
        solution, metrics, summary, failure = run_headless_hold(env.unwrapped, short_cfg)
        arrays = metrics.as_arrays()
        assert arrays["pelvis_pos_w"].ndim == 2 and arrays["pelvis_pos_w"].shape[1] == 3
        assert arrays["hand_attach_error_m"].shape[1] == 2
        assert arrays["foot_slip_m"].shape[1] == 2
        assert arrays["joint_torque"].ndim == 2
        assert np.isfinite(arrays["pelvis_pos_w"]).all()
        assert not bool(summary["any_nonfinite"])
        assert int(summary["max_nacon"]) < int(env.cfg.sim.nconmax)
        assert int(summary["max_nefc"]) < int(env.cfg.sim.njmax)
        assert float(summary["max_pelvis_drop_m"]) <= short_cfg.max_pelvis_drop_m
        assert float(summary["max_foot_slip_m"]) <= short_cfg.max_foot_slip_m
        assert float(summary["any_foot_contact_fraction"]) >= short_cfg.min_any_foot_contact_fraction
        assert failure == "ok"

        out = tmp_path / "hold_pose.npz"
        save_hold_pose_npz(str(out), solution, metrics, summary)
        assert out.is_file()
        loaded = np.load(out)
        assert "joint_pos" in loaded.files
        assert loaded["joint_pos"].shape == (29,)
    finally:
        env.close()


def test_hold_action_targets_ik_joints(hold_cfg, fixed_ladder_cfg):
    from train_mimic.tasks.climbing.debug_hold_pose import (
        build_hold_action,
        initialize_hold_pose_scene,
    )

    env = _make_hold_env(hold_cfg=hold_cfg, ladder_cfg=fixed_ladder_cfg)
    try:
        solution = initialize_hold_pose_scene(env.unwrapped, hold_cfg)
        action = build_hold_action(env.unwrapped, solution)
        assert action.shape == (1, env.action_manager.total_action_dim)
        assert torch.isfinite(action).all()
        # Neutral latch while force-seeded: do not request detach.
        assert float(action[0, -2].item()) == 0.0
        assert float(action[0, -1].item()) == 0.0
    finally:
        env.close()


def test_tracking_registry_untouched_by_hold_pose_import():
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    assert "General-Tracking-G1" in tasks
    assert "General-Climbing-G1" in tasks
