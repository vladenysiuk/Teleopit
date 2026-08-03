"""Stage 7 checks for reset robustness, scripted agents, and end-to-end MDP."""

from __future__ import annotations

import pytest
import torch

from train_mimic.tasks.climbing.config.hold_pose import HoldPoseConfig
from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.debug_mdp import (
    AssistedHoldScriptedAgent,
    FullMdpScriptedAgent,
    capacity_within_limits,
    make_mdp_fault_injection_hook,
    make_random_action,
    make_zero_action,
    observations_finite,
    pinned_ladder_cfg,
    randomized_ladder_cfg,
    rollout_signature,
    run_agent_rollout,
    run_scripted_hold_regression,
    seed_hand_latch,
    simulation_capacity_snapshot,
    stress_random_resets,
    stress_subset_resets,
)
from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.ladder.reward_state import ClimbRewardState
from train_mimic.tasks.climbing.reset_state import (
    capture_reset_state,
    history_is_post_reset,
    pollute_reset_sensitive_state,
)


@pytest.fixture
def pinned_ladder() -> LadderConfig:
    return pinned_ladder_cfg()


@pytest.fixture
def randomized_ladder() -> LadderConfig:
    return randomized_ladder_cfg()


@pytest.fixture
def hold_cfg() -> HoldPoseConfig:
    return HoldPoseConfig(hold_duration_s=2.0)


def _make_mdp_env(
    seed: int = 17,
    *,
    num_envs: int = 1,
    ladder_cfg: LadderConfig | None = None,
    episode_length_s: float = 2.0,
    play: bool = True,
    device: str = "cpu",
):
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import make_climbing_mdp_env_cfg

    cfg = make_climbing_mdp_env_cfg(
        num_envs=num_envs,
        seed=seed,
        ladder_cfg=ladder_cfg,
        episode_length_s=episode_length_s,
        play=play,
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()
    return env


def _make_hold_env(seed: int = 42, *, hold_cfg: HoldPoseConfig | None = None):
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import make_climbing_hold_pose_env_cfg

    cfg = make_climbing_hold_pose_env_cfg(
        num_envs=1,
        seed=seed,
        hold_cfg=hold_cfg or HoldPoseConfig(hold_duration_s=2.0),
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    return env


def _establish_left_attach(env, *, rung_id: int = 2) -> None:
    seed_hand_latch(env, hand="left", rung_id=rung_id, site_id=1)


def test_randomized_300_resets_with_post_steps(randomized_ladder: LadderConfig) -> None:
    env = _make_mdp_env(seed=3, ladder_cfg=randomized_ladder, episode_length_s=1.0)
    try:
        stats = stress_random_resets(
            env,
            num_resets=300,
            steps_after_reset=3,
            randomized_ladder=True,
        )
        assert stats["ok"]
        assert stats["resets"] == 300
        assert stats["stale_state_failures"] == 0
        assert stats["nonfinite_steps"] == 0
        assert stats["ladder_variants"] > 1
        assert stats["ladder_resampled"]
        assert capacity_within_limits(
            env,
            {"nacon_total": int(stats["max_nacon"]), "max_nefc": int(stats["max_nefc"])},
        )
    finally:
        env.close()


def test_pinned_geometry_reset_reproducibility(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(seed=11, ladder_cfg=pinned_ladder, episode_length_s=1.0)
    try:
        stats = stress_random_resets(
            env,
            num_resets=20,
            steps_after_reset=3,
            randomized_ladder=False,
        )
        assert stats["ok"]
        assert stats["ladder_variants"] == 1
    finally:
        env.close()


def test_no_stale_latch_after_full_reset(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(num_envs=1, seed=5, ladder_cfg=pinned_ladder)
    try:
        _establish_left_attach(env, rung_id=2)
        latch = ClimbLatchState.get(env)
        assert bool(latch.state.attached[0, 0].item())
        assert int(env.sim.data.eq_active[0].sum().item()) == 1

        env.reset()
        latch._ensure_state()
        assert not bool(latch.state.attached.any().item())
        assert int(latch.state.rung_id.min().item()) == -1
        assert int(env.sim.data.eq_active[0].sum().item()) == 0
        assert observations_finite(env)
    finally:
        env.close()


def test_no_stale_reward_memory_after_full_reset(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(seed=9, ladder_cfg=pinned_ladder)
    try:
        state = ClimbRewardState.get(env)
        state.max_rewarded_attachment_height_l[:] = 1.2
        state.valid_higher_attachment_count[:] = 4
        state.invalid_latch_count[:] = 3
        state.success_rewarded[:] = True

        env.reset()
        assert float(state.max_rewarded_attachment_height_l.max().item()) == float("-inf")
        assert int(state.valid_higher_attachment_count.max().item()) == 0
        assert int(state.invalid_latch_count.max().item()) == 0
        assert not bool(state.success_rewarded.any().item())
    finally:
        env.close()


def test_observation_history_cleared_after_reset(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(seed=13, ladder_cfg=pinned_ladder)
    try:
        for _ in range(12):
            env.step(make_random_action(env))
        pollute_reset_sensitive_state(env)
        env.reset()
        assert history_is_post_reset(env, env_idx=0, steps_since_reset=0)
        for step_i in range(4):
            env.step(make_zero_action(env))
            assert history_is_post_reset(env, env_idx=0, steps_since_reset=step_i + 1)
    finally:
        env.close()


def test_subset_reset_preserves_unselected_state(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(num_envs=4, seed=21, ladder_cfg=pinned_ladder)
    try:
        seed_hand_latch(env, hand="left", rung_id=2, site_id=1, env_ids=torch.tensor([0]))
        seed_hand_latch(env, hand="left", rung_id=4, site_id=1, env_ids=torch.tensor([1]))
        seed_hand_latch(env, hand="right", rung_id=3, site_id=1, env_ids=torch.tensor([2]))

        reset_ids = torch.tensor([1, 2], dtype=torch.int64)
        preserved_ids = torch.tensor([0, 3], dtype=torch.int64)
        stats = stress_subset_resets(
            env,
            reset_env_ids=reset_ids,
            preserved_env_ids=preserved_ids,
            steps_after_reset=3,
        )
        assert stats["ok"]
        latch = ClimbLatchState.get(env)
        assert bool(latch.state.attached[0, 0].item())
        assert not bool(latch.state.attached[1].any().item())
        assert not bool(latch.state.attached[2].any().item())
        assert not bool(latch.state.attached[3].any().item())
    finally:
        env.close()


def test_fixed_seed_and_actions_are_repeatable(pinned_ladder: LadderConfig) -> None:
    actions = [0, 1, 0, -1, 1]

    def _roll(seed: int) -> list[tuple[float, ...]]:
        env = _make_mdp_env(seed=seed, ladder_cfg=pinned_ladder, episode_length_s=3.0)
        try:
            gen = torch.Generator(device=env.device)
            gen.manual_seed(99)
            sigs: list[tuple[float, ...]] = []
            for step_idx in range(len(actions)):
                if actions[step_idx] == 0:
                    action = make_zero_action(env)
                else:
                    action = make_random_action(env, generator=gen)
                env.step(action)
                sigs.append(rollout_signature(env))
            return sigs
        finally:
            env.close()

    assert _roll(42) == _roll(42)


def test_zero_agent_rollout(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(seed=7, ladder_cfg=pinned_ladder, episode_length_s=1.5)
    try:
        summary = run_agent_rollout(
            env,
            lambda _env, _step: make_zero_action(_env),
            num_steps=60,
        )
        assert summary.steps == 60
        assert summary.nonfinite_obs_steps == 0
        assert summary.nonfinite_sim_steps == 0
        assert capacity_within_limits(
            env,
            {"nacon_total": summary.max_nacon, "max_nefc": summary.max_nefc},
        )
    finally:
        env.close()


def test_random_agent_rollout(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(seed=8, ladder_cfg=pinned_ladder, episode_length_s=1.5)
    try:
        generator = torch.Generator(device=env.device)
        generator.manual_seed(123)
        summary = run_agent_rollout(
            env,
            lambda _env, _step: make_random_action(_env, generator=generator),
            num_steps=80,
        )
        assert summary.steps == 80
        assert summary.nonfinite_obs_steps == 0
        assert summary.nonfinite_sim_steps == 0
        latch = ClimbLatchState.get(env)
        for env_idx in range(env.num_envs):
            active = latch.backend.read_active_eq_ids(env_idx, 0)
            active += latch.backend.read_active_eq_ids(env_idx, 1)
            assert len(active) <= 2
    finally:
        env.close()


def test_assisted_hold_scripted_regression(hold_cfg: HoldPoseConfig) -> None:
    env = _make_hold_env(seed=42, hold_cfg=hold_cfg)
    try:
        agent = AssistedHoldScriptedAgent(env, hold_cfg=hold_cfg)
        summary = run_agent_rollout(
            env,
            lambda e, s: agent(e, s),
            num_steps=sum(p.steps for p in agent.script),
        )
        assert summary.steps == sum(p.steps for p in agent.script)
        assert summary.nonfinite_obs_steps == 0
        assert summary.nonfinite_sim_steps == 0
    finally:
        env.close()


def test_scripted_hold_reproduces_stage6_thresholds(hold_cfg: HoldPoseConfig) -> None:
    short_cfg = HoldPoseConfig(hold_duration_s=2.5)
    env = _make_hold_env(seed=42, hold_cfg=short_cfg)
    try:
        summary, failure = run_scripted_hold_regression(env, short_cfg)
        assert failure == "ok"
        assert float(summary["max_pelvis_drop_m"]) <= short_cfg.max_pelvis_drop_m
        assert float(summary["max_foot_slip_m"]) <= short_cfg.max_foot_slip_m
        assert float(summary["any_foot_contact_fraction"]) >= short_cfg.min_any_foot_contact_fraction
    finally:
        env.close()


def test_full_mdp_scripted_rollout(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(seed=42, ladder_cfg=pinned_ladder, episode_length_s=5.0)
    try:
        agent = FullMdpScriptedAgent(env)
        fault_hook = make_mdp_fault_injection_hook(agent)
        summary = run_agent_rollout(
            env,
            lambda e, s: agent(e, s),
            num_steps=sum(p.steps for p in agent.script),
            pre_step_hook=fault_hook,
        )
        assert summary.steps == sum(p.steps for p in agent.script)
        assert summary.nonfinite_obs_steps == 0
        assert summary.nonfinite_sim_steps == 0
        assert capacity_within_limits(
            env,
            {"nacon_total": summary.max_nacon, "max_nefc": summary.max_nefc},
        )
    finally:
        env.close()


@pytest.mark.parametrize("num_envs", [1, 4, 16])
def test_increasing_batch_sizes_without_capacity_overflow(
    num_envs: int,
    pinned_ladder: LadderConfig,
) -> None:
    env = _make_mdp_env(num_envs=num_envs, seed=100 + num_envs, ladder_cfg=pinned_ladder)
    try:
        summary = run_agent_rollout(
            env,
            lambda _env, _step: make_zero_action(_env),
            num_steps=20,
        )
        assert summary.nonfinite_obs_steps == 0
        assert capacity_within_limits(
            env,
            {"nacon_total": summary.max_nacon, "max_nefc": summary.max_nefc},
        )
        contacts = ClimbContactState.get(env)
        contacts.update()
        assert contacts.state.hand_in_contact.shape == (num_envs, 2)
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU capacity test")
def test_gpu_batch_64_capacity(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(
        num_envs=64,
        seed=200,
        ladder_cfg=pinned_ladder,
        episode_length_s=1.0,
        device="cuda:0",
    )
    try:
        summary = run_agent_rollout(
            env,
            lambda _env, _step: make_zero_action(_env),
            num_steps=30,
        )
        assert summary.nonfinite_obs_steps == 0
        assert summary.nonfinite_sim_steps == 0
        assert summary.max_equalities_per_env <= 2
        assert summary.min_ncon_headroom > 0
        assert summary.min_njmax_headroom > 0
        assert capacity_within_limits(
            env,
            {"nacon_total": summary.max_nacon, "max_nefc": summary.max_nefc},
        )
    finally:
        env.close()


def test_simulation_capacity_snapshot_shape(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(seed=31, ladder_cfg=pinned_ladder)
    try:
        env.step(make_zero_action(env))
        cap = simulation_capacity_snapshot(env)
        assert {
            "nacon_total",
            "max_nefc",
            "active_equalities",
            "ncon_headroom",
            "njmax_headroom",
        }.issubset(cap.keys())
        assert capacity_within_limits(env, cap)
    finally:
        env.close()


def test_mdp_env_has_full_reward_and_termination_wiring(pinned_ladder: LadderConfig) -> None:
    from train_mimic.tasks.climbing.config.env import make_climbing_mdp_env_cfg

    cfg = make_climbing_mdp_env_cfg(num_envs=2, seed=0, ladder_cfg=pinned_ladder)
    assert "upward_progress" in cfg.rewards
    assert "success" in cfg.terminations
    assert "fall" in cfg.terminations
    assert "time_out" in cfg.terminations
    assert "max_head_height_l" in cfg.metrics
    reset_names = [name for name, term in cfg.events.items() if term.mode == "reset"]
    assert "reset_climb_latch" in reset_names
    assert "reset_climb_rewards" in reset_names
    assert reset_names.index("reset_climb_latch") < reset_names.index("reset_ladder")


def test_short_episode_timeout_in_mdp_rollout(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(seed=42, ladder_cfg=pinned_ladder, episode_length_s=0.25)
    try:
        summary = run_agent_rollout(
            env,
            lambda _env, _step: make_zero_action(_env),
            num_steps=120,
        )
        assert summary.completed_episodes >= 1 or summary.termination_counts.get("time_out", 0) >= 1
    finally:
        env.close()


def test_reset_state_snapshot_fields(pinned_ladder: LadderConfig) -> None:
    env = _make_mdp_env(seed=7, ladder_cfg=pinned_ladder)
    try:
        pollute_reset_sensitive_state(env)
        env.reset()
        snap = capture_reset_state(env, env_idx=0)
        assert snap.eq_active_count == 0
        assert not any(snap.latch_attached)
        assert snap.reward_valid_higher_attachments == 0
        assert snap.history_num_pushes >= 1
        assert abs(snap.history_prev_latch_t0[0]) < 0.5
        assert snap.episode_length == 0
    finally:
        env.close()


def test_tracking_registry_untouched_by_stage7_import() -> None:
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    assert "General-Tracking-G1" in tasks
    assert "General-Climbing-G1" in tasks
