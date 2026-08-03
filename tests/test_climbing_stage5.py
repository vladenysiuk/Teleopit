"""Stage 5 checks for General-Climbing-G1 rewards, metrics, and terminations."""

from __future__ import annotations

import pytest
import torch

from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.rewards import (
    ClimbingRewardConfig,
    ClimbingTerminationConfig,
)
from train_mimic.tasks.climbing.ladder.reward_state import ClimbRewardState
from train_mimic.tasks.climbing.mdp import rewards as climb_rewards
from train_mimic.tasks.climbing.mdp.common import (
    success_goal_height_l,
    top_active_rung_height_l,
)
from train_mimic.tasks.climbing.mdp.terminations import climbing_success_predicate


@pytest.fixture
def reward_cfg() -> ClimbingRewardConfig:
    return ClimbingRewardConfig(
        upward_progress_weight=10.0,
        new_attachment_weight=2.0,
        success_weight=20.0,
        success_pelvis_clearance_below_top_l=0.15,
        success_top_attach_margin_l=0.05,
        success_min_attached_hands=1,
        attachment_height_eps=1.0e-4,
        time_penalty_coeff=0.05,
        action_rate_weight=0.01,
        effort_weight=1.0e-4,
        invalid_latch_weight=0.5,
    )


def _success_params(reward_cfg: ClimbingRewardConfig) -> dict:
    return {
        "body_name": reward_cfg.progress_body,
        "pelvis_clearance_below_top_l": reward_cfg.success_pelvis_clearance_below_top_l,
        "top_attach_margin_l": reward_cfg.success_top_attach_margin_l,
        "min_attached_hands": reward_cfg.success_min_attached_hands,
    }


def _make_rewards_env(
    seed: int = 11,
    *,
    num_envs: int = 1,
    reward_cfg: ClimbingRewardConfig | None = None,
    ladder_cfg: LadderConfig | None = None,
    decimation: int | None = None,
):
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import make_climbing_rewards_env_cfg

    cfg = make_climbing_rewards_env_cfg(
        num_envs=num_envs,
        seed=seed,
        reward_cfg=reward_cfg,
        ladder_cfg=ladder_cfg,
    )
    if decimation is not None:
        cfg.decimation = decimation
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    return env


def _set_pelvis_ladder_height(env, value: float, *, env_idx: int = 0) -> None:
    from train_mimic.tasks.climbing.ladder.state import LadderRuntime, _quat_apply

    runtime = LadderRuntime.get(env.unwrapped)
    sample = runtime.sample
    assert sample is not None
    robot = env.unwrapped.scene["robot"]
    pelvis_id = robot.body_names.index("pelvis")
    local_up = torch.zeros(3, device=env.unwrapped.device, dtype=torch.float32)
    local_up[2] = 1.0
    world_up = _quat_apply(sample.frame_quat[env_idx], local_up)
    target_pelvis = sample.frame_pos[env_idx] + world_up * value
    current_pelvis = robot.data.body_link_pos_w[env_idx, pelvis_id]
    delta = target_pelvis - current_pelvis
    root = robot.data.root_link_pos_w.clone()
    root[env_idx] = root[env_idx] + delta
    pose = torch.cat([root, robot.data.root_link_quat_w], dim=-1)
    robot.write_root_link_pose_to_sim(pose)
    env.unwrapped.sim.forward()


def _set_hand_height_only(env, *, delta_l: float, hand: str = "left", env_idx: int = 0) -> None:
    from train_mimic.tasks.climbing.config.robot import LEFT_HAND_POINT, RIGHT_HAND_POINT
    from train_mimic.tasks.climbing.ladder.state import LadderRuntime, _quat_apply

    runtime = LadderRuntime.get(env.unwrapped)
    sample = runtime.sample
    assert sample is not None
    geom_name = LEFT_HAND_POINT if hand == "left" else RIGHT_HAND_POINT
    model = env.unwrapped.sim.mj_model
    prefix = "robot/"
    geom_id = model.geom(f"{prefix}{geom_name}").id
    sim = env.unwrapped.sim
    local_up = torch.zeros(3, device=env.unwrapped.device, dtype=torch.float32)
    local_up[2] = 1.0
    world_up = _quat_apply(sample.frame_quat[env_idx], local_up)
    pos = sim.data.geom_xpos.clone()
    pos[env_idx, geom_id] = pos[env_idx, geom_id] + world_up * delta_l
    sim.data.geom_xpos[:] = pos


def _contrib(raw: torch.Tensor, weight: float, dt: float, *, scale_dt: bool = True) -> float:
    value = float(raw[0].item()) * weight
    if scale_dt:
        value *= dt
    return value


def test_general_climbing_env_has_reward_terms(reward_cfg: ClimbingRewardConfig) -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    from train_mimic.tasks.climbing.config.constants import CLIMBING_TASK_ID

    cfg = load_env_cfg(CLIMBING_TASK_ID)
    expected = {
        "upward_progress",
        "new_higher_attachment",
        "success",
        "time_penalty",
        "action_rate",
        "effort",
        "invalid_latch",
    }
    assert expected.issubset(set(cfg.rewards.keys()))
    assert "_step_recorder" not in cfg.rewards
    assert "success" in cfg.terminations
    assert "fall" in cfg.terminations
    assert "max_head_height_l" in cfg.metrics
    assert "head_height_l" in cfg.metrics
    assert "max_pelvis_height_l" not in cfg.metrics
    assert "pelvis_height_l" not in cfg.metrics
    assert "hand_contact_fraction" in cfg.metrics


def test_unchanged_state_only_cost_terms(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    progress = climb_rewards.upward_progress(env.unwrapped, body_name="pelvis")
    attach = climb_rewards.new_higher_attachment(
        env.unwrapped, height_eps=reward_cfg.attachment_height_eps
    )
    assert float(progress[0].item()) == pytest.approx(0.0, abs=1e-5)
    assert float(attach[0].item()) == pytest.approx(0.0, abs=1e-5)
    time_mask = climb_rewards.time_penalty_mask(env.unwrapped, **_success_params(reward_cfg))
    assert float(time_mask[0].item()) == pytest.approx(1.0, abs=1e-5)
    env.close()


def test_pelvis_rise_gives_exact_progress(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    from train_mimic.tasks.climbing.mdp.common import body_ladder_relative_height

    state = ClimbRewardState.get(env.unwrapped)
    h0 = body_ladder_relative_height(env.unwrapped, body_name="pelvis")
    state.max_rewarded_progress_height_l[:] = h0
    target = h0 + 0.1
    _set_pelvis_ladder_height(env, float(target[0].item()))
    progress = climb_rewards.upward_progress(env.unwrapped, body_name="pelvis")
    dt = env.unwrapped.step_dt
    assert float(progress[0].item()) == pytest.approx(0.1 / dt, rel=1e-3, abs=1e-3)
    assert _contrib(progress, reward_cfg.upward_progress_weight, dt) == pytest.approx(
        reward_cfg.upward_progress_weight * 0.1, rel=1e-3, abs=1e-4
    )
    env.close()


def test_rise_fall_rise_below_max_pays_nothing(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    from train_mimic.tasks.climbing.mdp.common import body_ladder_relative_height

    state = ClimbRewardState.get(env.unwrapped)
    h0 = float(body_ladder_relative_height(env.unwrapped, body_name="pelvis")[0].item())
    state.max_rewarded_progress_height_l[:] = h0
    _set_pelvis_ladder_height(env, h0 + 0.12)
    first = climb_rewards.upward_progress(env.unwrapped, body_name="pelvis")
    assert float(first[0].item()) > 0.0
    _set_pelvis_ladder_height(env, h0 + 0.02)
    dip = climb_rewards.upward_progress(env.unwrapped, body_name="pelvis")
    assert float(dip[0].item()) == pytest.approx(0.0, abs=1e-6)
    _set_pelvis_ladder_height(env, h0 + 0.08)
    second_rise = climb_rewards.upward_progress(env.unwrapped, body_name="pelvis")
    assert float(second_rise[0].item()) == pytest.approx(0.0, abs=1e-6)
    env.close()


def test_hand_rise_alone_gives_zero_progress(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    state = ClimbRewardState.get(env.unwrapped)
    climb_rewards.upward_progress(env.unwrapped, body_name="pelvis")
    anchor = state.max_rewarded_progress_height_l.clone()
    _set_hand_height_only(env, delta_l=0.15)
    progress = climb_rewards.upward_progress(env.unwrapped, body_name="pelvis")
    assert float(progress[0].item()) == pytest.approx(0.0, abs=1e-4)
    assert torch.allclose(state.max_rewarded_progress_height_l, anchor, atol=1e-4)
    env.close()


def test_first_higher_attachment_pays_once(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState

    latch = ClimbLatchState.get(env.unwrapped)
    latch._ensure_state()
    latch.state.attach_event[0, 0] = True
    latch.state.rung_id[0, 0] = 3
    first = climb_rewards.new_higher_attachment(
        env.unwrapped, height_eps=reward_cfg.attachment_height_eps
    )
    second = climb_rewards.new_higher_attachment(
        env.unwrapped, height_eps=reward_cfg.attachment_height_eps
    )
    dt = env.unwrapped.step_dt
    assert float(first[0].item()) == pytest.approx(1.0 / dt, rel=1e-5)
    assert float(second[0].item()) == pytest.approx(0.0, abs=1e-8)
    env.close()


def test_detach_reattach_same_rung_does_not_farm(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState

    latch = ClimbLatchState.get(env.unwrapped)
    latch._ensure_state()
    state = ClimbRewardState.get(env.unwrapped)
    latch.state.attach_event[0, 0] = True
    latch.state.rung_id[0, 0] = 2
    first = climb_rewards.new_higher_attachment(
        env.unwrapped, height_eps=reward_cfg.attachment_height_eps
    )
    assert float(first[0].item()) > 0.0
    latch.state.attach_event[0, 0] = True
    latch.state.rung_id[0, 0] = 2
    second = climb_rewards.new_higher_attachment(
        env.unwrapped, height_eps=reward_cfg.attachment_height_eps
    )
    assert float(second[0].item()) == pytest.approx(0.0, abs=1e-8)
    assert int(state.valid_higher_attachment_count[0].item()) == 1
    env.close()


def test_two_hands_same_height_pay_once(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState

    latch = ClimbLatchState.get(env.unwrapped)
    latch._ensure_state()
    state = ClimbRewardState.get(env.unwrapped)
    latch.state.attach_event[0, 0] = True
    latch.state.attach_event[0, 1] = True
    latch.state.rung_id[0, 0] = 2
    latch.state.rung_id[0, 1] = 2
    bonus = climb_rewards.new_higher_attachment(
        env.unwrapped, height_eps=reward_cfg.attachment_height_eps
    )
    dt = env.unwrapped.step_dt
    assert float(bonus[0].item()) == pytest.approx(1.0 / dt, rel=1e-5)
    assert int(state.valid_higher_attachment_count[0].item()) == 1
    env.close()


def test_attachment_height_tolerance(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState

    latch = ClimbLatchState.get(env.unwrapped)
    latch._ensure_state()
    state = ClimbRewardState.get(env.unwrapped)
    latch.state.attach_event[0, 0] = True
    latch.state.rung_id[0, 0] = 2
    climb_rewards.new_higher_attachment(env.unwrapped, height_eps=reward_cfg.attachment_height_eps)
    paid = float(state.max_rewarded_attachment_height_l[0].item())
    # Tiny increase below eps must not pay again.
    state.max_rewarded_attachment_height_l[0] = paid - 0.5 * reward_cfg.attachment_height_eps
    latch.state.attach_event[0, 0] = True
    latch.state.rung_id[0, 0] = 2
    again = climb_rewards.new_higher_attachment(
        env.unwrapped, height_eps=reward_cfg.attachment_height_eps
    )
    assert float(again[0].item()) == pytest.approx(0.0, abs=1e-8)
    env.close()


def test_invalid_latch_one_penalty_per_attempt(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState

    latch = ClimbLatchState.get(env.unwrapped)
    latch._ensure_state()
    state = ClimbRewardState.get(env.unwrapped)
    dt = env.unwrapped.step_dt
    totals = []
    for _ in range(5):
        latch.state.invalid_attach_request[0, 0] = True
        latch.state.latch_command[0, 0] = 1.0
        raw = climb_rewards.invalid_latch_request(env.unwrapped)
        totals.append(float(raw[0].item()))
    assert totals[0] == pytest.approx(1.0 / dt, rel=1e-5)
    assert all(v == pytest.approx(0.0, abs=1e-8) for v in totals[1:])
    assert int(state.invalid_latch_count[0].item()) == 1
    # Release command, then a new attempt pays again.
    latch.state.latch_command[0, 0] = 0.0
    latch.state.invalid_attach_request[0, 0] = False
    climb_rewards.invalid_latch_request(env.unwrapped)
    latch.state.invalid_attach_request[0, 0] = True
    latch.state.latch_command[0, 0] = 1.0
    second = climb_rewards.invalid_latch_request(env.unwrapped)
    assert float(second[0].item()) == pytest.approx(1.0 / dt, rel=1e-5)
    assert int(state.invalid_latch_count[0].item()) == 2
    env.close()


def test_one_physical_second_time_cost(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    dt = env.unwrapped.step_dt
    steps = int(round(1.0 / dt))
    total = 0.0
    params = _success_params(reward_cfg)
    for _ in range(steps):
        mask = climb_rewards.time_penalty_mask(env.unwrapped, **params)
        total += _contrib(mask, -reward_cfg.time_penalty_coeff, dt)
    assert total == pytest.approx(-reward_cfg.time_penalty_coeff, rel=1e-3, abs=1e-3)
    env.close()


def test_event_magnitudes_invariant_under_step_dt(reward_cfg: ClimbingRewardConfig) -> None:
    contribs = []
    for decimation in (1, 2):
        env = _make_rewards_env(seed=21, reward_cfg=reward_cfg, decimation=decimation)
        from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
        from train_mimic.tasks.climbing.mdp.common import body_ladder_relative_height

        state = ClimbRewardState.get(env.unwrapped)
        h0 = body_ladder_relative_height(env.unwrapped, body_name="pelvis")
        state.max_rewarded_progress_height_l[:] = h0
        _set_pelvis_ladder_height(env, float((h0 + 0.1)[0].item()))
        progress = climb_rewards.upward_progress(env.unwrapped, body_name="pelvis")
        latch = ClimbLatchState.get(env.unwrapped)
        latch._ensure_state()
        latch.state.attach_event[0, 0] = True
        latch.state.rung_id[0, 0] = 3
        attach = climb_rewards.new_higher_attachment(
            env.unwrapped, height_eps=reward_cfg.attachment_height_eps
        )
        dt = env.unwrapped.step_dt
        contribs.append(
            (
                _contrib(progress, reward_cfg.upward_progress_weight, dt),
                _contrib(attach, reward_cfg.new_attachment_weight, dt),
            )
        )
        env.close()
    assert contribs[0][0] == pytest.approx(contribs[1][0], rel=1e-3, abs=1e-4)
    assert contribs[0][1] == pytest.approx(contribs[1][1], rel=1e-3, abs=1e-4)
    assert contribs[0][0] == pytest.approx(reward_cfg.upward_progress_weight * 0.1, rel=1e-3)
    assert contribs[0][1] == pytest.approx(reward_cfg.new_attachment_weight, rel=1e-3)


def test_success_bonus_and_termination(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
    from train_mimic.tasks.climbing.ladder.state import LadderRuntime

    latch = ClimbLatchState.get(env.unwrapped)
    latch._ensure_state()
    runtime = LadderRuntime.get(env.unwrapped)
    assert runtime.sample is not None
    top_id = int(runtime.sample.num_active[0].item()) - 1
    latch.state.attached[0, 0] = True
    latch.state.rung_id[0, 0] = top_id
    goal = float(
        success_goal_height_l(
            env.unwrapped,
            pelvis_clearance_below_top_l=reward_cfg.success_pelvis_clearance_below_top_l,
        )[0].item()
    )
    _set_pelvis_ladder_height(env, goal + 0.02)
    params = _success_params(reward_cfg)
    bonus = climb_rewards.success_bonus(env.unwrapped, **params)
    again = climb_rewards.success_bonus(env.unwrapped, **params)
    success = climbing_success_predicate(env.unwrapped, **params)
    dt = env.unwrapped.step_dt
    assert float(bonus[0].item()) == pytest.approx(1.0 / dt, rel=1e-5)
    assert float(again[0].item()) == pytest.approx(0.0, abs=1e-8)
    assert bool(success[0].item())
    env.close()


def test_success_requires_near_top_attachment(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState

    latch = ClimbLatchState.get(env.unwrapped)
    latch._ensure_state()
    latch.state.attached[0, 0] = True
    latch.state.rung_id[0, 0] = 0  # bottom rung
    goal = float(
        success_goal_height_l(
            env.unwrapped,
            pelvis_clearance_below_top_l=reward_cfg.success_pelvis_clearance_below_top_l,
        )[0].item()
    )
    _set_pelvis_ladder_height(env, goal + 0.05)
    success = climbing_success_predicate(env.unwrapped, **_success_params(reward_cfg))
    assert not bool(success[0].item())
    env.close()


@pytest.mark.parametrize(
    ("num_rungs", "spacing"),
    [
        (4, 0.22),
        (10, 0.28),
    ],
)
def test_success_attainable_on_short_and_tall_ladders(
    reward_cfg: ClimbingRewardConfig,
    num_rungs: int,
    spacing: float,
) -> None:
    ladder_cfg = LadderConfig(
        min_active_rungs=num_rungs,
        max_active_rungs=num_rungs,
        spacing_min=spacing,
        spacing_max=spacing,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
    )
    env = _make_rewards_env(seed=7, reward_cfg=reward_cfg, ladder_cfg=ladder_cfg)
    from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState

    top_h = float(top_active_rung_height_l(env.unwrapped)[0].item())
    goal = float(
        success_goal_height_l(
            env.unwrapped,
            pelvis_clearance_below_top_l=reward_cfg.success_pelvis_clearance_below_top_l,
        )[0].item()
    )
    expected_top = ladder_cfg.base_height + (num_rungs - 1) * spacing
    assert top_h == pytest.approx(expected_top, abs=1e-5)
    assert goal == pytest.approx(expected_top - reward_cfg.success_pelvis_clearance_below_top_l)

    latch = ClimbLatchState.get(env.unwrapped)
    latch._ensure_state()
    latch.state.attached[0, 0] = True
    latch.state.rung_id[0, 0] = num_rungs - 1
    _set_pelvis_ladder_height(env, goal + 0.01)
    assert bool(climbing_success_predicate(env.unwrapped, **_success_params(reward_cfg))[0].item())
    env.close()


def test_partial_reset_clears_only_selected_envs(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(num_envs=2, reward_cfg=reward_cfg)
    state = ClimbRewardState.get(env.unwrapped)
    state.max_rewarded_progress_height_l[:] = torch.tensor([1.1, 1.2])
    state.max_rewarded_attachment_height_l[:] = torch.tensor([0.4, 0.5])
    state.valid_higher_attachment_count[:] = torch.tensor([2, 3])
    state.invalid_latch_count[:] = torch.tensor([1, 4])
    state.success_rewarded[:] = torch.tensor([True, True])
    state.reset(env.unwrapped, torch.tensor([1], dtype=torch.int64), progress_body="pelvis")
    assert float(state.max_rewarded_progress_height_l[0].item()) == pytest.approx(1.1)
    assert float(state.max_rewarded_attachment_height_l[0].item()) == pytest.approx(0.4)
    assert int(state.valid_higher_attachment_count[0].item()) == 2
    assert int(state.invalid_latch_count[0].item()) == 1
    assert bool(state.success_rewarded[0].item())
    assert float(state.max_rewarded_attachment_height_l[1].item()) == float("-inf")
    assert int(state.valid_higher_attachment_count[1].item()) == 0
    assert int(state.invalid_latch_count[1].item()) == 0
    assert not bool(state.success_rewarded[1].item())
    env.close()


def test_global_translation_invariance(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    from train_mimic.tasks.climbing.ladder.state import LadderRuntime

    climb_rewards.upward_progress(env.unwrapped, body_name="pelvis")
    runtime = LadderRuntime.get(env.unwrapped)
    sample = runtime.sample
    assert sample is not None
    shift = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32)
    sample.frame_pos = sample.frame_pos + shift
    sample.rung_pos_w = sample.rung_pos_w + shift.unsqueeze(1)
    robot = env.unwrapped.scene["robot"]
    root = robot.data.root_link_pos_w + shift
    pose = torch.cat([root, robot.data.root_link_quat_w], dim=-1)
    robot.write_root_link_pose_to_sim(pose)
    progress = climb_rewards.upward_progress(env.unwrapped, body_name="pelvis")
    assert float(progress[0].item()) == pytest.approx(0.0, abs=1e-4)
    env.close()


def test_reward_terms_have_batch_shape(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(num_envs=4, reward_cfg=reward_cfg)
    params = _success_params(reward_cfg)
    terms = {
        "progress": climb_rewards.upward_progress(env.unwrapped, body_name="pelvis"),
        "attach": climb_rewards.new_higher_attachment(
            env.unwrapped, height_eps=reward_cfg.attachment_height_eps
        ),
        "success": climb_rewards.success_bonus(env.unwrapped, **params),
        "time": climb_rewards.time_penalty_mask(env.unwrapped, **params),
        "invalid": climb_rewards.invalid_latch_request(env.unwrapped),
    }
    for name, value in terms.items():
        assert value.shape == (4,), name
        assert value.device.type == "cpu", name
    env.close()


def test_reward_manager_matches_manual_sum(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    # Snapshot state so both paths see identical memory before mutation.
    state = ClimbRewardState.get(env.unwrapped)
    max_pelvis = state.max_rewarded_progress_height_l.clone()
    max_attach = state.max_rewarded_attachment_height_l.clone()
    success_flag = state.success_rewarded.clone()
    invalid_armed = state.invalid_latch_armed.clone()
    invalid_count = state.invalid_latch_count.clone()
    attach_count = state.valid_higher_attachment_count.clone()

    manual = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    dt = env.unwrapped.step_dt
    for name, term_cfg in env.unwrapped.cfg.rewards.items():
        del name
        params = dict(term_cfg.params or {})
        raw = term_cfg.func(env.unwrapped, **params)
        manual += raw * float(term_cfg.weight) * dt

    # Restore memory mutated by the manual pass, then use the manager.
    state.max_rewarded_progress_height_l[:] = max_pelvis
    state.max_rewarded_attachment_height_l[:] = max_attach
    state.success_rewarded[:] = success_flag
    state.invalid_latch_armed[:] = invalid_armed
    state.invalid_latch_count[:] = invalid_count
    state.valid_higher_attachment_count[:] = attach_count
    computed = env.unwrapped.reward_manager.compute(env.unwrapped.step_dt)
    assert torch.allclose(computed, manual, atol=1e-5)
    env.close()


def test_reset_clears_attachment_memory(reward_cfg: ClimbingRewardConfig) -> None:
    env = _make_rewards_env(reward_cfg=reward_cfg)
    state = ClimbRewardState.get(env.unwrapped)
    state.max_rewarded_attachment_height_l[:] = 1.2
    state.valid_higher_attachment_count[:] = 3
    state.invalid_latch_count[:] = 2
    env.reset()
    assert float(state.max_rewarded_attachment_height_l[0].item()) == float("-inf")
    assert int(state.valid_higher_attachment_count[0].item()) == 0
    assert int(state.invalid_latch_count[0].item()) == 0
    env.close()


def test_configure_climbing_rewards_respects_optional_overload() -> None:
    pytest.importorskip("mjlab")
    import train_mimic.tasks  # noqa: F401
    from train_mimic.tasks.climbing.config.env import make_general_climbing_env_cfg

    cfg = make_general_climbing_env_cfg(
        num_envs=1,
        reward_cfg=ClimbingRewardConfig(latch_overload_weight=0.25),
        termination_cfg=ClimbingTerminationConfig(enable_latch_overload=True),
    )
    assert "latch_overload" in cfg.rewards
    assert "latch_overload" in cfg.terminations


def test_mean_successful_time_to_success_ignores_failure_sentinel() -> None:
    import math

    from train_mimic.tasks.climbing.mdp.metrics import (
        correct_mean_time_to_success,
        mean_successful_time_to_success,
    )

    values = torch.tensor([-1.0, 100.0, -1.0, 200.0])
    assert float(mean_successful_time_to_success(values).item()) == pytest.approx(150.0)
    assert math.isnan(float(mean_successful_time_to_success(torch.tensor([-1.0, -1.0])).item()))

    contaminated = float(values.mean().item())
    success_rate = float((values >= 0).float().mean().item())
    assert correct_mean_time_to_success(contaminated, success_rate) == pytest.approx(150.0)
    assert math.isnan(correct_mean_time_to_success(contaminated, 0.0))

def test_time_to_success_episode_metric_averages_successful_only(
    reward_cfg: ClimbingRewardConfig,
) -> None:
    pytest.importorskip("mjlab")
    env = _make_rewards_env(reward_cfg=reward_cfg, num_envs=4)
    base = env.unwrapped
    mgr = base.metrics_manager
    assert "time_to_success" in mgr.active_terms
    idx = mgr.active_terms.index("time_to_success")
    mgr._step_values[:, idx] = torch.tensor([-1.0, 100.0, -1.0, 200.0], device=base.device)
    mgr._step_count[:] = 1
    extras = mgr.reset(torch.arange(4, device=base.device))
    key = "Episode_Metrics/time_to_success"
    assert key in extras
    assert float(extras[key].item()) == pytest.approx(150.0)

    mgr._step_values[:, idx] = torch.tensor([-1.0, -1.0, -1.0, -1.0], device=base.device)
    mgr._step_count[:] = 1
    extras = mgr.reset(torch.arange(4, device=base.device))
    assert key not in extras
    env.close()
