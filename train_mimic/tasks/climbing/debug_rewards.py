"""Debug helpers for climbing reward and termination inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from train_mimic.tasks.climbing.config.latch import LatchConfig
from train_mimic.tasks.climbing.debug_latch import build_latch_action
from train_mimic.tasks.climbing.debug_probe import (
    nudge_robot_along_ladder_up,
    nudge_robot_away_from_ladder,
    nudge_robot_toward_rung,
    place_robot_hand_near_rung,
    _zero_root_velocity,
)
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.ladder.reward_state import ClimbRewardState
from train_mimic.tasks.climbing.ladder.state import LadderRuntime
from train_mimic.tasks.climbing.mdp.common import (
    body_ladder_relative_height,
    success_goal_height_l,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

MotionMode = Literal["hold", "slide_robot_to_rung", "raise_pelvis", "invalid_attach"]


@dataclass
class RewardScriptPhase:
    name: str
    steps: int
    latch_left: float = 0.0
    latch_right: float = 0.0
    motion_mode: MotionMode = "hold"
    hand: str = "left"
    rung_id: int = 3
    separation_m: float = 0.08
    pelvis_delta_l: float = 0.0
    approach_step_m: float = 0.01
    seed_demo_attach: bool = False


DEFAULT_REWARD_SCRIPT: tuple[RewardScriptPhase, ...] = (
    RewardScriptPhase(name="baseline_hold", steps=10, motion_mode="hold"),
    RewardScriptPhase(
        name="hand_only_no_progress",
        steps=8,
        motion_mode="slide_robot_to_rung",
        hand="left",
        rung_id=2,
        approach_step_m=0.006,
    ),
    RewardScriptPhase(
        name="attach_higher_rung",
        steps=6,
        motion_mode="slide_robot_to_rung",
        hand="left",
        rung_id=3,
        separation_m=0.04,
        latch_left=1.0,
        approach_step_m=0.006,
    ),
    RewardScriptPhase(
        name="wait_after_attach",
        steps=20,
        motion_mode="hold",
        latch_left=0.0,
    ),
    RewardScriptPhase(
        name="raise_pelvis",
        steps=12,
        motion_mode="raise_pelvis",
        pelvis_delta_l=0.08,
        latch_left=0.0,
    ),
    RewardScriptPhase(
        name="invalid_latch_request",
        steps=4,
        motion_mode="invalid_attach",
        latch_left=1.0,
    ),
    RewardScriptPhase(
        name="trigger_success",
        steps=16,
        motion_mode="raise_pelvis",
        pelvis_delta_l=0.55,
        latch_left=0.0,
        seed_demo_attach=True,
    ),
)


def initialize_rewards_debug_scene(env: ManagerBasedRlEnv) -> None:
    """Seed the robot near a mid ladder rung for reward inspection."""
    place_robot_hand_near_rung(env, hand="left", rung_id=2, separation_m=0.10)
    robot = env.scene["robot"]
    root = robot.data.root_link_pos_w.clone()
    root[:, 2] = torch.maximum(root[:, 2], torch.tensor(0.85, device=env.device))
    root[:, 0] = torch.minimum(root[:, 0], torch.tensor(0.35, device=env.device))
    pose = torch.cat([root, robot.data.root_link_quat_w], dim=-1)
    robot.write_root_link_pose_to_sim(pose)
    zero_vel = torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)
    robot.write_root_link_velocity_to_sim(zero_vel)
    env.sim.forward()


def _seed_demo_attachment(env: ManagerBasedRlEnv, *, hand: str = "left", rung_id: int | None = None) -> None:
    """Physically latch one hand to a rung for success-demo phases.

    Places the hand near the rung, activates the MuJoCo connect equality, and
    syncs Python latch state so logs match what the viewer shows.
    """
    from train_mimic.tasks.climbing.config.robot import HAND_INDEX

    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    hand_idx = HAND_INDEX[hand]
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    if rung_id is None:
        rung_id = int(runtime.sample.num_active[0].item()) - 1

    # Snap close to the latch site so the connect equality does not yank hard.
    place_robot_hand_near_rung(env, hand=hand, rung_id=rung_id, separation_m=0.002)
    env.sim.forward()

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    rung_ids = torch.full((env.num_envs,), rung_id, device=env.device, dtype=torch.int64)
    site_ids, site_ok = latch._select_nearest_site(env_ids, hand_idx, rung_ids)
    if not bool(torch.all(site_ok).item()):
        # Fallback: middle site on the rung (always valid in compiled topology).
        mid = latch.ladder_cfg.sites_per_rung // 2
        site_ids = torch.full((env.num_envs,), mid, device=env.device, dtype=torch.int64)

    for local_idx in range(env.num_envs):
        eid = env_ids[local_idx : local_idx + 1]
        sid = int(site_ids[local_idx].item())
        eq_id = int(latch.topology.eq_ids[hand_idx, rung_id, sid].item())
        if eq_id < 0:
            raise RuntimeError(
                f"No latch equality for hand={hand!r} rung={rung_id} site={sid}."
            )
        latch.backend.set_active(eid, hand_idx, eq_id, active=True)
        latch.state.attached[eid, hand_idx] = True
        latch.state.rung_id[eid, hand_idx] = rung_id
        latch.state.site_id[eid, hand_idx] = sid
        latch.state.active_eq_id[eid, hand_idx] = eq_id

    # Do not set attach_event: this is a debug force-latch for success demos,
    # not a payable higher-attachment transition.
    latch.state.invalid_attach_request[:, hand_idx] = False
    env.sim.forward()


def apply_reward_script_phase(
    env: ManagerBasedRlEnv,
    phase: RewardScriptPhase,
    *,
    phase_step: int,
    latch_cfg: LatchConfig,
    script_complete: bool = False,
) -> None:
    del latch_cfg
    if script_complete or phase.motion_mode == "hold":
        _zero_root_velocity(env)
        return
    if phase.seed_demo_attach and phase_step == 0:
        # Attach to the top active rung so ladder-relative success can fire.
        _seed_demo_attachment(env, hand=phase.hand, rung_id=None)
    if phase.motion_mode == "slide_robot_to_rung":
        nudge_robot_toward_rung(
            env,
            hand=phase.hand,
            rung_id=phase.rung_id,
            separation_m=phase.separation_m,
            step_m=phase.approach_step_m,
        )
        return
    if phase.motion_mode == "raise_pelvis":
        per_step = phase.pelvis_delta_l / max(phase.steps, 1)
        nudge_robot_along_ladder_up(env, step_l=per_step)
        return
    if phase.motion_mode == "invalid_attach":
        nudge_robot_away_from_ladder(env, step_m=0.05)
        return


def print_reward_breakdown(env: ManagerBasedRlEnv, *, header: str, env_idx: int = 0) -> None:
    """Print cached per-term rewards from the last env step (does not re-evaluate terms)."""
    manager = env.reward_manager
    dt = env.step_dt
    scale_dt = getattr(env.cfg, "scale_rewards_by_dt", True)
    pelvis_h = float(body_ladder_relative_height(env, body_name="pelvis")[env_idx].item())
    print(header)
    print(f"  pelvis_height_l={pelvis_h:.4f}")
    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    attached = latch.state.attached[env_idx].detach().cpu().tolist()
    print(f"  hand_attached={attached}")
    total = 0.0
    for idx, name in enumerate(manager.active_terms):
        if name.startswith("_"):
            continue
        term_cfg = manager.get_term_cfg(name)
        rate = float(manager._step_reward[env_idx, idx].item())
        raw_val = rate / float(term_cfg.weight) if term_cfg.weight != 0.0 else 0.0
        step_contrib = rate * dt if scale_dt else rate
        total += step_contrib
        print(
            f"  {name}: raw={raw_val:+.5f} weight={term_cfg.weight:+.4g} "
            f"step={step_contrib:+.5f}"
        )
    print(f"  total_step_reward={total:+.5f}")


def print_episode_metrics(env: ManagerBasedRlEnv, *, header: str, env_idx: int = 0) -> None:
    state = ClimbRewardState.get(env)
    current_h = float(body_ladder_relative_height(env, body_name="pelvis")[env_idx].item())
    goal_h = float(success_goal_height_l(env, pelvis_clearance_below_top_l=0.15)[env_idx].item())
    metrics = getattr(env, "metrics_manager", None)
    max_h = float(state.max_rewarded_pelvis_height_l[env_idx].item())
    hand_frac = 0.0
    if metrics is not None and "max_pelvis_height_l" in metrics.active_terms:
        idx = metrics.active_terms.index("max_pelvis_height_l")
        max_h = float(metrics._step_values[env_idx, idx].item())
    if metrics is not None and "hand_contact_fraction" in metrics.active_terms:
        idx = metrics.active_terms.index("hand_contact_fraction")
        # Per-step indicator; episode mean is only finalized on reset.
        hand_frac = float(metrics._step_values[env_idx, idx].item())
    print(header)
    print(f"  max_pelvis_height_l={max_h:.4f}")
    print(f"  current_pelvis_height_l={current_h:.4f}")
    print(f"  success_goal_height_l={goal_h:.4f}")
    print(f"  valid_higher_attachments={int(state.valid_higher_attachment_count[env_idx].item())}")
    print(f"  invalid_latch_count={int(state.invalid_latch_count[env_idx].item())}")
    print(f"  hand_contact_indicator={hand_frac:.3f}")
    print(f"  time_to_success={int(state.time_to_success_steps[env_idx].item())}")


def build_reward_action(
    env: ManagerBasedRlEnv,
    *,
    latch_left: float,
    latch_right: float,
) -> torch.Tensor:
    return build_latch_action(env, latch_left=latch_left, latch_right=latch_right)
