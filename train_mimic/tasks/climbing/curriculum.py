"""Curriculum runtime helpers: reset-time sampling and physics application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from train_mimic.tasks.climbing.config.curriculum import CurriculumConfig
from train_mimic.tasks.climbing.config.hold_pose import HoldPoseConfig
from train_mimic.tasks.climbing.debug_hold_pose import (
    apply_hold_pose_to_env_ids,
    seed_initial_hand_latches,
)
from train_mimic.tasks.climbing.ladder.hold_ik import HoldPoseSolution, solve_hold_pose_ik
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.ladder.state import LadderRuntime

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

HOLD_POSE_CACHE_KEY = "climb_hold_pose_solution_cache"
LADDER_PHYSICS_CACHE_KEY = "climb_ladder_physics_state"


@dataclass
class _LadderPhysicsState:
    """Per-env sampled rung sliding friction applied at reset."""

    rung_sliding_friction: torch.Tensor


def _resolve_env_ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    return env_ids


def _per_env_generator(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    *,
    base_seed: int,
) -> torch.Generator:
    gen = torch.Generator(device=env.device)
    gen.manual_seed(int(base_seed) + int(env_ids[0].item()) * 9973)
    return gen


def _sample_rung_sliding_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    curriculum: CurriculumConfig,
) -> torch.Tensor:
    lo, hi = curriculum.rung_sliding_friction_range
    if not curriculum.randomize_rung_physics or lo == hi:
        value = float(curriculum.rung_friction[0])
        return torch.full((env_ids.numel(),), value, device=env.device, dtype=torch.float32)
    gen = _per_env_generator(env, env_ids, base_seed=int(env.cfg.seed))
    return torch.empty(env_ids.numel(), device=env.device, dtype=torch.float32).uniform_(
        lo, hi, generator=gen
    )


def _sample_capture_radius(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    curriculum: CurriculumConfig,
) -> torch.Tensor:
    lo, hi = curriculum.latch_capture_radius_range
    if not curriculum.randomize_latch_capture or lo == hi:
        value = float(curriculum.latch_capture_radius)
        return torch.full((env_ids.numel(),), value, device=env.device, dtype=torch.float32)
    gen = _per_env_generator(env, env_ids, base_seed=int(env.cfg.seed) + 17)
    return torch.empty(env_ids.numel(), device=env.device, dtype=torch.float32).uniform_(
        lo, hi, generator=gen
    )


def apply_ladder_physics_randomization(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    curriculum_cfg: CurriculumConfig,
) -> None:
    """Sample and apply rung sliding friction for selected environments."""
    if not curriculum_cfg.randomize_rung_physics:
        return

    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    runtime = LadderRuntime.get(env)
    friction = _sample_rung_sliding_friction(env, env_ids, curriculum_cfg)
    sim = env.sim
    for local_i, env_idx in enumerate(env_ids.tolist()):
        mu = float(friction[local_i].item())
        friction_vec = list(curriculum_cfg.rung_friction)
        friction_vec[0] = mu
        for geom_id in runtime.topology.rung_geom_ids:
            sim.model.geom_friction[int(env_idx), geom_id] = torch.tensor(
                friction_vec,
                device=env.device,
                dtype=torch.float32,
            )

    state = env.extras.get(LADDER_PHYSICS_CACHE_KEY)
    if state is None:
        state = _LadderPhysicsState(
            rung_sliding_friction=torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        )
        env.extras[LADDER_PHYSICS_CACHE_KEY] = state
    state.rung_sliding_friction[env_ids] = friction


def apply_latch_capture_randomization(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    curriculum_cfg: CurriculumConfig,
) -> None:
    """Sample per-env latch capture radius when the curriculum axis is enabled."""
    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    latch = env.extras.get(ClimbLatchState.EXTRA_KEY)
    if latch is None:
        return
    latch._ensure_state()
    radii = _sample_capture_radius(env, env_ids, curriculum_cfg)
    latch.state.capture_radius[env_ids] = radii


def _get_or_solve_hold_pose(
    env: ManagerBasedRlEnv,
    hold_cfg: HoldPoseConfig,
    *,
    env_idx: int,
) -> HoldPoseSolution:
    cache: dict[int, HoldPoseSolution] = env.extras.setdefault(HOLD_POSE_CACHE_KEY, {})
    if env_idx not in cache:
        cache[env_idx] = solve_hold_pose_ik(env, hold_cfg, env_idx=env_idx)
    return cache[env_idx]


def reset_climb_initial_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    curriculum_cfg: CurriculumConfig,
) -> None:
    """Apply assisted hold IK pose, optional noise, and initial hand attachments."""
    if not curriculum_cfg.use_hold_pose_reset:
        return

    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    hold_cfg = curriculum_cfg.resolved_hold_pose
    per_env_ik = (
        curriculum_cfg.randomize_ladder_pose
        or curriculum_cfg.randomize_rung_count
        or curriculum_cfg.randomize_spacing
    )

    for local_i, env_idx in enumerate(env_ids.tolist()):
        if per_env_ik:
            solution = solve_hold_pose_ik(env, hold_cfg, env_idx=int(env_idx))
        else:
            solution = _get_or_solve_hold_pose(env, hold_cfg, env_idx=0)

        target_ids = env_ids[local_i : local_i + 1]
        pos_noise = curriculum_cfg.root_pos_noise_std
        yaw_noise = curriculum_cfg.root_yaw_noise_rad
        joint_noise = curriculum_cfg.joint_pos_noise_std
        if not curriculum_cfg.randomize_initial_pose:
            pos_noise = (0.0, 0.0, 0.0)
            yaw_noise = 0.0
            joint_noise = 0.0

        gen = _per_env_generator(env, target_ids, base_seed=int(env.cfg.seed) + int(env_idx) * 7919)
        apply_hold_pose_to_env_ids(
            env,
            solution,
            target_ids,
            spawn_root_lift_m=hold_cfg.spawn_root_lift_m,
            root_pos_noise_std=pos_noise,
            root_yaw_noise_rad=yaw_noise,
            joint_pos_noise_std=joint_noise,
            generator=gen,
        )

        attach_gen = _per_env_generator(
            env, target_ids, base_seed=int(env.cfg.seed) + int(env_idx) * 6151
        )
        seed_initial_hand_latches(
            env,
            hold_cfg,
            curriculum_cfg,
            env_ids=target_ids,
            generator=attach_gen,
        )

    env.sim.forward()


def invalidate_hold_pose_cache(env: ManagerBasedRlEnv) -> None:
    """Drop cached IK solutions after ladder layout changes."""
    env.extras.pop(HOLD_POSE_CACHE_KEY, None)
