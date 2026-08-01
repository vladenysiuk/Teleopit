"""Debug helpers for hand–ladder contact inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from train_mimic.tasks.climbing.config.robot import (
    HAND_INDEX,
    HAND_NAMES,
    HAND_POINT_CONTYPE,
    LADDER_CONAFFINITY,
    LADDER_CONTYPE,
    LEFT_HAND_POINT,
    RIGHT_HAND_POINT,
)
from train_mimic.tasks.climbing.ladder.state import (
    LadderRuntime,
    rung_local_centers_from_sample,
    _quat_apply,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# Shift probe center outward along the rung so a 0.45 m cylinder covering one
# hand does not also intersect the opposite hand (~0.25 m apart on G1 stand).
_PROBE_LATERAL_OFFSET_FRAC = 0.35

# Modest along-rung offset when driving the *robot* to a fixed ladder rung.
# Left hand → +Y, right hand → −Y (G1 standing convention). Do not reuse the
# large probe offset — that exists only so a teleported cylinder hits one hand.
_ROBOT_REACH_LATERAL_M = 0.08

# World-up pull component during latched pull demo (keeps rung clear of ladder).
_PULL_UP_STEP_FRAC = 0.35

# Allow vertical root correction this far in front of the ladder frame (m).
# Lateral correction stays gated tighter to avoid rail collisions.
_VERTICAL_CORRECTION_FRAME_MARGIN_M = 0.55
_LATERAL_BAY_FRAME_MARGIN_M = 0.12


def _normalized(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1e-8)


def ladder_forward_direction_w(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Unit ladder-frame +X expressed in world (approach from in front of the robot)."""
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    local_x = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    local_x[:, 0] = 1.0
    return _normalized(_quat_apply(runtime.sample.frame_quat, local_x))


def ladder_lateral_direction_w(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Unit ladder-frame +Y expressed in world (along the rung)."""
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    local_y = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    local_y[:, 1] = 1.0
    return _normalized(_quat_apply(runtime.sample.frame_quat, local_y))


def _world_to_frame_local(
    env: ManagerBasedRlEnv,
    world_pos: torch.Tensor,
    *,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    frame_pos = runtime.sample.frame_pos[env_ids]
    frame_quat = runtime.sample.frame_quat[env_ids]
    return quat_apply_inverse(frame_quat, world_pos - frame_pos)


def _set_rung_center_local(
    env: ManagerBasedRlEnv,
    rung_id: int,
    local_center: torch.Tensor,
    *,
    env_ids: torch.Tensor,
) -> None:
    runtime = LadderRuntime.get(env)
    geom_id = runtime.topology.rung_geom_ids[rung_id]
    env.sim.model.geom_pos[env_ids, geom_id] = local_center
    site_x = -runtime.topology.site_normal_offset
    for site_idx, site_id in enumerate(runtime.topology.rung_site_ids[rung_id]):
        site_pos = local_center.clone()
        site_pos[:, 0] = local_center[:, 0] + site_x
        site_pos[:, 1] = runtime.topology.site_y_offsets[site_idx]
        env.sim.model.site_pos[env_ids, site_id] = site_pos


def _set_rung_center_world(
    env: ManagerBasedRlEnv,
    rung_id: int,
    world_center: torch.Tensor,
    *,
    env_ids: torch.Tensor,
) -> None:
    _set_rung_center_local(
        env,
        rung_id,
        _world_to_frame_local(env, world_center, env_ids=env_ids),
        env_ids=env_ids,
    )


def get_rung_center_pos_w(
    env: ManagerBasedRlEnv,
    rung_id: int,
    *,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Live rung collision-center in world frame, shape ``[N, 3]``."""
    runtime = LadderRuntime.get(env)
    geom_id = runtime.topology.rung_geom_ids[rung_id]
    pos = env.sim.data.geom_xpos[:, geom_id]
    if env_ids is None:
        return pos
    return pos[env_ids]


def get_rung_mocap_pos_w(
    env: ManagerBasedRlEnv,
    rung_id: int,
    *,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward-compatible alias for ``get_rung_center_pos_w``."""
    return get_rung_center_pos_w(env, rung_id, env_ids=env_ids)


def set_probe_rung_hand_only_collision(env: ManagerBasedRlEnv, rung_id: int) -> None:
    """Restrict one rung geom so it collides with hand points only (not legs/body)."""
    runtime = LadderRuntime.get(env)
    geom_id = runtime.topology.rung_geom_ids[rung_id]
    # Warp/CPU model field used by the active simulator.
    env.sim.model.geom_conaffinity[geom_id] = HAND_POINT_CONTYPE
    env.sim.mj_model.geom_conaffinity[geom_id] = HAND_POINT_CONTYPE


def set_active_rungs_hand_only_collision(env: ManagerBasedRlEnv) -> None:
    """Hand-only collision for every active rung on the sampled ladder."""
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        return
    for rung_id in range(runtime.cfg.max_rungs):
        if bool(runtime.sample.active_mask[0, rung_id].item()):
            set_probe_rung_hand_only_collision(env, rung_id)


def restore_rung_default_collision(env: ManagerBasedRlEnv, rung_id: int) -> None:
    """Restore a rung geom to the default ladder collision groups (body + hand points)."""
    runtime = LadderRuntime.get(env)
    geom_id = runtime.topology.rung_geom_ids[rung_id]
    env.sim.model.geom_conaffinity[geom_id] = LADDER_CONAFFINITY
    env.sim.model.geom_contype[geom_id] = LADDER_CONTYPE
    env.sim.mj_model.geom_conaffinity[geom_id] = LADDER_CONAFFINITY
    env.sim.mj_model.geom_contype[geom_id] = LADDER_CONTYPE


def restore_active_rungs_default_collision(env: ManagerBasedRlEnv) -> None:
    """Default ladder collision for every active rung (robot body + hand points)."""
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        return
    for rung_id in range(runtime.cfg.max_rungs):
        if bool(runtime.sample.active_mask[0, rung_id].item()):
            restore_rung_default_collision(env, rung_id)


def get_hand_pos_w(env: ManagerBasedRlEnv, hand: str) -> torch.Tensor:
    """World position of a hand contact geom, shape ``[N, 3]``."""
    if hand not in HAND_INDEX:
        raise ValueError(f"hand must be one of {HAND_NAMES}, got {hand!r}.")

    robot = env.scene["robot"]
    point_name = LEFT_HAND_POINT if hand == "left" else RIGHT_HAND_POINT
    geom_ids, _ = robot.find_geoms([point_name], preserve_order=True)
    return robot.data.geom_pos_w[:, geom_ids[0]]


def hand_target_for_rung_contact_w(
    env: ManagerBasedRlEnv,
    *,
    hand: str,
    rung_id: int,
    separation_m: float,
    lateral_mode: str = "reach",
) -> torch.Tensor:
    """World hand position that places ``hand`` on ``rung_id`` with ``separation_m``.

    ``separation_m`` is the hand-centre → rung-centre gap along ladder +X
    (positive = hand on the robot side of the rung).

    ``lateral_mode``:
      - ``"reach"``: small ±Y offset for robot approach to a fixed ladder rung
      - ``"probe_inverse"``: large offset matching ``probe_target_pos_w`` (tests)
    """
    if hand not in HAND_INDEX:
        raise ValueError(f"hand must be one of {HAND_NAMES}, got {hand!r}.")
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")

    rung_pos = runtime.sample.rung_pos_w[:, rung_id]
    forward = ladder_forward_direction_w(env)
    lateral = ladder_lateral_direction_w(env)
    # G1: left hand is +Y, right hand is −Y in the upright ladder frame.
    side = 1.0 if hand == "left" else -1.0
    if lateral_mode == "probe_inverse":
        # Inverse of probe_target_pos_w (probe uses +lateral*side from the hand).
        lateral_shift = side * _PROBE_LATERAL_OFFSET_FRAC * runtime.cfg.rung_length
        return rung_pos - forward * separation_m - lateral * lateral_shift
    if lateral_mode == "reach":
        return (
            rung_pos
            - forward * separation_m
            + lateral * (side * _ROBOT_REACH_LATERAL_M)
        )
    raise ValueError(f"Unknown lateral_mode={lateral_mode!r}.")


def _sim_physics_dt(env: ManagerBasedRlEnv) -> float:
    return float(env.cfg.sim.mujoco.timestep)


def _env_step_dt(env: ManagerBasedRlEnv) -> float:
    return float(env.step_dt)


def _zero_root_velocity(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    robot = env.scene["robot"]
    zero = torch.zeros(env_ids.numel(), 6, device=env.device, dtype=torch.float32)
    robot.write_root_link_velocity_to_sim(zero, env_ids=env_ids)


def _clamp_root_velocity_against_rails(
    env: ManagerBasedRlEnv,
    *,
    root_pos: torch.Tensor,
    lin_vel: torch.Tensor,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """Zero lateral root velocity that would drive the body through a side rail."""
    runtime = LadderRuntime.get(env)
    cfg = runtime.cfg
    margin = cfg.rail_radius + 0.04
    left_limit = -cfg.rail_half_width - margin
    right_limit = cfg.rail_half_width + margin
    y = root_pos[:, 1]
    vel_y = lin_vel[:, 1].clone()
    outside_left = y <= left_limit
    outside_right = y >= right_limit
    vel_y = torch.where(outside_left & (vel_y > 0.0), torch.zeros_like(vel_y), vel_y)
    vel_y = torch.where(outside_right & (vel_y < 0.0), torch.zeros_like(vel_y), vel_y)
    lin_vel = lin_vel.clone()
    lin_vel[:, 1] = vel_y
    return lin_vel


def _write_root_linear_velocity_toward(
    env: ManagerBasedRlEnv,
    *,
    direction_w: torch.Tensor,
    step_m: float,
    env_ids: torch.Tensor,
) -> None:
    """Set root linear velocity for one physics substep (contact can block motion)."""
    robot = env.scene["robot"]
    dt = _sim_physics_dt(env)
    speed = step_m / dt
    direction = _normalized(direction_w)
    lin_vel = direction * speed
    ang_vel = robot.data.root_link_vel_w[env_ids, 3:6]
    robot.write_root_link_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)


def place_robot_hand_near_rung(
    env: ManagerBasedRlEnv,
    *,
    hand: str,
    rung_id: int,
    separation_m: float,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Teleport the root so ``hand`` sits near a fixed rung (latch debug init only).

    Does not run during per-step approach — that still uses velocity drive so
    contacts can block motion. Use this once to skip the long collapse-prone walk.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)

    robot = env.scene["robot"]
    hand_pos = get_hand_pos_w(env, hand)[env_ids]
    target = hand_target_for_rung_contact_w(
        env,
        hand=hand,
        rung_id=rung_id,
        separation_m=separation_m,
        lateral_mode="reach",
    )[env_ids]
    delta = target - hand_pos
    root_pos = robot.data.root_link_pos_w[env_ids] + delta
    root_quat = robot.data.root_link_quat_w[env_ids]
    robot.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(
        torch.zeros(env_ids.numel(), 6, device=env.device, dtype=torch.float32),
        env_ids=env_ids,
    )


def nudge_robot_toward_rung(
    env: ManagerBasedRlEnv,
    *,
    hand: str,
    rung_id: int,
    separation_m: float,
    step_m: float = 0.012,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Drive the robot root toward a fixed ladder rung via velocity (not teleport).

    Root motion is decomposed in the ladder frame:
    - forward always;
    - vertical once within ``_VERTICAL_CORRECTION_FRAME_MARGIN_M`` of the frame
      (keeps hand height on the rung without waiting for full bay entry);
    - lateral only inside the bay (avoids cutting through side rails).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)

    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    robot = env.scene["robot"]
    hand_pos = get_hand_pos_w(env, hand)[env_ids]
    target = hand_target_for_rung_contact_w(
        env,
        hand=hand,
        rung_id=rung_id,
        separation_m=separation_m,
        lateral_mode="reach",
    )[env_ids]
    delta = target - hand_pos

    forward = ladder_forward_direction_w(env)[env_ids]
    lateral = ladder_lateral_direction_w(env)[env_ids]
    up = torch.zeros_like(forward)
    up[:, 2] = 1.0

    forward_mag = (delta * forward).sum(-1, keepdim=True)
    lateral_mag = (delta * lateral).sum(-1, keepdim=True)
    vertical_mag = (delta * up).sum(-1, keepdim=True)

    root = robot.data.root_link_pos_w[env_ids]
    frame_x = runtime.sample.frame_pos[env_ids, 0:1]
    allow_vertical = (root[:, 0:1] >= frame_x - _VERTICAL_CORRECTION_FRAME_MARGIN_M).to(delta.dtype)
    in_bay = (root[:, 0:1] >= frame_x - _LATERAL_BAY_FRAME_MARGIN_M).to(delta.dtype)
    direction = (
        forward * forward_mag
        + lateral * lateral_mag * in_bay
        + up * vertical_mag * allow_vertical
    )
    dist = torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
    direction = direction / dist
    step = torch.minimum(torch.linalg.norm(delta, dim=-1, keepdim=True), torch.full_like(dist, step_m))

    dt = _sim_physics_dt(env)
    speed = step.squeeze(-1) / dt
    lin_vel = direction * speed.unsqueeze(-1)
    lin_vel = _clamp_root_velocity_against_rails(
        env, root_pos=root, lin_vel=lin_vel, env_ids=env_ids
    )
    ang_vel = robot.data.root_link_vel_w[env_ids, 3:6]
    robot.write_root_link_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)


def nudge_robot_away_from_ladder(
    env: ManagerBasedRlEnv,
    *,
    step_m: float = 0.008,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Drive the robot root away from the ladder (−ladder X / back toward spawn).

    The robot approaches from −ladder X, so "pull away" is opposite the approach
    direction. Using +ladder X was incorrectly shoving the body into the rungs.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)

    robot = env.scene["robot"]
    forward = ladder_forward_direction_w(env)[env_ids]
    # Away from the ladder: back toward the robot spawn (−ladder forward).
    away = -forward
    # Small upward component keeps the latched hand from dragging into lower rungs.
    up = torch.zeros_like(away)
    up[:, 2] = 1.0
    direction = _normalized(away + up * _PULL_UP_STEP_FRAC)
    lin_vel = direction * (step_m / _sim_physics_dt(env))
    root = robot.data.root_link_pos_w[env_ids]
    lin_vel = _clamp_root_velocity_against_rails(
        env, root_pos=root, lin_vel=lin_vel, env_ids=env_ids
    )
    ang_vel = robot.data.root_link_vel_w[env_ids, 3:6]
    robot.write_root_link_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)


def nudge_robot_along_ladder_up(
    env: ManagerBasedRlEnv,
    *,
    step_l: float,
    max_speed: float = 0.35,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Drive the robot root upward along the ladder +Z axis by ``step_l`` this env step."""
    from train_mimic.tasks.climbing.ladder.relative_rungs import ladder_upward_axis_w

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    upward = _normalized(ladder_upward_axis_w(runtime.sample.frame_quat)[env_ids])
    speed = min(step_l / _env_step_dt(env), max_speed)
    lin_vel = upward * speed
    robot = env.scene["robot"]
    ang_vel = robot.data.root_link_vel_w[env_ids, 3:6]
    robot.write_root_link_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)


def restore_rung_to_ladder_sample(
    env: ManagerBasedRlEnv,
    rung_id: int,
    *,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Put a rung back on its sampled ladder pose."""
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    local_centers = rung_local_centers_from_sample(runtime.sample, runtime.cfg)
    _set_rung_center_local(env, rung_id, local_centers[env_ids, rung_id], env_ids=env_ids)


def park_probe_rung(env: ManagerBasedRlEnv, rung_id: int, *, env_ids: torch.Tensor | None = None) -> None:
    """Move a probe rung to the inactive parking pose (off-scene)."""
    runtime = LadderRuntime.get(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    inactive = torch.tensor(runtime.cfg.inactive_rung_pose, device=env.device, dtype=torch.float32)
    inactive = inactive.unsqueeze(0).expand(env_ids.numel(), 3)
    _set_rung_center_local(env, rung_id, inactive, env_ids=env_ids)


def nudge_probe_rung_away_from_ladder(
    env: ManagerBasedRlEnv,
    rung_id: int,
    *,
    step_m: float,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Translate a latched probe rung away from the ladder (+world up), not toward it."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    world_center = get_rung_center_pos_w(env, rung_id, env_ids=env_ids)
    forward = ladder_forward_direction_w(env)[env_ids]
    up = torch.zeros(env_ids.numel(), 3, device=env.device, dtype=torch.float32)
    up[:, 2] = 1.0
    delta = (-forward * step_m) + (up * step_m * _PULL_UP_STEP_FRAC)
    _set_rung_center_world(env, rung_id, world_center + delta, env_ids=env_ids)


def restore_all_active_rungs_to_ladder(env: ManagerBasedRlEnv) -> None:
    """Restore every active rung to its sampled ladder pose."""
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        return
    runtime.apply()


def probe_target_pos_w(
    env: ManagerBasedRlEnv,
    *,
    hand: str,
    separation_m: float,
) -> torch.Tensor:
    """World position for a probe rung center near ``hand`` (shape ``[N, 3]``)."""
    if hand not in HAND_INDEX:
        raise ValueError(f"hand must be one of {HAND_NAMES}, got {hand!r}.")

    runtime = LadderRuntime.get(env)
    robot = env.scene["robot"]
    point_name = LEFT_HAND_POINT if hand == "left" else RIGHT_HAND_POINT
    geom_ids, _ = robot.find_geoms([point_name], preserve_order=True)
    hand_pos = robot.data.geom_pos_w[:, geom_ids[0]]

    forward = ladder_forward_direction_w(env)
    lateral = ladder_lateral_direction_w(env)
    side = 1.0 if hand == "left" else -1.0
    lateral_shift = side * _PROBE_LATERAL_OFFSET_FRAC * runtime.cfg.rung_length
    return hand_pos + forward * separation_m + lateral * lateral_shift


def slide_probe_rung_toward_hand(
    env: ManagerBasedRlEnv,
    *,
    hand: str,
    rung_id: int,
    separation_m: float,
    alpha: float,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Interpolate a rung mocap from its ladder sample pose toward the hand target.

    ``alpha=0`` keeps the rung on the ladder; ``alpha=1`` places it at the hand
    contact pose for ``separation_m``.
    """
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)

    alpha = float(max(0.0, min(1.0, alpha)))
    ladder_pos = runtime.sample.rung_pos_w
    hand_target = probe_target_pos_w(env, hand=hand, separation_m=separation_m)
    target_pos = (1.0 - alpha) * ladder_pos[:, rung_id] + alpha * hand_target
    _set_rung_center_world(env, rung_id, target_pos[env_ids], env_ids=env_ids)


def place_probe_rung_near_hand(
    env: ManagerBasedRlEnv,
    *,
    hand: str,
    rung_id: int,
    separation_m: float,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Move one mocap rung in front of a hand point for contact probing.

    Approach is along ladder-frame +X (usually world +X for an upright ladder),
    so the cylinder stays clear of the torso/legs. The center is also shifted
    outward along the rung axis so the long cylinder does not also hit the
    opposite hand.
    """
    if hand not in HAND_INDEX:
        raise ValueError(f"hand must be one of {HAND_NAMES}, got {hand!r}.")

    runtime = LadderRuntime.get(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)

    target_pos = probe_target_pos_w(env, hand=hand, separation_m=separation_m)
    _set_rung_center_world(env, rung_id, target_pos[env_ids], env_ids=env_ids)


def rung_mocap_pos_numpy(
    env: ManagerBasedRlEnv,
    rung_id: int,
    env_idx: int,
) -> np.ndarray:
    """Single-env mocap rung center as float64 numpy (for debug overlays)."""
    pos = get_rung_mocap_pos_w(env, rung_id)[env_idx].detach().cpu().numpy()
    return np.asarray(pos, dtype=np.float64)
