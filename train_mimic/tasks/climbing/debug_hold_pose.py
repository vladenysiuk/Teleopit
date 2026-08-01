"""Debug helpers for Stage 6 static hold-pose validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from train_mimic.tasks.climbing.config.hold_pose import HoldPoseConfig
from train_mimic.tasks.climbing.config.robot import HAND_INDEX
from train_mimic.tasks.climbing.debug_latch import build_latch_action
from train_mimic.tasks.climbing.ladder.hold_ik import HoldPoseSolution, solve_hold_pose_ik
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.ladder.state import LadderRuntime, _quat_apply
from train_mimic.tasks.climbing.mdp.observations import action_term_slice

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

FailureClass = Literal[
    "ok",
    "foot_slip",
    "foot_rolling_off",
    "infeasible_ik",
    "torque_saturation",
    "poor_pd_gains",
    "latch_compliance",
    "collision_penetration",
    "solver_capacity",
    "nan_or_nonfinite",
    "pelvis_drop",
    "lost_foot_contact",
]


@dataclass
class HoldPoseMetrics:
    """Time-series logger for a brief gravity hold."""

    dt: float
    pelvis_pos_w: list[np.ndarray] = field(default_factory=list)
    pelvis_drop_m: list[float] = field(default_factory=list)
    hand_attach_error_m: list[np.ndarray] = field(default_factory=list)
    foot_axis_dist_m: list[np.ndarray] = field(default_factory=list)
    foot_slip_m: list[np.ndarray] = field(default_factory=list)
    foot_in_contact: list[np.ndarray] = field(default_factory=list)
    joint_torque: list[np.ndarray] = field(default_factory=list)
    torque_saturated: list[np.ndarray] = field(default_factory=list)
    contact_vertical_force: list[float] = field(default_factory=list)
    nacon: list[int] = field(default_factory=list)
    nefc: list[int] = field(default_factory=list)
    nonfinite: list[bool] = field(default_factory=list)
    _pelvis0: np.ndarray | None = None
    _foot0: np.ndarray | None = None

    def record(self, sample: dict) -> None:
        pelvis = np.asarray(sample["pelvis_pos_w"], dtype=np.float64)
        if self._pelvis0 is None:
            self._pelvis0 = pelvis.copy()
        if self._foot0 is None:
            self._foot0 = np.asarray(sample["foot_pos_w"], dtype=np.float64).copy()

        foot_pos = np.asarray(sample["foot_pos_w"], dtype=np.float64)
        # Tangential slip along the rung axis (ladder Y). Vertical sag of a
        # briefly unloaded foot is not counted as cylinder sliding.
        axis = np.asarray(sample["rung_axis_w"], dtype=np.float64)
        delta = foot_pos - self._foot0
        tangential = np.abs(delta @ axis)
        self.pelvis_pos_w.append(pelvis)
        self.pelvis_drop_m.append(float(self._pelvis0[2] - pelvis[2]))
        self.hand_attach_error_m.append(np.asarray(sample["hand_attach_error_m"], dtype=np.float64))
        self.foot_axis_dist_m.append(np.asarray(sample["foot_axis_dist_m"], dtype=np.float64))
        self.foot_slip_m.append(tangential)
        self.foot_in_contact.append(np.asarray(sample["foot_in_contact"], dtype=bool))
        self.joint_torque.append(np.asarray(sample["joint_torque"], dtype=np.float64))
        self.torque_saturated.append(np.asarray(sample["torque_saturated"], dtype=bool))
        self.contact_vertical_force.append(float(sample["contact_vertical_force"]))
        self.nacon.append(int(sample["nacon"]))
        self.nefc.append(int(sample["nefc"]))
        self.nonfinite.append(bool(sample["nonfinite"]))

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "pelvis_pos_w": np.asarray(self.pelvis_pos_w, dtype=np.float64),
            "pelvis_drop_m": np.asarray(self.pelvis_drop_m, dtype=np.float64),
            "hand_attach_error_m": np.asarray(self.hand_attach_error_m, dtype=np.float64),
            "foot_axis_dist_m": np.asarray(self.foot_axis_dist_m, dtype=np.float64),
            "foot_slip_m": np.asarray(self.foot_slip_m, dtype=np.float64),
            "foot_in_contact": np.asarray(self.foot_in_contact, dtype=bool),
            "joint_torque": np.asarray(self.joint_torque, dtype=np.float64),
            "torque_saturated": np.asarray(self.torque_saturated, dtype=bool),
            "contact_vertical_force": np.asarray(self.contact_vertical_force, dtype=np.float64),
            "nacon": np.asarray(self.nacon, dtype=np.int64),
            "nefc": np.asarray(self.nefc, dtype=np.int64),
            "nonfinite": np.asarray(self.nonfinite, dtype=bool),
            "dt": np.asarray([self.dt], dtype=np.float64),
        }

    def summary(self) -> dict[str, float | bool | str]:
        if not self.pelvis_drop_m:
            return {"steps": 0, "failure_class": "nan_or_nonfinite"}
        drop = float(np.max(self.pelvis_drop_m))
        slip = np.asarray(self.foot_slip_m, dtype=np.float64)
        contact = np.asarray(self.foot_in_contact, dtype=bool)
        hand_err = np.asarray(self.hand_attach_error_m, dtype=np.float64)
        sat = np.asarray(self.torque_saturated, dtype=bool)
        # Major leg joints: hips + knees (indices 0,3,6,9 in 29-dof ordering).
        leg_idx = np.array([0, 3, 6, 9], dtype=np.int64)
        leg_sat_frac = float(sat[:, leg_idx].mean()) if sat.size else 1.0
        both_frac = float(contact.mean()) if contact.size else 0.0
        any_frac = float(contact.any(axis=1).mean()) if contact.size else 0.0
        return {
            "steps": len(self.pelvis_drop_m),
            "duration_s": len(self.pelvis_drop_m) * self.dt,
            "max_pelvis_drop_m": drop,
            "max_foot_slip_m": float(np.max(slip)) if slip.size else float("inf"),
            "mean_foot_contact_fraction": both_frac,
            "any_foot_contact_fraction": any_frac,
            "max_hand_attach_error_m": float(np.max(hand_err)) if hand_err.size else float("inf"),
            "leg_saturation_fraction": leg_sat_frac,
            "max_nacon": int(np.max(self.nacon)) if self.nacon else 0,
            "max_nefc": int(np.max(self.nefc)) if self.nefc else 0,
            "any_nonfinite": bool(np.any(self.nonfinite)),
        }


def classify_hold_failure(
    summary: dict[str, float | bool | str],
    cfg: HoldPoseConfig,
    *,
    ik_ok: bool,
) -> FailureClass:
    """Classify hold failure before changing physics parameters."""
    if not ik_ok:
        return "infeasible_ik"
    if bool(summary.get("any_nonfinite", False)):
        return "nan_or_nonfinite"
    if int(summary.get("max_nacon", 0)) >= 700 or int(summary.get("max_nefc", 0)) >= 2400:
        return "solver_capacity"
    if float(summary.get("max_hand_attach_error_m", 0.0)) > cfg.max_hand_attach_error_m:
        return "latch_compliance"
    if float(summary.get("max_pelvis_drop_m", 0.0)) > cfg.max_pelvis_drop_m:
        return "pelvis_drop"
    if float(summary.get("max_foot_slip_m", 0.0)) > cfg.max_foot_slip_m:
        return "foot_slip"
    if (
        float(summary.get("mean_foot_contact_fraction", 0.0)) < cfg.min_foot_contact_fraction
        or float(summary.get("any_foot_contact_fraction", 0.0)) < cfg.min_any_foot_contact_fraction
    ):
        return "lost_foot_contact"
    if float(summary.get("leg_saturation_fraction", 0.0)) > cfg.max_leg_saturation_fraction:
        return "torque_saturation"
    return "ok"


def _seed_hand_latch(
    env: ManagerBasedRlEnv,
    *,
    hand: str,
    rung_id: int,
    site_id: int,
) -> None:
    """Force-activate one hand connect equality and sync Python latch state.

    Bypasses the one-step contact delay. Pose must already place the hand near
    the target site so the soft constraint does not yank.
    """
    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    hand_idx = HAND_INDEX[hand]
    eq_id = int(latch.topology.eq_ids[hand_idx, rung_id, site_id].item())
    if eq_id < 0:
        raise RuntimeError(f"Missing latch equality for {hand} rung={rung_id} site={site_id}")
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    latch.backend.set_active(env_ids, hand_idx, eq_id, active=True)
    latch.state.attached[env_ids, hand_idx] = True
    latch.state.rung_id[env_ids, hand_idx] = rung_id
    latch.state.site_id[env_ids, hand_idx] = site_id
    latch.state.active_eq_id[env_ids, hand_idx] = eq_id
    latch.state.attach_event[env_ids, hand_idx] = False
    latch.state.invalid_attach_request[env_ids, hand_idx] = False


def apply_hold_pose_to_env(
    env: ManagerBasedRlEnv,
    solution: HoldPoseSolution,
    *,
    env_idx: int = 0,
    spawn_root_lift_m: float = 0.0,
) -> None:
    """One-shot write of IK pose + zero velocity (never call per-step)."""
    robot = env.scene["robot"]
    n = env.num_envs
    device = env.device
    joint_pos = torch.zeros(n, solution.joint_pos.shape[0], device=device, dtype=torch.float32)
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[env_idx] = torch.as_tensor(solution.joint_pos, device=device, dtype=torch.float32)
    # Broadcast the same hold pose to all envs for the single-env debug path.
    if n == 1:
        joint_pos[:] = joint_pos[env_idx]
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    root = np.asarray(solution.root_pos, dtype=np.float64).copy()
    root[2] += float(spawn_root_lift_m)
    root_pos = torch.as_tensor(root, device=device, dtype=torch.float32).view(1, 3)
    root_quat = torch.as_tensor(solution.root_quat_wxyz, device=device, dtype=torch.float32).view(1, 4)
    if n != 1:
        root_pos = root_pos.expand(n, -1).clone()
        root_quat = root_quat.expand(n, -1).clone()
        root_pos[env_idx] = torch.as_tensor(root, device=device, dtype=torch.float32)
        root_quat[env_idx] = torch.as_tensor(solution.root_quat_wxyz, device=device, dtype=torch.float32)
    else:
        root_pos = root_pos.clone()
        root_quat = root_quat.clone()
    robot.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1))
    robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device, dtype=torch.float32))
    env.sim.forward()


def initialize_hold_pose_scene(
    env: ManagerBasedRlEnv,
    cfg: HoldPoseConfig,
    *,
    env_idx: int = 0,
) -> HoldPoseSolution:
    """Solve IK, apply pose once, and force-seed both hand latches."""
    solution = solve_hold_pose_ik(env, cfg, env_idx=env_idx)
    apply_hold_pose_to_env(
        env,
        solution,
        env_idx=env_idx,
        spawn_root_lift_m=cfg.spawn_root_lift_m,
    )
    _seed_hand_latch(
        env,
        hand="left",
        rung_id=solution.targets.left_hand_rung_id,
        site_id=solution.targets.left_hand_site_id,
    )
    _seed_hand_latch(
        env,
        hand="right",
        rung_id=solution.targets.right_hand_rung_id,
        site_id=solution.targets.right_hand_site_id,
    )
    env.sim.forward()
    return solution


def build_hold_action(
    env: ManagerBasedRlEnv,
    solution: HoldPoseSolution,
    *,
    latch_left: float = 0.0,
    latch_right: float = 0.0,
) -> torch.Tensor:
    """PD hold of the IK joint pose with neutral latch commands by default."""
    action = build_latch_action(env, latch_left=latch_left, latch_right=latch_right)
    joint_term = env.action_manager.get_term("joint_pos")
    scale = joint_term.scale
    offset = joint_term.offset
    q_des = torch.as_tensor(solution.joint_pos, device=env.device, dtype=torch.float32).unsqueeze(0)
    q_des = q_des.expand(env.num_envs, -1)
    if isinstance(scale, float):
        raw = (q_des - offset) / scale
    else:
        raw = (q_des - offset) / scale
    action_term_slice(action, env, "joint_pos")[:] = raw
    return action


def _foot_axis_distance(
    env: ManagerBasedRlEnv,
    *,
    site_name: str,
    rung_id: int,
    env_idx: int = 0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Distance from sole site to rung axis, sole world pos, and rung axis dir."""
    runtime = LadderRuntime.get(env)
    sample = runtime.sample
    assert sample is not None
    robot = env.scene["robot"]
    site_id = robot.site_names.index(site_name)
    sole = robot.data.site_pos_w[env_idx, site_id]
    center = sample.rung_pos_w[env_idx, rung_id]
    axis = _quat_apply(
        sample.frame_quat[env_idx].unsqueeze(0),
        torch.tensor([[0.0, 1.0, 0.0]], device=env.device, dtype=torch.float32),
    )[0]
    rel = sole - center
    axial = torch.dot(rel, axis)
    radial = rel - axis * axial
    dist = float(torch.linalg.norm(radial).item())
    return (
        dist,
        sole.detach().cpu().numpy().astype(np.float64),
        axis.detach().cpu().numpy().astype(np.float64),
    )


def _hand_attach_errors(env: ManagerBasedRlEnv, *, env_idx: int = 0) -> np.ndarray:
    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    diag = latch.geometry_diagnostics(env_ids=torch.tensor([env_idx], device=env.device))
    residual = diag["equality_residual"][0].detach().cpu().numpy().astype(np.float64)
    return residual


def collect_hold_sample(
    env: ManagerBasedRlEnv,
    cfg: HoldPoseConfig,
    *,
    env_idx: int = 0,
) -> dict:
    """Collect one-step hold diagnostics from simulator task state."""
    robot = env.scene["robot"]
    pelvis_id = robot.body_names.index("pelvis")
    pelvis = robot.data.body_link_pos_w[env_idx, pelvis_id].detach().cpu().numpy()

    left_dist, left_pos, axis_w = _foot_axis_distance(
        env, site_name="left_foot", rung_id=cfg.left_foot_rung_id, env_idx=env_idx
    )
    right_dist, right_pos, _ = _foot_axis_distance(
        env, site_name="right_foot", rung_id=cfg.right_foot_rung_id, env_idx=env_idx
    )
    # Contact proxy: sole-site radial distance to the cylinder axis.
    contact_tol = cfg.foot_contact_axis_tol_m
    foot_in_contact = np.array([left_dist <= contact_tol, right_dist <= contact_tol], dtype=bool)

    force = getattr(robot.data, "actuator_force", None)
    if force is None:
        torque = np.zeros(29, dtype=np.float64)
        saturated = np.zeros(29, dtype=bool)
    else:
        torque = force[env_idx].detach().cpu().numpy().astype(np.float64)
        limits = getattr(robot.data, "actuator_force_limits", None)
        if limits is None:
            saturated = np.zeros_like(torque, dtype=bool)
        else:
            lim = limits[env_idx].detach().cpu().numpy().astype(np.float64)
            saturated = np.abs(torque) >= (0.98 * np.maximum(lim, 1.0e-6))

    sim = env.sim
    nacon = int(sim.data.nacon.item()) if hasattr(sim.data, "nacon") else int(getattr(sim.mj_data, "ncon", 0))
    nefc = int(sim.data.nefc.item()) if hasattr(sim.data, "nefc") else int(getattr(sim.mj_data, "nefc", 0))

    # Approximate total vertical support from contact normal forces when available.
    contact_vertical = 0.0
    if hasattr(sim.data, "contact") and hasattr(sim.data.contact, "frame"):
        # Fallback: use sum of |actuator forces| as a coarse load proxy when Warp
        # contact force aggregation is unavailable in this debug path.
        contact_vertical = float(np.sum(np.abs(torque)))

    qpos = robot.data.root_link_pos_w
    qvel = robot.data.root_link_lin_vel_w
    nonfinite = bool(
        (not torch.isfinite(qpos).all().item())
        or (not torch.isfinite(qvel).all().item())
        or (not np.isfinite(torque).all())
    )

    return {
        "pelvis_pos_w": pelvis,
        "hand_attach_error_m": _hand_attach_errors(env, env_idx=env_idx),
        "foot_axis_dist_m": np.array([left_dist, right_dist], dtype=np.float64),
        "foot_pos_w": np.stack([left_pos, right_pos], axis=0),
        "rung_axis_w": axis_w,
        "foot_in_contact": foot_in_contact,
        "joint_torque": torque,
        "torque_saturated": saturated,
        "contact_vertical_force": contact_vertical,
        "nacon": nacon,
        "nefc": nefc,
        "nonfinite": nonfinite,
    }


def print_hold_summary(
    summary: dict[str, float | bool | str],
    *,
    failure_class: FailureClass,
    header: str = "[hold-pose summary]",
) -> None:
    print(header)
    for key in (
        "duration_s",
        "max_pelvis_drop_m",
        "max_foot_slip_m",
        "mean_foot_contact_fraction",
        "any_foot_contact_fraction",
        "max_hand_attach_error_m",
        "leg_saturation_fraction",
        "max_nacon",
        "max_nefc",
        "any_nonfinite",
    ):
        if key in summary:
            print(f"  {key}={summary[key]}")
    print(f"  failure_class={failure_class}")


def save_hold_pose_npz(
    path: str,
    solution: HoldPoseSolution,
    metrics: HoldPoseMetrics,
    summary: dict[str, float | bool | str],
) -> None:
    """Persist validated pose + hold metrics for debug/test reuse."""
    arrays = metrics.as_arrays()
    np.savez_compressed(
        path,
        root_pos=solution.root_pos,
        root_quat_wxyz=solution.root_quat_wxyz,
        joint_pos=solution.joint_pos,
        hand_errors_m=solution.hand_errors_m,
        foot_errors_m=solution.foot_errors_m,
        left_hand_rung_id=solution.targets.left_hand_rung_id,
        right_hand_rung_id=solution.targets.right_hand_rung_id,
        left_hand_site_id=solution.targets.left_hand_site_id,
        right_hand_site_id=solution.targets.right_hand_site_id,
        left_foot_rung_id=solution.targets.left_foot_rung_id,
        right_foot_rung_id=solution.targets.right_foot_rung_id,
        summary_keys=np.array(list(summary.keys())),
        summary_values=np.array([str(summary[k]) for k in summary.keys()]),
        **arrays,
    )


def run_headless_hold(
    env: ManagerBasedRlEnv,
    cfg: HoldPoseConfig,
    *,
    env_idx: int = 0,
) -> tuple[HoldPoseSolution, HoldPoseMetrics, dict, FailureClass]:
    """Initialize pose/latches and simulate a gravity hold without a viewer."""
    solution = initialize_hold_pose_scene(env, cfg, env_idx=env_idx)
    metrics = HoldPoseMetrics(dt=float(env.step_dt))
    steps = int(round(cfg.hold_duration_s / env.step_dt))
    for i in range(max(steps, 1)):
        action = build_hold_action(env, solution)
        env.step(action)
        if i >= cfg.settle_steps:
            metrics.record(collect_hold_sample(env, cfg, env_idx=env_idx))
    summary = metrics.summary()
    failure = classify_hold_failure(
        summary,
        cfg,
        ik_ok=solution.finite and solution.within_joint_limits and float(solution.hand_errors_m.max()) < 0.06,
    )
    return solution, metrics, summary, failure
