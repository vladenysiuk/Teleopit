"""Least-squares / mink IK for a static four-contact ladder-standing pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from train_mimic.tasks.climbing.config.hold_pose import HoldPoseConfig
from train_mimic.tasks.climbing.config.robot import (
    LEFT_HAND_POINT_SITE,
    RIGHT_HAND_POINT_SITE,
    _get_g1_climbing_spec,
)
from train_mimic.tasks.climbing.ladder.generator import (
    latch_site_normal_offset,
    site_y_offsets,
)
from train_mimic.tasks.climbing.ladder.state import LadderRuntime, _quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


LEFT_FOOT_SITE = "left_foot"
RIGHT_FOOT_SITE = "right_foot"
LEFT_ANKLE_BODY = "left_ankle_roll_link"
RIGHT_ANKLE_BODY = "right_ankle_roll_link"


@dataclass(frozen=True)
class HoldPoseTargets:
    """World-frame contact targets for one hold pose."""

    left_hand_pos_w: np.ndarray
    right_hand_pos_w: np.ndarray
    left_foot_pos_w: np.ndarray
    right_foot_pos_w: np.ndarray
    pelvis_pos_w: np.ndarray
    torso_quat_wxyz: np.ndarray
    # Flat plant: ankle +X toward ladder, +Z up (matches G1 sole layout).
    foot_quat_wxyz: np.ndarray
    left_hand_rung_id: int
    right_hand_rung_id: int
    left_hand_site_id: int
    right_hand_site_id: int
    left_foot_rung_id: int
    right_foot_rung_id: int


@dataclass(frozen=True)
class HoldPoseSolution:
    """Deterministic IK result for Stage 6 hold-pose initialization."""

    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos: np.ndarray
    targets: HoldPoseTargets
    hand_errors_m: np.ndarray
    foot_errors_m: np.ndarray
    max_joint_limit_violation: float
    finite: bool

    @property
    def within_joint_limits(self) -> bool:
        return self.max_joint_limit_violation <= 1.0e-3


def _climbing_seed_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Heuristic upright ladder-stand seed (feet@1 / hands@3)."""
    qpos = model.qpos0.copy()
    # freejoint stays at qpos0; joint block starts at index 7.
    j = qpos[7:]
    # Mild squat: knees open, ankles near flat (not deep wedge).
    j[0] = -0.45  # left hip pitch
    j[3] = 0.90  # left knee
    j[4] = -0.10  # left ankle pitch (flat plant)
    j[5] = 0.0  # left ankle roll
    j[6] = -0.45  # right hip pitch
    j[9] = 0.90  # right knee
    j[10] = -0.10  # right ankle pitch
    j[11] = 0.0  # right ankle roll
    # Arms: reach up/forward two rung spacings.
    j[15] = 0.85  # left shoulder pitch
    j[16] = 0.20  # left shoulder roll
    j[18] = 1.05  # left elbow
    j[22] = 0.85  # right shoulder pitch
    j[23] = -0.20  # right shoulder roll
    j[25] = 1.05  # right elbow
    return qpos


def _rotmat_to_wxyz(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to wxyz quaternion."""
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), 1.0e-12)


def build_hold_pose_targets(
    env: ManagerBasedRlEnv,
    cfg: HoldPoseConfig,
    *,
    env_idx: int = 0,
) -> HoldPoseTargets:
    """Compute latch-site and foot-top targets from the current ladder sample."""
    import torch

    runtime = LadderRuntime.get(env)
    sample = runtime.sample
    if sample is None:
        raise RuntimeError("LadderRuntime has no sample; reset the env first.")
    ladder_cfg = runtime.cfg

    for rung_id in (
        cfg.left_foot_rung_id,
        cfg.right_foot_rung_id,
        cfg.left_hand_rung_id,
        cfg.right_hand_rung_id,
    ):
        if rung_id >= ladder_cfg.max_rungs:
            raise ValueError(f"rung_id {rung_id} exceeds max_rungs={ladder_cfg.max_rungs}")
        if not bool(sample.active_mask[env_idx, rung_id].item()):
            raise ValueError(f"Hold-pose rung {rung_id} is inactive for env {env_idx}.")

    y_offsets = site_y_offsets(ladder_cfg)
    if cfg.left_hand_site_id >= len(y_offsets) or cfg.right_hand_site_id >= len(y_offsets):
        raise ValueError("Hand site ids exceed sites_per_rung.")

    device = sample.frame_pos.device
    dtype = sample.frame_pos.dtype
    frame_quat = sample.frame_quat[env_idx]
    local_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    local_y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    local_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    axis_x = _quat_apply(frame_quat.unsqueeze(0), local_x.unsqueeze(0))[0]
    axis_y = _quat_apply(frame_quat.unsqueeze(0), local_y.unsqueeze(0))[0]
    axis_z = _quat_apply(frame_quat.unsqueeze(0), local_z.unsqueeze(0))[0]

    site_n = float(latch_site_normal_offset(ladder_cfg))

    def hand_target(rung_id: int, site_id: int) -> np.ndarray:
        center = sample.rung_pos_w[env_idx, rung_id]
        y = float(y_offsets[site_id])
        pos = center - axis_x * site_n + axis_y * y
        return pos.detach().cpu().numpy().astype(np.float64)

    def foot_sole_target(rung_id: int, foot_y: float) -> np.ndarray:
        center = sample.rung_pos_w[env_idx, rung_id]
        # Sole site on the cylinder top-front, lifted clear of the volume.
        pos = (
            center
            - axis_x * float(cfg.foot_approach_offset_m)
            + axis_y * float(foot_y)
            + axis_z * float(ladder_cfg.rung_radius + cfg.foot_clearance_m)
        )
        return pos.detach().cpu().numpy().astype(np.float64)

    foot_h = float(sample.rung_heights[env_idx, cfg.left_foot_rung_id].item())
    hand_h = float(sample.rung_heights[env_idx, cfg.left_hand_rung_id].item())
    # Mild crouch biased toward the feet so cylinders carry more load.
    mid_h = (
        float(cfg.pelvis_height_foot_weight) * foot_h
        + float(cfg.pelvis_height_hand_weight) * hand_h
        + float(cfg.pelvis_height_offset_m)
    )
    pelvis = (
        sample.frame_pos[env_idx]
        - axis_x * float(cfg.pelvis_standoff_m)
        + axis_z * mid_h
    )

    # Face the ladder: robot +X aligns with ladder +X.
    ax = axis_x.detach().cpu().numpy()
    ay = axis_y.detach().cpu().numpy()
    az = axis_z.detach().cpu().numpy()
    torso_mat = np.stack([ax, ay, az], axis=1)
    # Flat foot plant uses the same ladder-aligned axes (sole −Z down).
    foot_mat = torso_mat.copy()
    return HoldPoseTargets(
        left_hand_pos_w=hand_target(cfg.left_hand_rung_id, cfg.left_hand_site_id),
        right_hand_pos_w=hand_target(cfg.right_hand_rung_id, cfg.right_hand_site_id),
        left_foot_pos_w=foot_sole_target(cfg.left_foot_rung_id, cfg.left_foot_y_m),
        right_foot_pos_w=foot_sole_target(cfg.right_foot_rung_id, cfg.right_foot_y_m),
        pelvis_pos_w=pelvis.detach().cpu().numpy().astype(np.float64),
        torso_quat_wxyz=_rotmat_to_wxyz(torso_mat),
        foot_quat_wxyz=_rotmat_to_wxyz(foot_mat),
        left_hand_rung_id=cfg.left_hand_rung_id,
        right_hand_rung_id=cfg.right_hand_rung_id,
        left_hand_site_id=cfg.left_hand_site_id,
        right_hand_site_id=cfg.right_hand_site_id,
        left_foot_rung_id=cfg.left_foot_rung_id,
        right_foot_rung_id=cfg.right_foot_rung_id,
    )


def _plant_ankles_and_realign_soles(
    configuration,
    model: mujoco.MjModel,
    targets: HoldPoseTargets,
    cfg: HoldPoseConfig,
    *,
    tasks: list | None = None,
    limits: list | None = None,
    solver: str = "daqp",
) -> np.ndarray:
    """Force a flat ankle plant, realign soles, then re-reach hand latch sites.

    Mink often leaves deep pitch / opposing roll that buries the ankle/heel
    volume even when the sole-site cost looks small. Clamping ankles without
    a root correction drops the soles into the cylinder. A pure root lift
    without re-reaching hands leaves large latch residuals that yank the body
    down under gravity — refine arms afterward.
    """
    import mink

    qpos = configuration.data.qpos.copy()
    # Joint order matches G1 29-DoF: L/R ankle pitch at 4/10, roll at 5/11.
    pitch = float(cfg.ankle_pitch_plant_rad)
    pitch = max(pitch, float(cfg.ankle_pitch_min_rad))
    qpos[7 + 4] = pitch
    qpos[7 + 10] = pitch
    roll_lim = float(cfg.ankle_roll_limit_rad)
    qpos[7 + 5] = float(np.clip(qpos[7 + 5], -roll_lim, roll_lim))
    qpos[7 + 11] = float(np.clip(qpos[7 + 11], -roll_lim, roll_lim))
    configuration.update(q=qpos)
    mujoco.mj_forward(model, configuration.data)
    data = configuration.data

    left = np.asarray(data.site(LEFT_FOOT_SITE).xpos, dtype=np.float64)
    right = np.asarray(data.site(RIGHT_FOOT_SITE).xpos, dtype=np.float64)
    delta = 0.5 * (
        (np.asarray(targets.left_foot_pos_w, dtype=np.float64) - left)
        + (np.asarray(targets.right_foot_pos_w, dtype=np.float64) - right)
    )
    # Never lower soles into the rung; optional extra spawn lift stacks on top.
    delta[2] = max(float(delta[2]), 0.0) + float(cfg.spawn_root_lift_m)
    qpos[0:3] = qpos[0:3] + delta
    configuration.update(q=qpos)
    mujoco.mj_forward(model, configuration.data)

    if tasks is not None and limits is not None:
        # Short refine: pull hands back to latch sites while holding foot plant.
        for _ in range(60):
            vel = mink.solve_ik(
                configuration,
                tasks,
                cfg.ik_dt,
                solver,
                cfg.ik_damping,
                limits=limits,
            )
            configuration.integrate_inplace(vel, cfg.ik_dt)
            # Re-apply ankle plant every iter so refine cannot re-bury feet.
            q = configuration.data.qpos.copy()
            q[7 + 4] = pitch
            q[7 + 10] = pitch
            q[7 + 5] = float(np.clip(q[7 + 5], -roll_lim, roll_lim))
            q[7 + 11] = float(np.clip(q[7 + 11], -roll_lim, roll_lim))
            configuration.update(q=q)
        mujoco.mj_forward(model, configuration.data)
        # Final sole realign (Z-only) if refine drifted feet down.
        data = configuration.data
        left = np.asarray(data.site(LEFT_FOOT_SITE).xpos, dtype=np.float64)
        right = np.asarray(data.site(RIGHT_FOOT_SITE).xpos, dtype=np.float64)
        dz = max(
            float(targets.left_foot_pos_w[2] - left[2]),
            float(targets.right_foot_pos_w[2] - right[2]),
            0.0,
        )
        if dz > 1.0e-4:
            q = configuration.data.qpos.copy()
            q[2] += dz
            configuration.update(q=q)
            mujoco.mj_forward(model, configuration.data)

    return configuration.data.qpos.copy()


def solve_hold_pose_ik(
    env: ManagerBasedRlEnv,
    cfg: HoldPoseConfig,
    *,
    env_idx: int = 0,
) -> HoldPoseSolution:
    """Solve a deterministic four-contact hold pose with mink least-squares IK."""
    import mink

    targets = build_hold_pose_targets(env, cfg, env_idx=env_idx)
    runtime = LadderRuntime.get(env)
    sample = runtime.sample
    assert sample is not None

    spec = _get_g1_climbing_spec()
    model = spec.compile()
    configuration = mink.Configuration(model)

    seed = _climbing_seed_qpos(model)
    # Keep the freejoint on the robot-facing side of the ladder (−ladder X).
    seed[0:3] = targets.pelvis_pos_w
    seed[3:7] = targets.torso_quat_wxyz
    configuration.update(q=seed)

    hand_tasks = [
        mink.FrameTask(
            frame_name=LEFT_HAND_POINT_SITE,
            frame_type="site",
            position_cost=cfg.hand_position_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        ),
        mink.FrameTask(
            frame_name=RIGHT_HAND_POINT_SITE,
            frame_type="site",
            position_cost=cfg.hand_position_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        ),
    ]
    foot_tasks = [
        mink.FrameTask(
            frame_name=LEFT_FOOT_SITE,
            frame_type="site",
            position_cost=cfg.foot_position_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        ),
        mink.FrameTask(
            frame_name=RIGHT_FOOT_SITE,
            frame_type="site",
            position_cost=cfg.foot_position_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        ),
    ]
    ankle_ori_tasks: list = []
    if float(cfg.foot_orientation_cost) > 0.0:
        ankle_ori_tasks = [
            mink.FrameTask(
                frame_name=LEFT_ANKLE_BODY,
                frame_type="body",
                position_cost=0.0,
                orientation_cost=cfg.foot_orientation_cost,
                lm_damping=1.0,
            ),
            mink.FrameTask(
                frame_name=RIGHT_ANKLE_BODY,
                frame_type="body",
                position_cost=0.0,
                orientation_cost=cfg.foot_orientation_cost,
                lm_damping=1.0,
            ),
        ]
    pelvis_task = mink.FrameTask(
        frame_name="pelvis",
        frame_type="body",
        position_cost=cfg.pelvis_position_cost,
        orientation_cost=cfg.torso_orientation_cost,
        lm_damping=1.0,
    )
    torso_task = mink.FrameTask(
        frame_name="torso_link",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=cfg.torso_orientation_cost,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=cfg.posture_cost)
    posture_task.set_target(seed)

    ori = mink.SO3(np.asarray(targets.torso_quat_wxyz, dtype=np.float64))
    foot_ori = mink.SO3(np.asarray(targets.foot_quat_wxyz, dtype=np.float64))
    hand_tasks[0].set_target(
        mink.SE3.from_rotation_and_translation(ori, targets.left_hand_pos_w)
    )
    hand_tasks[1].set_target(
        mink.SE3.from_rotation_and_translation(ori, targets.right_hand_pos_w)
    )
    foot_tasks[0].set_target(
        mink.SE3.from_rotation_and_translation(foot_ori, targets.left_foot_pos_w)
    )
    foot_tasks[1].set_target(
        mink.SE3.from_rotation_and_translation(foot_ori, targets.right_foot_pos_w)
    )
    for ankle_task in ankle_ori_tasks:
        # Orientation only; translation target unused.
        ankle_task.set_target(
            mink.SE3.from_rotation_and_translation(foot_ori, targets.pelvis_pos_w)
        )
    pelvis_task.set_target(
        mink.SE3.from_rotation_and_translation(ori, targets.pelvis_pos_w)
    )
    torso_task.set_target(
        mink.SE3.from_rotation_and_translation(ori, targets.pelvis_pos_w)
    )

    tasks = [*hand_tasks, *foot_tasks, *ankle_ori_tasks, pelvis_task, torso_task, posture_task]
    limits = [mink.ConfigurationLimit(model)]

    # Prefer daqp (common with mink); fall back to quadprog-compatible solvers.
    solver = cfg.ik_solver
    try:
        mink.solve_ik(configuration, tasks, cfg.ik_dt, solver, cfg.ik_damping, limits=limits)
    except Exception:
        solver = "quadprog"
        try:
            mink.solve_ik(configuration, tasks, cfg.ik_dt, solver, cfg.ik_damping, limits=limits)
        except Exception:
            solver = "daqp"

    prev_err = float("inf")
    for _ in range(cfg.ik_max_iters):
        vel = mink.solve_ik(
            configuration,
            tasks,
            cfg.ik_dt,
            solver,
            cfg.ik_damping,
            limits=limits,
        )
        configuration.integrate_inplace(vel, cfg.ik_dt)
        err = float(np.linalg.norm(np.concatenate([task.compute_error(configuration) for task in tasks])))
        if abs(prev_err - err) < 1.0e-4:
            break
        prev_err = err

    qpos = _plant_ankles_and_realign_soles(
        configuration,
        model,
        targets,
        cfg,
        tasks=tasks,
        limits=limits,
        solver=solver,
    )
    data = configuration.data

    hand_errs = np.array(
        [
            np.linalg.norm(data.site(LEFT_HAND_POINT_SITE).xpos - targets.left_hand_pos_w),
            np.linalg.norm(data.site(RIGHT_HAND_POINT_SITE).xpos - targets.right_hand_pos_w),
        ],
        dtype=np.float64,
    )
    foot_errs = np.array(
        [
            np.linalg.norm(data.site(LEFT_FOOT_SITE).xpos - targets.left_foot_pos_w),
            np.linalg.norm(data.site(RIGHT_FOOT_SITE).xpos - targets.right_foot_pos_w),
        ],
        dtype=np.float64,
    )

    joint_pos = qpos[7:].astype(np.float64)
    limit_violation = 0.0
    for jnt_id in range(model.njnt):
        if model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qadr = int(model.jnt_qposadr[jnt_id])
        lo, hi = model.jnt_range[jnt_id]
        if hi <= lo:
            continue
        q = float(qpos[qadr])
        limit_violation = max(limit_violation, lo - q, q - hi)

    finite = bool(np.isfinite(qpos).all())
    return HoldPoseSolution(
        root_pos=qpos[0:3].astype(np.float64),
        root_quat_wxyz=qpos[3:7].astype(np.float64),
        joint_pos=joint_pos,
        targets=targets,
        hand_errors_m=hand_errs,
        foot_errors_m=foot_errs,
        max_joint_limit_violation=float(limit_violation),
        finite=finite,
    )
