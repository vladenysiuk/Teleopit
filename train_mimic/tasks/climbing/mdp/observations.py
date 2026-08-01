"""Climbing observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as mjlab_mdp
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul

from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.observations import RelativeRungObsConfig
from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.ladder.relative_rungs import RelativeRungProvider, world_to_body
from train_mimic.tasks.climbing.ladder.state import LadderRuntime

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def action_term_slice(
    actions: torch.Tensor,
    env: ManagerBasedRlEnv,
    term_name: str,
) -> torch.Tensor:
    """Slice a flat action tensor by registered action-term name."""
    idx = 0
    for name in env.action_manager.active_terms:
        dim = env.action_manager.get_term(name).action_dim
        if name == term_name:
            return actions[:, idx : idx + dim]
        idx += dim
    known = ", ".join(env.action_manager.active_terms)
    raise KeyError(f"Unknown action term '{term_name}'. Active terms: {known}")


def _ladder_relative_heights(
    env: ManagerBasedRlEnv,
    *,
    reference_body: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return ladder up-axis, reference height, and per-rung ladder-relative heights."""
    runtime = env.extras.get(LadderRuntime.EXTRA_KEY)
    if runtime is None or runtime.sample is None:
        return None

    from train_mimic.tasks.climbing.ladder.relative_rungs import (
        ladder_relative_height,
        ladder_relative_heights_from_world,
        ladder_upward_axis_w,
    )

    sample = runtime.sample
    robot = env.scene["robot"]
    ref_pos_w = robot.data.body_link_pos_w[:, robot.body_names.index(reference_body)]
    upward_axis_w = ladder_upward_axis_w(sample.frame_quat)
    ref_height_l = ladder_relative_height(ref_pos_w, sample.frame_pos, upward_axis_w)
    rung_heights_l = ladder_relative_heights_from_world(
        sample.rung_pos_w,
        sample.frame_pos,
        upward_axis_w,
    )
    return upward_axis_w, ref_height_l, rung_heights_l


def _zeros_like_env(env: ManagerBasedRlEnv, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(env.num_envs, *shape, device=env.device, dtype=torch.float32)


def hand_in_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Left/right hand contact flags as float ``[B, 2]``."""
    contacts = env.extras.get(ClimbContactState.EXTRA_KEY)
    if contacts is None:
        return _zeros_like_env(env, (2,))
    contacts.update()
    return contacts.state.hand_in_contact.to(dtype=torch.float32)


def hand_attached(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Left/right latch attachment flags as float ``[B, 2]``."""
    latch = env.extras.get(ClimbLatchState.EXTRA_KEY)
    if latch is None:
        return _zeros_like_env(env, (2,))
    latch._ensure_state()
    return latch.state.attached.to(dtype=torch.float32)


def attached_rung_rel_height(
    env: ManagerBasedRlEnv,
    *,
    reference_body: str = "pelvis",
) -> torch.Tensor:
    """Attached rung ladder-relative height minus reference height; zero when detached."""
    latch = env.extras.get(ClimbLatchState.EXTRA_KEY)
    heights = _ladder_relative_heights(env, reference_body=reference_body)
    if latch is None or heights is None:
        return _zeros_like_env(env, (2,))
    latch._ensure_state()

    _, ref_height_l, rung_heights_l = heights
    rel = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)
    for hand_idx in range(2):
        attached = latch.state.attached[:, hand_idx]
        rung_id = latch.state.rung_id[:, hand_idx]
        valid = attached & (rung_id >= 0)
        if not bool(torch.any(valid).item()):
            continue
        env_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        ids = rung_id[env_ids]
        rung_h = rung_heights_l[env_ids, ids]
        rel[env_ids, hand_idx] = rung_h - ref_height_l[env_ids]
    return rel


def prev_joint_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Previous joint-position action slice."""
    return action_term_slice(mjlab_mdp.last_action(env), env, "joint_pos")


def prev_latch_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Previous latch action slice."""
    return action_term_slice(mjlab_mdp.last_action(env), env, "latch")


def relative_rung_endpoints_torso(
    env: ManagerBasedRlEnv,
    *,
    ladder_cfg: LadderConfig,
    obs_cfg: RelativeRungObsConfig,
) -> torch.Tensor:
    """Privileged relative-rung endpoints in torso frame, shape ``[B, K, 7]``."""
    del ladder_cfg
    provider = env.extras.get(RelativeRungProvider.EXTRA_KEY)
    if provider is None:
        return _zeros_like_env(env, (obs_cfg.num_rungs, 7))
    return provider.compute().endpoints


def body_height_w(env: ManagerBasedRlEnv, *, body_name: str) -> torch.Tensor:
    """World-frame body height ``[B, 1]``."""
    robot = env.scene["robot"]
    body_id = robot.body_names.index(body_name)
    height = robot.data.body_link_pos_w[:, body_id, 2:3]
    return height


def ladder_frame_pos_torso(
    env: ManagerBasedRlEnv,
    *,
    ladder_cfg: LadderConfig,
) -> torch.Tensor:
    """Ladder frame origin expressed in torso frame ``[B, 3]``."""
    del ladder_cfg
    runtime = env.extras.get(LadderRuntime.EXTRA_KEY)
    if runtime is None or runtime.sample is None:
        return _zeros_like_env(env, (3,))

    robot = env.scene["robot"]
    torso_id = robot.body_names.index("torso_link")
    frame_pos = runtime.sample.frame_pos
    return world_to_body(
        robot.data.body_link_pos_w[:, torso_id],
        robot.data.body_link_quat_w[:, torso_id],
        frame_pos.unsqueeze(1),
    ).squeeze(1)


def ladder_frame_ori_torso(
    env: ManagerBasedRlEnv,
    *,
    ladder_cfg: LadderConfig,
) -> torch.Tensor:
    """First two columns of ladder-frame rotation in torso frame ``[B, 6]``.

    Uses a 6D orientation representation (two torso-frame axes) rather than a
    raw quaternion to avoid double-cover ambiguity.
    """
    del ladder_cfg
    runtime = env.extras.get(LadderRuntime.EXTRA_KEY)
    if runtime is None or runtime.sample is None:
        return _zeros_like_env(env, (6,))

    robot = env.scene["robot"]
    torso_id = robot.body_names.index("torso_link")
    rel_quat = quat_mul(
        quat_inv(robot.data.body_link_quat_w[:, torso_id]),
        runtime.sample.frame_quat,
    )
    # Canonicalize quaternion hemisphere before converting to rotation matrix.
    sign = torch.where(rel_quat[:, :1] < 0.0, -1.0, 1.0)
    rel_quat = rel_quat * sign
    mat = matrix_from_quat(rel_quat)
    return mat[..., :2].reshape(env.num_envs, 6)


def _active_rung_window_state(
    env: ManagerBasedRlEnv,
    *,
    obs_cfg: RelativeRungObsConfig,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Shared critic window state: ladder-relative heights ``[B, K]`` and mask ``[B, K]``."""
    provider = env.extras.get(RelativeRungProvider.EXTRA_KEY)
    heights_ctx = _ladder_relative_heights(env, reference_body=obs_cfg.reference_body)
    if provider is None or heights_ctx is None:
        return None

    _, ref_height_l, rung_heights_l = heights_ctx
    obs = provider.compute()
    heights = torch.zeros(
        env.num_envs, obs_cfg.num_rungs, device=env.device, dtype=torch.float32
    )
    valid_mask = torch.zeros(
        env.num_envs, obs_cfg.num_rungs, device=env.device, dtype=torch.float32
    )
    for slot in range(obs_cfg.num_rungs):
        valid = obs.valid_mask[:, slot]
        if not bool(torch.any(valid).item()):
            continue
        env_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        rung_id = obs.rung_indices[env_ids, slot]
        heights[env_ids, slot] = rung_heights_l[env_ids, rung_id] - ref_height_l[env_ids]
        valid_mask[env_ids, slot] = 1.0
    return heights, valid_mask


def active_rung_heights_l_rel(
    env: ManagerBasedRlEnv,
    *,
    ladder_cfg: LadderConfig,
    obs_cfg: RelativeRungObsConfig,
) -> torch.Tensor:
    """Windowed rung heights relative to reference, along ladder up axis ``[B, K]``."""
    del ladder_cfg
    state = _active_rung_window_state(env, obs_cfg=obs_cfg)
    if state is None:
        return _zeros_like_env(env, (obs_cfg.num_rungs,))
    return state[0]


def active_rung_valid_mask(
    env: ManagerBasedRlEnv,
    *,
    ladder_cfg: LadderConfig,
    obs_cfg: RelativeRungObsConfig,
) -> torch.Tensor:
    """Explicit validity mask for critic rung-height slots ``[B, K]``."""
    del ladder_cfg
    state = _active_rung_window_state(env, obs_cfg=obs_cfg)
    if state is None:
        return _zeros_like_env(env, (obs_cfg.num_rungs,))
    return state[1]
