"""Privileged relative-rung geometry in the robot torso frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.observations import RelativeRungObsConfig
from train_mimic.tasks.climbing.ladder.state import LadderRuntime, _quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def quat_inv_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Invert unit quaternions in wxyz layout."""
    inv = quat.clone()
    inv[..., 1:4] = -inv[..., 1:4]
    return inv


def world_to_body(
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    points_w: torch.Tensor,
) -> torch.Tensor:
    """Express world-frame points in a body frame.

    ``p_body = R_body^T (p_world - p_body_world)``.
    """
    delta = points_w - pos_w.unsqueeze(-2)
    quat_inv = quat_inv_wxyz(quat_w)
    if delta.ndim == 2:
        return _quat_apply(quat_inv, delta)
    batch, count, _ = delta.shape
    quat_inv_b = quat_inv
    if quat_inv_b.ndim == 1:
        quat_inv_b = quat_inv_b.unsqueeze(0).expand(batch, 4)
    quat_flat = quat_inv_b.unsqueeze(1).expand(batch, count, 4).reshape(-1, 4)
    delta_flat = delta.reshape(-1, 3)
    return _quat_apply(quat_flat, delta_flat).reshape(batch, count, 3)


def ladder_upward_axis_w(frame_quat: torch.Tensor) -> torch.Tensor:
    """Return the ladder +Z axis in world frame, shape ``[..., 3]``."""
    local_up = torch.zeros(
        *frame_quat.shape[:-1],
        3,
        device=frame_quat.device,
        dtype=frame_quat.dtype,
    )
    local_up[..., 2] = 1.0
    if frame_quat.ndim == 1:
        return _quat_apply(frame_quat.unsqueeze(0), local_up).squeeze(0)
    return _quat_apply(frame_quat, local_up)


def ladder_relative_height(
    pos_w: torch.Tensor,
    frame_pos: torch.Tensor,
    upward_axis_w: torch.Tensor,
) -> torch.Tensor:
    """Scalar height along the ladder upward axis: ``u_L^T (p - p_L)``."""
    delta = pos_w - frame_pos
    if delta.ndim == 3:
        axis = upward_axis_w.unsqueeze(-2)
    else:
        axis = upward_axis_w
    return (delta * axis).sum(dim=-1)


def ladder_relative_heights_from_world(
    positions_w: torch.Tensor,
    frame_pos: torch.Tensor,
    upward_axis_w: torch.Tensor,
) -> torch.Tensor:
    """Batch ladder-relative heights for world-frame positions.

    ``positions_w`` may be ``[B, N, 3]`` or ``[B, 3]``.
    """
    if positions_w.ndim == 2:
        return ladder_relative_height(positions_w, frame_pos, upward_axis_w)
    delta = positions_w - frame_pos.unsqueeze(-2)
    axis = upward_axis_w.unsqueeze(-2)
    return (delta * axis).sum(dim=-1)


def select_rung_window_indices(
    *,
    active_mask: torch.Tensor,
    rung_heights_l: torch.Tensor,
    reference_height_l: torch.Tensor,
    num_rungs: int,
    rungs_below: int,
    rungs_above: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ``num_rungs`` slots with deterministic below/above ordering.

    Heights are ladder-relative scalars ``h = u_L^T (p - p_L)``, not world Z.

    Returns:
        rung_indices: ``[B, K]`` int64 rung IDs, ``-1`` when padded.
        valid_mask: ``[B, K]`` bool, ``True`` for real active rungs.
    """
    batch, max_rungs = active_mask.shape
    device = active_mask.device
    rung_ids = torch.arange(max_rungs, device=device, dtype=torch.int64)
    heights = rung_heights_l.to(dtype=torch.float32)

    selected_ids = torch.full((batch, num_rungs), -1, dtype=torch.int64, device=device)
    valid = torch.zeros(batch, num_rungs, dtype=torch.bool, device=device)

    for env_idx in range(batch):
        active = active_mask[env_idx]
        if not bool(torch.any(active).item()):
            continue

        active_ids = rung_ids[active]
        active_heights = heights[env_idx, active]
        order = torch.argsort(active_heights, stable=True)
        active_ids = active_ids[order]
        active_heights = active_heights[order]

        ref_h = reference_height_l[env_idx]
        below_mask = active_heights <= ref_h
        below_ids = active_ids[below_mask]
        above_ids = active_ids[~below_mask]

        pick_below = below_ids[-rungs_below:] if below_ids.numel() > 0 else below_ids
        pick_above = above_ids[:rungs_above] if above_ids.numel() > 0 else above_ids
        chosen = torch.cat((pick_below, pick_above))

        count = min(int(chosen.numel()), num_rungs)
        if count > 0:
            selected_ids[env_idx, :count] = chosen[:count]
            valid[env_idx, :count] = True

    return selected_ids, valid


def compute_rung_endpoints_world(
    *,
    rung_centers_w: torch.Tensor,
    rung_quat_w: torch.Tensor,
    half_length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rung endpoints in world frame.

    Rungs span local +Y in the ladder frame; ``rung_quat_w`` rotates that axis.
    """
    local_y = torch.tensor([0.0, 1.0, 0.0], device=rung_centers_w.device, dtype=torch.float32)
    axis_w = _quat_apply(rung_quat_w.reshape(-1, 4), local_y.view(1, 3).expand(rung_quat_w.shape[0], 3))
    axis_w = axis_w.reshape(*rung_quat_w.shape[:-1], 3)
    offset = axis_w * half_length
    return rung_centers_w - offset, rung_centers_w + offset


@dataclass
class RelativeRungObservation:
    """Batched relative-rung exteroception."""

    endpoints: torch.Tensor
    """Shape ``[B, K, 7]``: endpoint_a(3), endpoint_b(3), valid(1)."""

    rung_indices: torch.Tensor
    """Shape ``[B, K]`` selected rung IDs for debugging only; not actor input."""

    valid_mask: torch.Tensor
    """Shape ``[B, K]`` validity mask."""


class RelativeRungProvider:
    """Compute privileged relative-rung geometry from simulator task state."""

    EXTRA_KEY = "relative_rung_provider"

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        ladder_cfg: LadderConfig,
        obs_cfg: RelativeRungObsConfig,
    ) -> None:
        self.env = env
        self.ladder_cfg = ladder_cfg
        self.obs_cfg = obs_cfg
        self._robot = env.scene["robot"]
        self._reference_body_id = self._robot.body_names.index(obs_cfg.reference_body)
        self._torso_body_id = self._robot.body_names.index(obs_cfg.torso_body)

    @classmethod
    def attach(
        cls,
        env: ManagerBasedRlEnv,
        *,
        ladder_cfg: LadderConfig,
        obs_cfg: RelativeRungObsConfig | None = None,
    ) -> RelativeRungProvider:
        provider = cls(env, ladder_cfg=ladder_cfg, obs_cfg=obs_cfg or RelativeRungObsConfig())
        env.extras[cls.EXTRA_KEY] = provider
        return provider

    @classmethod
    def get(cls, env: ManagerBasedRlEnv) -> RelativeRungProvider:
        provider = env.extras.get(cls.EXTRA_KEY)
        if provider is None:
            raise RuntimeError("RelativeRungProvider is not attached to the environment.")
        return provider

    def compute(self) -> RelativeRungObservation:
        runtime = LadderRuntime.get(self.env)
        if runtime.sample is None:
            raise RuntimeError("Ladder sample unavailable for relative-rung observations.")

        sample = runtime.sample
        batch = self.env.num_envs
        k = self.obs_cfg.num_rungs
        device = self.env.device

        ref_pos_w = self._robot.data.body_link_pos_w[:, self._reference_body_id]
        upward_axis_w = ladder_upward_axis_w(sample.frame_quat)
        ref_height_l = ladder_relative_height(ref_pos_w, sample.frame_pos, upward_axis_w)
        rung_heights_l = ladder_relative_heights_from_world(
            sample.rung_pos_w,
            sample.frame_pos,
            upward_axis_w,
        )

        rung_indices, valid_mask = select_rung_window_indices(
            active_mask=sample.active_mask,
            rung_heights_l=rung_heights_l,
            reference_height_l=ref_height_l,
            num_rungs=k,
            rungs_below=self.obs_cfg.rungs_below,
            rungs_above=self.obs_cfg.rungs_above,
        )

        endpoints = torch.zeros(batch, k, 7, device=device, dtype=torch.float32)
        torso_pos_w = self._robot.data.body_link_pos_w[:, self._torso_body_id]
        torso_quat_w = self._robot.data.body_link_quat_w[:, self._torso_body_id]

        half_length = self.ladder_cfg.rung_length * 0.5
        for slot in range(k):
            rung_id = rung_indices[:, slot]
            slot_valid = valid_mask[:, slot]
            if not bool(torch.any(slot_valid).item()):
                continue

            env_ids = torch.nonzero(slot_valid, as_tuple=False).squeeze(-1)
            ids = rung_id[env_ids]
            centers = sample.rung_pos_w[env_ids, ids]
            quats = sample.rung_quat_w[env_ids, ids]
            end_a_w, end_b_w = compute_rung_endpoints_world(
                rung_centers_w=centers,
                rung_quat_w=quats,
                half_length=half_length,
            )
            end_a_b = world_to_body(
                torso_pos_w[env_ids],
                torso_quat_w[env_ids],
                end_a_w.unsqueeze(1),
            ).squeeze(1)
            end_b_b = world_to_body(
                torso_pos_w[env_ids],
                torso_quat_w[env_ids],
                end_b_w.unsqueeze(1),
            ).squeeze(1)
            endpoints[env_ids, slot, 0:3] = end_a_b
            endpoints[env_ids, slot, 3:6] = end_b_b
            endpoints[env_ids, slot, 6] = 1.0

        return RelativeRungObservation(
            endpoints=endpoints,
            rung_indices=rung_indices,
            valid_mask=valid_mask,
        )

    def format_debug_lines(self, env_idx: int = 0) -> list[str]:
        obs = self.compute()
        lines = [f"relative_rungs env={env_idx} K={self.obs_cfg.num_rungs}"]
        for slot in range(self.obs_cfg.num_rungs):
            if not bool(obs.valid_mask[env_idx, slot].item()):
                lines.append(f"  slot {slot:02d}: invalid")
                continue
            rung_id = int(obs.rung_indices[env_idx, slot].item())
            end_a = obs.endpoints[env_idx, slot, 0:3].tolist()
            end_b = obs.endpoints[env_idx, slot, 3:6].tolist()
            lines.append(
                f"  slot {slot:02d}: rung={rung_id:02d} "
                f"endpoint_a_torso={end_a} endpoint_b_torso={end_b}"
            )
        return lines
