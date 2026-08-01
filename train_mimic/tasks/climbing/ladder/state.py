"""Sampled ladder state, topology indexing, and batched mocap updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch

from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.ladder.generator import (
    expected_rung_geom_names,
    expected_site_names,
    latch_site_normal_offset,
    site_y_offsets,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Per-env rung layout writes these model fields at reset/resample. mjlab viewers
# only copy batched model fields into the host MjModel when they are registered
# via Simulation.expand_model_fields().
_LADDER_RUNTIME_MODEL_FIELDS = ("geom_pos", "site_pos")


def _quat_from_yaw_pitch(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Build wxyz quaternions from yaw (about Z) and pitch (about Y)."""
    half_yaw = yaw * 0.5
    half_pitch = pitch * 0.5
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    w = cy * cp
    x = cy * sp
    y = sy * cp
    z = -sy * sp
    quat = torch.stack([w, x, y, z], dim=-1)
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vectors by wxyz quaternions."""
    q_w = quat[..., 0:1]
    q_vec = quat[..., 1:4]
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec, t, dim=-1)


@dataclass
class LadderSample:
    """Per-batch sampled ladder layout."""

    active_mask: torch.Tensor
    rung_heights: torch.Tensor
    num_active: torch.Tensor
    frame_pos: torch.Tensor
    frame_quat: torch.Tensor
    rung_pos_w: torch.Tensor
    rung_quat_w: torch.Tensor


def rung_local_centers_from_sample(
    sample: LadderSample,
    cfg: LadderConfig,
) -> torch.Tensor:
    """Rung-center positions in the ladder frame, shape ``[B, max_rungs, 3]``."""
    local = torch.zeros(
        sample.rung_heights.shape[0],
        cfg.max_rungs,
        3,
        device=sample.rung_heights.device,
        dtype=torch.float32,
    )
    local[..., 2] = sample.rung_heights
    inactive = torch.tensor(cfg.inactive_rung_pose, device=local.device, dtype=torch.float32)
    inactive = inactive.view(1, 1, 3).expand_as(local)
    return torch.where(sample.active_mask.unsqueeze(-1), local, inactive)


@dataclass(frozen=True)
class LadderTopology:
    """Compiled model indices for the fixed ladder topology."""

    entity_prefix: str
    frame_mocap_id: int
    frame_body_id: int
    rung_geom_ids: tuple[int, ...]
    rung_site_ids: tuple[tuple[int, ...], ...]
    site_y_offsets: tuple[float, ...]
    site_normal_offset: float
    """Ladder-frame −X offset from rung axis to latch target sites."""

    @classmethod
    def from_model(cls, model: mujoco.MjModel, cfg: LadderConfig, *, entity_name: str = "ladder") -> LadderTopology:
        prefix = f"{entity_name}/"
        frame_body_id = model.body(f"{prefix}frame").id
        frame_mocap_id = int(model.body_mocapid[frame_body_id])
        if frame_mocap_id < 0:
            raise RuntimeError(f"Ladder frame body {prefix}frame is not mocap.")

        rung_geom_ids: list[int] = []
        rung_site_ids: list[tuple[int, ...]] = []

        for rung_id in range(cfg.max_rungs):
            geom_name = f"{prefix}{expected_rung_geom_names(cfg)[rung_id]}"
            rung_geom_ids.append(model.geom(geom_name).id)
            site_ids = tuple(
                model.site(f"{prefix}{site_name}").id
                for site_name in expected_site_names(cfg, rung_id)
            )
            rung_site_ids.append(site_ids)

        return cls(
            entity_prefix=prefix,
            frame_mocap_id=frame_mocap_id,
            frame_body_id=frame_body_id,
            rung_geom_ids=tuple(rung_geom_ids),
            rung_site_ids=tuple(rung_site_ids),
            site_y_offsets=site_y_offsets(cfg),
            site_normal_offset=latch_site_normal_offset(cfg),
        )

    @property
    def max_rungs(self) -> int:
        return len(self.rung_geom_ids)


class LadderSampler:
    """Deterministic per-environment ladder sampling."""

    def __init__(self, cfg: LadderConfig, *, seed: int | None = None, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self._generator = torch.Generator(device=device)
        if seed is not None:
            self._generator.manual_seed(seed)

    def sample(self, num_envs: int) -> LadderSample:
        cfg = self.cfg
        active_counts = torch.randint(
            cfg.min_active_rungs,
            cfg.max_active_rungs + 1,
            (num_envs,),
            generator=self._generator,
            device=self.device,
        )
        active_mask = torch.zeros(num_envs, cfg.max_rungs, dtype=torch.bool, device=self.device)
        rung_heights = torch.zeros(num_envs, cfg.max_rungs, dtype=torch.float32, device=self.device)

        for env_idx in range(num_envs):
            count = int(active_counts[env_idx].item())
            active_mask[env_idx, :count] = True
            if count > 0:
                rung_heights[env_idx, 0] = cfg.base_height
            if count > 1:
                gaps = torch.empty(count - 1, device=self.device).uniform_(
                    cfg.spacing_min,
                    cfg.spacing_max,
                    generator=self._generator,
                )
                rung_heights[env_idx, 1:count] = cfg.base_height + torch.cumsum(gaps, dim=0)

        distance = torch.empty(num_envs, device=self.device).uniform_(
            cfg.ladder_distance_range[0],
            cfg.ladder_distance_range[1],
            generator=self._generator,
        )
        yaw = torch.empty(num_envs, device=self.device).uniform_(
            cfg.ladder_yaw_range[0],
            cfg.ladder_yaw_range[1],
            generator=self._generator,
        )
        pitch = torch.empty(num_envs, device=self.device).uniform_(
            cfg.ladder_tilt_range[0],
            cfg.ladder_tilt_range[1],
            generator=self._generator,
        )
        frame_quat = _quat_from_yaw_pitch(yaw, pitch)
        frame_pos = torch.zeros(num_envs, 3, device=self.device)
        frame_pos[:, 0] = distance

        rung_pos_w, rung_quat_w = self._compute_rung_poses(
            frame_pos=frame_pos,
            frame_quat=frame_quat,
            rung_heights=rung_heights,
            active_mask=active_mask,
        )
        return LadderSample(
            active_mask=active_mask,
            rung_heights=rung_heights,
            num_active=active_counts,
            frame_pos=frame_pos,
            frame_quat=frame_quat,
            rung_pos_w=rung_pos_w,
            rung_quat_w=rung_quat_w,
        )

    def _compute_rung_poses(
        self,
        *,
        frame_pos: torch.Tensor,
        frame_quat: torch.Tensor,
        rung_heights: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        num_envs = frame_pos.shape[0]
        local = torch.zeros(num_envs, cfg.max_rungs, 3, device=self.device)
        local[..., 2] = rung_heights
        inactive_local = torch.tensor(cfg.inactive_rung_pose, device=self.device, dtype=torch.float32)
        inactive = inactive_local.view(1, 1, 3).expand(num_envs, cfg.max_rungs, 3)
        local = torch.where(active_mask.unsqueeze(-1), local, inactive)

        frame_pos_exp = frame_pos.unsqueeze(1).expand(-1, cfg.max_rungs, -1)
        frame_quat_exp = frame_quat.unsqueeze(1).expand(-1, cfg.max_rungs, -1)
        rung_pos_w = _quat_apply(frame_quat_exp.reshape(-1, 4), local.reshape(-1, 3))
        rung_pos_w = rung_pos_w.reshape(num_envs, cfg.max_rungs, 3) + frame_pos_exp
        rung_quat_w = frame_quat.unsqueeze(1).expand(-1, cfg.max_rungs, -1).clone()
        return rung_pos_w, rung_quat_w


class LadderRuntime:
    """Attach to an mjlab env and drive batched ladder frame + rung layout."""

    EXTRA_KEY = "ladder_runtime"

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        cfg: LadderConfig,
        *,
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.topology = LadderTopology.from_model(env.sim.mj_model, cfg)
        self.sampler = LadderSampler(cfg, seed=seed, device=env.device)
        self.sample: LadderSample | None = None
        self._ensure_model_fields_expanded()

    @staticmethod
    def _ensure_model_fields_expanded_for(env: ManagerBasedRlEnv) -> None:
        missing = [
            field
            for field in _LADDER_RUNTIME_MODEL_FIELDS
            if field not in env.sim.expanded_fields
        ]
        if missing:
            env.sim.expand_model_fields(tuple(missing))

    def _ensure_model_fields_expanded(self) -> None:
        self._ensure_model_fields_expanded_for(self.env)

    @classmethod
    def attach(
        cls,
        env: ManagerBasedRlEnv,
        cfg: LadderConfig,
        *,
        seed: int | None = None,
    ) -> LadderRuntime:
        runtime = cls(env, cfg, seed=seed)
        env.extras[cls.EXTRA_KEY] = runtime
        return runtime

    @classmethod
    def get(cls, env: ManagerBasedRlEnv) -> LadderRuntime:
        runtime = env.extras.get(cls.EXTRA_KEY)
        if runtime is None:
            raise RuntimeError("LadderRuntime is not attached to the environment.")
        return runtime

    def resample(self, env_ids: torch.Tensor | None = None) -> LadderSample:
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.int64)
        if env_ids.numel() == self.env.num_envs:
            self.sample = self.sampler.sample(self.env.num_envs)
            self.apply(env_ids)
            return self.sample

        if self.sample is None:
            self.sample = self.sampler.sample(self.env.num_envs)

        replacement = self.sampler.sample(env_ids.numel())
        for field_name in (
            "active_mask",
            "rung_heights",
            "num_active",
            "frame_pos",
            "frame_quat",
            "rung_pos_w",
            "rung_quat_w",
        ):
            full = getattr(self.sample, field_name)
            full[env_ids] = getattr(replacement, field_name)
        self.apply(env_ids)
        return self.sample

    def apply(self, env_ids: torch.Tensor | None = None) -> None:
        if self.sample is None:
            raise RuntimeError("No ladder sample to apply; call resample() first.")
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.int64)

        sim = self.env.sim
        frame_id = self.topology.frame_mocap_id
        sim.data.mocap_pos[env_ids, frame_id] = self.sample.frame_pos[env_ids]
        sim.data.mocap_quat[env_ids, frame_id] = self.sample.frame_quat[env_ids]

        local_centers = rung_local_centers_from_sample(self.sample, self.cfg)
        site_x = -self.topology.site_normal_offset
        for rung_idx, geom_id in enumerate(self.topology.rung_geom_ids):
            center = local_centers[env_ids, rung_idx].clone()
            sim.model.geom_pos[env_ids, geom_id] = center
            for site_idx, site_id in enumerate(self.topology.rung_site_ids[rung_idx]):
                site_pos = center.clone()
                site_pos[:, 0] = center[:, 0] + site_x
                site_pos[:, 1] = self.topology.site_y_offsets[site_idx]
                sim.model.site_pos[env_ids, site_id] = site_pos

    def format_debug_lines(self, env_idx: int = 0) -> list[str]:
        if self.sample is None:
            return ["ladder: (no sample)"]
        sample = self.sample
        lines = [
            f"ladder env={env_idx} active={int(sample.num_active[env_idx].item())} "
            f"frame_pos={sample.frame_pos[env_idx].tolist()} "
            f"frame_quat={sample.frame_quat[env_idx].tolist()}",
        ]
        for rung_id in range(self.cfg.max_rungs):
            if not bool(sample.active_mask[env_idx, rung_id].item()):
                continue
            height = float(sample.rung_heights[env_idx, rung_id].item())
            pos = sample.rung_pos_w[env_idx, rung_id].tolist()
            lines.append(f"  rung {rung_id:02d}: height={height:.3f} pos_w={pos}")
        return lines

    def inactive_rung_indices(self) -> tuple[int, ...]:
        return tuple(range(self.cfg.max_rungs))

    @staticmethod
    def expected_geom_names(cfg: LadderConfig) -> tuple[str, ...]:
        return expected_rung_geom_names(cfg)
