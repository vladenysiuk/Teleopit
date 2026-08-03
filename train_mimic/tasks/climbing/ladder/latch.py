"""Attach/detach latch backend and runtime state for General-Climbing-G1."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import mujoco
import torch

from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.latch import LatchConfig
from train_mimic.tasks.climbing.config.robot import (
    HAND_INDEX,
    HAND_NAMES,
    LEFT_HAND_POINT_SITE,
    RIGHT_HAND_POINT_SITE,
)
from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.generator import expected_site_names
from train_mimic.tasks.climbing.ladder.state import LadderRuntime, _quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class LatchCommand(IntEnum):
    """Discrete latch intent after hysteresis."""

    NEUTRAL = 0
    ATTACH = 1
    DETACH = 2


def latch_constraint_name(hand: str, rung_id: int, site_id: int) -> str:
    """Stable connect-equality name for one hand/rung/site triple."""
    return f"latch_{hand}_rung_{rung_id:02d}_site_{site_id:02d}"


def add_latch_equalities_to_spec(spec: mujoco.MjSpec, ladder_cfg: LadderConfig, latch_cfg: LatchConfig) -> None:
    """Predeclare inactive site connect constraints for all hand/rung/site pairs."""
    hand_sites = {
        "left": f"robot/{LEFT_HAND_POINT_SITE}",
        "right": f"robot/{RIGHT_HAND_POINT_SITE}",
    }
    for hand in HAND_NAMES:
        hand_site = hand_sites[hand]
        for rung_id in range(ladder_cfg.max_rungs):
            for site_name in expected_site_names(ladder_cfg, rung_id):
                ladder_site = f"ladder/{site_name}"
                eq = spec.add_equality(
                    type=mujoco.mjtEq.mjEQ_CONNECT,
                    name=latch_constraint_name(hand, rung_id, int(site_name.rsplit("_", 1)[-1])),
                    name1=hand_site,
                    name2=ladder_site,
                    objtype=mujoco.mjtObj.mjOBJ_SITE,
                    active=0,
                    solref=list(latch_cfg.solref),
                    solimp=list(latch_cfg.solimp),
                )
                del eq  # silence unused; spec owns the equality


@dataclass(frozen=True)
class LatchTopology:
    """Compiled equality indices for predeclared latch constraints."""

    eq_ids: torch.Tensor
    """Shape ``[2, max_rungs, sites_per_rung]`` int64 equality indices."""

    eq_names: tuple[str, ...]
    hand_site_ids: tuple[int, int]
    rung_site_ids: tuple[tuple[int, ...], ...]

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        ladder_cfg: LadderConfig,
        *,
        robot_prefix: str = "robot/",
        ladder_prefix: str = "ladder/",
    ) -> LatchTopology:
        eq_ids = torch.full(
            (len(HAND_NAMES), ladder_cfg.max_rungs, ladder_cfg.sites_per_rung),
            -1,
            dtype=torch.int64,
        )
        eq_names: list[str] = []
        for hand_idx, hand in enumerate(HAND_NAMES):
            for rung_id in range(ladder_cfg.max_rungs):
                for site_id in range(ladder_cfg.sites_per_rung):
                    name = latch_constraint_name(hand, rung_id, site_id)
                    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
                    if eq_id < 0:
                        raise RuntimeError(f"Missing latch equality constraint: {name}")
                    eq_ids[hand_idx, rung_id, site_id] = eq_id
                    eq_names.append(name)

        hand_site_ids = (
            model.site(f"{robot_prefix}{LEFT_HAND_POINT_SITE}").id,
            model.site(f"{robot_prefix}{RIGHT_HAND_POINT_SITE}").id,
        )
        rung_site_ids: list[tuple[int, ...]] = []
        for rung_id in range(ladder_cfg.max_rungs):
            site_ids = tuple(
                model.site(f"{ladder_prefix}{site_name}").id
                for site_name in expected_site_names(ladder_cfg, rung_id)
            )
            rung_site_ids.append(site_ids)

        expected_neq = len(HAND_NAMES) * ladder_cfg.max_rungs * ladder_cfg.sites_per_rung
        if model.neq != expected_neq:
            raise RuntimeError(
                f"Expected {expected_neq} latch equalities, compiled model has {model.neq}."
            )
        return cls(
            eq_ids=eq_ids,
            eq_names=tuple(eq_names),
            hand_site_ids=hand_site_ids,
            rung_site_ids=tuple(rung_site_ids),
        )


@dataclass
class HandLatchState:
    """Batched latch state for both hands."""

    attached: torch.Tensor
    rung_id: torch.Tensor
    site_id: torch.Tensor
    active_eq_id: torch.Tensor
    attach_event: torch.Tensor
    detach_event: torch.Tensor
    overload_detach_event: torch.Tensor
    overload_lockout: torch.Tensor
    """True until latch command returns to neutral after an overload break."""
    invalid_attach_request: torch.Tensor
    latch_command: torch.Tensor
    capture_radius: torch.Tensor
    """Per-environment attach eligibility radius (m), shape ``[B]``."""
    equality_force: torch.Tensor
    """Latest connect equality force magnitude (N) per hand, shape ``[B, 2]``."""


class LatchBackend(ABC):
    """Interface for activating/deactivating hand latch constraints."""

    @abstractmethod
    def set_active(
        self,
        env_ids: torch.Tensor,
        hand_idx: int,
        eq_id: int,
        *,
        active: bool,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def deactivate_hand(self, env_ids: torch.Tensor, hand_idx: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def deactivate_all(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_active_eq_ids(self, env_idx: int, hand_idx: int) -> list[int]:
        raise NotImplementedError


class ConnectEqualityLatchBackend(LatchBackend):
    """Activate predeclared MuJoCo connect equalities via ``eq_active``."""

    def __init__(self, env: ManagerBasedRlEnv, topology: LatchTopology) -> None:
        self._env = env
        self._topology = topology
        self._eq_active = env.sim.data.eq_active
        if self._eq_active.shape[1] != len(topology.eq_names):
            raise RuntimeError(
                "Simulation eq_active width does not match latch topology: "
                f"{self._eq_active.shape[1]} vs {len(topology.eq_names)}."
            )
        self._hand_active_eq = torch.full(
            (env.num_envs, len(HAND_NAMES)),
            -1,
            dtype=torch.int64,
            device=env.device,
        )

    @property
    def hand_active_eq(self) -> torch.Tensor:
        return self._hand_active_eq

    def set_active(
        self,
        env_ids: torch.Tensor,
        hand_idx: int,
        eq_id: int,
        *,
        active: bool,
    ) -> None:
        self.deactivate_hand(env_ids, hand_idx)
        if active:
            self._eq_active[env_ids, eq_id] = True
            self._hand_active_eq[env_ids, hand_idx] = eq_id
        else:
            self._hand_active_eq[env_ids, hand_idx] = -1

    def deactivate_hand(self, env_ids: torch.Tensor, hand_idx: int) -> None:
        prev = self._hand_active_eq[env_ids, hand_idx]
        valid = prev >= 0
        if torch.any(valid):
            active_envs = env_ids[valid]
            active_eq = prev[valid]
            self._eq_active[active_envs, active_eq] = False
        self._hand_active_eq[env_ids, hand_idx] = -1

    def deactivate_all(self, env_ids: torch.Tensor) -> None:
        for hand_idx in range(len(HAND_NAMES)):
            self.deactivate_hand(env_ids, hand_idx)

    def read_active_eq_ids(self, env_idx: int, hand_idx: int) -> list[int]:
        eq_id = int(self._hand_active_eq[env_idx, hand_idx].item())
        return [] if eq_id < 0 else [eq_id]


class ClimbLatchState:
    """Per-hand attach/detach latch with hysteresis and site selection."""

    EXTRA_KEY = "climb_latch_state"
    NONE_RUNG_ID = -1
    NONE_SITE_ID = -1

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        ladder_cfg: LadderConfig,
        latch_cfg: LatchConfig,
    ) -> None:
        self.env = env
        self.ladder_cfg = ladder_cfg
        self.latch_cfg = latch_cfg
        self.topology = LatchTopology.from_model(env.sim.mj_model, ladder_cfg)
        self.backend = ConnectEqualityLatchBackend(env, self.topology)
        self.state: HandLatchState | None = None

    @classmethod
    def attach(
        cls,
        env: ManagerBasedRlEnv,
        *,
        ladder_cfg: LadderConfig,
        latch_cfg: LatchConfig,
    ) -> ClimbLatchState:
        adapter = cls(env, ladder_cfg=ladder_cfg, latch_cfg=latch_cfg)
        adapter._ensure_state()
        env.extras[cls.EXTRA_KEY] = adapter
        return adapter

    @classmethod
    def get(cls, env: ManagerBasedRlEnv) -> ClimbLatchState:
        adapter = env.extras.get(cls.EXTRA_KEY)
        if adapter is None:
            raise RuntimeError("ClimbLatchState is not attached to the environment.")
        return adapter

    def _ensure_state(self) -> HandLatchState:
        if self.state is None:
            batch = self.env.num_envs
            device = self.env.device
            num_hands = len(HAND_NAMES)
            self.state = HandLatchState(
                attached=torch.zeros(batch, num_hands, dtype=torch.bool, device=device),
                rung_id=torch.full(
                    (batch, num_hands),
                    self.NONE_RUNG_ID,
                    dtype=torch.int64,
                    device=device,
                ),
                site_id=torch.full(
                    (batch, num_hands),
                    self.NONE_SITE_ID,
                    dtype=torch.int64,
                    device=device,
                ),
                active_eq_id=torch.full(
                    (batch, num_hands),
                    -1,
                    dtype=torch.int64,
                    device=device,
                ),
                attach_event=torch.zeros(batch, num_hands, dtype=torch.bool, device=device),
                detach_event=torch.zeros(batch, num_hands, dtype=torch.bool, device=device),
                overload_detach_event=torch.zeros(
                    batch, num_hands, dtype=torch.bool, device=device
                ),
                overload_lockout=torch.zeros(batch, num_hands, dtype=torch.bool, device=device),
                invalid_attach_request=torch.zeros(batch, num_hands, dtype=torch.bool, device=device),
                latch_command=torch.zeros(batch, num_hands, dtype=torch.float32, device=device),
                capture_radius=torch.full(
                    (batch,),
                    float(self.latch_cfg.capture_radius),
                    dtype=torch.float32,
                    device=device,
                ),
                equality_force=torch.zeros(batch, num_hands, dtype=torch.float32, device=device),
            )
        return self.state

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Deactivate equalities first, then clear Python latch state."""
        if env_ids is None:
            env_ids = slice(None)
        if isinstance(env_ids, slice):
            env_ids_tensor = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.int64)
        else:
            env_ids_tensor = env_ids

        # Deactivate before clearing IDs so stale connect constraints cannot
        # pull against resampled ladder sites on the next step.
        self.backend.deactivate_all(env_ids_tensor)

        if self.state is not None:
            self.state.attached[env_ids] = False
            self.state.rung_id[env_ids] = self.NONE_RUNG_ID
            self.state.site_id[env_ids] = self.NONE_SITE_ID
            self.state.active_eq_id[env_ids] = -1
            self.state.attach_event[env_ids] = False
            self.state.detach_event[env_ids] = False
            self.state.overload_detach_event[env_ids] = False
            self.state.overload_lockout[env_ids] = False
            self.state.invalid_attach_request[env_ids] = False
            self.state.latch_command[env_ids] = 0.0
            self.state.capture_radius[env_ids] = float(self.latch_cfg.capture_radius)
            self.state.equality_force[env_ids] = 0.0

    @staticmethod
    def decode_command(raw: torch.Tensor, latch_cfg: LatchConfig) -> torch.Tensor:
        """Map continuous latch actions to discrete intents."""
        cmd = torch.full_like(raw, LatchCommand.NEUTRAL, dtype=torch.int64)
        cmd = torch.where(raw > latch_cfg.attach_threshold, int(LatchCommand.ATTACH), cmd)
        cmd = torch.where(raw < latch_cfg.detach_threshold, int(LatchCommand.DETACH), cmd)
        return cmd

    def _hand_site_pos_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        site_xpos = self.env.sim.data.site_xpos
        hand_ids = torch.tensor(self.topology.hand_site_ids, device=self.env.device, dtype=torch.int64)
        return site_xpos[env_ids][:, hand_ids, :]

    def _rung_site_pos_w(self, env_ids: torch.Tensor, rung_id: int) -> torch.Tensor:
        site_xpos = self.env.sim.data.site_xpos
        site_ids = torch.tensor(
            self.topology.rung_site_ids[rung_id],
            device=self.env.device,
            dtype=torch.int64,
        )
        return site_xpos[env_ids][:, site_ids, :]

    def _select_nearest_site(
        self,
        env_ids: torch.Tensor,
        hand_idx: int,
        rung_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hand_pos = self._hand_site_pos_w(env_ids)[:, hand_idx, :]
        batch = env_ids.numel()
        site_id = torch.full((batch,), self.NONE_SITE_ID, dtype=torch.int64, device=self.env.device)
        valid = torch.zeros(batch, dtype=torch.bool, device=self.env.device)
        for local_idx in range(batch):
            rid = int(rung_ids[local_idx].item())
            site_pos = self._rung_site_pos_w(env_ids[local_idx : local_idx + 1], rid)[0]
            dist = torch.linalg.norm(site_pos - hand_pos[local_idx], dim=-1)
            radius = float(self.state.capture_radius[env_ids[local_idx]].item())
            eligible = dist <= radius
            if not bool(torch.any(eligible).item()):
                continue
            masked = torch.where(eligible, dist, torch.full_like(dist, 1.0e9))
            best_site = int(masked.argmin().item())
            site_id[local_idx] = best_site
            valid[local_idx] = True
        return site_id, valid

    def _equality_force_magnitudes(self) -> torch.Tensor:
        """Connect equality force magnitude (N) for each attached hand ``[B, 2]``.

        Uses MuJoCo / MuJoCo-Warp ``efc.force`` rows whose ``efc.id`` matches the
        active latch equality. Contact-sensor force is *not* a substitute: a
        latched hand can transmit large equality loads while contact normal is
        near zero.
        """
        state = self._ensure_state()
        forces = torch.zeros_like(state.equality_force)
        efc = getattr(self.env.sim.data, "efc", None)
        if efc is None or not hasattr(efc, "force") or not hasattr(efc, "id"):
            return forces

        efc_force = efc.force
        efc_id = efc.id
        # Warp pads efc rows; only the first ``ne`` entries are equality rows.
        ne = getattr(self.env.sim.data, "ne", None)

        for hand_idx in range(len(HAND_NAMES)):
            attached = state.attached[:, hand_idx]
            if not bool(torch.any(attached).item()):
                continue
            env_ids = torch.nonzero(attached, as_tuple=False).squeeze(-1)
            for env_idx in env_ids.tolist():
                eq_id = int(state.active_eq_id[env_idx, hand_idx].item())
                if eq_id < 0:
                    continue
                n_eq = int(ne[env_idx].item()) if ne is not None else int(efc_force.shape[1])
                if n_eq <= 0:
                    continue
                ids = efc_id[env_idx, :n_eq]
                mask = ids == eq_id
                if not bool(torch.any(mask).item()):
                    continue
                components = efc_force[env_idx, :n_eq][mask]
                forces[env_idx, hand_idx] = torch.linalg.norm(components)
        return forces

    def apply_overload_breaks(self) -> HandLatchState:
        """Detach hands whose connect equality force exceeds ``break_force``.

        Intended to run every physics substep (``LatchAction.apply_actions``) so
        overload cannot accumulate across a full policy step.
        """
        state = self._ensure_state()
        state.overload_detach_event[:] = False
        state.equality_force[:] = self._equality_force_magnitudes()

        break_force = self.latch_cfg.break_force
        if break_force is None:
            return state

        threshold = float(break_force)
        env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.int64)
        for hand_idx in range(len(HAND_NAMES)):
            overload = state.attached[:, hand_idx] & (state.equality_force[:, hand_idx] > threshold)
            if not bool(torch.any(overload).item()):
                continue
            ids = env_ids[overload]
            self.backend.deactivate_hand(ids, hand_idx)
            state.attached[ids, hand_idx] = False
            state.rung_id[ids, hand_idx] = self.NONE_RUNG_ID
            state.site_id[ids, hand_idx] = self.NONE_SITE_ID
            state.active_eq_id[ids, hand_idx] = -1
            state.detach_event[ids, hand_idx] = True
            state.overload_detach_event[ids, hand_idx] = True
            state.overload_lockout[ids, hand_idx] = True
        return state

    def update(self, raw_latch_actions: torch.Tensor) -> HandLatchState:
        """Apply hysteresis latch commands and sync equality constraints."""
        state = self._ensure_state()
        state.attach_event[:] = False
        state.detach_event[:] = False
        state.overload_detach_event[:] = False
        state.invalid_attach_request[:] = False
        state.latch_command[:] = raw_latch_actions

        contacts = ClimbContactState.get(self.env).update()
        commands = self.decode_command(raw_latch_actions, self.latch_cfg)
        env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.int64)

        for hand_idx, hand in enumerate(HAND_NAMES):
            cmd = commands[:, hand_idx]
            # Clear overload lockout once the latch command leaves ATTACH.
            unlock = state.overload_lockout[:, hand_idx] & (cmd != int(LatchCommand.ATTACH))
            if torch.any(unlock):
                state.overload_lockout[env_ids[unlock], hand_idx] = False

            attached = state.attached[:, hand_idx]
            detach_mask = attached & (cmd == int(LatchCommand.DETACH))
            if torch.any(detach_mask):
                ids = env_ids[detach_mask]
                self.backend.deactivate_hand(ids, hand_idx)
                state.attached[ids, hand_idx] = False
                state.rung_id[ids, hand_idx] = self.NONE_RUNG_ID
                state.site_id[ids, hand_idx] = self.NONE_SITE_ID
                state.active_eq_id[ids, hand_idx] = -1
                state.detach_event[ids, hand_idx] = True

            attach_mask = (
                (~state.attached[:, hand_idx])
                & (cmd == int(LatchCommand.ATTACH))
                & (~state.overload_lockout[:, hand_idx])
            )
            if torch.any(attach_mask):
                ids = env_ids[attach_mask]
                in_contact = contacts.hand_in_contact[ids, hand_idx]
                rung_id = contacts.hand_rung_id[ids, hand_idx]

                invalid = ~in_contact
                if torch.any(invalid):
                    bad_ids = ids[invalid]
                    state.invalid_attach_request[bad_ids, hand_idx] = True

                valid = in_contact
                if torch.any(valid):
                    valid_ids = ids[valid]
                    valid_rung = rung_id[valid]
                    site_id, site_ok = self._select_nearest_site(valid_ids, hand_idx, valid_rung)

                    bad_site = ~site_ok
                    if torch.any(bad_site):
                        state.invalid_attach_request[valid_ids[bad_site], hand_idx] = True

                    good = site_ok
                    if torch.any(good):
                        good_ids = valid_ids[good]
                        good_rung = valid_rung[good]
                        good_site = site_id[good]
                        for local_idx in range(good_ids.numel()):
                            eid = good_ids[local_idx : local_idx + 1]
                            rid = int(good_rung[local_idx].item())
                            sid = int(good_site[local_idx].item())
                            eq_id = int(self.topology.eq_ids[hand_idx, rid, sid].item())
                            self.backend.set_active(eid, hand_idx, eq_id, active=True)
                            state.attached[eid, hand_idx] = True
                            state.rung_id[eid, hand_idx] = rid
                            state.site_id[eid, hand_idx] = sid
                            state.active_eq_id[eid, hand_idx] = eq_id
                            state.attach_event[eid, hand_idx] = True

            locked_attach = (
                (~state.attached[:, hand_idx])
                & (cmd == int(LatchCommand.ATTACH))
                & state.overload_lockout[:, hand_idx]
            )
            if torch.any(locked_attach):
                state.invalid_attach_request[env_ids[locked_attach], hand_idx] = True

            # Attached hands ignore attach/neutral; touching another rung does not switch.
            still_attached = state.attached[:, hand_idx]
            if torch.any(still_attached):
                ids = env_ids[still_attached]
                state.active_eq_id[ids, hand_idx] = self.backend.hand_active_eq[ids, hand_idx]

        return state

    def geometry_diagnostics(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Latch/contact geometry metrics for Stage 3 validation.

        Returns tensors shaped ``[N, 2]`` (left/right) unless noted:

        - ``hand_to_rung_axis_dist``: hand-sphere centre to rung axis distance
        - ``signed_separation``: axis distance − (r_rung + r_hand); negative = penetration
        - ``contact_normal_force``: contact-frame |force_x| from ClimbContactState
        - ``equality_residual``: ||hand_site − attached_rung_site|| (inf if detached)
        """
        from train_mimic.tasks.climbing.config.robot import HAND_POINT_RADIUS

        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.int64)
        state = self._ensure_state()
        contacts = ClimbContactState.get(self.env)
        if contacts.state is None:
            contacts.update()
        assert contacts.state is not None

        n = env_ids.numel()
        device = self.env.device
        axis_dist = torch.full((n, 2), float("inf"), dtype=torch.float32, device=device)
        signed_sep = torch.full((n, 2), float("inf"), dtype=torch.float32, device=device)
        eq_residual = torch.full((n, 2), float("inf"), dtype=torch.float32, device=device)
        contact_force = contacts.state.hand_contact_force[env_ids].clone()

        hand_pos = self._hand_site_pos_w(env_ids)
        runtime = LadderRuntime.get(self.env)
        if runtime.sample is None:
            raise RuntimeError("LadderRuntime has no sample.")
        sum_radii = self.ladder_cfg.rung_radius + HAND_POINT_RADIUS

        for hand_idx in range(len(HAND_NAMES)):
            attached = state.attached[env_ids, hand_idx]
            if not bool(torch.any(attached).item()):
                # Still report geometry for hands currently in contact.
                in_contact = contacts.state.hand_in_contact[env_ids, hand_idx]
                if not bool(torch.any(in_contact).item()):
                    continue
                rung_ids = contacts.state.hand_rung_id[env_ids, hand_idx]
            else:
                in_contact = attached
                rung_ids = state.rung_id[env_ids, hand_idx]

            for local_idx in range(n):
                if not bool(in_contact[local_idx].item()):
                    continue
                rid = int(rung_ids[local_idx].item())
                if rid < 0:
                    continue
                eid = env_ids[local_idx : local_idx + 1]
                rung_center = runtime.sample.rung_pos_w[eid, rid]
                frame_quat = runtime.sample.frame_quat[eid]
                # Axis = ladder-frame Y through rung centre.
                local_y = torch.zeros(1, 3, device=device, dtype=torch.float32)
                local_y[:, 1] = 1.0
                axis_dir = _quat_apply(frame_quat, local_y)
                axis_dir = axis_dir / torch.linalg.norm(axis_dir, dim=-1, keepdim=True).clamp_min(1e-8)
                delta = hand_pos[local_idx : local_idx + 1, hand_idx] - rung_center
                along = (delta * axis_dir).sum(dim=-1, keepdim=True)
                radial = delta - along * axis_dir
                dist = torch.linalg.norm(radial, dim=-1)
                axis_dist[local_idx, hand_idx] = dist
                signed_sep[local_idx, hand_idx] = dist - sum_radii

                if bool(state.attached[eid, hand_idx].item()):
                    sid = int(state.site_id[eid, hand_idx].item())
                    rung_site = self.env.sim.data.site_xpos[eid, self.topology.rung_site_ids[rid][sid]]
                    eq_residual[local_idx, hand_idx] = torch.linalg.norm(
                        hand_pos[local_idx : local_idx + 1, hand_idx] - rung_site
                    )

        return {
            "hand_to_rung_axis_dist": axis_dist,
            "signed_separation": signed_sep,
            "contact_normal_force": contact_force,
            "equality_residual": eq_residual,
        }

    def format_debug_lines(self, env_idx: int = 0) -> list[str]:
        if self.state is None:
            return ["latch: (no state)"]
        lines = [f"latch env={env_idx}"]
        for hand in HAND_NAMES:
            idx = HAND_INDEX[hand]
            attached = bool(self.state.attached[env_idx, idx].item())
            rung_id = int(self.state.rung_id[env_idx, idx].item())
            site_id = int(self.state.site_id[env_idx, idx].item())
            eq_id = int(self.state.active_eq_id[env_idx, idx].item())
            cmd = float(self.state.latch_command[env_idx, idx].item())
            invalid = bool(self.state.invalid_attach_request[env_idx, idx].item())
            eq_name = "none"
            if eq_id >= 0:
                eq_name = self.topology.eq_names[eq_id]
            lines.append(
                f"  {hand}: attached={attached} rung_id={rung_id} site_id={site_id} "
                f"eq={eq_name} cmd={cmd:.2f} invalid_attach={invalid}"
            )
        env_ids = torch.tensor([env_idx], device=self.env.device, dtype=torch.int64)
        try:
            diag = self.geometry_diagnostics(env_ids)
        except RuntimeError:
            return lines
        for hand in HAND_NAMES:
            idx = HAND_INDEX[hand]
            axis_d = float(diag["hand_to_rung_axis_dist"][0, idx].item())
            sep = float(diag["signed_separation"][0, idx].item())
            force = float(diag["contact_normal_force"][0, idx].item())
            resid = float(diag["equality_residual"][0, idx].item())
            if not torch.isfinite(torch.tensor(axis_d)):
                continue
            lines.append(
                f"  {hand} geom: axis_dist={axis_d:.4f} signed_sep={sep:.4f} "
                f"n_force={force:.2f} eq_resid={resid:.4f}"
            )
        return lines

    def expected_latch_equality_count(self) -> int:
        return len(HAND_NAMES) * self.ladder_cfg.max_rungs * self.ladder_cfg.sites_per_rung
