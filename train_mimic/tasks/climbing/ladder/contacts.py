"""Hand–ladder contact identity for General-Climbing-G1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from train_mimic.tasks.climbing.config.robot import HAND_INDEX, HAND_NAMES
from train_mimic.tasks.climbing.ladder.state import LadderRuntime

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass
class HandContactState:
    """Batched hand–rung contact identity."""

    hand_in_contact: torch.Tensor
    hand_rung_id: torch.Tensor
    hand_rung_height: torch.Tensor
    hand_contact_force: torch.Tensor
    hand_contact_pos: torch.Tensor


class ClimbContactState:
    """Resolve per-hand ladder contact from mjlab contact sensors.

    Multiple simultaneous contacts on one hand are reduced deterministically:
    among **active** rungs with ``found > 0``, pick the rung whose contact
    normal force is largest. Ties prefer the **lower** rung index.

    Normal force is ``|force[..., 0]|`` from mjlab ``ContactSensor`` with
    ``global_frame=False`` (default): force is expressed in the MuJoCo contact
    frame where column 0 is the contact normal (primary → secondary).
    """

    EXTRA_KEY = "climb_contact_state"
    NONE_RUNG_ID = -1
    SENSOR_PREFIX = "hand_rung_contact"

    def __init__(self, env: ManagerBasedRlEnv, *, max_rungs: int) -> None:
        self.env = env
        self._max_rungs = max_rungs
        self._sensor_names = {
            hand: tuple(
                f"{self.SENSOR_PREFIX}_{hand}_rung_{rung_id:02d}"
                for rung_id in range(self._max_rungs)
            )
            for hand in HAND_NAMES
        }
        self.state: HandContactState | None = None

    @classmethod
    def attach(
        cls,
        env: ManagerBasedRlEnv,
        *,
        max_rungs: int,
    ) -> ClimbContactState:
        adapter = cls(env, max_rungs=max_rungs)
        env.extras[cls.EXTRA_KEY] = adapter
        return adapter

    @classmethod
    def get(cls, env: ManagerBasedRlEnv) -> ClimbContactState:
        adapter = env.extras.get(cls.EXTRA_KEY)
        if adapter is None:
            raise RuntimeError("ClimbContactState is not attached to the environment.")
        return adapter

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self.state is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self.state.hand_in_contact[env_ids] = False
        self.state.hand_rung_id[env_ids] = self.NONE_RUNG_ID
        self.state.hand_rung_height[env_ids] = 0.0
        self.state.hand_contact_force[env_ids] = 0.0
        self.state.hand_contact_pos[env_ids] = 0.0

    def update(self) -> HandContactState:
        env = self.env
        runtime = LadderRuntime.get(env)
        if runtime.sample is None:
            raise RuntimeError("LadderRuntime has no sample; resample before contact update.")

        batch = env.num_envs
        device = env.device
        num_hands = len(HAND_NAMES)

        if self.state is None:
            self.state = HandContactState(
                hand_in_contact=torch.zeros(batch, num_hands, dtype=torch.bool, device=device),
                hand_rung_id=torch.full(
                    (batch, num_hands),
                    self.NONE_RUNG_ID,
                    dtype=torch.int64,
                    device=device,
                ),
                hand_rung_height=torch.zeros(batch, num_hands, dtype=torch.float32, device=device),
                hand_contact_force=torch.zeros(batch, num_hands, dtype=torch.float32, device=device),
                hand_contact_pos=torch.zeros(batch, num_hands, 3, dtype=torch.float32, device=device),
            )

        active_mask = runtime.sample.active_mask
        rung_heights = runtime.sample.rung_heights

        for hand in HAND_NAMES:
            hand_idx = HAND_INDEX[hand]
            best_force = torch.zeros(batch, dtype=torch.float32, device=device)
            best_rung = torch.full((batch,), self.NONE_RUNG_ID, dtype=torch.int64, device=device)
            best_pos = torch.zeros(batch, 3, dtype=torch.float32, device=device)

            for rung_id, sensor_name in enumerate(self._sensor_names[hand]):
                sensor = env.scene[sensor_name]
                data = sensor.data
                found = data.found
                if found is None:
                    continue
                if found.ndim == 1:
                    found = found.unsqueeze(-1)
                in_contact = (found[:, 0] > 0) & active_mask[:, rung_id]

                force = data.force
                if force is None:
                    normal_force = torch.zeros(batch, dtype=torch.float32, device=device)
                else:
                    normal_force = force[:, 0, 0].abs()

                pos = data.pos
                if pos is None:
                    contact_pos = torch.zeros(batch, 3, dtype=torch.float32, device=device)
                else:
                    contact_pos = pos[:, 0, :]

                lower_index_wins = (best_rung < 0) | (rung_id < best_rung)
                replace = in_contact & (
                    (normal_force > best_force)
                    | ((normal_force == best_force) & lower_index_wins)
                )
                best_force = torch.where(replace, normal_force, best_force)
                best_rung = torch.where(replace, torch.full_like(best_rung, rung_id), best_rung)
                best_pos = torch.where(replace.unsqueeze(-1), contact_pos, best_pos)

            in_contact = best_rung >= 0
            self.state.hand_in_contact[:, hand_idx] = in_contact
            self.state.hand_rung_id[:, hand_idx] = torch.where(
                in_contact,
                best_rung,
                torch.full_like(best_rung, self.NONE_RUNG_ID),
            )
            self.state.hand_contact_force[:, hand_idx] = torch.where(in_contact, best_force, torch.zeros_like(best_force))
            self.state.hand_contact_pos[:, hand_idx] = best_pos
            height_values = rung_heights.gather(1, best_rung.clamp_min(0).unsqueeze(1)).squeeze(1)
            self.state.hand_rung_height[:, hand_idx] = torch.where(
                in_contact,
                height_values,
                torch.zeros_like(height_values),
            )

        return self.state

    def format_debug_lines(self, env_idx: int = 0) -> list[str]:
        if self.state is None:
            return ["contacts: (no state)"]
        lines = [f"contacts env={env_idx}"]
        for hand in HAND_NAMES:
            idx = HAND_INDEX[hand]
            in_contact = bool(self.state.hand_in_contact[env_idx, idx].item())
            rung_id = int(self.state.hand_rung_id[env_idx, idx].item())
            height = float(self.state.hand_rung_height[env_idx, idx].item())
            force = float(self.state.hand_contact_force[env_idx, idx].item())
            pos = self.state.hand_contact_pos[env_idx, idx].tolist()
            lines.append(
                f"  {hand}: contact={in_contact} rung_id={rung_id} "
                f"height={height:.3f} force={force:.3f} pos={pos}"
            )
        return lines


def hand_rung_contact_sensor_names(cfg_max_rungs: int) -> tuple[str, ...]:
    """Stable sensor names for each hand/rung contact pair."""
    names: list[str] = []
    for hand in HAND_NAMES:
        for rung_id in range(cfg_max_rungs):
            names.append(f"{ClimbContactState.SENSOR_PREFIX}_{hand}_rung_{rung_id:02d}")
    return tuple(names)
