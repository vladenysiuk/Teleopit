"""Per-episode reward event memory (reward-needed quantities only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass
class ClimbRewardState:
    """Batched reward event memory.

    Metrics that only need per-step indicators (contact, torque saturation,
    current head height) are handled by the metrics manager. This store keeps
    quantities that reward terms themselves must remember across steps.
    """

    max_rewarded_progress_height_l: torch.Tensor
    max_rewarded_attachment_height_l: torch.Tensor
    success_rewarded: torch.Tensor
    valid_higher_attachment_count: torch.Tensor
    invalid_latch_count: torch.Tensor
    invalid_latch_armed: torch.Tensor
    time_to_success_steps: torch.Tensor

    EXTRA_KEY = "climb_reward_state"

    @classmethod
    def attach(cls, env: ManagerBasedRlEnv) -> ClimbRewardState:
        batch = env.num_envs
        device = env.device
        state = cls(
            max_rewarded_progress_height_l=torch.zeros(batch, device=device, dtype=torch.float32),
            max_rewarded_attachment_height_l=torch.full(
                (batch,), float("-inf"), device=device, dtype=torch.float32
            ),
            success_rewarded=torch.zeros(batch, dtype=torch.bool, device=device),
            valid_higher_attachment_count=torch.zeros(batch, device=device, dtype=torch.int64),
            invalid_latch_count=torch.zeros(batch, device=device, dtype=torch.int64),
            invalid_latch_armed=torch.ones(batch, dtype=torch.bool, device=device),
            time_to_success_steps=torch.full((batch,), -1, device=device, dtype=torch.int64),
        )
        env.extras[cls.EXTRA_KEY] = state
        return state

    @classmethod
    def get(cls, env: ManagerBasedRlEnv) -> ClimbRewardState:
        state = env.extras.get(cls.EXTRA_KEY)
        if state is None:
            raise RuntimeError("ClimbRewardState is not attached to the environment.")
        return state

    def reset(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None,
        *,
        progress_body: str,
    ) -> None:
        """Clear per-episode memory and anchor progress tracking to the current pose."""
        if env_ids is None:
            env_ids = slice(None)
        from train_mimic.tasks.climbing.mdp.common import body_ladder_relative_height

        current_h = body_ladder_relative_height(env, body_name=progress_body)
        self.max_rewarded_progress_height_l[env_ids] = current_h[env_ids]
        self.max_rewarded_attachment_height_l[env_ids] = float("-inf")
        self.success_rewarded[env_ids] = False
        self.valid_higher_attachment_count[env_ids] = 0
        self.invalid_latch_count[env_ids] = 0
        self.invalid_latch_armed[env_ids] = True
        self.time_to_success_steps[env_ids] = -1
