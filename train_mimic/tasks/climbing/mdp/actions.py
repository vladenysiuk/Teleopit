"""Climbing latch action term."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.latch import LatchConfig
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class LatchActionCfg(ActionTermCfg):
    """Two-dimensional attach/detach latch commands (left, right)."""

    ladder_cfg: LadderConfig
    latch_cfg: LatchConfig = LatchConfig()

    def build(self, env: ManagerBasedRlEnv) -> ActionTerm:
        return LatchAction(cfg=self, env=env)


class LatchAction(ActionTerm):
    """Process hysteresis latch commands and drive equality constraints."""

    cfg: LatchActionCfg

    def __init__(self, cfg: LatchActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg=cfg, env=env)
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._latch: ClimbLatchState | None = None

    def _get_latch(self) -> ClimbLatchState:
        if self._latch is None:
            self._latch = ClimbLatchState.get(self._env)
        return self._latch

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions.to(self.device)
        self._get_latch().update(self._raw_actions)

    def apply_actions(self) -> None:
        # Overload break must run every physics substep using the previous step's
        # equality forces; process_actions alone would let a slingshot accumulate
        # across the full decimation window.
        self._get_latch().apply_overload_breaks()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
