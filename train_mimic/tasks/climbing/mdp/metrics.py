"""Climbing episode metrics.

Per-step indicators use mjlab ``reduce="mean"`` / ``"last"``. Running maxima
are class-based terms (mjlab MetricsTermCfg has no ``reduce="max"`` yet).
Stateful counters that rewards already maintain are reported with ``last``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.reward_state import ClimbRewardState
from train_mimic.tasks.climbing.mdp.common import body_ladder_relative_height
from train_mimic.tasks.climbing.mdp.terminations import climbing_success_predicate

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.metrics_manager import MetricsTermCfg


def _actuator_torque_saturated(env: ManagerBasedRlEnv, *, rtol: float = 0.98) -> torch.Tensor:
    robot = env.scene["robot"]
    force = getattr(robot.data, "actuator_force", None)
    if force is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    limits = getattr(robot.data, "actuator_force_limits", None)
    if limits is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.any(torch.abs(force) >= (limits * rtol), dim=-1)


class max_head_height_l:
    """Running maximum head / progress-body ladder-relative height (reduce=last).

    Body follows ``ClimbingRewardConfig.progress_body`` (default ``d435i_link``).
    """

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv) -> None:
        self._env = env
        self._body_name = str(cfg.params.get("body_name", "d435i_link"))
        self._max = torch.full(
            (env.num_envs,), float("-inf"), device=env.device, dtype=torch.float32
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # Managers construct before startup events; ladder may be absent at init.
        try:
            height = body_ladder_relative_height(self._env, body_name=self._body_name)
        except RuntimeError:
            self._max[env_ids] = float("-inf")
            return
        self._max[env_ids] = height[env_ids]

    def __call__(self, env: ManagerBasedRlEnv, *, body_name: str = "d435i_link") -> torch.Tensor:
        del body_name
        height = body_ladder_relative_height(env, body_name=self._body_name)
        self._max = torch.maximum(self._max, height)
        return self._max


def head_height_l(env: ManagerBasedRlEnv, *, body_name: str = "d435i_link") -> torch.Tensor:
    """Current head / progress-body ladder-relative height ``[B]``."""
    return body_ladder_relative_height(env, body_name=body_name)


def hand_contact_indicator(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Mean hand-contact indicator this step ``[B]`` in ``[0, 1]``."""
    contacts = ClimbContactState.get(env)
    contacts.update()
    return contacts.state.hand_in_contact.to(dtype=torch.float32).mean(dim=-1)


def torque_saturation_indicator(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Any-actuator torque saturation indicator this step ``[B]``."""
    return _actuator_torque_saturated(env).to(dtype=torch.float32)


def valid_higher_attachments(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Count of paid higher-attachment events this episode ``[B]``."""
    return ClimbRewardState.get(env).valid_higher_attachment_count.to(dtype=torch.float32)


def invalid_latch_count(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Count of invalid latch attempts this episode ``[B]``."""
    return ClimbRewardState.get(env).invalid_latch_count.to(dtype=torch.float32)


def episode_success(
    env: ManagerBasedRlEnv,
    *,
    body_name: str = "d435i_link",
    pelvis_clearance_below_top_l: float,
    top_attach_margin_l: float,
    min_attached_hands: int,
) -> torch.Tensor:
    """Binary success flag for the current step ``[B]``."""
    return climbing_success_predicate(
        env,
        body_name=body_name,
        pelvis_clearance_below_top_l=pelvis_clearance_below_top_l,
        top_attach_margin_l=top_attach_margin_l,
        min_attached_hands=min_attached_hands,
    ).to(dtype=torch.float32)


def time_to_success_steps(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Steps until first success, or ``-1`` if not yet successful ``[B]``."""
    return ClimbRewardState.get(env).time_to_success_steps.to(dtype=torch.float32)
