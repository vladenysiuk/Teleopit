"""Climbing reward terms.

With mjlab ``scale_by_dt=True``, dense rates and sparse events must return
physical rates so that ``w * f * dt`` is independent of control frequency:

* progress: ``f = Δh_novel / dt`` → contribution ``w Δh_novel``
* events: ``f = 1_event / dt`` → contribution ``w``
* time mask: ``f = 1`` → contribution ``w dt``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.ladder.reward_state import ClimbRewardState
from train_mimic.tasks.climbing.mdp.common import (
    body_ladder_relative_height,
    gather_active_rung_heights_l,
)
from train_mimic.tasks.climbing.mdp.terminations import climbing_success_predicate

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _event_rate(env: ManagerBasedRlEnv, event: torch.Tensor) -> torch.Tensor:
    return event.to(dtype=torch.float32) / float(env.step_dt)


def upward_progress(env: ManagerBasedRlEnv, *, body_name: str = "d435i_link") -> torch.Tensor:
    """Novel maximum progress-body ladder-relative height rate ``[B]``.

    Default body is the head proxy ``d435i_link`` (G1 29-DoF has no head body).
    Uses ``max_rewarded_progress_height_l`` so rise–fall–rise below the historical
    maximum cannot farm progress. Total progress reward telescopes to
    ``w * (max_t h_t - h_0)``.
    """
    state = ClimbRewardState.get(env)
    height_l = body_ladder_relative_height(env, body_name=body_name)
    new_max = torch.maximum(state.max_rewarded_progress_height_l, height_l)
    delta = new_max - state.max_rewarded_progress_height_l
    state.max_rewarded_progress_height_l = new_max
    return delta / float(env.step_dt)


def new_higher_attachment(
    env: ManagerBasedRlEnv,
    *,
    height_eps: float = 1e-4,
) -> torch.Tensor:
    """One-off bonus rate when attaching above the prior paid height ``[B]``."""
    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    state = ClimbRewardState.get(env)
    candidate = torch.full(
        (env.num_envs,), float("-inf"), device=env.device, dtype=torch.float32
    )

    for hand_idx in range(2):
        attach_event = latch.state.attach_event[:, hand_idx]
        if not bool(torch.any(attach_event).item()):
            continue
        env_ids = torch.nonzero(attach_event, as_tuple=False).squeeze(-1)
        rung_ids = latch.state.rung_id[env_ids, hand_idx]
        heights = gather_active_rung_heights_l(env, rung_ids, env_ids=env_ids)
        candidate[env_ids] = torch.maximum(candidate[env_ids], heights)

    pays = candidate > (state.max_rewarded_attachment_height_l + float(height_eps))
    if torch.any(pays):
        state.max_rewarded_attachment_height_l = torch.where(
            pays, candidate, state.max_rewarded_attachment_height_l
        )
        state.valid_higher_attachment_count = state.valid_higher_attachment_count + pays.to(
            dtype=torch.int64
        )
    return _event_rate(env, pays)


def success_bonus(
    env: ManagerBasedRlEnv,
    *,
    body_name: str = "d435i_link",
    pelvis_clearance_below_top_l: float,
    top_attach_margin_l: float,
    min_attached_hands: int,
) -> torch.Tensor:
    """One-off success bonus rate on the first qualifying step ``[B]``."""
    state = ClimbRewardState.get(env)
    success_now = climbing_success_predicate(
        env,
        body_name=body_name,
        pelvis_clearance_below_top_l=pelvis_clearance_below_top_l,
        top_attach_margin_l=top_attach_margin_l,
        min_attached_hands=min_attached_hands,
    )
    first_success = success_now & (~state.success_rewarded)
    state.success_rewarded |= success_now
    if torch.any(first_success):
        state.time_to_success_steps[first_success] = env.episode_length_buf[first_success]
    return _event_rate(env, first_success)


def time_penalty_mask(
    env: ManagerBasedRlEnv,
    *,
    body_name: str = "d435i_link",
    pelvis_clearance_below_top_l: float,
    top_attach_margin_l: float,
    min_attached_hands: int,
) -> torch.Tensor:
    """Returns ``1`` while the episode has not yet reached success ``[B]``."""
    not_success = ~climbing_success_predicate(
        env,
        body_name=body_name,
        pelvis_clearance_below_top_l=pelvis_clearance_below_top_l,
        top_attach_margin_l=top_attach_margin_l,
        min_attached_hands=min_attached_hands,
    )
    return not_success.to(dtype=torch.float32)


def invalid_latch_request(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One penalty rate per invalid attach attempt (edge-triggered) ``[B]``."""
    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    state = ClimbRewardState.get(env)
    invalid = latch.state.invalid_attach_request.any(dim=-1)
    commanding_attach = latch.state.latch_command > 0.5
    any_attach_cmd = commanding_attach.any(dim=-1)

    pay = invalid & state.invalid_latch_armed
    state.invalid_latch_count = state.invalid_latch_count + pay.to(dtype=torch.int64)
    state.invalid_latch_armed = state.invalid_latch_armed.clone()
    state.invalid_latch_armed[pay] = False
    state.invalid_latch_armed[~any_attach_cmd] = True
    return _event_rate(env, pay)


def latch_overload_penalty(
    env: ManagerBasedRlEnv,
    *,
    force_threshold: float = 500.0,
) -> torch.Tensor:
    """Optional overload penalty when connect equality force exceeds threshold ``[B]``."""
    from train_mimic.tasks.climbing.mdp.terminations import latch_overload

    return latch_overload(env, force_threshold=force_threshold).to(dtype=torch.float32)
