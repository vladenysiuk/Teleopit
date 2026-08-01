"""Climbing termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.mdp.common import (
    body_height_w,
    body_ladder_frame_distance_xy,
    body_ladder_relative_height,
    gather_active_rung_heights_l,
    success_goal_height_l,
    top_active_rung_height_l,
    top_active_rung_id,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def climbing_success_predicate(
    env: ManagerBasedRlEnv,
    *,
    body_name: str = "pelvis",
    pelvis_clearance_below_top_l: float,
    top_attach_margin_l: float,
    min_attached_hands: int,
) -> torch.Tensor:
    """Success predicate shared by reward bonus and termination ``[B]`` bool.

    Requires pelvis height at/above a ladder-relative goal derived from the top
    active rung, at least ``min_attached_hands`` attached, and at least one hand
    attached near the top rung (same rung id or within ``top_attach_margin_l``).
    """
    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    height_l = body_ladder_relative_height(env, body_name=body_name)
    goal_h = success_goal_height_l(
        env, pelvis_clearance_below_top_l=pelvis_clearance_below_top_l
    )
    attached = latch.state.attached
    support_ok = attached.sum(dim=-1) >= min_attached_hands
    height_ok = height_l >= goal_h

    top_h = top_active_rung_height_l(env)
    top_id = top_active_rung_id(env)
    near_top = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for hand_idx in range(2):
        hand_attached = attached[:, hand_idx]
        if not bool(torch.any(hand_attached).item()):
            continue
        env_ids = torch.nonzero(hand_attached, as_tuple=False).squeeze(-1)
        rung_ids = latch.state.rung_id[env_ids, hand_idx]
        rung_h = gather_active_rung_heights_l(env, rung_ids, env_ids=env_ids)
        on_top = rung_ids == top_id[env_ids]
        within_margin = rung_h >= (top_h[env_ids] - float(top_attach_margin_l))
        near_top[env_ids] |= on_top | within_margin
    return height_ok & support_ok & near_top


def climbing_success(
    env: ManagerBasedRlEnv,
    *,
    body_name: str = "pelvis",
    pelvis_clearance_below_top_l: float,
    top_attach_margin_l: float,
    min_attached_hands: int,
) -> torch.Tensor:
    """Terminate when the configured success predicate is satisfied."""
    return climbing_success_predicate(
        env,
        body_name=body_name,
        pelvis_clearance_below_top_l=pelvis_clearance_below_top_l,
        top_attach_margin_l=top_attach_margin_l,
        min_attached_hands=min_attached_hands,
    )


def fallen_or_far_from_ladder(
    env: ManagerBasedRlEnv,
    *,
    body_name: str = "pelvis",
    minimum_height_w: float,
    max_ladder_distance_xy: float,
) -> torch.Tensor:
    """Terminate when the robot falls or leaves the ladder bay."""
    too_low = body_height_w(env, body_name=body_name) < minimum_height_w
    too_far = body_ladder_frame_distance_xy(env, body_name=body_name) > max_ladder_distance_xy
    return too_low | too_far


def latch_overload(
    env: ManagerBasedRlEnv,
    *,
    force_threshold: float,
) -> torch.Tensor:
    """Optional termination when attached-hand contact force exceeds threshold."""
    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    contacts = ClimbContactState.get(env)
    contacts.update()
    overload = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for hand_idx in range(2):
        attached = latch.state.attached[:, hand_idx]
        if not bool(torch.any(attached).item()):
            continue
        force = contacts.state.hand_contact_force[:, hand_idx]
        overload |= attached & (force > force_threshold)
    return overload
