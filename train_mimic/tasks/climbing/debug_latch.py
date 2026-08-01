"""Debug helpers for latch attach/detach inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from train_mimic.tasks.climbing.config.latch import LatchConfig
from train_mimic.tasks.climbing.mdp.observations import action_term_slice
from train_mimic.tasks.climbing.debug_probe import (
    nudge_robot_away_from_ladder,
    nudge_robot_toward_rung,
    place_robot_hand_near_rung,
    restore_active_rungs_default_collision,
    restore_all_active_rungs_to_ladder,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

MotionMode = Literal[
    "hold_ladder",
    "slide_robot_to_rung",
    "pull_robot",
    "hold",
]

# Rung indices on the fixed 6-active-rung debug ladder (heights increase with id).
# Auto script is left-hand only (standing pose cannot reliably dual-latch).
_LEFT_RUNG_A = 2
_LEFT_RUNG_B = 3

# Hand-centre → rung-centre gap along ladder +X. Resting contact ≈ r_rung+r_hand
# (0.045 m with defaults). Use a slight overshoot so contact sensors stay true
# through the one-step attach delay.
_CLEARANCE_SEPARATION_M = 0.12
_CONTACT_SEPARATION_M = 0.040
_PRESS_SEPARATION_M = 0.038


@dataclass
class LatchScriptPhase:
    name: str
    hand: str
    rung_id: int
    separation_m: float
    latch_left: float
    latch_right: float
    steps: int
    motion_mode: MotionMode = "hold_ladder"
    """How the robot moves this phase; rung geoms stay on the sampled ladder."""
    approach_step_m: float = 0.012
    """Per-step root translation toward a rung when motion_mode='slide_robot_to_rung'."""
    pull_step_m: float = 0.008
    """Per-step root translation away from the ladder when motion_mode='pull_robot'."""


DEFAULT_LATCH_SCRIPT: tuple[LatchScriptPhase, ...] = (
    LatchScriptPhase(
        name="settle_near_rung_a",
        hand="left",
        rung_id=_LEFT_RUNG_A,
        separation_m=_CLEARANCE_SEPARATION_M,
        latch_left=0.0,
        latch_right=0.0,
        steps=16,
        motion_mode="slide_robot_to_rung",
        approach_step_m=0.008,
    ),
    LatchScriptPhase(
        name="left_approach_rung_a",
        hand="left",
        rung_id=_LEFT_RUNG_A,
        separation_m=_CONTACT_SEPARATION_M,
        latch_left=0.0,
        latch_right=0.0,
        steps=40,
        motion_mode="slide_robot_to_rung",
    ),
    LatchScriptPhase(
        name="left_touch",
        hand="left",
        rung_id=_LEFT_RUNG_A,
        separation_m=_PRESS_SEPARATION_M,
        latch_left=0.0,
        latch_right=0.0,
        steps=20,
        motion_mode="slide_robot_to_rung",
        approach_step_m=0.002,
    ),
    LatchScriptPhase(
        name="left_attach",
        hand="left",
        rung_id=_LEFT_RUNG_A,
        separation_m=_PRESS_SEPARATION_M,
        latch_left=1.0,
        latch_right=0.0,
        steps=16,
        motion_mode="slide_robot_to_rung",
        approach_step_m=0.001,
    ),
    LatchScriptPhase(
        name="left_pull_demo_a",
        hand="left",
        rung_id=_LEFT_RUNG_A,
        separation_m=0.0,
        latch_left=1.0,
        latch_right=0.0,
        steps=35,
        motion_mode="pull_robot",
        pull_step_m=0.006,
    ),
    LatchScriptPhase(
        name="left_detach",
        hand="left",
        rung_id=_LEFT_RUNG_A,
        separation_m=_PRESS_SEPARATION_M,
        latch_left=-1.0,
        latch_right=0.0,
        steps=20,
        motion_mode="hold",
    ),
    LatchScriptPhase(
        name="left_approach_rung_b",
        hand="left",
        rung_id=_LEFT_RUNG_B,
        separation_m=_CONTACT_SEPARATION_M,
        latch_left=0.0,
        latch_right=0.0,
        steps=35,
        motion_mode="slide_robot_to_rung",
    ),
    # Attach immediately while approach contact is still live — a separate
    # retouch phase was bouncing the hand just out of the contact band.
    LatchScriptPhase(
        name="left_reattach",
        hand="left",
        rung_id=_LEFT_RUNG_B,
        separation_m=_PRESS_SEPARATION_M,
        latch_left=1.0,
        latch_right=0.0,
        steps=24,
        motion_mode="slide_robot_to_rung",
        approach_step_m=0.0015,
    ),
    LatchScriptPhase(
        name="left_pull_demo_b",
        hand="left",
        rung_id=_LEFT_RUNG_B,
        separation_m=0.0,
        latch_left=1.0,
        latch_right=0.0,
        steps=35,
        motion_mode="pull_robot",
        pull_step_m=0.006,
    ),
)


def build_latch_action(
    env: ManagerBasedRlEnv,
    *,
    latch_left: float,
    latch_right: float,
) -> torch.Tensor:
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    latch = action_term_slice(action, env, "latch")
    latch[:, 0] = latch_left
    latch[:, 1] = latch_right
    return action


def initialize_latch_debug_scene(env: ManagerBasedRlEnv) -> None:
    """Full ladder at startup, robot seeded near the first attach rung.

    Seeding avoids the long root-velocity walk from the default spawn, during
    which zero joint actions let the standing pose collapse and miss contact.
    """
    restore_all_active_rungs_to_ladder(env)
    restore_active_rungs_default_collision(env)
    # Start with a clear gap; scripted velocity approach closes to contact.
    place_robot_hand_near_rung(
        env,
        hand="left",
        rung_id=_LEFT_RUNG_A,
        separation_m=_CLEARANCE_SEPARATION_M,
    )
    env.sim.forward()


def apply_latch_script_phase(
    env: ManagerBasedRlEnv,
    phase: LatchScriptPhase,
    *,
    phase_step: int = 0,
) -> None:
    """Update robot root motion for the current scripted latch phase."""
    # One-shot re-seed when switching hands/rungs while another latch may hold
    # the body: velocity drive alone cannot overcome a connect constraint enough
    # to place the free hand on target.
    if (
        phase_step == 0
        and phase.motion_mode == "slide_robot_to_rung"
        and phase.name == "left_approach_rung_b"
    ):
        place_robot_hand_near_rung(
            env,
            hand=phase.hand,
            rung_id=phase.rung_id,
            separation_m=_CLEARANCE_SEPARATION_M,
        )
        env.sim.forward()

    if phase.motion_mode == "hold_ladder":
        return

    if phase.motion_mode == "slide_robot_to_rung":
        nudge_robot_toward_rung(
            env,
            hand=phase.hand,
            rung_id=phase.rung_id,
            separation_m=phase.separation_m,
            step_m=phase.approach_step_m,
        )
        return

    if phase.motion_mode == "pull_robot":
        nudge_robot_away_from_ladder(env, step_m=phase.pull_step_m)
        return

    if phase.motion_mode == "hold":
        return


def run_latch_script_step(
    env: ManagerBasedRlEnv,
    phase: LatchScriptPhase,
    *,
    phase_step: int,
    latch_cfg: LatchConfig,
) -> None:
    del phase_step, latch_cfg
    apply_latch_script_phase(env, phase)
    action = build_latch_action(
        env,
        latch_left=phase.latch_left,
        latch_right=phase.latch_right,
    )
    env.step(action)
