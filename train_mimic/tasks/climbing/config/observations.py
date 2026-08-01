"""Observation configuration for General-Climbing-G1."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mjlab_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from train_mimic.tasks.climbing.config.ladder import LadderConfig

# Default proprioception history length (matches tracking task).
PROPRIO_HISTORY_LENGTH = 10

# Relative rung window: K=6 slots = 2 below + 4 above pelvis height.
DEFAULT_RELATIVE_RUNG_K = 6
DEFAULT_RUNGS_BELOW = 2
DEFAULT_RUNGS_ABOVE = 4

# Per-rung ladder exteroception: endpoint_a(3) + endpoint_b(3) + valid(1).
RELATIVE_RUNG_FEATURE_DIM = 7

LadderObservationMode = Literal["relative_rungs", "depth"]


@dataclass(frozen=True)
class LadderObservationConfig:
    """Swappable ladder perception mode for env and model wiring."""

    mode: LadderObservationMode = "relative_rungs"
    depth_height: int = 64
    depth_width: int = 64

    def __post_init__(self) -> None:
        if self.mode not in ("relative_rungs", "depth"):
            raise ValueError(
                f"Unsupported ladder_observation.mode={self.mode!r}. "
                "Expected 'relative_rungs' or 'depth'."
            )
        if self.depth_height <= 0 or self.depth_width <= 0:
            raise ValueError("depth_height and depth_width must be positive.")


def validate_ladder_observation_config(cfg: LadderObservationConfig) -> None:
    """Fail fast when an unsupported ladder perception backend is requested."""
    if cfg.mode == "depth":
        raise NotImplementedError(
            "ladder_observation.mode='depth' requires DepthCameraProvider, which is "
            "not implemented yet. Use mode='relative_rungs' until the camera provider "
            "exists."
        )


@dataclass(frozen=True)
class RelativeRungObsConfig:
    """Privileged relative-rung observation window."""

    num_rungs: int = DEFAULT_RELATIVE_RUNG_K
    rungs_below: int = DEFAULT_RUNGS_BELOW
    rungs_above: int = DEFAULT_RUNGS_ABOVE
    reference_body: str = "pelvis"
    torso_body: str = "torso_link"

    def __post_init__(self) -> None:
        if self.num_rungs <= 0:
            raise ValueError("num_rungs must be positive.")
        if self.rungs_below < 0 or self.rungs_above < 0:
            raise ValueError("rungs_below and rungs_above must be non-negative.")
        if self.rungs_below + self.rungs_above != self.num_rungs:
            raise ValueError(
                "rungs_below + rungs_above must equal num_rungs "
                f"({self.rungs_below} + {self.rungs_above} != {self.num_rungs})."
            )


def _shared_proprio_terms(*, enable_noise: bool) -> dict[str, ObservationTermCfg]:
    """Proprioception + contact/latch terms shared by actor and critic."""
    from train_mimic.tasks.climbing.mdp import observations as climb_mdp

    def _noise(**kwargs: float) -> Unoise | None:
        return Unoise(**kwargs) if enable_noise else None

    return {
        "robot_joint_pos_rel": ObservationTermCfg(
            func=mjlab_mdp.joint_pos_rel,
            params={"biased": True},
            noise=_noise(n_min=-0.01, n_max=0.01),
        ),
        "robot_joint_vel": ObservationTermCfg(
            func=mjlab_mdp.joint_vel_rel,
            noise=_noise(n_min=-0.5, n_max=0.5),
        ),
        "robot_base_ang_vel_b": ObservationTermCfg(
            func=mjlab_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=_noise(n_min=-0.2, n_max=0.2),
        ),
        "robot_projected_gravity_b": ObservationTermCfg(
            func=mjlab_mdp.projected_gravity,
            noise=_noise(n_min=-0.05, n_max=0.05),
        ),
        "prev_joint_action": ObservationTermCfg(func=climb_mdp.prev_joint_action),
        "prev_latch_action": ObservationTermCfg(func=climb_mdp.prev_latch_action),
        "hand_in_contact": ObservationTermCfg(func=climb_mdp.hand_in_contact),
        "hand_attached": ObservationTermCfg(func=climb_mdp.hand_attached),
        "attached_rung_rel_height": ObservationTermCfg(
            func=climb_mdp.attached_rung_rel_height,
            params={"reference_body": "pelvis"},
        ),
    }


def _ladder_term(ladder_cfg: LadderConfig, obs_cfg: RelativeRungObsConfig) -> ObservationTermCfg:
    from train_mimic.tasks.climbing.mdp import observations as climb_mdp

    return ObservationTermCfg(
        func=climb_mdp.relative_rung_endpoints_torso,
        params={"ladder_cfg": ladder_cfg, "obs_cfg": obs_cfg},
    )


def _privileged_critic_terms(ladder_cfg: LadderConfig) -> dict[str, ObservationTermCfg]:
    from train_mimic.tasks.climbing.mdp import observations as climb_mdp

    return {
        "robot_base_lin_vel_b": ObservationTermCfg(
            func=mjlab_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "pelvis_height_w": ObservationTermCfg(
            func=climb_mdp.body_height_w,
            params={"body_name": "pelvis"},
        ),
        "ladder_frame_pos_torso": ObservationTermCfg(
            func=climb_mdp.ladder_frame_pos_torso,
            params={"ladder_cfg": ladder_cfg},
        ),
        "ladder_frame_ori_torso": ObservationTermCfg(
            func=climb_mdp.ladder_frame_ori_torso,
            params={"ladder_cfg": ladder_cfg},
        ),
        "active_rung_heights_l_rel": ObservationTermCfg(
            func=climb_mdp.active_rung_heights_l_rel,
            params={"ladder_cfg": ladder_cfg, "obs_cfg": RelativeRungObsConfig()},
        ),
        "active_rung_valid_mask": ObservationTermCfg(
            func=climb_mdp.active_rung_valid_mask,
            params={"ladder_cfg": ladder_cfg, "obs_cfg": RelativeRungObsConfig()},
        ),
    }


def configure_climbing_observations(
    cfg: ManagerBasedRlEnvCfg,
    *,
    ladder_cfg: LadderConfig,
    relative_rung_cfg: RelativeRungObsConfig | None = None,
    ladder_observation_cfg: LadderObservationConfig | None = None,
    proprio_history_length: int = PROPRIO_HISTORY_LENGTH,
) -> None:
    """Wire separated observation groups into a climbing env cfg."""
    relative_rung_cfg = relative_rung_cfg or RelativeRungObsConfig()
    ladder_observation_cfg = ladder_observation_cfg or LadderObservationConfig()
    validate_ladder_observation_config(ladder_observation_cfg)

    actor_proprio_terms = _shared_proprio_terms(enable_noise=True)
    critic_proprio_terms = _shared_proprio_terms(enable_noise=False)

    if ladder_observation_cfg.mode == "relative_rungs":
        ladder_term = _ladder_term(ladder_cfg, relative_rung_cfg)
    else:
        raise NotImplementedError(
            "Depth ladder observations are not wired into the env yet. "
            "Use ladder_observation.mode='relative_rungs'."
        )

    cfg.observations = {
        "actor_proprio": ObservationGroupCfg(
            terms=actor_proprio_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "actor_proprio_history": ObservationGroupCfg(
            terms=deepcopy(actor_proprio_terms),
            concatenate_terms=True,
            enable_corruption=True,
            history_length=proprio_history_length,
            flatten_history_dim=False,
        ),
        "actor_ladder": ObservationGroupCfg(
            terms={"relative_rungs": ladder_term},
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic_proprio": ObservationGroupCfg(
            terms=critic_proprio_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic_proprio_history": ObservationGroupCfg(
            terms=deepcopy(critic_proprio_terms),
            concatenate_terms=True,
            enable_corruption=False,
            history_length=proprio_history_length,
            flatten_history_dim=False,
        ),
        "critic_ladder": ObservationGroupCfg(
            terms={"relative_rungs": deepcopy(ladder_term)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic_privileged": ObservationGroupCfg(
            terms=_privileged_critic_terms(ladder_cfg),
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

def actor_proprio_dim() -> int:
    """Expected flat actor proprio dimension (29+29+3+3+29+2+2+2+2)."""
    return 29 + 29 + 3 + 3 + 29 + 2 + 2 + 2 + 2


def critic_privileged_dim(*, num_rungs: int = DEFAULT_RELATIVE_RUNG_K) -> int:
    """Expected flat critic privileged dimension (3+1+3+6+2K)."""
    return 3 + 1 + 3 + 6 + 2 * num_rungs


__all__ = [
    "DEFAULT_RELATIVE_RUNG_K",
    "DEFAULT_RUNGS_ABOVE",
    "DEFAULT_RUNGS_BELOW",
    "LadderObservationConfig",
    "LadderObservationMode",
    "PROPRIO_HISTORY_LENGTH",
    "RELATIVE_RUNG_FEATURE_DIM",
    "RelativeRungObsConfig",
    "actor_proprio_dim",
    "configure_climbing_observations",
    "critic_privileged_dim",
    "validate_ladder_observation_config",
]
