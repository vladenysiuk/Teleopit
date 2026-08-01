"""Environment configuration for General-Climbing-G1."""

from __future__ import annotations

import mujoco

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from train_mimic.tasks.climbing.config.hold_pose import HoldPoseConfig
from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.latch import LatchConfig
from train_mimic.tasks.climbing.config.curriculum import CurriculumConfig
from train_mimic.tasks.climbing.config.observations import (
    LadderObservationConfig,
    RelativeRungObsConfig,
    configure_climbing_observations,
    validate_ladder_observation_config,
)
from train_mimic.tasks.climbing.config.rewards import (
    ClimbingRewardConfig,
    ClimbingTerminationConfig,
    configure_climbing_rewards,
)
from train_mimic.tasks.climbing.config.robot import (
    HAND_POINT_GEOM_NAMES,
    LEFT_HAND_POINT,
    RIGHT_HAND_POINT,
    make_climbing_robot_cfg,
    make_ladder_collision_cfg,
)
from train_mimic.tasks.climbing.config.scene import make_ladder_entity_cfg
from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.latch import add_latch_equalities_to_spec
from train_mimic.tasks.climbing.mdp import events as climb_events
from train_mimic.tasks.climbing.mdp.actions import LatchActionCfg


def _hand_rung_contact_sensors(ladder_cfg: LadderConfig) -> tuple[ContactSensorCfg, ...]:
    sensors: list[ContactSensorCfg] = []
    for hand, geom_name in (("left", LEFT_HAND_POINT), ("right", RIGHT_HAND_POINT)):
        for rung_id in range(ladder_cfg.max_rungs):
            sensors.append(
                ContactSensorCfg(
                    name=f"{ClimbContactState.SENSOR_PREFIX}_{hand}_rung_{rung_id:02d}",
                    primary=ContactMatch(
                        mode="geom",
                        pattern=geom_name,
                        entity="robot",
                    ),
                    secondary=ContactMatch(
                        mode="geom",
                        pattern=f"rung_{rung_id:02d}_geom",
                        entity="ladder",
                    ),
                    # force is contact-frame: column 0 = normal (primary→secondary).
                    # Keep global_frame=False so |force_x| is the contact normal force.
                    fields=("found", "force", "pos"),
                    reduce="maxforce",
                    num_slots=1,
                    global_frame=False,
                )
            )
    return tuple(sensors)


def make_ladder_entity_cfg_with_collision(cfg: LadderConfig):
    entity_cfg = make_ladder_entity_cfg(cfg)
    entity_cfg.collisions = (make_ladder_collision_cfg(),)
    return entity_cfg


def _latch_equality_count(ladder_cfg: LadderConfig) -> int:
    return 2 * ladder_cfg.max_rungs * ladder_cfg.sites_per_rung


def _recommended_nconmax(ladder_cfg: LadderConfig, *, hold_pose: bool = False) -> int:
    """Contact buffer capacity per environment.

    Four-contact hold and scripted fall nudges can briefly stack foot, hand,
    body, and rail contacts against the per-env ``nconmax`` budget.
    """
    # Hand points × rungs plus feet/body/rail pairs during hold perturbation.
    base = 192 + 4 * ladder_cfg.max_rungs
    if hold_pose:
        return max(768, base)
    return max(384, base)


def _recommended_njmax(ladder_cfg: LadderConfig, *, hold_pose: bool = False) -> int:
    """Constraint-row capacity per environment (connect latches + contacts)."""
    latch_rows = _latch_equality_count(ladder_cfg) * 3
    base = max(1200, latch_rows + 600)
    if hold_pose:
        return max(base, 2560)
    return base


def make_climbing_scene_cfg(
    ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
    *,
    num_envs: int,
    env_spacing: float,
    sensors: tuple[ContactSensorCfg, ...],
) -> SceneCfg:
    def spec_fn(spec: mujoco.MjSpec) -> None:
        add_latch_equalities_to_spec(spec, ladder_cfg, latch_cfg)

    return SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        num_envs=num_envs,
        env_spacing=env_spacing,
        entities={
            "robot": make_climbing_robot_cfg(),
            "ladder": make_ladder_entity_cfg_with_collision(ladder_cfg),
        },
        sensors=sensors,
        spec_fn=spec_fn,
    )


def make_general_climbing_env_cfg(
    *,
    num_envs: int = 4096,
    seed: int = 0,
    ladder_cfg: LadderConfig | None = None,
    latch_cfg: LatchConfig | None = None,
    relative_rung_cfg: RelativeRungObsConfig | None = None,
    ladder_observation_cfg: LadderObservationConfig | None = None,
    reward_cfg: ClimbingRewardConfig | None = None,
    termination_cfg: ClimbingTerminationConfig | None = None,
    curriculum_cfg: CurriculumConfig | None = None,
    play: bool = False,
    env_spacing: float = 4.0,
) -> ManagerBasedRlEnvCfg:
    """General-Climbing-G1 env: G1 + ladder + contacts + latch + observations + rewards."""
    ladder_cfg = ladder_cfg or LadderConfig()
    latch_cfg = latch_cfg or LatchConfig()
    relative_rung_cfg = relative_rung_cfg or RelativeRungObsConfig()
    ladder_observation_cfg = ladder_observation_cfg or LadderObservationConfig()
    reward_cfg = reward_cfg or ClimbingRewardConfig()
    termination_cfg = termination_cfg or ClimbingTerminationConfig()
    validate_ladder_observation_config(ladder_observation_cfg)
    cfg = ManagerBasedRlEnvCfg(
        decimation=4,
        scene=make_climbing_scene_cfg(
            ladder_cfg,
            latch_cfg,
            num_envs=num_envs,
            env_spacing=env_spacing,
            sensors=_hand_rung_contact_sensors(ladder_cfg),
        ),
        observations={},
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=G1_ACTION_SCALE,
                use_default_offset=True,
            ),
            "latch": LatchActionCfg(
                entity_name="robot",
                ladder_cfg=ladder_cfg,
                latch_cfg=latch_cfg,
            ),
        },
        events={
            # Safe reset order: deactivate equalities → clear Python latch/contact
            # state → resample ladder sites → curriculum physics/pose. Moving sites
            # while an old connect remains active can produce a one-step impulse.
            "reset_climb_latch": EventTermCfg(
                func=climb_events.reset_climb_latch_state,
                mode="reset",
            ),
            "reset_climb_contacts": EventTermCfg(
                func=climb_events.reset_climb_contact_state,
                mode="reset",
            ),
            "reset_ladder": EventTermCfg(
                func=climb_events.reset_ladder_sample,
                mode="reset",
                params={"ladder_cfg": ladder_cfg},
            ),
            "reset_climb_rewards": EventTermCfg(
                func=climb_events.reset_climb_reward_state,
                mode="reset",
                params={"progress_body": reward_cfg.progress_body},
            ),
            "attach_climb_contacts": EventTermCfg(
                func=climb_events.attach_climb_contact_state,
                mode="startup",
                params={"ladder_cfg": ladder_cfg},
            ),
            "attach_climb_latch": EventTermCfg(
                func=climb_events.attach_climb_latch_state,
                mode="startup",
                params={"ladder_cfg": ladder_cfg, "latch_cfg": latch_cfg},
            ),
            "attach_relative_rungs": EventTermCfg(
                func=climb_events.attach_relative_rung_provider,
                mode="startup",
                params={
                    "ladder_cfg": ladder_cfg,
                    "relative_rung_cfg": relative_rung_cfg,
                },
            ),
            "attach_climb_rewards": EventTermCfg(
                func=climb_events.attach_climb_reward_state,
                mode="startup",
                params={"progress_body": reward_cfg.progress_body},
            ),
        },
        rewards={},
        terminations={},
        episode_length_s=20.0,
        sim=SimulationCfg(
            njmax=_recommended_njmax(ladder_cfg),
            nconmax=_recommended_nconmax(ladder_cfg),
            mujoco=MujocoCfg(timestep=0.005),
        ),
        seed=seed,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.0,
            elevation=-15.0,
            azimuth=90.0,
        ),
    )
    configure_climbing_observations(
        cfg,
        ladder_cfg=ladder_cfg,
        relative_rung_cfg=relative_rung_cfg,
        ladder_observation_cfg=ladder_observation_cfg,
    )
    configure_climbing_rewards(
        cfg,
        reward_cfg=reward_cfg,
        term_cfg=termination_cfg,
    )
    if curriculum_cfg is not None:
        cfg.events["reset_climb_curriculum_physics"] = EventTermCfg(
            func=climb_events.reset_climb_curriculum_physics,
            mode="reset",
            params={"curriculum_cfg": curriculum_cfg},
        )
        cfg.events["reset_climb_curriculum_pose"] = EventTermCfg(
            func=climb_events.reset_climb_curriculum_pose,
            mode="reset",
            params={"curriculum_cfg": curriculum_cfg},
        )
    if play:
        cfg.observations["actor_proprio"].enable_corruption = False
        cfg.observations["actor_proprio_history"].enable_corruption = False
    return cfg


def make_climbing_rewards_env_cfg(
    *,
    num_envs: int = 1,
    seed: int = 0,
    ladder_cfg: LadderConfig | None = None,
    latch_cfg: LatchConfig | None = None,
    reward_cfg: ClimbingRewardConfig | None = None,
    termination_cfg: ClimbingTerminationConfig | None = None,
    env_spacing: float = 4.0,
) -> ManagerBasedRlEnvCfg:
    """Single-env rewards debug configuration with deterministic ladder sampling."""
    ladder_cfg = ladder_cfg or LadderConfig(
        min_active_rungs=6,
        max_active_rungs=6,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
    )
    reward_cfg = reward_cfg or ClimbingRewardConfig()
    termination_cfg = termination_cfg or ClimbingTerminationConfig(
        enable_success_termination=False,
        enable_fall_termination=False,
    )
    cfg = make_general_climbing_env_cfg(
        num_envs=num_envs,
        seed=seed,
        ladder_cfg=ladder_cfg,
        latch_cfg=latch_cfg or LatchConfig(),
        reward_cfg=reward_cfg,
        termination_cfg=termination_cfg,
        env_spacing=env_spacing,
        play=True,
    )
    cfg.decimation = 1
    cfg.episode_length_s = 1.0e9
    return cfg


def make_climbing_observations_env_cfg(
    *,
    num_envs: int = 1,
    seed: int = 0,
    ladder_cfg: LadderConfig | None = None,
    latch_cfg: LatchConfig | None = None,
    relative_rung_cfg: RelativeRungObsConfig | None = None,
    env_spacing: float = 4.0,
) -> ManagerBasedRlEnvCfg:
    """Single-env observations debug configuration with deterministic ladder sampling."""
    ladder_cfg = ladder_cfg or LadderConfig(
        min_active_rungs=6,
        max_active_rungs=6,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
    )
    cfg = make_general_climbing_env_cfg(
        num_envs=num_envs,
        seed=seed,
        ladder_cfg=ladder_cfg,
        latch_cfg=latch_cfg or LatchConfig(),
        relative_rung_cfg=relative_rung_cfg,
        env_spacing=env_spacing,
        play=True,
    )
    cfg.decimation = 1
    cfg.episode_length_s = 1.0e9
    return cfg


def make_climbing_contacts_env_cfg(
    *,
    num_envs: int = 1,
    seed: int = 0,
    ladder_cfg: LadderConfig | None = None,
    latch_cfg: LatchConfig | None = None,
    env_spacing: float = 4.0,
) -> ManagerBasedRlEnvCfg:
    """Single-env contact debug configuration with deterministic ladder sampling."""
    ladder_cfg = ladder_cfg or LadderConfig(
        min_active_rungs=6,
        max_active_rungs=6,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
    )
    cfg = make_general_climbing_env_cfg(
        num_envs=num_envs,
        seed=seed,
        ladder_cfg=ladder_cfg,
        latch_cfg=latch_cfg,
        env_spacing=env_spacing,
    )
    cfg.decimation = 1
    cfg.episode_length_s = 1.0e9
    return cfg


def make_climbing_latch_env_cfg(
    *,
    num_envs: int = 1,
    seed: int = 0,
    ladder_cfg: LadderConfig | None = None,
    latch_cfg: LatchConfig | None = None,
    env_spacing: float = 4.0,
) -> ManagerBasedRlEnvCfg:
    """Single-env latch debug configuration with deterministic ladder sampling."""
    ladder_cfg = ladder_cfg or LadderConfig(
        min_active_rungs=6,
        max_active_rungs=6,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
    )
    latch_cfg = latch_cfg or LatchConfig()
    return make_climbing_contacts_env_cfg(
        num_envs=num_envs,
        seed=seed,
        ladder_cfg=ladder_cfg,
        latch_cfg=latch_cfg,
        env_spacing=env_spacing,
    )


def make_climbing_hold_pose_env_cfg(
    *,
    num_envs: int = 1,
    seed: int = 0,
    ladder_cfg: LadderConfig | None = None,
    latch_cfg: LatchConfig | None = None,
    hold_cfg: HoldPoseConfig | None = None,
    env_spacing: float = 4.0,
) -> ManagerBasedRlEnvCfg:
    """Single-env static hold-pose env: fixed ladder, no pose randomization."""
    from copy import deepcopy

    from mjlab.utils.spec_config import CollisionCfg

    from train_mimic.tasks.climbing.config.robot import (
        ROBOT_BODY_CONAFFINITY,
        ROBOT_BODY_CONTYPE,
    )

    hold_cfg = hold_cfg or HoldPoseConfig()
    ladder_cfg = ladder_cfg or LadderConfig(
        min_active_rungs=6,
        max_active_rungs=6,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
        # Stage 6 may raise cylinder radius/friction within the agreed ranges.
        rung_radius=hold_cfg.rung_radius_m,
        rung_friction=hold_cfg.rung_friction,
    )
    # Slightly stiffer connect latch for the static hold (still soft; not welded).
    hold_latch_cfg = latch_cfg or LatchConfig(solref=(0.012, 1.0))
    cfg = make_general_climbing_env_cfg(
        num_envs=num_envs,
        seed=seed,
        ladder_cfg=ladder_cfg,
        latch_cfg=hold_latch_cfg,
        reward_cfg=ClimbingRewardConfig(),
        termination_cfg=ClimbingTerminationConfig(
            enable_success_termination=False,
            enable_fall_termination=False,
        ),
        env_spacing=env_spacing,
        play=True,
    )
    # Stage-6-allowed foot friction + PD stiffening for cylinder standing.
    # Do not raise effort limits (forbidden: unlimited actuator torques).
    robot_cfg = deepcopy(cfg.scene.entities["robot"])
    robot_cfg.collisions = (
        CollisionCfg(
            geom_names_expr=(r".*_collision$",),
            contype=ROBOT_BODY_CONTYPE,
            conaffinity=ROBOT_BODY_CONAFFINITY,
            condim=3,
            priority={r"^(left|right)_foot[1-7]_collision$": 1},
            friction={r"^(left|right)_foot[1-7]_collision$": (hold_cfg.foot_friction,)},
            disable_other_geoms=False,
        ),
    )
    scaled_actuators = []
    for actuator in robot_cfg.articulation.actuators:
        act = deepcopy(actuator)
        act.stiffness = float(act.stiffness) * hold_cfg.pd_stiffness_scale
        act.damping = float(act.damping) * hold_cfg.pd_damping_scale
        scaled_actuators.append(act)
    robot_cfg.articulation.actuators = tuple(scaled_actuators)
    cfg.scene.entities["robot"] = robot_cfg
    cfg.decimation = 1
    cfg.episode_length_s = 1.0e9
    cfg.sim.njmax = _recommended_njmax(ladder_cfg, hold_pose=True)
    cfg.sim.nconmax = _recommended_nconmax(ladder_cfg, hold_pose=True)
    return cfg


def make_climbing_mdp_env_cfg(
    *,
    num_envs: int = 1,
    seed: int = 0,
    episode_length_s: float = 5.0,
    ladder_cfg: LadderConfig | None = None,
    latch_cfg: LatchConfig | None = None,
    reward_cfg: ClimbingRewardConfig | None = None,
    termination_cfg: ClimbingTerminationConfig | None = None,
    env_spacing: float = 4.0,
    play: bool = False,
    decimation: int = 1,
) -> ManagerBasedRlEnvCfg:
    """Full General-Climbing-G1 MDP for reset/agent/capacity validation (Stage 7)."""
    ladder_cfg = ladder_cfg or LadderConfig(
        min_active_rungs=4,
        max_active_rungs=8,
    )
    cfg = make_general_climbing_env_cfg(
        num_envs=num_envs,
        seed=seed,
        ladder_cfg=ladder_cfg,
        latch_cfg=latch_cfg or LatchConfig(),
        reward_cfg=reward_cfg or ClimbingRewardConfig(),
        termination_cfg=termination_cfg or ClimbingTerminationConfig(),
        env_spacing=env_spacing,
        play=play,
    )
    cfg.decimation = decimation
    cfg.episode_length_s = episode_length_s
    return cfg


def verify_hand_point_geoms(model) -> None:
    """Fail fast if canonical hand point geoms are missing from the compiled model."""
    prefix = "robot/"
    for geom_name in HAND_POINT_GEOM_NAMES:
        full = f"{prefix}{geom_name}"
        if full not in {model.geom(i).name for i in range(model.ngeom)}:
            raise RuntimeError(f"Missing climbing hand geom: {full}")
