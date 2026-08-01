"""Scene-only environment configuration for ladder debugging (Stage 1)."""

from __future__ import annotations

from functools import partial

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.ladder.generator import build_ladder_spec
from train_mimic.tasks.climbing.mdp import events as ladder_events


def make_ladder_entity_cfg(cfg: LadderConfig) -> EntityCfg:
    """Entity configuration for the fixed-topology ladder."""
    return EntityCfg(
        spec_fn=partial(build_ladder_spec, cfg),
        init_state=EntityCfg.InitialStateCfg(
            pos=(cfg.ladder_distance, 0.0, 0.0),
        ),
    )


def make_ladder_scene_env_cfg(
    *,
    num_envs: int = 1,
    seed: int = 0,
    ladder_cfg: LadderConfig | None = None,
    env_spacing: float = 4.0,
) -> ManagerBasedRlEnvCfg:
    """Minimal mjlab env for ladder scene inspection without task registration."""
    ladder_cfg = ladder_cfg or LadderConfig()
    return ManagerBasedRlEnvCfg(
        decimation=1,
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=num_envs,
            env_spacing=env_spacing,
            entities={"ladder": make_ladder_entity_cfg(ladder_cfg)},
        ),
        observations={},
        actions={},
        events={
            "reset_ladder": EventTermCfg(
                func=ladder_events.reset_ladder_sample,
                mode="reset",
                params={"ladder_cfg": ladder_cfg},
            ),
        },
        rewards={},
        terminations={},
        episode_length_s=1.0e9,
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
        seed=seed,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            distance=3.0,
            elevation=-15.0,
            azimuth=90.0,
        ),
    )
