"""Stage 1 checks for ladder specification, generation, and sampling."""

from __future__ import annotations

import mujoco
import pytest
import torch

from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.ladder.generator import (
    build_ladder_spec,
    expected_rung_geom_names,
    expected_site_names,
    latch_site_normal_offset,
)
from train_mimic.tasks.climbing.ladder.geometry import create_rung_geometry
from train_mimic.tasks.climbing.ladder.state import LadderSampler, LadderTopology


@pytest.fixture
def small_cfg() -> LadderConfig:
    return LadderConfig(max_rungs=6, min_active_rungs=2, max_active_rungs=4)


def test_ladder_config_rejects_unsupported_geometry_kind() -> None:
    with pytest.raises(ValueError, match="geometry_kind"):
        LadderConfig(geometry_kind="box")


def test_create_rung_geometry_rejects_unsupported_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_rung_geometry("box")


def test_build_ladder_spec_has_stable_topology(small_cfg: LadderConfig) -> None:
    spec = build_ladder_spec(small_cfg)
    model = spec.compile()

    assert expected_rung_geom_names(small_cfg) == tuple(
        f"rung_{i:02d}_geom" for i in range(small_cfg.max_rungs)
    )
    assert model.nmocap == 1
    assert model.ngeom == small_cfg.max_rungs + 2
    assert model.nsite == small_cfg.max_rungs * small_cfg.sites_per_rung
    assert int(model.body_mocapid[model.body("frame").id]) >= 0

    for rung_id in range(small_cfg.max_rungs):
        geom_name = f"rung_{rung_id:02d}_geom"
        geom_id = model.geom(geom_name).id
        assert model.geom_bodyid[geom_id] == model.body("frame").id
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert model.geom_size[geom_id][0] == pytest.approx(small_cfg.rung_radius)
        assert model.geom_size[geom_id][1] == pytest.approx(small_cfg.rung_length * 0.5)
        assert model.geom_friction[geom_id][0] == pytest.approx(small_cfg.rung_friction[0])

        site_names = expected_site_names(small_cfg, rung_id)
        assert len(site_names) == small_cfg.sites_per_rung
        expected_x = -latch_site_normal_offset(small_cfg)
        x_positions = [model.site_pos[model.site(name).id][0] for name in site_names]
        y_positions = [model.site_pos[model.site(name).id][1] for name in site_names]
        assert all(x == pytest.approx(expected_x) for x in x_positions)
        assert y_positions[0] == pytest.approx(-small_cfg.rung_length * 0.5)
        assert y_positions[-1] == pytest.approx(small_cfg.rung_length * 0.5)
        assert len(set(y_positions)) == small_cfg.sites_per_rung


def test_sampler_reproducibility_and_batching(small_cfg: LadderConfig) -> None:
    sample_a = LadderSampler(small_cfg, seed=7).sample(3)
    sample_b = LadderSampler(small_cfg, seed=7).sample(3)
    sample_c = LadderSampler(small_cfg, seed=8).sample(3)

    assert torch.equal(sample_a.active_mask, sample_b.active_mask)
    assert torch.allclose(sample_a.rung_heights, sample_b.rung_heights)
    assert not torch.equal(sample_a.active_mask, sample_c.active_mask)


def test_sampler_active_count_and_spacing_invariants(small_cfg: LadderConfig) -> None:
    sample = LadderSampler(small_cfg, seed=123).sample(8)

    assert torch.all(sample.num_active >= small_cfg.min_active_rungs)
    assert torch.all(sample.num_active <= small_cfg.max_active_rungs)
    assert torch.all(sample.active_mask.sum(dim=1) == sample.num_active)

    for env_idx in range(sample.num_active.shape[0]):
        count = int(sample.num_active[env_idx].item())
        heights = sample.rung_heights[env_idx, :count]
        assert torch.all(heights[1:] > heights[:-1])
        if count > 1:
            gaps = heights[1:] - heights[:-1]
            assert torch.all(gaps >= small_cfg.spacing_min)
            assert torch.all(gaps <= small_cfg.spacing_max)


def test_inactive_rungs_are_parked_outside_scene(small_cfg: LadderConfig) -> None:
    sample = LadderSampler(small_cfg, seed=99).sample(4)
    inactive = ~sample.active_mask
    inactive_z = sample.rung_pos_w[inactive, 2]
    active_z = sample.rung_pos_w[sample.active_mask, 2]

    assert torch.all(inactive_z < -10.0)
    assert torch.all(active_z >= small_cfg.base_height - 0.5)


def test_debug_vis_site_world_positions_use_sampled_rung_pose(small_cfg: LadderConfig) -> None:
    import numpy as np

    from train_mimic.tasks.climbing.debug_vis import _quat_apply_wxyz
    from train_mimic.tasks.climbing.ladder.generator import build_ladder_spec, expected_site_names

    model = build_ladder_spec(small_cfg).compile()
    sample = LadderSampler(small_cfg, seed=5).sample(1)

    rung_id = 0
    rung_pos = sample.rung_pos_w[0, rung_id].detach().cpu().numpy()
    rung_quat = sample.rung_quat_w[0, rung_id].detach().cpu().numpy()
    site_name = expected_site_names(small_cfg, rung_id)[0]
    local = np.asarray(model.site_pos[model.site(site_name).id], dtype=np.float64)
    world = rung_pos + _quat_apply_wxyz(rung_quat, local)

    assert np.all(np.isfinite(world))
    assert np.linalg.norm(world - rung_pos) == pytest.approx(np.linalg.norm(local), rel=1e-5)
    # Latch targets sit on the robot-facing side of the rung axis (−ladder X).
    assert local[0] == pytest.approx(-latch_site_normal_offset(small_cfg))


@pytest.mark.parametrize("num_envs", [1, 4])
def test_mjlab_scene_reset_preserves_topology(num_envs: int, small_cfg: LadderConfig) -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.scene import make_ladder_scene_env_cfg
    from train_mimic.tasks.climbing.ladder.state import LadderRuntime

    env_cfg = make_ladder_scene_env_cfg(num_envs=num_envs, seed=11, ladder_cfg=small_cfg)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env.reset()
    runtime = LadderRuntime.get(env)

    model_before = env.sim.mj_model
    counts_before = (model_before.nbody, model_before.ngeom, model_before.nsite, model_before.nmocap)
    topology_before = runtime.topology

    env.reset()
    model_after = env.sim.mj_model
    counts_after = (model_after.nbody, model_after.ngeom, model_after.nsite, model_after.nmocap)

    assert counts_before == counts_after
    assert topology_before.rung_geom_ids == runtime.topology.rung_geom_ids
    assert topology_before.rung_site_ids == runtime.topology.rung_site_ids

    action = torch.zeros(env.num_envs, 0)
    env.step(action)
    env.reset()
    runtime.resample()

    assert runtime.sample is not None
    assert runtime.sample.active_mask.shape == (num_envs, small_cfg.max_rungs)
    env.close()


def test_ladder_runtime_registers_viewer_model_fields() -> None:
    """Batched rung geom_pos/site_pos must sync to the native MuJoCo viewer."""
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.ladder import LadderConfig
    from train_mimic.tasks.climbing.config.scene import make_ladder_scene_env_cfg
    from train_mimic.tasks.climbing.ladder.state import LadderRuntime

    env_cfg = make_ladder_scene_env_cfg(
        num_envs=1,
        seed=7,
        ladder_cfg=LadderConfig(max_rungs=4, min_active_rungs=2, max_active_rungs=4),
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env.reset()
    runtime = LadderRuntime.get(env)
    assert runtime.sample is not None

    expanded = env.sim.expanded_fields
    assert "geom_pos" in expanded
    assert "site_pos" in expanded

    mj_model = env.sim.mj_model
    sync_fields = expanded & {"geom_pos", "site_pos", "body_pos", "body_quat"}
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts, sync_model_fields

    disable_model_sameframe_shortcuts(mj_model)
    sync_model_fields(mj_model, env.sim.model, sync_fields, 0)

    rung_geom_id = runtime.topology.rung_geom_ids[0]
    expected_z = float(runtime.sample.rung_heights[0, 0].item())
    assert mj_model.geom_pos[rung_geom_id, 2] == pytest.approx(expected_z, abs=1e-5)
    env.close()


def test_tracking_registry_still_unchanged() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    from train_mimic.tasks.tracking.config.constants import GENERAL_TRACKING_TASK

    load_env_cfg(GENERAL_TRACKING_TASK)
