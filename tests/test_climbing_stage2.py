"""Stage 2 checks for General-Climbing-G1 contact identity."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import pytest
import torch

from train_mimic.tasks.climbing.config.constants import CLIMBING_TASK_ID
from train_mimic.tasks.climbing.config.env import (
    make_climbing_contacts_env_cfg,
    make_general_climbing_env_cfg,
    verify_hand_point_geoms,
)
from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.robot import (
    HAND_POINT_CONAFFINITY,
    HAND_POINT_CONTYPE,
    HAND_POINT_GEOM_NAMES,
    HAND_POINT_OFFSETS,
    HAND_POINT_RADIUS,
    LADDER_CONAFFINITY,
    LADDER_CONTYPE,
    LEFT_HAND_POINT,
    LEFT_WRIST_BODY,
    RIGHT_HAND_POINT,
    RIGHT_WRIST_BODY,
    ROBOT_BODY_CONAFFINITY,
    ROBOT_BODY_CONTYPE,
    _get_g1_climbing_spec,
)
from train_mimic.tasks.climbing.debug_probe import place_probe_rung_near_hand
from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.state import LadderRuntime


# Empirical clearance with 15 mm hand sphere + 30 mm rung, ladder-+X probe.
CONTACT_CLEARANCE_M = 0.05
CONTACT_PENETRATION_M = 0.005


@pytest.fixture
def small_ladder_cfg() -> LadderConfig:
    return LadderConfig(
        max_rungs=6,
        min_active_rungs=4,
        max_active_rungs=4,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
    )


def test_climbing_hand_point_spec_adds_spheres_and_sites() -> None:
    spec = _get_g1_climbing_spec()
    model = spec.compile()
    for geom_name in HAND_POINT_GEOM_NAMES:
        geom = model.geom(geom_name)
        assert int(geom.type[0]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    assert model.site("left_hand_point_site").id >= 0
    assert model.site("right_hand_point_site").id >= 0
    left = model.site("left_hand_point_site").pos
    right = model.site("right_hand_point_site").pos
    assert tuple(left) == pytest.approx(HAND_POINT_OFFSETS[LEFT_WRIST_BODY])
    assert tuple(right) == pytest.approx(HAND_POINT_OFFSETS[RIGHT_WRIST_BODY])


def test_general_climbing_task_is_registered() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(CLIMBING_TASK_ID)
    assert "robot" in cfg.scene.entities
    assert "ladder" in cfg.scene.entities
    assert cfg.actions["joint_pos"].scale  # G1 action scale wired
    assert len(cfg.scene.sensors) == 24  # 12 rungs x 2 hands


def test_tracking_registry_still_unchanged() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    from train_mimic.app import DEFAULT_TASK
    from train_mimic.tasks.tracking.config.constants import GENERAL_TRACKING_TASK

    assert DEFAULT_TASK == GENERAL_TRACKING_TASK
    load_env_cfg(GENERAL_TRACKING_TASK)


def _make_contacts_env(seed: int = 7, *, num_envs: int = 1, ladder_cfg: LadderConfig | None = None):
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_climbing_contacts_env_cfg(num_envs=num_envs, seed=seed, ladder_cfg=ladder_cfg)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    verify_hand_point_geoms(env.sim.mj_model)
    return env


def _step_zero(env) -> None:
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    env.step(action)


def _contact_state(env):
    contacts = ClimbContactState.get(env.unwrapped)
    return contacts.update()


@pytest.mark.parametrize("num_envs", [1, 2])
def test_contact_tensors_have_expected_shape_and_dtype(num_envs: int, small_ladder_cfg: LadderConfig) -> None:
    env = _make_contacts_env(num_envs=num_envs, ladder_cfg=small_ladder_cfg)
    state = _contact_state(env)
    assert state.hand_in_contact.shape == (num_envs, 2)
    assert state.hand_in_contact.dtype == torch.bool
    assert state.hand_rung_id.shape == (num_envs, 2)
    assert state.hand_rung_id.dtype == torch.int64
    assert state.hand_rung_height.shape == (num_envs, 2)
    assert state.hand_contact_force.shape == (num_envs, 2)
    assert state.hand_contact_pos.shape == (num_envs, 2, 3)
    env.close()


def test_far_hand_has_no_contact(small_ladder_cfg: LadderConfig) -> None:
    env = _make_contacts_env(ladder_cfg=small_ladder_cfg)
    place_probe_rung_near_hand(
        env.unwrapped,
        hand="left",
        rung_id=2,
        separation_m=CONTACT_CLEARANCE_M + 0.05,
    )
    _step_zero(env)
    state = _contact_state(env)
    assert not bool(state.hand_in_contact[0, 0].item())
    assert int(state.hand_rung_id[0, 0].item()) == ClimbContactState.NONE_RUNG_ID
    env.close()


def test_touching_hand_reports_rung_id_height_and_force(small_ladder_cfg: LadderConfig) -> None:
    env = _make_contacts_env(seed=7, ladder_cfg=small_ladder_cfg)
    rung_id = 3
    place_probe_rung_near_hand(
        env.unwrapped,
        hand="left",
        rung_id=rung_id,
        separation_m=-CONTACT_PENETRATION_M,
    )
    _step_zero(env)
    state = _contact_state(env)
    assert bool(state.hand_in_contact[0, 0].item())
    assert int(state.hand_rung_id[0, 0].item()) == rung_id
    assert float(state.hand_contact_force[0, 0].item()) > 0.0

    from train_mimic.tasks.climbing.ladder.state import LadderRuntime

    runtime = LadderRuntime.get(env.unwrapped)
    expected_height = float(runtime.sample.rung_heights[0, rung_id].item())
    assert float(state.hand_rung_height[0, 0].item()) == pytest.approx(expected_height)
    env.close()


def test_left_and_right_hands_are_independent(small_ladder_cfg: LadderConfig) -> None:
    from train_mimic.tasks.climbing.debug_probe import set_probe_rung_hand_only_collision

    env = _make_contacts_env(seed=7, ladder_cfg=small_ladder_cfg)
    set_probe_rung_hand_only_collision(env.unwrapped, 3)

    place_probe_rung_near_hand(
        env.unwrapped, hand="left", rung_id=3, separation_m=-CONTACT_PENETRATION_M
    )
    _step_zero(env)
    state = _contact_state(env)
    assert bool(state.hand_in_contact[0, 0].item())
    assert not bool(state.hand_in_contact[0, 1].item())
    assert int(state.hand_rung_id[0, 0].item()) == 3

    place_probe_rung_near_hand(
        env.unwrapped, hand="right", rung_id=3, separation_m=-CONTACT_PENETRATION_M
    )
    _step_zero(env)
    state = _contact_state(env)
    assert bool(state.hand_in_contact[0, 1].item())
    assert not bool(state.hand_in_contact[0, 0].item())
    assert int(state.hand_rung_id[0, 1].item()) == 3
    env.close()


def test_inactive_rung_contact_is_ignored(small_ladder_cfg: LadderConfig) -> None:
    env = _make_contacts_env(seed=7, ladder_cfg=small_ladder_cfg)
    inactive_rung_id = small_ladder_cfg.max_rungs - 1
    place_probe_rung_near_hand(
        env.unwrapped,
        hand="left",
        rung_id=inactive_rung_id,
        separation_m=-CONTACT_PENETRATION_M,
    )
    _step_zero(env)
    state = _contact_state(env)
    assert not bool(state.hand_in_contact[0, 0].item())
    assert int(state.hand_rung_id[0, 0].item()) == ClimbContactState.NONE_RUNG_ID
    env.close()


def test_reset_clears_stale_contact_state(small_ladder_cfg: LadderConfig) -> None:
    env = _make_contacts_env(seed=7, ladder_cfg=small_ladder_cfg)
    place_probe_rung_near_hand(env.unwrapped, hand="left", rung_id=2, separation_m=-CONTACT_PENETRATION_M)
    _step_zero(env)
    state = _contact_state(env)
    assert bool(state.hand_in_contact[0, 0].item())

    env.reset()
    state = _contact_state(env)
    assert not bool(state.hand_in_contact[0, 0].item())
    assert int(state.hand_rung_id[0, 0].item()) == ClimbContactState.NONE_RUNG_ID
    env.close()


def test_multiple_contact_reduction_prefers_max_force_then_lower_index() -> None:
    """Unit test tie rule without requiring a multi-touch sim pose."""
    batch = 1
    device = "cpu"
    runtime_sample = SimpleNamespace(
        active_mask=torch.tensor([[True, True, True, False]], device=device),
        rung_heights=torch.tensor([[0.3, 0.5, 0.7, 0.0]], device=device),
    )

    class FakeSensor:
        def __init__(self, found: float, force: float) -> None:
            self.data = SimpleNamespace(
                found=torch.tensor([[found]], device=device),
                force=torch.tensor([[[force, 0.0, 0.0]]], device=device),
                pos=torch.tensor([[[1.0, 2.0, 3.0]]], device=device),
            )

    scene = {
        "hand_rung_contact_left_rung_00": FakeSensor(1.0, 5.0),
        "hand_rung_contact_left_rung_01": FakeSensor(1.0, 5.0),
        "hand_rung_contact_left_rung_02": FakeSensor(1.0, 9.0),
        "hand_rung_contact_left_rung_03": FakeSensor(0.0, 0.0),
        "hand_rung_contact_right_rung_00": FakeSensor(0.0, 0.0),
        "hand_rung_contact_right_rung_01": FakeSensor(0.0, 0.0),
        "hand_rung_contact_right_rung_02": FakeSensor(0.0, 0.0),
        "hand_rung_contact_right_rung_03": FakeSensor(0.0, 0.0),
    }

    env = SimpleNamespace(
        num_envs=batch,
        device=device,
        scene=scene,
        extras={
            LadderRuntime.EXTRA_KEY: SimpleNamespace(
                sample=runtime_sample,
                cfg=LadderConfig(max_rungs=4, min_active_rungs=2, max_active_rungs=4),
            ),
        },
    )

    adapter = ClimbContactState(env, max_rungs=4)  # type: ignore[arg-type]
    env.extras[ClimbContactState.EXTRA_KEY] = adapter
    state = adapter.update()
    assert int(state.hand_rung_id[0, 0].item()) == 2

    scene["hand_rung_contact_left_rung_02"].data.force[..., 0, 0] = 5.0
    state = adapter.update()
    assert int(state.hand_rung_id[0, 0].item()) == 0


def test_robot_body_is_repelled_by_ladder_rail(small_ladder_cfg: LadderConfig) -> None:
    """Body-rail collision must resolve in mujoco_warp (symmetric conaffinity filter)."""
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.robot import (
        LADDER_CONAFFINITY,
        LADDER_CONTYPE,
        ROBOT_BODY_CONAFFINITY,
        ROBOT_BODY_CONTYPE,
    )

    assert ROBOT_BODY_CONAFFINITY & LADDER_CONTYPE
    assert ROBOT_BODY_CONTYPE & LADDER_CONAFFINITY

    env = _make_contacts_env(ladder_cfg=small_ladder_cfg)
    robot = env.scene["robot"]
    mjm = env.sim.mj_model
    pelvis_id = mjm.geom("robot/pelvis_collision").id
    assert int(mjm.geom_conaffinity[pelvis_id]) & LADDER_CONTYPE

    rail_id = mjm.geom("ladder/rail_left").id
    rail_pos = env.sim.data.geom_xpos[0, rail_id].detach().cpu().clone()
    rail_pos[2] = 0.77
    pose = torch.cat([rail_pos.unsqueeze(0), robot.data.root_link_quat_w[0:1]], dim=-1)
    robot.write_root_link_pose_to_sim(pose)
    robot.write_root_link_velocity_to_sim(torch.zeros(1, 6, device=env.device))

    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for _ in range(30):
        env.step(action)

    root = robot.data.root_link_pos_w[0]
    assert float(torch.linalg.norm(root[:2] - rail_pos[:2]).item()) > 0.05
    env.close()


def test_general_climbing_env_has_contact_sensors_for_all_rungs() -> None:
    cfg = make_general_climbing_env_cfg(num_envs=1, seed=0)
    names = {sensor.name for sensor in cfg.scene.sensors}
    for hand, point in (("left", LEFT_HAND_POINT), ("right", RIGHT_HAND_POINT)):
        del point
        for rung_id in range(12):
            assert f"hand_rung_contact_{hand}_rung_{rung_id:02d}" in names


def test_contact_sensors_use_contact_frame_force() -> None:
    """Normal force is contact-frame column 0, not an implicit world-axis default."""
    cfg = make_general_climbing_env_cfg(num_envs=1, seed=0)
    assert len(cfg.scene.sensors) > 0
    for sensor in cfg.scene.sensors:
        assert sensor.global_frame is False
        assert "force" in sensor.fields


def test_contact_normal_force_matches_known_axis_contact(small_ladder_cfg: LadderConfig) -> None:
    """With ladder-+X approach, contact-frame |force_x| must be the dominant force."""
    env = _make_contacts_env(seed=7, ladder_cfg=small_ladder_cfg)
    place_probe_rung_near_hand(
        env.unwrapped,
        hand="left",
        rung_id=3,
        separation_m=-CONTACT_PENETRATION_M,
    )
    _step_zero(env)
    state = _contact_state(env)
    assert bool(state.hand_in_contact[0, 0].item())
    force = float(state.hand_contact_force[0, 0].item())
    assert force > 0.0

    sensor = env.scene["hand_rung_contact_left_rung_03"]
    force_vec = sensor.data.force[0, 0]
    normal = float(force_vec[0].abs().item())
    tangent = float(torch.linalg.norm(force_vec[1:]).item())
    assert normal == pytest.approx(force, rel=1e-5, abs=1e-5)
    assert normal >= tangent
    env.close()


def test_collision_mask_regression_matrix(small_ladder_cfg: LadderConfig) -> None:
    """Permanent structural coverage after Warp-driven mask changes."""
    # Symmetric body↔ladder filter required by mujoco_warp broadphase.
    assert ROBOT_BODY_CONTYPE & LADDER_CONAFFINITY
    assert LADDER_CONTYPE & ROBOT_BODY_CONAFFINITY
    # Hand↔ladder
    assert HAND_POINT_CONTYPE & LADDER_CONAFFINITY
    assert LADDER_CONTYPE & HAND_POINT_CONAFFINITY
    # Hand must not self-collide with robot body geoms.
    assert not (HAND_POINT_CONTYPE & ROBOT_BODY_CONAFFINITY)
    assert not (ROBOT_BODY_CONTYPE & HAND_POINT_CONAFFINITY)

    env = _make_contacts_env(ladder_cfg=small_ladder_cfg)
    mjm = env.sim.mj_model
    hand_id = mjm.geom(f"robot/{LEFT_HAND_POINT}").id
    pelvis_id = mjm.geom("robot/pelvis_collision").id
    rung_id = mjm.geom("ladder/rung_00_geom").id
    rail_id = mjm.geom("ladder/rail_left").id

    assert int(mjm.geom_contype[hand_id]) == HAND_POINT_CONTYPE
    assert int(mjm.geom_conaffinity[hand_id]) == HAND_POINT_CONAFFINITY
    assert int(mjm.geom_contype[pelvis_id]) == ROBOT_BODY_CONTYPE
    assert int(mjm.geom_conaffinity[pelvis_id]) & LADDER_CONTYPE
    assert int(mjm.geom_contype[rung_id]) == LADDER_CONTYPE
    assert int(mjm.geom_conaffinity[rung_id]) == LADDER_CONAFFINITY
    assert int(mjm.geom_contype[rail_id]) == LADDER_CONTYPE
    assert int(mjm.geom_conaffinity[rail_id]) == LADDER_CONAFFINITY
    assert float(mjm.geom_size[hand_id][0]) == pytest.approx(HAND_POINT_RADIUS)
    env.close()


def test_robot_body_is_repelled_by_active_rung(small_ladder_cfg: LadderConfig) -> None:
    """Body–rung collision (not only rails) must resolve under default masks."""
    env = _make_contacts_env(ladder_cfg=small_ladder_cfg)
    robot = env.scene["robot"]
    runtime = LadderRuntime.get(env.unwrapped)
    assert runtime.sample is not None
    active = torch.where(runtime.sample.active_mask[0])[0]
    rung_idx = int(active[0].item())
    rung_pos = env.sim.data.geom_xpos[0, runtime.topology.rung_geom_ids[rung_idx]].detach().cpu().clone()

    pose = torch.cat([rung_pos.unsqueeze(0), robot.data.root_link_quat_w[0:1]], dim=-1)
    robot.write_root_link_pose_to_sim(pose)
    robot.write_root_link_velocity_to_sim(torch.zeros(1, 6, device=env.device))

    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for _ in range(30):
        env.step(action)

    root = robot.data.root_link_pos_w[0]
    assert float(torch.linalg.norm(root - rung_pos).item()) > 0.05
    env.close()
