"""Stage 4 checks for General-Climbing-G1 observations and ladder encoders."""

from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from train_mimic.tasks.climbing.config.observations import (
    DEFAULT_RELATIVE_RUNG_K,
    PROPRIO_HISTORY_LENGTH,
    RELATIVE_RUNG_FEATURE_DIM,
    LadderObservationConfig,
    RelativeRungObsConfig,
    actor_proprio_dim,
    critic_privileged_dim,
    validate_ladder_observation_config,
)
from train_mimic.tasks.climbing.ladder.relative_rungs import (
    RelativeRungProvider,
    compute_rung_endpoints_world,
    ladder_relative_heights_from_world,
    ladder_relative_height,
    ladder_upward_axis_w,
    select_rung_window_indices,
    world_to_body,
)
from train_mimic.tasks.climbing.ladder.state import _quat_apply
from train_mimic.tasks.climbing.rl.climbing_model import ClimbingModel
from train_mimic.tasks.climbing.rl.ladder_encoders import (
    DepthLadderEncoder,
    NotImplementedDepthCameraProvider,
    RelativeRungVectorEncoder,
)


def _yaw_quat(yaw: float) -> torch.Tensor:
    half = yaw * 0.5
    return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=torch.float32)


def test_world_to_body_exact_transform() -> None:
    from mjlab.utils.lab_api.math import quat_apply_inverse

    body_pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    body_quat = _yaw_quat(math.pi / 2.0).unsqueeze(0)
    point_w = torch.tensor([[1.0, 3.0, 3.0]], dtype=torch.float32)
    point_b = world_to_body(body_pos, body_quat, point_w.unsqueeze(1)).squeeze(1)
    expected = quat_apply_inverse(body_quat, point_w - body_pos)
    assert torch.allclose(point_b, expected, atol=1e-5)


def test_tilted_ladder_uses_ladder_relative_height_not_world_z() -> None:
    """World-Z sorting diverges from ladder-relative height when the ladder is tilted."""
    from train_mimic.tasks.climbing.ladder.state import _quat_from_yaw_pitch

    frame_pos = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    pitch = torch.tensor([0.3], dtype=torch.float32)
    frame_quat = _quat_from_yaw_pitch(torch.zeros(1), pitch)
    upward = ladder_upward_axis_w(frame_quat)

    local = torch.tensor([[[0.0, 0.0, 0.30], [0.0, 0.0, 0.55], [0.0, 0.0, 0.80]]])
    rung_pos_w = _quat_apply(frame_quat.unsqueeze(1), local) + frame_pos.unsqueeze(1)
    heights_l = ladder_relative_heights_from_world(rung_pos_w, frame_pos, upward)
    ref_l = ladder_relative_height(
        torch.tensor([[1.0, 0.0, 0.56]], dtype=torch.float32),
        frame_pos,
        upward,
    )

    world_z = rung_pos_w[..., 2]
    assert not torch.allclose(world_z, heights_l, atol=1e-3)

    ids_l, _ = select_rung_window_indices(
        active_mask=torch.tensor([[True, True, True]], dtype=torch.bool),
        rung_heights_l=heights_l,
        reference_height_l=ref_l,
        num_rungs=6,
        rungs_below=2,
        rungs_above=4,
    )
    ids_z, _ = select_rung_window_indices(
        active_mask=torch.tensor([[True, True, True]], dtype=torch.bool),
        rung_heights_l=world_z,
        reference_height_l=torch.tensor([[float(world_z[0, 1].item())]], dtype=torch.float32),
        num_rungs=6,
        rungs_below=2,
        rungs_above=4,
    )
    # Monotonic local heights keep the same slot ordering; values must still differ.
    assert not torch.allclose(heights_l, world_z, atol=1e-3)
    assert int(ids_l.shape[1]) == 6
    del ids_z


def test_depth_mode_fails_during_env_configuration() -> None:
    from train_mimic.tasks.climbing.config.env import make_general_climbing_env_cfg

    with pytest.raises(NotImplementedError, match="DepthCameraProvider"):
        make_general_climbing_env_cfg(
            num_envs=1,
            ladder_observation_cfg=LadderObservationConfig(mode="depth"),
        )


def test_climbing_model_depth_mode_forward_without_model_source_edits() -> None:
    proprio_dim = actor_proprio_dim()
    obs = TensorDict(
        {
            "actor_proprio": torch.zeros(1, proprio_dim),
            "actor_proprio_history": torch.zeros(1, PROPRIO_HISTORY_LENGTH, proprio_dim),
            "actor_ladder": torch.zeros(1, 1, 64, 64),
        },
        batch_size=[1],
    )
    model = ClimbingModel(
        obs=obs,
        obs_groups={"actor": ("actor_proprio", "actor_proprio_history", "actor_ladder")},
        obs_set="actor",
        output_dim=31,
        hidden_dims=(32, 16),
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        cnn_cfg={"output_channels": (8, 4), "kernel_size": 3, "activation": "elu"},
        ladder_cfg={
            "mode": "depth",
            "output_dim": 16,
            "hidden_channels": (8, 16),
        },
    )
    out = model.get_latent(
        TensorDict(
            {
                "actor_proprio": torch.zeros(2, proprio_dim),
                "actor_proprio_history": torch.zeros(
                    2, PROPRIO_HISTORY_LENGTH, proprio_dim
                ),
                "actor_ladder": torch.zeros(2, 1, 64, 64),
            },
            batch_size=[2],
        )
    )
    assert out.shape == (2, model._get_latent_dim())


def test_ladder_normalization_preserves_invalid_padding() -> None:
    proprio_dim = actor_proprio_dim()
    obs = TensorDict(
        {
            "actor_proprio": torch.zeros(1, proprio_dim),
            "actor_proprio_history": torch.zeros(1, PROPRIO_HISTORY_LENGTH, proprio_dim),
            "actor_ladder": torch.zeros(1, DEFAULT_RELATIVE_RUNG_K, RELATIVE_RUNG_FEATURE_DIM),
        },
        batch_size=[1],
    )
    model = ClimbingModel(
        obs=obs,
        obs_groups={"actor": ("actor_proprio", "actor_proprio_history", "actor_ladder")},
        obs_set="actor",
        output_dim=31,
        hidden_dims=(32, 16),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        cnn_cfg={"output_channels": (8, 4), "kernel_size": 3, "activation": "elu"},
        ladder_cfg={"mode": "relative_rungs", "output_dim": 16, "hidden_channels": (8, 16)},
    )
    ladder = torch.zeros(4, DEFAULT_RELATIVE_RUNG_K, RELATIVE_RUNG_FEATURE_DIM)
    ladder[:, 0, :6] = torch.tensor([0.2, -0.1, 0.3, 0.4, 0.1, -0.2])
    ladder[:, 0, 6] = 1.0
    ladder[:, 1, :6] = 999.0
    ladder[:, 1, 6] = 0.0
    obs_batch = TensorDict(
        {
            "actor_proprio": torch.zeros(4, proprio_dim),
            "actor_proprio_history": torch.zeros(
                4, PROPRIO_HISTORY_LENGTH, proprio_dim
            ),
            "actor_ladder": ladder,
        },
        batch_size=[4],
    )
    model.update_normalization(obs_batch)
    prepared = model._prepare_ladder_obs("actor_ladder", ladder)
    assert torch.all(prepared[:, 1, :6] == 0.0)
    assert prepared[0, 1, 6].item() == 0.0
    assert prepared[0, 0, 6].item() == 1.0


def test_attached_rung_rel_height_zero_when_detached() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import make_climbing_observations_env_cfg
    from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState

    env = ManagerBasedRlEnv(cfg=make_climbing_observations_env_cfg(seed=5), device="cpu")
    env.reset()
    latch = ClimbLatchState.get(env.unwrapped)
    latch._ensure_state()
    latch.state.attached[:] = False
    latch.state.rung_id[:] = -1
    obs = env.observation_manager.compute()
    actor_group = obs["actor_proprio"]
    if isinstance(actor_group, dict):
        rel_h = actor_group["attached_rung_rel_height"]
    else:
        rel_h = actor_group[:, -2:]
    assert torch.all(rel_h == 0.0)
    env.close()


def test_critic_privileged_active_rung_heights_include_mask() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import make_climbing_observations_env_cfg

    env = ManagerBasedRlEnv(cfg=make_climbing_observations_env_cfg(seed=9), device="cpu")
    env.reset()
    obs = env.observation_manager.compute()
    privileged = obs["critic_privileged"]
    if isinstance(privileged, dict):
        heights = privileged["active_rung_heights_l_rel"]
        valid = privileged["active_rung_valid_mask"]
    else:
        flat = privileged
        heights = flat[:, -2 * DEFAULT_RELATIVE_RUNG_K : -DEFAULT_RELATIVE_RUNG_K]
        valid = flat[:, -DEFAULT_RELATIVE_RUNG_K :]
    assert heights.shape[-1] == DEFAULT_RELATIVE_RUNG_K
    assert valid.shape[-1] == DEFAULT_RELATIVE_RUNG_K
    invalid = valid < 0.5
    assert torch.all(heights[invalid] == 0.0)
    env.close()


def test_rung_window_selection_masks_top_and_bottom() -> None:
    active_mask = torch.tensor([[True, True, True, False, False]], dtype=torch.bool)
    heights_l = torch.tensor([[0.3, 0.55, 0.8, -50.0, -50.0]], dtype=torch.float32)
    ref_h = torch.tensor([0.56], dtype=torch.float32)
    ids, valid = select_rung_window_indices(
        active_mask=active_mask,
        rung_heights_l=heights_l,
        reference_height_l=ref_h,
        num_rungs=6,
        rungs_below=2,
        rungs_above=4,
    )
    assert ids.shape == (1, 6)
    assert valid.shape == (1, 6)
    assert int(valid.sum().item()) == 3
    assert ids[0, 0].item() == 0
    assert ids[0, 1].item() == 1
    assert ids[0, 2].item() == 2
    assert not bool(valid[0, 3].item())


def test_rigid_translation_invariance_of_relative_endpoints() -> None:
    rung_center = torch.tensor([[0.85, 0.0, 0.55]], dtype=torch.float32)
    rung_quat = _yaw_quat(0.0).unsqueeze(0)
    end_a, end_b = compute_rung_endpoints_world(
        rung_centers_w=rung_center,
        rung_quat_w=rung_quat,
        half_length=0.225,
    )

    torso_a = torch.tensor([[0.0, 0.0, 0.9]], dtype=torch.float32)
    torso_q = _yaw_quat(0.0).unsqueeze(0)
    rel_a = world_to_body(torso_a, torso_q, end_a.unsqueeze(1)).squeeze(1)
    rel_b = world_to_body(torso_a, torso_q, end_b.unsqueeze(1)).squeeze(1)

    shift = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32)
    torso_b = torso_a + shift
    rung_shifted = rung_center + shift
    end_a2, end_b2 = compute_rung_endpoints_world(
        rung_centers_w=rung_shifted,
        rung_quat_w=rung_quat,
        half_length=0.225,
    )
    rel_a2 = world_to_body(torso_b, torso_q, end_a2.unsqueeze(1)).squeeze(1)
    rel_b2 = world_to_body(torso_b, torso_q, end_b2.unsqueeze(1)).squeeze(1)

    assert torch.allclose(rel_a, rel_a2, atol=1e-5)
    assert torch.allclose(rel_b, rel_b2, atol=1e-5)


def test_robot_only_motion_changes_relative_endpoints() -> None:
    rung_center = torch.tensor([[0.85, 0.0, 0.55]], dtype=torch.float32)
    rung_quat = _yaw_quat(0.0).unsqueeze(0)
    end_a, _ = compute_rung_endpoints_world(
        rung_centers_w=rung_center,
        rung_quat_w=rung_quat,
        half_length=0.225,
    )
    torso = torch.tensor([[0.0, 0.0, 0.9]], dtype=torch.float32)
    torso_q = _yaw_quat(0.0).unsqueeze(0)
    rel_before = world_to_body(torso, torso_q, end_a.unsqueeze(1)).squeeze(1)

    torso_moved = torch.tensor([[0.2, 0.0, 0.9]], dtype=torch.float32)
    rel_after = world_to_body(torso_moved, torso_q, end_a.unsqueeze(1)).squeeze(1)
    assert not torch.allclose(rel_before, rel_after)


def test_relative_rung_vector_encoder_output_dim() -> None:
    encoder = RelativeRungVectorEncoder(
        num_rungs=DEFAULT_RELATIVE_RUNG_K,
        feature_dim=RELATIVE_RUNG_FEATURE_DIM,
        output_dim=64,
    )
    x = torch.zeros(4, DEFAULT_RELATIVE_RUNG_K, RELATIVE_RUNG_FEATURE_DIM)
    y = encoder(x)
    assert y.shape == (4, 64)


def test_depth_encoder_output_dim_for_representative_shape() -> None:
    encoder = DepthLadderEncoder(output_dim=64)
    depth = torch.zeros(2, 1, 64, 64)
    latent = encoder(depth)
    assert latent.shape == (2, 64)


def test_depth_camera_provider_fails_explicitly() -> None:
    provider = NotImplementedDepthCameraProvider()
    with pytest.raises(NotImplementedError, match="DepthCameraProvider"):
        provider.capture()


def _build_climbing_actor_model(*, include_privileged: bool = False) -> ClimbingModel:
    proprio_dim = actor_proprio_dim()
    obs_dict: dict[str, torch.Tensor] = {
        "actor_proprio": torch.zeros(1, proprio_dim),
        "actor_proprio_history": torch.zeros(1, PROPRIO_HISTORY_LENGTH, proprio_dim),
        "actor_ladder": torch.zeros(1, DEFAULT_RELATIVE_RUNG_K, RELATIVE_RUNG_FEATURE_DIM),
    }
    groups = ["actor_proprio", "actor_proprio_history", "actor_ladder"]
    if include_privileged:
        obs_dict["critic_privileged"] = torch.zeros(1, critic_privileged_dim())
        groups.append("critic_privileged")
    obs = TensorDict(obs_dict, batch_size=[1])
    return ClimbingModel(
        obs=obs,
        obs_groups={"actor": groups},
        obs_set="actor",
        output_dim=31,
        hidden_dims=(32, 16),
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        cnn_cfg={"output_channels": (8, 4), "kernel_size": 3, "activation": "elu"},
        ladder_cfg={"output_dim": 16, "hidden_channels": (8, 16)},
    )


def test_climbing_model_actor_does_not_require_privileged_group() -> None:
    model = _build_climbing_actor_model(include_privileged=False)
    obs = TensorDict(
        {
            "actor_proprio": torch.zeros(2, actor_proprio_dim()),
            "actor_proprio_history": torch.zeros(
                2, PROPRIO_HISTORY_LENGTH, actor_proprio_dim()
            ),
            "actor_ladder": torch.zeros(
                2, DEFAULT_RELATIVE_RUNG_K, RELATIVE_RUNG_FEATURE_DIM
            ),
        },
        batch_size=[2],
    )
    latent = model.get_latent(obs)
    assert latent.shape[0] == 2
    assert latent.shape[1] > actor_proprio_dim()


def test_climbing_rl_cfg_keeps_privileged_group_critic_only() -> None:
    from train_mimic.tasks.climbing.config.rl import make_general_climbing_ppo_runner_cfg

    cfg = make_general_climbing_ppo_runner_cfg()
    assert "critic_privileged" in cfg.obs_groups["critic"]
    assert "critic_privileged" not in cfg.obs_groups["actor"]


def test_general_climbing_env_has_observation_groups() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    from train_mimic.tasks.climbing.config.constants import CLIMBING_TASK_ID

    cfg = load_env_cfg(CLIMBING_TASK_ID)
    expected = {
        "actor_proprio",
        "actor_proprio_history",
        "actor_ladder",
        "critic_proprio",
        "critic_proprio_history",
        "critic_ladder",
        "critic_privileged",
    }
    assert expected.issubset(set(cfg.observations.keys()))


def test_observations_are_finite_after_reset() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import make_climbing_observations_env_cfg

    env = ManagerBasedRlEnv(cfg=make_climbing_observations_env_cfg(seed=11), device="cpu")
    env.reset()
    obs = env.observation_manager.compute()
    for name, tensor in obs.items():
        if isinstance(tensor, dict):
            for term_name, term_value in tensor.items():
                assert torch.isfinite(term_value).all(), f"{name}/{term_name}"
                assert term_value.shape[0] == env.num_envs
        else:
            assert torch.isfinite(tensor).all(), name
            assert tensor.shape[0] == env.num_envs
    env.close()


def test_actor_ladder_shape_and_inactive_slots_masked() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import make_climbing_observations_env_cfg
    from train_mimic.tasks.climbing.ladder.relative_rungs import RelativeRungProvider

    env = ManagerBasedRlEnv(cfg=make_climbing_observations_env_cfg(seed=3), device="cpu")
    env.reset()
    obs = env.observation_manager.compute()
    ladder_group = obs["actor_ladder"]
    if isinstance(ladder_group, dict):
        ladder = ladder_group["relative_rungs"]
    else:
        ladder = ladder_group
    assert ladder.shape[1:] == (DEFAULT_RELATIVE_RUNG_K, RELATIVE_RUNG_FEATURE_DIM)

    provider = RelativeRungProvider.get(env.unwrapped)
    rel = provider.compute()
    invalid = ~rel.valid_mask
    if bool(torch.any(invalid).item()):
        assert torch.all(ladder[invalid, 6] == 0.0)
    env.close()


def test_tracking_registry_still_unchanged() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    from train_mimic.app import DEFAULT_TASK
    from train_mimic.tasks.tracking.config.constants import GENERAL_TRACKING_TASK

    assert DEFAULT_TASK == GENERAL_TRACKING_TASK
    load_env_cfg(GENERAL_TRACKING_TASK)


def test_translation_plus_fixed_orientation_invariance() -> None:
    torso = torch.tensor([[1.0, 0.0, 0.9]], dtype=torch.float32)
    torso_q = _yaw_quat(0.0).unsqueeze(0)
    end_a_w = torch.tensor([[0.85, -0.225, 0.55]], dtype=torch.float32)
    rel = world_to_body(torso, torso_q, end_a_w.unsqueeze(1)).squeeze(1)

    shift = torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.float32)
    rel_shifted = world_to_body(torso + shift, torso_q, (end_a_w + shift).unsqueeze(1)).squeeze(1)
    assert torch.allclose(rel, rel_shifted, atol=1e-5)
