"""Stage 3 checks for General-Climbing-G1 attach/detach latch."""

from __future__ import annotations

import mujoco
import pytest
import torch

from train_mimic.tasks.climbing.config.env import (
    _recommended_njmax,
    make_climbing_latch_env_cfg,
    make_general_climbing_env_cfg,
)
from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.latch import LatchConfig
from train_mimic.tasks.climbing.debug_latch import build_latch_action
from train_mimic.tasks.climbing.debug_probe import (
    nudge_probe_rung_away_from_ladder,
    place_probe_rung_near_hand,
    set_probe_rung_hand_only_collision,
)
from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.latch import (
    ClimbLatchState,
    LatchCommand,
    LatchTopology,
    latch_constraint_name,
)
from train_mimic.tasks.climbing.ladder.state import LadderRuntime
from train_mimic.tasks.climbing.mdp.actions import LatchActionCfg

CONTACT_PENETRATION_M = 0.005
CONTACT_CLEARANCE_M = 0.05


@pytest.fixture
def small_ladder_cfg() -> LadderConfig:
    return LadderConfig(
        max_rungs=6,
        min_active_rungs=4,
        max_active_rungs=4,
        sites_per_rung=3,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
    )


@pytest.fixture
def latch_cfg() -> LatchConfig:
    # Stage-3 transition / pull diagnostics intentionally keep an unbreakable,
    # stiffer latch so residual bounds and hold sequences stay comparable to the
    # original Stage-3 gate. Training defaults are softer + break_force=500.
    return LatchConfig(
        attach_threshold=0.5,
        detach_threshold=-0.5,
        capture_radius=0.15,
        break_force=None,
        solref=(0.02, 1.0),
    )


def _make_latch_env(
    seed: int = 7,
    *,
    num_envs: int = 1,
    ladder_cfg: LadderConfig | None = None,
    latch_cfg: LatchConfig | None = None,
):
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_climbing_latch_env_cfg(
        num_envs=num_envs,
        seed=seed,
        ladder_cfg=ladder_cfg,
        latch_cfg=latch_cfg,
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    return env


def _step_latch(
    env,
    *,
    latch_left: float = 0.0,
    latch_right: float = 0.0,
) -> None:
    action = build_latch_action(env, latch_left=latch_left, latch_right=latch_right)
    env.step(action)


def _establish_hand_contact(env, *, hand: str, rung_id: int) -> None:
    """Step once with neutral latch so contact sensors reflect the current probe pose."""
    set_probe_rung_hand_only_collision(env.unwrapped, rung_id)
    place_probe_rung_near_hand(
        env.unwrapped,
        hand=hand,
        rung_id=rung_id,
        separation_m=-CONTACT_PENETRATION_M,
    )
    _step_latch(env)


def _latch_state(env):
    return ClimbLatchState.get(env.unwrapped).state


def test_general_climbing_env_declares_latch_equalities(small_ladder_cfg: LadderConfig) -> None:
    cfg = make_general_climbing_env_cfg(num_envs=1, seed=0, ladder_cfg=small_ladder_cfg)
    assert cfg.actions["latch"].ladder_cfg.max_rungs == small_ladder_cfg.max_rungs
    assert isinstance(cfg.actions["latch"], LatchActionCfg)
    assert cfg.sim.njmax == _recommended_njmax(small_ladder_cfg)


def test_latch_topology_matches_compiled_model(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    topology = LatchTopology.from_model(env.sim.mj_model, small_ladder_cfg)
    expected = 2 * small_ladder_cfg.max_rungs * small_ladder_cfg.sites_per_rung
    assert env.sim.mj_model.neq == expected
    assert topology.eq_ids.shape == (
        2,
        small_ladder_cfg.max_rungs,
        small_ladder_cfg.sites_per_rung,
    )
    env.close()


def test_action_dim_includes_two_latch_commands(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    assert env.action_manager.total_action_dim == 31
    assert env.action_manager.action_term_dim == [29, 2]
    env.close()


def test_hysteresis_decode() -> None:
    cfg = LatchConfig(attach_threshold=0.5, detach_threshold=-0.5)
    raw = torch.tensor([[0.6, 0.0], [0.0, -0.6], [0.1, -0.1]])
    cmd = ClimbLatchState.decode_command(raw, cfg)
    assert int(cmd[0, 0].item()) == int(LatchCommand.ATTACH)
    assert int(cmd[0, 1].item()) == int(LatchCommand.NEUTRAL)
    assert int(cmd[1, 1].item()) == int(LatchCommand.DETACH)
    assert int(cmd[2, 0].item()) == int(LatchCommand.NEUTRAL)
    assert int(cmd[2, 1].item()) == int(LatchCommand.NEUTRAL)


@pytest.mark.parametrize(
    ("contact", "command", "expect_attached", "expect_invalid"),
    [
        (False, LatchCommand.ATTACH, False, True),
        (True, LatchCommand.NEUTRAL, False, False),
        (True, LatchCommand.DETACH, False, False),
        (True, LatchCommand.ATTACH, True, False),
    ],
)
def test_detached_transition_table(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
    contact: bool,
    command: LatchCommand,
    expect_attached: bool,
    expect_invalid: bool,
) -> None:
    env = _make_latch_env(ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    separation = -CONTACT_PENETRATION_M if contact else CONTACT_CLEARANCE_M + 0.05
    set_probe_rung_hand_only_collision(env.unwrapped, 2)
    place_probe_rung_near_hand(env.unwrapped, hand="left", rung_id=2, separation_m=separation)
    if contact:
        _step_latch(env)
    cmd_value = {
        LatchCommand.ATTACH: 1.0,
        LatchCommand.DETACH: -1.0,
        LatchCommand.NEUTRAL: 0.0,
    }[command]
    _step_latch(env, latch_left=cmd_value)
    state = _latch_state(env)
    assert bool(state.attached[0, 0].item()) == expect_attached
    assert bool(state.invalid_attach_request[0, 0].item()) == expect_invalid
    if expect_attached:
        assert int(state.rung_id[0, 0].item()) == 2
        assert int(state.active_eq_id[0, 0].item()) >= 0
    env.close()


def test_attached_hand_ignores_reattach_and_other_rung_contact(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    _establish_hand_contact(env, hand="left", rung_id=2)
    _step_latch(env, latch_left=1.0)
    state = _latch_state(env)
    assert bool(state.attached[0, 0].item())
    attached_rung = int(state.rung_id[0, 0].item())
    attached_eq = int(state.active_eq_id[0, 0].item())

    place_probe_rung_near_hand(
        env.unwrapped, hand="left", rung_id=3, separation_m=-CONTACT_PENETRATION_M
    )
    _step_latch(env, latch_left=1.0)
    state = _latch_state(env)
    assert bool(state.attached[0, 0].item())
    assert int(state.rung_id[0, 0].item()) == attached_rung
    assert int(state.active_eq_id[0, 0].item()) == attached_eq
    env.close()


def test_detach_releases_constraint(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    _establish_hand_contact(env, hand="left", rung_id=2)
    _step_latch(env, latch_left=1.0)
    assert bool(_latch_state(env).attached[0, 0].item())

    _step_latch(env, latch_left=-1.0)
    state = _latch_state(env)
    assert not bool(state.attached[0, 0].item())
    assert int(state.active_eq_id[0, 0].item()) == -1
    assert not bool(env.sim.data.eq_active[0].any().item())
    env.close()


def test_exactly_one_active_constraint_per_hand(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(num_envs=1, ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    latch = ClimbLatchState.get(env.unwrapped)
    _establish_hand_contact(env, hand="left", rung_id=2)
    _step_latch(env, latch_left=1.0)
    active = env.sim.data.eq_active[0].to(torch.bool)
    assert int(active.sum().item()) == 1
    eq_id = int(_latch_state(env).active_eq_id[0, 0].item())
    assert bool(active[eq_id].item())
    assert latch.topology.eq_names[eq_id].startswith("latch_left_")
    env.close()


def test_both_hands_attach_independently(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    _establish_hand_contact(env, hand="left", rung_id=2)
    _step_latch(env, latch_left=1.0)
    _establish_hand_contact(env, hand="right", rung_id=3)
    _step_latch(env, latch_left=1.0, latch_right=1.0)
    state = _latch_state(env)
    assert bool(state.attached[0, 0].item())
    assert bool(state.attached[0, 1].item())
    assert int(state.rung_id[0, 0].item()) == 2
    assert int(state.rung_id[0, 1].item()) == 3
    assert int(env.sim.data.eq_active[0].sum().item()) == 2
    env.close()


def test_reset_clears_latch_state_and_equalities(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    _establish_hand_contact(env, hand="left", rung_id=2)
    _step_latch(env, latch_left=1.0)
    assert bool(_latch_state(env).attached[0, 0].item())

    env.reset()
    state = _latch_state(env)
    assert not bool(state.attached[0, 0].item())
    assert int(state.rung_id[0, 0].item()) == ClimbLatchState.NONE_RUNG_ID
    assert not bool(env.sim.data.eq_active[0].any().item())
    env.close()


def test_overload_break_detaches_and_lockouts_reattach(
    small_ladder_cfg: LadderConfig,
) -> None:
    """Connect equality overload must break; sticky attach must not re-weld."""
    env = _make_latch_env(
        ladder_cfg=small_ladder_cfg,
        latch_cfg=LatchConfig(
            attach_threshold=0.5,
            detach_threshold=-0.5,
            capture_radius=0.15,
            break_force=200.0,
            solref=(0.02, 1.0),
        ),
    )
    _establish_hand_contact(env, hand="left", rung_id=2)
    _step_latch(env, latch_left=1.0)
    assert bool(_latch_state(env).attached[0, 0].item())

    robot = env.unwrapped.scene["robot"]
    broke = False
    for _ in range(30):
        vel = torch.zeros(1, 6, device=env.device, dtype=torch.float32)
        vel[0, 0] = -5.0
        robot.write_root_link_velocity_to_sim(vel)
        _step_latch(env, latch_left=1.0)
        state = _latch_state(env)
        if not bool(state.attached[0, 0].item()):
            broke = True
            assert bool(state.overload_lockout[0, 0].item())
            break
    assert broke, "expected connect overload to detach the latch"

    # Holding ATTACH after overload must not immediately re-weld.
    for _ in range(5):
        _step_latch(env, latch_left=1.0)
        assert not bool(_latch_state(env).attached[0, 0].item())
        assert bool(_latch_state(env).overload_lockout[0, 0].item())

    # Neutral command clears lockout; a later attach can succeed again.
    _step_latch(env, latch_left=0.0)
    assert not bool(_latch_state(env).overload_lockout[0, 0].item())
    env.close()


def test_pull_keeps_hand_near_attached_site(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    _establish_hand_contact(env, hand="left", rung_id=2)
    _step_latch(env, latch_left=1.0)
    latch = ClimbLatchState.get(env.unwrapped)
    site_id = int(_latch_state(env).site_id[0, 0].item())
    hand_site_id = latch.topology.hand_site_ids[0]
    rung_site_id = latch.topology.rung_site_ids[2][site_id]
    hand_before = env.sim.data.site_xpos[0, hand_site_id].clone()

    max_penetration = 0.0
    max_force = 0.0
    max_residual = 0.0
    for _ in range(20):
        nudge_probe_rung_away_from_ladder(env.unwrapped, 2, step_m=0.004)
        _step_latch(env, latch_left=1.0)
        diag = latch.geometry_diagnostics()
        sep = float(diag["signed_separation"][0, 0].item())
        if torch.isfinite(torch.tensor(sep)):
            max_penetration = max(max_penetration, max(0.0, -sep))
        max_force = max(max_force, float(diag["contact_normal_force"][0, 0].item()))
        resid = float(diag["equality_residual"][0, 0].item())
        if torch.isfinite(torch.tensor(resid)):
            max_residual = max(max_residual, resid)

    hand_after = env.sim.data.site_xpos[0, hand_site_id]
    rung_site_after = env.sim.data.site_xpos[0, rung_site_id]
    error = torch.linalg.norm(hand_after - rung_site_after).item()
    travel = torch.linalg.norm(hand_after - hand_before).item()
    assert error < 0.12
    assert travel > 0.01
    # External latch target: steady penetration only a few millimetres.
    # Residual bound includes soft-constraint compliance under probe pull plus
    # lateral site discretization from the hand-only probe offset.
    assert max_penetration < 0.008
    assert max_residual < 0.12
    assert max_force < 5.0e4
    assert torch.isfinite(env.sim.data.qvel).all()
    env.close()


def test_latch_target_sites_are_offset_from_rung_axis(small_ladder_cfg: LadderConfig) -> None:
    from train_mimic.tasks.climbing.ladder.generator import latch_site_normal_offset
    from train_mimic.tasks.climbing.ladder.state import LadderRuntime

    env = _make_latch_env(ladder_cfg=small_ladder_cfg)
    runtime = LadderRuntime.get(env.unwrapped)
    assert runtime.sample is not None
    offset = latch_site_normal_offset(small_ladder_cfg)
    assert runtime.topology.site_normal_offset == pytest.approx(offset)

    rung_id = 2
    geom_id = runtime.topology.rung_geom_ids[rung_id]
    site_id = runtime.topology.rung_site_ids[rung_id][0]
    # Batched local model fields after apply().
    center = env.sim.model.geom_pos[0, geom_id]
    site = env.sim.model.site_pos[0, site_id]
    assert float(site[0].item()) == pytest.approx(float(center[0].item()) - offset, abs=1e-5)
    env.close()


def test_nearest_site_is_selected(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    latch = ClimbLatchState.get(env.unwrapped)
    _establish_hand_contact(env, hand="left", rung_id=2)
    _step_latch(env, latch_left=1.0)
    site_id = int(_latch_state(env).site_id[0, 0].item())

    hand_pos = env.sim.data.site_xpos[0, latch.topology.hand_site_ids[0]]
    dists = []
    for sid in range(small_ladder_cfg.sites_per_rung):
        rung_site = latch.topology.rung_site_ids[2][sid]
        pos = env.sim.data.site_xpos[0, rung_site]
        dists.append(float(torch.linalg.norm(hand_pos - pos).item()))
    assert site_id == int(min(range(len(dists)), key=dists.__getitem__))
    env.close()


def test_hysteresis_prevents_threshold_chatter(latch_cfg: LatchConfig) -> None:
    cfg = latch_cfg
    values = torch.tensor([0.49, 0.51, 0.49, 0.51])
    cmds = [int(ClimbLatchState.decode_command(v.unsqueeze(0).unsqueeze(0), cfg)[0, 0].item()) for v in values]
    assert cmds[0] == int(LatchCommand.NEUTRAL)
    assert cmds[1] == int(LatchCommand.ATTACH)
    assert cmds[2] == int(LatchCommand.NEUTRAL)
    assert cmds[3] == int(LatchCommand.ATTACH)


def test_latch_state_tensors_are_finite_after_steps(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    set_probe_rung_hand_only_collision(env.unwrapped, 2)
    place_probe_rung_near_hand(
        env.unwrapped, hand="left", rung_id=2, separation_m=-CONTACT_PENETRATION_M
    )
    for cmd in (0.0, 1.0, 1.0, -1.0, 1.0):
        _step_latch(env, latch_left=cmd)
    state = _latch_state(env)
    for field in (
        state.hand_rung_height if hasattr(state, "hand_rung_height") else (),
    ):
        del field
    assert torch.isfinite(state.latch_command).all()
    assert torch.isfinite(env.sim.data.site_xpos).all()
    env.close()


def test_constraint_names_are_stable(small_ladder_cfg: LadderConfig) -> None:
    assert latch_constraint_name("left", 2, 1) == "latch_left_rung_02_site_01"


def test_cpu_eq_active_matches_backend_state(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    latch = ClimbLatchState.get(env.unwrapped)
    _establish_hand_contact(env, hand="left", rung_id=2)
    _step_latch(env, latch_left=1.0)
    eq_id = int(_latch_state(env).active_eq_id[0, 0].item())
    sim_active = bool(env.sim.data.eq_active[0, eq_id].item())
    assert sim_active
    assert latch.backend.read_active_eq_ids(0, 0) == [eq_id]
    assert int(env.sim.data.eq_active[0].sum().item()) == 1
    env.close()


@pytest.mark.parametrize("num_envs", [1, 2])
def test_batched_latch_state_shape(
    num_envs: int,
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(num_envs=num_envs, ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    state = _latch_state(env)
    assert state.attached.shape == (num_envs, 2)
    assert state.rung_id.shape == (num_envs, 2)
    assert state.site_id.shape == (num_envs, 2)
    assert state.latch_command.shape == (num_envs, 2)
    env.close()


def test_per_world_latch_isolation(
    latch_cfg: LatchConfig,
) -> None:
    """Attaching in one env must not change eq_active or sites in other envs."""
    ladder_cfg = LadderConfig(
        max_rungs=6,
        min_active_rungs=6,
        max_active_rungs=6,
        sites_per_rung=3,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
    )
    env = _make_latch_env(
        num_envs=4,
        ladder_cfg=ladder_cfg,
        latch_cfg=latch_cfg,
    )
    latch = ClimbLatchState.get(env.unwrapped)
    runtime = LadderRuntime.get(env.unwrapped)
    assert runtime.sample is not None
    assert env.sim.model.site_pos.shape[0] == 4

    plans = (
        (0, "left", 2),
        (1, "left", 5),
        (2, "right", 3),
    )
    for env_idx, hand, rung_id in plans:
        set_probe_rung_hand_only_collision(env.unwrapped, rung_id)
        place_probe_rung_near_hand(
            env.unwrapped,
            hand=hand,
            rung_id=rung_id,
            separation_m=-CONTACT_PENETRATION_M,
            env_ids=torch.tensor([env_idx], dtype=torch.int64),
        )

    # Probe moves are per-env: env0 rung2 local pose differs from env3's sampled pose.
    geom_id = runtime.topology.rung_geom_ids[2]
    assert not torch.allclose(
        env.sim.model.geom_pos[0, geom_id],
        env.sim.model.geom_pos[3, geom_id],
    )

    # Neutral step so contact sensors latch to probe poses.
    _step_latch(env)
    action = build_latch_action(env, latch_left=0.0, latch_right=0.0)
    action[0, 29] = 1.0  # env0 left attach
    action[1, 29] = 1.0  # env1 left attach
    action[2, 30] = 1.0  # env2 right attach
    # env3 remains detached
    env.step(action)

    state = _latch_state(env)
    assert bool(state.attached[0, 0].item()) and int(state.rung_id[0, 0].item()) == 2
    assert bool(state.attached[1, 0].item()) and int(state.rung_id[1, 0].item()) == 5
    assert bool(state.attached[2, 1].item()) and int(state.rung_id[2, 1].item()) == 3
    assert not bool(state.attached[3].any().item())

    eq_active = env.sim.data.eq_active.to(torch.bool)
    for env_idx, hand, rung_id in plans:
        hand_idx = 0 if hand == "left" else 1
        eq_id = int(state.active_eq_id[env_idx, hand_idx].item())
        assert eq_id >= 0
        assert bool(eq_active[env_idx, eq_id].item())
        # No other env has this equality active.
        for other in range(4):
            if other == env_idx:
                continue
            assert not bool(eq_active[other, eq_id].item())
        assert int(eq_active[env_idx].sum().item()) == 1

    assert int(eq_active[3].sum().item()) == 0

    # Subset reset of envs 1 and 2 clears only those worlds.
    env.reset(env_ids=torch.tensor([1, 2], dtype=torch.int64))
    state = _latch_state(env)
    eq_active = env.sim.data.eq_active.to(torch.bool)
    assert bool(state.attached[0, 0].item())
    assert not bool(state.attached[1].any().item())
    assert not bool(state.attached[2].any().item())
    assert not bool(state.attached[3].any().item())
    assert int(eq_active[0].sum().item()) == 1
    assert int(eq_active[1].sum().item()) == 0
    assert int(eq_active[2].sum().item()) == 0

    diag = latch.geometry_diagnostics()
    resid = float(diag["equality_residual"][0, 0].item())
    sep = float(diag["signed_separation"][0, 0].item())
    assert torch.isfinite(torch.tensor(resid))
    assert resid < 0.12
    assert sep > -0.008  # no deep penetration into the rung
    env.close()


def test_subset_reset_keeps_first_step_finite(
    small_ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    env = _make_latch_env(num_envs=2, ladder_cfg=small_ladder_cfg, latch_cfg=latch_cfg)
    _establish_hand_contact(env, hand="left", rung_id=2)
    action = build_latch_action(env, latch_left=1.0, latch_right=0.0)
    env.step(action)
    assert bool(_latch_state(env).attached[0, 0].item())

    env.reset(env_ids=torch.tensor([0], dtype=torch.int64))
    _step_latch(env)
    assert torch.isfinite(env.sim.data.qvel).all()
    assert not bool(_latch_state(env).attached[0].any().item())
    env.close()


def test_reset_event_order_deactivates_before_ladder_resample() -> None:
    cfg = make_general_climbing_env_cfg(num_envs=1, seed=0)
    reset_names = [name for name, term in cfg.events.items() if term.mode == "reset"]
    assert reset_names.index("reset_climb_latch") < reset_names.index("reset_ladder")
    assert reset_names.index("reset_climb_contacts") < reset_names.index("reset_ladder")


def test_tracking_registry_still_unchanged() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    from train_mimic.app import DEFAULT_TASK
    from train_mimic.tasks.tracking.config.constants import GENERAL_TRACKING_TASK

    assert DEFAULT_TASK == GENERAL_TRACKING_TASK
    load_env_cfg(GENERAL_TRACKING_TASK)
