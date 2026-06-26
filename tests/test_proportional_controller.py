"""Unit tests for the privileged proportional controller (no Unity needed)."""

import numpy as np

from MPC.MPC_Agent import MPCConfig, ProportionalController, MultiAgentProportionalController


def make_state(gx_m, gy_m, gz_m):
    """Build a kinematic state with a given goal offset (in metres).
    Only the first 3 dims (localGoalPos / 50) matter to the controller."""
    state = np.zeros(13, dtype=np.float32)
    state[0] = gx_m / 50.0
    state[1] = gy_m / 50.0
    state[2] = gz_m / 50.0
    return state


def test_goal_straight_ahead_commands_forward_only():
    c = ProportionalController(MPCConfig())
    a = c.get_action(make_state(0.0, 0.0, 10.0))
    forward, lateral, ascent, yaw = a
    assert forward > 0.9          # full forward (10 m saturates)
    assert abs(lateral) < 1e-6
    assert abs(ascent) < 1e-6
    assert abs(yaw) < 1e-6        # already facing the goal


def test_goal_to_the_right_commands_lateral_and_yaw():
    c = ProportionalController(MPCConfig())
    a = c.get_action(make_state(10.0, 0.0, 1.0))
    assert a[1] > 0.5             # lateral toward +x (right)
    assert a[3] > 0.0             # yaw turns toward the goal


def test_goal_above_commands_ascent():
    c = ProportionalController(MPCConfig())
    a = c.get_action(make_state(0.0, 10.0, 1.0))
    assert a[2] > 0.5


def test_actions_within_bounds():
    c = ProportionalController(MPCConfig())
    a = c.get_action(make_state(-100.0, 80.0, -60.0))
    assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_proportional_slowdown_near_goal():
    c = ProportionalController(MPCConfig(p_gain_translation=0.5))
    far = c.get_action(make_state(0.0, 0.0, 10.0))[0]
    near = c.get_action(make_state(0.0, 0.0, 1.0))[0]
    assert far > near             # slows as it approaches
    assert near == 0.5            # 0.5 gain * 1 m


def test_multiagent_interface_matches_mpc():
    config = MPCConfig(kinematic_dim=13, kinematic_offset=0)
    mac = MultiAgentProportionalController(num_agents=3, config=config)
    vector_obs = np.zeros((3, 92), dtype=np.float32)
    vector_obs[:, 2] = 0.2  # goal 10 m ahead for all
    actions, info = mac.get_actions(vector_obs, goals=np.zeros((3, 3)))
    assert actions.shape == (3, 4)
    assert np.all(actions[:, 0] > 0.9)        # all command forward
    assert 'solve_time' in info
    assert mac.update_dynamics([]) == 0.0     # no-op


def test_kinematic_offset_respected():
    # State block placed AFTER a 5-dim raycast block (offset=5)
    config = MPCConfig(kinematic_dim=13, kinematic_offset=5)
    mac = MultiAgentProportionalController(num_agents=1, config=config)
    vector_obs = np.zeros((1, 92), dtype=np.float32)
    vector_obs[0, 5 + 2] = 0.2  # localGoalPos.z at offset 5
    actions, _ = mac.get_actions(vector_obs, goals=np.zeros((1, 3)))
    assert actions[0, 0] > 0.9
