import numpy as np
import pytest

from MAPPO.rollout_buffer import RolloutBuffer


def make_buffer(num_steps=3, num_agents=1, obs_dim=2, action_dim=1, gamma=0.5, lam=0.5):
    return RolloutBuffer(num_steps, num_agents, (obs_dim,), action_dim,
                         gamma=gamma, gae_lambda=lam)


def test_gae_hand_computed():
    """GAE against a hand-computed 3-step example with a mid-buffer done.

    gamma = lambda = 0.5, rewards = [1, 1, 1], values = [0.5, 0.5, 0.5],
    dones = [0, 1, 0], last_value = 0.5.

    t=2: delta = 1 + 0.5*0.5 - 0.5 = 0.75            -> A2 = 0.75
    t=1: done -> delta = 1 + 0 - 0.5 = 0.5, no carry -> A1 = 0.5
    t=0: delta = 1 + 0.5*0.5 - 0.5 = 0.75
         A0 = 0.75 + 0.5*0.5 * A1 = 0.75 + 0.125     -> A0 = 0.875
    returns = A + V = [1.375, 1.0, 1.25]
    """
    buf = make_buffer()
    obs = np.zeros((1, 2), dtype=np.float32)
    for t in range(3):
        buf.store(obs, obs, np.zeros((1, 1)), reward=[1.0],
                  done=[1.0 if t == 1 else 0.0], value=[0.5], log_prob=[0.0])

    buf.compute_returns_and_advantages(np.array([0.5]))

    np.testing.assert_allclose(buf.advantages[:, 0], [0.875, 0.5, 0.75], atol=1e-6)
    np.testing.assert_allclose(buf.returns[:, 0], [1.375, 1.0, 1.25], atol=1e-6)


def test_done_blocks_bootstrap_across_episodes():
    """The value of the post-done state must not leak into the advantage."""
    buf = make_buffer(num_steps=2)
    obs = np.zeros((1, 2), dtype=np.float32)
    # Step 0 ends an episode; step 1 starts a new one with a huge value.
    buf.store(obs, obs, np.zeros((1, 1)), reward=[1.0], done=[1.0],
              value=[0.0], log_prob=[0.0])
    buf.store(obs, obs, np.zeros((1, 1)), reward=[0.0], done=[0.0],
              value=[1000.0], log_prob=[0.0])

    buf.compute_returns_and_advantages(np.array([0.0]))
    # A0 = r0 - V0 = 1.0, untouched by the 1000.0 value at t=1
    assert buf.advantages[0, 0] == pytest.approx(1.0)


def test_store_overflow_raises():
    buf = make_buffer(num_steps=1)
    obs = np.zeros((1, 2), dtype=np.float32)
    buf.store(obs, obs, np.zeros((1, 1)), [0.0], [0.0], [0.0], [0.0])
    with pytest.raises(ValueError, match="full"):
        buf.store(obs, obs, np.zeros((1, 1)), [0.0], [0.0], [0.0], [0.0])


def test_get_requires_full_buffer_and_resets():
    buf = make_buffer(num_steps=2)
    obs = np.zeros((1, 2), dtype=np.float32)
    buf.store(obs, obs, np.zeros((1, 1)), [0.0], [0.0], [0.0], [0.0])
    with pytest.raises(ValueError, match="not full"):
        buf.get()
    buf.store(obs, obs, np.zeros((1, 1)), [0.0], [0.0], [0.0], [0.0])
    buf.compute_returns_and_advantages(np.array([0.0]))
    data = buf.get()
    assert data['observations'].shape == (2, 1, 2)
    assert buf.ptr == 0
