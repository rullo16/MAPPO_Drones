"""
End-to-end smoke test: two full collect -> GAE -> train cycles against a
scripted dummy environment, exercising the same code path as train.py
(act / truncation handling / store / compute_returns / train).

This is the test that catches "HEAD doesn't run" before a 24-hour GPU job
finds out the hard way.
"""

import numpy as np
import torch

from MAPPO.mappo_agent import MAPPOAgent
from MAPPO.rollout_buffer import RolloutBuffer

CAM_SHAPE = (1, 36, 36)
VEC_SHAPE = (6,)
ACTION_DIM = 2
NUM_AGENTS = 2
ROLLOUT = 16


class DummyEnv:
    """Random observations/rewards; episode ends every `episode_len` steps."""

    def __init__(self, episode_len=5, seed=0):
        self.rng = np.random.default_rng(seed)
        self.episode_len = episode_len
        self.t = 0

    def reset(self):
        self.t = 0
        return self._obs()

    def _obs(self):
        cam = self.rng.random((NUM_AGENTS, *CAM_SHAPE), dtype=np.float32)
        vec = self.rng.random((NUM_AGENTS, *VEC_SHAPE), dtype=np.float32)
        return cam, vec

    def step(self, actions):
        assert actions.shape == (NUM_AGENTS, ACTION_DIM)
        self.t += 1
        rewards = self.rng.standard_normal(NUM_AGENTS).astype(np.float32)
        # one agent terminates at the episode boundary, the other is truncated
        done = self.t >= self.episode_len
        dones = np.array([done, False], dtype=np.float32)
        return self._obs(), rewards, dones, done


def test_two_full_training_cycles():
    torch.manual_seed(0)
    np.random.seed(0)

    config = {
        'rollout_length': ROLLOUT,
        'num_minibatches': 2,
        'ppo_epochs': 2,
        'max_steps': ROLLOUT * 2,
    }
    agent = MAPPOAgent(CAM_SHAPE, VEC_SHAPE, ACTION_DIM, NUM_AGENTS, config)
    buffer = RolloutBuffer(ROLLOUT, NUM_AGENTS, (agent.encoded_obs_dim,), ACTION_DIM)
    env = DummyEnv()

    cam, vec = env.reset()
    encoded = agent.encode_observations(cam, vec)

    for _cycle in range(2):
        for _ in range(ROLLOUT):
            actions, log_probs, values = agent.get_action(encoded)
            (cam_n, vec_n), rewards, dones, episode_over = env.step(actions)
            encoded_next = agent.encode_observations(cam_n, vec_n)

            intrinsic = agent.compute_intrinsic_rewards(encoded, encoded_next, actions)
            total_rewards = rewards + intrinsic * agent.curiosity_coef

            if episode_over:
                # time-limit correction for the truncated agent (mirrors train.py)
                next_values = agent.get_values(encoded_next)
                truncated = dones < 0.5
                total_rewards = total_rewards + truncated * 0.99 * next_values
                store_dones = np.ones_like(dones)
            else:
                store_dones = dones

            buffer.store(encoded.cpu().numpy(), encoded_next.cpu().numpy(),
                         actions, total_rewards, store_dones, values, log_probs)

            if episode_over:
                cam, vec = env.reset()
                encoded = agent.encode_observations(cam, vec)
            else:
                encoded = encoded_next

        buffer.compute_returns_and_advantages(agent.get_values(encoded))
        stats = agent.train(buffer)

        for key, value in stats.items():
            assert np.isfinite(value), f"{key} is not finite after cycle {_cycle}"

    assert agent.update_count == 2
