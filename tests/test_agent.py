import numpy as np
import pytest
import torch

from MAPPO.mappo_agent import MAPPOAgent, validate_config, DEFAULT_CONFIG
from MAPPO.rollout_buffer import RolloutBuffer

CAM_SHAPE = (1, 36, 36)
VEC_SHAPE = (8,)
ACTION_DIM = 3
NUM_AGENTS = 2

SMALL_CONFIG = {
    'rollout_length': 16,
    'num_minibatches': 2,
    'ppo_epochs': 2,
    'max_steps': 64,
}


@pytest.fixture
def agent():
    torch.manual_seed(0)
    np.random.seed(0)
    return MAPPOAgent(CAM_SHAPE, VEC_SHAPE, ACTION_DIM, NUM_AGENTS, SMALL_CONFIG)


def full_buffer(agent, seed=0):
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(
        num_steps=agent.num_steps,
        num_agents=NUM_AGENTS,
        obs_shape=(agent.encoded_obs_dim,),
        action_dim=ACTION_DIM,
    )
    for t in range(agent.num_steps):
        obs = rng.standard_normal((NUM_AGENTS, agent.encoded_obs_dim)).astype(np.float32)
        next_obs = rng.standard_normal((NUM_AGENTS, agent.encoded_obs_dim)).astype(np.float32)
        actions = np.tanh(rng.standard_normal((NUM_AGENTS, ACTION_DIM))).astype(np.float32)
        buf.store(obs, next_obs, actions,
                  reward=rng.standard_normal(NUM_AGENTS),
                  done=np.zeros(NUM_AGENTS),
                  value=rng.standard_normal(NUM_AGENTS),
                  log_prob=rng.standard_normal(NUM_AGENTS) - 2.0)
    buf.compute_returns_and_advantages(np.zeros(NUM_AGENTS))
    return buf


def test_unknown_config_key_raises():
    with pytest.raises(ValueError, match="Unknown config keys"):
        validate_config({'num_mini_batches': 8})  # the historical typo'd key
    with pytest.raises(ValueError, match="Unknown config keys"):
        MAPPOAgent(CAM_SHAPE, VEC_SHAPE, ACTION_DIM, NUM_AGENTS, {'lr': 1e-4})


def test_defaults_applied():
    cfg = validate_config({})
    assert cfg == DEFAULT_CONFIG


def test_encoded_obs_dim_derived_from_encoder(agent):
    assert agent.encoded_obs_dim == agent.vision_encoder.output_dim + VEC_SHAPE[0]


def test_encoder_is_frozen(agent):
    assert not agent.vision_encoder.training  # eval mode
    assert all(not p.requires_grad for p in agent.vision_encoder.parameters())
    # and excluded from the optimizer
    opt_params = {id(p) for g in agent.optimizer.param_groups for p in g['params']}
    assert all(id(p) not in opt_params for p in agent.vision_encoder.parameters())


def test_act_and_get_values_shapes(agent):
    cam = np.random.rand(NUM_AGENTS, *CAM_SHAPE).astype(np.float32)
    vec = np.random.rand(NUM_AGENTS, *VEC_SHAPE).astype(np.float32)
    actions, log_probs, values, encoded = agent.act(cam, vec)
    assert actions.shape == (NUM_AGENTS, ACTION_DIM)
    assert log_probs.shape == (NUM_AGENTS,)
    assert values.shape == (NUM_AGENTS,)
    assert encoded.shape == (NUM_AGENTS, agent.encoded_obs_dim)
    assert agent.get_values(encoded).shape == (NUM_AGENTS,)


def test_intrinsic_rewards_clipped(agent):
    obs = torch.randn(NUM_AGENTS, agent.encoded_obs_dim)
    next_obs = torch.randn(NUM_AGENTS, agent.encoded_obs_dim)
    actions = torch.randn(NUM_AGENTS, ACTION_DIM)
    for _ in range(5):
        r = agent.compute_intrinsic_rewards(obs, next_obs, actions)
    assert r.shape == (NUM_AGENTS,)
    clip = agent.intrinsic_reward_clip
    assert np.all(r >= -clip) and np.all(r <= clip)


def test_train_runs_and_returns_finite_stats(agent):
    """The test that would have caught both historical curiosity-indexing
    bugs and the tuple-vs-int shape check."""
    stats = agent.train(full_buffer(agent))
    for key in ('policy_loss', 'value_loss', 'entropy', 'approx_kl',
                'clip_fraction', 'explained_variance'):
        assert np.isfinite(stats[key]), f"{key} is not finite"
    assert agent.update_count == 1


def test_train_updates_actor_and_critic(agent):
    before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    critic_before = {k: v.clone() for k, v in agent.critic.state_dict().items()}
    agent.train(full_buffer(agent))
    actor_moved = any(not torch.equal(before[k], v)
                      for k, v in agent.actor.state_dict().items())
    critic_moved = any(not torch.equal(critic_before[k], v)
                       for k, v in agent.critic.state_dict().items())
    assert actor_moved, "actor parameters did not change after train()"
    assert critic_moved, "critic parameters did not change after train()"


def test_entropy_coef_decays(agent):
    coefs = []
    for i in range(3):
        agent.train(full_buffer(agent, seed=i))
        coefs.append(agent.entropy_coef)
    assert coefs[0] >= coefs[1] >= coefs[2]
    assert coefs[-1] >= agent.entropy_coef_final


def test_checkpoint_round_trip(tmp_path, agent):
    agent.train(full_buffer(agent))
    extra = {'total_steps': 12345, 'best_reward': 7.5, 'curriculum_stage': 2}
    path = tmp_path / "ckpt.pth"
    agent.save(path, extra=extra)

    torch.manual_seed(99)
    fresh = MAPPOAgent(CAM_SHAPE, VEC_SHAPE, ACTION_DIM, NUM_AGENTS, SMALL_CONFIG)
    loaded_extra = fresh.load(path)

    assert loaded_extra == extra
    assert fresh.update_count == agent.update_count
    for k, v in agent.actor.state_dict().items():
        assert torch.equal(fresh.actor.state_dict()[k], v)
    for k, v in agent.critic.state_dict().items():
        assert torch.equal(fresh.critic.state_dict()[k], v)
    assert fresh.intrinsic_reward_normalizer.count == agent.intrinsic_reward_normalizer.count
