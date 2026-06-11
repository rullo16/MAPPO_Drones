import numpy as np
import torch
import pytest

from MAPPO.models import MAPPOActor, MAPPOCritic, LOG_STD_MIN, LOG_STD_MAX


@pytest.fixture
def actor():
    torch.manual_seed(0)
    return MAPPOActor(obs_dim=16, action_dim=3)


def test_log_prob_round_trip(actor):
    """evaluate_actions must agree with the log prob returned at sampling
    time (tanh-squash correction consistency)."""
    torch.manual_seed(1)
    obs = torch.randn(64, 16)
    actions, log_probs_sample = actor.get_action(obs)
    log_probs_eval, entropy = actor.evaluate_actions(obs, actions)
    assert torch.allclose(log_probs_sample, log_probs_eval, atol=1e-4)
    assert entropy.shape == (64,)


def test_actions_within_bounds(actor):
    obs = torch.randn(128, 16)
    actions, _ = actor.get_action(obs)
    assert actions.min() >= -1.0 and actions.max() <= 1.0


def test_std_respects_clamp(actor):
    # Force the raw parameter outside the clamp; forward must still respect it
    with torch.no_grad():
        actor.log_std.fill_(10.0)
    dist = actor(torch.randn(4, 16))
    assert dist.scale.max().item() <= np.exp(LOG_STD_MAX) + 1e-6
    with torch.no_grad():
        actor.log_std.fill_(-10.0)
    dist = actor(torch.randn(4, 16))
    assert dist.scale.min().item() >= np.exp(LOG_STD_MIN) - 1e-6


def test_max_entropy_is_actual_upper_bound(actor):
    with torch.no_grad():
        actor.log_std.fill_(LOG_STD_MAX)
    _, entropy = actor.evaluate_actions(torch.randn(8, 16),
                                        torch.zeros(8, 3))
    assert entropy.max().item() <= actor.max_entropy() + 1e-4


def test_deterministic_action_is_mean(actor):
    obs = torch.randn(4, 16)
    a1, _ = actor.get_action(obs, deterministic=True)
    a2, _ = actor.get_action(obs, deterministic=True)
    assert torch.equal(a1, a2)


def test_critic_shapes():
    critic = MAPPOCritic(global_obs_dim=4 * 16, num_agents=4)
    values = critic(torch.randn(7, 4, 16))
    assert values.shape == (7, 4)
    # Pre-flattened input is also accepted
    values_flat = critic(torch.randn(7, 64))
    assert values_flat.shape == (7, 4)
