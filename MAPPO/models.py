import numpy as np
import torch
import torch.nn as nn

# Bounds for the policy's log standard deviation. A state-independent,
# clamped log_std keeps the entropy bounded and removes one common path
# to entropy blow-up (std saturating at its ceiling).
LOG_STD_MIN = float(np.log(0.05))
LOG_STD_MAX = float(np.log(0.8))


class MAPPOActor(nn.Module):
    """
    Decentralized policy net.

    "Decentralized" because each agent makes decisions based on its own
    observation only; there is no communication between agents during
    execution, which is critical for real-world deployment.

    All agents share the same network parameters (parameter sharing), which
    yields faster learning, better generalization and smaller models.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        """
        obs_dim: Dimension of encoded observations (vision + vector)
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size
        """
        super().__init__()
        self.action_dim = action_dim

        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)

        # State-independent log std (standard PPO choice). Initialised at
        # log(0.3) and clamped to [LOG_STD_MIN, LOG_STD_MAX] in forward().
        self.log_std = nn.Parameter(torch.full((action_dim,), float(np.log(0.3))))

        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal init; small gain for the policy head prevents large
        initial policy changes.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)

    def max_entropy(self):
        """Upper bound on the (pre-tanh) policy entropy, used by training
        health monitors to detect a policy stuck at maximum randomness."""
        return self.action_dim * (0.5 * np.log(2 * np.pi * np.e) + LOG_STD_MAX)

    def forward(self, obs):
        """
        Compute the action distribution.
        obs: (batch_size, obs_dim)

        returns a Normal distribution over pre-tanh actions
        """
        features = self.feature_net(obs)
        action_mean = self.mean_layer(features)
        action_std = torch.exp(torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
        action_std = action_std.expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy.

        obs: Observations
        deterministic: If True, return mean (for evaluation)

        returns
        action: Sampled action, tanh-squashed to [-1, 1]
        log_prob: Log probability of the action (with squash correction)
        """
        dist = self.forward(obs)

        if deterministic:
            action_raw = dist.mean
        else:
            action_raw = dist.sample()

        action = torch.tanh(action_raw)

        log_prob = dist.log_prob(action_raw)
        # tanh squash correction
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob

    def evaluate_actions(self, obs, actions):
        """
        Evaluate log prob and entropy of given (squashed) actions.
        Used during training to compute the policy loss.

        returns
        log_probs: Log probability of actions under the current policy
        entropy: Entropy of the action distribution
        """
        dist = self.forward(obs)

        actions_unsquashed = torch.atanh(actions.clamp(-0.9999, 0.9999))

        log_prob = dist.log_prob(actions_unsquashed)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class MAPPOCritic(nn.Module):
    """
    Centralized value network.

    "Centralized" because it sees all agents' observations during training,
    which yields better value estimates and helps with credit assignment.
    Only used during training (the CTDE pattern).
    """

    def __init__(self, global_obs_dim, hidden_dim=512, num_agents=4):
        """
        global_obs_dim: Total observation dimension (num_agents * obs_dim)
        hidden_dim: Hidden layer size
        num_agents: Number of agents
        """
        super().__init__()
        self.num_agents = num_agents

        self.value_net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_agents)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, global_obs):
        """
        Compute value estimates.
        global_obs: (batch_size, num_agents, obs_dim) - all agents' observations

        returns values: (batch_size, num_agents) - per-agent value estimates
        """
        if global_obs.dim() == 3:
            batch_size = global_obs.shape[0]
            global_obs_flat = global_obs.reshape(batch_size, -1)
        else:
            global_obs_flat = global_obs

        return self.value_net(global_obs_flat)
