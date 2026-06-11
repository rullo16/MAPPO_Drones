import numpy as np


class RolloutBuffer:
    """
    On-policy buffer for storing trajectories and computing advantages.
    Unlike an off-policy replay buffer, this buffer:
    - Only stores the most recent rollout
    - Computes advantages using Generalized Advantage Estimation (GAE)
    - Gets cleared after each update

    GAE reduces variance in advantage estimates and balances the
    bias-variance tradeoff with the lambda parameter.

    Done-flag convention: dones[t] == 1 means the state reached AFTER the
    action at step t is terminal (or the episode was truncated and the
    caller has already applied the time-limit bootstrap correction to
    rewards[t]). GAE will not bootstrap across a done boundary.
    """

    def __init__(self, num_steps, num_agents, obs_shape, action_dim, gamma=0.99, gae_lambda=0.95):
        """
        num_steps: Number of steps to collect before updating (e.g., 2048)
        num_agents: Number of agents in the environment
        obs_shape: Shape of encoded observations, e.g., (348,)
        action_dim: Dimension of action space
        gamma: Discount factor for future rewards
        gae_lambda: GAE lambda parameter
        """
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = np.zeros((num_steps, num_agents, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((num_steps, num_agents, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.values = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_agents), dtype=np.float32)

        self.advantages = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_agents), dtype=np.float32)

        self.ptr = 0

    def store(self, obs, next_obs, action, reward, done, value, log_prob):
        """
        Store one transition for all agents.
        obs: (num_agents, obs_dim) - encoded observations
        next_obs: (num_agents, obs_dim) - encoded next observations
        action: (num_agents, action_dim) - actions taken
        reward: (num_agents,) - rewards received
        done: (num_agents,) - done flags (see class docstring for convention)
        value: (num_agents,) - value estimates from critic
        log_prob: (num_agents,) - log probability of actions
        """
        if self.ptr >= self.num_steps:
            raise ValueError(
                f"Buffer full! Called store() {self.ptr + 1} times but buffer size is {self.num_steps}"
            )

        self.observations[self.ptr] = obs
        self.next_observations[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr += 1

    def compute_returns_and_advantages(self, last_values):
        """
        Compute advantages using GAE:

        A_t = delta_t + (gamma*lambda) * delta_{t+1} + ...
        where delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

        last_values: (num_agents,) - value estimates for the state following
                     the final stored transition (used to bootstrap)
        """
        if self.ptr != self.num_steps:
            raise ValueError(f"Buffer not full! Only {self.ptr}/{self.num_steps} transitions")

        advantages = np.zeros_like(self.rewards)
        last_gae = 0

        for t in reversed(range(self.num_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            if t == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values

    def get(self):
        """
        Get all stored data and reset the buffer.

        Advantage normalization is intentionally NOT done here — the agent
        normalizes advantages once over the whole batch in train().
        """
        if self.ptr != self.num_steps:
            raise ValueError(f"Buffer not full! Only {self.ptr}/{self.num_steps} transitions")

        self.ptr = 0

        return {
            'observations': self.observations.copy(),
            'next_observations': self.next_observations.copy(),
            'actions': self.actions.copy(),
            'returns': self.returns.copy(),
            'advantages': self.advantages.copy(),
            'log_probs': self.log_probs.copy(),
            'values': self.values.copy()
        }
