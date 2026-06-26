"""
MAPPO agent (CTDE):
1. RolloutBuffer  - stores trajectories, computes advantages with GAE
2. MAPPOActor     - decentralized policy (parameter-shared across agents)
3. MAPPOCritic    - centralized value function
4. CuriosityModule- ICM intrinsic reward (operates on encoded features)

Design notes:
- The vision encoder is FROZEN during RL. The rollout buffer stores encoded
  features (not raw pixels), so gradients cannot reach the encoder anyway;
  freezing it makes that explicit. Pretrain it with
  PretrainFeatureExtraction.ipynb and load it before training.
- Encoded observations are [frozen visual features | raw vector obs]. The
  actor/critic learn their own processing of the vector part end-to-end
  (a separate pre-encoding vector MLP would be untrainable for the same
  reason the encoder is frozen).
- Config keys are validated strictly: unknown keys raise, so a typo can
  never silently fall back to a default.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .models import MAPPOActor, MAPPOCritic
from .rollout_buffer import RolloutBuffer
from .vision_encoders import EfficientVisionEncoder
from .curiosity import CuriosityModule, RunningMeanStd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The only accepted hyperparameters, with their defaults.
DEFAULT_CONFIG = {
    'learning_rate': 3e-4,
    'clip_param': 0.2,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.005,          # initial entropy bonus
    'entropy_coef_final': 0.001,    # linearly decayed to this over max_steps
    'max_grad_norm': 0.5,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'rollout_length': 2048,
    'num_minibatches': 8,
    'ppo_epochs': 4,
    'max_steps': 3_000_000,
    'curiosity_coef': 0.01,
    'reward_clip': 10.0,            # extrinsic reward clip (applied by caller)
    'intrinsic_reward_clip': 5.0,   # clip for normalized intrinsic rewards
    'target_kl': 0.02,              # early-stop updates when KL > 1.5x this
    'vision_output_dim': 256,
}


def validate_config(config):
    """Merge user config over defaults; raise on unknown keys."""
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(
            f"Unknown config keys: {sorted(unknown)}. "
            f"Accepted keys: {sorted(DEFAULT_CONFIG)}"
        )
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    return merged


class MAPPOAgent:
    """
    MAPPO agent.

    Fuses frozen visual features with raw vector observations, implements
    the PPO clipped objective with a centralized critic, and manages the
    ICM curiosity module.
    """

    def __init__(self, camera_shape, vector_shape, action_dim, num_agents, config):
        """
        camera_shape: Shape of camera observations (C, H, W)
        vector_shape: Shape of vector observations, e.g. (92,)
        action_dim: Dimension of action space
        num_agents: Number of agents
        config: Dict of hyperparameters (validated against DEFAULT_CONFIG)
        """
        config = validate_config(config)
        self.config = config

        self.device = device
        self.camera_shape = camera_shape
        self.vector_shape = vector_shape
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.lr = config['learning_rate']
        self.clip_param = config['clip_param']
        self.value_loss_coef = config['value_loss_coef']
        self.entropy_coef_initial = config['entropy_coef']
        self.entropy_coef_final = config['entropy_coef_final']
        self.entropy_coef = self.entropy_coef_initial
        self.max_grad_norm = config['max_grad_norm']
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.num_steps = config['rollout_length']
        self.num_minibatches = config['num_minibatches']
        self.ppo_epochs = config['ppo_epochs']
        self.curiosity_coef = config['curiosity_coef']
        self.intrinsic_reward_clip = config['intrinsic_reward_clip']
        self.target_kl = config['target_kl']

        self.total_updates = max(1, config['max_steps'] // self.num_steps)
        self.update_count = 0

        # --- Vision encoder: frozen, eval-mode (see module docstring) ---
        self.vision_encoder = EfficientVisionEncoder(
            input_shape=camera_shape,
            output_dim=config['vision_output_dim'],
        ).to(device)
        self.vision_encoder.eval()
        self.vision_encoder.requires_grad_(False)

        # Encoded obs = frozen visual features + raw vector obs
        self.encoded_obs_dim = self.vision_encoder.output_dim + vector_shape[0]

        self.actor = MAPPOActor(self.encoded_obs_dim, action_dim).to(device)
        self.critic = MAPPOCritic(
            self.encoded_obs_dim * num_agents, num_agents=num_agents
        ).to(device)

        self.curiosity_module = CuriosityModule(
            obs_dim=self.encoded_obs_dim,
            action_dim=action_dim,
            hidden_dim=512
        ).to(device)

        self.intrinsic_reward_normalizer = RunningMeanStd()

        # Single optimizer over the trainable networks only.
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr},
            {'params': self.curiosity_module.parameters(), 'lr': self.lr},
        ], eps=1e-5)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_updates,
            eta_min=self.lr * 0.1
        )

    def load_pretrained_encoder(self, state_dict):
        """Load pretrained vision encoder weights. Strict by design: a
        checkpoint from a different architecture must fail loudly instead of
        silently matching zero keys (which is what strict=False did)."""
        self.vision_encoder.load_state_dict(state_dict, strict=True)
        self.vision_encoder.eval()

    def encode_observations(self, camera_obs, vector_obs):
        """
        Encode raw observations: frozen visual features + raw vector obs.

        camera_obs: (num_agents, C, H, W), float in [0, 1] or uint8
        vector_obs: (num_agents, vector_dim)

        returns encoded_obs: (num_agents, encoded_obs_dim) tensor on device
        """
        if isinstance(camera_obs, np.ndarray):
            camera_obs = torch.from_numpy(camera_obs).to(self.device)
        if camera_obs.dtype == torch.uint8:
            camera_obs = camera_obs.float() / 255.0
        else:
            camera_obs = camera_obs.float()

        if isinstance(vector_obs, np.ndarray):
            vector_obs = torch.from_numpy(vector_obs).float().to(self.device)

        with torch.no_grad():
            visual_features = self.vision_encoder(camera_obs)

        return torch.cat([visual_features, vector_obs], dim=-1)

    def compute_intrinsic_rewards(self, obs, next_obs, actions):
        """
        Compute normalized, clipped intrinsic rewards from the ICM forward
        model's prediction error.

        obs, next_obs: (num_agents, encoded_obs_dim) tensors
        actions: (num_agents, action_dim) tensor or ndarray

        returns intrinsic_rewards: (num_agents,) ndarray
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float().to(self.device)

        with torch.no_grad():
            intrinsic_rewards, _ = self.curiosity_module(obs, next_obs, actions)
            intrinsic_rewards = intrinsic_rewards.cpu().numpy()

        self.intrinsic_reward_normalizer.update(intrinsic_rewards)
        intrinsic_rewards = (
            (intrinsic_rewards - self.intrinsic_reward_normalizer.mean)
            / (self.intrinsic_reward_normalizer.std + 1e-8)
        )
        return np.clip(intrinsic_rewards, -self.intrinsic_reward_clip, self.intrinsic_reward_clip)

    @torch.no_grad()
    def get_action(self, encoded_obs, deterministic=False):
        """
        Get actions, log probs and values for already-encoded observations.
        Use encode_observations() once per step and pass the result here —
        this avoids redundant encoder forward passes.

        encoded_obs: (num_agents, encoded_obs_dim) tensor
        """
        actions, log_probs = self.actor.get_action(encoded_obs, deterministic)
        values = self.critic(encoded_obs.unsqueeze(0)).squeeze(0)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy()
        )

    @torch.no_grad()
    def get_values(self, encoded_obs):
        """Centralized value estimates for one timestep of encoded obs."""
        return self.critic(encoded_obs.unsqueeze(0)).squeeze(0).cpu().numpy()

    def act(self, camera_obs, vector_obs, deterministic=False):
        """
        Convenience wrapper: encode once and return everything the rollout
        loop needs.

        returns (actions, log_probs, values, encoded_obs)
        """
        encoded_obs = self.encode_observations(camera_obs, vector_obs)
        actions, log_probs, values = self.get_action(encoded_obs, deterministic)
        return actions, log_probs, values, encoded_obs

    def _current_entropy_coef(self):
        frac = min(1.0, self.update_count / self.total_updates)
        return self.entropy_coef_initial + frac * (self.entropy_coef_final - self.entropy_coef_initial)

    def train(self, rollout_buffer: RolloutBuffer):
        """
        Update policy using PPO on one full rollout.

        returns stats: dict of averaged training statistics
        """
        data = rollout_buffer.get()

        obs = torch.tensor(data['observations'], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(data['next_observations'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(data['actions'], dtype=torch.float32, device=self.device)
        returns = torch.tensor(data['returns'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(data['advantages'], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(data['log_probs'], dtype=torch.float32, device=self.device)

        assert obs.dim() == 3, f"Expected (num_steps, num_agents, obs_dim), got {tuple(obs.shape)}"
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            raise ValueError("NaN/Inf detected in observations")
        if torch.isnan(advantages).any():
            raise ValueError("NaN detected in advantages")

        # Normalize advantages once over the whole batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_steps = obs.shape[0]
        timesteps_per_minibatch = max(1, num_steps // self.num_minibatches)

        self.entropy_coef = self._current_entropy_coef()

        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': [],
            'explained_variance': []
        }

        early_stop = False
        for epoch in range(self.ppo_epochs):
            if early_stop:
                break
            # Shuffle timesteps; agents stay grouped per timestep so the
            # centralized critic always sees a coherent global state.
            indices = torch.randperm(num_steps, device=self.device)

            for start in range(0, num_steps, timesteps_per_minibatch):
                end = min(start + timesteps_per_minibatch, num_steps)
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]                      # (mb, A, D)
                mb_next_obs = next_obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]

                mb_obs_flat = mb_obs.reshape(-1, self.encoded_obs_dim)
                mb_next_obs_flat = mb_next_obs.reshape(-1, self.encoded_obs_dim)
                mb_actions_flat = mb_actions.reshape(-1, self.action_dim)
                mb_advantages_flat = mb_advantages.reshape(-1)
                mb_old_log_probs_flat = mb_old_log_probs.reshape(-1)
                mb_returns_flat = mb_returns.reshape(-1)

                log_probs, entropy = self.actor.evaluate_actions(mb_obs_flat, mb_actions_flat)

                log_ratio = log_probs - mb_old_log_probs_flat
                ratio = torch.exp(log_ratio)

                # PPO clipped objective
                surr1 = ratio * mb_advantages_flat
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * mb_advantages_flat
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(mb_obs).reshape(-1)
                value_loss = F.mse_loss(values_pred, mb_returns_flat)

                entropy_loss = -entropy.mean()

                # Curiosity loss on a subsample of flattened transitions.
                # obs/next_obs/actions are all indexed with the SAME flat
                # indices so the pairs stay temporally aligned.
                curiosity_batch_size = min(512, len(mb_obs_flat))
                curiosity_indices = torch.randperm(len(mb_obs_flat), device=self.device)[:curiosity_batch_size]
                _, curiosity_loss = self.curiosity_module(
                    mb_obs_flat[curiosity_indices],
                    mb_next_obs_flat[curiosity_indices],
                    mb_actions_flat[curiosity_indices]
                )

                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                    + self.curiosity_coef * curiosity_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters())
                    + list(self.curiosity_module.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    # Low-bias KL estimator: E[(r - 1) - log r]
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()

                    y_pred = values_pred.cpu().numpy()
                    y_true = mb_returns_flat.cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = float(np.clip(1 - np.var(y_true - y_pred) / (var_y + 1e-8), -1.0, 1.0))

                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(entropy.mean().item())
                stats['approx_kl'].append(approx_kl)
                stats['clip_fraction'].append(clip_fraction)
                stats['explained_variance'].append(explained_var)

                # Per-minibatch KL early stop
                if approx_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break

        self.scheduler.step()
        self.update_count += 1

        result = {k: float(np.mean(v)) for k, v in stats.items()}
        result['entropy_coef'] = self.entropy_coef
        result['curiosity_coef'] = self.curiosity_coef
        result['learning_rate'] = self.optimizer.param_groups[0]['lr']
        result['early_stopped'] = float(early_stop)
        return result

    def save(self, path, extra=None):
        """Save full training state. `extra` is an arbitrary dict for
        loop-level state (total_steps, curriculum stage, best reward, ...).

        Atomic: writes to a temp file and renames it over the target, so a
        failed write (disk full, OneDrive/antivirus lock) can never corrupt
        an existing checkpoint."""
        path = str(path)
        tmp_path = path + '.tmp'
        try:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'vision_encoder': self.vision_encoder.state_dict(),
                'curiosity_module': self.curiosity_module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'update_count': self.update_count,
                'intrinsic_normalizer': {
                    'mean': self.intrinsic_reward_normalizer.mean,
                    'var': self.intrinsic_reward_normalizer.var,
                    'count': self.intrinsic_reward_normalizer.count,
                },
                'config': self.config,
                'extra': extra or {},
            }, tmp_path)
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        print(f"Model saved to {path}")

    def load(self, path):
        """Restore full training state. Returns the `extra` dict that was
        passed to save()."""
        # weights_only=False: checkpoints are produced by save() above and
        # contain plain Python/numpy training state alongside the tensors.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.vision_encoder.load_state_dict(checkpoint['vision_encoder'])
        self.vision_encoder.eval()
        self.curiosity_module.load_state_dict(checkpoint['curiosity_module'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.update_count = checkpoint.get('update_count', 0)
        norm = checkpoint.get('intrinsic_normalizer')
        if norm is not None:
            self.intrinsic_reward_normalizer.mean = norm['mean']
            self.intrinsic_reward_normalizer.var = norm['var']
            self.intrinsic_reward_normalizer.count = norm['count']
        print(f"Model loaded from {path}")
        return checkpoint.get('extra', {})
