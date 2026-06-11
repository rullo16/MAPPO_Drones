"""
MAPPO training entry point (replaces the duplicated training notebooks).

Usage:
    python -m MAPPO.train --curriculum                 # 4-stage curriculum
    python -m MAPPO.train --env-path ./Env/FinalLevel/DroneFlightv1   # single env
    python -m MAPPO.train --curriculum --resume saved_models_mappo/mappo_final.pth

Unity ML-Agents is only imported inside main(), so this module stays
importable (and testable) without a Unity installation.
"""

import argparse
import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from MAPPO.mappo_agent import MAPPOAgent, DEFAULT_CONFIG
from MAPPO.rollout_buffer import RolloutBuffer

DEFAULT_ENV_PATHS = [
    './Env/Level1/DroneFlightv1',
    './Env/Level1.5/DroneFlightv1',
    './Env/Level2/DroneFlightv1',
    './Env/FinalLevel/DroneFlightv1',
]

# Anneal the curiosity reward weight to zero over this many env steps.
CURIOSITY_ANNEAL_STEPS = 1_000_000

# Training-health monitors (see check_training_health)
ENTROPY_STUCK_UPDATES = 30   # abort if entropy pinned at max this long
KL_DEAD_UPDATES = 30         # abort if policy stops moving this long
SUCCESS_WARN_STEPS = 500_000 # warn if success still < 5% after this many steps


class FourStageCurriculum:
    """
    Curriculum over environment builds.

    Gates are ASCENDING (each stage is harder, so later stages demand more),
    and every stage has a max-steps fallback so a single stage can never
    trap the whole run.
    """

    def __init__(self, env_paths, thresholds=(0.6, 0.7, 0.8),
                 min_steps_per_stage=250_000, max_steps_per_stage=750_000):
        assert len(env_paths) == len(thresholds) + 1, \
            "Need one success threshold per non-final stage"
        self.env_paths = env_paths
        self.thresholds = thresholds
        self.min_steps_per_stage = min_steps_per_stage
        self.max_steps_per_stage = max_steps_per_stage
        self.current_stage = 1
        self.stage_start_step = 0

    @property
    def num_stages(self):
        return len(self.env_paths)

    def should_advance(self, total_steps, success_rate):
        if self.current_stage >= self.num_stages:
            return False

        steps_in_stage = total_steps - self.stage_start_step
        threshold = self.thresholds[self.current_stage - 1]

        earned = steps_in_stage >= self.min_steps_per_stage and success_rate >= threshold
        forced = steps_in_stage >= self.max_steps_per_stage

        if earned or forced:
            if forced and not earned:
                print(f"⚠️  Stage {self.current_stage}: max steps reached "
                      f"(success {success_rate:.1%} < {threshold:.0%}) — advancing anyway")
            self.current_stage += 1
            self.stage_start_step = total_steps
            return True
        return False

    def get_current_env_path(self):
        return self.env_paths[self.current_stage - 1]


def get_agent_obs(obs, agent, cam_key=1, vec_keys=(0, 2)):
    """
    Extract one agent's observation from the env dict.
    Returns camera (CHW, float32 in [0,1]) and vector (1D float32).
    """
    data = obs[agent]
    if isinstance(data, dict) and "observation" in data:
        data = data["observation"]

    if isinstance(data, dict) and ("camera_obs" in data and "vector_obs" in data):
        cam = np.asarray(data["camera_obs"])
        vec = np.asarray(data["vector_obs"]).reshape(-1)
    else:
        cam = np.asarray(data[cam_key])
        v0 = np.asarray(data[vec_keys[0]]).reshape(-1)
        v1 = np.asarray(data[vec_keys[1]]).reshape(-1)
        vec = np.concatenate([v0, v1], axis=0)

    if cam.ndim != 3:
        raise AssertionError(f"Camera must be 3D, got {cam.shape}")
    if cam.shape[-1] in (1, 3, 4):  # HWC -> CHW
        cam = np.transpose(cam, (2, 0, 1))

    cam = cam.astype(np.float32, copy=False)
    if cam.max() > 1.5:  # uint8 range; normalize exactly once
        cam = cam / 255.0

    return cam, vec.astype(np.float32, copy=False)


def gather_observations(obs_dict, agents, cam_shape, vec_shape):
    """Stack per-agent observations; zero-fill (and report) missing agents."""
    num_agents = len(agents)
    camera_obs = np.zeros((num_agents, *cam_shape), dtype=np.float32)
    vector_obs = np.zeros((num_agents, *vec_shape), dtype=np.float32)
    missing = []
    for i, agent_id in enumerate(agents):
        if agent_id in obs_dict:
            camera_obs[i], vector_obs[i] = get_agent_obs(obs_dict, agent_id)
        else:
            missing.append(agent_id)
    return camera_obs, vector_obs, missing


class TrainingHealthMonitor:
    """
    Watches for the failure modes that previously wasted a full training run:
    - policy entropy pinned at its theoretical max (policy = pure noise)
    - approx KL ~ 0 (policy not moving at all)
    - success rate flat near zero deep into training
    """

    def __init__(self, max_entropy):
        self.max_entropy = max_entropy
        self.entropy_stuck = 0
        self.kl_dead = 0

    def check(self, stats, total_steps, success_rate):
        """Returns None, or an abort message."""
        if stats['entropy'] > 0.98 * self.max_entropy:
            self.entropy_stuck += 1
        else:
            self.entropy_stuck = 0

        if stats['approx_kl'] < 1e-4:
            self.kl_dead += 1
        else:
            self.kl_dead = 0

        if self.entropy_stuck >= ENTROPY_STUCK_UPDATES:
            return (f"Entropy has been within 2% of its maximum "
                    f"({self.max_entropy:.3f}) for {self.entropy_stuck} consecutive "
                    f"updates — the policy is pure noise. Check entropy_coef and "
                    f"reward scales before burning more compute.")
        if self.kl_dead >= KL_DEAD_UPDATES:
            return (f"approx_kl < 1e-4 for {self.kl_dead} consecutive updates — "
                    f"the policy is not moving. Check learning rate, advantage "
                    f"scales, and that gradients are flowing.")
        if total_steps >= SUCCESS_WARN_STEPS and success_rate < 0.05:
            print(f"⚠️  HEALTH: success rate is {success_rate:.1%} after "
                  f"{total_steps:,} steps. Consider stopping to diagnose.")
        return None


def parse_args():
    p = argparse.ArgumentParser(description="MAPPO training")
    p.add_argument('--curriculum', action='store_true',
                   help='Use the 4-stage curriculum over --env-paths')
    p.add_argument('--env-path', type=str, default=None,
                   help='Single environment build (no curriculum)')
    p.add_argument('--env-paths', nargs='+', default=DEFAULT_ENV_PATHS,
                   help='Environment builds for the curriculum, easiest first')
    p.add_argument('--max-steps', type=int, default=DEFAULT_CONFIG['max_steps'])
    p.add_argument('--resume', type=str, default=None,
                   help='Checkpoint to resume from (restores steps/stage too)')
    p.add_argument('--pretrained-encoder', type=str,
                   default='SavedModels/feature_extractor_contrastive_init.pth')
    p.add_argument('--save-dir', type=str, default='./saved_models_mappo')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-graphics', action='store_true')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--success-reward-threshold', type=float, default=15.0,
                   help='An episode counts as a success if ANY agent receives '
                        'a single-step reward above this (proxy for reaching '
                        'the target; keep in sync with the Unity reward code)')
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--save-every', type=int, default=10)
    # Hyperparameter overrides
    p.add_argument('--learning-rate', type=float, default=DEFAULT_CONFIG['learning_rate'])
    p.add_argument('--entropy-coef', type=float, default=DEFAULT_CONFIG['entropy_coef'])
    p.add_argument('--curiosity-coef', type=float, default=DEFAULT_CONFIG['curiosity_coef'])
    p.add_argument('--rollout-length', type=int, default=DEFAULT_CONFIG['rollout_length'])
    return p.parse_args()


def main():
    args = parse_args()

    # Imported here so the module is testable without Unity installed
    from mlagents_envs.environment import UnityEnvironment as UE
    from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv as UPZBE

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = dict(DEFAULT_CONFIG)
    config.update({
        'learning_rate': args.learning_rate,
        'entropy_coef': args.entropy_coef,
        'curiosity_coef': args.curiosity_coef,
        'rollout_length': args.rollout_length,
        'max_steps': args.max_steps,
    })

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Environment ----------------
    if args.curriculum:
        curriculum = FourStageCurriculum(args.env_paths)
        env_path = curriculum.get_current_env_path()
    else:
        curriculum = None
        env_path = args.env_path or args.env_paths[-1]

    def open_env(path):
        unity_env = UE(file_name=path, seed=args.seed, no_graphics=args.no_graphics)
        wrapped = UPZBE(unity_env)
        first_obs = wrapped.reset()
        return wrapped, first_obs

    print(f"Loading Unity environment: {env_path}")
    env, obs = open_env(env_path)
    agents = sorted(env.agents)
    num_agents = len(agents)

    cam_shape = env.observation_space(agents[0])[1].shape
    vec_dim = (env.observation_space(agents[0])[0].shape[0]
               + env.observation_space(agents[0])[2].shape[0])
    vec_shape = (vec_dim,)
    action_dim = env.action_space(agents[0]).shape[0]

    print(f"  Agents: {num_agents} | Camera: {cam_shape} | "
          f"Vector dim: {vec_dim} | Action dim: {action_dim}")

    # ---------------- Agent ----------------
    agent = MAPPOAgent(
        camera_shape=cam_shape,
        vector_shape=vec_shape,
        action_dim=action_dim,
        num_agents=num_agents,
        config=config,
    )

    if os.path.exists(args.pretrained_encoder):
        state_dict = torch.load(args.pretrained_encoder, map_location=device)
        agent.load_pretrained_encoder(state_dict)
        print(f"✓ Pretrained vision encoder loaded from {args.pretrained_encoder}")
    else:
        print(f"⚠️  No pretrained encoder at {args.pretrained_encoder}.\n"
              f"   The encoder is FROZEN during RL, so without pretraining the "
              f"visual features are a fixed random projection. Run "
              f"PretrainFeatureExtraction.ipynb first for usable features.")

    # ---------------- Resume ----------------
    total_steps = 0
    num_updates = 0
    best_reward = -float('inf')
    if args.resume:
        extra = agent.load(args.resume)
        total_steps = extra.get('total_steps', 0)
        num_updates = extra.get('num_updates', agent.update_count)
        best_reward = extra.get('best_reward', -float('inf'))
        if curriculum is not None and 'curriculum_stage' in extra:
            curriculum.current_stage = extra['curriculum_stage']
            curriculum.stage_start_step = extra.get('curriculum_stage_start_step', total_steps)
            stage_path = curriculum.get_current_env_path()
            if stage_path != env_path:
                env.close()
                env_path = stage_path
                env, obs = open_env(env_path)
                agents = sorted(env.agents)
        print(f"✓ Resumed at step {total_steps:,} (update {num_updates}, "
              f"best reward {best_reward:.2f})")

    buffer = RolloutBuffer(
        num_steps=config['rollout_length'],
        num_agents=num_agents,
        obs_shape=(agent.encoded_obs_dim,),
        action_dim=action_dim,
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
    )

    # ---------------- W&B ----------------
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "MAPPO_Drones"),
            entity=os.getenv("WANDB_ENTITY"),  # None -> default account
            config={**config, 'curriculum': bool(curriculum), 'seed': args.seed},
        )

    monitor = TrainingHealthMonitor(max_entropy=agent.actor.max_entropy())

    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_successes = deque(maxlen=100)
    current_episode_reward = np.zeros(num_agents)
    current_episode_length = 0
    success_rate = 0.0
    curiosity_coef_initial = config['curiosity_coef']

    def checkpoint_extra():
        extra = {
            'total_steps': total_steps,
            'num_updates': num_updates,
            'best_reward': best_reward,
        }
        if curriculum is not None:
            extra['curriculum_stage'] = curriculum.current_stage
            extra['curriculum_stage_start_step'] = curriculum.stage_start_step
        return extra

    print(f"Starting training: {config['max_steps']:,} steps, "
          f"rollout {config['rollout_length']}, "
          f"{config['max_steps'] // config['rollout_length']:,} updates expected")

    # Encode the initial observation once; each step reuses the previous
    # step's next-encoding, so the encoder runs once per new observation.
    camera_obs, vector_obs, _ = gather_observations(obs, agents, cam_shape, vec_shape)
    encoded_obs = agent.encode_observations(camera_obs, vector_obs)

    try:
        while total_steps < config['max_steps']:

            # ---- Curriculum stage transition ----
            if curriculum is not None and curriculum.should_advance(total_steps, success_rate):
                stage = curriculum.current_stage
                stage_ckpt = save_dir / f"mappo_stage{stage - 1}_checkpoint.pth"
                agent.save(stage_ckpt, extra=checkpoint_extra())
                env.close()
                env_path = curriculum.get_current_env_path()
                print(f"\n=== Curriculum: advancing to stage {stage}: {env_path} ===\n")
                env, obs = open_env(env_path)
                agents = sorted(env.agents)
                episode_successes.clear()  # stale stats must not chain advances
                current_episode_reward = np.zeros(num_agents)
                current_episode_length = 0
                camera_obs, vector_obs, _ = gather_observations(obs, agents, cam_shape, vec_shape)
                encoded_obs = agent.encode_observations(camera_obs, vector_obs)

            # Anneal curiosity weight to 0 over the first CURIOSITY_ANNEAL_STEPS
            agent.curiosity_coef = curiosity_coef_initial * max(
                0.0, 1.0 - total_steps / CURIOSITY_ANNEAL_STEPS)

            # ================= COLLECTION =================
            for _ in range(config['rollout_length']):
                if not obs:
                    obs = env.reset()
                    agents = sorted(env.agents)
                    camera_obs, vector_obs, _ = gather_observations(obs, agents, cam_shape, vec_shape)
                    encoded_obs = agent.encode_observations(camera_obs, vector_obs)
                    current_episode_reward = np.zeros(num_agents)
                    current_episode_length = 0

                actions, log_probs, values = agent.get_action(encoded_obs)

                action_dict = {aid: act for aid, act in zip(agents, actions)}
                next_obs, reward_dict, done_dict, _ = env.step(action_dict)

                rewards = np.array([reward_dict.get(a, 0.0) for a in agents])
                dones = np.array([done_dict.get(a, False) for a in agents], dtype=np.float32)
                current_episode_reward += rewards

                train_rewards = np.clip(rewards, -config['reward_clip'], config['reward_clip'])

                camera_next, vector_next, missing = gather_observations(
                    next_obs, agents, cam_shape, vec_shape)
                if missing:
                    print(f"⚠️  Missing observations zero-filled for agents: {missing}")
                encoded_next = agent.encode_observations(camera_next, vector_next)

                intrinsic = agent.compute_intrinsic_rewards(encoded_obs, encoded_next, actions)
                total_rewards = train_rewards + intrinsic * agent.curiosity_coef

                episode_over = bool(dones.any()) or (len(done_dict) > 0 and all(done_dict.values()))

                if episode_over:
                    # The whole env resets, so every agent's trajectory ends
                    # here. Agents that did NOT terminate are truncated: apply
                    # the time-limit correction (bootstrap with V(s')) and
                    # mark them done so GAE doesn't leak across the reset.
                    next_values = agent.get_values(encoded_next)
                    truncated = dones < 0.5
                    total_rewards = total_rewards + truncated * config['gamma'] * next_values
                    store_dones = np.ones_like(dones)
                else:
                    store_dones = dones

                buffer.store(
                    obs=encoded_obs.cpu().numpy(),
                    next_obs=encoded_next.cpu().numpy(),
                    action=actions,
                    reward=total_rewards,
                    done=store_dones,
                    value=values,
                    log_prob=log_probs,
                )

                current_episode_length += 1
                total_steps += 1

                if episode_over:
                    episode_rewards.append(current_episode_reward.mean())
                    episode_lengths.append(current_episode_length)
                    # Success proxy: any agent got the big terminal reward.
                    episode_successes.append(
                        float(np.any(rewards > args.success_reward_threshold)))

                    obs = env.reset()
                    agents = sorted(env.agents)
                    camera_obs, vector_obs, _ = gather_observations(obs, agents, cam_shape, vec_shape)
                    encoded_obs = agent.encode_observations(camera_obs, vector_obs)
                    current_episode_reward = np.zeros(num_agents)
                    current_episode_length = 0
                else:
                    obs = next_obs
                    encoded_obs = encoded_next

            # ================= UPDATE =================
            last_values = agent.get_values(encoded_obs)
            buffer.compute_returns_and_advantages(last_values)
            train_stats = agent.train(buffer)
            num_updates += 1

            mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
            success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0

            # ---- Health monitors ----
            abort_msg = monitor.check(train_stats, total_steps, success_rate)
            if abort_msg:
                print(f"\n🛑 TRAINING ABORTED BY HEALTH MONITOR:\n   {abort_msg}\n")
                break

            # ---- Logging ----
            if num_updates % args.log_every == 0:
                print(f"\nStep {total_steps:,} | Update {num_updates}"
                      + (f" | Stage {curriculum.current_stage}" if curriculum else ""))
                print(f"  Reward (100ep):  {mean_reward:8.2f}")
                print(f"  Success rate:    {success_rate:8.1%}")
                print(f"  Episode length:  {mean_length:8.1f}")
                print(f"  Policy loss:     {train_stats['policy_loss']:8.4f}")
                print(f"  Value loss:      {train_stats['value_loss']:8.4f}")
                print(f"  Entropy:         {train_stats['entropy']:8.4f}"
                      f"  (max {monitor.max_entropy:.2f})")
                print(f"  KL divergence:   {train_stats['approx_kl']:8.4f}")
                print(f"  Clip fraction:   {train_stats['clip_fraction']:8.1%}")
                print(f"  Explained var:   {train_stats['explained_variance']:8.1%}")

                if use_wandb:
                    import wandb
                    log = {f"train/{k}": v for k, v in train_stats.items()}
                    log.update({
                        'train/reward_mean': mean_reward,
                        'train/success_rate': success_rate,
                        'train/episode_length': mean_length,
                        'train/total_steps': total_steps,
                    })
                    if curriculum is not None:
                        log['train/curriculum_stage'] = curriculum.current_stage
                    wandb.log(log, step=total_steps)

            # ---- Checkpoints ----
            if num_updates % args.save_every == 0:
                agent.save(save_dir / f"mappo_checkpoint_{total_steps:08d}.pth",
                           extra=checkpoint_extra())

            if episode_rewards and mean_reward > best_reward:
                best_reward = mean_reward
                agent.save(save_dir / "mappo_best.pth", extra=checkpoint_extra())

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at {total_steps:,} steps ({num_updates} updates)")

    agent.save(save_dir / "mappo_final.pth", extra=checkpoint_extra())
    env.close()
    if use_wandb:
        import wandb
        wandb.finish()

    print(f"\nDone. Total steps: {total_steps:,} | Updates: {num_updates}")
    if episode_rewards:
        print(f"Final reward (100ep): {np.mean(episode_rewards):.2f}")
    if episode_successes:
        print(f"Final success rate:   {np.mean(episode_successes):.1%}")


if __name__ == '__main__':
    main()
