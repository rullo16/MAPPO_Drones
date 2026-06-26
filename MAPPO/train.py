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


def prune_checkpoints(save_dir, keep):
    """Delete all but the newest `keep` periodic checkpoints to bound disk
    usage (best/final/stage checkpoints are never pruned)."""
    if keep <= 0:
        return
    checkpoints = sorted(Path(save_dir).glob("mappo_checkpoint_*.pth"))
    for old in checkpoints[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


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
    p.add_argument('--keep-checkpoints', type=int, default=5,
                   help='How many periodic checkpoints to retain (0 = all)')
    p.add_argument('--reward-clip', type=float, default=DEFAULT_CONFIG['reward_clip'],
                   help='Clip extrinsic rewards to +/- this for TRAINING. Keep it '
                        'above the env\'s terminal success bonus, or the incentive '
                        'to finish gets squashed relative to shaping reward. '
                        '(Success DETECTION always uses raw rewards.)')
    # Hyperparameter overrides
    p.add_argument('--learning-rate', type=float, default=DEFAULT_CONFIG['learning_rate'])
    p.add_argument('--entropy-coef', type=float, default=DEFAULT_CONFIG['entropy_coef'])
    p.add_argument('--curiosity-coef', type=float, default=DEFAULT_CONFIG['curiosity_coef'])
    p.add_argument('--rollout-length', type=int, default=DEFAULT_CONFIG['rollout_length'])
    return p.parse_args()


def main():
    args = parse_args()

    # Imported here so the module is testable without Unity installed
    from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
    from MAPPO.unity_env import UnityMultiAgentEnv

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
        'reward_clip': args.reward_clip,
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
        # Consume the env's StatsRecorder side channel; without it,
        # mlagents_envs warns "Unknown side channel data received" forever.
        # UnityMultiAgentEnv drives the low-level API so each drone's
        # episode ends and respawns INDEPENDENTLY — no global resets.
        stats_channel = StatsSideChannel()
        wrapped = UnityMultiAgentEnv(
            file_name=path, seed=args.seed, no_graphics=args.no_graphics,
            side_channels=[stats_channel])
        return wrapped, stats_channel

    print(f"Loading Unity environment: {env_path}")
    env, env_stats = open_env(env_path)
    num_agents = env.num_agents

    cam_shape = env.camera_shape
    vec_dim = env.vector_dim
    vec_shape = (vec_dim,)
    action_dim = env.action_dim

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
    # "Best" is selected by SUCCESS RATE (reward as tiebreak), not by training
    # reward: training reward mixes exploration + curiosity/shaping and shifts
    # scale between curriculum stages, so reward-best can freeze on a weak
    # early-stage checkpoint while the policy keeps improving (observed: a
    # reward-best checkpoint scored 0% deterministic while the final scored
    # ~100%). Reset on curriculum advance so best reflects the hardest stage.
    best_success = -float('inf')
    best_reward = -float('inf')
    if args.resume:
        extra = agent.load(args.resume)
        total_steps = extra.get('total_steps', 0)
        num_updates = extra.get('num_updates', agent.update_count)
        best_reward = extra.get('best_reward', -float('inf'))
        best_success = extra.get('best_success', -float('inf'))
        if curriculum is not None and 'curriculum_stage' in extra:
            curriculum.current_stage = extra['curriculum_stage']
            curriculum.stage_start_step = extra.get('curriculum_stage_start_step', total_steps)
            stage_path = curriculum.get_current_env_path()
            if stage_path != env_path:
                env.close()
                env_path = stage_path
                env, env_stats = open_env(env_path)
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
    # Per-agent accumulators: agents reset INDEPENDENTLY inside Unity
    # (EndEpisode respawns one drone while the others keep flying), so
    # episodes are tracked per agent and the env is never globally reset.
    agent_episode_reward = np.zeros(num_agents)
    agent_episode_length = np.zeros(num_agents, dtype=np.int64)
    success_rate = 0.0
    curiosity_coef_initial = config['curiosity_coef']

    def checkpoint_extra():
        extra = {
            'total_steps': total_steps,
            'num_updates': num_updates,
            'best_reward': best_reward,
            'best_success': best_success,
        }
        if curriculum is not None:
            extra['curriculum_stage'] = curriculum.current_stage
            extra['curriculum_stage_start_step'] = curriculum.stage_start_step
        return extra

    def try_save(path):
        """A failed checkpoint write (disk full, file lock) must not kill a
        multi-hour training run; saves are atomic so nothing gets corrupted."""
        try:
            agent.save(path, extra=checkpoint_extra())
            return True
        except (RuntimeError, OSError) as e:
            print(f"⚠️  Checkpoint save failed for {path}: {e}")
            print(f"   Check free disk space (and OneDrive/antivirus locks on "
                  f"the save dir). Training continues.")
            return False

    print(f"Starting training: {config['max_steps']:,} steps, "
          f"rollout {config['rollout_length']}, "
          f"{config['max_steps'] // config['rollout_length']:,} updates expected")

    # Encode the initial observation once; each step reuses the previous
    # step's next-encoding, so the encoder runs once per new observation.
    camera_obs, vector_obs = env.reset()
    encoded_obs = agent.encode_observations(camera_obs, vector_obs)

    try:
        while total_steps < config['max_steps']:

            # ---- Curriculum stage transition ----
            if curriculum is not None and curriculum.should_advance(total_steps, success_rate):
                stage = curriculum.current_stage
                stage_ckpt = save_dir / f"mappo_stage{stage - 1}_checkpoint.pth"
                try_save(stage_ckpt)
                env.close()
                env_path = curriculum.get_current_env_path()
                print(f"\n=== Curriculum: advancing to stage {stage}: {env_path} ===\n")
                env, env_stats = open_env(env_path)
                episode_successes.clear()  # stale stats must not chain advances
                # Reset best tracking so mappo_best.pth reflects the new
                # (harder) stage, not a high score from an easier one.
                best_success = -float('inf')
                best_reward = -float('inf')
                agent_episode_reward[:] = 0.0
                agent_episode_length[:] = 0
                camera_obs, vector_obs = env.reset()
                encoded_obs = agent.encode_observations(camera_obs, vector_obs)

            # Anneal curiosity weight to 0 over the first CURIOSITY_ANNEAL_STEPS
            agent.curiosity_coef = curiosity_coef_initial * max(
                0.0, 1.0 - total_steps / CURIOSITY_ANNEAL_STEPS)

            # ================= COLLECTION =================
            missing_obs_steps = 0
            # Raw (pre-clip) step-reward extremes this rollout: directly
            # answers "does any step reward ever exceed the success
            # threshold?" when diagnosing a 0% success rate.
            raw_reward_max = -float('inf')
            raw_reward_min = float('inf')
            for _ in range(config['rollout_length']):
                actions, log_probs, values = agent.get_action(encoded_obs)

                camera_next, vector_next, rewards, dones, interrupted, stale = env.step(actions)

                agent_episode_reward += rewards
                agent_episode_length += 1
                raw_reward_max = max(raw_reward_max, float(rewards.max()))
                raw_reward_min = min(raw_reward_min, float(rewards.min()))

                train_rewards = np.clip(rewards, -config['reward_clip'], config['reward_clip'])

                if stale.any():
                    # A slot produced no fresh decision within the adapter's
                    # micro-step budget; its obs is the previous one. Should
                    # be rare with synchronized DecisionRequesters.
                    missing_obs_steps += 1
                    if missing_obs_steps <= 3:
                        print(f"⚠️  Stale observations for agent slots: "
                              f"{np.flatnonzero(stale > 0.5).tolist()}")

                encoded_next = agent.encode_observations(camera_next, vector_next)

                # Per-agent dones are TRUE terminals: each drone respawns on
                # its own inside Unity while the others keep flying — there
                # is no global reset. GAE masks bootstraps per agent via the
                # stored done flags. Intrinsic reward is masked at terminals
                # (the "next" obs is the respawn state — its novelty is not
                # signal) and at stale slots.
                intrinsic = agent.compute_intrinsic_rewards(encoded_obs, encoded_next, actions)
                total_rewards = (train_rewards
                                 + intrinsic * agent.curiosity_coef * (1.0 - dones) * (1.0 - stale))

                if interrupted.any():
                    # ML-Agents flagged a built-in MaxStep interruption: a
                    # truncation, not a true terminal — bootstrap with V(s').
                    next_values = agent.get_values(encoded_next)
                    total_rewards = total_rewards + interrupted * config['gamma'] * next_values

                buffer.store(
                    obs=encoded_obs.cpu().numpy(),
                    next_obs=encoded_next.cpu().numpy(),
                    action=actions,
                    reward=total_rewards,
                    done=dones,
                    value=values,
                    log_prob=log_probs,
                )

                total_steps += 1

                for i in np.flatnonzero(dones > 0.5):
                    episode_rewards.append(float(agent_episode_reward[i]))
                    episode_lengths.append(int(agent_episode_length[i]))
                    # Success proxy: the terminal step carried the goal bonus
                    # (env pays >= 20 on success).
                    episode_successes.append(
                        float(rewards[i] > args.success_reward_threshold))
                    agent_episode_reward[i] = 0.0
                    agent_episode_length[i] = 0

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
                print(f"  Step reward raw: [{raw_reward_min:.2f}, {raw_reward_max:.2f}]"
                      f"  (success needs > {args.success_reward_threshold})")
                if missing_obs_steps:
                    print(f"  ⚠️  Mid-episode missing obs: {missing_obs_steps} steps this rollout")

                if use_wandb:
                    import wandb
                    log = {f"train/{k}": v for k, v in train_stats.items()}
                    # Unity-side StatsRecorder values (mean per rollout)
                    for stat_key, stat_values in env_stats.get_and_reset_stats().items():
                        log[f"env/{stat_key}"] = float(np.mean([v for v, _ in stat_values]))
                    log.update({
                        'train/reward_mean': mean_reward,
                        'train/success_rate': success_rate,
                        'train/episode_length': mean_length,
                        'train/total_steps': total_steps,
                        'train/step_reward_raw_max': raw_reward_max,
                        'train/step_reward_raw_min': raw_reward_min,
                    })
                    if curriculum is not None:
                        log['train/curriculum_stage'] = curriculum.current_stage
                    wandb.log(log, step=total_steps)

            # ---- Checkpoints ----
            if num_updates % args.save_every == 0:
                if try_save(save_dir / f"mappo_checkpoint_{total_steps:08d}.pth"):
                    prune_checkpoints(save_dir, args.keep_checkpoints)

            # Best by success rate, with mean reward as tiebreak.
            if episode_successes and (
                    success_rate > best_success
                    or (success_rate == best_success and mean_reward > best_reward)):
                best_success = success_rate
                best_reward = mean_reward
                try_save(save_dir / "mappo_best.pth")

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at {total_steps:,} steps ({num_updates} updates)")

    try_save(save_dir / "mappo_final.pth")
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
