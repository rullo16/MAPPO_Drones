"""
Deterministic evaluation of a trained MAPPO checkpoint.

The training curves include exploration noise (actions are sampled). For
thesis-grade numbers, this script loads a checkpoint and runs the policy
DETERMINISTICALLY (mean action, no exploration, no learning) for a fixed
number of agent-episodes, then reports success rate, mean reward and mean
episode length — using the SAME success definition as MAPPO/train.py so the
numbers are directly comparable.

Usage:
    python -m MAPPO.evaluate --checkpoint saved_models_mappo/mappo_best.pth \
        --env-path ./Env/FinalLevel/DroneFlightv1 --num-episodes 200

    # Evaluate the same checkpoint on every curriculum level:
    python -m MAPPO.evaluate --checkpoint saved_models_mappo/mappo_best.pth --all-levels

Results are printed as a table and saved to <out> (JSON).

Unity ML-Agents is imported only inside the run functions, so this module
stays importable (and the summary logic testable) without a Unity install.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from MAPPO.mappo_agent import MAPPOAgent, DEFAULT_CONFIG

DEFAULT_ENV_PATHS = [
    './Env/Level1/DroneFlightv1',
    './Env/Level1.5/DroneFlightv1',
    './Env/Level2/DroneFlightv1',
    './Env/FinalLevel/DroneFlightv1',
]


def summarize(rewards, lengths, successes):
    """Aggregate per-episode records into a results dict. Pure function so it
    can be unit-tested without Unity."""
    n = len(rewards)
    if n == 0:
        return {'episodes': 0, 'success_rate': 0.0, 'reward_mean': 0.0,
                'reward_std': 0.0, 'length_mean': 0.0, 'length_std': 0.0}
    rewards = np.asarray(rewards, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    successes = np.asarray(successes, dtype=np.float64)
    return {
        'episodes': int(n),
        'success_rate': float(successes.mean()),
        'reward_mean': float(rewards.mean()),
        'reward_std': float(rewards.std()),
        'length_mean': float(lengths.mean()),
        'length_std': float(lengths.std()),
    }


def evaluate_checkpoint(checkpoint_path, env_path, num_episodes, seed,
                        no_graphics, success_reward_threshold):
    """Load a checkpoint, run `num_episodes` deterministic agent-episodes on
    one env build, and return a summary dict."""
    from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
    from MAPPO.unity_env import UnityMultiAgentEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # The checkpoint carries the config it was trained with; use it so the
    # network shapes match exactly.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(DEFAULT_CONFIG)
    config.update(ckpt.get('config', {}))

    env = UnityMultiAgentEnv(file_name=env_path, seed=seed,
                             no_graphics=no_graphics,
                             side_channels=[StatsSideChannel()])
    try:
        agent = MAPPOAgent(
            camera_shape=env.camera_shape,
            vector_shape=(env.vector_dim,),
            action_dim=env.action_dim,
            num_agents=env.num_agents,
            config=config,
        )
        agent.load(checkpoint_path)

        rewards, lengths, successes = [], [], []
        ep_reward = np.zeros(env.num_agents)
        ep_length = np.zeros(env.num_agents, dtype=np.int64)

        cam, vec = env.reset()
        # Pre-encode once; reuse next encoding each step.
        encoded = agent.encode_observations(cam, vec)

        # Cap total steps so a non-terminating policy can't hang the eval.
        max_total_steps = num_episodes * 2000 + 10000
        steps = 0
        while len(rewards) < num_episodes and steps < max_total_steps:
            actions, _, _ = agent.get_action(encoded, deterministic=True)
            cam, vec, step_rewards, dones, interrupted, stale = env.step(actions)
            encoded = agent.encode_observations(cam, vec)

            ep_reward += step_rewards
            ep_length += 1
            steps += 1

            for i in np.flatnonzero(dones > 0.5):
                rewards.append(float(ep_reward[i]))
                lengths.append(int(ep_length[i]))
                successes.append(float(step_rewards[i] > success_reward_threshold))
                ep_reward[i] = 0.0
                ep_length[i] = 0

        if len(rewards) < num_episodes:
            print(f"  ⚠️  Only collected {len(rewards)}/{num_episodes} episodes "
                  f"within the step budget.")
        return summarize(rewards, lengths, successes)
    finally:
        env.close()


def print_table(results_by_level):
    """results_by_level: dict[level_name -> summary dict]."""
    header = f"{'Level':<28} {'Episodes':>9} {'Success':>9} {'Reward':>16} {'Ep.Length':>14}"
    print("\n" + "=" * len(header))
    print("DETERMINISTIC EVALUATION")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for level, s in results_by_level.items():
        print(f"{level:<28} {s['episodes']:>9} {s['success_rate']:>8.1%} "
              f"{s['reward_mean']:>8.2f}±{s['reward_std']:<6.2f} "
              f"{s['length_mean']:>7.1f}±{s['length_std']:<5.1f}")
    print("=" * len(header) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Deterministic MAPPO evaluation")
    p.add_argument('--checkpoint', required=True,
                   help='Path to a .pth checkpoint (e.g. mappo_best.pth)')
    p.add_argument('--env-path', default=None,
                   help='Single environment build to evaluate on')
    p.add_argument('--all-levels', action='store_true',
                   help='Evaluate on every build in --env-paths')
    p.add_argument('--env-paths', nargs='+', default=DEFAULT_ENV_PATHS,
                   help='Builds used by --all-levels')
    p.add_argument('--num-episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=123,
                   help='Eval seed (different from training seed by default)')
    p.add_argument('--no-graphics', action='store_true')
    p.add_argument('--success-reward-threshold', type=float, default=15.0,
                   help='Per-step reward above which a terminal counts as success '
                        '(keep in sync with the Unity reward code and train.py)')
    p.add_argument('--out', default='eval_results.json',
                   help='Where to write the results JSON')
    return p.parse_args()


def main():
    args = parse_args()

    if args.all_levels:
        levels = args.env_paths
    elif args.env_path:
        levels = [args.env_path]
    else:
        levels = [args.env_paths[-1]]

    results_by_level = {}
    for env_path in levels:
        name = Path(env_path).parent.name or env_path
        print(f"\nEvaluating {args.checkpoint} on {env_path} ...")
        results_by_level[name] = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            env_path=env_path,
            num_episodes=args.num_episodes,
            seed=args.seed,
            no_graphics=args.no_graphics,
            success_reward_threshold=args.success_reward_threshold,
        )

    print_table(results_by_level)

    out = {
        'checkpoint': args.checkpoint,
        'num_episodes': args.num_episodes,
        'seed': args.seed,
        'success_reward_threshold': args.success_reward_threshold,
        'results': results_by_level,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Results written to {args.out}")


if __name__ == '__main__':
    main()
