"""
Calibration probe for the proportional controller.

The hand-coded controller needs the correct mapping from the observation's
goal block to the 4 actions (which state index is forward/right/up, and with
what sign). Rather than guess, this probes the live env: from reset it applies
+1 on each action axis (others zero) for a few steps and reports how every
state dimension changes. The goal dimensions are the ones that move most under
translation; their sign tells you how to steer.

Usage:
    python -m MPC.calibrate --env-path ./Env/Level1/DroneFlightv1 --no-graphics
"""

import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env-path', required=True)
    p.add_argument('--no-graphics', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--probe-steps', type=int, default=5)
    p.add_argument('--frame-size', type=int, default=14,
                   help='Drone-state frame size; the current frame is the last '
                        'FRAME_SIZE entries of the vector obs (it is stacked)')
    args = p.parse_args()

    from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
    from MAPPO.unity_env import UnityMultiAgentEnv

    env = UnityMultiAgentEnv(file_name=args.env_path, seed=args.seed,
                             no_graphics=args.no_graphics,
                             side_channels=[StatsSideChannel()])
    try:
        print(f"num_agents={env.num_agents}  action_dim={env.action_dim}")
        print(f"vector_dim={env.vector_dim}")
        print(f"vector_block_slices (start,end per 1D sensor): {env.vector_block_slices}")
        # The drone-state sensor is frame-stacked, so the CURRENT frame is the
        # last FRAME_SIZE entries; its first 3 are localGoalPos. (Do NOT use the
        # smallest block — that is the raycast sensor.)
        kin_start = env.vector_dim - args.frame_size
        kin_end = env.vector_dim
        kin = args.frame_size
        print(f"current drone-state frame -> offset={kin_start}, size={kin}")
        print(f"localGoalPos candidate = state[{kin_start}:{kin_start+3}]\n")

        cam, vec = env.reset()
        print(f"sample full vector obs (agent 0):\n{np.round(vec[0], 3)}\n")
        print(f"current frame state[0:{kin}] (agent 0):\n"
              f"{np.round(vec[0, kin_start:kin_end], 3)}\n")

        action_names = ['forward', 'lateral', 'ascent', 'yaw'][:env.action_dim]
        print(f"Per-axis probe (+1 for {args.probe_steps} steps), mean Δ of each "
              f"state dim in the current frame (agent 0).")
        print("The first 3 dims are localGoalPos [x=right, y=up, z=forward]:")
        print(f"{'action':<10} " + " ".join(f"d{j:<6}" for j in range(kin)))

        for a in range(env.action_dim):
            cam, vec = env.reset()
            before = vec[0, kin_start:kin_end].copy()
            act = np.zeros((env.num_agents, env.action_dim), dtype=np.float32)
            act[:, a] = 1.0
            for _ in range(args.probe_steps):
                cam, vec, r, dones, interrupted, stale = env.step(act)
                if dones[0] > 0.5:   # agent 0 reset mid-probe; stop early
                    break
            after = vec[0, kin_start:kin_end]
            delta = after - before
            print(f"{action_names[a]:<10} " + " ".join(f"{d:>7.3f}" for d in delta))

        print("\nRead-off: the 3 state dims with the largest |Δ| under the "
              "translation axes are localGoalPos. For each, the action that "
              "decreases it is the one to steer with (command ∝ +state if +action "
              "decreases that dim).")
    finally:
        env.close()


if __name__ == '__main__':
    main()
