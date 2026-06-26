import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class MPCConfig:
    # Planning Horizon
    horizon: int = 20
    num_samples: int = 1000  # MPPI samples per step
    dt: float = 0.1
    
    # MPPI Parameters
    temperature: float = 0.5  # Lambda in MPPI (exploration noise variance)
    noise_sigma: float = 0.3  # Action smoothing noise

    # Cost Weights
    goal_weight: float = 20.0
    collision_weight: float = 50.0
    control_weight: float = 0.01
    velocity_weight: float = 1.0
    
    # Constraints
    max_velocity: float = 10.0
    min_separation: float = 1.5
    obstacle_buffer: float = 0.5

    # Dimensions
    full_state_dim: int = 92   # Full vector obs from Unity
    # The drone state lives in the LOCAL frame: the env observes
    # [localGoalPos(3), rotationQuat(4), localVel(3), localAngVel(3), speed(1)]
    # — there is NO world position. The MPC state is the first 13 of that
    # block (speed scalar dropped); state[:3] is the goal offset in the
    # drone's frame, so the planning target is always the ORIGIN.
    kinematic_dim: int = 13
    kinematic_offset: int = 0  # start of the state block inside the full vector obs
    action_dim: int = 4        # 4 continuous actions
    hidden_dim: int = 256

    # Training
    dynamics_lr: float = 1e-3
    dynamics_buffer_size: int = 100000
    dynamics_batch_size: int = 256

    # Proportional-controller baseline gains (see ProportionalController).
    # Translation gain is applied to the goal offset in METERS: 0.5 saturates
    # the command at ~2 m, i.e. full speed toward the goal until ~2 m out.
    p_gain_translation: float = 0.5
    p_gain_yaw: float = 0.3

class LearnedDynamicsModel(nn.Module):
    """
    Predicts NEXT kinematic state given CURRENT kinematic state and ACTION.
    State: [x, y, z, vx, vy, vz, rx, ry, rz, wx, wy, wz] (12 dims)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim) # Predicts Delta State
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Predict delta
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return state + delta

class DynamicsBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0
        self.capacity = capacity

    def add(self, state, action, next_state):
        idx = self.ptr
        self.states[idx] = torch.tensor(state)
        self.actions[idx] = torch.tensor(action)
        self.next_states[idx] = torch.tensor(next_state)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=device)
        return self.states[idx], self.actions[idx], self.next_states[idx]

class MPPIController:
    """
    Model Predictive Path Integral (MPPI) Controller.
    Parallelized on GPU for high-speed batch shooting.
    """
    def __init__(self, agent_id: int, config: MPCConfig):
        self.agent_id = agent_id
        self.config = config
        
        # Dynamics specific to Kinematics
        self.dynamics = LearnedDynamicsModel(config.kinematic_dim, config.action_dim).to(device)
        self.optimizer = optim.Adam(self.dynamics.parameters(), lr=config.dynamics_lr)
        self.buffer = DynamicsBuffer(config.dynamics_buffer_size, config.kinematic_dim, config.action_dim)
        
        # Mean action sequence for warm start
        self.U_mean = torch.zeros(config.horizon, config.action_dim, device=device)

    def get_action(self, current_kinematics_np, goal_np, other_agent_positions_np=None):
        """
        Run MPPI optimization loop.
        """
        K = self.config.num_samples
        H = self.config.horizon
        
        # Prepare tensors
        state = torch.tensor(current_kinematics_np, dtype=torch.float32, device=device) # (12,)
        goal = torch.tensor(goal_np, dtype=torch.float32, device=device)                # (3,) or (12,)
        
        # 1. Sample perturbation noise
        noise = torch.randn(K, H, self.config.action_dim, device=device) * self.config.noise_sigma
        
        # 2. Create perturbed action sequences: u = u_mean + noise
        # (K, H, A)
        perturbed_actions = self.U_mean.unsqueeze(0) + noise
        perturbed_actions = torch.clamp(perturbed_actions, -1.0, 1.0)
        
        # 3. Rollout Dynamics (Batch mode)
        # States: (K, H+1, State_Dim)
        states = torch.zeros(K, H + 1, self.config.kinematic_dim, device=device)
        states[:, 0] = state # Set initial state for all K samples
        
        curr_states = states[:, 0]
        
        with torch.no_grad():
            for t in range(H):
                next_states = self.dynamics(curr_states, perturbed_actions[:, t])
                states[:, t+1] = next_states
                curr_states = next_states

        # 4. Compute Costs (Vectorized)
        costs = torch.zeros(K, device=device)
        
        # Distance to goal (Use only Position x,y,z which are indices 0,1,2)
        # Assuming goal is just position (3,)
        pred_pos = states[:, 1:, :3] # (K, H, 3)
        
        # Goal Cost (Sum over Horizon)
        dist_sq = torch.sum((pred_pos - goal[:3])**2, dim=2)
        costs += dist_sq.sum(dim=1) * self.config.goal_weight
        
        # Collision Cost (with other agents)
        if other_agent_positions_np is not None:
            others = torch.tensor(other_agent_positions_np, device=device) # (Num_Others, 3)
            # pred_pos: (K, H, 3) vs others: (N, 3) -> Distance matrix
            # Simple check: distance to nearest neighbor at each step
            for t in range(H):
                my_pos_t = pred_pos[:, t, :] # (K, 3)
                # Broadcast distance calc
                # (K, 1, 3) - (1, N, 3) -> (K, N, 3)
                dists = torch.norm(my_pos_t.unsqueeze(1) - others.unsqueeze(0), dim=2)
                # Penalize if dist < threshold
                collision_mask = dists < self.config.min_separation
                costs += collision_mask.sum(dim=1) * self.config.collision_weight

        # Control Cost
        costs += torch.sum(perturbed_actions**2, dim=(1,2)) * self.config.control_weight

        # 5. MPPI Update Rule (Information Theoretic)
        # Weights = exp(-1/lambda * (Cost - minCost))
        costs = costs - torch.min(costs)
        weights = torch.exp(-costs / self.config.temperature)
        weights = weights / (torch.sum(weights) + 1e-6) # Normalize
        
        # Weighted average of noise
        # (K, 1, 1) * (K, H, A) -> Sum over K -> (H, A)
        weighted_noise = torch.sum(weights.view(-1, 1, 1) * noise, dim=0)
        
        # Update mean trajectory
        self.U_mean = torch.clamp(self.U_mean + weighted_noise, -1.0, 1.0)
        
        # Return first action
        action = self.U_mean[0].cpu().numpy()
        
        # Warm start shift (slide window)
        self.U_mean = torch.roll(self.U_mean, -1, dims=0)
        self.U_mean[-1] = torch.zeros(self.config.action_dim, device=device)
        
        return action

    def train(self):
        if self.buffer.size < self.config.dynamics_batch_size:
            return 0.0
        
        s, a, ns = self.buffer.sample(self.config.dynamics_batch_size)
        pred_ns = self.dynamics(s, a)
        loss = F.mse_loss(pred_ns, ns)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class MultiAgentMPC:
    def __init__(self, num_agents, config: MPCConfig):
        self.num_agents = num_agents
        self.config = config
        self.controllers = [MPPIController(i, config) for i in range(num_agents)]
        
    def extract_kinematics(self, vector_obs):
        """
        Extracts the drone-state block from the full Unity vector obs
        (which also contains raycasts, whose position in the concatenation
        is sensor-order dependent — set config.kinematic_offset from
        UnityMultiAgentEnv.vector_block_slices).
        """
        o = self.config.kinematic_offset
        return vector_obs[:, o:o + self.config.kinematic_dim]

    def get_actions(self, vector_obs, goals):
        """
        vector_obs: (N, 92)
        goals: (N, 3)
        """
        actions = []
        kinematics = self.extract_kinematics(vector_obs)  # (N, kinematic_dim)

        start_t = time.time()

        for i in range(self.num_agents):
            # NOTE: state[:3] is each drone's LOCAL goal offset, not a world
            # position, so positions cannot be shared between drones for
            # collision avoidance — the env does not observe world positions.
            action = self.controllers[i].get_action(
                kinematics[i],
                goals[i],
                None
            )
            actions.append(action)
            
        solve_time = time.time() - start_t
        
        return np.array(actions), {'solve_time': solve_time}

    def update_dynamics(self, transitions):
        losses = []
        o = self.config.kinematic_offset
        d = self.config.kinematic_dim
        for t in transitions:
            # t is {agent_id: {state, action, next_state}}
            for i, data in t.items():
                # Important: Only store the kinematic block, not the full obs
                k_state = data['state'][o:o + d]
                k_next = data['next_state'][o:o + d]

                self.controllers[i].buffer.add(k_state, data['action'], k_next)
                
        for controller in self.controllers:
            loss = controller.train()
            losses.append(loss)
            
        return np.mean(losses)
    
    def save(self, path):
        torch.save([c.dynamics.state_dict() for c in self.controllers], path)

    def load(self, path):
        dicts = torch.load(path)
        for i, c in enumerate(self.controllers):
            c.dynamics.load_state_dict(dicts[i])


class ProportionalController:
    """
    Privileged proportional controller for open goal-reaching.

    Reads the goal direction directly from the observation (localGoalPos — the
    goal expressed in the drone's own frame) and commands velocity straight
    toward it, with a mild yaw term to face the goal. No model, no learning:
    a correct, transparent classical reference. It has NO obstacle sensing, so
    by construction it solves open levels and fails on cluttered ones — which
    is exactly the comparison point we want against MAPPO.

    Kinematic-block layout (Unity local frame, x=right, y=up, z=forward):
        [0:3] localGoalPos / 50     (goal offset, normalized)
        [3:7] rotation quaternion
        [7:10] localVel / normalSpeed
        [10:13] localAngVel / 5
    Action layout (matches DroneAgentCamera.OnActionReceived):
        [forward(+z), lateral(+x), ascent(+y), yaw]
    """

    def __init__(self, config: MPCConfig):
        self.config = config

    def get_action(self, state, goal=None):
        # state[:3] is localGoalPos / 50; recover the offset in metres.
        gx, gy, gz = state[0] * 50.0, state[1] * 50.0, state[2] * 50.0
        kp = self.config.p_gain_translation
        forward = np.clip(kp * gz, -1.0, 1.0)   # toward goal along local +z
        lateral = np.clip(kp * gx, -1.0, 1.0)   # along local +x (right)
        ascent = np.clip(kp * gy, -1.0, 1.0)    # along local +y (up)
        # Turn to face the goal in the horizontal plane (gentle; translation
        # already strafes toward the goal, so yaw is a refinement).
        yaw = np.clip(self.config.p_gain_yaw * np.arctan2(gx, gz), -1.0, 1.0)
        return np.array([forward, lateral, ascent, yaw], dtype=np.float32)


class MultiAgentProportionalController:
    """
    Drop-in replacement for MultiAgentMPC exposing the same interface
    (get_actions / update_dynamics / save / load), so the MPC notebook and
    the per-level evaluation run unchanged. There is no dynamics model, so
    update_dynamics is a no-op and the exploration/training phase can be
    skipped entirely (set exploration_steps=0 or just call the eval directly).
    """

    def __init__(self, num_agents, config: MPCConfig):
        self.num_agents = num_agents
        self.config = config
        self.controllers = [ProportionalController(config) for _ in range(num_agents)]

    def extract_kinematics(self, vector_obs):
        o = self.config.kinematic_offset
        return vector_obs[:, o:o + self.config.kinematic_dim]

    def get_actions(self, vector_obs, goals):
        kinematics = self.extract_kinematics(vector_obs)  # (N, kinematic_dim)
        start_t = time.time()
        actions = [self.controllers[i].get_action(kinematics[i])
                   for i in range(self.num_agents)]
        return np.array(actions), {'solve_time': time.time() - start_t}

    def update_dynamics(self, transitions):
        return 0.0  # no model to train

    def save(self, path):
        torch.save({'config': self.config}, path)

    def load(self, path):
        pass