"""
Low-level ML-Agents adapter with INDEPENDENT per-agent episodes.

The PettingZoo wrapper (UnityParallelEnv) permanently removes an agent from
the live set when its episode ends and forces a global reset() once all have
terminated — so one drone crashing yanks every other drone back to spawn.
This adapter talks to the low-level mlagents_envs API directly
(get_steps / set_actions), where a terminated agent simply respawns inside
Unity and keeps producing decision steps, exactly matching the env design
("Only this agent resets!").

API:
    env = UnityMultiAgentEnv(file_name, seed=..., no_graphics=..., side_channels=[...])
    cam, vec = env.reset()                       # (N,C,H,W) float32, (N,D) float32
    cam, vec, rewards, dones, interrupted, stale = env.step(actions)  # actions (N,A) in [-1,1]

- dones[i] = 1 marks a TRUE terminal for slot i (success/timeout/crash);
  the returned obs for that slot is the respawned episode's first obs.
- interrupted[i] = 1 when ML-Agents flags the episode as interrupted
  (built-in MaxStep) — a truncation, which the caller may bootstrap.
- stale[i] = 1 if slot i produced no fresh decision within the micro-step
  budget (its obs is the previous one); should be rare with synchronized
  DecisionRequesters.
"""

import numpy as np


class UnityMultiAgentEnv:

    def __init__(self, file_name, seed=0, no_graphics=False, side_channels=None,
                 max_micro_steps=20):
        # Imported lazily so this module stays importable without Unity/mlagents
        from mlagents_envs.environment import UnityEnvironment, ActionTuple
        self._ActionTuple = ActionTuple
        self._max_micro_steps = max_micro_steps

        self._env = UnityEnvironment(
            file_name=file_name, seed=seed, no_graphics=no_graphics,
            side_channels=side_channels or [])
        self._env.reset()

        names = list(self._env.behavior_specs)
        if len(names) != 1:
            raise RuntimeError(f"Expected exactly one behavior, got: {names}")
        self.behavior_name = names[0]
        spec = self._env.behavior_specs[self.behavior_name]

        self.action_dim = spec.action_spec.continuous_size
        if self.action_dim == 0:
            raise RuntimeError("Behavior has no continuous actions")

        obs_specs = spec.observation_specs
        cam_indices = [i for i, o in enumerate(obs_specs) if len(o.shape) == 3]
        if len(cam_indices) != 1:
            raise RuntimeError(
                f"Expected exactly one camera observation, got {len(cam_indices)}")
        self._cam_i = cam_indices[0]
        self._vec_is = [i for i, o in enumerate(obs_specs) if len(o.shape) == 1]

        cam_shape = tuple(obs_specs[self._cam_i].shape)
        # ML-Agents sends visual obs as HWC; detect and convert to CHW.
        self._cam_hwc = cam_shape[-1] in (1, 3, 4) and cam_shape[0] not in (1, 3, 4)
        self.camera_shape = ((cam_shape[-1], cam_shape[0], cam_shape[1])
                             if self._cam_hwc else cam_shape)
        self.vector_dim = int(sum(obs_specs[i].shape[0] for i in self._vec_is))
        # (start, end) slice of each 1D sensor inside the concatenated vector
        # obs, in spec order — lets consumers (e.g. the MPC baseline) locate a
        # specific sensor block (drone state vs raycasts) without guessing.
        self.vector_block_slices = []
        offset = 0
        for i in self._vec_is:
            size = int(obs_specs[i].shape[0])
            self.vector_block_slices.append((offset, offset + size))
            offset += size

        ds, _ = self._env.get_steps(self.behavior_name)
        if len(ds) == 0:
            raise RuntimeError("No agents requested decisions after reset()")
        self.num_agents = len(ds)

        self._cam = np.zeros((self.num_agents, *self.camera_shape), np.float32)
        self._vec = np.zeros((self.num_agents, self.vector_dim), np.float32)
        self._init_slots(ds)
        self._ingest_decisions(ds)
        self._ds = ds

    # ------------------------------------------------------------------
    # Slot management: maps (possibly per-episode) ML-Agents agent ids to
    # stable slot indices 0..N-1. Works whether ids persist across episodes
    # or a fresh id appears for each new episode: a terminal id's slot is
    # retired and recycled for the next unknown id.
    # ------------------------------------------------------------------

    def _init_slots(self, ds):
        ids = sorted(int(a) for a in ds.agent_id)
        self._slot_of = {aid: slot for slot, aid in enumerate(ids)}
        self._retired = {}  # insertion-ordered: aid -> slot

    def _slot(self, aid):
        aid = int(aid)
        slot = self._slot_of.get(aid)
        if slot is not None:
            # The id continued past a terminal (stable-id case): un-retire it.
            self._retired.pop(aid, None)
            return slot
        # Unknown id: a respawned agent with a fresh episode id — recycle the
        # oldest retired slot.
        if not self._retired:
            raise RuntimeError(
                f"Unknown agent id {aid} and no retired slot to recycle "
                f"(more live agents than slots?)")
        old_aid = next(iter(self._retired))
        slot = self._retired.pop(old_aid)
        del self._slot_of[old_aid]
        self._slot_of[aid] = slot
        return slot

    def _retire(self, aid, slot):
        self._retired[int(aid)] = slot

    # ------------------------------------------------------------------

    def _extract(self, steps, j):
        cam = np.asarray(steps.obs[self._cam_i][j])
        if cam.dtype == np.uint8:
            cam = cam.astype(np.float32) / 255.0
        if self._cam_hwc:
            cam = np.transpose(cam, (2, 0, 1))
        vec = np.concatenate(
            [np.asarray(steps.obs[i][j], dtype=np.float32).ravel()
             for i in self._vec_is])
        return cam.astype(np.float32, copy=False), vec

    def _ingest_decisions(self, ds):
        for j, aid in enumerate(ds.agent_id):
            slot = self._slot(aid)
            self._cam[slot], self._vec[slot] = self._extract(ds, j)

    def reset(self):
        self._env.reset()
        ds, _ = self._env.get_steps(self.behavior_name)
        self._init_slots(ds)
        self._ingest_decisions(ds)
        self._ds = ds
        return self._cam.copy(), self._vec.copy()

    def step(self, actions):
        """
        actions: (num_agents, action_dim) in [-1, 1], indexed by slot.

        Advances the simulation until every slot has produced a fresh
        decision (or the micro-step budget runs out). Within one call, an
        agent that needs multiple engine decisions repeats its given action;
        rewards accumulate and done flags are OR-ed, so the caller sees one
        aligned transition per slot.
        """
        N = self.num_agents
        rewards = np.zeros(N, np.float32)
        dones = np.zeros(N, np.float32)
        interrupted = np.zeros(N, np.float32)
        seen = np.zeros(N, bool)

        for _ in range(self._max_micro_steps):
            if len(self._ds) > 0:
                arr = np.zeros((len(self._ds), self.action_dim), np.float32)
                for j, aid in enumerate(self._ds.agent_id):
                    arr[j] = actions[self._slot(aid)]
                self._env.set_actions(self.behavior_name,
                                      self._ActionTuple(continuous=arr))
            self._env.step()
            ds, ts = self._env.get_steps(self.behavior_name)

            for j, aid in enumerate(ts.agent_id):
                slot = self._slot(aid)
                rewards[slot] += float(ts.reward[j])
                dones[slot] = 1.0
                if bool(ts.interrupted[j]):
                    interrupted[slot] = 1.0
                self._retire(aid, slot)

            for j, aid in enumerate(ds.agent_id):
                slot = self._slot(aid)
                rewards[slot] += float(ds.reward[j])
                self._cam[slot], self._vec[slot] = self._extract(ds, j)
                seen[slot] = True

            self._ds = ds
            if seen.all():
                break

        stale = (~seen).astype(np.float32)
        return (self._cam.copy(), self._vec.copy(),
                rewards, dones, interrupted, stale)

    def close(self):
        self._env.close()
