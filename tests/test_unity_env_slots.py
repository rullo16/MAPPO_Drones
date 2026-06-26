"""Slot-management tests for UnityMultiAgentEnv (no Unity required).

The adapter maps ML-Agents agent ids to stable slot indices. Ids may
persist across episodes or change per episode (a fresh id after each
EndEpisode); both must keep slots stable and never collide.
"""

import pytest

from MAPPO.unity_env import UnityMultiAgentEnv


class FakeDecisionSteps:
    def __init__(self, agent_ids):
        self.agent_id = agent_ids


def make_adapter(initial_ids):
    env = object.__new__(UnityMultiAgentEnv)  # skip __init__ (needs Unity)
    env._init_slots(FakeDecisionSteps(initial_ids))
    return env


def test_initial_slots_are_sorted_and_stable():
    env = make_adapter([7, 3, 5])
    assert env._slot(3) == 0
    assert env._slot(5) == 1
    assert env._slot(7) == 2
    # repeated lookups never move an agent
    assert env._slot(5) == 1


def test_stable_ids_survive_terminal_and_continue():
    env = make_adapter([1, 2, 3])
    slot = env._slot(2)
    env._retire(2, slot)
    # Same id reappears (stable-id case): keeps its slot, leaves retirement
    assert env._slot(2) == slot
    assert 2 not in env._retired


def test_fresh_episode_ids_recycle_retired_slots():
    env = make_adapter([1, 2, 3])
    env._retire(2, env._slot(2))
    # A brand-new id (respawn with fresh episode id) takes the retired slot
    assert env._slot(99) == 1
    assert 2 not in env._slot_of
    # The new id is now stable
    assert env._slot(99) == 1


def test_multiple_retirements_recycle_in_order():
    env = make_adapter([10, 20, 30])
    env._retire(10, env._slot(10))
    env._retire(30, env._slot(30))
    # Oldest retirement is recycled first
    assert env._slot(111) == env._slot_of[111] == 0
    assert env._slot(222) == 2


def test_unknown_id_with_no_retired_slot_raises():
    env = make_adapter([1, 2])
    with pytest.raises(RuntimeError, match="no retired slot"):
        env._slot(99)
