"""Tests for the Unity-independent parts of MAPPO/evaluate.py."""

import numpy as np

from MAPPO.evaluate import summarize


def test_summarize_empty():
    s = summarize([], [], [])
    assert s['episodes'] == 0
    assert s['success_rate'] == 0.0
    assert s['reward_mean'] == 0.0


def test_summarize_basic():
    rewards = [10.0, 20.0, 30.0]
    lengths = [100, 200, 300]
    successes = [1.0, 0.0, 1.0]
    s = summarize(rewards, lengths, successes)
    assert s['episodes'] == 3
    assert s['success_rate'] == 2 / 3
    assert s['reward_mean'] == 20.0
    assert s['length_mean'] == 200.0
    assert s['reward_std'] == np.std(rewards)


def test_summarize_all_success():
    s = summarize([5.0, 5.0], [50, 50], [1.0, 1.0])
    assert s['success_rate'] == 1.0
    assert s['length_std'] == 0.0
