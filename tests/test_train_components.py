"""Unit tests for the Unity-independent pieces of MAPPO/train.py."""

import numpy as np

from MAPPO.train import FourStageCurriculum, TrainingHealthMonitor, gather_observations


class TestCurriculum:
    def paths(self):
        return ['lvl1', 'lvl2', 'lvl3', 'lvl4']

    def test_starts_at_stage_one(self):
        cur = FourStageCurriculum(self.paths())
        assert cur.current_stage == 1
        assert cur.get_current_env_path() == 'lvl1'

    def test_no_advance_before_min_steps(self):
        cur = FourStageCurriculum(self.paths(), min_steps_per_stage=1000,
                                  max_steps_per_stage=5000)
        assert not cur.should_advance(total_steps=500, success_rate=1.0)

    def test_advance_on_success(self):
        cur = FourStageCurriculum(self.paths(), thresholds=(0.6, 0.7, 0.8),
                                  min_steps_per_stage=1000, max_steps_per_stage=5000)
        assert cur.should_advance(total_steps=1500, success_rate=0.65)
        assert cur.current_stage == 2
        assert cur.get_current_env_path() == 'lvl2'
        # the new stage's threshold is higher
        assert not cur.should_advance(total_steps=3000, success_rate=0.65)

    def test_forced_advance_at_max_steps(self):
        """A stage can never trap the run forever."""
        cur = FourStageCurriculum(self.paths(), min_steps_per_stage=1000,
                                  max_steps_per_stage=5000)
        assert cur.should_advance(total_steps=5000, success_rate=0.0)
        assert cur.current_stage == 2

    def test_final_stage_never_advances(self):
        cur = FourStageCurriculum(self.paths())
        cur.current_stage = 4
        assert not cur.should_advance(total_steps=10**9, success_rate=1.0)


class TestHealthMonitor:
    def stats(self, entropy=0.0, kl=0.01):
        return {'entropy': entropy, 'approx_kl': kl}

    def test_aborts_on_pinned_entropy(self):
        mon = TrainingHealthMonitor(max_entropy=5.0)
        msg = None
        for _ in range(35):
            msg = mon.check(self.stats(entropy=4.99), 1000, 0.0)
            if msg:
                break
        assert msg is not None and "noise" in msg

    def test_aborts_on_dead_kl(self):
        mon = TrainingHealthMonitor(max_entropy=5.0)
        msg = None
        for _ in range(35):
            msg = mon.check(self.stats(kl=1e-6), 1000, 0.0)
            if msg:
                break
        assert msg is not None and "not moving" in msg

    def test_counters_reset_on_healthy_update(self):
        mon = TrainingHealthMonitor(max_entropy=5.0)
        for _ in range(25):
            assert mon.check(self.stats(entropy=4.99), 1000, 0.0) is None
        mon.check(self.stats(entropy=2.0), 1000, 0.0)  # healthy
        assert mon.entropy_stuck == 0


def test_gather_observations_zero_fills_missing():
    cam_shape, vec_shape = (1, 8, 8), (3,)
    obs_dict = {
        'a': {'camera_obs': np.ones((1, 8, 8)), 'vector_obs': np.ones(3)},
        # 'b' missing
    }
    cams, vecs, missing = gather_observations(obs_dict, ['a', 'b'], cam_shape, vec_shape)
    assert missing == ['b']
    assert cams[0].sum() > 0 and cams[1].sum() == 0
    assert vecs[1].sum() == 0
