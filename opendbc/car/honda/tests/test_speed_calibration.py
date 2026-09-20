import math
import unittest

from opendbc.car.honda.speed_calibration import BATCH_DURATION, MAX_SCALE, MIN_SCALE, TIME_CONSTANT, CrvSpeedScaleEstimator, control_speed_scale, load_scale, persist_scale


class FakeParams:
  def __init__(self, value=None):
    self.value = value
    self.writes = []

  def get(self, _key, return_default=False):
    return self.value

  def put(self, _key, value):
    self.writes.append(value)
    self.value = value


def observe(estimator, scale, speed=30.0, duration=BATCH_DURATION):
  for _ in range(int(duration / 0.1)):
    estimator.update(speed, speed * scale, 0.1)


class TestCrvSpeedCalibration(unittest.TestCase):
  def test_adaptation_time_constant_is_five_minutes(self):
    self.assertEqual(TIME_CONSTANT, 5.0 * 60.0)

  def test_load_scale_defaults_and_bounds(self):
    self.assertEqual(load_scale(FakeParams()), 1.0)
    self.assertEqual(load_scale(FakeParams("invalid")), 1.0)
    self.assertEqual(load_scale(FakeParams("2.0")), MAX_SCALE)
    self.assertEqual(load_scale(FakeParams("0.1")), MIN_SCALE)

  def test_uses_through_origin_scale_and_adapts_slowly(self):
    estimator = CrvSpeedScaleEstimator()
    target = 1.036
    observe(estimator, target)
    expected = 1.0 + (target - 1.0) * (1.0 - math.exp(-BATCH_DURATION / TIME_CONSTANT))
    self.assertAlmostEqual(estimator.scale, expected, places=6)

  def test_converges_across_speeds_without_offset(self):
    estimator = CrvSpeedScaleEstimator()
    target = 1.036
    for _ in range(60):
      for speed in (10.0, 20.0, 30.0):
        observe(estimator, target, speed=speed, duration=BATCH_DURATION / 3)
    expected = 1.0 + (target - 1.0) * (1.0 - math.exp(-(60 * BATCH_DURATION) / TIME_CONSTANT))
    self.assertAlmostEqual(estimator.scale, expected, places=4)

  def test_rejects_invalid_and_out_of_bounds_samples(self):
    estimator = CrvSpeedScaleEstimator()
    for _ in range(1000):
      self.assertFalse(estimator.update(5.0, 5.5, 0.1))
      self.assertFalse(estimator.update(float("nan"), 30.0, 0.1))
    for _ in range(10):
      observe(estimator, 1.2)
    self.assertEqual(estimator.scale, 1.0)
    self.assertEqual(CrvSpeedScaleEstimator(2.0).scale, MAX_SCALE)

  def test_scale_is_applied_at_all_speeds(self):
    learned_scale = 1.036
    for speed in (0.0, 1.0, 5.0, 10.0, 30.0):
      self.assertEqual(control_speed_scale(speed, learned_scale), learned_scale)

  def test_persistence_is_rate_limited(self):
    estimator = CrvSpeedScaleEstimator()
    self.assertFalse(estimator.should_persist(600.0))
    estimator.scale += 0.001
    self.assertTrue(estimator.should_persist(600.0))
    estimator.mark_persisted(600.0)
    self.assertFalse(estimator.should_persist(601.0))

  def test_persistence_writes_registered_float_type(self):
    estimator = CrvSpeedScaleEstimator()
    estimator.scale = 1.00218
    params = FakeParams()

    self.assertTrue(persist_scale(params, estimator, 600.0))
    self.assertEqual(params.writes, [1.00218])
    self.assertIs(type(params.writes[0]), float)
    self.assertFalse(persist_scale(params, estimator, 601.0))
