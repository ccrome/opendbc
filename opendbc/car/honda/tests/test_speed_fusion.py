"""Recorded CR-V transmission-speed excursion must not become ego acceleration."""
from types import SimpleNamespace
import unittest

import numpy as np

from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.values import CAR


def sensor(fingerprint):
  result = CarState.__new__(CarState)
  result.CP = SimpleNamespace(carFingerprint=fingerprint, wheelSpeedFactor=1.0)
  return result


class SpeedState:
  def __init__(self, speed):
    self._x = [[speed], [0.]]

  @property
  def x(self):
    return self._x

  def set_x(self, state):
    self._x = state


def observer(speed):
  cs = sensor(CAR.HONDA_CRV_5G)
  cs.v_ego_kf = SpeedState(speed)
  cs.crv_speed_covariance = np.diag([.0002, 1.])
  cs.crv_speed_initialized = True
  cs.crv_speed_dropout_time = 0.
  cs.crv_high_speed_dropout_active = False
  cs.crv_speed_dropout_timeout = False
  cs._apply_crv_speed_scale = lambda measurement, _cluster: measurement
  return cs


class TestHondaSpeedFusion(unittest.TestCase):
  def test_recorded_oct3_bookmark3_braking_remains_monotonic(self):
    # Route 0000003f--43ddadcc82, segment 6, t=391.343–391.692 s.
    # Bus 1 WHEEL_SPEEDS mean versus ENGINE_DATA XMISSION_SPEED, in m/s.
    wheels = [1.186, 1.078, 1.02225, .9275, .90475, .83475, .8035, .7335]
    transmission = [1.178, 1.114, 1.114, 1.289, 1.208, .878, .656, .517]
    cs = sensor(CAR.HONDA_CRV_5G)
    raw = [cs.get_raw_speed(wheel, trans) for wheel, trans in zip(wheels, transmission, strict=True)]
    self.assertTrue(np.all(np.diff(raw) <= 0.0), raw)

  def test_crv_transmission_excursions_do_not_change_wheel_velocity(self):
    cs = sensor(CAR.HONDA_CRV_5G)
    for speed in (.05, .5, 1., 3., 10., 30.):
      for excursion in (-.3, 0., .3):
        self.assertEqual(cs.get_raw_speed(speed, max(0., speed + excursion)), speed)

  def test_other_hondas_retain_existing_transmission_blend(self):
    cs = sensor(CAR.HONDA_CIVIC)
    cs.CP.wheelSpeedFactor = 1.01
    for wheel in (.1, 1., 3., 6., 20.):
      weight = float(np.interp(wheel, [1., 6.], [0., 1.]))
      expected = (1. - weight) * .7 * cs.CP.wheelSpeedFactor + weight * wheel
      self.assertEqual(cs.get_raw_speed(wheel, .7), expected)

  def test_single_wheel_dropout_at_50_mph_uses_remaining_channels(self):
    speed = 50. * 0.44704
    cs = observer(speed)
    estimate, _, _, uncertainty, _ = cs.update_crv_speed_observer([0., speed, speed, speed], 0.)
    self.assertEqual(cs.crv_wheel_count, 3)
    self.assertTrue(cs.crv_speed_measurement_valid)
    self.assertAlmostEqual(estimate, speed, places=2)
    self.assertGreater(uncertainty, 0.)

  def test_fewer_valid_wheels_reduce_confidence(self):
    speed = 50. * 0.44704
    three = observer(speed)
    one = observer(speed)
    _, _, _, three_std, _ = three.update_crv_speed_observer([0., speed, speed, speed], 0.)
    _, _, _, one_std, _ = one.update_crv_speed_observer([0., 0., 0., speed], 0.)
    self.assertEqual(one.crv_wheel_count, 1)
    self.assertGreater(one_std, three_std)

  def test_all_wheel_dropout_is_prediction_with_growing_uncertainty(self):
    speed = 50. * 0.44704
    cs = observer(speed)
    first, _, raw, first_std, _ = cs.update_crv_speed_observer([0.] * 4, 0.)
    second, _, _, second_std, _ = cs.update_crv_speed_observer([0.] * 4, 0.)
    self.assertEqual(cs.crv_wheel_count, 0)
    self.assertFalse(cs.crv_speed_measurement_valid)
    self.assertAlmostEqual(first, speed, places=4)
    self.assertAlmostEqual(second, speed, places=4)
    self.assertAlmostEqual(raw, speed, places=4)
    self.assertGreater(second_std, first_std)

  def test_recovered_wheels_update_prediction_and_contract_uncertainty(self):
    speed = 50. * 0.44704
    cs = observer(speed)
    _, _, _, _, _ = cs.update_crv_speed_observer([0.] * 4, 0.)
    _, _, _, dropout_std, _ = cs.update_crv_speed_observer([0.] * 4, 0.)
    estimate, _, _, recovered_std, _ = cs.update_crv_speed_observer([speed] * 4, 0.)
    self.assertEqual(cs.crv_wheel_count, 4)
    self.assertTrue(cs.crv_speed_measurement_valid)
    self.assertAlmostEqual(estimate, speed, places=2)
    self.assertLess(recovered_std, dropout_std)

  def test_inconsistent_positive_wheel_is_rejected(self):
    speed = 50. * 0.44704
    cs = observer(speed)
    estimate, _, _, _, _ = cs.update_crv_speed_observer([100., 0., 0., 0.], 0.)
    self.assertEqual(cs.crv_wheel_count, 0)
    self.assertFalse(cs.crv_speed_measurement_valid)
    self.assertAlmostEqual(estimate, speed, places=3)

  def test_single_wheel_excursion_does_not_shift_median_speed(self):
    speed = 2.0
    cs = observer(speed)
    estimate, _, _, _, _ = cs.update_crv_speed_observer([speed + .3, speed, speed, speed], 0.)
    self.assertEqual(cs.crv_wheel_count, 4)
    self.assertAlmostEqual(estimate, speed, places=2)

  def test_sustained_mid_high_speed_dropout_triggers_takeover_after_half_second(self):
    speed = 50. * 0.44704
    cs = observer(speed)
    for _ in range(49):
      cs.update_crv_speed_observer([0.] * 4, 0.)
    self.assertFalse(cs.crv_speed_dropout_timeout)
    cs.update_crv_speed_observer([0.] * 4, 0.)
    self.assertTrue(cs.crv_speed_dropout_timeout)
    self.assertAlmostEqual(cs.crv_speed_dropout_time, .5)

  def test_low_speed_wheel_censoring_does_not_require_takeover(self):
    cs = observer(.3)
    for _ in range(200):
      cs.update_crv_speed_observer([0.] * 4, 0.)
    self.assertFalse(cs.crv_speed_dropout_timeout)
    self.assertEqual(cs.crv_speed_dropout_time, 0.)

  def test_two_recovered_wheels_clear_the_dropout_timeout(self):
    speed = 50. * 0.44704
    cs = observer(speed)
    for _ in range(60):
      cs.update_crv_speed_observer([0.] * 4, 0.)
    self.assertTrue(cs.crv_speed_dropout_timeout)
    cs.update_crv_speed_observer([speed, speed, 0., 0.], 0.)
    self.assertFalse(cs.crv_speed_dropout_timeout)
    self.assertEqual(cs.crv_speed_dropout_time, 0.)


if __name__ == "__main__":
  unittest.main()
