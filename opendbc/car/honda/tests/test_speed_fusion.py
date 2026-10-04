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


if __name__ == "__main__":
  unittest.main()
