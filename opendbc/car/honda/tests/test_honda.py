import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus
from opendbc.car.honda.carcontroller import longitudinal_control_allowed
from opendbc.car.honda.hondacan import CanBus, create_acc_commands
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC, HondaFlags


class TestHondaFingerprint(unittest.TestCase):
  def test_crv_bosch_actuator_delay_matches_speed_regulation_calibration(self):
    cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
    self.assertAlmostEqual(cp.longitudinalActuatorDelay, 0.8)

  def test_tja_bosch_only(self):
    for car_model in CAR:
      if car_model.config.flags & HondaFlags.BOSCH_TJA_CONTROL:
        assert car_model.config.flags & HondaFlags.BOSCH, "Nidec car found with TJA control"

  def test_brake_hold_inhibits_longitudinal_control(self):
    assert not longitudinal_control_allowed(True, True, True)
    assert not longitudinal_control_allowed(True, False, True)
    assert longitudinal_control_allowed(True, True, False)
    assert not longitudinal_control_allowed(False, True, False)

  def test_brake_hold_sends_no_bosch_longitudinal_command(self):
    cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
    can = CanBus(cp)
    packer = CANPacker(DBC[CAR.HONDA_CRV_5G][Bus.pt])
    parser = CANParser(DBC[CAR.HONDA_CRV_5G][Bus.pt],
                       [('ACC_CONTROL', 0), ('ACC_CONTROL_ON', 0)], can.pt)

    allowed = longitudinal_control_allowed(True, True, True)
    messages = create_acc_commands(packer, can, allowed, allowed, 1.7, 1300.0, 0, cp)
    parser.update([(1_000_000_000, messages)])

    assert parser.vl['ACC_CONTROL']['ACCEL_COMMAND'] == 0.0
    assert parser.vl['ACC_CONTROL']['GAS_COMMAND'] == -30000.0
    assert parser.vl['ACC_CONTROL_ON']['CONTROL_ON'] == 0.0

  def test_negative_accel_never_becomes_gas_command(self):
    cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
    can = CanBus(cp)
    packer = CANPacker(DBC[CAR.HONDA_CRV_5G][Bus.pt])
    parser = CANParser(DBC[CAR.HONDA_CRV_5G][Bus.pt], [('ACC_CONTROL', 0)], can.pt)

    for accel in (-0.21, -0.20, -0.19, -0.01):
      messages = create_acc_commands(packer, can, True, True, accel, 100.0, 0, cp)
      parser.update([(1_000_000_000, messages)])
      assert parser.vl['ACC_CONTROL']['BRAKE_REQUEST'] == 1
      assert parser.vl['ACC_CONTROL']['GAS_COMMAND'] == -30000.0

    messages = create_acc_commands(packer, can, True, True, 0.01, 100.0, 0, cp)
    parser.update([(1_000_000_001, messages)])
    assert parser.vl['ACC_CONTROL']['BRAKE_REQUEST'] == 0
    assert parser.vl['ACC_CONTROL']['GAS_COMMAND'] == 100.0
