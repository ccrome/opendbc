import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus
from opendbc.car.honda.carcontroller import longitudinal_control_allowed, update_crv_bosch_long_mode
from opendbc.car.honda.hondacan import CanBus, create_acc_commands
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC, HondaFlags


class TestHondaFingerprint(unittest.TestCase):
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
    messages = create_acc_commands(packer, can, allowed, allowed, 1.7, 1300.0, False, False, 0, cp)
    parser.update([(1_000_000_000, messages)])

    assert parser.vl['ACC_CONTROL']['ACCEL_COMMAND'] == 0.0
    assert parser.vl['ACC_CONTROL']['GAS_COMMAND'] == -30000.0
    assert parser.vl['ACC_CONTROL_ON']['CONTROL_ON'] == 0.0

  def test_crv_bosch_long_mode_boundaries(self):
    assert update_crv_bosch_long_mode(-0.151, "coast", True, False) == "brake"
    assert update_crv_bosch_long_mode(-0.150, "coast", True, False) == "coast"
    assert update_crv_bosch_long_mode(-0.050, "brake", True, False) == "brake"
    assert update_crv_bosch_long_mode(-0.049, "brake", True, False) == "coast"
    assert update_crv_bosch_long_mode(0.050, "coast", True, False) == "coast"
    assert update_crv_bosch_long_mode(0.051, "coast", True, False) == "gas"
    assert update_crv_bosch_long_mode(0.050, "gas", True, False) == "coast"
    assert update_crv_bosch_long_mode(1.0, "gas", False, False) == "coast"
    assert update_crv_bosch_long_mode(0.0, "coast", True, True) == "brake"

  def test_acc_control_uses_explicit_output_mode(self):
    cp = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
    can = CanBus(cp)
    packer = CANPacker(DBC[CAR.HONDA_CRV_5G][Bus.pt])
    parser = CANParser(DBC[CAR.HONDA_CRV_5G][Bus.pt], [('ACC_CONTROL', 0)], can.pt)

    cases = (
      (0.0, 500.0, False, False, -30000.0, 0.0),
      (0.1, 500.0, False, True, 500.0, 0.0),
      (-0.2, 500.0, True, False, -30000.0, 1.0),
    )
    for accel, gas, braking, gas_allowed, expected_gas, expected_brake in cases:
      messages = create_acc_commands(packer, can, True, True, accel, gas, braking, gas_allowed, 0, cp)
      parser.update([(1_000_000_000, messages)])
      assert parser.vl['ACC_CONTROL']['GAS_COMMAND'] == expected_gas
      assert parser.vl['ACC_CONTROL']['BRAKE_REQUEST'] == expected_brake
