import time

import numpy as np
from collections import defaultdict

from opendbc.can import CANDefine, CANParser
from opendbc.car import Bus, DT_CTRL, create_button_events, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.speed_calibration import CrvSpeedScaleEstimator, control_speed_scale, load_scale, persist_scale
from opendbc.car.honda.values import CAR, DBC, STEER_THRESHOLD, HondaFlags, CruiseButtons, CruiseSettings, \
                                                 GearShifter, CarControllerParams
from opendbc.car.interfaces import CarStateBase
from openpilot.common.params import Params

from opendbc.sunnypilot.car.honda.carstate_ext import CarStateExt

TransmissionType = structs.CarParams.TransmissionType
ButtonType = structs.CarState.ButtonEvent.Type

CRV_SPEED_DROPOUT_MIN_SPEED = 5.0  # m/s; low-speed wheel censoring is expected
CRV_SPEED_DROPOUT_TAKEOVER_TIME = 0.5  # seconds of fewer than two valid wheels

BUTTONS_DICT = {CruiseButtons.RES_ACCEL: ButtonType.accelCruise, CruiseButtons.DECEL_SET: ButtonType.decelCruise,
                CruiseButtons.MAIN: ButtonType.mainCruise, CruiseButtons.CANCEL: ButtonType.cancel}
SETTINGS_BUTTONS_DICT = {CruiseSettings.DISTANCE: ButtonType.gapAdjustCruise, CruiseSettings.LKAS: ButtonType.lkas}


class CarState(CarStateBase, CarStateExt):
  def __init__(self, CP, CP_SP):
    CarStateBase.__init__(self, CP, CP_SP)
    CarStateExt.__init__(self, CP, CP_SP)
    can_define = CANDefine(DBC[CP.carFingerprint][Bus.pt])

    if CP.transmissionType != TransmissionType.manual:
      self.gearbox_msg = "GEARBOX_AUTO"
      if CP.transmissionType == TransmissionType.cvt:
        self.gearbox_msg = "GEARBOX_CVT"
      self.shifter_values = can_define.dv[self.gearbox_msg]["GEAR_SHIFTER"]

    self.car_state_scm_msg = "SCM_FEEDBACK"
    if CP.flags & HondaFlags.NIDEC_ALT_SCM_MESSAGES:
      self.car_state_scm_msg = "SCM_BUTTONS"

    self.brake_error_msg = "HYBRID_BRAKE_ERROR" if CP.flags & HondaFlags.HYBRID else "STANDSTILL"

    self.steer_status_values = defaultdict(lambda: "UNKNOWN", can_define.dv["STEER_STATUS"]["STEER_STATUS"])

    self.brake_switch_prev = False
    self.brake_switch_active = False
    self.low_speed_alert = False

    self.dynamic_v_cruise_units = bool(self.CP.flags & (HondaFlags.BOSCH_RADARLESS | HondaFlags.BOSCH_ALT_RADAR | HondaFlags.BOSCH_CANFD))
    self.cruise_setting = 0
    self.v_cruise_pcm_prev = 0

    # When available we use cp.vl["CAR_SPEED"]["ROUGH_CAR_SPEED_2"] to populate vEgoCluster
    # However, on cars without a digital speedometer this is not always present (HRV, FIT, CRV 2016, ILX and RDX)
    self.dash_speed_seen = False
    self.is_metric = False
    self.v_cruise_factor = 1.
    self.crv_speed_scale = None
    self.crv_speed_scale_params = None
    self.crv_speed_scale_last_update = None
    if CP.carFingerprint == CAR.HONDA_CRV_5G:
      self.crv_speed_scale_params = Params()
      self.crv_speed_scale = CrvSpeedScaleEstimator(load_scale(self.crv_speed_scale_params))
      self.crv_speed_covariance = np.diag([0.3, 10.0])
      self.crv_speed_initialized = False
      self.crv_speed_measurement_valid = False
      self.crv_wheel_count = 0
      self.crv_speed_dropout_time = 0.
      self.crv_high_speed_dropout_active = False
      self.crv_speed_dropout_timeout = False

  def _get_cluster_speed(self, cp) -> float:
    if self.CP.carFingerprint in (CAR.HONDA_ODYSSEY_TWN,):
      return 0.0

    self.dash_speed_seen = self.dash_speed_seen or cp.vl["CAR_SPEED"]["ROUGH_CAR_SPEED_2"] > 1e-3
    if not self.dash_speed_seen:
      return 0.0

    conversion = CV.KPH_TO_MS if self.is_metric else CV.MPH_TO_MS
    return cp.vl["CAR_SPEED"]["ROUGH_CAR_SPEED_2"] * conversion

  def _apply_crv_speed_scale(self, unscaled_v_ego: float, cluster_speed: float) -> float:
    if self.crv_speed_scale is None:
      return unscaled_v_ego

    now = time.monotonic()
    dt = 0.0 if self.crv_speed_scale_last_update is None else now - self.crv_speed_scale_last_update
    self.crv_speed_scale_last_update = now
    self.crv_speed_scale.update(unscaled_v_ego, cluster_speed, dt)
    persist_scale(self.crv_speed_scale_params, self.crv_speed_scale, now)
    return unscaled_v_ego * control_speed_scale(unscaled_v_ego, self.crv_speed_scale.scale)

  def get_raw_speed(self, v_wheel: float, v_transmission: float) -> float:
    """Fuse the speed sources before calibration and the speed observer."""
    # The CR-V transmission signal can rise during smooth wheel deceleration
    # (Oct. 3 bookmark 3). Differentiating it creates a false acceleration and
    # an emergency brake pulse. Use the four-wheel mean for the CR-V; retain
    # the original low-speed transmission blend for every other Honda.
    wheel_weight_low = float(self.CP.carFingerprint == CAR.HONDA_CRV_5G)
    v_weight = float(np.interp(v_wheel, [1., 6.], [wheel_weight_low, 1.]))
    return (1. - v_weight) * v_transmission * self.CP.wheelSpeedFactor + v_weight * v_wheel

  def update_crv_speed_observer(self, wheel_speeds, cluster_speed):
    """Fuse valid wheel channels; coast the state and grow covariance on dropout."""
    speeds = np.asarray(wheel_speeds, dtype=float)
    predicted_speed = max(0., float(self.v_ego_kf.x[0][0]))
    valid = np.isfinite(speeds) & (speeds > 0.)
    # Zero is a censored low-speed reading while motion is expected. Treat it
    # as standstill evidence only once the independent motion prediction is
    # already effectively at rest. This gate applies at every vehicle speed.
    if predicted_speed <= 0.05 and np.all(np.isfinite(speeds)) and np.all(speeds == 0.):
      valid[:] = True
    dt = DT_CTRL
    transition = np.array([[1., dt], [0., 1.]])
    state = np.asarray(self.v_ego_kf.x, dtype=float).reshape(2)
    predicted_state = transition @ state
    predicted_state[0] = max(0., predicted_state[0])
    # Calibrated from route-data-new: high-speed aEgo jerk RMS is 2.35 m/s^3,
    # so acceleration uncertainty grows at the corresponding spectral rate.
    process_covariance = np.array([[0., 0.], [0., 5.5 * dt]])
    covariance = transition @ self.crv_speed_covariance @ transition.T + process_covariance
    if self.crv_speed_initialized:
      # Reject a positive but physically inconsistent channel using the
      # observer's current uncertainty and conservative single-wheel noise.
      # As covariance grows during dropout this gate widens accordingly.
      innovation_limit = 4. * np.sqrt(covariance[0, 0] + 1.2)
      valid &= np.abs(speeds - predicted_state[0]) <= innovation_limit
    count = int(np.count_nonzero(valid))
    self.crv_wheel_count = count
    self.crv_speed_measurement_valid = count > 0
    self.crv_high_speed_dropout_active = (self.crv_high_speed_dropout_active or
                                          (count < 2 and predicted_speed >= CRV_SPEED_DROPOUT_MIN_SPEED))
    self.crv_speed_dropout_time = (self.crv_speed_dropout_time + dt
                                   if self.crv_high_speed_dropout_active else 0.)
    self.crv_speed_dropout_time *= float(count < 2)
    self.crv_high_speed_dropout_active = self.crv_high_speed_dropout_active and count < 2
    self.crv_speed_dropout_timeout = self.crv_speed_dropout_time >= CRV_SPEED_DROPOUT_TAKEOVER_TIME

    raw_measurement = predicted_state[0]
    if count:
      wheel_measurement = float(np.median(speeds[valid]))
      scaled_measurement = self._apply_crv_speed_scale(wheel_measurement, cluster_speed)
      scale = scaled_measurement / wheel_measurement if wheel_measurement > 0. else 1.
      scatter = float(np.var(speeds[valid])) if count > 1 else 0.
      # Wheel-mean residual against carState.vEgo is 0.014 m/s RMS above
      # 5 m/s on route-data-new. Model the corresponding per-channel noise
      # and increase variance for fewer channels and wheel disagreement.
      measurement_variance = (0.0008 / count + scatter / count) * scale ** 2
      if not self.crv_speed_initialized:
        state = np.array([scaled_measurement, 0.])
        covariance = np.diag([measurement_variance, 1.])
        self.crv_speed_initialized = True
      else:
        innovation_variance = covariance[0, 0] + measurement_variance
        gain = covariance[:, 0] / innovation_variance
        innovation = scaled_measurement - predicted_state[0]
        state = predicted_state + gain * innovation
        observation = np.array([1., 0.])
        residual = np.eye(2) - np.outer(gain, observation)
        covariance = (residual @ covariance @ residual.T
                      + measurement_variance * np.outer(gain, gain))
      raw_measurement = scaled_measurement
    else:
      # No wheel measurement: use the acceleration state only for prediction.
      # Covariance continues to grow until valid wheel data returns.
      state = predicted_state

    self.crv_speed_covariance = covariance
    self.v_ego_kf.set_x([[float(state[0])], [float(state[1])]])
    return (float(state[0]), float(state[1]), float(raw_measurement),
            float(np.sqrt(max(covariance[0, 0], 0.))), float(np.sqrt(max(covariance[1, 1], 0.))))

  def update(self, can_parsers) -> tuple[structs.CarState, structs.CarStateSP]:
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]
    if self.CP.enableBsm:
      cp_body = can_parsers[Bus.body]

    ret = structs.CarState()
    ret_sp = structs.CarStateSP()

    # update prevs, update must run once per loop
    prev_cruise_buttons = self.cruise_buttons
    prev_cruise_setting = self.cruise_setting
    self.cruise_setting = cp.vl["SCM_BUTTONS"]["CRUISE_SETTING"]
    self.cruise_buttons = cp.vl["SCM_BUTTONS"]["CRUISE_BUTTONS"]

    # used for car hud message
    # TODO: find CAR_SPEED for HONDA_ODYSSEY_TWN or use ACC_HUD w/ detection
    self.is_metric = self.CP.carFingerprint in (CAR.HONDA_ODYSSEY_TWN,) or not cp.vl["CAR_SPEED"]["IMPERIAL_UNIT"]
    self.v_cruise_factor = CV.MPH_TO_MS if self.dynamic_v_cruise_units and not self.is_metric else CV.KPH_TO_MS

    # ******************* parse out can *******************

    # CR-V wheel channels that report zero while motion is predicted are
    # censored, not zero-speed measurements. Other Hondas keep their blend.
    wheel_speeds = [cp.vl["WHEEL_SPEEDS"][f"WHEEL_SPEED_{s}"] * CV.KPH_TO_MS for s in ("FL", "FR", "RL", "RR")]
    v_transmission = cp.vl["ENGINE_DATA"]["XMISSION_SPEED"] * CV.KPH_TO_MS
    cluster_speed = self._get_cluster_speed(cp)
    if self.crv_speed_scale is not None:
      ret.vEgo, ret.aEgo, ret.vEgoRaw, ret.vEgoStd, ret.aEgoStd = self.update_crv_speed_observer(wheel_speeds, cluster_speed)
      ret.vEgoMeasurementValid = self.crv_speed_measurement_valid
      ret.vEgoWheelCount = self.crv_wheel_count
      ret.vEgoDropoutTime = self.crv_speed_dropout_time
      ret.vehicleSensorsInvalid = self.crv_speed_dropout_timeout
      ret.standstill = (v_transmission < 1e-5 and np.all(np.asarray(wheel_speeds) == 0.) and ret.vEgo < 0.05)
    else:
      v_wheel = sum(wheel_speeds) / 4.0
      unscaled_v_ego = self.get_raw_speed(v_wheel, v_transmission)
      ret.vEgoRaw = unscaled_v_ego
      ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
      ret.vEgoStd = 0.
      ret.aEgoStd = 0.
      ret.vEgoMeasurementValid = True
      ret.vEgoWheelCount = 4
      ret.vEgoDropoutTime = 0.
      ret.standstill = cp.vl["ENGINE_DATA"]["XMISSION_SPEED"] < 1e-5

    # doorOpen is true if we can find any door open, but signal locations vary, and we may only see the driver's door
    # TODO: Test the eight Nidec cars without SCM signals for driver's door state, may be able to consolidate further
    if self.CP.flags & HondaFlags.HAS_ALL_DOOR_STATES:
      ret.doorOpen = any([cp.vl["DOORS_STATUS"]["DOOR_OPEN_FL"], cp.vl["DOORS_STATUS"]["DOOR_OPEN_FR"],
                          cp.vl["DOORS_STATUS"]["DOOR_OPEN_RL"], cp.vl["DOORS_STATUS"]["DOOR_OPEN_RR"]])
    elif "DRIVERS_DOOR_OPEN" in cp.vl["SCM_BUTTONS"]:
      ret.doorOpen = bool(cp.vl["SCM_BUTTONS"]["DRIVERS_DOOR_OPEN"])
    else:
      ret.doorOpen = bool(cp.vl["SCM_FEEDBACK"]["DRIVERS_DOOR_OPEN"])

    ret.seatbeltUnlatched = bool(cp.vl["SEATBELT_STATUS"]["SEATBELT_DRIVER_LAMP"] or not cp.vl["SEATBELT_STATUS"]["SEATBELT_DRIVER_LATCHED"])

    steer_status = self.steer_status_values[cp.vl["STEER_STATUS"]["STEER_STATUS"]]
    ret.steerFaultPermanent = steer_status not in ("NORMAL", "NO_TORQUE_ALERT_1", "NO_TORQUE_ALERT_2", "LOW_SPEED_LOCKOUT", "TMP_FAULT")
    if self.CP.flags & HondaFlags.BOSCH_ALT_RADAR:
      # TODO: See if this logic works for all other Honda
      min_steer_speed = max(CarControllerParams.STEER_GLOBAL_MIN_SPEED, self.CP.minSteerSpeed)
      expected_low_speed_lockout = steer_status == "LOW_SPEED_LOCKOUT" and ret.vEgo < min_steer_speed
      ret.steerFaultTemporary = steer_status != "NORMAL" and not expected_low_speed_lockout
    else:
      # LOW_SPEED_LOCKOUT is not worth a warning
      # NO_TORQUE_ALERT_2 can be caused by bump or steering nudge from driver
      # FIXME: the stock camera stops steering on NO_TORQUE_ALERT_1
      ret.steerFaultTemporary = steer_status not in ("NORMAL", "LOW_SPEED_LOCKOUT", "NO_TORQUE_ALERT_2")

    if (self.CP.carFingerprint == CAR.ACURA_MDX_4G) and (steer_status == "TJA_LOW_SPEED_LOCKOUT"):
      ret.steerFaultPermanent = False
      ret.steerFaultTemporary = False

    # All Honda EPS cut off slightly above standstill, some much higher
    # Don't alert in the near-standstill range, but alert for per-vehicle configured minimums above that
    if CarControllerParams.STEER_GLOBAL_MIN_SPEED < ret.vEgo < (self.CP.minSteerSpeed + 0.5):
      self.low_speed_alert = True
    elif ret.vEgo > (self.CP.minSteerSpeed + 1.):
      # TODO: better handle delayed steering enablement on ALT_RADAR cars
      self.low_speed_alert = False
    ret.lowSpeedAlert = self.low_speed_alert

    if self.CP.flags & HondaFlags.BOSCH_RADARLESS:
      ret.accFaulted = bool(cp.vl["CRUISE_FAULT_STATUS"]["CRUISE_FAULT"])
    else:
      if self.CP.openpilotLongitudinalControl:
        if (self.CP.carFingerprint == CAR.ACURA_MDX_4G) and (self.CP.flags & HondaFlags.BOSCH_ALT_BRAKE):
          ret.accFaulted = bool(cp.vl["BRAKE_MODULE"]["CRUISE_FAULT"])
        else:
          ret.accFaulted = bool(cp.vl[self.brake_error_msg]["BRAKE_ERROR_1"] or cp.vl[self.brake_error_msg]["BRAKE_ERROR_2"])

      # Log non-critical stock ACC/LKAS faults if Nidec (camera)
      if not (self.CP.flags & HondaFlags.BOSCH):
        ret.carFaultedNonCritical = bool(cp_cam.vl["ACC_HUD"]["ACC_PROBLEM"] or cp_cam.vl["LKAS_HUD"]["LKAS_PROBLEM"])

    ret.espDisabled = cp.vl["VSA_STATUS"]["ESP_DISABLED"] != 0

    if self.dash_speed_seen:
      ret.vEgoCluster = cluster_speed

    ret.steeringAngleDeg = cp.vl["STEERING_SENSORS"]["STEER_ANGLE"]
    ret.steeringRateDeg = cp.vl["STEERING_SENSORS"]["STEER_ANGLE_RATE"]

    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_stalk(
      250, cp.vl["SCM_FEEDBACK"]["LEFT_BLINKER"], cp.vl["SCM_FEEDBACK"]["RIGHT_BLINKER"])
    ret.brakeHoldActive = cp.vl["VSA_STATUS"]["BRAKE_HOLD_ACTIVE"] == 1
    ret.parkingBrake = bool(cp.vl[self.car_state_scm_msg]["PARKING_BRAKE_ON"])

    if self.CP.transmissionType == TransmissionType.manual:
      ret.gearShifter = GearShifter.reverse if bool(cp.vl[self.car_state_scm_msg]["REVERSE_LIGHT"]) else GearShifter.drive
    else:
      gear_position = self.shifter_values.get(cp.vl[self.gearbox_msg]["GEAR_SHIFTER"], None)
      ret.gearShifter = self.parse_gear_shifter(gear_position)

    ret.gasPressed = cp.vl["POWERTRAIN_DATA"]["PEDAL_GAS"] > 1e-5

    ret.steeringTorque = cp.vl["STEER_STATUS"]["STEER_TORQUE_SENSOR"]
    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD.get(self.CP.carFingerprint, 1200)

    if self.CP.flags & HondaFlags.BOSCH:
      # The PCM always manages its own cruise control state, but doesn't publish it
      if self.CP.flags & HondaFlags.BOSCH_RADARLESS:
        ret.cruiseState.nonAdaptive = cp_cam.vl["ACC_HUD"]["CRUISE_CONTROL_LABEL"] != 0

      if not self.CP.openpilotLongitudinalControl:
        # ACC_HUD is on camera bus on radarless cars
        acc_hud = cp_cam.vl["ACC_HUD"] if self.CP.flags & HondaFlags.BOSCH_RADARLESS else cp.vl["ACC_HUD"]
        ret.cruiseState.nonAdaptive = acc_hud["CRUISE_CONTROL_LABEL"] != 0
        ret.cruiseState.standstill = acc_hud["CRUISE_SPEED"] == 252.

        # On set, cruise set speed pulses between 254~255 and the set speed prev is set to avoid this.
        ret.cruiseState.speed = self.v_cruise_pcm_prev if acc_hud["CRUISE_SPEED"] > 160.0 else acc_hud["CRUISE_SPEED"] * self.v_cruise_factor
        self.v_cruise_pcm_prev = ret.cruiseState.speed
    else:
      ret.cruiseState.speed = cp.vl["CRUISE"]["CRUISE_SPEED_PCM"] * CV.KPH_TO_MS

    if self.CP.flags & HondaFlags.BOSCH_ALT_BRAKE:
      ret.brakePressed = cp.vl["BRAKE_MODULE"]["BRAKE_PRESSED"] != 0
    else:
      # brake switch has shown some single time step noise, so only considered when
      # switch is on for at least 2 consecutive CAN samples
      # brake switch rises earlier than brake pressed but is never 1 when in park
      brake_switch_vals = cp.vl_all["POWERTRAIN_DATA"]["BRAKE_SWITCH"]
      if len(brake_switch_vals):
        brake_switch = cp.vl["POWERTRAIN_DATA"]["BRAKE_SWITCH"] != 0
        if len(brake_switch_vals) > 1:
          self.brake_switch_prev = brake_switch_vals[-2] != 0
        self.brake_switch_active = brake_switch and self.brake_switch_prev
        self.brake_switch_prev = brake_switch
      ret.brakePressed = (cp.vl["POWERTRAIN_DATA"]["BRAKE_PRESSED"] != 0) or self.brake_switch_active

    ret.deprecated.brake = cp.vl["VSA_STATUS"]["USER_BRAKE"]
    ret.cruiseState.enabled = cp.vl["POWERTRAIN_DATA"]["ACC_STATUS"] != 0
    ret.cruiseState.available = bool(cp.vl[self.car_state_scm_msg]["MAIN_ON"])

    # Gets rid of Pedal Grinding noise when brake is pressed at slow speeds for some models
    if self.CP.carFingerprint in (CAR.HONDA_PILOT, CAR.HONDA_RIDGELINE):
      if ret.deprecated.brake > 0.1:
        ret.brakePressed = True

    if self.CP.flags & HondaFlags.BOSCH:
      # TODO: find the radarless AEB_STATUS bit and make sure ACCEL_COMMAND is correct to enable AEB alerts
      if not (self.CP.flags & HondaFlags.BOSCH_RADARLESS):
        ret.stockAeb = (not self.CP.openpilotLongitudinalControl) and bool(cp.vl["ACC_CONTROL"]["AEB_STATUS"] and cp.vl["ACC_CONTROL"]["ACCEL_COMMAND"] < -1e-5)
    else:
      ret.stockAeb = bool(cp_cam.vl["BRAKE_COMMAND"]["AEB_REQ_1"] and cp_cam.vl["BRAKE_COMMAND"]["COMPUTER_BRAKE"] > 1e-5)

    self.acc_hud = False
    self.lkas_hud = False
    if not (self.CP.flags & HondaFlags.BOSCH):
      ret.stockFcw = cp_cam.vl["BRAKE_COMMAND"]["FCW"] != 0
      self.acc_hud = cp_cam.vl["ACC_HUD"]
      self.stock_brake = cp_cam.vl["BRAKE_COMMAND"]
    if self.CP.flags & HondaFlags.BOSCH_RADARLESS:
      self.lkas_hud = cp_cam.vl["LKAS_HUD"]

    if self.CP.enableBsm:
      # BSM messages are on B-CAN, requires a panda forwarding B-CAN messages to CAN 0
      # more info here: https://github.com/commaai/openpilot/pull/1867
      ret.leftBlindspot = cp_body.vl["BSM_STATUS_LEFT"]["BSM_ALERT"] == 1
      ret.rightBlindspot = cp_body.vl["BSM_STATUS_RIGHT"]["BSM_ALERT"] == 1

    ret.buttonEvents = [
      *create_button_events(self.cruise_buttons, prev_cruise_buttons, BUTTONS_DICT),
      *create_button_events(self.cruise_setting, prev_cruise_setting, SETTINGS_BUTTONS_DICT),
    ]

    CarStateExt.update(self, ret, can_parsers)

    return ret, ret_sp

  def get_can_parsers(self, CP, CP_SP):
    parsers = {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], [], CanBus(CP).pt),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], [], CanBus(CP).camera),
    }
    if CP.enableBsm:
      parsers[Bus.body] = CANParser(DBC[CP.carFingerprint][Bus.body], [], CanBus(CP).radar)

    return parsers
