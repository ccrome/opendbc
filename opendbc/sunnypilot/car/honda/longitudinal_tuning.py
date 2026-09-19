"""Fixed longitudinal configuration for the fifth-generation Honda CR-V."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HondaCrvLongitudinalTune:
  tune_id: str = "safeV1"
  revision: int = 1
  actuator_delay: float = 0.8
  ki_bp: tuple[float, ...] = (0.0, 5.0, 35.0)
  ki_v: tuple[float, ...] = (0.0, 0.0, 0.0)
  following_time: float = 1.75
  brake_entry_accel: float = -0.15
  brake_release_accel: float = -0.05
  gas_entry_accel: float = 0.05
  low_speed_max: float = 5.0
  launch_accel_max: float = 0.8
  close_lead_distance: float = 6.0
  full_launch_distance: float = 10.0
  closing_v_rel: float = -1.0
  closing_time_gap: float = 2.0
  guard_release_time: float = 0.5
  stop_release_v_rel: float = 0.2
  lead_probability: float = 0.5


CRV_LONGITUDINAL_TUNE = HondaCrvLongitudinalTune()
