"""Slow CR-V wheel-speed calibration against the Honda cluster indication."""

import math


PARAM_KEY = "HondaCrvSpeedScale"
DEFAULT_SCALE = 1.0
MIN_SCALE = 0.98
MAX_SCALE = 1.06
MIN_SPEED = 20.0 / 2.236936  # 20 mph, m/s
MAX_SPEED = 45.0  # m/s
BATCH_DURATION = 30.0  # seconds of eligible driving
TIME_CONSTANT = 5.0 * 60.0  # seconds of eligible driving
PERSIST_INTERVAL = 5.0 * 60.0  # seconds of wall time
PERSIST_DELTA = 1e-4
APPLY_START_SPEED = 5.0  # m/s
APPLY_FULL_SPEED = 10.0  # m/s


def clamp_scale(value: float) -> float:
  return max(MIN_SCALE, min(MAX_SCALE, value))


def load_scale(params) -> float:
  value = params.get(PARAM_KEY, return_default=True)
  try:
    return clamp_scale(float(value))
  except (TypeError, ValueError):
    return DEFAULT_SCALE


def control_speed_scale(unscaled_speed: float, learned_scale: float) -> float:
  """Blend in the correction above low-speed stop-and-go operation."""
  learned_scale = clamp_scale(learned_scale)
  if unscaled_speed <= APPLY_START_SPEED:
    return 1.0
  if unscaled_speed >= APPLY_FULL_SPEED:
    return learned_scale
  fraction = (unscaled_speed - APPLY_START_SPEED) / (APPLY_FULL_SPEED - APPLY_START_SPEED)
  return 1.0 + fraction * (learned_scale - 1.0)


class CrvSpeedScaleEstimator:
  """Bounded, slowly adapting cluster-to-CAN speed scale estimator."""

  def __init__(self, initial_scale: float = DEFAULT_SCALE):
    self.scale = clamp_scale(initial_scale)
    self._sum_xx = 0.0
    self._sum_xy = 0.0
    self._duration = 0.0
    self._last_persist_time = 0.0
    self._last_persisted_scale = self.scale

  @staticmethod
  def eligible(unscaled_speed: float, cluster_speed: float) -> bool:
    return (math.isfinite(unscaled_speed) and math.isfinite(cluster_speed)
            and MIN_SPEED <= unscaled_speed <= MAX_SPEED
            and MIN_SPEED <= cluster_speed <= MAX_SPEED)

  def update(self, unscaled_speed: float, cluster_speed: float, dt: float) -> bool:
    if not self.eligible(unscaled_speed, cluster_speed):
      return False
    dt = max(0.0, min(float(dt), 0.1))
    if dt == 0.0:
      return False
    self._sum_xx += dt * unscaled_speed * unscaled_speed
    self._sum_xy += dt * unscaled_speed * cluster_speed
    self._duration += dt
    if self._duration < BATCH_DURATION:
      return False
    batch_duration = self._duration
    observed_scale = self._sum_xy / self._sum_xx
    self._sum_xx = self._sum_xy = self._duration = 0.0
    if not MIN_SCALE <= observed_scale <= MAX_SCALE:
      return False
    alpha = 1.0 - math.exp(-batch_duration / TIME_CONSTANT)
    self.scale = clamp_scale(self.scale + alpha * (observed_scale - self.scale))
    return True

  def should_persist(self, now: float) -> bool:
    return (now - self._last_persist_time >= PERSIST_INTERVAL
            and abs(self.scale - self._last_persisted_scale) >= PERSIST_DELTA)

  def mark_persisted(self, now: float) -> None:
    self._last_persist_time = now
    self._last_persisted_scale = self.scale


def persist_scale(params, estimator: CrvSpeedScaleEstimator, now: float) -> bool:
  """Persist a learned scale using the native type registered for the param."""
  if not estimator.should_persist(now):
    return False
  # HondaCrvSpeedScale is registered as FLOAT. Params.put deliberately does
  # not coerce strings, so passing a formatted string here crashes card.
  params.put(PARAM_KEY, float(estimator.scale))
  estimator.mark_persisted(now)
  return True
