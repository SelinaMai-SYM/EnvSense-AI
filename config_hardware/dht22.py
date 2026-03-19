from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DHT22Config:
    pin: int = 4


class DHT22Reader:
    """
    Read DHT22 temperature and humidity.

    If hardware libraries are unavailable or mock_mode=True, this falls back
    to a realistic synthetic signal with drift and noise.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        mock_mode: bool,
        interval_sec: int = 10,
        seed: Optional[int] = 1234,
    ) -> None:
        self.mock_mode = mock_mode
        self.interval_sec = interval_sec
        self._rng = random.Random(seed)

        self.pin = int(cfg.get("pin", 4))
        self._step = 0

        # Internal state for mock drift
        self._temp_C = 23.5 + self._rng.uniform(-1.0, 1.0)
        self._humidity = 45.0 + self._rng.uniform(-5.0, 5.0)

        self._adafruit_dht = None
        if not mock_mode:
            try:
                import Adafruit_DHT  # type: ignore

                self._adafruit_dht = Adafruit_DHT
            except Exception:
                self._adafruit_dht = None
                self.mock_mode = True

    def _read_mock(self) -> Dict[str, float]:
        self._step += 1

        # Slow drift + daily-like oscillation
        t = self._step
        drift = 0.015 * math.sin(2 * math.pi * t / 600.0)  # ~ every 100 minutes (if interval=10s)
        daily = 0.6 * math.sin(2 * math.pi * t / 360.0)  # ~ every 60 minutes

        # Add a bit of occupancy/heat effect by linking temp to humidity changes
        noise_t = self._rng.gauss(0.0, 0.08)
        noise_h = self._rng.gauss(0.0, 0.6)

        # Anti-correlation: when temp rises, humidity tends to drop slightly.
        temp_target = 23.0 + daily + drift
        humidity_target = 48.0 - 1.2 * (temp_target - 23.0) + 6.0 * math.sin(2 * math.pi * t / 420.0)

        self._temp_C += 0.08 * (temp_target - self._temp_C) + noise_t
        self._humidity += 0.06 * (humidity_target - self._humidity) + noise_h

        # Clamp to plausible ranges
        self._temp_C = float(max(15.0, min(35.0, self._temp_C)))
        self._humidity = float(max(20.0, min(90.0, self._humidity)))

        return {"temp_C": float(self._temp_C), "humidity": float(self._humidity)}

    def read(self) -> Dict[str, float]:
        """
        Returns:
          { "temp_C": float, "humidity": float }
        """

        if self.mock_mode or self._adafruit_dht is None:
            return self._read_mock()

        # Real hardware path
        try:
            humidity, temp = self._adafruit_dht.read_retry(self._adafruit_dht.DHT22, self.pin)
            if temp is None or humidity is None:
                # Sensor temporarily disconnected or unstable reading: fall back to mock.
                return self._read_mock()

            return {"temp_C": float(temp), "humidity": float(humidity)}
        except Exception:
            # Fail gracefully in case the device or library is not accessible
            self.mock_mode = True
            return self._read_mock()

