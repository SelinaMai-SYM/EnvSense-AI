from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DHT22Config:
    pin: int = 4


class DHT22Reader:
    """
    Read DHT22 temperature and humidity.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
    ) -> None:
        self.pin = int(cfg.get("pin", 4))
        try:
            import Adafruit_DHT  # type: ignore

            self._adafruit_dht = Adafruit_DHT
        except Exception as exc:
            raise RuntimeError(
                "DHT22 real-hardware mode requires Adafruit_DHT. Install dependency and check environment."
            ) from exc

    def read(self) -> Dict[str, float]:
        """
        Returns:
          { "temp_C": float, "humidity": float }
        """

        try:
            humidity, temp = self._adafruit_dht.read_retry(self._adafruit_dht.DHT22, self.pin)
            if temp is None or humidity is None:
                raise RuntimeError("DHT22 returned empty reading. Check wiring/power/pin configuration.")

            return {"temp_C": float(temp), "humidity": float(humidity)}
        except Exception as exc:
            raise RuntimeError("Failed to read DHT22 sensor.") from exc

