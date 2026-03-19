from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DHT22Config:
    pin: int = 21


class DHT22Reader:
    """
    Read DHT22 temperature and humidity.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
    ) -> None:
        # Use BCM GPIO numbering (same convention as workshop example).
        self.pin = int(cfg.get("pin", 21))
        self._method = ""
        self._sensor = None
        self._adafruit_dht_legacy = None

        # Prefer CircuitPython `adafruit_dht` first (workshop style), then
        # fallback to legacy `Adafruit_DHT`.
        try:
            import adafruit_dht  # type: ignore
            import board  # type: ignore

            board_pin_name = f"D{self.pin}"
            board_pin = getattr(board, board_pin_name, None)
            if board_pin is None:
                raise RuntimeError(
                    f"GPIO pin mapping failed for {board_pin_name}. "
                    "Use BCM pin numbers that exist in `board` module."
                )

            self._sensor = adafruit_dht.DHT22(board_pin, use_pulseio=False)
            self._method = "adafruit_dht"
        except Exception:
            try:
                import Adafruit_DHT  # type: ignore

                self._adafruit_dht_legacy = Adafruit_DHT
                self._sensor = (Adafruit_DHT.DHT22, self.pin)
                self._method = "Adafruit_DHT"
            except Exception as exc:
                raise RuntimeError(
                    "DHT22 requires `adafruit-circuitpython-dht` (preferred) "
                    "or legacy `Adafruit_DHT`. Please install dependencies."
                ) from exc

    def read(self) -> Dict[str, float]:
        """
        Returns:
          { "temp_C": float, "humidity": float }
        """

        try:
            if self._method == "adafruit_dht":
                temp = getattr(self._sensor, "temperature", None)
                humidity = getattr(self._sensor, "humidity", None)
            else:
                if self._adafruit_dht_legacy is None or self._sensor is None:
                    raise RuntimeError("DHT22 backend not initialized.")
                dht_model, pin = self._sensor
                humidity, temp = self._adafruit_dht_legacy.read_retry(dht_model, pin)

            if temp is None or humidity is None:
                raise RuntimeError(
                    "DHT22 returned empty reading. Check wiring, power, pull-up resistor, and GPIO pin."
                )

            return {"temp_C": float(temp), "humidity": float(humidity)}
        except Exception as exc:
            raise RuntimeError("Failed to read DHT22 sensor.") from exc

