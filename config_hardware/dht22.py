from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict

logger = logging.getLogger("envsense.dht22")


@dataclass
class DHT22Config:
    pin: int = 21
    pin_candidates: tuple[int, ...] | None = None


class DHT22Reader:
    """
    Read DHT22 temperature and humidity.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
    ) -> None:
        # Use BCM GPIO numbering (same convention as workshop example).
        configured_pin = int(cfg.get("pin", 21))
        self.pin_candidates = self._resolve_pin_candidates(cfg, default_pin=configured_pin)
        self.pin = self.pin_candidates[0]
        self.max_retries = max(1, int(cfg.get("max_retries", 3)))
        self.retry_delay_sec = max(0.0, float(cfg.get("retry_delay_sec", 1.0)))
        self.warn_after_consecutive_failures = max(1, int(cfg.get("warn_after_consecutive_failures", 3)))
        self._method = ""
        self._adafruit_dht = None
        self._board = None
        self._adafruit_dht_legacy = None
        self._adafruit_sensors: Dict[int, Any] = {}
        self._active_pin: int | None = None
        self._consecutive_failures = 0
        self._last_good_reading: Dict[str, float] | None = None

        self._initialize_backend()

    @staticmethod
    def _resolve_pin_candidates(cfg: Dict[str, Any], *, default_pin: int) -> list[int]:
        raw_candidates = cfg.get("pin_candidates")
        if raw_candidates is None:
            return [default_pin]

        if not isinstance(raw_candidates, (list, tuple)):
            raw_candidates = [raw_candidates]

        candidates: list[int] = []
        for raw_candidate in raw_candidates:
            try:
                candidate = int(raw_candidate)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid DHT22 pin candidate: %r", raw_candidate)
                continue

            if candidate not in candidates:
                candidates.append(candidate)

        return candidates or [default_pin]

    def _initialize_backend(self) -> None:
        # Prefer CircuitPython `adafruit_dht` first (workshop style), then
        # fallback to legacy `Adafruit_DHT`.
        try:
            import adafruit_dht  # type: ignore
            import board  # type: ignore

            self._adafruit_dht = adafruit_dht
            self._board = board

            if any(self._board_pin_for(pin) is not None for pin in self.pin_candidates):
                self._method = "adafruit_dht"
                return
        except Exception:
            self._adafruit_dht = None
            self._board = None

        try:
            import Adafruit_DHT  # type: ignore

            self._adafruit_dht_legacy = Adafruit_DHT
            self._method = "Adafruit_DHT"
        except Exception as exc:
            raise RuntimeError(
                "DHT22 requires `adafruit-circuitpython-dht` (preferred) "
                "or legacy `Adafruit_DHT`. Please install dependencies."
            ) from exc

    def _board_pin_for(self, pin: int) -> Any:
        if self._board is None:
            return None
        return getattr(self._board, f"D{pin}", None)

    def _pin_attempt_order(self) -> list[int]:
        if self._active_pin is None or self._active_pin not in self.pin_candidates:
            return list(self.pin_candidates)
        return [self._active_pin] + [pin for pin in self.pin_candidates if pin != self._active_pin]

    def _close_adafruit_sensor(self, pin: int) -> None:
        sensor = self._adafruit_sensors.pop(pin, None)
        if sensor is None:
            return
        try:
            sensor.exit()
        except Exception:
            pass

    def _remember_active_pin(self, pin: int) -> None:
        if self._active_pin == pin:
            return
        logger.info("DHT22 now using BCM pin %d", pin)
        self._active_pin = pin

    def _read_with_adafruit_dht(self) -> Dict[str, float]:
        if self._adafruit_dht is None:
            raise RuntimeError("DHT22 adafruit_dht backend not initialized.")

        errors: list[str] = []
        last_exc: Exception | None = None

        for pin in self._pin_attempt_order():
            board_pin = self._board_pin_for(pin)
            if board_pin is None:
                errors.append(f"BCM {pin}: board pin mapping unavailable")
                continue

            try:
                sensor = self._adafruit_sensors.get(pin)
                if sensor is None:
                    sensor = self._adafruit_dht.DHT22(board_pin, use_pulseio=False)
                    self._adafruit_sensors[pin] = sensor

                temp = getattr(sensor, "temperature", None)
                humidity = getattr(sensor, "humidity", None)
                if temp is None or humidity is None:
                    raise RuntimeError(
                        f"DHT22 returned empty reading on BCM pin {pin}. "
                        "Check wiring, power, pull-up resistor, and GPIO pin."
                    )

                reading = {"temp_C": float(temp), "humidity": float(humidity)}
                self._remember_active_pin(pin)
                return reading
            except Exception as exc:
                last_exc = exc
                errors.append(f"BCM {pin}: {exc}")
                self._close_adafruit_sensor(pin)

        raise RuntimeError(
            f"DHT22 read failed on candidate BCM pins {self.pin_candidates}. "
            + "; ".join(errors)
        ) from last_exc

    def _read_with_legacy_dht(self) -> Dict[str, float]:
        if self._adafruit_dht_legacy is None:
            raise RuntimeError("DHT22 Adafruit_DHT backend not initialized.")

        errors: list[str] = []
        last_exc: Exception | None = None

        for pin in self._pin_attempt_order():
            try:
                humidity, temp = self._adafruit_dht_legacy.read_retry(self._adafruit_dht_legacy.DHT22, pin)
                if temp is None or humidity is None:
                    raise RuntimeError(
                        f"DHT22 returned empty reading on BCM pin {pin}. "
                        "Check wiring, power, pull-up resistor, and GPIO pin."
                    )

                reading = {"temp_C": float(temp), "humidity": float(humidity)}
                self._remember_active_pin(pin)
                return reading
            except Exception as exc:
                last_exc = exc
                errors.append(f"BCM {pin}: {exc}")

        raise RuntimeError(
            f"DHT22 read failed on candidate BCM pins {self.pin_candidates}. "
            + "; ".join(errors)
        ) from last_exc

    def read(self) -> Dict[str, float]:
        """
        Returns:
          { "temp_C": float, "humidity": float }
        """
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if self._method == "adafruit_dht":
                    reading = self._read_with_adafruit_dht()
                else:
                    reading = self._read_with_legacy_dht()
                self._consecutive_failures = 0
                self._last_good_reading = reading
                return reading
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_sec)

        self._consecutive_failures += 1
        if self._consecutive_failures >= self.warn_after_consecutive_failures:
            logger.warning(
                (
                    "DHT22 read failed %d consecutive times. "
                    "Using fallback reading and continuing."
                ),
                self._consecutive_failures,
                exc_info=last_exc,
            )

        if self._last_good_reading is not None:
            return self._last_good_reading

        return {"temp_C": float("nan"), "humidity": float("nan")}

