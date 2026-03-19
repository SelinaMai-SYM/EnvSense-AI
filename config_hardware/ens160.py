from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ENS160Config:
    i2c_bus: int = 1
    address: int = 0x53


class ENS160Reader:
    """
    Read ENS160 eCO2 and TVOC over I2C.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
    ) -> None:
        self.i2c_bus = int(cfg.get("i2c_bus", 1))
        # Config may specify 0x53 style address in YAML; Python YAML loads ints correctly
        self.address = int(cfg.get("address", 0x53))

        self._adafruit_ens160 = None
        self._i2c_device = None

        try:
            import board  # type: ignore
            import busio  # type: ignore
            import adafruit_ens160  # type: ignore

            i2c = busio.I2C(board.SCL, board.SDA)
            try:
                self._adafruit_ens160 = adafruit_ens160.ENS160(i2c, addr=self.address)  # type: ignore[arg-type]
            except TypeError:
                self._adafruit_ens160 = adafruit_ens160.ENS160(i2c)  # type: ignore[call-arg]
        except Exception as exc:
            raise RuntimeError(
                "ENS160 real-hardware mode requires I2C + adafruit_ens160 libraries."
            ) from exc

    def read(self) -> Dict[str, float]:
        """
        Returns:
          { "eco2_ppm": float, "tvoc": float }
        """

        try:
            # Adafruit library exposes eco2 and tvoc values in ppm / ppb.
            eco2 = getattr(self._adafruit_ens160, "eco2", None)
            tvoc = getattr(self._adafruit_ens160, "tvoc", None)
            if eco2 is None or tvoc is None:
                raise RuntimeError("ENS160 returned empty reading. Check I2C wiring/address.")
            return {"eco2_ppm": float(eco2), "tvoc": float(tvoc)}
        except Exception as exc:
            raise RuntimeError("Failed to read ENS160 sensor.") from exc

