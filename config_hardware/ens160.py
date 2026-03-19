from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ENS160Config:
    i2c_bus: int = 1
    address: int = 0x53


class ENS160Reader:
    """
    Read ENS160 eCO2 and TVOC over I2C.

    If hardware libraries are unavailable or mock_mode=True, this falls back
    to a realistic synthetic signal with drift and occasional spikes.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        mock_mode: bool,
        interval_sec: int = 10,
        seed: Optional[int] = 4321,
    ) -> None:
        self.mock_mode = mock_mode
        self.interval_sec = interval_sec
        self._rng = random.Random(seed)
        self._step = 0

        self.i2c_bus = int(cfg.get("i2c_bus", 1))
        # Config may specify 0x53 style address in YAML; Python YAML loads ints correctly
        self.address = int(cfg.get("address", 0x53))

        self._adafruit_ens160 = None
        self._i2c_device = None

        if not mock_mode:
            try:
                import board  # type: ignore
                import busio  # type: ignore

                # The Adafruit library tries to probe the given address.
                import adafruit_ens160  # type: ignore

                i2c = busio.I2C(board.SCL, board.SDA)
                # Some versions accept `addr` kwarg; if not, the constructor will raise
                try:
                    self._adafruit_ens160 = adafruit_ens160.ENS160(i2c, addr=self.address)  # type: ignore[arg-type]
                except TypeError:
                    self._adafruit_ens160 = adafruit_ens160.ENS160(i2c)  # type: ignore[call-arg]

            except Exception:
                self._adafruit_ens160 = None
                self.mock_mode = True

    def _read_mock(self) -> Dict[str, float]:
        self._step += 1
        t = self._step

        # Occupancy-like signal (0..1) with ~60 minutes period (if interval=10s => 360 steps)
        occ = 0.5 * (1.0 + math.sin(2 * math.pi * t / 360.0))

        baseline_eco2 = 700.0 + 70.0 * math.sin(2 * math.pi * t / 900.0)
        target_eco2 = baseline_eco2 + 500.0 * occ

        # Relax towards the target + small noise
        noise = self._rng.gauss(0.0, 12.0)
        if not hasattr(self, "_eco2_ppm"):
            self._eco2_ppm = 850.0
        self._eco2_ppm += 0.07 * (target_eco2 - self._eco2_ppm) + noise

        # Occasional "burst" during high occupancy
        if occ > 0.72 and self._rng.random() < 0.03:
            self._eco2_ppm += self._rng.uniform(80.0, 180.0)

        # TVOC tends to follow occupancy and eco2 excursions
        baseline_tvoc = 120.0 + 25.0 * math.sin(2 * math.pi * t / 500.0)
        tvoc_target = baseline_tvoc + 220.0 * occ + max(0.0, self._eco2_ppm - 900.0) * 0.08

        if not hasattr(self, "_tvoc"):
            self._tvoc = 160.0
        self._tvoc += 0.09 * (tvoc_target - self._tvoc) + self._rng.gauss(0.0, 4.5)

        if self._rng.random() < 0.012:
            self._tvoc += self._rng.uniform(80.0, 160.0)

        self._eco2_ppm = float(max(400.0, min(2200.0, self._eco2_ppm)))
        self._tvoc = float(max(20.0, min(1200.0, self._tvoc)))

        return {"eco2_ppm": float(self._eco2_ppm), "tvoc": float(self._tvoc)}

    def read(self) -> Dict[str, float]:
        """
        Returns:
          { "eco2_ppm": float, "tvoc": float }
        """

        if self.mock_mode or self._adafruit_ens160 is None:
            return self._read_mock()

        try:
            # Adafruit library exposes eco2 and tvoc values in ppm / ppb.
            eco2 = getattr(self._adafruit_ens160, "eco2", None)
            tvoc = getattr(self._adafruit_ens160, "tvoc", None)
            if eco2 is None or tvoc is None:
                return self._read_mock()
            return {"eco2_ppm": float(eco2), "tvoc": float(tvoc)}
        except Exception:
            self.mock_mode = True
            return self._read_mock()

