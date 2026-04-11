from __future__ import annotations

import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class OLEDDisplay:
    """
    Optional SSD1306 OLED display.

    If OLED libs/hardware are unavailable, initialization is skipped and `show()` becomes a no-op.
    """

    def __init__(self, cfg: Dict[str, Any], *, enabled: bool) -> None:
        self.enabled = bool(enabled)

        self._display: Optional[Any] = None
        self._width = 128
        self._height = 64

        if not self.enabled:
            return

        try:
            import board  # type: ignore
            import busio  # type: ignore
            import adafruit_ssd1306  # type: ignore

            i2c = busio.I2C(board.SCL, board.SDA)
            # Many adafruit_ssd1306 instances accept reset pin; we keep defaults.
            self._display = adafruit_ssd1306.SSD1306_I2C(self._width, self._height, i2c, addr=cfg.get("address", 0x3C))
            self._display.fill(0)
            self._display.show()
        except Exception as e:
            logger.warning("OLED unavailable, falling back gracefully: %s", e)
            self.enabled = False
            self._display = None

    def show(self, values: Dict[str, float], mode_result_text: str) -> None:
        """
        Show current values plus a mode-specific recommendation line.

        Args:
          values: {"temp_C":..., "humidity":..., "eco2_ppm":..., "tvoc":...}
          mode_result_text: short text (1-2 lines) shown on screen.
        """
        if not self.enabled or self._display is None:
            return

        try:
            self._display.fill(0)

            # Use built-in PIL drawing if available.
            from PIL import Image, ImageDraw, ImageFont  # type: ignore

            image = Image.new("1", (self._width, self._height))
            draw = ImageDraw.Draw(image)

            text = [
                f"T {values.get('temp_C', 0):.1f}C  H {values.get('humidity', 0):.0f}%",
                f"eCO2 {values.get('eco2_ppm', 0):.0f}  TVOC {values.get('tvoc', 0):.0f}",
                mode_result_text[:24],
            ]

            y = 0
            for line in text:
                draw.text((0, y), line, fill=255)
                y += 16

            self._display.image(image)
            self._display.show()
        except Exception:
            # Fail gracefully; never crash the main logger loop
            return

