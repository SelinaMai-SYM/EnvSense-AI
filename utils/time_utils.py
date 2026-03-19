from __future__ import annotations

from datetime import datetime, timezone


def now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

