from __future__ import annotations

from typing import Iterable, List, Optional


def ewma(values: Iterable[float], alpha: float = 0.3) -> List[float]:
    """Exponentially-weighted moving average for plotting/robustness."""
    out: List[float] = []
    last: Optional[float] = None
    for v in values:
        if last is None:
            last = float(v)
        else:
            last = alpha * float(v) + (1 - alpha) * last
        out.append(last)
    return out

