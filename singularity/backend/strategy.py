"""
Strategy: EMA-momentum + breakout, tuned to fire often (medium frequency).

It enters when short-term momentum and a recent-range breakout agree, which
triggers far more often than a plain crossover — suited to many trades per
session. Exits are handled by the bracket's take-profit / stop, so the
risk:reward ratio is enforced on every trade.

HONEST NOTE: this is a reasonable, conventional momentum rule. It is NOT a
proven money-maker. Momentum breakouts whipsaw in choppy markets and frequent
trading multiplies commission/slippage drag. The adaptive layer (memory.py)
sizes around it; it does not turn a non-edge into an edge. Test in paper.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def ema(values: Sequence[float], length: int) -> Optional[float]:
    if len(values) < length:
        return None
    k = 2 / (length + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def generate_signal(
    closes: Sequence[float],
    fast: int,
    slow: int,
    breakout_lookback: int,
    have_position: bool,
    allow_short: bool,
) -> Optional[str]:
    if have_position:
        return None
    need = max(slow, breakout_lookback) + 2
    if len(closes) < need:
        return None

    fast_e = ema(closes, fast)
    slow_e = ema(closes, slow)
    if fast_e is None or slow_e is None:
        return None

    window = closes[-(breakout_lookback + 1):-1]  # exclude current bar
    recent_high = max(window)
    recent_low = min(window)
    price = closes[-1]

    # Long: upward momentum + break of recent high
    if fast_e > slow_e and price >= recent_high:
        return "BUY"
    # Short (optional): downward momentum + break of recent low
    if allow_short and fast_e < slow_e and price <= recent_low:
        return "SELL"
    return None
