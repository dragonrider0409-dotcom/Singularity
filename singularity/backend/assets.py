"""
Build IB contracts for any market: US/international stocks, crypto, forex.

This is what lets Singularity trade "any stock" plus crypto and FX — IBKR
exposes them all through different contract classes, and this module picks
the right one (and the right historical-data and sizing conventions).
"""

from __future__ import annotations

from dataclasses import dataclass

from ib_async import Stock, Crypto, Forex, Contract

from .config import BotConfig


@dataclass
class AssetSpec:
    contract: Contract
    what_to_show: str   # historical data type valid for this asset
    use_rth: bool       # 24/7 markets ignore RTH
    fractional: bool    # crypto/fx allow fractional size
    min_tick: float


def build(cfg: BotConfig) -> AssetSpec:
    sym = cfg.symbol.upper().strip()

    if cfg.asset_type == "crypto":
        exch = cfg.exchange if cfg.exchange not in ("SMART", "") else "PAXOS"
        return AssetSpec(
            contract=Crypto(sym, exch, cfg.currency or "USD"),
            what_to_show="AGGTRADES",
            use_rth=False,
            fractional=True,
            min_tick=0.01,
        )

    if cfg.asset_type == "forex":
        # symbol like EURUSD, or base/quote inferred
        pair = sym if len(sym) == 6 else (sym + (cfg.currency or "USD"))
        return AssetSpec(
            contract=Forex(pair),
            what_to_show="MIDPOINT",
            use_rth=False,
            fractional=True,
            min_tick=0.00005,
        )

    # default: stock (US via SMART, or any international exchange/currency)
    exch = cfg.exchange or "SMART"
    return AssetSpec(
        contract=Stock(sym, exch, cfg.currency or "USD"),
        what_to_show="TRADES",
        use_rth=cfg.rth_only,
        fractional=False,
        min_tick=0.01,
    )


def duration_for(bar_size: str) -> str:
    """Pick a sensible history window for a given bar size (for fast loops)."""
    bs = bar_size.lower()
    if "sec" in bs:
        return "1800 S"      # 30 min of seconds-bars
    if "min" in bs:
        return "1 D"
    return "5 D"
