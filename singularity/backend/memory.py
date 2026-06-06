"""
Persistent learning / memory.

This is the honest version of "improves with every trade and memorizes":

  * Every completed round-trip is written to a SQLite file that survives
    restarts. That is the memory.
  * Trades are bucketed by CONTEXT (hour-of-day + short-term trend regime +
    volatility band). For each context we track wins, losses, and net P&L.
  * Before taking a new trade, the engine asks this module for a SIZE SCALE
    for the current context. Contexts that have historically paid out get
    scaled up (toward max_size_scale); contexts that have repeatedly lost get
    scaled down and, past a sample threshold, gated out entirely.
  * Estimates use Bayesian shrinkage toward neutral, so a couple of lucky or
    unlucky early trades don't make it overconfident.

What this is NOT: a guarantee of profit. Adaptation means it tilts toward
its own past winners; if the underlying strategy has no real edge, learning
to size into noise will not manufacture one. It reduces damage from bad
contexts and presses good ones — that is all an honest learner can claim.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .config import ROOT

DB_PATH = ROOT / "singularity_memory.db"


class Memory:
    def __init__(self, path: Path = DB_PATH) -> None:
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        with self.conn:
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS trades(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT, symbol TEXT, asset_type TEXT, action TEXT,
                    qty REAL, entry REAL, exit REAL, pnl REAL,
                    context TEXT, win INTEGER)"""
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS equity(
                    ts TEXT, net_liq REAL)"""
            )

    # ---- context key ------------------------------------------------------
    @staticmethod
    def context_key(closes) -> str:
        """Bucket the current market state into a learnable context."""
        hour = dt.datetime.now().strftime("%H")
        trend = "flat"
        vol = "lo"
        if len(closes) >= 20:
            recent = closes[-20:]
            slope = recent[-1] - recent[0]
            trend = "up" if slope > 0 else "down" if slope < 0 else "flat"
            avg = sum(recent) / len(recent)
            spread = (max(recent) - min(recent)) / avg if avg else 0
            vol = "hi" if spread > 0.004 else "lo"
        return f"h{hour}|{trend}|{vol}"

    # ---- writes -----------------------------------------------------------
    def record_trade(self, *, symbol, asset_type, action, qty, entry, exit_, pnl, context):
        win = 1 if pnl is not None and pnl > 0 else 0
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO trades(ts,symbol,asset_type,action,qty,entry,exit,pnl,context,win)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (dt.datetime.now().isoformat(timespec="seconds"), symbol, asset_type,
                 action, qty, entry, exit_, pnl, context, win),
            )

    def record_equity(self, net_liq: float):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO equity(ts,net_liq) VALUES(?,?)",
                (dt.datetime.now().isoformat(timespec="seconds"), net_liq),
            )

    # ---- the learned signal ----------------------------------------------
    def context_expectancy(self, context: str):
        """Shrunken expectancy (avg P&L) and sample size for a context."""
        row = self.conn.execute(
            "SELECT COUNT(*) n, AVG(pnl) avg_pnl, SUM(win) wins FROM trades WHERE context=?",
            (context,),
        ).fetchone()
        n = row["n"] or 0
        if n == 0:
            return 0.0, 0, 0.0
        avg = row["avg_pnl"] or 0.0
        wins = row["wins"] or 0
        # shrink avg toward 0 with pseudo-count of 5 neutral observations
        shrunk = avg * n / (n + 5)
        winrate = wins / n
        return shrunk, n, winrate

    def size_scale(self, context: str, *, lo: float, hi: float, gate_after: int) -> float:
        """
        Map a context's history to a position-size multiplier.
          - unknown / thin history  -> 1.0 (neutral)
          - positive expectancy      -> scale up toward `hi`
          - negative expectancy      -> scale down toward `lo`; gate to 0 if
            the loss is persistent past `gate_after` samples.
        """
        exp, n, _ = self.context_expectancy(context)
        if n < 3:
            return 1.0
        if exp >= 0:
            # confidence grows with sample size
            conf = min(1.0, n / 20)
            return round(1.0 + (hi - 1.0) * conf * min(1.0, exp / 25.0), 3)
        # negative expectancy
        if n >= gate_after and exp < -5:
            return 0.0  # gate out a reliably losing context
        conf = min(1.0, n / 20)
        return round(max(lo, 1.0 + (lo - 1.0) * conf), 3)

    # ---- reads for the UI -------------------------------------------------
    def summary(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) n, SUM(win) wins, SUM(pnl) net FROM trades"
        ).fetchone()
        n = row["n"] or 0
        contexts = self.conn.execute(
            "SELECT context, COUNT(*) n, AVG(pnl) avg_pnl, SUM(win) wins "
            "FROM trades GROUP BY context ORDER BY avg_pnl DESC"
        ).fetchall()
        return {
            "total_trades": n,
            "wins": row["wins"] or 0,
            "win_rate": round((row["wins"] or 0) / n, 3) if n else None,
            "net_pnl": round(row["net"] or 0.0, 2),
            "contexts": [
                {
                    "context": c["context"],
                    "trades": c["n"],
                    "win_rate": round(c["wins"] / c["n"], 2) if c["n"] else 0,
                    "avg_pnl": round(c["avg_pnl"] or 0.0, 2),
                }
                for c in contexts[:12]
            ],
        }

    def trade_pnls(self, limit: int = 100):
        rows = self.conn.execute(
            "SELECT ts, symbol, action, pnl FROM trades ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def equity_curve(self, limit: int = 500):
        rows = self.conn.execute(
            "SELECT ts, net_liq FROM equity ORDER BY rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


memory = Memory()
