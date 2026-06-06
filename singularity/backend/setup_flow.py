"""
Guided setup. Defines the steps the user must complete before trading, tracks
which ones he's finished (persisted per user), and auto-verifies the ones the
app can check itself (the IB connection, the first paper run).

The frontend turns this into the notification: when he logs in it shows the
next unfinished step; as steps complete the progress bar fills; when all are
done it shows the all-clear.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .config import ROOT

DB_PATH = ROOT / "singularity_app.db"

# key, title, what to do, and whether the app verifies it automatically.
STEPS = [
    {
        "key": "install_gateway",
        "title": "Install & open IB Gateway",
        "detail": "Download IB Gateway from Interactive Brokers, install it, and "
                  "launch it. (Trader Workstation works too.) This is the only "
                  "extra program you need — it's how Singularity reaches the market.",
        "auto": False,
    },
    {
        "key": "enable_api",
        "title": "Turn on the API in Gateway",
        "detail": "In Gateway: Configure → Settings → API → Settings. Check "
                  "'Enable ActiveX and Socket Clients' and 'Download open orders "
                  "on connection', and add 127.0.0.1 as a trusted IP.",
        "auto": False,
    },
    {
        "key": "login_ib",
        "title": "Log into your IB account (use Paper first)",
        "detail": "Log into Gateway with your Interactive Brokers credentials. "
                  "Choose the PAPER account while you're testing — it trades "
                  "simulated money so nothing is at risk.",
        "auto": False,
    },
    {
        "key": "verify_connection",
        "title": "Connect Singularity to Gateway",
        "detail": "Click 'Test connection'. Singularity will reach Gateway on the "
                  "port for your selected mode. A green check means you're wired up.",
        "auto": True,
    },
    {
        "key": "configure",
        "title": "Pick your symbol and risk:reward",
        "detail": "In Configuration, set the symbol (e.g. AAPL, BTC, EURUSD), your "
                  "risk:reward and dollars-per-trade, then press Save.",
        "auto": False,
    },
    {
        "key": "paper_run",
        "title": "Run it in Paper and watch",
        "detail": "Press Start while in Paper mode. Watch the equity curve and the "
                  "'what it has learned' panel. Let it run before considering Live.",
        "auto": True,
    },
]
STEP_KEYS = [s["key"] for s in STEPS]


class SetupTracker:
    def __init__(self, path: Path = DB_PATH) -> None:
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS setup_progress("
                "username TEXT, step TEXT, done_at TEXT, PRIMARY KEY(username, step))"
            )

    def _done_set(self, username: str) -> set[str]:
        rows = self.conn.execute(
            "SELECT step FROM setup_progress WHERE username=?", (username,)
        ).fetchall()
        return {r["step"] for r in rows}

    def mark(self, username: str, step: str) -> None:
        if step not in STEP_KEYS:
            raise ValueError("Unknown setup step.")
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO setup_progress(username,step,done_at) "
                "VALUES(?,?,datetime('now'))",
                (username, step),
            )

    def unmark(self, username: str, step: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "DELETE FROM setup_progress WHERE username=? AND step=?",
                (username, step),
            )

    def progress(self, username: str) -> dict:
        done = self._done_set(username)
        steps = [
            {**s, "done": s["key"] in done}
            for s in STEPS
        ]
        next_step = next((s for s in steps if not s["done"]), None)
        return {
            "steps": steps,
            "completed": len(done),
            "total": len(STEPS),
            "complete": len(done) >= len(STEPS),
            "next_key": next_step["key"] if next_step else None,
        }


setup_tracker = SetupTracker()
