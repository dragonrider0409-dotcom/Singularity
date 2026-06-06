"""
Local accounts: sign-up / log-in for the desktop app.

This is on-device protection — accounts live in a SQLite file next to the app,
passwords are PBKDF2-hashed, and a session token gates the dashboard. It is not
a cloud service; it keeps the local app behind a login on his machine.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from .config import ROOT

DB_PATH = ROOT / "singularity_app.db"
ITER = 200_000
SESSION_TTL = 60 * 60 * 12  # 12 hours


class Auth:
    def __init__(self, path: Path = DB_PATH) -> None:
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS users("
                "username TEXT PRIMARY KEY, pw TEXT, created TEXT)"
            )
        self._sessions: dict[str, tuple[str, float]] = {}

    # ---- hashing ----------------------------------------------------------
    @staticmethod
    def _hash(pw: str, salt: bytes | None = None) -> str:
        salt = salt or os.urandom(16)
        h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, ITER)
        return f"{salt.hex()}:{h.hex()}"

    @staticmethod
    def _verify(pw: str, stored: str) -> bool:
        try:
            salt_hex, h_hex = stored.split(":")
            calc = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), ITER)
            return secrets.compare_digest(calc.hex(), h_hex)
        except Exception:
            return False

    # ---- accounts ---------------------------------------------------------
    def user_exists(self, username: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM users WHERE username=?", (username,)
        ).fetchone() is not None

    def signup(self, username: str, password: str) -> str:
        username = username.strip().lower()
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters.")
        if self.user_exists(username):
            raise ValueError("That username already exists — try logging in.")
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO users(username,pw,created) VALUES(?,?,datetime('now'))",
                (username, self._hash(password)),
            )
        return self._new_session(username)

    def login(self, username: str, password: str) -> str:
        username = username.strip().lower()
        row = self.conn.execute(
            "SELECT pw FROM users WHERE username=?", (username,)
        ).fetchone()
        if not row or not self._verify(password, row["pw"]):
            raise ValueError("Incorrect username or password.")
        return self._new_session(username)

    # ---- sessions ---------------------------------------------------------
    def _new_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        self._sessions[token] = (username, time.time())
        return token

    def user_for_token(self, token: str) -> str | None:
        rec = self._sessions.get(token or "")
        if not rec:
            return None
        username, created = rec
        if time.time() - created > SESSION_TTL:
            self._sessions.pop(token, None)
            return None
        return username

    def logout(self, token: str) -> None:
        self._sessions.pop(token or "", None)


auth = Auth()
