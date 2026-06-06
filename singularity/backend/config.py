"""Runtime configuration for Singularity."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

APP_NAME = "Singularity"

# Persist next to the executable / project root.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "singularity_config.json"

PORTS = {
    "paper_tws": 7497, "live_tws": 7496,
    "paper_gateway": 4002, "live_gateway": 4001,
}

AssetType = Literal["stock", "crypto", "forex"]


class BotConfig(BaseModel):
    # ---- required adjustable knobs ----------------------------------------
    symbol: str = Field("AAPL", description="Ticker / pair, e.g. AAPL, BTC, EURUSD")
    risk_reward_ratio: float = Field(1.8, gt=0, le=20)

    # ---- asset routing (any market) ---------------------------------------
    asset_type: AssetType = Field("stock")
    exchange: str = Field("SMART", description="SMART, PAXOS (crypto), IDEALPRO (fx), LSE, etc.")
    currency: str = Field("USD")

    # ---- risk / sizing -----------------------------------------------------
    risk_per_trade_usd: float = Field(40.0, gt=0)
    stop_loss_pct: float = Field(0.4, gt=0, le=25)
    max_notional_usd: float = Field(5000.0, gt=0, description="Hard cap on position value.")
    max_open_positions: int = Field(1, ge=1, le=10)
    allow_short: bool = Field(False)

    # ---- medium-frequency loop --------------------------------------------
    poll_seconds: int = Field(10, ge=3, le=3600)
    bar_size: str = Field("10 secs")
    fast_ema: int = Field(8, ge=1, le=200)
    slow_ema: int = Field(21, ge=2, le=400)
    breakout_lookback: int = Field(20, ge=2, le=300)

    # ---- adaptive learning -------------------------------------------------
    learning_enabled: bool = Field(True)
    min_size_scale: float = Field(0.3, ge=0, le=1)
    max_size_scale: float = Field(1.6, ge=1, le=5)
    gate_after_samples: int = Field(8, ge=1, description="Min trades before gating a bad context.")

    # ---- connection --------------------------------------------------------
    mode: Literal["paper", "live"] = Field("paper")
    connection: Literal["tws", "gateway"] = Field("gateway")
    host: str = Field("127.0.0.1")
    client_id: int = Field(21, ge=0)
    rth_only: bool = Field(False, description="Crypto/fx ignore this; stocks honor it.")

    def port(self) -> int:
        return PORTS[f"{self.mode}_{self.connection}"]


_lock = threading.Lock()


def load_config() -> BotConfig:
    if CONFIG_PATH.exists():
        try:
            return BotConfig(**json.loads(CONFIG_PATH.read_text()))
        except Exception:
            pass
    cfg = BotConfig()
    save_config(cfg)
    return cfg


def save_config(cfg: BotConfig) -> None:
    with _lock:
        CONFIG_PATH.write_text(json.dumps(cfg.model_dump(), indent=2))
