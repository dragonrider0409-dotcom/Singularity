"""API schemas for Singularity."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class StatusResponse(BaseModel):
    running: bool
    connected: bool
    mode: str
    symbol: str
    asset_type: str
    risk_reward_ratio: float
    open_positions: int
    last_signal: Optional[str] = None
    last_message: Optional[str] = None
    pnl_realized: Optional[float] = None
    net_liquidation: Optional[float] = None


class AccountResponse(BaseModel):
    connected: bool
    account: Optional[str] = None
    net_liquidation: Optional[float] = None
    buying_power: Optional[float] = None
    cash: Optional[float] = None


class TradeRecord(BaseModel):
    time: str
    symbol: str
    action: str
    quantity: float
    entry: float
    take_profit: float
    stop_loss: float
    size_scale: float
    context: str
    status: str


class ActionResponse(BaseModel):
    ok: bool
    message: str


class Point(BaseModel):
    t: str
    v: float


class ChartsResponse(BaseModel):
    equity: List[Point]
    trade_pnls: List[Point]


class MemoryResponse(BaseModel):
    total_trades: int
    wins: int
    win_rate: Optional[float] = None
    net_pnl: float
    contexts: list
