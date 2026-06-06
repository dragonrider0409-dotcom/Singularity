"""Singularity API: auth + guided setup + trading control + dashboard host."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .auth import auth
from .config import BotConfig
from .engine import engine
from .memory import memory
from .models import (AccountResponse, ActionResponse, ChartsResponse,
                     MemoryResponse, Point, StatusResponse, TradeRecord)
from .setup_flow import setup_tracker


def _frontend_dir() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "frontend"


FRONTEND = _frontend_dir()
app = FastAPI(title="Singularity", version="3.0.0")


def current_user(authorization: str = Header(None)) -> str:
    token = (authorization or "").replace("Bearer ", "").strip()
    user = auth.user_for_token(token)
    if not user:
        raise HTTPException(401, detail="Please log in.")
    return user


class Creds(BaseModel):
    username: str
    password: str


class StepBody(BaseModel):
    step: str


@app.post("/api/signup")
def signup(c: Creds):
    try:
        token = auth.signup(c.username, c.password)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return {"token": token, "username": c.username.strip().lower()}


@app.post("/api/login")
def login(c: Creds):
    try:
        token = auth.login(c.username, c.password)
    except ValueError as e:
        raise HTTPException(401, detail=str(e))
    return {"token": token, "username": c.username.strip().lower()}


@app.post("/api/logout")
def logout(authorization: str = Header(None)):
    auth.logout((authorization or "").replace("Bearer ", "").strip())
    return {"ok": True}


@app.get("/api/me")
def me(user: str = Depends(current_user)):
    return {"username": user}


@app.get("/api/setup")
def get_setup(user: str = Depends(current_user)):
    return setup_tracker.progress(user)


@app.post("/api/setup/complete")
def complete_step(b: StepBody, user: str = Depends(current_user)):
    try:
        setup_tracker.mark(user, b.step)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return setup_tracker.progress(user)


@app.post("/api/setup/reset-step")
def reset_step(b: StepBody, user: str = Depends(current_user)):
    setup_tracker.unmark(user, b.step)
    return setup_tracker.progress(user)


@app.post("/api/setup/verify-connection")
async def verify_connection(user: str = Depends(current_user)):
    ok, msg = await engine.client.test_connection(engine.config)
    if ok:
        setup_tracker.mark(user, "verify_connection")
    return {"ok": ok, "message": msg, "progress": setup_tracker.progress(user)}


@app.get("/api/status", response_model=StatusResponse)
def status(user: str = Depends(current_user)) -> StatusResponse:
    c = engine.config
    conn = engine.client.connected
    return StatusResponse(
        running=engine.running, connected=conn, mode=c.mode, symbol=c.symbol,
        asset_type=c.asset_type, risk_reward_ratio=c.risk_reward_ratio,
        open_positions=engine.client.open_position_count(c.symbol) if conn else 0,
        last_signal=engine.last_signal, last_message=engine.last_message,
        pnl_realized=engine.client.realized_pnl() if conn else None,
        net_liquidation=engine.client.net_liquidation() if conn else None,
    )


@app.get("/api/config", response_model=BotConfig)
def get_config(user: str = Depends(current_user)) -> BotConfig:
    return engine.config


@app.post("/api/config", response_model=ActionResponse)
def set_config(cfg: BotConfig, user: str = Depends(current_user)) -> ActionResponse:
    engine.update_config(cfg)
    setup_tracker.mark(user, "configure")
    return ActionResponse(ok=True, message="Saved.")


@app.get("/api/account", response_model=AccountResponse)
def account(user: str = Depends(current_user)) -> AccountResponse:
    if not engine.client.connected:
        return AccountResponse(connected=False)
    s = engine.client.account_summary()
    return AccountResponse(connected=True, account=s.get("account"),
                           net_liquidation=s.get("NetLiquidation"),
                           buying_power=s.get("BuyingPower"), cash=s.get("TotalCashValue"))


@app.get("/api/trades", response_model=list[TradeRecord])
def trades(user: str = Depends(current_user)) -> list[TradeRecord]:
    return list(engine.trades)


@app.get("/api/charts", response_model=ChartsResponse)
def charts(user: str = Depends(current_user)) -> ChartsResponse:
    eq = [Point(t=p["ts"], v=p["net_liq"]) for p in memory.equity_curve()]
    tp = [Point(t=p["ts"], v=(p["pnl"] or 0.0)) for p in memory.trade_pnls()]
    return ChartsResponse(equity=eq, trade_pnls=tp)


@app.get("/api/memory", response_model=MemoryResponse)
def memory_summary(user: str = Depends(current_user)) -> MemoryResponse:
    return MemoryResponse(**memory.summary())


@app.post("/api/start", response_model=ActionResponse)
async def start(user: str = Depends(current_user)) -> ActionResponse:
    try:
        await engine.start()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, detail=(
            f"Could not start: {e}. Is IB Gateway/TWS running with the API "
            f"enabled on port {engine.config.port()} and logged into a "
            f"{engine.config.mode} account?"))
    if engine.config.mode == "paper":
        setup_tracker.mark(user, "paper_run")
    return ActionResponse(ok=True, message=engine.last_message)


@app.post("/api/stop", response_model=ActionResponse)
async def stop(flatten: bool = False, user: str = Depends(current_user)) -> ActionResponse:
    await engine.stop(flatten=flatten)
    return ActionResponse(ok=True, message=engine.last_message)


@app.post("/api/flatten", response_model=ActionResponse)
async def flatten(user: str = Depends(current_user)) -> ActionResponse:
    if not engine.client.connected:
        raise HTTPException(400, detail="Not connected.")
    n = await engine.client.flatten(engine.config.symbol)
    return ActionResponse(ok=True, message=f"Flatten sent for {n} position(s).")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=str(FRONTEND)), name="static")
