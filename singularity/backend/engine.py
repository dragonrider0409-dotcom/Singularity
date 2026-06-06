"""Trading engine: the loop that ties strategy, learning, and orders together."""

from __future__ import annotations

import asyncio
import datetime as dt
from collections import deque
from typing import Deque, Optional

from .config import BotConfig, load_config, save_config
from .ib_client import IBClient
from .memory import memory
from .models import TradeRecord
from .strategy import generate_signal


class Engine:
    def __init__(self) -> None:
        self.client = IBClient()
        self.config: BotConfig = load_config()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.last_signal: Optional[str] = None
        self.last_message = "Idle."
        self.trades: Deque[TradeRecord] = deque(maxlen=200)

        # round-trip tracking for learning
        self._in_trade = False
        self._entry_px = 0.0
        self._entry_action = ""
        self._entry_context = ""
        self._pnl_baseline = 0.0

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return
        await self.client.connect(self.config)
        await self.client.prepare(self.config)
        self._running = True
        self.last_message = (
            f"Live on {self.config.symbol} [{self.config.asset_type}] "
            f"{self.config.mode.upper()} · R:R {self.config.risk_reward_ratio}"
        )
        self._task = asyncio.create_task(self._loop())

    async def stop(self, flatten: bool = False) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if flatten:
            try:
                await self.client.flatten(self.config.symbol)
            except Exception as e:  # noqa: BLE001
                self.last_message = f"Flatten error: {e}"
        self.client.disconnect()
        self.last_message = "Stopped."

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._step()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.last_message = f"Loop error: {e}"
            await asyncio.sleep(self.config.poll_seconds)

    async def _step(self) -> None:
        cfg = self.config

        # record equity for the chart + learning
        nlv = self.client.net_liquidation()
        if nlv is not None:
            memory.record_equity(nlv)

        # detect a round-trip close -> learn from it
        pos = self.client.position_size(cfg.symbol)
        if self._in_trade and pos == 0:
            realized = self.client.realized_pnl()
            pnl = round(realized - self._pnl_baseline, 2)
            memory.record_trade(
                symbol=cfg.symbol, asset_type=cfg.asset_type,
                action=self._entry_action, qty=0, entry=self._entry_px,
                exit_=None, pnl=pnl, context=self._entry_context,
            )
            self.last_message = f"Trade closed · P&L {pnl:+.2f} · learned context {self._entry_context}"
            self._in_trade = False

        have_position = self.client.open_position_count(cfg.symbol) >= cfg.max_open_positions

        closes = await self.client.recent_closes(cfg)
        need = max(cfg.slow_ema, cfg.breakout_lookback) + 2
        if len(closes) < need:
            self.last_message = "Gathering bars…"
            return

        signal = generate_signal(
            closes, cfg.fast_ema, cfg.slow_ema, cfg.breakout_lookback,
            have_position, cfg.allow_short,
        )
        self.last_signal = signal or "FLAT"
        if signal is None:
            return

        # learned position-size scale for the current context
        context = memory.context_key(closes)
        scale = 1.0
        if cfg.learning_enabled:
            scale = memory.size_scale(
                context, lo=cfg.min_size_scale, hi=cfg.max_size_scale,
                gate_after=cfg.gate_after_samples,
            )
        if scale <= 0:
            self.last_message = f"Skipped {signal}: context {context} gated by learning."
            return

        entry = await self.client.last_price()
        if entry is None:
            self.last_message = "No price available."
            return

        order = await self.client.place_bracket(cfg, signal, entry, scale)

        # arm round-trip tracking
        self._in_trade = True
        self._entry_px = order["entry"]
        self._entry_action = order["action"]
        self._entry_context = context
        self._pnl_baseline = self.client.realized_pnl()

        self.trades.appendleft(TradeRecord(
            time=dt.datetime.now().strftime("%H:%M:%S"), symbol=cfg.symbol,
            action=order["action"], quantity=order["quantity"], entry=order["entry"],
            take_profit=order["take_profit"], stop_loss=order["stop_loss"],
            size_scale=scale, context=context, status="submitted",
        ))
        self.last_message = (
            f"{order['action']} {order['quantity']} {cfg.symbol} @ {order['entry']} "
            f"| TP {order['take_profit']} | SL {order['stop_loss']} | size×{scale} [{context}]"
        )

    def update_config(self, new: BotConfig) -> None:
        self.config = new
        save_config(new)
        self.client.spec = None  # re-prepare next cycle


engine = Engine()
