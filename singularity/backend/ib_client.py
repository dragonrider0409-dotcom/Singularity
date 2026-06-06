"""IB connection + order management for any asset class."""

from __future__ import annotations

import math
from typing import List, Optional

from ib_async import IB, LimitOrder, MarketOrder, StopOrder

from .assets import AssetSpec, build, duration_for
from .config import BotConfig


class IBClient:
    def __init__(self) -> None:
        self.ib = IB()
        self.spec: Optional[AssetSpec] = None

    @property
    def connected(self) -> bool:
        return self.ib.isConnected()

    async def connect(self, cfg: BotConfig) -> None:
        if self.connected:
            return
        await self.ib.connectAsync(cfg.host, cfg.port(), clientId=cfg.client_id, timeout=12)
        self.ib.reqMarketDataType(3)  # delayed fallback if no live subscription

    def disconnect(self) -> None:
        if self.connected:
            self.ib.disconnect()
        self.spec = None

    async def test_connection(self, cfg: BotConfig) -> tuple[bool, str]:
        """Try to reach Gateway/TWS. Used by the setup checklist."""
        if self.connected:
            return True, "Already connected to IB."
        try:
            await self.ib.connectAsync(cfg.host, cfg.port(), clientId=cfg.client_id, timeout=8)
            ok = self.ib.isConnected()
            acct = self.account_summary().get("account")
            self.ib.disconnect()
            if ok:
                return True, f"Connected to IB ({cfg.mode}{f', account {acct}' if acct else ''})."
            return False, "Reached the port but the API did not respond."
        except Exception as e:  # noqa: BLE001
            return False, (
                f"Could not connect on port {cfg.port()}: {e}. Make sure Gateway "
                f"is open, logged into a {cfg.mode} account, and the API is enabled."
            )

    async def prepare(self, cfg: BotConfig) -> AssetSpec:
        spec = build(cfg)
        qualified = await self.ib.qualifyContractsAsync(spec.contract)
        if not qualified:
            raise ValueError(f"Could not resolve '{cfg.symbol}' as {cfg.asset_type}.")
        spec.contract = qualified[0]
        self.spec = spec
        return spec

    # ---- account ----------------------------------------------------------
    def account_summary(self) -> dict:
        out: dict = {}
        for v in self.ib.accountValues():
            if v.tag in ("NetLiquidation", "BuyingPower", "TotalCashValue") and v.currency in ("USD", "BASE", ""):
                try:
                    out[v.tag] = float(v.value)
                    out["account"] = v.account
                except ValueError:
                    pass
        return out

    def net_liquidation(self) -> Optional[float]:
        return self.account_summary().get("NetLiquidation")

    def position_size(self, symbol: str) -> float:
        sym = symbol.upper().strip()
        return sum(p.position for p in self.ib.positions() if p.contract.symbol == sym)

    def open_position_count(self, symbol: str) -> int:
        sym = symbol.upper().strip()
        return sum(1 for p in self.ib.positions() if p.contract.symbol == sym and p.position != 0)

    def realized_pnl(self) -> float:
        total = 0.0
        for pnl in self.ib.pnl():
            if pnl.realizedPnL is not None and not math.isnan(pnl.realizedPnL):
                total += pnl.realizedPnL
        return round(total, 2)

    # ---- market data ------------------------------------------------------
    async def recent_closes(self, cfg: BotConfig, n: int = 200) -> List[float]:
        if self.spec is None:
            await self.prepare(cfg)
        bars = await self.ib.reqHistoricalDataAsync(
            self.spec.contract, endDateTime="",
            durationStr=duration_for(cfg.bar_size),
            barSizeSetting=cfg.bar_size,
            whatToShow=self.spec.what_to_show,
            useRTH=self.spec.use_rth, formatDate=1,
        )
        return [b.close for b in bars][-n:]

    async def last_price(self) -> Optional[float]:
        if self.spec is None:
            return None
        t = self.ib.reqMktData(self.spec.contract, "", False, False)
        await self.ib.sleep(1.5)
        px = t.marketPrice()
        if px is None or math.isnan(px):
            px = t.close
        self.ib.cancelMktData(self.spec.contract)
        if px is None or math.isnan(px):
            return None
        return float(px)

    # ---- sizing + bracket -------------------------------------------------
    def size_and_levels(self, cfg: BotConfig, entry: float, action: str, size_scale: float):
        distance = entry * (cfg.stop_loss_pct / 100.0)
        if distance <= 0:
            raise ValueError("Stop distance is zero.")
        raw = (cfg.risk_per_trade_usd / distance) * size_scale
        # respect notional cap
        max_by_notional = cfg.max_notional_usd / entry
        qty = min(raw, max_by_notional)
        if self.spec and self.spec.fractional:
            qty = round(qty, 6)
        else:
            qty = float(int(qty))
        qty = max(qty, (0.0001 if (self.spec and self.spec.fractional) else 1.0))

        if action == "BUY":
            stop = entry - distance
            take = entry + cfg.risk_reward_ratio * distance
        else:
            stop = entry + distance
            take = entry - cfg.risk_reward_ratio * distance
        r = lambda x: round(x, 4 if (self.spec and self.spec.fractional) else 2)
        return qty, r(entry), r(take), r(stop)

    async def place_bracket(self, cfg: BotConfig, action: str, entry: float, size_scale: float):
        if self.spec is None:
            await self.prepare(cfg)
        qty, entry_px, take_px, stop_px = self.size_and_levels(cfg, entry, action, size_scale)
        opp = "SELL" if action == "BUY" else "BUY"

        parent = LimitOrder(action, qty, entry_px)
        parent.orderId = self.ib.client.getReqId(); parent.transmit = False
        take = LimitOrder(opp, qty, take_px)
        take.orderId = self.ib.client.getReqId(); take.parentId = parent.orderId; take.transmit = False
        stop = StopOrder(opp, qty, stop_px)
        stop.orderId = self.ib.client.getReqId(); stop.parentId = parent.orderId; stop.transmit = True

        oca = f"sing-{parent.orderId}"
        for o in (take, stop):
            o.ocaGroup = oca; o.ocaType = 1
        if not self.spec.use_rth and cfg.asset_type == "stock":
            for o in (parent, take, stop):
                o.outsideRth = True

        for o in (parent, take, stop):
            self.ib.placeOrder(self.spec.contract, o)
        return {"action": action, "quantity": qty, "entry": entry_px,
                "take_profit": take_px, "stop_loss": stop_px}

    async def flatten(self, symbol: str) -> int:
        sym = symbol.upper().strip()
        for tr in self.ib.openTrades():
            if tr.contract.symbol == sym:
                self.ib.cancelOrder(tr.order)
        closed = 0
        for p in self.ib.positions():
            if p.contract.symbol == sym and p.position != 0:
                act = "SELL" if p.position > 0 else "BUY"
                self.ib.placeOrder(p.contract, MarketOrder(act, abs(p.position)))
                closed += 1
        return closed
