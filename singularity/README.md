# Singularity

An adaptive, multi-asset, medium-frequency trading bot for **Interactive
Brokers**. Opens as a desktop window with a local **login / sign-up**, a guided
**setup checklist that tracks the user's progress**, live equity/P&L charts, and
a learning layer that remembers and adapts across trades.

- **Any market:** US & international stocks, crypto, forex (IB contract routing).
- **Medium frequency:** fast loop on seconds/minute bars, momentum-breakout entries.
- **Risk:reward enforced** on every trade via bracket orders (entry + take-profit + stop).
- **Adaptive learning:** every closed trade is recorded to a local database;
  position size is tilted toward market contexts that have historically paid out
  and gated away from ones that repeatedly lose.
- **No-code dashboard:** the operator only ever sees a web page.
- **Ships as one app:** bundled with PyInstaller — the end user installs no
  Python, pip, or editor.

## Honest limits (read these)

- **There is no "guaranteed profitable" strategy, and this isn't one.** The
  built-in momentum-breakout rule is a sensible starting point, not a proven
  edge. The learning layer *adapts* sizing from history — it cannot manufacture
  an edge that isn't there. Most automated retail daytrading loses money.
- **The learning is real but modest:** it reduces size in losing contexts and
  presses winning ones, using shrinkage so it doesn't overreact to a few trades.
- **IB Gateway is required.** Interactive Brokers' API only works with their own
  TWS or IB Gateway running locally and logged in. No bot can avoid that — see
  SHIPPING.md. Everything *else* the friend installs is zero.
- **Defaults to paper trading.** Keep it there until you trust it.
- **US PDT rule:** 4+ day trades in 5 business days in a margin account requires
  $25,000 minimum equity. Frequent trading also multiplies commissions/slippage.
- Not affiliated with Interactive Brokers. Not financial advice.

## Run from source (for development)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py            # opens http://127.0.0.1:8000
```

## Build the shippable app

```bash
python build.py          # output in dist/  (Singularity / Singularity.exe)
```

See **SHIPPING.md** for how to send it to a non-technical friend.

## Layout

```
singularity/
├── run.py / build.py / singularity.spec
├── requirements.txt
├── backend/
│   ├── main.py        API + serves dashboard
│   ├── engine.py      MFT loop, learned sizing, round-trip detection
│   ├── ib_client.py   IB connection, multi-asset orders
│   ├── assets.py      stock / crypto / forex contract routing
│   ├── strategy.py    momentum-breakout entries  (edit to use your own)
│   ├── memory.py      SQLite learning + adaptive sizing
│   ├── config.py / models.py
└── frontend/index.html   dashboard, charts, learning panel (no external libs)
```

To plug in your own strategy, edit `backend/strategy.py → generate_signal()`.
Everything else keeps working.
