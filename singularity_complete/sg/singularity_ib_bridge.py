#!/usr/bin/env python3
"""
Singularity IB Bridge  v2.0
─────────────────────────────────────────────────────────────────────────────
Runs on your LOCAL machine. Connects to IB Gateway / TWS and exposes a
REST API on port 8765 that the Singularity bot reads from the browser.

Setup
─────
pip install ib_insync flask flask-cors

Usage
─────
# Paper trading (port 7497):
python singularity_ib_bridge.py --ib-port 7497 --account U1234567

# Live trading (port 7496):
python singularity_ib_bridge.py --ib-port 7496 --account U1234567

# Custom IB Gateway host:
python singularity_ib_bridge.py --ib-host 192.168.1.10 --ib-port 7497 --account U1234567

Endpoints
─────────
GET  /status      → connection status + account ID
GET  /ping        → latency check (returns {"pong": true})
GET  /account     → NAV, day P&L, total P&L
GET  /positions   → open positions
GET  /trades      → today's closed trades
POST /order       → place order  { symbol, action, qty, order_type, price?, strategy? }
POST /close       → close position { symbol, qty, side }
"""

import argparse, threading, time, logging
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('SingularityBridge')

app = Flask(__name__)
CORS(app)  # Allow browser (Vercel/localhost) to call bridge

# ── IB connection state ────────────────────────────────────────────────────────
ib       = None
ACCOUNT  = ''
IB_HOST  = '127.0.0.1'
IB_PORT  = 7497
_connected = False
_trades_today = []  # filled orders accumulated during session

def connect_ib():
    global ib, _connected
    try:
        from ib_insync import IB as IBClient
        ib = IBClient()
        ib.connect(IB_HOST, IB_PORT, clientId=1, readonly=False)
        _connected = ib.isConnected()
        if _connected:
            log.info(f'✓ Connected to IB Gateway at {IB_HOST}:{IB_PORT} · Account: {ACCOUNT}')
            # Subscribe to fills
            ib.execDetailsEvent += on_fill
        else:
            log.warning('IB Gateway not connected')
    except Exception as e:
        log.error(f'IB connect error: {e}')
        _connected = False

def on_fill(trade, fill):
    """Record fills as they come in."""
    t = {
        'symbol':      fill.contract.symbol,
        'action':      fill.execution.side,
        'qty':         fill.execution.shares,
        'fill_price':  fill.execution.price,
        'ts':          int(time.time() * 1000),
        'strategy':    getattr(trade.order, 'orderRef', 'manual'),
        'order_id':    fill.execution.orderId,
    }
    _trades_today.append(t)
    log.info(f"Fill: {t['action']} {t['qty']} {t['symbol']} @ {t['fill_price']}")

def keep_alive():
    """Re-connect if IB drops the connection."""
    global _connected
    while True:
        time.sleep(30)
        if ib and not ib.isConnected():
            log.warning('IB disconnected — reconnecting…')
            _connected = False
            try:
                ib.connect(IB_HOST, IB_PORT, clientId=1, readonly=False)
                _connected = ib.isConnected()
                if _connected:
                    log.info('Reconnected.')
            except Exception as e:
                log.error(f'Reconnect error: {e}')

# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.route('/ping')
def ping():
    return jsonify({'pong': True, 'ts': int(time.time()*1000)})

@app.route('/status')
def status():
    global _connected
    if ib:
        _connected = ib.isConnected()
    return jsonify({'connected': _connected, 'account': ACCOUNT,
                    'ib_host': IB_HOST, 'ib_port': IB_PORT,
                    'mode': 'paper' if IB_PORT in (7497,4002) else 'live'})

@app.route('/account')
def account():
    if not ib or not _connected:
        return jsonify({'error': 'not connected', 'nav': 0, 'dpnl': 0, 'tpnl': 0})
    try:
        vals = {v.tag: v.value for v in ib.accountValues() if v.account == ACCOUNT or not ACCOUNT}
        nav  = float(vals.get('NetLiquidation', 0) or 0)
        dpnl = float(vals.get('DayTradesRemainingT+0', 0) or 0)  # fallback
        # Better P&L fields
        dpnl = float(vals.get('UnrealizedPnL', 0) or 0) + float(vals.get('RealizedPnL', 0) or 0)
        return jsonify({'nav': nav, 'dpnl': dpnl, 'tpnl': dpnl})
    except Exception as e:
        return jsonify({'error': str(e), 'nav': 0, 'dpnl': 0, 'tpnl': 0})

@app.route('/positions')
def positions():
    if not ib or not _connected:
        return jsonify([])
    try:
        from ib_insync import util
        pos_list = []
        for p in ib.positions():
            if ACCOUNT and p.account != ACCOUNT:
                continue
            sym = p.contract.symbol
            qty = p.position
            ep  = float(p.avgCost or 0)
            # Get market price from ticker
            tick = ib.ticker(p.contract)
            mp = float(tick.marketPrice() if tick else ep)
            pos_list.append({
                'symbol':      sym,
                'qty':         abs(qty),
                'side':        'buy' if qty > 0 else 'sell',
                'entry':       ep,
                'avg_cost':    ep,
                'marketPrice': mp,
                'market_price':mp,
                'position':    qty,
            })
        return jsonify(pos_list)
    except Exception as e:
        return jsonify([])

@app.route('/trades')
def trades():
    """Return today's filled orders."""
    return jsonify(_trades_today[-50:])  # last 50

@app.route('/order', methods=['POST'])
def place_order():
    if not ib or not _connected:
        return jsonify({'error': 'not connected'}), 503
    data       = request.get_json(force=True)
    symbol     = data.get('symbol', '').upper().strip()
    action     = data.get('action', 'BUY').upper()   # BUY | SELL
    qty        = int(data.get('qty', 1))
    order_type = data.get('order_type', 'MKT').upper()  # MKT | LMT | STP
    price      = data.get('price')
    strategy   = data.get('strategy', 'singularity')

    if not symbol or qty < 1:
        return jsonify({'error': 'invalid symbol or qty'}), 400
    if action not in ('BUY', 'SELL'):
        return jsonify({'error': 'action must be BUY or SELL'}), 400

    try:
        from ib_insync import Stock, MarketOrder, LimitOrder, StopOrder
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)

        if order_type == 'MKT':
            order = MarketOrder(action, qty)
        elif order_type == 'LMT':
            if not price:
                return jsonify({'error': 'price required for LMT order'}), 400
            order = LimitOrder(action, qty, float(price))
        elif order_type == 'STP':
            if not price:
                return jsonify({'error': 'price required for STP order'}), 400
            order = StopOrder(action, qty, float(price))
        else:
            return jsonify({'error': 'unknown order_type'}), 400

        # Tag order with strategy name for reference
        order.orderRef = strategy

        trade = ib.placeOrder(contract, order)
        log.info(f'Order placed: {action} {qty} {symbol} ({order_type}) · ref={strategy} · id={trade.order.orderId}')
        return jsonify({
            'status':   'submitted',
            'order_id': trade.order.orderId,
            'symbol':   symbol,
            'action':   action,
            'qty':      qty,
            'type':     order_type,
            'strategy': strategy,
        })
    except Exception as e:
        log.error(f'Order error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/close', methods=['POST'])
def close_position():
    data   = request.get_json(force=True)
    symbol = data.get('symbol','').upper().strip()
    qty    = int(data.get('qty', 1))
    side   = data.get('side', 'long')
    action = 'SELL' if side in ('long','buy') else 'BUY'
    return place_order.__wrapped__() if hasattr(place_order,'__wrapped__') else \
           (lambda: (request.json.__setitem__('action', action) or
                     request.json.__setitem__('order_type','MKT')) or place_order())()

# ── TradingView webhook support ─────────────────────────────────────────────────
@app.route('/webhook/tradingview', methods=['POST'])
def tradingview_webhook():
    """
    Receives TradingView alerts and converts them to IB orders.
    Configure your TradingView alert webhook URL as:
        http://YOUR_MACHINE_IP:8765/webhook/tradingview
    
    Alert message format (JSON):
        {"action":"buy","symbol":"NVDA","qty":10,"strategy":"momentum"}
    or plaintext:
        "BUY NVDA 10"
    """
    try:
        raw = request.data.decode('utf-8').strip()
        log.info(f'TradingView webhook: {raw}')
        
        # Try JSON first
        try:
            import json
            payload = json.loads(raw)
            action   = payload.get('action','buy').upper()
            symbol   = payload.get('symbol','').upper().strip()
            qty      = int(payload.get('qty', 1))
            strategy = payload.get('strategy', 'tradingview')
        except Exception:
            # Plaintext format: "BUY NVDA 10" or "BUY NVDA"
            parts = raw.upper().split()
            action   = parts[0] if parts else 'BUY'
            symbol   = parts[1] if len(parts) > 1 else ''
            qty      = int(parts[2]) if len(parts) > 2 else 1
            strategy = 'tradingview'

        if not symbol:
            return jsonify({'error': 'no symbol'}), 400
        if action not in ('BUY','SELL'):
            action = 'BUY' if action in ('LONG','ENTER','BUY') else 'SELL'

        if not ib or not _connected:
            log.warning('TradingView signal received but IB not connected')
            return jsonify({'status': 'queued', 'note': 'IB not connected — signal logged'}), 202

        # Route to IB
        from ib_insync import Stock, MarketOrder
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        order    = MarketOrder(action, qty)
        order.orderRef = strategy
        trade = ib.placeOrder(contract, order)
        log.info(f'TradingView → IB: {action} {qty} {symbol}')
        return jsonify({'status': 'submitted', 'order_id': trade.order.orderId})
    except Exception as e:
        log.error(f'Webhook error: {e}')
        return jsonify({'error': str(e)}), 500

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Singularity IB Bridge')
    parser.add_argument('--ib-host',  default='127.0.0.1', help='IB Gateway host')
    parser.add_argument('--ib-port',  type=int, default=7497, help='IB port (7497=paper, 7496=live)')
    parser.add_argument('--account',  default='',  help='IB Account ID (e.g. U1234567)')
    parser.add_argument('--port',     type=int, default=8765, help='Bridge listen port')
    args = parser.parse_args()

    IB_HOST  = args.ib_host
    IB_PORT  = args.ib_port
    ACCOUNT  = args.account

    mode = 'PAPER' if IB_PORT in (7497, 4002) else 'LIVE'
    print(f"""
╔══════════════════════════════════════════════╗
║     SINGULARITY IB BRIDGE  v2.0              ║
║  Mode:    {mode:<35}║
║  IB:      {IB_HOST}:{IB_PORT:<27}║
║  Account: {ACCOUNT or '(not set)':<34}║
║  Bridge:  http://127.0.0.1:{args.port:<18}║
╚══════════════════════════════════════════════╝
Connecting to IB Gateway…
""")

    # Connect in background thread so Flask can start
    t = threading.Thread(target=connect_ib, daemon=True)
    t.start()
    t.join(timeout=8)  # wait up to 8s for IB connection

    # Start keep-alive thread
    ka = threading.Thread(target=keep_alive, daemon=True)
    ka.start()

    print(f'Bridge listening on http://0.0.0.0:{args.port}')
    print('TradingView webhook: http://YOUR_IP:' + str(args.port) + '/webhook/tradingview')
    print('Press Ctrl+C to stop.\n')
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
