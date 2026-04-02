"""
api/surveillance.py — AEQUITAS Market Surveillance Engine
All routes consolidated into one serverless function (under 12 limit).

Routes (all GET):
  /api/surveillance/indices  — global indices across 3 timezones
  /api/surveillance/sectors  — sector ETF heat map
  /api/surveillance/scan     — unusual volume + big money scanner
  /api/surveillance/insiders — SEC Form 4 insider filings
  /api/surveillance/news     — RSS news + sector classification
"""
import sys, os, math, json, urllib.request, re
from http.server import BaseHTTPRequestHandler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from helpers import send_json, send_err, send_cors, get_qs

# ── Sector / keyword classification ──────────────────────────────────
SECTOR_KEYWORDS = {
    'energy_oil':   ['oil','petroleum','crude','opec','pipeline','refin','chevron','exxon','shell','bp ','energy','lng','natural gas','barrel'],
    'defense':      ['defense','defence','military','pentagon','weapon','missile','fighter','nato','army','navy','air force','lockheed','boeing','raytheon','northrop','l3','general dynamics'],
    'tech':         ['apple','microsoft','google','alphabet','nvidia','semiconductor','chip','ai','artificial intelligence','cloud','software','meta','amazon','tesla','cyber'],
    'finance':      ['fed','federal reserve','interest rate','bank','jpmorgan','goldman','morgan stanley','blackrock','hedge fund','private equity','ipo','spac','lending'],
    'pharma_health':['fda','drug','pharmaceutical','biotech','clinical trial','approval','vaccine','cancer','merger','acquisition','pfizer','moderna','eli lilly'],
    'real_estate':  ['real estate','reit','housing','mortgage','interest rate','construction','property'],
    'consumer':     ['retail','consumer','walmart','target','amazon','e-commerce','spending','inflation','cpi'],
    'commodities':  ['gold','silver','copper','wheat','corn','soybean','commodity','futures','lithium','uranium'],
    'macro':        ['recession','gdp','inflation','unemployment','federal reserve','treasury','yield curve','dollar','currency','forex'],
    'geopolitical': ['war','conflict','sanction','china','russia','taiwan','ukraine','iran','israel','election','tariff','trade'],
}

def classify_sector(text):
    text_lower = text.lower()
    scores = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[sector] = score
    if not scores:
        return 'general'
    return max(scores, key=scores.get)

SECTOR_LABELS = {
    'energy_oil': 'Energy / Oil', 'defense': 'Defense / Military',
    'tech': 'Technology', 'finance': 'Finance / Banking',
    'pharma_health': 'Pharma / Health', 'real_estate': 'Real Estate',
    'consumer': 'Consumer / Retail', 'commodities': 'Commodities',
    'macro': 'Macro / Rates', 'geopolitical': 'Geopolitical',
    'general': 'General Market',
}

def sentiment_score(text):
    """Simple rule-based sentiment — positive/negative word counting."""
    pos = ['beat','surge','soar','rally','gain','rise','profit','growth','record','strong',
           'outperform','upgrade','buy','bullish','boom','approval','win','exceed','positive']
    neg = ['miss','fall','drop','crash','loss','decline','cut','downgrade','sell','bearish',
           'recession','layoff','bankrupt','fraud','investigation','warning','below','concern']
    t   = text.lower()
    ps  = sum(1 for w in pos if w in t)
    ns  = sum(1 for w in neg if w in t)
    if ps > ns: return 'bullish'
    if ns > ps: return 'bearish'
    return 'neutral'

def _safe(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    return v

# ── INDICES ───────────────────────────────────────────────────────────
INDICES = {
    'asia': [
        ('^N225',    'Nikkei 225',      'JP'),
        ('^HSI',     'Hang Seng',       'HK'),
        ('000001.SS','Shanghai Comp.',  'CN'),
        ('^AXJO',    'ASX 200',         'AU'),
        ('^KS11',    'KOSPI',           'KR'),
    ],
    'europe': [
        ('^GDAXI',   'DAX',             'DE'),
        ('^FTSE',    'FTSE 100',        'GB'),
        ('^FCHI',    'CAC 40',          'FR'),
        ('^STOXX50E','Euro Stoxx 50',   'EU'),
        ('^IBEX',    'IBEX 35',         'ES'),
    ],
    'us': [
        ('^GSPC',    'S&P 500',         'US'),
        ('^IXIC',    'NASDAQ',          'US'),
        ('^DJI',     'Dow Jones',       'US'),
        ('^RUT',     'Russell 2000',    'US'),
        ('^VIX',     'VIX',             'US'),
        ('DXY=X',    'Dollar Index',    'US'),
        ('^TNX',     '10Y Treasury',    'US'),
        ('GLD',      'Gold',            'US'),
        ('CL=F',     'WTI Crude',       'US'),
    ],
}

def fetch_indices():
    import yfinance as yf
    result = {}
    for region, tickers in INDICES.items():
        region_data = []
        symbols = [t[0] for t in tickers]
        try:
            raw = yf.download(symbols, period='5d', interval='1d',
                              auto_adjust=True, progress=False, threads=True)
            closes = raw['Close'] if hasattr(raw.columns, 'levels') else raw
            for sym, name, country in tickers:
                try:
                    col = sym if sym in closes.columns else closes.columns[0]
                    prices = closes[sym].dropna() if sym in closes.columns else None
                    if prices is None or len(prices) < 2:
                        region_data.append({'sym':sym,'name':name,'country':country,
                                            'price':None,'chg':None,'chg_pct':None,'error':'no data'})
                        continue
                    px    = float(prices.iloc[-1])
                    px_1  = float(prices.iloc[-2])
                    chg   = px - px_1
                    chg_p = chg / px_1 * 100 if px_1 else 0
                    region_data.append({
                        'sym': sym, 'name': name, 'country': country,
                        'price':   round(px, 4) if px < 10000 else round(px, 0),
                        'chg':     round(chg, 4),
                        'chg_pct': round(chg_p, 2),
                        'alert':   abs(chg_p) > 1.5,
                    })
                except Exception as e:
                    region_data.append({'sym':sym,'name':name,'country':country,'error':str(e)})
        except Exception as e:
            region_data = [{'error': str(e)}]
        result[region] = region_data
    return result

# ── SECTORS ───────────────────────────────────────────────────────────
SECTOR_ETFS = [
    ('XLK',  'Technology'),    ('XLF',  'Financials'),  ('XLE',  'Energy'),
    ('XLV',  'Healthcare'),    ('XLI',  'Industrials'),  ('XLY',  'Consumer Disc.'),
    ('XLP',  'Consumer Stpl'),  ('XLB',  'Materials'),   ('XLU',  'Utilities'),
    ('XLRE', 'Real Estate'),   ('XLC',  'Comm. Services'),
    ('ITA',  'Aerospace/Def.'), ('XBI',  'Biotech'),     ('GLD',  'Gold'),
    ('USO',  'Oil'),           ('TLT',  'Long Bonds'),   ('UUP',  'US Dollar'),
]

def fetch_sectors():
    import yfinance as yf, numpy as np
    symbols = [s for s,_ in SECTOR_ETFS]
    try:
        raw    = yf.download(symbols, period='5d', interval='1d',
                              auto_adjust=True, progress=False, threads=True)
        closes = raw['Close'] if hasattr(raw.columns, 'levels') else raw
        vols   = raw['Volume'] if hasattr(raw.columns, 'levels') else None
        result = []
        for sym, name in SECTOR_ETFS:
            try:
                prices = closes[sym].dropna() if sym in closes.columns else None
                if prices is None or len(prices) < 2:
                    result.append({'sym':sym,'name':name,'chg_pct':None,'volume':None})
                    continue
                px    = float(prices.iloc[-1])
                px_1  = float(prices.iloc[-2])
                chg_p = (px - px_1) / px_1 * 100 if px_1 else 0
                vol   = None
                if vols is not None and sym in vols.columns:
                    vol_s = vols[sym].dropna()
                    vol   = int(vol_s.iloc[-1]) if len(vol_s) > 0 else None
                # 5d trend
                week_chg = (px / float(prices.iloc[0]) - 1) * 100 if len(prices) >= 5 else 0
                result.append({
                    'sym': sym, 'name': name,
                    'price':    round(px, 2),
                    'chg_pct':  round(chg_p, 2),
                    'week_pct': round(week_chg, 2),
                    'volume':   vol,
                    'signal':   'bullish' if chg_p > 0.5 else ('bearish' if chg_p < -0.5 else 'neutral'),
                })
            except Exception as e:
                result.append({'sym':sym,'name':name,'error':str(e)})
        return result
    except Exception as e:
        return [{'error': str(e)}]

# ── UNUSUAL VOLUME SCANNER ────────────────────────────────────────────
SCAN_UNIVERSE = [
    # Mega cap + most liquid
    'AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','AVGO','JPM','V',
    'UNH','XOM','LLY','MA','JNJ','PG','HD','COST','MRK','ABBV',
    'BAC','CRM','NFLX','AMD','ORCL','CVX','KO','PEP','WMT','DIS',
    # ETFs (show big money flows)
    'SPY','QQQ','IWM','GLD','SLV','USO','TLT','HYG','EEM','FXI',
    # Defense + geo-sensitive
    'LMT','RTX','NOC','BA','GD','L3T','KTOS','PLTR',
    # Energy
    'XOM','CVX','COP','EOG','SLB','HAL','MPC','VLO',
    # Finance
    'JPM','GS','MS','BAC','WFC','C','BLK','SCHW',
    # Biotech (FDA catalyst)
    'MRNA','BNTX','REGN','GILD','BIIB','VRTX','AMGN',
]
SCAN_UNIVERSE = list(dict.fromkeys(SCAN_UNIVERSE))  # dedupe

def fetch_unusual_volume():
    import yfinance as yf, numpy as np
    alerts = []
    # Download 21 days to get rolling average
    try:
        raw    = yf.download(SCAN_UNIVERSE, period='21d', interval='1d',
                              auto_adjust=True, progress=False, threads=True)
        closes = raw['Close']  if hasattr(raw.columns,'levels') else raw
        vols   = raw['Volume'] if hasattr(raw.columns,'levels') else None
        if vols is None:
            return {'alerts':[],'error':'No volume data'}

        for sym in SCAN_UNIVERSE:
            try:
                if sym not in vols.columns or sym not in closes.columns:
                    continue
                v_series = vols[sym].dropna()
                p_series = closes[sym].dropna()
                if len(v_series) < 5 or len(p_series) < 2:
                    continue

                v_today  = float(v_series.iloc[-1])
                v_20avg  = float(v_series.iloc[:-1].mean())
                v_median = float(v_series.iloc[:-1].median())
                px_today = float(p_series.iloc[-1])
                px_prev  = float(p_series.iloc[-2])
                chg_pct  = (px_today - px_prev) / px_prev * 100 if px_prev else 0
                notional = v_today * px_today   # dollar volume today

                # Trigger conditions
                vol_ratio     = v_today / v_20avg if v_20avg > 0 else 0
                above_median  = v_today > max(v_median * 1.5, 100_000)
                big_notional  = notional > 1_000_000
                vol_spike     = vol_ratio > 2.5  # 2.5× average

                if (vol_spike or above_median) and big_notional:
                    # Classify likely category
                    ticker_info = yf.Ticker(sym)
                    info        = ticker_info.info or {}
                    sector      = info.get('sector', 'Unknown')
                    industry    = info.get('industry', '')
                    name        = info.get('shortName', sym)

                    severity = 'extreme' if vol_ratio > 5 else ('high' if vol_ratio > 3 else 'elevated')
                    alerts.append({
                        'sym':          sym,
                        'name':         name,
                        'sector':       sector,
                        'industry':     industry,
                        'price':        round(px_today, 2),
                        'chg_pct':      round(chg_pct, 2),
                        'volume':       int(v_today),
                        'vol_avg_20d':  int(v_20avg),
                        'vol_ratio':    round(vol_ratio, 1),
                        'notional_mm':  round(notional / 1_000_000, 1),
                        'severity':     severity,
                        'direction':    'buy' if chg_pct > 0.3 else ('sell' if chg_pct < -0.3 else 'neutral'),
                        'flag':         'INSIDER?' if (vol_ratio > 4 and abs(chg_pct) < 1.0) else
                                        ('BIG MOVE' if abs(chg_pct) > 3 else 'UNUSUAL VOL'),
                    })
            except Exception:
                continue

        # Sort by vol_ratio descending
        alerts.sort(key=lambda x: x.get('vol_ratio', 0), reverse=True)
        return {'alerts': alerts[:30], 'scanned': len(SCAN_UNIVERSE), 'triggered': len(alerts)}
    except Exception as e:
        import traceback
        return {'alerts': [], 'error': str(e), 'trace': traceback.format_exc().splitlines()[-1]}

# ── SEC INSIDER FILINGS ───────────────────────────────────────────────
def fetch_insiders():
    """
    Pull recent Form 4 (insider transactions) and 13F (institutional holdings)
    from SEC EDGAR RSS feeds — free, no API key needed.
    """
    filings = []
    headers = {'User-Agent': 'AEQUITAS Research aequitas@research.io'}

    # Form 4 recent filings feed
    try:
        url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&dateb=&owner=include&count=40&search_text=&output=atom'
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml = resp.read().decode('utf-8', errors='ignore')

        # Parse entries from Atom feed
        entries = re.findall(r'<entry>(.*?)</entry>', xml, re.DOTALL)
        for entry in entries[:20]:
            title   = re.search(r'<title>(.*?)</title>', entry)
            updated = re.search(r'<updated>(.*?)</updated>', entry)
            link    = re.search(r'<link[^>]*href="([^"]+)"', entry)
            summary = re.search(r'<summary[^>]*>(.*?)</summary>', entry, re.DOTALL)

            title_txt   = re.sub(r'<[^>]+>', '', title.group(1))   if title   else ''
            updated_txt = updated.group(1)[:10]                     if updated else ''
            link_txt    = link.group(1)                             if link    else ''
            summary_txt = re.sub(r'<[^>]+>', '', summary.group(1)) if summary else ''

            # Extract ticker if visible in title/summary
            ticker_match = re.search(r'\b([A-Z]{1,5})\b', title_txt)
            ticker = ticker_match.group(1) if ticker_match else '—'

            # Skip if no real content
            if not title_txt.strip():
                continue

            filings.append({
                'type':    'Form 4',
                'title':   title_txt.strip(),
                'date':    updated_txt,
                'ticker':  ticker,
                'link':    link_txt,
                'summary': summary_txt.strip()[:200],
                'sentiment': sentiment_score(title_txt + ' ' + summary_txt),
            })
    except Exception as e:
        filings.append({'type': 'Form 4', 'error': str(e), 'title': 'SEC feed unavailable'})

    # Known whale recent 13F notes (static curated list — 13F filed quarterly)
    whales = [
        {'name':'Berkshire Hathaway (Buffett)', 'recent':'Added OXY, AAPL, BAC. Exited HP. Q4 2024.', 'focus':'Value / Energy / Finance'},
        {'name':'Pershing Square (Ackman)',     'recent':'New position in Alphabet. Exited bonds. Q4 2024.', 'focus':'Concentrated equity'},
        {'name':'Bridgewater Associates',       'recent':'Increased SPY, GLD. Reduced EEM. Q4 2024.', 'focus':'All-weather / Macro'},
        {'name':'Tiger Global',                 'recent':'Added Microsoft, NVDA. Reduced consumer names.', 'focus':'Tech growth'},
        {'name':'Elliott Management',           'recent':'Activist stake in Starbucks, BP. Q4 2024.', 'focus':'Activist / Special situations'},
        {'name':'Citadel (Griffin)',             'recent':'Rotated into energy, defense. Q4 2024.', 'focus':'Multi-strategy'},
        {'name':'Point72 (Cohen)',               'recent':'New healthcare positions. Q4 2024.', 'focus':'Long/short equity'},
    ]

    return {'form4': filings[:15], 'whales': whales}

# ── NEWS RSS FEED ─────────────────────────────────────────────────────
NEWS_FEEDS = [
    ('Reuters',     'https://feeds.reuters.com/reuters/businessNews'),
    ('Reuters Mkts','https://feeds.reuters.com/reuters/usDomesticNews'),
    ('AP Business', 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml'),
    ('MarketWatch', 'https://feeds.marketwatch.com/marketwatch/topstories'),
    ('Seeking Alpha','https://seekingalpha.com/market_currents.xml'),
]

def fetch_news():
    headlines = []
    headers   = {'User-Agent': 'Mozilla/5.0 AEQUITAS/1.0'}

    for source, url in NEWS_FEEDS:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml = resp.read().decode('utf-8', errors='ignore')

            items = re.findall(r'<item>(.*?)</item>', xml, re.DOTALL)
            for item in items[:8]:
                title   = re.search(r'<title[^>]*><!\[CDATA\[(.*?)\]\]></title>|<title[^>]*>(.*?)</title>', item, re.DOTALL)
                pubdate = re.search(r'<pubDate>(.*?)</pubDate>', item)
                link    = re.search(r'<link[^>]*>(.*?)</link>|<link[^>]*/>', item)
                desc    = re.search(r'<description[^>]*><!\[CDATA\[(.*?)\]\]></description>|<description[^>]*>(.*?)</description>', item, re.DOTALL)

                title_txt = ''
                if title:
                    title_txt = (title.group(1) or title.group(2) or '').strip()
                title_txt = re.sub(r'<[^>]+>', '', title_txt).strip()
                if not title_txt or len(title_txt) < 10: continue

                desc_txt = ''
                if desc:
                    desc_txt = (desc.group(1) or desc.group(2) or '').strip()
                desc_txt = re.sub(r'<[^>]+>', '', desc_txt)[:300]

                date_txt = pubdate.group(1)[:16] if pubdate else ''
                link_txt = ''
                if link:
                    link_txt = (link.group(1) or '').strip()

                full_text = title_txt + ' ' + desc_txt
                sector    = classify_sector(full_text)
                sent      = sentiment_score(full_text)

                # Keywords that flag as high-impact
                impact_words = ['acquisition','merger','ipo','fda','billion','trillion',
                                'crash','surge','record','ban','sanction','war','conflict',
                                'beat','miss','guidance','layoff','bankrupt','investigation']
                impact = any(w in full_text.lower() for w in impact_words)

                headlines.append({
                    'source':    source,
                    'title':     title_txt,
                    'desc':      desc_txt[:200],
                    'date':      date_txt,
                    'link':      link_txt,
                    'sector':    sector,
                    'sector_label': SECTOR_LABELS.get(sector, sector),
                    'sentiment': sent,
                    'impact':    impact,
                })
        except Exception as e:
            headlines.append({'source': source, 'error': str(e), 'title': f'{source} feed unavailable'})

    headlines.sort(key=lambda x: x.get('impact', False), reverse=True)
    return headlines[:40]


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)

    def do_GET(self):
        p = self.path
        try:
            if '/indices' in p:
                return send_json(self, fetch_indices())
            elif '/sectors' in p:
                return send_json(self, {'sectors': fetch_sectors()})
            elif '/scan' in p:
                return send_json(self, fetch_unusual_volume())
            elif '/insiders' in p:
                return send_json(self, fetch_insiders())
            elif '/news' in p:
                return send_json(self, {'headlines': fetch_news()})
            # Default: return summary of all (lighter)
            return send_json(self, {'endpoints': [
                '/api/surveillance/indices',
                '/api/surveillance/sectors',
                '/api/surveillance/scan',
                '/api/surveillance/insiders',
                '/api/surveillance/news',
            ]})
        except Exception as e:
            import traceback
            send_err(self, str(e) + ' | ' + traceback.format_exc().splitlines()[-1])

    def log_message(self, *a): pass
