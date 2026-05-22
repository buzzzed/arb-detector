"""
Kalshi × Polymarket Real-Time Arbitrage Detector
-------------------------------------------------
Polls both prediction-market platforms every POLL_INTERVAL seconds, matches
contracts by title similarity, and surfaces fee-adjusted arbitrage windows.

Endpoints
  GET /          → live dashboard (HTML)
  GET /stream    → SSE stream of market data (consumed by the dashboard)
  GET /api/pairs → current matched pairs as JSON (REST API)
  GET /health    → health check

Run locally:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8001
"""

import asyncio, json, os, re, time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

# ── API roots ──────────────────────────────────────────────────────────────────
KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA_URL  = "https://gamma-api.polymarket.com"

# ── Fee schedule ───────────────────────────────────────────────────────────────
KALSHI_WIN_FEE = 0.07   # 7 % of net profit, deducted at settlement
POLY_TAKER_FEE = 0.02   # 2 % taker fee on trade value, charged upfront

# ── Tuning ─────────────────────────────────────────────────────────────────────
POLL_INTERVAL   = 30     # seconds between full refreshes (be kind to free APIs)
MATCH_THRESHOLD = 0.28   # minimum combined similarity score to pair two markets
MIN_ARB_PROFIT  = 0.003  # flag as arb if guaranteed profit > 0.3 % of notional
MIN_VOLUME      = 10     # skip markets with < $10 lifetime volume
MAX_PAGES       = 5      # API pages to fetch per platform per cycle

# ── Title normalizer ───────────────────────────────────────────────────────────
_STOP = {
    "will","the","a","an","be","is","are","in","on","at","to","for","of","and",
    "or","by","than","that","this","it","its","does","do","did","have","has",
    "had","would","could","should","may","might","can","going","ever","more",
    "most","least","before","after","during","about","whether","win","lose",
    "reach","exceed","fall","rise","what","who","when","where","how","which",
    "any","all","some","end","year","month","week","day","first","last","next",
    "yes","no","get","make","take","give","come","go","see","know","say","from",
    "with","into","then","been","there","their","they","we","he","she","you","i",
}
_ALIASES = {
    "bitcoin": "btc", "ethereum": "eth", "ether": "eth", "solana": "sol",
    "dogecoin": "doge", "ripple": "xrp", "cardano": "ada", "polkadot": "dot",
    "trump": "trump", "donald": "trump", "harris": "harris", "kamala": "harris",
    "biden": "biden",
}

def _tokens(title: str) -> set[str]:
    t = title.lower()
    for k, v in _ALIASES.items():
        t = re.sub(rf"\b{k}\b", v, t)
    t = re.sub(r"[^\w\s\.\$\%]", " ", t)
    return {tok for tok in t.split() if tok not in _STOP and len(tok) > 1}

def _sim(a: str, b: str) -> float:
    sa, sb = _tokens(a), _tokens(b)
    if not sa or not sb:
        return 0.0
    jaccard = len(sa & sb) / len(sa | sb)
    seq     = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return round(0.60 * jaccard + 0.40 * seq, 4)

# ── Kalshi client ──────────────────────────────────────────────────────────────
async def _fetch_kalshi(client: httpx.AsyncClient) -> list[dict]:
    events: list[dict] = []
    cursor: Optional[str] = None
    for _ in range(MAX_PAGES):
        params: dict = {"status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            r = await client.get(f"{KALSHI_URL}/events", params=params, timeout=12)
            r.raise_for_status()
            data = r.json()
        except Exception:
            break
        events.extend(data.get("events", []))
        cursor = data.get("cursor")
        if not cursor or len(data.get("events", [])) < 200:
            break

    async def _event_markets(ev: dict) -> list[dict]:
        try:
            r = await client.get(
                f"{KALSHI_URL}/markets",
                params={"event_ticker": ev["event_ticker"], "limit": 50},
                timeout=10,
            )
            r.raise_for_status()
            markets = r.json().get("markets", [])
            for m in markets:
                m["event_title"] = ev.get("title", m.get("title", ""))
            return markets
        except Exception:
            return []

    all_markets: list[dict] = []
    CHUNK = 20
    for i in range(0, len(events), CHUNK):
        results = await asyncio.gather(*[_event_markets(ev) for ev in events[i : i + CHUNK]])
        for r in results:
            all_markets.extend(r)
    return all_markets

def _norm_kalshi(m: dict) -> Optional[dict]:
    try:
        ya  = float(m.get("yes_ask_dollars") or m.get("yes_ask", 0) or 0)
        na  = float(m.get("no_ask_dollars")  or m.get("no_ask",  0) or 0)
        yb  = float(m.get("yes_bid_dollars") or m.get("yes_bid", 0) or 0)
        nb  = float(m.get("no_bid_dollars")  or m.get("no_bid",  0) or 0)
        vol = float(m.get("volume_fp") or m.get("volume", 0) or 0)
        if ya <= 0 or na <= 0 or vol < MIN_VOLUME:
            return None
        if ya > 1:
            ya, na, yb, nb = ya / 100, na / 100, yb / 100, nb / 100
        ticker    = m.get("ticker", "")
        raw_title = m.get("title", "")
        if re.match(r"^(yes|no)\s+\w", raw_title, re.IGNORECASE):
            return None
        event_title = m.get("event_title") or raw_title
        sub_title   = m.get("yes_sub_title", "").strip()
        is_multi    = (ticker != m.get("event_ticker", ticker)) and sub_title and sub_title.lower() not in ("yes", "no")
        title       = f"{event_title}: {sub_title}" if is_multi else event_title
        return {
            "platform"  : "Kalshi",
            "id"        : ticker,
            "title"     : title,
            "url"       : f"https://kalshi.com/markets/{ticker}",
            "yes_ask"   : ya,
            "no_ask"    : na,
            "yes_bid"   : yb,
            "no_bid"    : nb,
            "volume"    : vol,
            "close_time": m.get("close_time", ""),
        }
    except Exception:
        return None

# ── Polymarket client ──────────────────────────────────────────────────────────
async def _fetch_poly(client: httpx.AsyncClient) -> list[dict]:
    out: list[dict] = []
    offset = 0
    for _ in range(MAX_PAGES):
        try:
            r = await client.get(
                f"{GAMMA_URL}/markets",
                params={"active": "true", "closed": "false", "limit": 100, "offset": offset},
                timeout=12,
            )
            r.raise_for_status()
            batch = r.json()
        except Exception:
            break
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        if len(batch) < 100:
            break
        offset += 100
    return out

def _norm_poly(m: dict) -> Optional[dict]:
    try:
        outcomes = json.loads(m.get("outcomes",      '["Yes","No"]'))
        prices   = json.loads(m.get("outcomePrices", "[0,0]"))
        if len(outcomes) != 2 or len(prices) != 2:
            return None
        vol = float(m.get("volume24hr") or m.get("volume") or 0)
        if vol < MIN_VOLUME:
            return None
        yi  = next((i for i, o in enumerate(outcomes) if o.lower() in ("yes", "true")), 0)
        ni  = 1 - yi
        yp  = float(prices[yi])
        np_ = float(prices[ni])
        if yp <= 0 or np_ <= 0:
            return None
        slug = m.get("slug") or m.get("id", "")
        return {
            "platform"  : "Polymarket",
            "id"        : str(m.get("id", "")),
            "title"     : m.get("question", ""),
            "url"       : f"https://polymarket.com/event/{slug}",
            "yes_ask"   : yp,
            "no_ask"    : np_,
            "yes_bid"   : yp,
            "no_bid"    : np_,
            "volume"    : vol,
            "close_time": m.get("endDate", ""),
        }
    except Exception:
        return None

# ── Market matcher ─────────────────────────────────────────────────────────────
def _match(kalshi: list[dict], poly: list[dict]) -> list[dict]:
    scored: list[tuple[float, dict, dict]] = []
    for km in kalshi:
        for pm in poly:
            s = _sim(km["title"], pm["title"])
            if s >= MATCH_THRESHOLD:
                scored.append((s, km, pm))

    scored.sort(key=lambda x: -x[0])
    used_k:  set[str] = set()
    used_pm: set[str] = set()
    pairs:   list[dict] = []
    for score, km, pm in scored:
        if km["id"] in used_k or pm["id"] in used_pm:
            continue
        used_k.add(km["id"])
        used_pm.add(pm["id"])
        pairs.append({"kalshi": km, "poly": pm, "sim": round(score, 3)})
    return pairs

# ── Arb calculator ─────────────────────────────────────────────────────────────
def _arb(km: dict, pm: dict) -> dict:
    """
    Strategy A: Long YES on Kalshi + Long NO on Polymarket
    Strategy B: Long NO on Kalshi  + Long YES on Polymarket

    Kalshi fee:      7 % of net profit at settlement  → payout = 1 - 0.07*(1-price)
    Polymarket fee:  2 % taker fee upfront            → effective cost = price * 1.02

    Guaranteed profit = min(pnl_if_YES, pnl_if_NO) - total_cost
    """
    K_Y = km["yes_ask"]
    K_N = km["no_ask"]
    P_Y = pm["yes_ask"] * (1 + POLY_TAKER_FEE)
    P_N = pm["no_ask"]  * (1 + POLY_TAKER_FEE)

    ky_payout = 1 - KALSHI_WIN_FEE * (1 - K_Y)
    kn_payout = 1 - KALSHI_WIN_FEE * (1 - K_N)

    cost_a    = K_Y + P_N
    profit_a  = min(ky_payout - cost_a, 1.0 - cost_a)

    cost_b    = K_N + P_Y
    profit_b  = min(1.0 - cost_b, kn_payout - cost_b)

    def _s(label, cost, profit, pnl_y, pnl_n):
        return {
            "strategy"  : label,
            "cost"      : round(cost,   4),
            "min_profit": round(profit, 4),
            "profit_pct": round(profit * 100, 2),
            "pnl_if_yes": round(pnl_y, 4),
            "pnl_if_no" : round(pnl_n, 4),
            "is_arb"    : profit > MIN_ARB_PROFIT,
        }

    return {
        "A"              : _s("YES Kalshi / NO Polymarket",  cost_a, profit_a, ky_payout - cost_a, 1.0 - cost_a),
        "B"              : _s("NO Kalshi  / YES Polymarket", cost_b, profit_b, 1.0 - cost_b, kn_payout - cost_b),
        "best_profit_pct": round(max(profit_a, profit_b) * 100, 2),
        "has_arb"        : profit_a > MIN_ARB_PROFIT or profit_b > MIN_ARB_PROFIT,
        "yes_spread"     : round(km["yes_ask"] - pm["yes_ask"], 4),
    }

# ── Shared in-memory state ─────────────────────────────────────────────────────
_state: dict = {
    "ts": 0.0, "kalshi_n": 0, "poly_n": 0,
    "matched_n": 0, "arb_n": 0, "pairs": [], "error": None,
}

# ── Background poller ──────────────────────────────────────────────────────────
async def _poll() -> None:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        while True:
            try:
                kalshi_raw, poly_raw = await asyncio.gather(
                    _fetch_kalshi(client), _fetch_poly(client)
                )

                # Run the CPU-bound normalize → match → analyze pipeline in a
                # worker thread so /health and /stream stay responsive on
                # single-core hosts.
                def _crunch():
                    k  = [x for m in kalshi_raw if (x := _norm_kalshi(m))]
                    p_ = [x for m in poly_raw   if (x := _norm_poly(m))]
                    pp = _match(k, p_)
                    out = [{**p, "arb": _arb(p["kalshi"], p["poly"])} for p in pp]
                    out.sort(key=lambda x: (not x["arb"]["has_arb"], -x["arb"]["best_profit_pct"]))
                    return k, p_, out

                kalshi, poly, analyzed = await asyncio.to_thread(_crunch)

                _state.update({
                    "ts"       : time.time(),
                    "kalshi_n" : len(kalshi),
                    "poly_n"   : len(poly),
                    "matched_n": len(analyzed),
                    "arb_n"    : sum(1 for x in analyzed if x["arb"]["has_arb"]),
                    "pairs"    : analyzed[:500],
                    "error"    : None,
                })
            except Exception as exc:
                _state["error"] = str(exc)

            await asyncio.sleep(POLL_INTERVAL)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Arb Detector — Kalshi × Polymarket")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def _start() -> None:
    asyncio.create_task(_poll())

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse((Path(__file__).parent / "dashboard.html").read_text())

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": _state["ts"], "kalshi_n": _state["kalshi_n"], "poly_n": _state["poly_n"]})

@app.get("/api/pairs")
async def api_pairs() -> JSONResponse:
    """REST endpoint — returns current matched pairs and arb analysis."""
    return JSONResponse({
        "ts"       : _state["ts"],
        "kalshi_n" : _state["kalshi_n"],
        "poly_n"   : _state["poly_n"],
        "matched_n": _state["matched_n"],
        "arb_n"    : _state["arb_n"],
        "pairs"    : _state["pairs"],
    })

@app.get("/stream")
async def stream(request: Request) -> EventSourceResponse:
    async def gen():
        last_ts = 0.0
        while True:
            if await request.is_disconnected():
                break
            if _state["ts"] != last_ts:
                last_ts = _state["ts"]
                yield {"event": "update", "data": json.dumps(_state, default=str)}
            await asyncio.sleep(1)
    return EventSourceResponse(gen())
