"""
cycle_classifier.py

Classify a semiconductor company by industry cycle type.

Why this matters: a generic DCF assumes all companies in a sector have similar
growth profiles, but semis are heterogeneous. Memory companies (MU, WDC) see
3-4 year boom/bust cycles tied to DRAM/NAND pricing. AI-infrastructure-exposed
companies (NVDA, AMD, AVGO, ASML) are riding a 10-15 year data center buildout
analogous to internet infrastructure 1996-2008. Diversified analog companies
(TXN, ADI, NXPI) sell into auto, industrial, and comms end markets with much
muted cyclicality.

Each cycle type gets a different high-growth runway and terminal growth
assumption. This classifier returns that profile for a given ticker.
"""


# Hardcoded ticker → cycle mapping. This is domain judgment, not something
# to derive from data. A semi analyst would just know MU is memory-cycle and
# NVDA is AI-infrastructure. We encode that knowledge directly.
SEMI_CYCLE_MAP = {
    # ─── Memory cycle ─────────────────────────────────────────────────
    # DRAM/NAND pricing drives 3-4 year cycles. Boom/bust dynamics.
    "MU":   "memory",   # Micron - DRAM + NAND, some HBM (AI-adjacent)
    "WDC":  "memory",   # Western Digital - NAND + HDD
    "STX":  "memory",   # Seagate - HDD primarily, similar cycle dynamics

    # ─── AI infrastructure ────────────────────────────────────────────
    # Structural 10-15 year data center buildout.
    "NVDA": "ai_infrastructure",  # GPUs - the picks-and-shovels play
    "AMD":  "ai_infrastructure",  # CPUs + GPUs, increasingly data-center
    "AVGO": "ai_infrastructure",  # custom silicon for hyperscalers
    "ASML": "ai_infrastructure",  # lithography monopoly - all AI fabs use them
    "KLAC": "ai_infrastructure",  # process control equipment
    "LRCX": "ai_infrastructure",  # etch + deposition equipment
    "AMAT": "ai_infrastructure",  # broad semi equipment
    "MRVL": "ai_infrastructure",  # networking + custom data infra chips
    "TSM":  "ai_infrastructure",  # TSMC foundry - manufactures the AI chips
    "ARM":  "ai_infrastructure",  # chip architecture, post-IPO
    "INTC": "ai_infrastructure",  # turnaround story, AI ambitions

    # ─── Diversified analog ───────────────────────────────────────────
    # Multiple end markets (auto/industrial/comms), muted cyclicality.
    "TXN":  "diversified_analog",  # Texas Instruments
    "ADI":  "diversified_analog",  # Analog Devices
    "NXPI": "diversified_analog",  # NXP - auto-heavy
    "MPWR": "diversified_analog",  # Monolithic Power
    "MCHP": "diversified_analog",  # Microchip
    "ON":   "diversified_analog",  # ON Semi - auto/industrial
    "SWKS": "diversified_analog",  # Skyworks - wireless
    "QRVO": "diversified_analog",  # Qorvo - RF
    "QCOM": "diversified_analog",  # Qualcomm - mature mobile
}


# Profile per cycle type: runway length and terminal growth assumption.
# These numbers reflect cycle dynamics, not company-specific quality.
CYCLE_PROFILES = {
    "memory": {
        "runway_years":    6,
        "terminal_growth": 0.020,  # 2.0% - mature, commodity-like
        "rationale":       (
            "Memory cycle (DRAM/NAND). Historical 3-4 year price-driven cycles. "
            "Short runway reflects boom/bust dynamics; long-term growth tracks "
            "global data/storage demand."
        ),
    },
    "ai_infrastructure": {
        "runway_years":    13,
        "terminal_growth": 0.035,  # 3.5% - elevated for infra buildout
        "rationale":       (
            "AI infrastructure cycle. Structural data center buildout, "
            "analogous to internet infrastructure 1996-2008. Long runway "
            "reflects multi-year capex commitments across hyperscalers."
        ),
    },
    "diversified_analog": {
        "runway_years":    9,
        "terminal_growth": 0.028,  # 2.8% - steady mature growth
        "rationale":       (
            "Diversified analog. Exposure across auto, industrial, and comms "
            "end markets. Muted cyclicality, slower but steadier growth than "
            "AI-infrastructure peers."
        ),
    },
}


def classify_semi_cycle(ticker):
    """
    Classify a semiconductor company by industry cycle type.

    Parameters
    ----------
    ticker : str
        Stock ticker (case-insensitive)

    Returns
    -------
    dict with keys:
        cycle           : 'memory' | 'ai_infrastructure' | 'diversified_analog'
        runway_years    : int, high-growth period length
        terminal_growth : float, long-run growth rate (e.g. 0.035 = 3.5%)
        rationale       : str, human-readable explanation
        source          : 'hardcoded' | 'fallback' | 'default'
    """
    ticker = ticker.upper()

    # ─── Step 1: hardcoded lookup ───
    # Covers the ~22 semis we care about. Domain knowledge, not derived.
    if ticker in SEMI_CYCLE_MAP:
        cycle = SEMI_CYCLE_MAP[ticker]
        profile = CYCLE_PROFILES[cycle].copy()
        profile["cycle"]  = cycle
        profile["source"] = "hardcoded"
        return profile

    # ─── Step 2: yfinance fallback ───
    # For unknown semis, try to classify via the industry field. Wrapped in
    # try/except because yfinance can rate-limit or return weird shapes.
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        industry = (info.get("industry") or "").lower()

        # Semi equipment makers ride the same wave as AI-infra companies
        # (their revenue is tied to fab capex, which is currently AI-driven)
        if "semiconductor equipment" in industry or "semiconductor materials" in industry:
            cycle = "ai_infrastructure"
            profile = CYCLE_PROFILES[cycle].copy()
            profile["cycle"]     = cycle
            profile["source"]    = "fallback"
            profile["rationale"] = (
                f"Unknown semi ticker; yfinance industry '{industry}' → "
                f"classified as semi equipment → AI infrastructure cycle."
            )
            return profile

        if "semiconductor" in industry:
            # Generic semi - default to diversified analog (conservative)
            cycle = "diversified_analog"
            profile = CYCLE_PROFILES[cycle].copy()
            profile["cycle"]     = cycle
            profile["source"]    = "fallback"
            profile["rationale"] = (
                f"Unknown semi ticker; yfinance industry '{industry}' → "
                f"classified as generic semi → diversified analog (conservative)."
            )
            return profile
    except Exception:
        # yfinance call failed (rate limit, network, malformed response, etc.)
        # Fall through to final default below.
        pass

    # ─── Step 3: final default ───
    # If we got here, we couldn't classify the ticker at all. Use the most
    # conservative assumption (shorter runway, lower terminal growth).
    cycle = "diversified_analog"
    profile = CYCLE_PROFILES[cycle].copy()
    profile["cycle"]     = cycle
    profile["source"]    = "default"
    profile["rationale"] = (
        f"No classification available for {ticker}; "
        f"defaulting to diversified analog (conservative)."
    )
    return profile