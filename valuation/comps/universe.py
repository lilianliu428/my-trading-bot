"""
Comps peer universe — ticker lists per sub-sector bucket.

Ticker sets are kept as plain config so they can be edited without touching
code. v1 selection is manual; see LAYER_2_COMPS_SPEC.md "Universe (Q1)" for
selection criteria (US-listed/USD, TTM revenue > $500M, 3+ years of history,
classified into one of the buckets below).
"""

COMPS_UNIVERSE = {
    "semiconductors": [
        # AI infrastructure
        "NVDA", "AMD", "AVGO", "MRVL", "INTC", "QCOM",
        # Memory
        "MU", "WDC", "STX",
        # Analog / diversified
        "TXN", "ADI", "NXPI", "MCHP", "ON",
        # Equipment (context, not core)
        "AMAT", "LRCX", "KLAC",
        # Foundry / manufacturing
        "TSM", "ASML",
        # Additional peers to reach ~30-40
        "SWKS", "MPWR", "CRUS", "SLAB", "LSCC",
        "ARM", "COHR", "IPGP", "MTSI", "OLED",
        # --- Expansion (valuation/comps/universe_expansion.py) ---
        # Sourced from yf.Industry("semiconductors" /
        # "semiconductor-equipment-materials").top_companies, filtered by
        # classify()=="semiconductors", market cap > $500M, TTM revenue >
        # $100M, and a complete features.py feature vector. 35 tickers.
        "ACLS", "ACMR", "ALAB", "ALGM", "AMBA", "AMKR", "AOSL", "CAMT",
        "CEVA", "COHU", "CRDO", "DIOD", "ENTG", "FORM", "GFS", "ICHR",
        "KLIC", "LASR", "MXL", "NVMI", "ONTO", "PI", "PLAB", "POWI",
        "Q", "QRVO", "RMBS", "SITM", "SKYT", "SMTC", "SYNA", "TER",
        "UCTT", "VECO", "VSH",
    ],
    "mature_tech": [
        "MSFT", "GOOGL", "AAPL", "META", "AMZN",
        "ORCL", "CRM", "ADBE", "IBM", "CSCO",
        # Software
        "NOW", "INTU", "WDAY", "TEAM", "DDOG", "SNOW", "PANW",
        # Additional
        "NFLX", "DIS", "TMUS", "VZ", "T",
        # --- Expansion (valuation/comps/universe_expansion.py) ---
        # Sourced from yf.Industry("software-infrastructure" /
        # "software-application" / "information-technology-services" /
        # "communication-equipment" / "computer-hardware").top_companies,
        # filtered by classify()=="mature_tech", market cap > $500M, TTM
        # revenue > $100M, and a complete features.py feature vector.
        # 73 tickers. Note: classify() has no profitability screen for
        # the non-software industries above (Computer Hardware, IT
        # Services, Communication Equipment), so this list includes some
        # volatile/early-stage names (e.g. IONQ, CIFR, APLD, SMCI) despite
        # the "mature_tech" bucket label — an artifact of the existing
        # classifier, not filtered further here per task scope.
        "AAOI", "ACN", "ADP", "ADSK", "ADTN", "ANET", "APLD", "BDC",
        "BR", "BSY", "CACI", "CDNS", "CDW", "CHKP", "CIEN", "CIFR",
        "CNXC", "CRSR", "CTSH", "DBX", "DELL", "DGII", "DOCN", "DXC",
        "EFOR", "EPAM", "ESTC", "EXLS", "EXTR", "FCCN", "FFIV", "FIS",
        "G", "GILT", "GLOB", "HLIT", "HPE", "HPQ", "IDCC", "INGM",
        "INOD", "IONQ", "IT", "JKHY", "KD", "LDOS", "LITE", "LYFT",
        "MANH", "MSI", "NTAP", "NTGR", "P", "PAYC", "PEGA", "PENG",
        "PSN", "PTC", "QLYS", "ROP", "SAIC", "SMCI", "SSYS", "STX",
        "TDC", "UI", "VIAV", "VISN", "VSAT", "VYX", "WDC", "ZBRA", "ZM",
    ],
    "communication": [
        "META", "GOOGL", "NFLX", "TMUS", "T", "VZ",
        "DIS", "CMCSA", "CHTR", "EA", "ATVI",
    ],
    # Add more buckets as they arise
}

# Minimum TTM revenue for universe inclusion (avoid micro-caps with noisy
# financials). Documented as a selection criterion; v1 lists above are
# curated manually so this isn't enforced programmatically yet.
MIN_REVENUE_TTM = 500e6


def get_universe(bucket, exclude_ticker=None):
    """
    Return the peer ticker list for a bucket, optionally excluding one ticker
    (e.g. the target company itself, if it happens to also sit in the
    universe list).

    Returns an empty list if the bucket has no defined universe.
    """
    tickers = COMPS_UNIVERSE.get(bucket, [])
    if exclude_ticker is None:
        return list(tickers)
    exclude_ticker = exclude_ticker.upper()
    return [t for t in tickers if t.upper() != exclude_ticker]
