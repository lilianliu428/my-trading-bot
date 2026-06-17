"""
api/routers/valuation.py

Valuation endpoints. Currently exposes:
    GET /api/valuation/{ticker}    Single-ticker DCF valuation

Bucket inference is currently a hardcoded map covering ~12 tickers. This is
intentional for v1 — we'll replace it with a lookup against the fundamentals
table (which has GICS sector data) when we wire up the database in session 2.
"""

from fastapi import APIRouter, HTTPException
from api.schemas.valuation import ValuationResponse
from valuation.tech.mature.mature_dcf import compute_tech_intrinsic_value


# v1 bucket inference. Replace with database lookup in session 2.
TICKER_TO_BUCKET = {
    "MSFT":  "mature_tech",
    "AAPL":  "mature_tech",
    "GOOGL": "mature_tech",
    "ORCL":  "mature_tech",
    "META":  "communication",
    "NVDA":  "semiconductors",
    "AMD":   "semiconductors",
    "AVGO":  "semiconductors",
    "INTC":  "semiconductors",
    "MU":    "semiconductors",
    "TXN":   "semiconductors",
    "ADI":   "semiconductors",
}


# APIRouter is FastAPI's way to group related endpoints. They all share the
# same prefix and tag, set once here, instead of repeating on each endpoint.
router = APIRouter(
    prefix="/api/valuation",
    tags=["valuation"],
)


@router.get("/{ticker}", response_model=ValuationResponse)
def get_valuation(ticker: str):
    """
    Compute and return the model's fair value for a single ticker.

    Path parameter:
        ticker: stock ticker (case-insensitive, validated against supported list)

    Returns ValuationResponse. Returns 400 if ticker is not supported,
    500 if the underlying valuation engine throws.
    """
    ticker = ticker.upper()

    if ticker not in TICKER_TO_BUCKET:
        raise HTTPException(
            status_code=400,
            detail=f"Ticker '{ticker}' not in v1 supported list. Supported: {sorted(TICKER_TO_BUCKET.keys())}",
        )

    bucket = TICKER_TO_BUCKET[ticker]

    try:
        result = compute_tech_intrinsic_value(ticker, bucket=bucket)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Valuation engine failed for {ticker}: {type(e).__name__}: {e}",
        )

    if "error" in result:
        raise HTTPException(
            status_code=500,
            detail=f"Valuation engine returned error: {result['error']}",
        )

    # Map the engine's raw dict to the API response shape.
    return ValuationResponse(
        ticker=ticker,
        bucket=bucket,
        fair_value_per_share=result["per_share_value"],
        current_price=result["current_price"],
        upside_downside=result["upside_downside"],
        wacc=result["wacc"],
        high_growth_years=result["growth_profile"]["high_growth_years"],
        initial_growth=result["growth_profile"]["yearly_growth"][0],
        terminal_growth=result["growth_profile"]["yearly_growth"][-1],
	is_heavy_acquirer=result["goodwill_analysis"]["is_heavy_acquirer"],
        data_flags=result.get("data_flags", []),
    )
