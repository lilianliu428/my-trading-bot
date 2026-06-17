"""
api/schemas/valuation.py

Pydantic models defining the shape of valuation API responses.

Why Pydantic: FastAPI uses these models to (1) validate that the data we
return matches the declared shape, (2) auto-generate the JSON schema shown
in the /docs interface, (3) catch bugs where we accidentally return the
wrong type or forget a field.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ValuationResponse(BaseModel):
    """
    Response shape for GET /api/valuation/{ticker}.

    All monetary values are in USD. Percentages are expressed as decimals
    (e.g. 0.27 = 27%) so the dashboard/bot can format them however it wants.
    """

    ticker: str = Field(..., description="Stock ticker, uppercase")
    bucket: str = Field(..., description="Valuation bucket used (e.g. mature_tech, semiconductors)")

    # Headline numbers
    fair_value_per_share: float = Field(..., description="Model-computed fair value per share, USD")
    current_price: float = Field(..., description="Latest market price per share, USD")
    upside_downside: Optional[float] = Field(
        None,
        description="(fair_value - price) / price. None if fair_value is non-positive."
    )

    # Key drivers (so the dashboard can show 'why')
    wacc: float = Field(..., description="Weighted average cost of capital, decimal")
    high_growth_years: int = Field(..., description="Length of the high-growth phase, years")
    initial_growth: float = Field(..., description="Year-1 revenue growth rate, decimal")
    terminal_growth: float = Field(..., description="Long-run growth rate, decimal")

    # Status
    is_heavy_acquirer: bool = Field(..., description="Whether heavy-acquirer routing fired")
    data_flags: list[str] = Field(default_factory=list, description="Notes from the model about data issues or methodology choices")