"""
valuation/scenarios/models.py

Data classes defining the shape of a scenario and a scenario result.

A Scenario is a named story with:
  - A probability weight (must sum to 1.0 across all scenarios for a ticker)
  - Business driver assumptions (TAM share, end-market margin, etc.)
  - A narrative description (what's the story?)

The scenario's business drivers get translated into DCF inputs by the
translation.py module. The DCF then runs once per scenario; results
get combined into a probability-weighted final output.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScenarioDrivers:
    """
    Business-level assumptions for a single scenario.

    These are the *narrative* inputs (market share, margin, etc.) that
    get translated into DCF inputs (growth rate, runway, etc.) by
    translation.py.
    """
    # Market positioning
    target_year: int                  # year to project to (e.g. 2028)
    target_share: float               # decimal market share by target_year (0-1)
    target_margin: float              # operating margin at target_year (0-1)

    # Optional driver overrides — if you want to bypass the auto-derived DCF
    # inputs for a specific number, set it here. Mostly for one-off tickers.
    override_initial_growth: Optional[float] = None
    override_high_growth_years: Optional[int] = None
    override_terminal_growth: Optional[float] = None


@dataclass
class Scenario:
    """
    A single scenario for one ticker.

    Example:
        Scenario(
            name="Bull",
            probability=0.20,
            narrative="AMD captures 25% accelerator share by 2028, MI300X ramps.",
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.25,
                target_margin=0.28,
            )
        )
    """
    name: str
    probability: float                 # 0.0 to 1.0
    narrative: str                     # human-readable story
    drivers: ScenarioDrivers


@dataclass
class ScenarioResult:
    """
    Output for a single scenario after running the DCF.

    Contains both the inputs (so we know what we ran) and the outputs
    (fair value, key metrics). Used for both end-user display and debugging.
    """
    scenario_name: str
    probability: float
    narrative: str

    # Inputs that were derived from the scenario's drivers
    derived_initial_growth: float
    derived_high_growth_years: int
    derived_terminal_growth: float

    # DCF output for this scenario
    fair_value_per_share: float
    upside_downside: Optional[float]

    # Raw engine output (for debugging/details)
    raw_dcf_result: dict = field(default_factory=dict)


@dataclass
class ScenarioValuationOutput:
    """
    Final scenario-weighted output for a ticker.

    This is what the API/bot/dashboard ultimately consumes.
    """
    ticker: str
    industry: str                      # which industry sizing model was used
    current_price: float

    # Per-scenario results
    scenarios: list[ScenarioResult]

    # Aggregate statistics
    expected_value: float              # probability-weighted fair value
    expected_upside: Optional[float]   # (expected_value - price) / price
    range_low: float                   # min fair value across scenarios
    range_high: float                  # max fair value across scenarios
    mode_scenario: str                 # name of highest-probability scenario

    # Provenance
    data_flags: list[str] = field(default_factory=list)