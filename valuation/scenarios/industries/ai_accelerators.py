"""
valuation/scenarios/industries/ai_accelerators.py

Industry sizing model + scenarios for AI accelerator companies.

This file is responsible for two things:
1. TAM_BY_YEAR — total addressable market for AI accelerators by year.
   These are PLACEHOLDER numbers for v1. They need to be refined with real
   data (hyperscaler capex projections, analyst forecasts) in session 2.
2. SCENARIOS_BY_TICKER — per-ticker scenarios with probability + drivers.
   These define each scenario's narrative and target share/margin.

Companies covered: AMD, NVDA, AVGO (the three pure-play AI accelerator names
we're testing the framework on first).
"""

from valuation.scenarios.models import Scenario, ScenarioDrivers


# ════════════════════════════════════════════════════════════════════════
# TAM SIZING MODEL (v1 placeholders — refine with real data in session 2)
# ════════════════════════════════════════════════════════════════════════
# These numbers reflect rough thinking about the AI accelerator market:
# - 2025: ~$200B (driven by NVDA's run rate + competitive products)
# - 2028: ~$500B (consensus AI capex forecasts roughly imply this)
# - 2030: ~$700B (continued buildout, but slowing growth)
#
# Real numbers would come from: Dell'Oro Group reports, hyperscaler capex
# guidance ($MSFT/$META/$GOOG/$AMZN total capex × % AI accelerators),
# AI training compute requirements, inference workload scaling laws.
#
# For v1: placeholders. The framework works; the numbers need refining.
# ════════════════════════════════════════════════════════════════════════

TAM_BY_YEAR = {
    2025: 200e9,    # $200B
    2026: 280e9,    # ramp
    2027: 380e9,    # acceleration
    2028: 500e9,    # peak buildout
    2029: 620e9,
    2030: 700e9,    # continued growth, decelerating
}


def get_tam(year: int) -> float:
    """Get TAM for a given year, with linear interpolation between known years."""
    if year in TAM_BY_YEAR:
        return TAM_BY_YEAR[year]
    # Extrapolate beyond known range with 8% annual growth (conservative)
    known_years = sorted(TAM_BY_YEAR.keys())
    if year > known_years[-1]:
        years_beyond = year - known_years[-1]
        return TAM_BY_YEAR[known_years[-1]] * (1.08 ** years_beyond)
    if year < known_years[0]:
        years_before = known_years[0] - year
        return TAM_BY_YEAR[known_years[0]] / (1.30 ** years_before)
    # Interpolate within range
    lower_year = max(y for y in known_years if y < year)
    upper_year = min(y for y in known_years if y > year)
    fraction = (year - lower_year) / (upper_year - lower_year)
    return TAM_BY_YEAR[lower_year] + fraction * (TAM_BY_YEAR[upper_year] - TAM_BY_YEAR[lower_year])


# ════════════════════════════════════════════════════════════════════════
# SCENARIOS BY TICKER
# ════════════════════════════════════════════════════════════════════════
# For each ticker, four scenarios that span the realistic range of outcomes.
# Probabilities should sum to 1.0 per ticker.
#
# Share assumptions reflect *AI accelerator market share*, not overall
# semiconductor share. AMD bull case = 25% by 2028 (Lilian's input).
# NVDA's scenarios reflect that it's the incumbent — share losses are
# more catastrophic than share gains; bear case is "NVDA loses share".
# ════════════════════════════════════════════════════════════════════════

SCENARIOS_BY_TICKER = {
    # ─── AMD ──────────────────────────────────────────────────────────
    "AMD": [
        Scenario(
            name="Bear",
            probability=0.25,
            narrative=(
                "NVDA Blackwell extends technical lead. MI300X loses hyperscaler "
                "design wins. AMD AI accelerator revenue plateaus at ~$5B/year. "
                "Share falls to 2% by 2028."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.02,
                target_margin=0.15,  # weak margins in marginal share position
            ),
        ),
        Scenario(
            name="Base",
            probability=0.50,
            narrative=(
                "AMD holds current trajectory. MI300X ramps as expected with "
                "Meta, Microsoft as anchor customers. Captures ~8% share by 2028. "
                "Software stack (ROCm) matures but doesn't catch CUDA."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.08,
                target_margin=0.22,
            ),
        ),
        Scenario(
            name="Bull",
            probability=0.20,
            narrative=(
                "Lilian's bull case: AMD captures 25% of AI accelerator market "
                "by 2028 (~$125B revenue from accelerators alone, implied stock "
                "price ~$800). Software stack reaches parity. Multi-vendor "
                "diversification by hyperscalers drives massive share gains."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.25,
                target_margin=0.28,
            ),
        ),
        Scenario(
            name="Moonshot",
            probability=0.05,
            narrative=(
                "AMD becomes the AI inference standard. While NVDA dominates "
                "training, inference workloads (10x training in volume) shift "
                "to AMD on cost. Captures 40% share by 2028."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.40,
                target_margin=0.32,
            ),
        ),
    ],

    # ─── NVDA ─────────────────────────────────────────────────────────
    "NVDA": [
        Scenario(
            name="Bear",
            probability=0.20,
            narrative=(
                "Competitive pressure compounds. AMD MI300X + custom hyperscaler "
                "silicon (AVGO TPU, MSFT Maia, GOOGL TPU) erode share. NVDA "
                "falls to 55% share by 2028. Pricing power compresses margins."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.55,
                target_margin=0.45,  # margin compression but still high
            ),
        ),
        Scenario(
            name="Base",
            probability=0.50,
            narrative=(
                "NVDA maintains dominance. Blackwell → Rubin roadmap holds. "
                "Holds 70% share through 2028. CUDA moat persists for training. "
                "Inference market sees some share loss but training stays sticky."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.70,
                target_margin=0.55,
            ),
        ),
        Scenario(
            name="Bull",
            probability=0.25,
            narrative=(
                "AI buildout exceeds expectations. NVDA extends technical lead. "
                "Holds 80% share. Margins expand on software/networking attach. "
                "Total addressable market grows faster than consensus."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.80,
                target_margin=0.60,
            ),
        ),
        Scenario(
            name="Moonshot",
            probability=0.05,
            narrative=(
                "AI capex compounds beyond expectations. ASI/AGI race drives "
                "training compute requirements 10x. NVDA captures 85% share at "
                "expanded margins. Becomes the most valuable company in history."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.85,
                target_margin=0.62,
            ),
        ),
    ],

    # ─── AVGO ─────────────────────────────────────────────────────────
    "AVGO": [
        Scenario(
            name="Bear",
            probability=0.20,
            narrative=(
                "Custom silicon thesis fizzles. Hyperscalers consolidate back "
                "to NVDA's GPU platform. AVGO custom AI revenue stays at "
                "current ~$10B. VMware integration challenges drag overall."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.04,
                target_margin=0.25,
            ),
        ),
        Scenario(
            name="Base",
            probability=0.50,
            narrative=(
                "AVGO captures ~10% AI accelerator share via custom silicon "
                "deals (GOOGL TPU, MSFT Maia, others). Becomes default option "
                "for hyperscalers wanting non-NVDA alternative. Strong margins."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.10,
                target_margin=0.32,
            ),
        ),
        Scenario(
            name="Bull",
            probability=0.25,
            narrative=(
                "Custom silicon becomes dominant pattern as hyperscalers "
                "reject NVDA pricing. AVGO captures 18% share. Wins multiple "
                "new hyperscaler design wins (AAPL?, Oracle?, Tesla?)."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.18,
                target_margin=0.36,
            ),
        ),
        Scenario(
            name="Moonshot",
            probability=0.05,
            narrative=(
                "Custom silicon supplants merchant silicon for inference. "
                "AVGO becomes the new NVDA for inference workloads at "
                "hyperscaler scale. Captures 25% share."
            ),
            drivers=ScenarioDrivers(
                target_year=2028,
                target_share=0.25,
                target_margin=0.40,
            ),
        ),
    ],
}


def get_scenarios(ticker: str) -> list[Scenario]:
    """Return the list of scenarios for a ticker. Raises if ticker not configured."""
    ticker = ticker.upper()
    if ticker not in SCENARIOS_BY_TICKER:
        raise KeyError(
            f"No AI accelerator scenarios configured for {ticker}. "
            f"Supported: {list(SCENARIOS_BY_TICKER.keys())}"
        )
    scenarios = SCENARIOS_BY_TICKER[ticker]

    # Sanity check: probabilities should sum to 1.0
    total = sum(s.probability for s in scenarios)
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"Scenario probabilities for {ticker} sum to {total}, not 1.0. "
            f"Fix in SCENARIOS_BY_TICKER."
        )

    return scenarios