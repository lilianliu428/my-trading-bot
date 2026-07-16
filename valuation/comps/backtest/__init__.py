"""
Point-in-time backtest for the Layer 2 comps regression.

Data-reality note (see LAYER_2_COMPS_SPEC.md Q6 fallback clause): yfinance's
annual statements only go back ~5 years, not the 10 originally scoped. This
module clips the requested backtest window to whatever fundamentals are
actually available and flags the shortfall in BacktestResult.data_flags,
rather than reaching for a paid alternative data source.
"""
