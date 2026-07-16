"""
Layer 2: Regression-based comparable valuation engine.

Predicts what multiple (EV/Sales, EV/EBITDA) a company should trade at,
based on how the market prices similar companies, via log-linear regression
on growth/margin/ROIC/size. See LAYER_2_COMPS_SPEC.md for full design.
"""
