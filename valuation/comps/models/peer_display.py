"""
Peer comparison display — computed for every ticker in every bucket
regardless of whether a validated regression exists (universal display
per LAYER_2_FINAL_SPEC). Reuses similarity.find_nearest_peers() for the
5-nearest-peer selection (same 4-feature standardized-distance metric the
original comps engine validated, rather than reimplementing distance
computation), then layers on peer-group multiple medians/percentiles and
a feature-deviation table across the broader 10-feature candidate pool
for interpretability.

"Peer" throughout this module means the 5 NEAREST peers specifically
(by feature distance), not the whole bucket universe — the spec's
pseudocode doesn't scope this precisely; comparing against your closest
comps rather than a heterogeneous full bucket is the standard comps-desk
convention and the more useful number.
"""

import numpy as np

from valuation.comps.similarity import find_nearest_peers
from valuation.comps.output_schema import PeerDisplayOutput

# Broader set for the feature-deviation table (interpretability display,
# not a model input) — every candidate feature that's actually available,
# not just the 4 similarity.py uses for distance.
DEVIATION_FEATURES = [
    "trailing_growth", "operating_margin", "roic", "log_revenue",
    "gross_margin", "ebitda_margin", "r_and_d_intensity",
    "growth_stability", "fcf_conversion", "trailing_growth_5y",
]


def _percentile_of(value, population):
    population = [p for p in population if p is not None]
    if value is None or not population:
        return None
    return float(100.0 * sum(1 for p in population if p <= value) / len(population))


def compute_peer_display(target_features, universe_features, target_multiples, universe_multiples_by_ticker):
    """
    target_multiples: {"ev_sales": float|None, "ev_ebitda": float|None}
    universe_multiples_by_ticker: {ticker: {"ev_sales": .., "ev_ebitda": ..}}
    """
    peers = find_nearest_peers(target_features, universe_features, n=5)
    nearest_peers = [p[0] for p in peers]
    peer_distances = [p[1] for p in peers]

    features_by_ticker = {f.ticker: f for f in universe_features}

    peer_ev_sales = [
        universe_multiples_by_ticker[t]["ev_sales"] for t in nearest_peers
        if universe_multiples_by_ticker.get(t, {}).get("ev_sales") is not None
    ]
    peer_ev_ebitda = [
        universe_multiples_by_ticker[t]["ev_ebitda"] for t in nearest_peers
        if universe_multiples_by_ticker.get(t, {}).get("ev_ebitda") is not None
    ]

    target_ev_sales = target_multiples.get("ev_sales")
    target_ev_ebitda = target_multiples.get("ev_ebitda")

    feature_deviations = {}
    for feature in DEVIATION_FEATURES:
        target_val = getattr(target_features, feature, None)
        peer_vals = [v for v in (getattr(features_by_ticker.get(t), feature, None) for t in nearest_peers) if v is not None]
        if target_val is None or len(peer_vals) < 2:
            continue
        peer_std = float(np.std(peer_vals))
        if peer_std == 0:
            continue
        feature_deviations[feature] = (target_val - float(np.mean(peer_vals))) / peer_std

    return PeerDisplayOutput(
        nearest_peers=nearest_peers,
        peer_distances=peer_distances,
        peer_ev_sales_median=float(np.median(peer_ev_sales)) if peer_ev_sales else None,
        peer_ev_ebitda_median=float(np.median(peer_ev_ebitda)) if peer_ev_ebitda else None,
        target_ev_sales=target_ev_sales,
        target_ev_ebitda=target_ev_ebitda,
        target_ev_sales_percentile=_percentile_of(target_ev_sales, peer_ev_sales),
        target_ev_ebitda_percentile=_percentile_of(target_ev_ebitda, peer_ev_ebitda),
        feature_deviations=feature_deviations,
    )
