"""
Peer similarity — z-score-standardized Euclidean distance across the 4
regression features. Display-only: peers do NOT feed back into the fair
value estimate. Disagreement between the regression prediction and the peer
group is information, not error.
"""

import numpy as np

from valuation.comps.regression import FEATURE_ORDER


def find_nearest_peers(target, universe, n=5):
    """
    Return top-n most similar companies to target, by z-score-standardized
    Euclidean distance across the 4 features. Excludes target itself.

    Mean/std for standardization are computed across `universe` (companies
    with a complete feature vector). Companies with incomplete features are
    dropped from the candidate pool. Returns [(ticker, distance), ...]
    sorted by distance ascending — fewer than n if the pool is smaller.
    """
    candidates = [c for c in universe if c.ticker != target.ticker and c.is_complete()]
    if not candidates:
        return []

    X = np.array([[getattr(c, f) for f in FEATURE_ORDER] for c in candidates], dtype=float)
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    # Avoid divide-by-zero for a feature that's constant across the universe.
    safe_stds = np.where(feature_stds == 0, 1.0, feature_stds)

    X_std = (X - feature_means) / safe_stds

    if not target.is_complete():
        return []
    target_vec = np.array([getattr(target, f) for f in FEATURE_ORDER], dtype=float)
    target_std = (target_vec - feature_means) / safe_stds

    distances = np.sqrt(np.sum((X_std - target_std) ** 2, axis=1))

    ranked = sorted(zip([c.ticker for c in candidates], distances.tolist()), key=lambda pair: pair[1])
    return ranked[:n]


if __name__ == "__main__":
    from valuation.comps.features import compute_features_for_ticker
    from valuation.comps.universe import get_universe

    bucket = "semiconductors"
    universe_tickers = get_universe(bucket, exclude_ticker="AMD")
    universe_features = [compute_features_for_ticker(t) for t in universe_tickers]

    target = compute_features_for_ticker("AMD")
    peers = find_nearest_peers(target, universe_features, n=5)
    print("\nAMD nearest peers:")
    for ticker, dist in peers:
        print(f"  {ticker}: {dist:.3f}")
