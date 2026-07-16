"""
Output dataclasses for the finalized, per-bucket-routed Layer 2 engine
(see LAYER_2_FINAL_SPEC — 6 diagnostic sessions: segmentation, forward
selection, sector-relative multiples, Ridge, tree regression, and this
final routing). Distinct from valuation/comps/regression.py's
CompsRegressionResult/MultipleObservation, which remain in place unchanged
and still back the general-purpose compute_comps_valuation() used by the
diagnostic modules.
"""

from dataclasses import dataclass, field


@dataclass
class Layer2RegressionOutput:
    multiple_type: str            # "ev_sales" or "ev_ebitda"
    predicted_multiple: float
    fair_value_per_share: float
    confidence: str                # "HIGH" or "MEDIUM"
    model_type: str                # e.g. "RandomForest 8-feature" or "OLS relative 2-feature"
    holdout_r2: float              # from the validating diagnostic session, not recomputed live
    formulation: str               # "absolute" or "relative"
    features_used: list = field(default_factory=list)
    features_missing: list = field(default_factory=list)


@dataclass
class PeerDisplayOutput:
    nearest_peers: list = field(default_factory=list)          # up to 5 tickers
    peer_distances: list = field(default_factory=list)
    peer_ev_sales_median: float | None = None
    peer_ev_ebitda_median: float | None = None
    target_ev_sales: float | None = None
    target_ev_ebitda: float | None = None
    target_ev_sales_percentile: float | None = None            # 0-100
    target_ev_ebitda_percentile: float | None = None
    feature_deviations: dict = field(default_factory=dict)     # feature name -> deviation in std devs vs peer group


@dataclass
class Layer2CompsResult:
    ticker: str
    bucket: str
    market_price: float | None

    ev_sales_regression: object = None     # Layer2RegressionOutput | None
    ev_ebitda_regression: object = None    # Layer2RegressionOutput | None

    peer_display: object = None            # PeerDisplayOutput | None

    data_flags: list = field(default_factory=list)
