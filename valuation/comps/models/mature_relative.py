"""
OLS-on-relative-multiple comps model for mature_tech EV/EBITDA — the only
validated Layer 2 result for this bucket (relative_regression.py
diagnostic session): hold-out R² = 0.437, training CV R² = 0.478
(gap = 0.041), features [fcf_conversion, gross_margin], linear (not
log-linear) regression on relative_multiple = (multiple - bucket_median) /
bucket_median.

mature_tech EV/Sales is intentionally NOT shipped — no model form (OLS,
Ridge, RandomForest, GradientBoosting) or feature set (4-feature,
10-feature) or formulation (absolute, relative) produced a validated
result for it across 6 diagnostic sessions. There is no fit_/predict_
function here for it; final_engine.py routes it to peer-display-only.
"""

import math

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from valuation.comps.regression import MIN_OBSERVATIONS

MATURE_RELATIVE_FEATURES = ["fcf_conversion", "gross_margin"]
VALIDATED_HOLDOUT_R2 = 0.437


class MatureRelativeModel:
    def __init__(self, model, scaler, bucket_median, feature_names):
        self.model = model
        self.scaler = scaler
        self.bucket_median = bucket_median
        self.feature_names = feature_names


def _value_or_none(features_obj, name):
    v = getattr(features_obj, name, None)
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def fit_mature_relative(observations):
    """
    Fit on the full mature_tech universe's ev_ebitda observations (live
    model — no train/holdout split; that was only for validation).
    Requires BOTH features present per training row — no partial
    tolerance (2-feature model, unlike semi_forest's 8).
    Returns (MatureRelativeModel | None, data_flags).
    """
    data_flags = []
    usable_raw_multiples = [
        obs.multiple_value for obs in observations
        if obs.multiple_type == "ev_ebitda" and obs.multiple_value is not None and obs.multiple_value > 0
    ]
    if len(usable_raw_multiples) < 2:
        data_flags.append("Fewer than 2 valid ev_ebitda observations — cannot compute bucket median")
        return None, data_flags

    bucket_median = float(np.median(usable_raw_multiples))
    if bucket_median == 0:
        data_flags.append("Bucket median ev_ebitda is exactly 0 — relative multiple undefined")
        return None, data_flags

    rows, targets = [], []
    for obs in observations:
        if obs.multiple_type != "ev_ebitda" or obs.multiple_value is None or obs.multiple_value <= 0:
            continue
        values = [_value_or_none(obs.features, f) for f in MATURE_RELATIVE_FEATURES]
        if any(v is None for v in values):
            continue
        rows.append(values)
        targets.append((obs.multiple_value - bucket_median) / bucket_median)

    if len(rows) < MIN_OBSERVATIONS:
        data_flags.append(f"Only {len(rows)} usable observations (< {MIN_OBSERVATIONS} minimum) — mature_relative not fit")
        return None, data_flags

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.array(rows, dtype=float))

    model = LinearRegression()
    model.fit(X_scaled, np.array(targets, dtype=float))

    data_flags.append(
        f"Fit on {len(rows)} observations, bucket_median={bucket_median:.3f} "
        f"(diagnostic-validated hold-out R²={VALIDATED_HOLDOUT_R2})"
    )

    return MatureRelativeModel(model, scaler, bucket_median, MATURE_RELATIVE_FEATURES), data_flags


def predict_mature_relative(model_bundle, target_features):
    """
    Returns (predicted_multiple, features_missing, data_flags). Requires
    both features present.
    """
    if model_bundle is None:
        return None, [], ["mature_relative model not available"]

    row = [_value_or_none(target_features, f) for f in model_bundle.feature_names]
    missing = [f for f, v in zip(model_bundle.feature_names, row) if v is None]
    if missing:
        return None, missing, [f"Target missing required feature(s) {missing} — insufficient data, no prediction"]

    x_scaled = model_bundle.scaler.transform(np.array([row], dtype=float))
    predicted_relative = float(model_bundle.model.predict(x_scaled)[0])
    predicted_multiple = model_bundle.bucket_median * (1 + predicted_relative)

    if predicted_multiple <= 0:
        return None, [], [f"Predicted multiple non-positive ({predicted_multiple:.2f}) — discarding implausible prediction"]

    return predicted_multiple, [], []
