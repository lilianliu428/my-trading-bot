"""
RandomForest comps model for the semiconductors bucket — both EV/Sales and
EV/EBITDA. Validated in the tree_regression.py diagnostic session:
hold-out R² = 0.330 (EV/Sales) and 0.598 (EV/EBITDA) against an 80/20
hold-out split (seed=42) of the 64-ticker semiconductors universe.

Two deliberate deviations from LAYER_2_FINAL_SPEC's literal pseudocode,
both necessary for this model to actually reproduce those hold-out
numbers rather than ship something never tested:

  - max_depth is validated PER MULTIPLE TYPE (7 for EV/Sales, 5 for
    EV/EBITDA) via GridSearchCV in the diagnostic, not a single fixed 5
    for both — the spec's fixed depth=5 only matches the EV/EBITDA result.
  - The feature set is the 8 candidates that survived the diagnostic's
    >20%-missing-data filter on the training set, not all 10 —
    trailing_growth_5y (63% missing) and fcf_conversion (27% missing) were
    excluded during validation, so training on all 10 here would fit a
    different, untested model.

Missing-feature handling (live prediction, not part of the diagnostic):
tolerates up to 2 of the 8 features missing via training-set-median
imputation (matches the task's ">=3 of 10 missing -> refuse" rule,
translated to this 8-feature set); refuses beyond that and reports
"insufficient data" instead of guessing.
"""

import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from valuation.comps.multiples import winsorize
from valuation.comps.regression import MIN_OBSERVATIONS

SEMI_FEATURES = [
    "trailing_growth", "growth_stability", "operating_margin", "gross_margin",
    "ebitda_margin", "roic", "r_and_d_intensity", "log_revenue",
]

MAX_DEPTH_BY_MULTIPLE = {"ev_sales": 7, "ev_ebitda": 5}
RF_FIXED_PARAMS = dict(n_estimators=100, min_samples_split=5, min_samples_leaf=3, random_state=42)
VALIDATED_HOLDOUT_R2 = {"ev_sales": 0.330, "ev_ebitda": 0.598}
MAX_MISSING_FEATURES = 2  # refuse at 3+ of 8 missing, per task's ">=3 of 10" rule


class SemiForestModel:
    def __init__(self, multiple_type, model, scaler, imputer, feature_names):
        self.multiple_type = multiple_type
        self.model = model
        self.scaler = scaler
        self.imputer = imputer
        self.feature_names = feature_names


def _value_or_nan(features_obj, name):
    v = getattr(features_obj, name, None)
    if v is None:
        return math.nan
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return math.nan
    return float(v)


def fit_semi_forest(observations, multiple_type):
    """
    Fit the live (full-universe) RandomForest for one multiple type — no
    train/holdout split here, that was only for validation. Training rows
    with more than MAX_MISSING_FEATURES missing are dropped; the rest are
    median-imputed. Returns (SemiForestModel | None, data_flags).
    """
    data_flags = []
    candidates = [
        obs for obs in observations
        if obs.multiple_type == multiple_type and obs.multiple_value is not None and obs.multiple_value > 0
    ]

    rows = []
    for obs in candidates:
        row = [_value_or_nan(obs.features, f) for f in SEMI_FEATURES]
        if sum(1 for v in row if math.isnan(v)) <= MAX_MISSING_FEATURES:
            rows.append((obs, row))

    dropped = len(candidates) - len(rows)
    if dropped:
        data_flags.append(f"Dropped {dropped} of {len(candidates)} training observations (>{MAX_MISSING_FEATURES} of {len(SEMI_FEATURES)} features missing)")

    if len(rows) < MIN_OBSERVATIONS:
        data_flags.append(f"Only {len(rows)} usable observations (< {MIN_OBSERVATIONS} minimum) — semi_forest not fit")
        return None, data_flags

    X_raw = np.array([r for _, r in rows], dtype=float)
    y_raw = np.array([obs.multiple_value for obs, _ in rows], dtype=float)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    y_wins = winsorize(y_raw)
    X_wins = np.column_stack([winsorize(X_imputed[:, i]) for i in range(X_imputed.shape[1])])
    y_log = np.log(y_wins)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_wins)

    max_depth = MAX_DEPTH_BY_MULTIPLE[multiple_type]
    model = RandomForestRegressor(max_depth=max_depth, **RF_FIXED_PARAMS)
    model.fit(X_scaled, y_log)

    data_flags.append(
        f"Fit on {len(rows)} observations, max_depth={max_depth} "
        f"(diagnostic-validated hold-out R²={VALIDATED_HOLDOUT_R2[multiple_type]})"
    )

    return SemiForestModel(multiple_type, model, scaler, imputer, SEMI_FEATURES), data_flags


def predict_semi_forest(model_bundle, target_features):
    """
    Returns (predicted_multiple, features_missing, data_flags).
    predicted_multiple is None if the model wasn't fit or the target is
    missing more than MAX_MISSING_FEATURES of the 8 features.
    """
    if model_bundle is None:
        return None, [], ["semi_forest model not available"]

    row = [_value_or_nan(target_features, f) for f in model_bundle.feature_names]
    missing = [f for f, v in zip(model_bundle.feature_names, row) if math.isnan(v)]

    if len(missing) > MAX_MISSING_FEATURES:
        return None, missing, [
            f"Target missing {len(missing)} of {len(model_bundle.feature_names)} features "
            f"(> {MAX_MISSING_FEATURES} tolerance) — insufficient data, no prediction"
        ]

    x_imputed = model_bundle.imputer.transform(np.array([row], dtype=float))
    x_scaled = model_bundle.scaler.transform(x_imputed)
    predicted_log = float(model_bundle.model.predict(x_scaled)[0])
    predicted_multiple = float(np.exp(predicted_log))

    flags = [f"Prediction used median-imputed values for: {missing}"] if missing else []
    return predicted_multiple, missing, flags
