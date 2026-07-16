"""
Comps regression — fits log(multiple) on the 4-feature vector and predicts
a target company's multiple with a standard error.

Model form (log-linear on the multiple, not on the features — the multiple
has a natural lower bound of zero and is skewed; features vary in both
directions):

    log(multiple) = intercept + b1*trailing_growth + b2*operating_margin
                     + b3*roic + b4*log_revenue + eps

Fit separately for ev_sales and ev_ebitda via statsmodels.OLS.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import statsmodels.api as sm

from valuation.comps.features import CompanyFeatures
from valuation.comps.multiples import winsorize

FEATURE_ORDER = ["trailing_growth", "operating_margin", "roic", "log_revenue"]

# Need enough degrees of freedom for 4 features + intercept to mean anything.
MIN_OBSERVATIONS = 8


def _feature_vector_complete(features_obj, feature_names):
    """
    Generic completeness/finiteness check for an arbitrary feature subset —
    unlike CompanyFeatures.is_complete(), which only checks the fixed
    original 4, this checks whichever feature_names a regression was
    actually asked to use (needed once fit_regression() supports feature
    subsets beyond FEATURE_ORDER, e.g. for feature-selection diagnostics).
    """
    for name in feature_names:
        value = getattr(features_obj, name, None)
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            return False
    return True


@dataclass
class MultipleObservation:
    ticker: str
    multiple_type: str              # "ev_sales" or "ev_ebitda"
    multiple_value: float           # observed multiple (e.g. 8.5 for 8.5x)
    features: CompanyFeatures       # for the regression right-hand side


@dataclass
class CompsRegressionResult:
    multiple_type: str
    n_observations: int             # how many companies in the regression
    r_squared: float | None
    coefficients: dict              # keyed by feature name
    intercept: float | None
    residual_std: float | None      # for standard errors on predictions
    data_flags: list = field(default_factory=list)
    # Internal: kept so predict_multiple() can compute the exact
    # residual_std * sqrt(1 + x'(X'X)^-1 x) standard error without refitting.
    # Not part of the spec's public field list — implementation detail.
    _model_result: object = None
    # Internal: (min, max) of each feature AFTER winsorization, used by
    # callers to flag when a target's feature vector is an extrapolation
    # (log-linear predictions can blow up multiplicatively outside the
    # fitted range). Not part of the spec's public field list.
    feature_ranges: dict = field(default_factory=dict)
    # Which features (and in what order) this regression was fit on.
    # Defaults to FEATURE_ORDER for backward compatibility with every
    # existing caller (orchestrator.py, backtest/framework.py, diagnostics)
    # that doesn't pass feature_names — predict_multiple() reads this back
    # so it builds the right-length vector regardless of what was fit.
    feature_names: list = field(default_factory=lambda: list(FEATURE_ORDER))


def fit_regression(observations, multiple_type, feature_names=None):
    """
    Fit log-linear regression on the universe for one multiple type.

    feature_names: which features to regress on, in order. Defaults to
    FEATURE_ORDER (the original 4) — every existing caller relies on this
    default. Pass an explicit subset/superset (e.g. from the feature
    candidate pool in feature_selection.py) to fit on something else; the
    fitted CompsRegressionResult remembers which features it used so
    predict_multiple() builds a matching vector later.

    Drops observations with an incomplete feature vector (for the
    requested feature_names) or a non-positive multiple value (log
    undefined). Winsorizes the multiple values and each feature column
    independently at the 1st/99th percentile before fitting.

    Returns a CompsRegressionResult. If fewer than MIN_OBSERVATIONS usable
    observations remain, returns a result with r_squared/coefficients/etc.
    all None and an explanatory data_flag.
    """
    feature_names = list(feature_names) if feature_names else list(FEATURE_ORDER)
    data_flags = []
    usable = [
        obs for obs in observations
        if obs.multiple_type == multiple_type
        and _feature_vector_complete(obs.features, feature_names)
        and obs.multiple_value is not None
        and obs.multiple_value > 0
    ]
    dropped = len(observations) - len(usable)
    if dropped:
        data_flags.append(f"Dropped {dropped} of {len(observations)} observations (missing features or non-positive multiple)")

    if len(usable) < MIN_OBSERVATIONS:
        data_flags.append(
            f"Only {len(usable)} usable observations (< {MIN_OBSERVATIONS} minimum) — regression not fit"
        )
        return CompsRegressionResult(
            multiple_type=multiple_type,
            n_observations=len(usable),
            r_squared=None,
            coefficients={},
            intercept=None,
            residual_std=None,
            data_flags=data_flags,
            feature_names=feature_names,
        )

    X_raw = np.array([[getattr(obs.features, f) for f in feature_names] for obs in usable], dtype=float)
    y_raw = np.array([obs.multiple_value for obs in usable], dtype=float)

    # Winsorize independently per feature and per multiple (Q4) — each has
    # different tail behavior, so cap them separately rather than jointly.
    y_wins = winsorize(y_raw)
    X_wins = np.column_stack([winsorize(X_raw[:, i]) for i in range(X_raw.shape[1])])
    data_flags.append("Winsorized multiple and each feature at [1%, 99%]")

    y_log = np.log(y_wins)
    # has_constant="add" forces the constant into column 0 even if a feature
    # happens to be (near-)constant across the universe — sm.add_constant's
    # default autodetection would otherwise silently skip adding it, which
    # breaks the fixed const-then-feature_names column assumption below.
    X_design = sm.add_constant(X_wins, has_constant="add")

    model_result = sm.OLS(y_log, X_design).fit()

    coefficients = {name: float(model_result.params[i + 1]) for i, name in enumerate(feature_names)}
    intercept = float(model_result.params[0])
    r_squared = float(model_result.rsquared)
    residual_std = float(np.sqrt(model_result.mse_resid))

    data_flags.append(f"Fit on {len(usable)} observations, R²={r_squared:.3f}")

    feature_ranges = {
        name: (float(np.min(X_wins[:, i])), float(np.max(X_wins[:, i])))
        for i, name in enumerate(feature_names)
    }

    return CompsRegressionResult(
        multiple_type=multiple_type,
        n_observations=len(usable),
        r_squared=r_squared,
        coefficients=coefficients,
        intercept=intercept,
        residual_std=residual_std,
        data_flags=data_flags,
        _model_result=model_result,
        feature_ranges=feature_ranges,
        feature_names=feature_names,
    )


def check_extrapolation(regression, features):
    """
    Flag any feature where the target falls outside the (winsorized)
    training range — log-linear predictions can extrapolate multiplicatively
    on outliers, so a wide-range prediction should be flagged as low-trust
    rather than presented at face value.

    Returns a list of human-readable warning strings (empty if in-range or
    regression/features unavailable).
    """
    if regression is None or not regression.feature_ranges or not _feature_vector_complete(features, regression.feature_names):
        return []
    warnings = []
    for name, (lo, hi) in regression.feature_ranges.items():
        value = getattr(features, name)
        if value < lo or value > hi:
            warnings.append(
                f"{name}={value:.3f} outside {regression.multiple_type} training range "
                f"[{lo:.3f}, {hi:.3f}] — prediction is an extrapolation, treat with caution"
            )
    return warnings


def predict_multiple(regression, features):
    """
    Predict a multiple for the target company.

    Returns (predicted_multiple, prediction_std_error). Prediction is
    exp(log_predicted) to reverse the log transform. prediction_std_error is
    on the LOG scale — it is residual_std * sqrt(1 + x'(X'X)^-1 x), i.e. the
    standard error of log(multiple), accounting for both residual variance
    and fitting uncertainty. Callers converting to dollar/fair-value terms
    should apply the delta-method approximation (se_level ~= level * se_log).

    Returns (None, None) if the regression couldn't be fit or the target's
    feature vector is incomplete.
    """
    if regression is None or regression._model_result is None:
        return None, None
    if not _feature_vector_complete(features, regression.feature_names):
        return None, None

    x = np.array([[1.0] + [getattr(features, f) for f in regression.feature_names]])
    pred = regression._model_result.get_prediction(x)
    predicted_log = float(pred.predicted_mean[0])
    se_log = float(pred.se_obs[0])

    predicted_multiple = float(np.exp(predicted_log))
    return predicted_multiple, se_log


if __name__ == "__main__":
    from valuation.comps.features import compute_features_for_ticker
    from valuation.comps.multiples import compute_multiples_for_ticker
    from valuation.comps.universe import get_universe

    bucket = "semiconductors"
    universe = get_universe(bucket)

    observations = []
    for t in universe:
        feats = compute_features_for_ticker(t)
        mults = compute_multiples_for_ticker(t)
        if mults["ev_sales"] is not None:
            observations.append(MultipleObservation(t, "ev_sales", mults["ev_sales"], feats))
        if mults["ev_ebitda"] is not None:
            observations.append(MultipleObservation(t, "ev_ebitda", mults["ev_ebitda"], feats))

    reg = fit_regression(observations, "ev_sales")
    print(f"\nev_sales regression: n={reg.n_observations} R²={reg.r_squared}")
    print(f"  coefficients={reg.coefficients}")
    print(f"  intercept={reg.intercept}  residual_std={reg.residual_std}")

    target_feats = compute_features_for_ticker("AMD")
    pred, se = predict_multiple(reg, target_feats)
    print(f"\nAMD predicted ev_sales={pred}  se_log={se}")
