"""
Diagnostic: does a log-linear reformulation of the mature_tech EV/EBITDA
relative-multiple model preserve predictive power while guaranteeing
positive predictions?

The current linear model (mature_relative.py, validated hold-out R²=0.437
in the relative_regression.py diagnostic session) regresses
relative_multiple = (multiple - bucket_median) / bucket_median directly.
Because that's a plain linear fit with no floor, it can predict a
relative_multiple below -1, i.e. a negative absolute multiple — which its
defensive guard correctly refuses to ship rather than return nonsense.
Confirmed in the prior session's Phase 4 end-to-end validation: GOOGL and
AAPL get NO EV/EBITDA prediction at all under the live engine because of
exactly this.

Hypothesis: fitting log(multiple / bucket_median) instead removes the
possibility of a negative predicted multiple by construction
(predicted_multiple = bucket_median * exp(...) > 0 always), using the same
universe, features, and 80/20 seed=42 split as the original model.

Math note worth keeping in mind when reading the comparison table: Pearson
correlation is invariant to positive affine transforms. The ORIGINAL
linear model's predicted_multiple = bucket_median * (1 + predicted_relative)
is a strictly increasing AFFINE function of predicted_relative, so
corr(predicted_relative, actual_relative)**2 and
corr(predicted_multiple, actual_multiple)**2 are mathematically IDENTICAL
for that model — its 0.437 is valid as both "relative-scale" and
"multiple-scale" R². That equivalence does NOT hold here: exp() is
non-linear, so this log-linear model's log-scale R² and multiple-scale R²
are genuinely different metrics. Both are computed and reported below.

Isolated diagnostic — does not modify semi_forest.py, peer_display.py,
final_engine.py, orchestrator.py, or composite.py. Only
valuation/comps/models/mature_relative.py is modified, and only if
run_decision() at the bottom finds this formulation should replace it
(same public names — fit_mature_relative, predict_mature_relative,
MATURE_RELATIVE_FEATURES, VALIDATED_HOLDOUT_R2 — so final_engine.py's
imports keep working unchanged either way).

Run directly: python3 -m valuation.comps.models.mature_relative_loglinear
"""

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from valuation.comps.regression import MIN_OBSERVATIONS

MATURE_RELATIVE_FEATURES = ["fcf_conversion", "gross_margin"]


@dataclass
class MatureRelativeLogLinearModel:
    model: object = None
    scaler: object = None
    bucket_median: float | None = None
    feature_names: list = field(default_factory=lambda: list(MATURE_RELATIVE_FEATURES))
    r_squared: float | None = None      # in-sample, LOG scale
    n_observations: int = 0
    data_flags: list = field(default_factory=list)


def _value_or_none(features_obj, name):
    v = getattr(features_obj, name, None)
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def fit_mature_relative_loglinear(observations, multiple_type="ev_ebitda", feature_names=None):
    """
    Fit log(multiple / bucket_median) ~ standardized features. Only
    positive-EBITDA-multiple tickers are usable (log requires it — same
    filter compute_multiples_for_ticker already applies upstream by
    returning None for non-positive EBITDA, so this is naturally enforced
    by the multiple_value > 0 check below).

    Signature matches the fit_fn shape diagnostics.loo_cross_validate()
    expects: (observations, multiple_type, feature_names=...) -> a result
    with .r_squared, so LOO CV can reuse that existing generalized loop.
    """
    feature_names = list(feature_names) if feature_names else list(MATURE_RELATIVE_FEATURES)
    data_flags = []

    usable_raw_multiples = [
        obs.multiple_value for obs in observations
        if obs.multiple_type == multiple_type and obs.multiple_value is not None and obs.multiple_value > 0
    ]
    if len(usable_raw_multiples) < 2:
        data_flags.append("Fewer than 2 valid ev_ebitda observations — cannot compute bucket median")
        return MatureRelativeLogLinearModel(feature_names=feature_names, data_flags=data_flags)

    bucket_median = float(np.median(usable_raw_multiples))
    if bucket_median <= 0:
        data_flags.append(f"Bucket median ev_ebitda non-positive ({bucket_median}) — log-relative undefined")
        return MatureRelativeLogLinearModel(feature_names=feature_names, data_flags=data_flags)

    rows, targets = [], []
    for obs in observations:
        if obs.multiple_type != multiple_type or obs.multiple_value is None or obs.multiple_value <= 0:
            continue
        values = [_value_or_none(obs.features, f) for f in feature_names]
        if any(v is None for v in values):
            continue
        rows.append(values)
        targets.append(math.log(obs.multiple_value / bucket_median))

    if len(rows) < MIN_OBSERVATIONS:
        data_flags.append(f"Only {len(rows)} usable observations (< {MIN_OBSERVATIONS} minimum) — log-linear not fit")
        return MatureRelativeLogLinearModel(bucket_median=bucket_median, feature_names=feature_names, data_flags=data_flags)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.array(rows, dtype=float))
    y = np.array(targets, dtype=float)

    model = LinearRegression()
    model.fit(X_scaled, y)
    r_squared = float(model.score(X_scaled, y))

    data_flags.append(f"Fit on {len(rows)} observations, bucket_median={bucket_median:.3f}, in-sample R² (log scale)={r_squared:.3f}")

    return MatureRelativeLogLinearModel(
        model=model, scaler=scaler, bucket_median=bucket_median, feature_names=feature_names,
        r_squared=r_squared, n_observations=len(rows), data_flags=data_flags,
    )


def predict_log_relative(regression, features):
    """Predict log(multiple / bucket_median) directly — the model's native output scale."""
    if regression is None or regression.model is None:
        return None
    row = [_value_or_none(features, f) for f in regression.feature_names]
    if any(v is None for v in row):
        return None
    x_scaled = regression.scaler.transform(np.array([row], dtype=float))
    return float(regression.model.predict(x_scaled)[0])


def predict_mature_relative_loglinear(regression, features):
    """
    Predict the absolute multiple (reversing log + bucket_median).
    Guaranteed positive by construction (exp() > 0 always) — unlike the
    linear model, no defensive positivity guard is needed here.

    Returns (predicted_multiple, None) — matches the fit_fn/predict_fn
    (value, se) calling convention used by loo_cross_validate() /
    forward_selection(); this model has no analytic per-prediction SE.
    """
    predicted_log_relative = predict_log_relative(regression, features)
    if predicted_log_relative is None or regression.bucket_median is None:
        return None, None
    return regression.bucket_median * math.exp(predicted_log_relative), None


def _corr_r2(predicted, actual):
    if len(predicted) < 2:
        return None
    p, a = np.array(predicted, dtype=float), np.array(actual, dtype=float)
    if np.std(p) == 0 or np.std(a) == 0:
        return None
    return float(np.corrcoef(p, a)[0, 1] ** 2)


def run_mature_ebitda_loglinear_diagnostic():
    from valuation.comps.universe import COMPS_UNIVERSE, get_universe
    from valuation.comps.feature_selection import compute_extended_features_for_ticker
    from valuation.comps.multiples import compute_multiples_for_ticker
    from valuation.comps.holdout_validation import split_universe, interpret, SPLIT_SEED
    from valuation.comps.regression import MultipleObservation
    from valuation.comps.diagnostics import loo_cross_validate
    from valuation.comps.models.mature_relative import (
        fit_mature_relative, predict_mature_relative, VALIDATED_HOLDOUT_R2 as LINEAR_HOLDOUT_R2,
    )

    bucket = "mature_tech"
    tickers = COMPS_UNIVERSE[bucket]
    train_tickers, holdout_tickers = split_universe(tickers)  # seed=42 — SAME split validated the linear model

    print(f"\n{'=' * 90}\nLOG-LINEAR REFORMULATION: mature_tech EV/EBITDA\n{'=' * 90}")
    print(f"Universe: {len(tickers)} -> train={len(train_tickers)}, holdout={len(holdout_tickers)} (seed={SPLIT_SEED})")

    print("Computing extended features + multiples for full universe...")
    features_by_ticker = {t: compute_extended_features_for_ticker(t) for t in tickers}
    multiples_by_ticker = {t: compute_multiples_for_ticker(t) for t in tickers}

    def build_obs(subset):
        obs = []
        for t in subset:
            f = features_by_ticker.get(t)
            m = multiples_by_ticker.get(t)
            if f is None or m is None or m.get("ev_ebitda") is None:
                continue
            obs.append(MultipleObservation(t, "ev_ebitda", m["ev_ebitda"], f))
        return obs

    train_obs = build_obs(train_tickers)
    holdout_obs = build_obs(holdout_tickers)

    # ---- Step 4: fit on training set, report coefficients + in-sample R² + LOO CV R² ----
    train_model = fit_mature_relative_loglinear(train_obs)
    print(f"\nFit: n={train_model.n_observations}, bucket_median={train_model.bucket_median}")
    if train_model.model is not None:
        coefs = dict(zip(train_model.feature_names, train_model.model.coef_))
        print(f"Coefficients (standardized): {coefs}")
        print(f"Intercept: {train_model.model.intercept_}")
    print(f"In-sample R² (log scale): {train_model.r_squared}")

    _loo_results, loo_cv_r2_multiple_scale = loo_cross_validate(
        train_obs, "ev_ebitda", feature_names=MATURE_RELATIVE_FEATURES,
        fit_fn=fit_mature_relative_loglinear, predict_fn=predict_mature_relative_loglinear,
    )
    print(f"LOO CV R² (multiple scale, via loo_cross_validate): {loo_cv_r2_multiple_scale}")

    # LOO CV R² on the model's native log scale too, for direct comparison
    # against in-sample R² (both log scale) and for step 6's log-scale ask.
    loo_log_predicted, loo_log_actual = [], []
    for obs in train_obs:
        held_out_ticker = obs.ticker
        fold_train = [o for o in train_obs if o.ticker != held_out_ticker]
        fold_model = fit_mature_relative_loglinear(fold_train)
        pred_log = predict_log_relative(fold_model, obs.features)
        if pred_log is None or fold_model.bucket_median is None:
            continue
        loo_log_predicted.append(pred_log)
        loo_log_actual.append(math.log(obs.multiple_value / fold_model.bucket_median))
    loo_cv_r2_log_scale = _corr_r2(loo_log_predicted, loo_log_actual)
    print(f"LOO CV R² (log scale, native to this model): {loo_cv_r2_log_scale}")

    # ---- Step 6: 80/20 hold-out validation, fit ONCE on training, predict holdout ----
    holdout_model = fit_mature_relative_loglinear(train_obs)
    predicted_multiples, actual_multiples = [], []
    predicted_log_rel, actual_log_rel = [], []
    for obs in holdout_obs:
        pred_mult, _ = predict_mature_relative_loglinear(holdout_model, obs.features)
        if pred_mult is None:
            continue
        pred_log = predict_log_relative(holdout_model, obs.features)
        predicted_multiples.append(pred_mult)
        actual_multiples.append(obs.multiple_value)
        predicted_log_rel.append(pred_log)
        actual_log_rel.append(math.log(obs.multiple_value / holdout_model.bucket_median))

    holdout_r2_multiple = _corr_r2(predicted_multiples, actual_multiples)
    holdout_r2_log = _corr_r2(predicted_log_rel, actual_log_rel)
    print(f"\nHold-out: {len(predicted_multiples)} of {len(holdout_obs)} evaluated")
    print(f"  Direct comparison  — Hold-out R² (multiple scale): {holdout_r2_multiple}")
    print(f"  Log-scale comparison — Hold-out R² (log scale):    {holdout_r2_log}")
    for v in predicted_multiples:
        if v <= 0:
            print(f"  [ANOMALY] non-positive predicted multiple {v} — should be impossible for log-linear, investigate")

    # Verdict: compare LOO CV R² and Hold-out R² on the SAME scale
    # (multiple scale) for a genuinely apples-to-apples gap, per the
    # task's own note that multiple-scale is "the critical column ...
    # which is comparable."
    verdict = interpret(loo_cv_r2_multiple_scale, holdout_r2_multiple)
    print(f"\nVerdict (LOO CV R² vs hold-out R², both multiple-scale): {verdict}")

    # ---- Step 7: comparison table ----
    print(f"\n{'-' * 100}")
    print("COMPARISON TABLE")
    print(f"{'-' * 100}")
    print(f"{'Model':<28}{'Train CV R²':>20}{'Hold-out R² (mult.)':>22}{'Verdict':>12}")
    print(f"{'Linear-on-relative (current)':<28}{'0.478 (linear-scale)':>20}{'0.437 (multiple-scale)':>22}{'Confirmed':>12}")
    tcv_str = f"{loo_cv_r2_multiple_scale:.3f} (mult.)" if loo_cv_r2_multiple_scale is not None else "N/A"
    hr2_str = f"{holdout_r2_multiple:.3f}" if holdout_r2_multiple is not None else "N/A"
    print(f"{'Log-linear-on-relative (new)':<28}{tcv_str:>20}{hr2_str:>22}{verdict:>12}")
    print(f"  (log-scale for reference: LOO CV R²={loo_cv_r2_log_scale}, hold-out R²={holdout_r2_log})")

    # ---- Step 8: GOOGL, AAPL, MSFT specific checks under BOTH formulations ----
    print(f"\n{'-' * 100}")
    print("SPECIFIC TICKER CHECK — does log-linear fix the known GOOGL/AAPL failures?")
    print(f"{'-' * 100}")
    ticker_results = {}
    for ticker in ("GOOGL", "AAPL", "MSFT"):
        universe_tickers = get_universe(bucket, exclude_ticker=ticker)
        target_features = features_by_ticker.get(ticker) or compute_extended_features_for_ticker(ticker)
        target_multiples = multiples_by_ticker.get(ticker) or compute_multiples_for_ticker(ticker)
        actual_multiple = target_multiples.get("ev_ebitda")

        loo_obs = []
        for t in universe_tickers:
            f = features_by_ticker.get(t)
            m = multiples_by_ticker.get(t)
            if f is None or m is None or m.get("ev_ebitda") is None:
                continue
            loo_obs.append(MultipleObservation(t, "ev_ebitda", m["ev_ebitda"], f))

        linear_model, _lf = fit_mature_relative(loo_obs)
        linear_pred, _lm, _lflags = predict_mature_relative(linear_model, target_features)

        loglinear_model = fit_mature_relative_loglinear(loo_obs)
        loglinear_pred, _ = predict_mature_relative_loglinear(loglinear_model, target_features)

        print(f"\n  {ticker}: actual_ev_ebitda={actual_multiple}")
        print(f"    Linear:     predicted={linear_pred}  {'REFUSED (non-positive)' if linear_pred is None else 'OK'}")
        print(f"    Log-linear: predicted={loglinear_pred}  positive={'YES' if (loglinear_pred is not None and loglinear_pred > 0) else 'NO/MISSING'}")

        ticker_results[ticker] = {"actual": actual_multiple, "linear_pred": linear_pred, "loglinear_pred": loglinear_pred}

    googl_aapl_fixed = all(
        ticker_results[t]["loglinear_pred"] is not None and ticker_results[t]["loglinear_pred"] > 0
        for t in ("GOOGL", "AAPL")
    )
    print(f"\nGOOGL/AAPL fixed under log-linear: {googl_aapl_fixed}")

    return {
        "loo_cv_r2_multiple_scale": loo_cv_r2_multiple_scale,
        "loo_cv_r2_log_scale": loo_cv_r2_log_scale,
        "holdout_r2_multiple": holdout_r2_multiple,
        "holdout_r2_log": holdout_r2_log,
        "verdict": verdict,
        "ticker_results": ticker_results,
        "googl_aapl_fixed": googl_aapl_fixed,
        "in_sample_r2_log": train_model.r_squared,
        "coefficients": dict(zip(train_model.feature_names, train_model.model.coef_)) if train_model.model else {},
    }


def run_decision(diagnostic_results):
    """
    Step 9: decide whether to switch mature_relative.py to this
    formulation. Criteria (task + constraints):
      - verdict in {Confirmed, Suspicious} (not Overfit)
      - hold-out R² (multiple scale) >= 0.35
      - valid (present, positive) predictions for GOOGL and AAPL
    """
    print(f"\n{'=' * 90}\nDECISION\n{'=' * 90}")
    verdict = diagnostic_results["verdict"]
    holdout_r2 = diagnostic_results["holdout_r2_multiple"]
    googl_aapl_fixed = diagnostic_results["googl_aapl_fixed"]

    not_overfit = verdict in ("Confirmed", "Suspicious")
    meets_bar = holdout_r2 is not None and holdout_r2 >= 0.35

    print(f"  Not overfit (Confirmed/Suspicious): {not_overfit}  (verdict={verdict})")
    print(f"  Hold-out R² >= 0.35: {meets_bar}  (holdout_r2={holdout_r2})")
    print(f"  GOOGL/AAPL fixed: {googl_aapl_fixed}")

    switch = not_overfit and meets_bar and googl_aapl_fixed
    print(f"\n  DECISION: {'SWITCH to log-linear' if switch else 'KEEP linear implementation'}")
    return switch


PREVIOUS_HOLDOUT_R2_MULTIPLE_SCALE = 0.055  # Session 10's result, for the comparison below


def rerun_verification():
    """
    Session 11: re-run this diagnostic (methodology unchanged from
    run_mature_ebitda_loglinear_diagnostic() above) to verify Session 10's
    finding — hold-out R²=0.055, a collapse from the linear model's 0.437
    — is robust to the transient network timeouts Session 10 encountered,
    not an artifact of a partial/corrupted fetch.

    Adds beyond the original diagnostic: explicit fetch-failure and
    data-quality-issue reporting, a wider set of special-case ticker
    predictions (mega-cap outliers, large mature, SaaS-adjacent), and a
    fresh fit of the LINEAR (currently shipped) model on the same
    universe/split as a sanity check that it still reproduces ~0.437.
    """
    from valuation.comps.universe import COMPS_UNIVERSE, get_universe
    from valuation.comps.feature_selection import compute_extended_features_for_ticker
    from valuation.comps.multiples import compute_multiples_for_ticker
    from valuation.comps.holdout_validation import split_universe, SPLIT_SEED
    from valuation.comps.regression import MultipleObservation
    from valuation.comps.diagnostics import loo_cross_validate
    from valuation.comps.models.mature_relative import fit_mature_relative, predict_mature_relative

    bucket = "mature_tech"
    tickers = COMPS_UNIVERSE[bucket]
    print(f"\n{'=' * 90}\nSESSION 11: RE-RUN VERIFICATION — mature_tech EV/EBITDA log-linear\n{'=' * 90}")
    print(f"Universe size: {len(tickers)} tickers")

    # ---- Step 1: fetch, reporting failures explicitly ----
    features_by_ticker, multiples_by_ticker, fetch_failures = {}, {}, []
    for t in tickers:
        try:
            features_by_ticker[t] = compute_extended_features_for_ticker(t)
        except Exception as e:
            fetch_failures.append((t, "features", f"{type(e).__name__}: {e}"))
            continue
        try:
            multiples_by_ticker[t] = compute_multiples_for_ticker(t)
        except Exception as e:
            fetch_failures.append((t, "multiples", f"{type(e).__name__}: {e}"))

    successfully_fetched = [t for t in tickers if t in features_by_ticker and t in multiples_by_ticker]
    print(f"Successfully fetched: {len(successfully_fetched)} of {len(tickers)}")
    if fetch_failures:
        print(f"Fetch failures ({len(fetch_failures)}):")
        for t, stage, err in fetch_failures:
            print(f"  {t} ({stage}): {err}")
    else:
        print("No fetch failures.")

    usable_tickers, quality_issues = [], []
    for t in successfully_fetched:
        m, f = multiples_by_ticker[t], features_by_ticker[t]
        if m.get("ev_ebitda") is None or m["ev_ebitda"] <= 0:
            quality_issues.append(f"{t}: no positive ev_ebitda")
            continue
        if _value_or_none(f, "fcf_conversion") is None or _value_or_none(f, "gross_margin") is None:
            quality_issues.append(f"{t}: missing fcf_conversion or gross_margin")
            continue
        usable_tickers.append(t)
    print(f"\nUsable for this model (positive ev_ebitda + both features): {len(usable_tickers)} of {len(successfully_fetched)}")
    print(f"Data quality issues ({len(quality_issues)}):")
    for q in quality_issues:
        print(f"  {q}")

    # ---- Step 2: same seed=42 split, refit ----
    train_tickers, holdout_tickers = split_universe(tickers)

    def build_obs(subset):
        obs = []
        for t in subset:
            f, m = features_by_ticker.get(t), multiples_by_ticker.get(t)
            if f is None or m is None or m.get("ev_ebitda") is None:
                continue
            obs.append(MultipleObservation(t, "ev_ebitda", m["ev_ebitda"], f))
        return obs

    train_obs = build_obs(train_tickers)
    holdout_obs = build_obs(holdout_tickers)
    print(f"\nTrain/holdout split (seed={SPLIT_SEED}): train={len(train_tickers)} tickers ({len(train_obs)} usable obs), "
          f"holdout={len(holdout_tickers)} tickers ({len(holdout_obs)} usable obs)")

    # ---- Step 3: refit log-linear, in-sample + LOO CV (both scales) ----
    train_model = fit_mature_relative_loglinear(train_obs)
    print(f"\nLog-linear fit: n={train_model.n_observations}, bucket_median={train_model.bucket_median}")
    if train_model.model is not None:
        print(f"Coefficients: {dict(zip(train_model.feature_names, train_model.model.coef_))}")
        print(f"Intercept: {train_model.model.intercept_}")
    print(f"In-sample R² (log scale): {train_model.r_squared}")

    _loo, loo_r2_mult = loo_cross_validate(
        train_obs, "ev_ebitda", feature_names=MATURE_RELATIVE_FEATURES,
        fit_fn=fit_mature_relative_loglinear, predict_fn=predict_mature_relative_loglinear,
    )
    print(f"LOO CV R² (multiple scale): {loo_r2_mult}")

    loo_log_pred, loo_log_act = [], []
    for obs in train_obs:
        fold_train = [o for o in train_obs if o.ticker != obs.ticker]
        fold_model = fit_mature_relative_loglinear(fold_train)
        pl = predict_log_relative(fold_model, obs.features)
        if pl is None or fold_model.bucket_median is None:
            continue
        loo_log_pred.append(pl)
        loo_log_act.append(math.log(obs.multiple_value / fold_model.bucket_median))
    loo_r2_log = _corr_r2(loo_log_pred, loo_log_act)
    print(f"LOO CV R² (log scale): {loo_r2_log}")

    # ---- Step 4: hold-out (multiple scale) ----
    predicted_m, actual_m = [], []
    for obs in holdout_obs:
        pm, _ = predict_mature_relative_loglinear(train_model, obs.features)
        if pm is not None:
            predicted_m.append(pm)
            actual_m.append(obs.multiple_value)
    holdout_r2_mult = _corr_r2(predicted_m, actual_m)
    print(f"\nHold-out R² (multiple scale): {holdout_r2_mult}  (n_evaluated={len(predicted_m)} of {len(holdout_obs)})")

    if holdout_r2_mult is not None:
        delta = abs(holdout_r2_mult - PREVIOUS_HOLDOUT_R2_MULTIPLE_SCALE)
        print(f"\nComparison to Session 10: previous={PREVIOUS_HOLDOUT_R2_MULTIPLE_SCALE}, this run={holdout_r2_mult:.4f}, |delta|={delta:.4f}")
        if delta > 0.10:
            print("MATERIAL DIFFERENCE (>0.10) — investigate cause.")
        else:
            print("Consistent with Session 10 (<=0.10 difference) — CONFIRMS the prior finding.")

    # ---- Step 5: sanity-check the LINEAR (shipped) model on the same universe/split ----
    linear_model, linear_flags = fit_mature_relative(train_obs)
    print(f"\nLinear (shipped) model sanity check: {linear_flags}")
    lin_pred_m, lin_act_m = [], []
    for obs in holdout_obs:
        pm, _missing, _flags = predict_mature_relative(linear_model, obs.features)
        if pm is not None:
            lin_pred_m.append(pm)
            lin_act_m.append(obs.multiple_value)
    linear_holdout_r2 = _corr_r2(lin_pred_m, lin_act_m)
    print(f"Linear hold-out R² (multiple scale, this run): {linear_holdout_r2}  (n_evaluated={len(lin_pred_m)} of {len(holdout_obs)})")
    if linear_holdout_r2 is not None:
        print(f"Expected ~0.437 from Session 7/10 — {'MATCHES' if abs(linear_holdout_r2 - 0.437) < 0.10 else 'CHECK: differs materially'}")

    # ---- Step 6: special-case ticker predictions ----
    special_tickers = ["GOOGL", "AAPL", "MSFT", "META", "CSCO", "ORCL", "CRM", "ADBE"]
    print(f"\n{'-' * 100}\nSPECIAL-CASE TICKER PREDICTIONS\n{'-' * 100}")
    print(f"{'Ticker':<8}{'Actual':>10}{'Linear pred':>14}{'LogLin pred':>14}")
    special_results = {}
    for ticker in special_tickers:
        target_features = features_by_ticker.get(ticker)
        target_multiples = multiples_by_ticker.get(ticker)
        if target_features is None or target_multiples is None:
            print(f"{ticker:<8}  FAILED TO FETCH")
            special_results[ticker] = None
            continue
        actual = target_multiples.get("ev_ebitda")

        universe_tickers = get_universe(bucket, exclude_ticker=ticker)
        loo_obs = []
        for t in universe_tickers:
            f, m = features_by_ticker.get(t), multiples_by_ticker.get(t)
            if f is None or m is None or m.get("ev_ebitda") is None:
                continue
            loo_obs.append(MultipleObservation(t, "ev_ebitda", m["ev_ebitda"], f))

        lm, _ = fit_mature_relative(loo_obs)
        lpred, _, _ = predict_mature_relative(lm, target_features)

        llm = fit_mature_relative_loglinear(loo_obs)
        llpred, _ = predict_mature_relative_loglinear(llm, target_features)

        actual_str = f"{actual:.2f}" if actual is not None else "N/A"
        lpred_str = f"{lpred:.2f}" if lpred is not None else "REFUSED"
        llpred_str = f"{llpred:.2f}" if llpred is not None else "N/A"
        print(f"{ticker:<8}{actual_str:>10}{lpred_str:>14}{llpred_str:>14}")
        special_results[ticker] = {"actual": actual, "linear_pred": lpred, "loglinear_pred": llpred}

    return {
        "successfully_fetched": len(successfully_fetched), "fetch_failures": fetch_failures,
        "usable_tickers": len(usable_tickers), "quality_issues": quality_issues,
        "in_sample_r2_log": train_model.r_squared, "loo_r2_multiple": loo_r2_mult, "loo_r2_log": loo_r2_log,
        "holdout_r2_multiple": holdout_r2_mult, "linear_holdout_r2_sanity": linear_holdout_r2,
        "coefficients": dict(zip(train_model.feature_names, train_model.model.coef_)) if train_model.model else {},
        "special_results": special_results,
    }


if __name__ == "__main__":
    results = run_mature_ebitda_loglinear_diagnostic()
    run_decision(results)
