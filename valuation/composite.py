"""
Top-level composite orchestrator — composes Layer 1 (DCF) + Layer 2 (Comps)
into a single side-by-side valuation for triangulation.
"""

import statistics

from valuation.tech import TECH_BUCKETS
from valuation.comps.orchestrator import compute_layer_2_comps, resolve_bucket


def _run_layer_1_dcf(ticker, bucket):
    """
    Layer 1 DCF, called in pure-baseline mode (layer_1_mode=True) — see
    valuation/inputs/growth.py docstrings on what that mode strips out
    (consensus growth signal, heavy-acquirer historical-revenue override).

    Tech buckets route through the tech DCF (R&D capitalization, ex-goodwill
    ROIC, SBC dilution). Other buckets fall back to the generic DCF, which
    doesn't support layer_1_mode — Layer 2's universe only covers tech
    buckets today, so this path is a documented best-effort fallback.
    """
    if bucket in TECH_BUCKETS:
        from valuation.tech.mature.mature_dcf import compute_tech_intrinsic_value
        return compute_tech_intrinsic_value(ticker, bucket=bucket, layer_1_mode=True)

    from valuation.dcf import compute_intrinsic_value_for_ticker
    result = compute_intrinsic_value_for_ticker(ticker, bucket=bucket)
    result.setdefault("data_flags", []).append(
        "Generic DCF does not support layer_1_mode — used as-is (consensus growth and "
        "heavy-acquirer overrides are not stripped for non-tech buckets)."
    )
    return result


def _build_signal_summary(dcf_value, comps_result):
    """
    Compare DCF fair value against the range of comps fair values and
    produce a short interpretation string. comps_result is a
    Layer2CompsResult (final_engine.py's per-bucket-routed engine) — its
    ev_sales_regression/ev_ebitda_regression are each a
    Layer2RegressionOutput | None, not shipped for every bucket (see
    LAYER_2_FINAL_SPEC's routing table).
    """
    comps_values = [
        v for v in (
            comps_result.ev_sales_regression.fair_value_per_share if comps_result and comps_result.ev_sales_regression else None,
            comps_result.ev_ebitda_regression.fair_value_per_share if comps_result and comps_result.ev_ebitda_regression else None,
        )
        if v is not None
    ]

    comps_range = (min(comps_values), max(comps_values)) if comps_values else None
    comps_median = statistics.median(comps_values) if comps_values else None

    divergence = None
    if comps_median is not None and dcf_value is not None:
        divergence = comps_median - dcf_value

    if dcf_value is None and comps_median is None:
        interpretation = "Neither DCF nor comps produced a usable fair value estimate."
    elif dcf_value is None:
        interpretation = "DCF unavailable; comps-only signal."
    elif comps_median is None:
        interpretation = "Comps unavailable; DCF-only signal."
    else:
        rel_divergence = divergence / dcf_value if dcf_value else 0
        if abs(rel_divergence) < 0.10:
            interpretation = (
                f"DCF (${dcf_value:.2f}) and comps (median ${comps_median:.2f}) converge within 10% — "
                "high-conviction signal."
            )
        elif divergence > 0:
            interpretation = (
                f"Comps (median ${comps_median:.2f}) see more upside than DCF (${dcf_value:.2f}) — "
                "the market is pricing peers richer than intrinsic cash flows justify, or the DCF's "
                "growth assumptions are conservative relative to the peer set."
            )
        else:
            interpretation = (
                f"DCF (${dcf_value:.2f}) sees more upside than comps (median ${comps_median:.2f}) — "
                "the peer group is pricing this business cheaper than its own cash flows justify, or "
                "the DCF's growth assumptions are aggressive relative to peers."
            )

    return {
        "dcf_value": dcf_value,
        "comps_range": comps_range,
        "divergence": divergence,
        "interpretation": interpretation,
    }


def _regression_output_dict(reg_output):
    """Layer2RegressionOutput -> plain dict for the composite result, or None."""
    if reg_output is None:
        return None
    return {
        "fair_value": reg_output.fair_value_per_share,
        "predicted_multiple": reg_output.predicted_multiple,
        "confidence": reg_output.confidence,
        "model_type": reg_output.model_type,
        "formulation": reg_output.formulation,
        "holdout_r2": reg_output.holdout_r2,
        "features_used": reg_output.features_used,
        "features_missing": reg_output.features_missing,
    }


def _peer_display_dict(peer_display):
    if peer_display is None:
        return None
    return {
        "nearest_peers": peer_display.nearest_peers,
        "peer_distances": peer_display.peer_distances,
        "peer_ev_sales_median": peer_display.peer_ev_sales_median,
        "peer_ev_ebitda_median": peer_display.peer_ev_ebitda_median,
        "target_ev_sales": peer_display.target_ev_sales,
        "target_ev_ebitda": peer_display.target_ev_ebitda,
        "target_ev_sales_percentile": peer_display.target_ev_sales_percentile,
        "target_ev_ebitda_percentile": peer_display.target_ev_ebitda_percentile,
        "feature_deviations": peer_display.feature_deviations,
    }


def compute_full_valuation(ticker, bucket=None, include_comps=True):
    """
    Full layered valuation: Layer 1 (DCF) + Layer 2 (Comps).

    Layer 2 methodology is routed per bucket by final_engine.py (see
    LAYER_2_FINAL_SPEC — 6 diagnostic sessions: segmentation, forward
    selection, sector-relative multiples, Ridge, tree regression):
        semiconductors -> RandomForest regression (both multiples, HIGH
                           confidence) + peer display
        mature_tech    -> OLS-relative EV/EBITDA regression only (MEDIUM
                           confidence; EV/Sales not shipped) + peer display
        everything else -> peer display only, no regression shipped

    Bucket resolution: explicit bucket wins; otherwise auto-derived via
    resolve_bucket() (classify() on fresh yfinance sector/industry data).
    """
    ticker = ticker.upper()
    data_flags = []

    bucket, bucket_flags = resolve_bucket(ticker, bucket)
    data_flags.extend(bucket_flags)

    dcf_raw = _run_layer_1_dcf(ticker, bucket)
    dcf_error = dcf_raw.get("error")
    dcf_fair_value = dcf_raw.get("per_share_value") if not dcf_error else None
    dcf_upside = dcf_raw.get("upside_downside") if not dcf_error else None
    market_price = dcf_raw.get("current_price")

    layer_1_dcf = {
        "fair_value": dcf_fair_value,
        "upside": dcf_upside,
        "details": dcf_raw,
    }

    comps_result = None
    layer_2_comps = None
    if include_comps:
        comps_result = compute_layer_2_comps(ticker, bucket=bucket)
        market_price = market_price or comps_result.market_price

        layer_2_comps = {
            "ev_sales_regression": _regression_output_dict(comps_result.ev_sales_regression),
            "ev_ebitda_regression": _regression_output_dict(comps_result.ev_ebitda_regression),
            "peer_display": _peer_display_dict(comps_result.peer_display),
        }
        data_flags.extend(comps_result.data_flags)

    if dcf_error:
        data_flags.append(f"Layer 1 DCF error: {dcf_error}")
    else:
        data_flags.extend(dcf_raw.get("data_flags", []))

    signal_summary = _build_signal_summary(dcf_fair_value, comps_result)

    return {
        "ticker": ticker,
        "bucket": bucket,
        "market_price": market_price,
        "layer_1_dcf": layer_1_dcf,
        "layer_2_comps": layer_2_comps,
        "signal_summary": signal_summary,
        "data_flags": data_flags,
    }


if __name__ == "__main__":
    for ticker, bucket in [
        ("AMD", "semiconductors"),
        ("NVDA", "semiconductors"),
        ("MSFT", "mature_tech"),
        ("AAPL", "mature_tech"),
        ("GOOGL", "mature_tech"),
    ]:
        result = compute_full_valuation(ticker, bucket=bucket)
        print(f"\n{'=' * 70}\n{result['ticker']} ({result['bucket']})\n{'=' * 70}")
        print(f"  Market price:  ${result['market_price']:.2f}" if result['market_price'] else "  Market price:  N/A")
        l1 = result["layer_1_dcf"]
        print(f"  Layer 1 DCF:   {'$%.2f' % l1['fair_value'] if l1['fair_value'] else 'N/A'}")
        l2 = result["layer_2_comps"]
        if l2:
            evs = l2["ev_sales_regression"]
            eve = l2["ev_ebitda_regression"]
            print(f"  Layer 2 EV/Sales:  {'$%.2f (%s confidence, %s)' % (evs['fair_value'], evs['confidence'], evs['model_type']) if evs else 'not shipped for this bucket'}")
            print(f"  Layer 2 EV/EBITDA: {'$%.2f (%s confidence, %s)' % (eve['fair_value'], eve['confidence'], eve['model_type']) if eve else 'not shipped for this bucket'}")
            pd = l2["peer_display"]
            if pd:
                print(f"  Peers: {pd['nearest_peers']}")
        print(f"  Signal: {result['signal_summary']['interpretation']}")
