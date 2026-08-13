# Alpha Research Layer — Results Report

*Cross-sectional factor test of the derived valuation features from Layers 1-2, per ALPHA_LAYER_SPEC.md.*

---

## Universe and panel

**Universe:** 51 tickers — the hand-curated "core" lists from `valuation/comps/universe.py`
(29 semiconductors + 22 mature_tech), rather than the full ~157-ticker universe (including the
auto-scraped "expansion" tickers). This was a data-availability constraint, not a design choice:
fetching the full universe hit escalating yfinance rate-limiting (observed fetch time went from
~0.3s/ticker to 60-140s/ticker and climbing across two attempts on the full universe). The core
51-ticker list is the pre-existing, non-cherry-picked curated subset — not selected to produce
favorable results — and comfortably clears both the 15-ticker IC threshold and 20-ticker decile
threshold on nearly every date. 2 tickers (TSM, ASML) were excluded for non-USD reporting
currency, an existing filter, not something new.

**Panel:** 1,393 (ticker, date) rows across 34 monthly rebalancing dates, 2022-09-28 to
2025-06-28 (window clipped to what point-in-time fundamentals + 12m-forward price data actually
support). Cross-sectional width: min 7, max 49, mean 41 tickers/date — the earliest 4 dates
(2022-09 to 2022-12) are thin (7-11 tickers) because most tickers' first fiscal-year snapshot
hadn't posted yet; from 2023-01 onward width is consistently 40-49.

**Feature coverage** (fraction of (ticker, date) rows with a non-null value):

| Feature | Coverage | Note |
|---|---|---|
| `gross_margin` | 100% | |
| `ev_ebitda` | 95% | |
| `rd_capitalized_margin` | 90% | null for non-R&D businesses (correct exclusion, not a gap) |
| `roic` | 89% | |
| `goodwill_ratio` | 89% | |
| `ex_goodwill_roic` | 85% | |
| `dcf_upside` | 83% | |
| `trailing_growth` | 58% | null whenever a ticker's *earliest* available snapshot is active (no prior-year revenue to compare against) |
| `sbc_dilution_rate` | 23% | needs a 3-year trailing window from an already-scarce ~4-5yr yfinance history; null for roughly the first fiscal year each ticker is in the panel |

No feature was silently filled — missing means excluded from that date's cross-section for that
alpha, per spec.

---

## Main results table

All 18 alphas × 2 horizons, sorted by `ic_ir` descending:

| alpha_id | horizon | mean_ic | std_ic | ic_ir | ic_tstat | pct_positive | sharpe | decile_spread | hit_rate | hit_rate_extremes | turnover | n_dates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A3 | 12 | 0.0859 | 0.1039 | **0.827** | **4.45** | 0.828 | 0.638 | 0.605 | 0.539 | 0.707 | 0.014 | 30 |
| A4 | 12 | 0.1285 | 0.1669 | 0.770 | 2.98 | 0.733 | 0.839 | 0.321 | 0.562 | 0.575 | 0.012 | **15** |
| A3 | 6 | 0.0321 | 0.0740 | 0.434 | 2.34 | 0.655 | 0.746 | 0.116 | 0.510 | 0.566 | 0.014 | 30 |
| A7 | 6 | 0.0876 | 0.2847 | 0.308 | 1.31 | 0.500 | -0.415 | -0.116 | 0.573 | 0.553 | 0.021 | **20** |
| A8n | 6 | 0.0499 | 0.1909 | 0.261 | 1.43 | 0.500 | -0.097 | -0.013 | 0.515 | 0.571 | 0.020 | 31 |
| A3n | 6 | 0.0446 | 0.1851 | 0.241 | 1.30 | 0.655 | -0.089 | -0.011 | 0.512 | 0.533 | 0.034 | 30 |
| A1 | 6 | 0.0471 | 0.1965 | 0.240 | 1.29 | 0.552 | -0.301 | -0.072 | 0.523 | 0.533 | 0.021 | 30 |
| C3 | 6 | 0.0507 | 0.2230 | 0.227 | 1.25 | 0.500 | -0.139 | -0.022 | 0.518 | 0.500 | 0.043 | 31 |
| A8 | 6 | 0.0280 | 0.2079 | 0.135 | 0.74 | 0.467 | -0.255 | -0.040 | 0.498 | 0.555 | 0.018 | 31 |
| A2n | 6 | 0.0108 | 0.1826 | 0.059 | 0.32 | 0.400 | -0.121 | -0.026 | 0.502 | 0.538 | 0.023 | 32 |
| A2 | 6 | 0.0060 | 0.1885 | 0.032 | 0.17 | 0.367 | -0.128 | -0.027 | 0.495 | 0.535 | 0.018 | 32 |
| A8n | 12 | 0.0047 | 0.1681 | 0.028 | 0.15 | 0.433 | -0.450 | -0.370 | 0.487 | 0.465 | 0.020 | 31 |
| A4 | 6 | 0.0083 | 0.3356 | 0.025 | 0.10 | 0.600 | 0.118 | 0.018 | 0.509 | 0.494 | 0.012 | **15** |
| A3n | 12 | 0.0031 | 0.1968 | 0.016 | 0.09 | 0.655 | 0.006 | 0.002 | 0.504 | 0.591 | 0.034 | 30 |
| C3 | 12 | -0.0003 | 0.1684 | -0.002 | -0.01 | 0.533 | -0.506 | -0.301 | 0.513 | 0.441 | 0.043 | 31 |
| C2 | 6 | -0.0004 | 0.1795 | -0.002 | -0.01 | 0.448 | -0.065 | -0.010 | 0.476 | 0.532 | 0.032 | 30 |
| A6 | 6 | -0.0020 | 0.2907 | -0.007 | -0.04 | 0.533 | -0.492 | -0.201 | 0.505 | 0.435 | 0.014 | 32 |
| A9n | 6 | -0.0034 | 0.2585 | -0.013 | -0.07 | 0.400 | -0.335 | -0.036 | 0.493 | 0.471 | 0.039 | 32 |
| A1n | 6 | -0.0104 | 0.2541 | -0.041 | -0.22 | 0.483 | -0.026 | -0.004 | 0.476 | 0.504 | 0.034 | 30 |
| A1n | 12 | -0.0124 | 0.2604 | -0.048 | -0.26 | 0.483 | 0.100 | 0.053 | 0.471 | 0.504 | 0.034 | 30 |
| A5n | 12 | -0.0126 | 0.2146 | -0.059 | -0.32 | 0.400 | 0.529 | 0.123 | 0.488 | 0.564 | 0.038 | 31 |
| A7 | 12 | -0.0219 | 0.3439 | -0.064 | -0.27 | 0.500 | -0.606 | -1.001 | 0.505 | 0.453 | 0.021 | **20** |
| A8 | 12 | -0.0185 | 0.1894 | -0.097 | -0.53 | 0.400 | -0.491 | -0.467 | 0.464 | 0.461 | 0.018 | 31 |
| A1 | 12 | -0.0180 | 0.1488 | -0.121 | -0.65 | 0.414 | -0.457 | -0.533 | 0.471 | 0.463 | 0.021 | 30 |
| A9 | 12 | -0.0200 | 0.1463 | -0.137 | -0.75 | 0.467 | -0.092 | -0.026 | 0.480 | 0.508 | 0.033 | 32 |
| A9n | 12 | -0.0432 | 0.2387 | -0.181 | -0.99 | 0.500 | -0.098 | -0.025 | 0.469 | 0.529 | 0.039 | 32 |
| C1 | 6 | -0.0351 | 0.1744 | -0.201 | -1.08 | 0.448 | -0.322 | -0.058 | 0.479 | 0.483 | 0.052 | 30 |
| A2n | 12 | -0.0443 | 0.1651 | -0.268 | -1.47 | 0.467 | -0.455 | -0.567 | 0.479 | 0.450 | 0.023 | 32 |
| C2 | 12 | -0.0350 | 0.1273 | -0.275 | -1.48 | 0.241 | -0.224 | -0.178 | **0.446** | 0.516 | 0.032 | 30 |
| A2 | 12 | -0.0513 | 0.1761 | -0.291 | -1.60 | 0.433 | -0.446 | -0.558 | 0.475 | 0.458 | 0.018 | 32 |
| A5 | 12 | -0.0407 | 0.1396 | -0.292 | -1.60 | 0.433 | -0.228 | -0.110 | 0.479 | 0.556 | 0.032 | 31 |
| A6 | 12 | -0.1561 | 0.2994 | -0.521 | -2.86 | 0.333 | -0.598 | -1.211 | **0.445** | **0.280** | 0.014 | 32 |
| A9 | 6 | -0.0727 | 0.1328 | -0.547 | -3.00 | 0.367 | -0.644 | -0.057 | 0.470 | 0.414 | 0.033 | 32 |
| A5n | 6 | -0.0804 | 0.1403 | -0.573 | -3.14 | 0.267 | 0.151 | 0.013 | **0.445** | 0.453 | 0.038 | 31 |
| C1 | 12 | -0.0861 | 0.1459 | -0.590 | -3.18 | 0.276 | -0.382 | -0.262 | 0.474 | 0.434 | 0.052 | 30 |
| A5 | 6 | -0.0827 | 0.1385 | -0.597 | -3.27 | 0.267 | -0.212 | -0.034 | **0.443** | 0.466 | 0.032 | 31 |

Bold `n_dates` = under the 20-date threshold spec asks to flag as underpowered/directional-only
(A4 at both horizons: 15; A7 at both horizons: 20, right at the line). Bold `hit_rate` /
`hit_rate_extremes` = below 0.45 (see "Hit rate" section below).

### Hit rate

Added per-company directional accuracy: for each date, split both the alpha and the forward
return at their own median, and check whether each ticker landed on the same side of both
splits. 0.50 is the null; `hit_rate_extremes` restricts the count to the top/bottom decile
(tercile if thin) by alpha value — where an actual long/short strategy would be trading, and
where a well-behaved alpha's conviction should translate into higher accuracy.

**A3 (12m) — the report's one holdout-validated finding — behaves exactly as a well-behaved
alpha should**: `hit_rate` 0.539 (modest but real edge over the 0.50 null), and
`hit_rate_extremes` 0.707 — accuracy jumps sharply when restricted to the names the alpha is
most convicted about. That's independent corroboration of the IC-based result from a completely
different angle (per-company accuracy rather than rank correlation).

**Four alphas show `hit_rate` below 0.45** (flagged per instruction, bold above): C2 (12m,
0.446), A6 (12m, 0.445), A5n (6m, 0.445), A5 (6m, 0.443). None of these are a surprise or an
inverted-signal discovery on their own — they're the same alphas already flagged in the main
table for strongly negative `ic_tstat` (A6 12m: -2.86; A9/A5/A5n/C1: all ≤ -3.0 except C2 at
-1.48, which is the only new entry to this list). The hit-rate lens simply corroborates the
negative-IC finding from a different angle rather than surfacing anything new. **A6 (12m)** is
the most extreme: `hit_rate_extremes` 0.280 — a well-behaved alpha's conviction bucket would show
*higher* accuracy than its median-split accuracy (0.445); here it's dramatically *lower*, meaning
the highest-conviction names were the most *wrong* — consistent with the memory-stock-rally
explanation already investigated in the "unexpectedly strong or inverted behavior" section below
(a handful of low-margin memory names — MU, STX, WDC — dominating the bottom-margin decile and
then rallying +560-960%). Not a new bug, just the same concentrated small-n effect showing up
in a second metric.

Turnover is low across the board (0.01-0.05), consistent with quarterly-at-most fundamental
updates — no sign of noisy/leaking inputs.

---

## Holdout validation

Dates split chronologically in half (17 early, 17 late — no shuffling). Top 3 alphas by
`ic_ir` on the early half only, then evaluated on the late half with no refitting.

**6-month horizon:**

| alpha_id | early_mean_ic | early_ic_ir | late_mean_ic | late_ic_ir | sign_held | verdict |
|---|---|---|---|---|---|---|
| A3n | 0.063 | 0.615 | 0.031 | 0.137 | True | Degrades |
| A8n | 0.139 | 0.576 | -0.018 | -0.175 | False | **Flips** |
| A8 | 0.120 | 0.457 | -0.042 | -0.349 | False | **Flips** |

**12-month horizon:**

| alpha_id | early_mean_ic | early_ic_ir | late_mean_ic | late_ic_ir | sign_held | verdict |
|---|---|---|---|---|---|---|
| A3n | 0.101 | **1.924** | -0.066 | -0.285 | False | **Flips** |
| A3 | 0.139 | 1.027 | 0.049 | 0.917 | True | **Holds** |
| A2 | 0.093 | 0.729 | -0.162 | -1.367 | False | **Flips** |

**A3n is the cautionary tale spec warned about:** it has the single best early-period `ic_ir`
of all 18 alphas at either horizon (1.92) and completely reverses sign out of sample. Taken alone
at full-period, or worse, at early-period-only, it would have looked like the standout result.
It isn't one — it's noise. A8 and A8n (baseline ROIC alphas) are *also* unstable across the
split, so this isn't a derived-feature-specific problem; it's a general caution about a
34-date, mostly-overlapping-observation panel and 18-way alpha comparison.

**A3 (12m) is the one alpha that clears the spec's bar for "yes":** `ic_tstat` 4.45 > 2 on the
full period, and it survives holdout (`late_ic_ir` 0.917 is well over half of `early_ic_ir`
1.027, sign held). A4 (12m) also clears `ic_tstat` > 2 on the full period (2.98) but *could not
be tested* in holdout — it had zero valid dates in the early half (its 3-year-lookback
requirement means it only becomes available in the later fiscal years each ticker is in the
panel), so per spec it's reported as **inconclusive**, not as a second "yes."

---

## Derived vs. baseline comparison

The sharpest version of the research question:

| derived | baseline | horizon | derived ic_ir | baseline ic_ir | derived tstat | baseline tstat | derived beats baseline |
|---|---|---|---|---|---|---|---|
| A1 (ex-goodwill ROIC) | A8 (reported ROIC) | 6m | 0.240 | 0.135 | 1.29 | 0.74 | Yes (but neither significant) |
| A1 | A8 | 12m | -0.121 | -0.098 | -0.65 | -0.53 | No |
| A1n | A8n | 6m | -0.041 | 0.261 | -0.22 | 1.43 | No |
| A1n | A8n | 12m | -0.048 | 0.028 | -0.26 | 0.15 | No |
| C1 (derived composite) | C3 (baseline composite) | 6m | -0.201 | 0.227 | -1.08 | 1.25 | No |
| C1 | C3 | 12m | -0.590 | -0.002 | -3.18 | -0.01 | No |

**Ex-goodwill ROIC does not beat reported ROIC** in this sample. It wins narrowly and
insignificantly at 6m raw, but loses (sometimes badly, as in C1 vs C3 at 12m) everywhere else,
including every bucket-neutralized comparison. The months of goodwill-adjustment work did not
translate into better cross-sectional return signal here — that should be stated plainly rather
than softened.

The one place a derived feature *does* show real signal is **A3 (`normalized_reinvestment`,
raw, not bucket-neutralized)** — but that's a different feature from the ones this specific
comparison tests, and it isn't part of either composite (C1/C2 use A1n/A2n, not A3n).

---

## Narrative summary

**Does any alpha clear `ic_tstat > 2` and survive holdout?** Yes, exactly one: **A3
(`rank(normalized_reinvestment)`, 12-month horizon)**. Full-period `ic_tstat` = 4.45, `ic_ir` =
0.83, and it holds up in the temporal holdout (early `ic_ir` 1.03 → late `ic_ir` 0.92, same
sign). This is a genuine, well-supported finding: the normalized reinvestment rate — Layer 1's
`max(median, weighted_3y)` reinvestment signal with ΔWC removed — carries real 12-month
cross-sectional return signal in this sample. A4 (`sbc_dilution_rate`, 12m) also clears the
t-stat bar on the full period (2.98) but couldn't be evaluated in holdout at all due to its own
data-availability constraint, so it's inconclusive rather than a second confirmed result.

**Do derived features beat the baseline features they were built to improve on?** Mostly no.
The direct comparison the spec asks for — ex-goodwill ROIC (A1/A1n) vs. reported ROIC (A8/A8n),
and the derived composite (C1) vs. the baseline composite (C3) — does not favor the derived
side. C1 is the single worst-performing alpha in the entire 18-alpha set at the 12-month horizon
(`ic_tstat` -3.18), driven by A1n's weak standalone performance. A9 (baseline value, ev_ebitda)
and A5 (dcf_upside) both show strong *negative* t-stats at 6m (-3.00 and -3.27) — over this
window, "cheap"-style value and "high implied upside" both cross-sectionally *underperformed*,
consistent with a growth/momentum-favoring regime (AI-driven tech/semi rally, 2022-2025) rather
than a value-friendly one. That's a real historical-period effect, not obviously a defect in
either signal.

**Any feature with unexpectedly strong or inverted behavior?** A6 (`gross_margin`, 12m) shows an
implausible-looking -121% decile spread. Investigated directly: it's real, not a lookahead bug.
The bottom-margin decile on 2025-06-28 included MU, STX, and WDC — memory/storage names that
returned +560% to +960% over the following 12 months during the 2025-2026 AI-driven memory
shortage rally — against a top-margin decile of software names (NOW, ADBE, TEAM, META) that were
mostly flat-to-down over the same window. With only ~4-5 names per decile leg, a handful of
extreme individual moves can swing the whole metric; this is a legitimate small-n concentration
effect, not a bug, and it's a reminder that decile spreads on a 40-49-name universe are noisier
than on a full 500+-name universe.

**Data limitations.** The universe is 51 tickers, not the full ~157 (yfinance rate-limiting
forced this reduction — see "Universe and panel" above). `sbc_dilution_rate` (A4) has only 23%
coverage and its two rows both sit at n_dates=15, under the spec's 20-date underpowered
threshold — read A4's t-stat as directional only. `trailing_growth` (A7, baseline) sits right at
the 20-date line for the same reason (data depth). More fundamentally: 34 monthly rebalancing
dates is not 34 independent observations. Fundamentals update once a year per ticker in
practice, so a t-stat computed from ~30 monthly dates substantially overstates the effective
sample size — treat every t-stat in this report, including A3's 4.45, as optimistic relative to
what a non-overlapping annual sample would show. This is exactly the caution the spec asked to
be stated plainly, and it's the main reason A3's result, while the strongest and holdout-
validated finding here, should be treated as suggestive rather than definitive without a larger,
longer-history universe to confirm it.

---

## Non-overlapping-return re-evaluation (Prof. Travis Johnson methodology)

The main results above correlate the signal at date T against the 12-month-forward return from
T. Consecutive monthly rebalancing dates share 11 of 12 months of that return window, so the
~30 dates are not independent observations and the reported t-stats (A3's 4.45 in particular)
overstate significance.

**Methodology.** Per the Hodrick-equivalence guidance received: `cov(signal_t, sum of r over
t+1..t+12) ≡ cov(sum of signal over t-12..t-1, r_t)`. Averaging the signal over the trailing 12
months and correlating against the *non-overlapping* 1-month-forward return targets the same
underlying economics while removing the return-window overlap. Two variants, since the order of
time-averaging vs. cross-sectional ranking is a real choice:

- **Variant (a)**: average the raw feature over the trailing 12 months, *then* rank
  cross-sectionally at date T. More faithful to the equation above.
- **Variant (b)**: rank cross-sectionally at each date first (i.e. today's standard alpha
  value), *then* average the trailing 12 months of that ranked signal. More robust to outliers
  in the raw feature.

Both variants call the *exact same, unmodified* `alphas.ALPHAS[alpha_id]` functions the original
evaluation uses — variant (a) feeds them a feature-smoothed panel, variant (b) smooths their
output. `operators.py`, `alphas.py`, and `panel.py` were not touched; the only new code is
`valuation/alpha/nonoverlap.py` plus two additive Newey-West functions in `evaluate.py`. Standard
errors are Newey-West (Bartlett kernel) with 2 and 3 lags, reported alongside the naive
`mean_ic / (std_ic / sqrt(n))` t-stat, as a safeguard against autocorrelation in the IC series
itself (signal persistence) — a separate concern from the return-window overlap this whole
exercise targets, per the guidance that signal persistence alone isn't the problem being fixed.

**A data-vintage caveat, discovered running this.** This session's scratchpad cache (all fetched
price/fundamentals data) had been cleared since the original report, so all 51 tickers were
re-fetched fresh from yfinance. Recomputing the *original* 12-month-overlapping evaluation on
this fresh pull — needed as an apples-to-apples "old" baseline, since comparing yesterday's frozen
numbers against today's freshly-pulled "new" numbers would confound methodology with data
vintage — gave **different numbers than the original report** (A3 12m: mean_ic 0.086→0.083,
`ic_tstat` 4.45→**7.09**; row count 1393→1345; minimum cross-sectional width in the ramp-up
months 7→3). This is pure yfinance day-to-day data drift (confirmed: no fetch errors, same
non-USD exclusions for TSM/ASML, same universe) — some tickers' earliest available annual
statement shifted, thinning several early-period cross-sections below the 15-ticker IC minimum.
It's an unplanned but relevant finding in its own right: it shows the pipeline's headline numbers
are sensitive not only to the overlap issue this exercise targets, but to ~24 hours of upstream
data revision — independent, reinforcing evidence that no single point estimate here should be
read as precise. The comparison below uses the fresh pull consistently for both "old" and "new"
columns, isolating the methodology effect from this drift.

**Extended panel.** A 1-month-forward target only needs one more month of price history per row
(vs. twelve), so the usable window extends well past the original evaluation's tail: 45
rebalancing dates (2022-09-28 to 2026-05-28) vs. the original 34 (2022-09-28 to 2025-06-28),
1,884 rows vs. 1,345 in the reproduced-old subset. `fwd_return_1m` coverage is 100%.

### Comparison table (sorted by `new_ic_ir_a`)

† = sign flip vs. old (variant a or b) — full list and discussion below.

| alpha_id | old_mean_ic | old_tstat | new_mean_ic (a) | new_tstat_NW2 (a) | new_tstat_NW3 (a) | new_mean_ic (b) | new_tstat_NW2 (b) | new_tstat_NW3 (b) | n_dates |
|---|---|---|---|---|---|---|---|---|---|
| A3 | 0.0831 | 7.09 | 0.0256 | 1.46 | 1.92 | 0.0164 | 0.94 | 1.24 | 27 |
| A1 † | -0.0530 | -2.64 | 0.0301 | 0.84 | 0.83 | 0.0190 | 0.53 | 0.52 | 27 |
| A1n † | -0.0491 | -1.00 | 0.0259 | 0.50 | 0.47 | 0.0772 | 1.21 | 1.12 | 27 |
| A8n † | -0.0429 | -2.23 | 0.0126 | 0.29 | 0.30 | 0.0101 | 0.23 | 0.23 | 27 |
| C2 † | -0.0561 | -3.59 | 0.0109 | 0.26 | 0.27 | 0.0313 | 0.77 | 0.77 | 27 |
| A5n † | -0.0232 | -0.58 | 0.0119 | 0.22 | 0.21 | 0.0145 | 0.26 | 0.26 | 27 |
| A8 † | -0.0741 | -3.23 | 0.0063 | 0.14 | 0.13 | 0.0050 | 0.11 | 0.11 | 27 |
| A9 | -0.0251 | -1.09 | -0.0305 | -0.58 | -0.60 | -0.0220 | -0.41 | -0.41 | 29 |
| A2n | -0.0609 | -2.29 | -0.0317 | -0.81 | -0.79 | -0.0512 | -1.31 | -1.34 | 29 |
| A3n † | 0.0101 | 0.25 | -0.0338 | -0.86 | -0.86 | -0.0430 | -1.17 | -1.19 | 27 |
| A2 | -0.0678 | -2.42 | -0.0395 | -0.97 | -0.95 | -0.0355 | -0.87 | -0.84 | 29 |
| A4 † | 0.0406 | 0.82 | -0.0615 | -0.72 | -0.74 | -0.0453 | -0.55 | -0.57 | 14 |
| A5 | -0.0590 | -2.28 | -0.0631 | -1.26 | -1.35 | -0.0610 | -1.22 | -1.29 | 27 |
| C3 | -0.0295 | -0.99 | -0.0483 | -1.25 | -1.30 | -0.0222 | -0.54 | -0.54 | 27 |
| C1 † | -0.1071 | -4.07 | -0.0505 | -1.30 | -1.34 | 0.0053 | 0.14 | 0.15 | 27 |
| A7 | -0.0138 | -0.15 | -0.1145 | **-2.02** | -1.98 | -0.1142 | **-2.06** | -2.02 | 17 |
| A9n | -0.0414 | -0.92 | -0.1079 | **-2.10** | **-2.16** | -0.0742 | -1.35 | -1.32 | 29 |
| A6 | -0.1694 | -3.26 | -0.1462 | **-2.58** | **-2.39** | -0.1405 | -2.56 | -2.38 | 29 |

**Only three alphas clear \|NW-tstat\| ≥ 2 under the new spec (bold above), and all three are
negative**: A6, A9n, A7. **No alpha shows a robust, statistically-distinguishable-from-zero
positive signal under the corrected methodology** — not even A3.

### Holdout under the new specification

Same protocol as the original holdout (chronological split, no shuffling, top 3 by early-period
`ic_ir`, evaluated on the late half with no refitting) — applied to the new spec's `fwd_return_1m`
IC series. 45 dates split at index 22 (early: through 2024-06-28; late: from 2024-07-28).

**Variant (a):**

| alpha_id | early_mean_ic | early_ic_ir | late_mean_ic | late_ic_ir | sign_held | verdict |
|---|---|---|---|---|---|---|
| C3 | 0.1143 | 1.385 | -0.0765 | -0.430 | False | Flips |
| A8n | 0.1672 | 1.126 | -0.0143 | -0.075 | False | Flips |
| A2n | 0.1318 | 1.063 | -0.0744 | -0.432 | False | Flips |

**Variant (b):**

| alpha_id | early_mean_ic | early_ic_ir | late_mean_ic | late_ic_ir | sign_held | verdict |
|---|---|---|---|---|---|---|
| C3 | 0.0984 | 1.468 | -0.0432 | -0.228 | False | Flips |
| A2 | 0.1463 | 1.222 | -0.0830 | -0.482 | False | Flips |
| A8 | 0.1723 | 1.025 | -0.0241 | -0.119 | False | Flips |

**Every single top-3 alpha in both variants flips sign from early to late.** Nothing survives
holdout under the new spec — a stronger, more general version of the A3n cautionary tale from the
original holdout. This should be read alongside a real limitation: the early-period ranking here
is based on only **4-6 valid dates per alpha** (vs. 13-15 in the original holdout), because the
trailing-12-month averaging requirement eats the first 11 months of any ticker's usable window on
top of the panel's already-thin early cross-section (3-16 tickers before 2023-04). A ranking built
on 4 dates is not a reliable ranking regardless of methodology — this holdout is underpowered on
its own terms, independent of the overlap-correction question.

### A3 specifically

A3 (`rank(normalized_reinvestment)`, the one alpha that cleared the bar in the original report)
was **not** among the top 3 by early-period `ic_ir` under either new-spec variant (ranked 7th of
16 in variant a, 9th of 16 in variant b) — so it's not part of the "official" holdout table above.
Computed directly instead:

| | old (12m overlapping) | new variant (a) | new variant (b) |
|---|---|---|---|
| Full-period mean_ic | 0.0831 | 0.0256 | 0.0164 |
| Full-period t-stat | 7.09 (naive) | 0.96 naive / 1.46 NW2 / **1.92 NW3** | 0.62 naive / 0.94 NW2 / 1.24 NW3 |
| n_dates | 27 | 27 | 27 |
| Early-period ic_ir (holdout) | — | 0.742 (n=4) | 0.522 (n=4) |
| Late-period ic_ir (holdout) | — | 0.143 (n=23) | 0.084 (n=23) |
| Sign held early→late | — | True | True |

**A3 does not clear a t-stat of 2 under the corrected methodology** — closest is variant (a)'s
NW3 estimate at 1.92, just short. **It does not survive holdout in any meaningful sense**: the
sign holds (positive in both halves), but `late_ic_ir` is roughly a fifth of `early_ic_ir` in
both variants — well below the 50%-retention bar used to classify "Holds" everywhere else in this
report, and the early-period estimate itself rests on only 4 dates. And the smoothing didn't only
cost A3 its t-stat: its raw `mean_ic` shrank by 69% (0.083→0.026) too, not just its significance —
so part of A3's original strength really was tied to the specific (unaveraged) timing of the
signal interacting with the overlapping 12-month window, not purely a statistical-inflation
artifact. **The one positive finding from the original report does not survive this more
rigorous test.** That is exactly the outcome this exercise was designed to be able to produce,
and it should be reported as such rather than reframed as a failure of the exercise.

### Interpretation

**The t-stat fell, as expected — that is the exercise working, not a failure.** A3's naive
t-stat dropped from 7.09 (this run's fresh 12m-overlapping recompute) to at most 1.92 (NW3,
variant a). More broadly, only 3 of 18 alphas clear \|NW-tstat\| ≥ 2 under the new spec, all
negative, none previously flagged as a "finding" — the correction didn't just weaken A3, it
weakened everything, which is the expected signature of a panel whose apparent precision was
substantially an overlap artifact.

**Mean IC changes were mixed, not uniform** — smoothing "helped" (increased \|mean_ic\| under
variant a) for 7 of 18 alphas and hurt the other 11. A3 is in the "hurt" group, and hurt
substantially (largest proportional drop of any alpha with a meaningful old signal). A9n is the
clearest case of smoothing *helping*: its old signal was weak and non-significant
(mean_ic -0.041, tstat -0.92) but strengthens under the new spec (mean_ic -0.108, NW2 -2.10) —
smoothing revealed a signal the overlapping-window evaluation didn't detect clearly, in the
opposite direction from A3.

**Variants (a) and (b) diverge materially for several alphas** — most notably A1n (new mean_ic
0.026 under (a) vs. 0.077 under (b) — nearly 3x), C1 (-0.050 vs. +0.005 — different sign), and C2
(0.011 vs. 0.031 — ~3x). These are alphas whose raw underlying feature is noisier/more
outlier-prone (A1n and C1/C2 both route through `ex_goodwill_roic`, which can take extreme values
for heavy-acquirer names like AVGO), which is exactly where variant (b)'s "rank first, then
average" order should be expected to diverge from variant (a)'s "average first, then rank" — (b)
is more robust to a single extreme raw-feature month, (a) is more exposed to it. A3 itself is
comparatively stable across variants (0.0256 vs. 0.0164 — same order of magnitude, same sign),
consistent with `normalized_reinvestment` already being a smoothed quantity by construction
(`max(median, weighted_3y)` in Layer 1) before this exercise smooths it again.

**Sign flips**: 8 alphas flip under variant (a) — A1, A1n, A8n, C2, A5n, A8, A3n, A4 — and a 9th
(C1) flips only under variant (b). Every flipped alpha has an old `\|tstat\|` under 3.6 and a new
`\|NW-tstat\|` under 1.3 in both directions — these are alphas crossing zero in a region where
neither the old nor the new estimate is statistically distinguishable from zero. The one strong
old alpha that does NOT flip (A6, old tstat -3.26) stays strongly negative and significant under
the new spec too (NW2 -2.58) — the correction doesn't manufacture instability in an already-clear
signal, only in already-marginal ones. No flip here should be read as economically meaningful;
they're bookkept above per the constraint to flag them, not because any of them constitutes a new
finding.

---

## Implementation notes

- New code, additive only: `valuation/alpha/{operators,panel,alphas,evaluate,runner}.py`.
- One additive extension to `valuation/comps/backtest/historical_data.py`: six new point-in-time
  fields (`goodwill`, `roic_ex_goodwill`, `goodwill_ratio`, `rd_capitalized_margin`,
  `sbc_dilution_rate`, `normalized_reinvestment`) computed per fiscal-year column using data
  already fetched there — no new network calls, and confirmed backward-compatible (existing
  `framework.py`/`dcf_approximation.py` callers regression-tested unaffected).
  `valuation/tech/shared/*.py` and `valuation/inputs/growth.py` (Layer 1/2 production code) were
  not modified — those modules are live-only (hardcoded to the latest fiscal column) and were
  deliberately not reused for point-in-time reconstruction.
- Unit tests for the three operators: `test_alpha_operators.py` (repo root, matches existing
  `test_comps_unit.py` convention) — all pass.
- No alphas were added, removed, or reweighted after seeing results. The 18-alpha list and every
  threshold (15-ticker IC minimum, 20-ticker decile minimum, ±5% SBC cap, etc.) match the spec
  as pre-declared.
- Added `hit_rate` / `hit_rate_extremes` (per-company median-split directional accuracy, plain
  and conviction-restricted) to `valuation/alpha/evaluate.py`, wired into `evaluate_alpha` and
  `main_results_table` (`valuation/alpha/runner.py`). Additive only — `operators.py`, `alphas.py`,
  and `panel.py` untouched; no existing metric, alpha definition, or threshold changed. Verified
  on synthetic data first (perfect/inverted/random alignment all recovered the expected 1.0 /
  0.0 / ~0.5) before re-running on the cached full panel — no data refetch needed since the
  universe, features, and alpha definitions are unchanged from the prior run.
- Added the non-overlapping-return re-evaluation as a new, separate module:
  `valuation/alpha/nonoverlap.py` (extended-panel builder, trailing-average signal variants a/b,
  comparison-table and holdout builders) plus two additive Newey-West functions in
  `evaluate.py` (`newey_west_se_of_mean`, `newey_west_tstat`), sanity-checked against known
  white-noise and AR(1) behavior before use. `operators.py`, `alphas.py`, and `panel.py` remain
  untouched; every alpha in the new module is evaluated by calling the same, unmodified
  `alphas.ALPHAS[alpha_id]` functions the original 12-month-overlapping evaluation uses. The
  original evaluation path is unchanged and still runs — this is a second, additive lens on the
  same panel-construction machinery, not a replacement.
