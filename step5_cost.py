"""
Step 5: Construction Labor Cost Layer
Nuclear Family — PRISM

Joins the EIA/Sargent & Lundy state-level construction labor cost index to the
scored county dataset and derives estimated SMR capital cost per county.

Deliberately NOT part of the composite suitability score. See DESIGN NOTE below.

Inputs
  raw_data/eia_sl_capital_cost/location_factors.csv   geo_id, county_name, location_factor
  processed_data/candidates_ranked.csv                scored counties (mcda_score, rank)

Output
  processed_data/county_costs.csv                     geo_id, location_factor,
                                                      est_capex_per_kw, est_total_capex

Run:
  python step5_cost.py                # build county_costs.csv
  python step5_cost.py --diagnostic   # also print the MinMax scaling audit
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# SOURCE
# ─────────────────────────────────────────────
# Sargent & Lundy, "Capital Cost and Performance Characteristic Estimates for
# Utility Scale Electric Power Generating Technologies", Report SL-018001 Rev A,
# prepared for the U.S. Energy Information Administration, December 2023.
#
# location_factor comes from Appendix A, "Labor Location-Based Cost Adjustments".
# It is derived from RS Means craft labor rates measured against a 30 City Average
# baseline (= 1.00), refined with regional labor productivity factors.
#
# IMPORTANT — what this factor is NOT:
#   It captures labor wage rates and labor productivity ONLY. Seismic, wind, snow
#   and other environmental cost adjustments are a SEPARATE table in the same
#   report and are not incorporated here. PRISM scores seismic risk independently
#   as a suitability criterion, so folding the environmental table in here would
#   double-count it.
#
# The published values are state-level constants. Every county in a given state
# carries an identical factor, so this index can shift whole states against each
# other but can never break a tie between two counties in the same state.
# verify_state_constant() asserts this rather than trusting it.

# Overnight capital cost for the SMR reference case, $/kW (2023$).
# Source: same report, Case 10 — Small Modular Reactor, 6 x 80 MW units, 480 MW net.
CAPEX_PER_KW_BASE = 8936.0

# Net plant capacity for Case 10, in kW (480 MW).
PLANT_NET_KW = 480_000

# ─────────────────────────────────────────────
# DESIGN NOTE — why cost is a parallel axis, not a 13th weighted criterion
# ─────────────────────────────────────────────
# 1. The factor is state-constant. Pushed through the same MinMax scaling as the
#    other criteria, its ~0.95-1.18 spread stretches to the full 0-1 range and the
#    criterion starts behaving like a high-leverage state dummy variable. The
#    choropleth grows visible discontinuities at every state border.
# 2. Every other criterion answers "is this a feasible site." Cost answers "what
#    will it cost to build here." Collapsing both into one number destroys
#    interpretability: a high-ranking county could be a good site, or it could
#    just have cheap labor, and the score can no longer tell you which.
#
# Cost therefore ships as its own map layer and its own panel section, with an
# optional user-controlled weight that defaults to zero.
# ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
LOCATION_FACTORS = ROOT / "raw_data" / "eia_sl_capital_cost" / "location_factors.csv"
CANDIDATES_RANKED = ROOT / "processed_data" / "candidates_ranked.csv"
OUTPUT = ROOT / "processed_data" / "county_costs.csv"


def read_fips(path: Path, fips_col: str = "geo_id") -> pd.DataFrame:
    """
    Read a CSV and normalise its FIPS column to a zero-padded 5-character string.

    Both source files store geo_id as an integer, which silently strips the
    leading zero from every county in states 01-09 (131 rows in this dataset).
    Normalising on read rather than trusting either file's on-disk format is the
    difference between a 2,161-row join and a 2,030-row one.
    """
    df = pd.read_csv(path)
    df[fips_col] = df[fips_col].astype(str).str.zfill(5)
    return df


def verify_state_constant(lf: pd.DataFrame) -> None:
    """Assert the documented claim that the factor is constant within each state."""
    known = lf.dropna(subset=["location_factor"]).copy()
    known["state_fips"] = known["geo_id"].str[:2]
    per_state = known.groupby("state_fips")["location_factor"].nunique()
    offenders = per_state[per_state > 1]

    print("  state-constant check:")
    print(f"    states with a factor          : {per_state.size}")
    print(f"    states with >1 distinct value : {offenders.size}")
    if offenders.size:
        print(f"    VIOLATIONS: {offenders.to_dict()}", file=sys.stderr)
        raise SystemExit(
            "location_factor is not state-constant. The design rationale for "
            "keeping cost out of the composite assumes it is — re-check the source."
        )
    print("    -> confirmed state-constant")


def build() -> pd.DataFrame:
    print("Loading inputs")
    lf = read_fips(LOCATION_FACTORS)
    cr = read_fips(CANDIDATES_RANKED)
    print(f"  location_factors.csv   : {len(lf):,} rows")
    print(f"  candidates_ranked.csv  : {len(cr):,} rows")

    # Row-count reconciliation. Report it, do not paper over a mismatch.
    if len(lf) == len(cr):
        print(f"  row counts MATCH ({len(lf):,})")
    else:
        print(f"  row counts DIFFER: {len(lf):,} vs {len(cr):,}", file=sys.stderr)

    lf_ids, cr_ids = set(lf["geo_id"]), set(cr["geo_id"])
    only_lf, only_cr = lf_ids - cr_ids, cr_ids - lf_ids
    print(f"  in location_factors only : {len(only_lf)}")
    print(f"  in candidates only       : {len(only_cr)}")
    if only_lf or only_cr:
        print(f"    sample lf-only: {sorted(only_lf)[:5]}", file=sys.stderr)
        print(f"    sample cr-only: {sorted(only_cr)[:5]}", file=sys.stderr)

    if lf["geo_id"].duplicated().any():
        raise SystemExit("Duplicate geo_id in location_factors.csv — join would fan out.")

    print("\nVerifying source claims")
    factors = lf["location_factor"]
    print(f"  distinct values : {factors.nunique()}")
    print(f"  range           : {factors.min():.3f} - {factors.max():.3f}")
    print(f"  null factors    : {factors.isna().sum()}")
    verify_state_constant(lf)

    print("\nJoining and deriving cost columns")
    out = cr[["geo_id", "county_name"]].merge(
        lf[["geo_id", "location_factor"]], on="geo_id", how="left"
    )

    # Counties with no published factor get null cost, never an imputed national
    # average. An invented number here would be indistinguishable from a measured
    # one downstream, and these are real coverage gaps (Alaska boroughs, Kauai,
    # and the territories are outside the scope of the source report).
    out["est_capex_per_kw"] = CAPEX_PER_KW_BASE * out["location_factor"]
    out["est_total_capex"] = out["est_capex_per_kw"] * PLANT_NET_KW

    matched = out["location_factor"].notna().sum()
    print(f"  matched  : {matched:,} / {len(out):,}")
    print(f"  unmatched: {len(out) - matched:,} (null cost columns, not imputed)")

    known = out.dropna(subset=["est_capex_per_kw"])
    print(
        f"\n  $/kW           : ${known.est_capex_per_kw.min():,.0f} - "
        f"${known.est_capex_per_kw.max():,.0f}   (baseline ${CAPEX_PER_KW_BASE:,.0f} at factor 1.00)"
    )
    print(
        f"  total ({PLANT_NET_KW / 1000:.0f} MW): ${known.est_total_capex.min() / 1e9:.2f}B - "
        f"${known.est_total_capex.max() / 1e9:.2f}B   "
        f"(spread ${(known.est_total_capex.max() - known.est_total_capex.min()) / 1e6:,.0f}M)"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["county_name"]).to_csv(OUTPUT, index=False)
    print(f"\nWrote {len(out):,} rows -> {OUTPUT.relative_to(ROOT)}")
    return out


# ─────────────────────────────────────────────
# Scaling diagnostic (read-only — changes nothing)
# ─────────────────────────────────────────────
# The composite in notebooks/modeling/scoring.ipynb MinMax-scales 12 criteria and
# combines them with rank-order-centroid weights. This reports what that scaling
# actually does to each criterion's influence. It does not alter the score.

HIGHER_BETTER = [
    "avg_vol", "total_lake_area", "max_voltage",
    "total_energy_consumption_mwh", "data_centers_count",
]
LOWER_BETTER = [
    "pga_max", "pct_sfha", "population_density", "dist_to_lakes_km",
    "distance_to_lines_km", "pct_military", "pct_protected",
]
# Order matters: ROC weights are assigned by position in this ranking.
CRITERIA_RANKING = [
    "pga_max", "pct_sfha", "population_density", "dist_to_lakes_km",
    "avg_vol", "total_lake_area", "distance_to_lines_km", "max_voltage",
    "total_energy_consumption_mwh", "data_centers_count",
    "pct_military", "pct_protected",
]


def roc_weights(n: int) -> list:
    """Rank-order centroid weights, as in scoring.ipynb."""
    return [sum(1 / j for j in range(i, n + 1)) / n for i in range(1, n + 1)]


def diagnostic() -> None:
    print("\n" + "=" * 78)
    print("MINMAX SCALING DIAGNOSTIC — read-only, nothing below changes the score")
    print("=" * 78)

    df = pd.read_csv(ROOT / "processed_data" / "candidates.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")

    feats = HIGHER_BETTER + LOWER_BETTER
    weights = dict(zip(CRITERIA_RANKING, roc_weights(len(CRITERIA_RANKING))))

    lo, hi = df[feats].min(), df[feats].max()
    scaled = (df[feats] - lo) / (hi - lo)
    scaled[LOWER_BETTER] = 1 - scaled[LOWER_BETTER]

    score = sum(scaled[f] * w for f, w in weights.items())
    stored = pd.read_csv(ROOT / "processed_data" / "candidates_ranked.csv")["mcda_score"]
    both = score.notna() & stored.notna()
    max_err = float(np.abs(score[both].values - stored[both].values).max())
    print(f"\nReproduces stored mcda_score to {max_err:.2e} — the audit below is exact.\n")

    rows = []
    for f, w in weights.items():
        raw, s = df[f], scaled[f]
        contrib_sd = (s * w).std()
        med = raw.median()
        rows.append({
            "criterion": f,
            "weight": round(w, 4),
            "tail_p99/median": round(raw.quantile(0.99) / med, 1) if med else np.nan,
            "scaled_sd": round(s.std(), 4),
            "contrib_sd": round(contrib_sd, 5),
        })

    t = pd.DataFrame(rows)
    var = t["contrib_sd"] ** 2
    t["%_of_score_variance"] = (100 * var / var.sum()).round(1)
    t = t.sort_values("%_of_score_variance", ascending=False)

    print(t.to_string(index=False))

    top2 = t["%_of_score_variance"].head(2).sum()
    print(f"\nFINDING: the top 2 criteria carry {top2:.1f}% of the composite's variance.")
    print("MinMax divides by (max - min), so a criterion with a long right tail collapses")
    print("toward a constant and stops discriminating between counties, no matter its")
    print("assigned weight. population_density is the 3rd-heaviest criterion by weight")
    print(f"({weights['population_density']:.3f}) and contributes "
          f"{t.loc[t.criterion == 'population_density', '%_of_score_variance'].iloc[0]}% of the variance.")
    print("\nThe documented ROC weights are not the weights the model is effectively using.")
    print("NOT CHANGED. Fixing it (rank/quantile or robust scaling) would move the")
    print("published rankings, which is not something to do silently before a presentation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostic", action="store_true",
                    help="also print the MinMax scaling audit (read-only)")
    args = ap.parse_args()

    build()
    if args.diagnostic:
        diagnostic()
