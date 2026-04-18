"""
Step 4: Multi-Objective Optimization (MOO)
Nuclear Family - Illinois Data Science Club Data Dive Spring 2026

Approach:
  - Run NSGA-II (pymoo) on candidates.csv with 6 objectives
  - Also extract manual Pareto front from full dataset
  - Compare overlap between MCDA top candidates and MOO Pareto front
  - Output pareto_front.csv with ranked counties
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# ─────────────────────────────────────────────
# CONFIG: map objectives to your column names
# Adjust these if your columns are named differently
# ─────────────────────────────────────────────
COLUMN_MAP = {
    "seismic_risk":     "pga_max",
    "flood_risk":       "pct_sfha",
    "pop_density":      "population_density",
    "water_access":     "avg_discharge",
    "grid_connectivity":"distance_to_lines_km",
    "energy_demand":    "total_energy_consumption_mwh",
}

# Columns that are "higher is better" — we negate them so pymoo minimizes
MAXIMIZE_COLS = {"water_access", "energy_demand"}

# Optional: column with MCDA suitability score from Step 2
SUITABILITY_COL = None  # set to None if not present

# Input / output paths
CANDIDATES_PATH = "processed_data/candidates.csv"
OUTPUT_PATH     = "pareto_front.csv"
PLOT_PATH       = "pareto_scatter.png"

# NSGA-II hyperparams
POP_SIZE   = 200
N_GEN      = 300
SEED       = 42


# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
df = pd.read_csv(CANDIDATES_PATH)
print(f"Loaded {len(df)} candidate counties")

# Validate columns exist
missing = [v for v in COLUMN_MAP.values() if v not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns in candidates.csv: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

# Build objective matrix (all minimization after negation)
obj_names = list(COLUMN_MAP.keys())
obj_cols  = [COLUMN_MAP[k] for k in obj_names]

X_raw = df[obj_cols].copy().values.astype(float)

# Normalize each objective to [0, 1] before optimization
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)
X_norm = (X_raw - X_min) / (X_max - X_min + 1e-9)

# Negate maximize objectives
for i, name in enumerate(obj_names):
    if name in MAXIMIZE_COLS:
        X_norm[:, i] = 1 - X_norm[:, i]  # flip: higher original → lower cost

n_counties, n_obj = X_norm.shape
print(f"Objectives: {obj_names}")
print(f"Shape: {X_norm.shape}")


# ─────────────────────────────────────────────
# 2. Define pymoo Problem
#    We treat each county index as a "solution"
#    This is a selection problem: we want the non-dominated counties
# ─────────────────────────────────────────────
def is_pareto_efficient(costs):
    """
    Find Pareto-efficient rows in a cost matrix (all minimize).
    Returns boolean mask.
    """
    n = len(costs)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if is_efficient[i]:
            # dominated if any other point is <= on all objectives and < on at least one
            dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
            dominated[i] = False
            is_efficient[is_efficient] = ~dominated[is_efficient]
    return is_efficient


class NuclearSitingProblem(Problem):
    """
    Wraps pre-computed normalized objective matrix.
    Decision variable = county index (integer), but we treat
    each county as a fixed point and find the Pareto front analytically.
    """
    def __init__(self, X_norm):
        self.X_obj = X_norm
        super().__init__(
            n_var=1,
            n_obj=X_norm.shape[1],
            n_ieq_constr=0,
            xl=0,
            xu=len(X_norm) - 1,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        idx = np.clip(x[:, 0].astype(int), 0, len(self.X_obj) - 1)
        out["F"] = self.X_obj[idx]


# ─────────────────────────────────────────────
# 3. Analytical Pareto front (faster & exact)
#    NSGA-II is also run for robustness scoring
# ─────────────────────────────────────────────
print("\nComputing analytical Pareto front...")
pareto_mask = is_pareto_efficient(X_norm)
pareto_df = df[pareto_mask].copy()
print(f"  {pareto_mask.sum()} Pareto-optimal counties found")


# ─────────────────────────────────────────────
# 4. NSGA-II run (validates / extends Pareto front)
# ─────────────────────────────────────────────
print("\nRunning NSGA-II...")
problem = NuclearSitingProblem(X_norm)
algorithm = NSGA2(
    pop_size=POP_SIZE,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True,
)
termination = get_termination("n_gen", N_GEN)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=SEED,
    verbose=False,
)

# Map NSGA-II results back to county indices
nsga2_indices = np.clip(res.X[:, 0].astype(int), 0, n_counties - 1)
nsga2_indices = np.unique(nsga2_indices)
nsga2_pareto_df = df.iloc[nsga2_indices].copy()
print(f"  NSGA-II Pareto front: {len(nsga2_pareto_df)} counties")


# ─────────────────────────────────────────────
# 5. Robustness scoring
#    How many of 6 objectives does each county "win" in pairwise comparison?
#    Also: what % of random objective weight combos does it appear in top 50?
# ─────────────────────────────────────────────
print("\nComputing robustness scores...")

N_WEIGHT_SAMPLES = 500
robustness = np.zeros(n_counties)

rng = np.random.default_rng(SEED)
for _ in range(N_WEIGHT_SAMPLES):
    w = rng.dirichlet(np.ones(n_obj))          # random weight vector summing to 1
    scores = X_norm @ w                         # weighted sum (all minimization)
    top50_idx = np.argsort(scores)[:50]
    robustness[top50_idx] += 1

robustness_pct = robustness / N_WEIGHT_SAMPLES  # fraction of weight combos where county is top-50


# ─────────────────────────────────────────────
# 6. Assemble output dataframe
# ─────────────────────────────────────────────
result_df = df.copy()
result_df["robustness_score"] = robustness_pct
result_df["on_analytical_pareto"] = pareto_mask
result_df["on_nsga2_pareto"] = result_df.index.isin(nsga2_indices)
result_df["on_both_fronts"] = result_df["on_analytical_pareto"] & result_df["on_nsga2_pareto"]

# Add normalized objective columns for reference
for i, name in enumerate(obj_names):
    result_df[f"norm_{name}"] = X_norm[:, i]

# Sort by robustness score
result_df = result_df.sort_values("robustness_score", ascending=False)

result_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {OUTPUT_PATH}")


# ─────────────────────────────────────────────
# 7. Overlap analysis (per the doc)
# ─────────────────────────────────────────────
if SUITABILITY_COL and SUITABILITY_COL in df.columns:
    top_mcda = df.nlargest(200, SUITABILITY_COL).index
    overlap = result_df[
        result_df["on_analytical_pareto"] & result_df.index.isin(top_mcda)
    ]
    print(f"\nOverlap Analysis:")
    print(f"  Top-200 MCDA counties:         {len(top_mcda)}")
    print(f"  Analytical Pareto front:        {pareto_mask.sum()}")
    print(f"  Overlap (both):                 {len(overlap)}")
    pct = len(overlap) / pareto_mask.sum() * 100
    print(f"  % of Pareto front in MCDA top: {pct:.1f}%")
    if pct >= 60:
        print("  → High overlap: MCDA weights are well-calibrated")
    elif pct >= 30:
        print("  → Moderate overlap: some good sites missed by MCDA weights")
    else:
        print("  → Low overlap: MCDA weights may need revisiting")

    # Counties on Pareto front but NOT in MCDA top 200 = missed by weights
    missed = result_df[
        result_df["on_analytical_pareto"] & ~result_df.index.isin(top_mcda)
    ]
    print(f"  Counties on Pareto front but missed by MCDA: {len(missed)}")


# ─────────────────────────────────────────────
# 8. Visualization: 2D scatter of two objectives, colored by robustness
# ─────────────────────────────────────────────
print("\nGenerating plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Nuclear Siting — Pareto Front Analysis", fontsize=14, fontweight="bold")

# Plot 1: Seismic vs Flood risk, colored by robustness
ax = axes[0]
sc = ax.scatter(
    X_norm[:, obj_names.index("seismic_risk")],
    X_norm[:, obj_names.index("flood_risk")],
    c=robustness_pct,
    cmap="YlOrRd_r",
    s=15,
    alpha=0.6,
    label="All candidates"
)
pareto_x = X_norm[pareto_mask, obj_names.index("seismic_risk")]
pareto_y = X_norm[pareto_mask, obj_names.index("flood_risk")]
ax.scatter(pareto_x, pareto_y, s=40, facecolors="none", edgecolors="blue",
           linewidths=1.2, label="Pareto front", zorder=5)
plt.colorbar(sc, ax=ax, label="Robustness score")
ax.set_xlabel("Seismic Risk (norm)")
ax.set_ylabel("Flood Risk (norm)")
ax.set_title("Seismic vs Flood Risk")
ax.legend(fontsize=8)

# Plot 2: Pop density vs Energy demand
ax = axes[1]
sc2 = ax.scatter(
    X_norm[:, obj_names.index("pop_density")],
    X_norm[:, obj_names.index("energy_demand")],
    c=robustness_pct,
    cmap="YlOrRd_r",
    s=15,
    alpha=0.6,
)
pareto_x2 = X_norm[pareto_mask, obj_names.index("pop_density")]
pareto_y2 = X_norm[pareto_mask, obj_names.index("energy_demand")]
ax.scatter(pareto_x2, pareto_y2, s=40, facecolors="none", edgecolors="blue",
           linewidths=1.2, label="Pareto front", zorder=5)
plt.colorbar(sc2, ax=ax, label="Robustness score")
ax.set_xlabel("Population Density (norm, lower=better)")
ax.set_ylabel("Energy Demand cost (norm, lower=more demand)")
ax.set_title("Pop Density vs Energy Demand")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved {PLOT_PATH}")
plt.show()

# ─────────────────────────────────────────────
# 9. Print top 20 counties
# ─────────────────────────────────────────────
id_cols = [c for c in ["county_name", "state", "GEOID", "NAME", "county_fips"] if c in df.columns]
display_cols = id_cols + ["robustness_score", "on_analytical_pareto", "on_nsga2_pareto"]
if SUITABILITY_COL and SUITABILITY_COL in df.columns:
    display_cols.append(SUITABILITY_COL)

print("\nTop 20 counties by robustness score:")
print(result_df[display_cols].head(20).to_string(index=False))
