"""
moo_pairwise_colored.py

Pairwise Pareto objective scatterplots in project theme colors.
Reproduces the logic from moo.ipynb without re-running the full notebook.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

BASE = Path(__file__).parent.parent.parent

# ── palette ───────────────────────────────────────────────────────────────────
COLORS = {
    'none':   '#D0C2E0',   # faded lavender
    'tier 3': '#87C4E8',   # sky blue  (global Pareto only)
    'tier 2': '#E85A9E',   # hot pink  (MCDA Pareto only)
    'tier 1': '#3D2B52',   # plum      (both fronts)
}
SIZES  = {'none': 6,  'tier 3': 14, 'tier 2': 22, 'tier 1': 28}
ALPHAS = {'none': 0.25, 'tier 3': 0.55, 'tier 2': 0.9, 'tier 1': 0.9}

TIER_LABELS = {
    'none':   'Unplaced',
    'tier 3': 'Tier 3 — global Pareto only',
    'tier 2': 'Tier 2 — MCDA Pareto only',
    'tier 1': 'Tier 1 — both fronts',
}

SHORT_AXIS = {
    'pga_max':                      'PGA Max',
    'pct_sfha':                     'Pct SFHA',
    'population_density':           'Pop. Density',
    'dist_to_lakes_km':             'Dist. to Lakes',
    'distance_to_lines_km':         'Dist. to Lines',
    'total_energy_consumption_mwh': 'Energy Consumption',
}

BG = '#F5F0F7'

# ── data + MOO (mirrors moo.ipynb) ───────────────────────────────────────────
df = pd.read_csv(BASE / 'processed_data' / 'candidates_ranked.csv')

objectives    = ['pga_max', 'pct_sfha', 'population_density',
                 'dist_to_lakes_km', 'distance_to_lines_km',
                 'total_energy_consumption_mwh']
higher_better = ['total_energy_consumption_mwh']
idx_hb        = [objectives.index(f) for f in higher_better]

df_clean = df.dropna(subset=objectives).reset_index(drop=True)

model = NonDominatedSorting()

matrix_full = df_clean[objectives].values.astype(float)
matrix_full[:, idx_hb] = -matrix_full[:, idx_hb]
pareto_full_idx = model.do(matrix_full, only_non_dominated_front=True)
pareto_full_df  = df_clean.index.values[pareto_full_idx]

n_top      = int(len(df_clean) / 10)
idx_top    = df_clean.nlargest(n_top, 'mcda_score').index.values
matrix_top = df_clean.loc[idx_top, objectives].values.astype(float)
matrix_top[:, idx_hb] = -matrix_top[:, idx_hb]
pareto_top_idx = model.do(matrix_top, only_non_dominated_front=True)
pareto_top_df  = idx_top[pareto_top_idx]

df_clean['in_pareto_full'] = df_clean.index.isin(pareto_full_df)
df_clean['in_pareto_top']  = df_clean.index.isin(pareto_top_df)

def assign_tier(row):
    if row['in_pareto_full'] and row['in_pareto_top']:   return 'tier 1'
    if (not row['in_pareto_full']) and row['in_pareto_top']: return 'tier 2'
    if row['in_pareto_full'] and (not row['in_pareto_top']): return 'tier 3'
    return 'none'

df_clean['tier'] = df_clean.apply(assign_tier, axis=1)

# ── figure ────────────────────────────────────────────────────────────────────
pairs = list(combinations(objectives, 2))   # 15 pairs
fig, axes = plt.subplots(4, 4, figsize=(22, 18))
fig.patch.set_facecolor(BG)

for ax, (ox, oy) in zip(axes.flat, pairs):
    ax.set_facecolor(BG)
    for tier in ['none', 'tier 3', 'tier 2', 'tier 1']:
        sub = df_clean[df_clean['tier'] == tier]
        ax.scatter(
            sub[ox], sub[oy],
            c=COLORS[tier], s=SIZES[tier], alpha=ALPHAS[tier],
            linewidths=0, label=TIER_LABELS[tier],
            rasterized=True,
        )
    ax.set_xlabel(SHORT_AXIS.get(ox, ox), fontsize=9, color='#4A3A5A')
    ax.set_ylabel(SHORT_AXIS.get(oy, oy), fontsize=9, color='#4A3A5A')
    ax.tick_params(labelsize=7.5, colors='#7B6A8A')
    for spine in ax.spines.values():
        spine.set_edgecolor('#D0C2E0')
    ax.set_facecolor(BG)

# hide the unused 16th panel but put the legend there
legend_ax = axes[3, 3]
legend_ax.set_facecolor(BG)
legend_ax.set_axis_off()
handles = [
    plt.scatter([], [], c=COLORS[t], s=60, alpha=0.9, label=TIER_LABELS[t])
    for t in ['tier 1', 'tier 2', 'tier 3', 'none']
]
legend_ax.legend(
    handles=handles,
    loc='center',
    fontsize=11,
    frameon=True,
    facecolor='white',
    edgecolor='#C8B8D8',
    labelcolor='#2A1A3A',
    title='Tier',
    title_fontsize=12,
)

fig.suptitle('PARETO OBJECTIVE PAIRWISE VIEWS',
             fontsize=17, fontweight='bold', color='#3D2B52', y=1.01)

plt.tight_layout(pad=1.2)
out = BASE / 'results' / 'pairwise_scatterplot_moo_colored.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved → {out}')
print(f'  Tier 1 (both): {(df_clean["tier"]=="tier 1").sum()}')
print(f'  Tier 2 (MCDA only): {(df_clean["tier"]=="tier 2").sum()}')
print(f'  Tier 3 (global only): {(df_clean["tier"]=="tier 3").sum()}')
