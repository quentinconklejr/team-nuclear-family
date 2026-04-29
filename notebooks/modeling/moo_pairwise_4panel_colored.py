"""
moo_pairwise_4panel_colored.py

4-panel curated Pareto objective scatterplot in project theme colors.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

BASE = Path(__file__).parent.parent.parent

COLORS = {
    'none':   '#D0C2E0',
    'tier 3': '#87C4E8',
    'tier 2': '#E85A9E',
    'tier 1': '#3D2B52',
}
SIZES  = {'none': 8,  'tier 3': 16, 'tier 2': 24, 'tier 1': 30}
ALPHAS = {'none': 0.25, 'tier 3': 0.6, 'tier 2': 0.9, 'tier 1': 0.92}

TIER_LABELS = {
    'none':   'none',
    'tier 3': 'tier 3',
    'tier 2': 'tier 2',
    'tier 1': 'tier 1',
}

BG = '#F5F0F7'

# ── data + MOO ────────────────────────────────────────────────────────────────
df = pd.read_csv(BASE / 'processed_data' / 'candidates_ranked.csv')

objectives    = ['pga_max', 'pct_sfha', 'population_density',
                 'dist_to_lakes_km', 'distance_to_lines_km',
                 'total_energy_consumption_mwh']
higher_better = ['total_energy_consumption_mwh']
idx_hb        = [objectives.index(f) for f in higher_better]

df_clean = df.dropna(subset=objectives).reset_index(drop=True)

model        = NonDominatedSorting()
matrix_full  = df_clean[objectives].values.astype(float)
matrix_full[:, idx_hb] = -matrix_full[:, idx_hb]
pareto_full_df = df_clean.index.values[model.do(matrix_full, only_non_dominated_front=True)]

n_top      = int(len(df_clean) / 10)
idx_top    = df_clean.nlargest(n_top, 'mcda_score').index.values
matrix_top = df_clean.loc[idx_top, objectives].values.astype(float)
matrix_top[:, idx_hb] = -matrix_top[:, idx_hb]
pareto_top_df = idx_top[model.do(matrix_top, only_non_dominated_front=True)]

df_clean['in_pareto_full'] = df_clean.index.isin(pareto_full_df)
df_clean['in_pareto_top']  = df_clean.index.isin(pareto_top_df)

def assign_tier(row):
    if row['in_pareto_full'] and row['in_pareto_top']:        return 'tier 1'
    if (not row['in_pareto_full']) and row['in_pareto_top']:  return 'tier 2'
    if row['in_pareto_full'] and (not row['in_pareto_top']):  return 'tier 3'
    return 'none'

df_clean['tier'] = df_clean.apply(assign_tier, axis=1)

# ── 4 curated pairs ───────────────────────────────────────────────────────────
pairs = [
    ('pga_max',            'dist_to_lakes_km',             'PGA Max vs. Dist. to Lakes'),
    ('pga_max',            'total_energy_consumption_mwh', 'PGA Max vs. Energy Consumption'),
    ('population_density', 'distance_to_lines_km',         'Pop. Density vs. Dist. to Lines'),
    ('pct_sfha',           'total_energy_consumption_mwh', 'Pct SFHA vs. Energy Consumption'),
]

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

for ax, (ox, oy, title) in zip(axes, pairs):
    ax.set_facecolor(BG)
    for tier in ['none', 'tier 3', 'tier 2', 'tier 1']:
        sub = df_clean[df_clean['tier'] == tier]
        ax.scatter(
            sub[ox], sub[oy],
            c=COLORS[tier], s=SIZES[tier], alpha=ALPHAS[tier],
            linewidths=0, label=TIER_LABELS[tier],
            rasterized=True,
        )
    ax.set_xlabel(ox, fontsize=10, color='#4A3A5A')
    ax.set_ylabel(oy, fontsize=10, color='#4A3A5A')
    ax.set_title(title, fontsize=12, fontweight='bold', color='#3D2B52', pad=8)
    ax.tick_params(labelsize=8.5, colors='#7B6A8A')
    for spine in ax.spines.values():
        spine.set_edgecolor('#D0C2E0')

axes[0].legend(fontsize=10, frameon=True,
               facecolor='white', edgecolor='#C8B8D8',
               labelcolor='#2A1A3A', markerscale=1.4)

fig.suptitle('PAIRWISE OBJECTIVE VIEWS',
             fontsize=15, fontweight='bold', color='#3D2B52', y=1.01)

plt.tight_layout(pad=1.8)
out = BASE / 'results' / 'pairwise_objective_views_colored.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved → {out}')
