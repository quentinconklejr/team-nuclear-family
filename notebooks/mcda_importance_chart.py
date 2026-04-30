"""
mcda_importance_chart.py

Horizontal bar chart of MCDA feature weights, styled to match project palette.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

BASE = '/Users/ninaschreiber/projects/active/team-nuclear-family'

# ── data ──────────────────────────────────────────────────────────────────────
weights = {
    'pga_max':                      0.2586,
    'pct_sfha':                     0.1753,
    'population_density':           0.1336,
    'dist_to_lakes_km':             0.1058,
    'avg_vol':                      0.0850,
    'total_lake_area':              0.0683,
    'distance_to_lines_km':         0.0544,
    'max_voltage':                  0.0425,
    'total_energy_consumption_mwh': 0.0321,
    'data_centers_count':           0.0229,
    'pct_military':                 0.0145,
    'pct_protected':                0.0069,
}

LABELS = {
    'pga_max':                      'Peak Ground Acceleration',
    'pct_sfha':                     'Severe Flood Hazard Area (%)',
    'population_density':           'Population Density',
    'dist_to_lakes_km':             'Distance to Nearest Lake',
    'avg_vol':                      'Average Lake Volume',
    'total_lake_area':              'Total Lake Area',
    'distance_to_lines_km':         'Distance to Transmission Lines',
    'max_voltage':                  'Max. Transmission Voltage',
    'total_energy_consumption_mwh': 'Total Energy Consumption',
    'data_centers_count':           'Data Centers Count',
    'pct_military':                 'Military Area (%)',
    'pct_protected':                'Protected Area (%)',
}

# sorted lowest to highest so highest sits at top of chart
features = sorted(weights, key=weights.get)
values   = [weights[f] for f in features]
labels   = [LABELS[f]  for f in features]
n        = len(features)

# ── palette: plum → hot pink → lavender → sky blue ───────────────────────────
PALETTE = [
    '#3D2B52',   # plum
    '#7B5EA7',   # mid purple
    '#B8A4D4',   # lavender
    '#E85A9E',   # hot pink
    '#87C4E8',   # sky blue
]

cmap = mcolors.LinearSegmentedColormap.from_list('mcda', PALETTE[::-1], N=256)
bar_colors = [cmap(i / (n - 1)) for i in range(n)]

# ── figure ────────────────────────────────────────────────────────────────────
BG = '#F5F0F7'
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

bars = ax.barh(range(n), values, color=bar_colors, height=0.68, zorder=2)

# value labels at end of each bar
for i, (bar, v) in enumerate(zip(bars, values)):
    ax.text(
        v + 0.003, i,
        f'{v:.1%}',
        va='center', ha='left',
        fontsize=10, color='#2A1A3A', fontweight='bold'
    )

ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=11.5, color='#2A1A3A')

ax.set_xlim(0, max(values) * 1.22)
ax.set_xlabel('Weight', fontsize=12, color='#4A3A5A', labelpad=8)

ax.set_title('MCDA FEATURES IMPORTANCE RANKING',
             fontsize=15, fontweight='bold', color='#3D2B52',
             pad=18, loc='left')

# clean up spines
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#C8B8D8')
ax.tick_params(axis='x', colors='#7B6A8A', labelsize=10)
ax.tick_params(axis='y', length=0)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

ax.set_axisbelow(True)
ax.xaxis.grid(True, color='#E0D4EC', linewidth=0.6, zorder=0)

plt.tight_layout(pad=1.5)
out = f'{BASE}/results/mcda_importance_chart.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved → {out}')
