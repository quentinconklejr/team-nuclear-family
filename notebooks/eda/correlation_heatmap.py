"""
correlation_heatmap.py

Correlation heatmap styled in project theme colors.
Sky blue (negative) -> blush (zero) -> hot pink -> plum (positive).
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
df = pd.read_csv(BASE / 'processed_data' / 'final_dataset.csv')

# numeric only, drop non-informative cols
drop = {'Unnamed: 0', 'geo_id', 'county_name'}
num_cols = [c for c in df.select_dtypes(include='number').columns if c not in drop]

SHORT = {
    'population':                   'population',
    'median_household_income':      'median HH income',
    'housing_units':                'housing units',
    'total_energy_consumption_mwh': 'energy consumption',
    'data_centers_count':           'data centers',
    'sfha_area':                    'SFHA area',
    'pct_sfha':                     'pct SFHA',
    'lake_count':                   'lake count',
    'total_lake_area':              'lake area',
    'avg_vol':                      'avg lake vol',
    'avg_depth':                    'avg lake depth',
    'avg_discharge':                'avg discharge',
    'dist_to_lakes_km':             'dist to lakes',
    'wetland_count':                'wetland count',
    'distance_to_rivers_km':        'dist to rivers',
    'rivers_count':                 'river count',
    'total_rivers_mile':            'river miles',
    'military_count':               'military count',
    'total_military_area_m':        'military area',
    'pct_military':                 'pct military',
    'plant_count':                  'plant count',
    'pga_max':                      'pga max',
    'distance_to_lines_km':         'dist to lines',
    'transmission_lines_count':     'line count',
    'max_voltage':                  'max voltage',
    'average_voltage':              'avg voltage',
    'protected_count':              'protected count',
    'total_protected_area_m':       'protected area',
    'pct_protected':                'pct protected',
    'county_area_km2':              'county area',
    'population_density':           'pop density',
}

corr = df[num_cols].corr()
labels = [SHORT.get(c, c) for c in num_cols]

# ── custom diverging colormap: sky blue → blush → hot pink → plum ─────────────
CMAP = mcolors.LinearSegmentedColormap.from_list(
    'theme_div',
    ['#87C4E8', '#C8B8D8', '#F5F0F7', '#E85A9E', '#3D2B52'],
    N=256
)

BG = '#F5F0F7'
n  = len(num_cols)
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

im = ax.imshow(corr.values, cmap=CMAP, vmin=-1, vmax=1, aspect='auto')

# tick labels
ax.set_xticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8.5, color='#2A1A3A')
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=8.5, color='#2A1A3A')

# colour bar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.ax.tick_params(labelsize=9, colors='#4A3A5A')
cbar.outline.set_edgecolor('#C8B8D8')
cbar.ax.set_facecolor(BG)

# grid lines between cells
for i in range(n + 1):
    ax.axhline(i - 0.5, color='white', linewidth=0.4)
    ax.axvline(i - 0.5, color='white', linewidth=0.4)

ax.set_title('CORRELATIONS HEATMAP', fontsize=15, fontweight='bold',
             color='#3D2B52', pad=16, loc='left')

plt.tight_layout(pad=1.5)
out = BASE / 'results' / 'correlation_heatmap.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved → {out}')
