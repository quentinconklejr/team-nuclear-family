"""
with_vs_without_plants_colored.py

Styled version of the 4-panel pairwise objective scatterplot from eda.ipynb,
comparing counties with and without existing nuclear plants.
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
df = pd.read_csv(BASE / 'processed_data' / 'final_dataset.csv')

has_plant = df[df['plant_count'] > 0]
no_plant  = df[df['plant_count'] == 0]

BG         = '#F5F0F7'
NO_PLANT   = '#B0AABC'   # medium grey-purple
HAS_PLANT  = '#E85A9E'   # bright pink

scatter_vars = [
    ('avg_discharge',            'distance_to_rivers_km',         'Lake Discharge vs. River Proximity'),
    ('transmission_lines_count', 'max_voltage',                    'Transmission Line Count vs. Max Voltage'),
    ('pga_max',                  'pct_sfha',                       'Seismic vs. Flood Risk'),
    ('population',               'total_energy_consumption_mwh',   'Population vs. Energy Consumption'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

for ax, (x, y, title) in zip(axes, scatter_vars):
    ax.set_facecolor(BG)

    ax.scatter(no_plant[x],  no_plant[y],
               color=NO_PLANT,  alpha=0.35, s=18, linewidths=0,
               label='no plant', rasterized=True)
    ax.scatter(has_plant[x], has_plant[y],
               color=HAS_PLANT, alpha=0.85, s=32, linewidths=0,
               label='has plant', zorder=3)

    ax.set_xlabel(x, fontsize=10, color='#4A3A5A')
    ax.set_ylabel(y, fontsize=10, color='#4A3A5A')
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
out = BASE / 'results' / 'with_vs_without_plants_colored.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'Saved → {out}')
