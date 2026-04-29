"""
ranked_sites_map.py

Aesthetic US county map showing ranked nuclear suitability.
5-tier color gradient from low (#C8B8D8) to high (#3D2B52).
White US land fill, transparent outside US boundaries.
No legend (user supplies separately).
"""

import matplotlib
matplotlib.use('Agg')

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

BASE = '/Users/ninaschreiber/projects/active/team-nuclear-family'

# ── palette: 5 tiers, low → high ─────────────────────────────────────────────
TIER_COLORS = {
    1: '#C8B8D8',  # low
    2: '#E0809C',
    3: '#E85A9E',
    4: '#8B3A7A',
    5: '#3D2B52',  # high
}
LAND       = '#F2EEF8'   # barely-there lavender — gives all 5 tiers contrast
BORDER     = '#D8CEEA'   # faint lilac county outlines
STATE_LINE = '#B8A4D4'   # soft lilac state lines

# ── load data ─────────────────────────────────────────────────────────────────
counties = gpd.read_file(f'{BASE}/processed_data/county_boundaries.geojson')
ranked   = pd.read_csv(f'{BASE}/processed_data/candidates_ranked.csv')

# normalise FIPS
counties['geoid'] = counties['geoid'].astype(str).str.zfill(5)
ranked['geo_id']  = ranked['geo_id'].astype(str).str.zfill(5)

# ── assign quintile tiers based on mcda_score (higher score = higher tier) ───
ranked = ranked.sort_values('mcda_score', ascending=True).reset_index(drop=True)
n = len(ranked)
ranked['tier'] = pd.qcut(ranked['mcda_score'], q=5, labels=False) + 1  # 0-based → 1-5
ranked['color'] = ranked['tier'].map(TIER_COLORS)

tier_counts = ranked['tier'].value_counts().sort_index()
for t, c in tier_counts.items():
    print(f'  Tier {t}: {c:,} counties  →  {TIER_COLORS[t]}')

# ── merge onto geometries ─────────────────────────────────────────────────────
counties = counties.merge(
    ranked[['geo_id', 'tier', 'color', 'mcda_score']],
    left_on='geoid', right_on='geo_id',
    how='left',
)

# ── CONUS only ────────────────────────────────────────────────────────────────
SKIP = {'02', '15', '60', '66', '69', '72', '78'}
conus = counties[~counties['state_fips'].isin(SKIP)].copy()

conus = conus.to_crs(epsg=5070)
rep = conus.geometry.representative_point()
conus['cx'] = rep.x
conus['cy'] = rep.y

states = conus.dissolve(by='state_fips')

# split into ranked vs unranked (excluded)
ranked_conus   = conus[conus['tier'].notna()].copy()
unranked_conus = conus[conus['tier'].isna()]

print(f'\nRanked CONUS counties: {len(ranked_conus):,}')
print(f'Unranked (excluded):   {len(unranked_conus):,}')

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 14))
fig.patch.set_alpha(0)
ax.set_facecolor('none')

# barely-lavender land fill — gives all tiers contrast without feeling dark
states.plot(ax=ax, color=LAND, alpha=1.0, zorder=0)

# ultra-light county mesh
conus.boundary.plot(ax=ax, color=BORDER, linewidth=0.07, alpha=0.28, zorder=1)

# state outlines
states.boundary.plot(ax=ax, color=STATE_LINE, linewidth=0.55, alpha=0.72, zorder=2)

# dots for each tier, low → high so high sits on top
for tier in [1, 2, 3, 4, 5]:
    df = ranked_conus[ranked_conus['tier'] == tier]
    ax.scatter(
        df['cx'], df['cy'],
        c=TIER_COLORS[tier],
        s=16,
        alpha=0.95,
        linewidths=0,
        zorder=3 + tier,  # tier 5 on top
    )

ax.set_axis_off()

plt.tight_layout(pad=0)
out = f'{BASE}/results/ranked_sites_map.png'
plt.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
plt.close()
print(f'\nSaved → {out}')
