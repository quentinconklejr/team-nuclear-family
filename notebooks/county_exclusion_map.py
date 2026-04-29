"""
county_exclusion_map.py

Aesthetic US county map: plum dots = kept candidates, pink dots = excluded.
White land fill, transparent outside US boundaries (for presentation slides).
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display needed

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

BASE = '/Users/ninaschreiber/projects/active/team-nuclear-family'

# ── palette ───────────────────────────────────────────────────────────────────
PLUM        = '#3D2B52'   # kept candidate counties
PINK        = '#EE85A9'   # excluded counties  (user specified "E85A9")
WHITE       = '#FFFFFF'   # land fill
BORDER      = '#E0D8EC'   # soft lilac county outlines
STATE_LINE  = '#C4B0D8'   # slightly stronger for state edges

# ── load data ─────────────────────────────────────────────────────────────────
counties = gpd.read_file(f'{BASE}/processed_data/county_boundaries.geojson')
cands    = pd.read_csv(f'{BASE}/processed_data/candidates.csv')

# align FIPS: geojson has zero-padded 5-char strings; csv has bare integers
counties['geoid'] = counties['geoid'].astype(str).str.zfill(5)
kept_ids = set(cands['geo_id'].astype(str).str.zfill(5))
counties['kept'] = counties['geoid'].isin(kept_ids)

# ── CONUS only ────────────────────────────────────────────────────────────────
SKIP = {'02', '15', '60', '66', '69', '72', '78'}
conus = counties[~counties['state_fips'].isin(SKIP)].copy()

# Albers Equal Area Conic — the standard CONUS projection
conus = conus.to_crs(epsg=5070)
rep = conus.geometry.representative_point()
conus['cx'] = rep.x
conus['cy'] = rep.y

states = conus.dissolve(by='state_fips')

kept_df     = conus[conus['kept']]
excluded_df = conus[~conus['kept']]

n_kept     = len(kept_df)
n_excluded = len(excluded_df)
print(f'Kept: {n_kept:,}   Excluded: {n_excluded:,}')

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 14))
fig.patch.set_alpha(0)     # transparent outside the US
ax.set_facecolor('none')   # transparent axes background

# white land fill — solid white inside US boundaries
states.plot(ax=ax, color=WHITE, alpha=1.0, zorder=0)

# ultra-light county mesh
conus.boundary.plot(ax=ax, color=BORDER, linewidth=0.07, alpha=0.30, zorder=1)

# state outlines — a touch more visible
states.boundary.plot(ax=ax, color=STATE_LINE, linewidth=0.55, alpha=0.75, zorder=2)

# ── dots ──────────────────────────────────────────────────────────────────────
# kept (plum) drawn first so pink excluded sit on top
ax.scatter(
    kept_df['cx'], kept_df['cy'],
    c=PLUM, s=11, alpha=0.78, linewidths=0,
    zorder=3,
)
# excluded (pink) on top
ax.scatter(
    excluded_df['cx'], excluded_df['cy'],
    c=PINK, s=11, alpha=0.80, linewidths=0,
    zorder=4,
)

ax.set_axis_off()

# ── legend ────────────────────────────────────────────────────────────────────
plum_patch = mpatches.Patch(
    facecolor=PLUM,
    label=f'Candidate  ·  {n_kept:,} counties',
)
pink_patch = mpatches.Patch(
    facecolor=PINK,
    label=f'Excluded   ·  {n_excluded:,} counties',
)
legend = ax.legend(
    handles=[plum_patch, pink_patch],
    loc='lower left',
    frameon=False,
    fontsize=14,
    labelcolor=PLUM,
    handlelength=1.1,
    handleheight=1.1,
    labelspacing=0.6,
)
for text in legend.get_texts():
    text.set_fontfamily('serif')
    text.set_fontstyle('italic')

# ── titles ────────────────────────────────────────────────────────────────────
ax.text(
    0.5, 0.975,
    'U.S. County Suitability Screening',
    transform=ax.transAxes,
    ha='center', va='top',
    fontsize=24, fontfamily='serif', fontstyle='italic',
    color=PLUM, alpha=0.92,
)
ax.text(
    0.5, 0.938,
    'screened by seismic risk · flood hazard · military presence · protected land',
    transform=ax.transAxes,
    ha='center', va='top',
    fontsize=11, fontfamily='serif', fontstyle='italic',
    color='#7B5FA0', alpha=0.85,
)

plt.tight_layout(pad=0)
out = f'{BASE}/results/county_exclusion_map.png'
plt.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
plt.close()
print(f'Saved → {out}')
