"""
ranked_sites_map_top5.py

Same 5-tier ranked-sites map as ranked_sites_map.py, but with periwinkle-blue
stars and annotated labels for the top-5 highest-scoring counties.
"""

import matplotlib
matplotlib.use('Agg')

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

BASE = '/Users/ninaschreiber/projects/active/team-nuclear-family'

# ── palette: 5 tiers, low → high ─────────────────────────────────────────────
TIER_COLORS = {
    1: '#C8B8D8',
    2: '#E0809C',
    3: '#E85A9E',
    4: '#8B3A7A',
    5: '#3D2B52',
}
LAND       = '#F2EEF8'
BORDER     = '#D8CEEA'
STATE_LINE = '#B8A4D4'

FIPS_TO_ABBR = {
    '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT',
    '10':'DE','11':'DC','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL',
    '18':'IN','19':'IA','20':'KS','21':'KY','22':'LA','23':'ME','24':'MD',
    '25':'MA','26':'MI','27':'MN','28':'MS','29':'MO','30':'MT','31':'NE',
    '32':'NV','33':'NH','34':'NJ','35':'NM','36':'NY','37':'NC','38':'ND',
    '39':'OH','40':'OK','41':'OR','42':'PA','44':'RI','45':'SC','46':'SD',
    '47':'TN','48':'TX','49':'UT','50':'VT','51':'VA','53':'WA','54':'WV',
    '55':'WI','56':'WY',
}

STAR_COLOR = '#7B8FCC'        # periwinkle blue
STAR_EDGE  = '#4A5A99'        # slightly deeper edge for definition
ARROW_COLOR = '#4A5A99'

# ── label offsets in EPSG:5070 metres (one per top-5 slot) ───────────────────
LABEL_OFFSETS = [
    (  450_000,  380_000),   # slot 1 — upper-right
    ( -450_000,  380_000),   # slot 2 — upper-left
    (  500_000,  -80_000),   # slot 3 — right
    ( -500_000,  150_000),   # slot 4 — left
    (  150_000, -430_000),   # slot 5 — below
]

# ── load data ─────────────────────────────────────────────────────────────────
counties = gpd.read_file(f'{BASE}/processed_data/county_boundaries.geojson')
ranked   = pd.read_csv(f'{BASE}/processed_data/candidates_ranked.csv')

counties['geoid']   = counties['geoid'].astype(str).str.zfill(5)
ranked['geo_id']    = ranked['geo_id'].astype(str).str.zfill(5)

# ── tiers ─────────────────────────────────────────────────────────────────────
ranked = ranked.sort_values('mcda_score', ascending=True).reset_index(drop=True)
ranked['tier']  = pd.qcut(ranked['mcda_score'], q=5, labels=False) + 1
ranked['color'] = ranked['tier'].map(TIER_COLORS)

# ── merge ─────────────────────────────────────────────────────────────────────
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

ranked_conus   = conus[conus['tier'].notna()].copy()
unranked_conus = conus[conus['tier'].isna()]

# ── top-5 sites ───────────────────────────────────────────────────────────────
top5 = ranked_conus.nlargest(5, 'mcda_score').reset_index(drop=True)

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 14))
fig.patch.set_alpha(0)
ax.set_facecolor('none')

states.plot(ax=ax, color=LAND, alpha=1.0, zorder=0)
conus.boundary.plot(ax=ax, color=BORDER, linewidth=0.07, alpha=0.28, zorder=1)
states.boundary.plot(ax=ax, color=STATE_LINE, linewidth=0.55, alpha=0.72, zorder=2)

for tier in [1, 2, 3, 4, 5]:
    df = ranked_conus[ranked_conus['tier'] == tier]
    ax.scatter(
        df['cx'], df['cy'],
        c=TIER_COLORS[tier],
        s=16,
        alpha=0.95,
        linewidths=0,
        zorder=3 + tier,
    )

# ── periwinkle stars for top-5 ────────────────────────────────────────────────
ax.scatter(
    top5['cx'], top5['cy'],
    marker='*',
    s=420,
    color=STAR_COLOR,
    edgecolors=STAR_EDGE,
    linewidths=0.8,
    zorder=20,
)

# ── annotations ───────────────────────────────────────────────────────────────
for rank, (_, row) in enumerate(top5.iterrows(), start=1):
    dx, dy = LABEL_OFFSETS[rank - 1]
    tx = row['cx'] + dx
    ty = row['cy'] + dy

    st = FIPS_TO_ABBR.get(str(row['state_fips']).zfill(2), '')
    label = f'#{rank}  {row["county_name"]}, {st}'

    ax.annotate(
        label,
        xy=(row['cx'], row['cy']),
        xytext=(tx, ty),
        fontsize=13.5,
        fontweight='bold',
        color='#2A3A7A',
        arrowprops=dict(
            arrowstyle='-|>',
            color=ARROW_COLOR,
            lw=1.2,
            shrinkA=4,
            shrinkB=10,
        ),
        bbox=dict(
            boxstyle='round,pad=0.35',
            facecolor='white',
            edgecolor=STAR_EDGE,
            alpha=0.88,
            linewidth=0.9,
        ),
        ha='center',
        va='center',
        zorder=25,
    )

ax.set_axis_off()
plt.tight_layout(pad=0)

out = f'{BASE}/results/ranked_sites_map_top5.png'
plt.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
plt.close()
print(f'Saved → {out}')
for rank, (_, row) in enumerate(top5.iterrows(), start=1):
    st = FIPS_TO_ABBR.get(str(row['state_fips']).zfill(2), '')
    print(f'  #{rank}  {row["county_name"]}, {st}  (score={row["mcda_score"]:.4f})')
