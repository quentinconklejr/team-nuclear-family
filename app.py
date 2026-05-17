import json
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Nuclear Reactor Siting Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── FIPS → state abbreviation (authoritative; GeoJSON state_abbr field is corrupted) ──
_FIPS_TO_STATE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "60": "AS", "66": "GU", "69": "MP",
    "72": "PR", "78": "VI",
}


# ── Counties excluded from map background display ────────────────────────────
# These are removed from the choropleth background layer only — scores are preserved.
# Criteria: TIGER water area >> land area AND boundary visually overlaps a Great Lake
# or the county is a remote island territory far off the continental map.
_DISPLAY_EXCLUDED_GEOIDS = {
    "26083",  # Keweenaw County, MI  — 91% Lake Superior water; peninsula extends deep into lake
    "25019",  # Nantucket County, MA — 85% water; offshore island, distorts Atlantic shelf
    "25007",  # Dukes County, MA     — 79% water; Martha's Vineyard + islands
    "15005",  # Kalawao County, HI   — 77% water; tiny Molokai peninsula island
}


# ── SMR scoring engine ───────────────────────────────────────────────────────
# Thresholds per NRC/IAEA guidance. Population density converted from /mi² to /km²
# (1 mi² = 2.590 km²): 200/mi²=77.2/km², 500/mi²=193.1/km², 1000/mi²=386.1/km²

def compute_smr_scores(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return df copy with 'smr_score' column (NaN = disqualified by mode rules)."""
    df = df.copy()
    pga = df["pga_max"].fillna(0.0)
    dens = df["population_density"].fillna(0.0)
    volt = df["max_voltage"].fillna(0.0)
    lake_d  = df["dist_to_lakes_km"].fillna(999.0)  if "dist_to_lakes_km"      in df.columns else pd.Series(999.0, index=df.index)
    river_d = df["distance_to_rivers_km"].fillna(999.0) if "distance_to_rivers_km" in df.columns else pd.Series(999.0, index=df.index)
    nearest = np.minimum(lake_d, river_d)

    # Seismic — identical across all modes
    s_seis = np.where(pga > 0.50, np.nan,
             np.where(pga > 0.30, 0.25,
             np.where(pga > 0.10, 0.65, 1.0)))

    # Population density
    if mode == "LWR":
        s_pop = np.where(dens > 193.1, np.nan,
                np.where(dens > 77.2,  0.35, 1.0))
    else:
        s_pop = np.where(dens > 386.1, np.nan,
                np.where(dens > 193.1, 0.25,
                np.where(dens > 77.2,  0.65, 1.0)))

    # Grid / transmission
    if mode == "LWR":
        s_tx = np.where(volt >= 345, 1.0,
               np.where(volt >= 230, 0.75,
               np.where(volt >= 138, 0.50, 0.25)))
    elif mode == "SMR - NuScale VOYGR":
        s_tx = np.where(volt >= 115, 1.0, 0.50)   # licensed for islanded operation
    else:  # SMR - General
        s_tx = np.where(volt >= 230, 1.0,
               np.where(volt >= 115, 0.65, 0.35))

    # Water proximity  (LWR base tiers: <5→1.0, <15→0.75, <30→0.50, else→0.25)
    if mode == "LWR":
        s_wat = np.where(nearest < 5,  1.0,
                np.where(nearest < 15, 0.75,
                np.where(nearest < 30, 0.50, 0.25)))
    elif mode == "SMR - NuScale VOYGR":        # 2-tier reduction (dry cooling)
        s_wat = np.where(nearest < 30, 1.0, 0.75)
    else:                                       # SMR - General, 1-tier reduction
        s_wat = np.where(nearest < 5,  1.0,
                np.where(nearest < 15, 1.0,
                np.where(nearest < 30, 0.75, 0.50)))

    disq = np.isnan(s_seis) | np.isnan(s_pop)
    df["smr_score"] = np.where(disq, np.nan, (s_seis + s_pop + s_tx + s_wat) / 4.0)
    return df


# ── Theme definitions ─────────────────────────────────────────────────────────

_THEMES = {
    "Light": {
        "map_style":      "carto-positron",
        "paper_bg":       "white",
        "font_color":     "#1a2e4a",
        "colorscale":     "YlGn",
        "county_fill":    "#d6dae0",
        "county_border":  "#b0b8c4",
        "cand_border":    "#4a5568",
        "pareto_border":  "#b45309",
        "coal_border":    "#7c3aed",
        "state_border":   "#3a4a5c",
    },
    "Dark": {
        "map_style":      "carto-darkmatter",
        "paper_bg":       "#0e1117",
        "font_color":     "#e2e8f0",
        "colorscale":     "YlGn",
        "county_fill":    "#2d3748",
        "county_border":  "#4a5568",
        "cand_border":    "#a0aec0",
        "pareto_border":  "#f6ad55",
        "coal_border":    "#a78bfa",
        "state_border":   "#a0aec0",
    },
    "High Contrast": {
        "map_style":      "carto-darkmatter",
        "paper_bg":       "#000000",
        "font_color":     "#ffffff",
        "colorscale":     [[0, "#003300"], [0.33, "#006600"], [0.66, "#00aa00"], [1, "#00ff44"]],
        "county_fill":    "#222222",
        "county_border":  "#555555",
        "cand_border":    "#ffffff",
        "pareto_border":  "#ffff00",
        "coal_border":    "#ff00ff",
        "state_border":   "#ffffff",
    },
}


# ── CSS builders ──────────────────────────────────────────────────────────────

def _build_css(theme: str, accessible: bool) -> str:
    fs_title  = "2.3rem"  if accessible else "1.9rem"
    fs_sub    = "1.1rem"  if accessible else "0.95rem"
    fs_hdr    = "1.2rem"  if accessible else "1.05rem"
    fs_lbl    = "1rem"    if accessible else "0.9rem"
    fs_metric = "1rem"    if accessible else "0.82rem"

    shared = f"""
.sr-only {{
    position: absolute; width: 1px; height: 1px; padding: 0;
    margin: -1px; overflow: hidden; clip: rect(0,0,0,0);
    white-space: nowrap; border: 0;
}}
.stSlider > label {{ font-weight: 600; font-size: {fs_lbl}; }}
div[data-testid="metric-container"] > div:first-child {{ font-size: {fs_metric}; }}
"""

    if theme == "Light":
        return f"""<style>
.main-title {{
    font-size: {fs_title}; font-weight: 700; color: #1a2e4a; margin-bottom: 0.15rem;
}}
.subtitle {{
    color: #4a5a6a; font-size: {fs_sub}; margin-bottom: 1.25rem; line-height: 1.5;
}}
.section-header {{
    font-size: {fs_hdr}; font-weight: 600; color: #1a2e4a;
    margin: 1rem 0 0.4rem 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.2rem;
}}
.pareto-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 0.75rem; font-weight: 700;
    background: #dcfce7; color: #14532d;
    vertical-align: middle; margin-left: 6px; border: 1px solid #14532d;
}}
.detail-metric {{ margin-bottom: 0.15rem; }}
[data-testid="stSidebar"] {{ background: #f1f5f9; }}
{shared}
</style>"""

    if theme == "Dark":
        return f"""<style>
/* ── Layout ── */
.stApp {{ background-color: #0e1117 !important; }}
.stApp > header {{ background-color: #0e1117 !important; }}
.block-container {{ background-color: #0e1117 !important; }}

/* ── Sidebar shell ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child {{
    background-color: #1a2035 !important;
}}

/* ── Sidebar text — labels, markdown, captions ── */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stText"],
section[data-testid="stSidebar"] [data-testid="stCaption"] p {{
    color: #e2e8f0 !important;
}}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {{
    color: #93c5fd !important;
}}
section[data-testid="stSidebar"] hr {{ border-color: #334155 !important; }}

/* ── Selectbox ── */
[data-baseweb="select"] > div {{
    background-color: #253050 !important;
    border-color: #4a5568 !important;
}}
[data-baseweb="select"] div[class*="ValueContainer"],
[data-baseweb="select"] div[class*="singleValue"],
[data-baseweb="select"] div[class*="placeholder"],
[data-baseweb="select"] input {{
    color: #e2e8f0 !important;
    background-color: transparent !important;
}}
[data-baseweb="select"] svg {{ fill: #94a3b8 !important; }}

/* Dropdown popup (renders outside sidebar DOM) */
[data-baseweb="popover"] > div {{
    background-color: #253050 !important;
    border-color: #4a5568 !important;
}}
[data-baseweb="menu"] li,
[data-baseweb="list"] li {{
    background-color: #253050 !important;
    color: #e2e8f0 !important;
}}
[data-baseweb="option"]:hover {{
    background-color: #334155 !important;
}}

/* ── Toggle ── */
[data-testid="stToggle"] label {{ color: #e2e8f0 !important; }}
[data-testid="stToggle"] p {{ color: #e2e8f0 !important; }}

/* ── Slider ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p {{ color: #e2e8f0 !important; }}

/* ── Metrics ── */
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {{ color: #e2e8f0 !important; }}

/* ── Alerts ── */
[data-testid="stAlert"],
div[class*="stAlert"] {{
    background-color: #1e2535 !important;
    border-color: #334155 !important;
}}
[data-testid="stAlert"] p,
div[class*="stAlert"] p {{ color: #e2e8f0 !important; }}
div[data-testid="stInfo"] {{ background-color: #0f2844 !important; border-color: #1e4976 !important; }}
div[data-testid="stSuccess"] {{ background-color: #0f2d1a !important; border-color: #166534 !important; }}
div[data-testid="stWarning"] {{ background-color: #2d2205 !important; border-color: #713f12 !important; }}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{ background-color: #1e2535 !important; }}

/* ── Our custom classes ── */
.main-title {{
    font-size: {fs_title}; font-weight: 700; color: #93c5fd; margin-bottom: 0.15rem;
}}
.subtitle {{
    color: #94a3b8; font-size: {fs_sub}; margin-bottom: 1.25rem; line-height: 1.5;
}}
.section-header {{
    font-size: {fs_hdr}; font-weight: 600; color: #93c5fd;
    margin: 1rem 0 0.4rem 0; border-bottom: 2px solid #334155; padding-bottom: 0.2rem;
}}
.pareto-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 0.75rem; font-weight: 700;
    background: #14532d; color: #86efac;
    vertical-align: middle; margin-left: 6px; border: 1px solid #86efac;
}}
.detail-metric {{ margin-bottom: 0.15rem; }}
{shared}
</style>"""

    # High Contrast
    return f"""<style>
/* ── Layout ── */
.stApp {{ background-color: #000000 !important; }}
.stApp > header {{ background-color: #000000 !important; }}
.block-container {{ background-color: #000000 !important; }}

/* ── Sidebar shell ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child {{
    background-color: #111111 !important;
}}

/* ── Sidebar text ── */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stText"],
section[data-testid="stSidebar"] [data-testid="stCaption"] p {{
    color: #ffffff !important;
}}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {{
    color: #ffff00 !important;
}}
section[data-testid="stSidebar"] hr {{ border-color: #555555 !important; }}

/* ── Selectbox ── */
[data-baseweb="select"] > div {{
    background-color: #222222 !important;
    border-color: #ffffff !important;
    border-width: 2px !important;
}}
[data-baseweb="select"] div[class*="ValueContainer"],
[data-baseweb="select"] div[class*="singleValue"],
[data-baseweb="select"] div[class*="placeholder"],
[data-baseweb="select"] input {{
    color: #ffffff !important;
    background-color: transparent !important;
}}
[data-baseweb="select"] svg {{ fill: #ffffff !important; }}

/* Dropdown popup */
[data-baseweb="popover"] > div {{
    background-color: #222222 !important;
    border: 2px solid #ffff00 !important;
}}
[data-baseweb="menu"] li,
[data-baseweb="list"] li {{
    background-color: #222222 !important;
    color: #ffffff !important;
}}
[data-baseweb="option"]:hover {{
    background-color: #333300 !important;
    color: #ffff00 !important;
}}

/* ── Toggle ── */
[data-testid="stToggle"] label,
[data-testid="stToggle"] p {{ color: #ffffff !important; }}

/* ── Slider ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p {{ color: #ffffff !important; }}

/* ── Metrics ── */
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {{ color: #ffffff !important; }}

/* ── Alerts ── */
[data-testid="stAlert"],
div[class*="stAlert"] {{
    background-color: #111111 !important;
    border: 2px solid #ffffff !important;
}}
[data-testid="stAlert"] p,
div[class*="stAlert"] p {{ color: #ffffff !important; }}
div[data-testid="stSuccess"] {{ border-color: #00ff00 !important; }}
div[data-testid="stWarning"] {{ border-color: #ffff00 !important; }}
div[data-testid="stInfo"] {{ border-color: #00ffff !important; }}

/* ── Focus rings for keyboard nav ── */
a:focus, button:focus, [tabindex]:focus,
[data-baseweb="select"]:focus-within {{
    outline: 3px solid #ffff00 !important;
    outline-offset: 2px !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{ background-color: #111111 !important; }}

/* ── Our custom classes ── */
.main-title {{
    font-size: {fs_title}; font-weight: 700; color: #ffff00 !important; margin-bottom: 0.15rem;
}}
.subtitle {{
    color: #e2e2e2 !important; font-size: {fs_sub}; margin-bottom: 1.25rem; line-height: 1.5;
}}
.section-header {{
    font-size: {fs_hdr}; font-weight: 600; color: #ffff00 !important;
    margin: 1rem 0 0.4rem 0; border-bottom: 2px solid #ffffff; padding-bottom: 0.2rem;
}}
.pareto-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 0;
    font-size: 0.75rem; font-weight: 700;
    background: #000000; color: #00ff00;
    vertical-align: middle; margin-left: 6px; border: 2px solid #00ff00;
}}
.detail-metric {{ margin-bottom: 0.15rem; }}
{shared}
</style>"""


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_candidates():
    df = pd.read_csv("processed_data/candidates_ranked.csv")
    df["geoid"] = df["geo_id"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_pareto():
    cols = [
        "geo_id", "on_nsga2_pareto", "on_both_fronts",
        "norm_seismic_risk", "norm_flood_risk", "norm_pop_density",
        "norm_water_access", "norm_grid_connectivity", "norm_energy_demand",
    ]
    pf = pd.read_csv("pareto_front.csv", usecols=cols)
    pf["geoid"] = pf["geo_id"].astype(str).str.zfill(5)
    return pf


@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    with urllib.request.urlopen(url) as r:
        return json.load(r)


@st.cache_data
def load_state_geojson():
    # Cartographic state polygons dissolved from Plotly counties — shoreline-clipped
    with open("processed_data/state_boundaries_cartographic.geojson") as f:
        return json.load(f)


@st.cache_data
def load_coal_counties():
    """EIA 860 2022 coal plant counties (operating + retired). Returns geoid → dict."""
    coal = pd.read_csv("processed_data/coal_counties.csv", dtype={"geoid": str})
    coal["geoid"] = coal["geoid"].str.zfill(5)
    return coal.set_index("geoid").to_dict(orient="index")



@st.cache_data
def build_geo_lookup(_geojson):
    result = {}
    for feat in _geojson["features"]:
        # Plotly GeoJSON uses feature.id (5-digit FIPS string), not properties.geoid
        fid = str(feat.get("id", ""))
        state_abbr = _FIPS_TO_STATE.get(fid[:2], "")
        p = feat["properties"]
        result[fid] = {
            "state_abbr": state_abbr,
            "county_name_full": p.get("NAME", "") + " " + p.get("LSAD", ""),
        }
    return result



candidates    = load_candidates()
pareto        = load_pareto()
geojson       = load_geojson()
geo_lookup    = build_geo_lookup(geojson)
state_geojson = load_state_geojson()
coal_lookup   = load_coal_counties()

# Plotly GeoJSON uses feature.id (not properties.geoid); exclude water-dominant counties
all_geoids = [
    str(f["id"]) for f in geojson["features"]
    if str(f["id"]) not in _DISPLAY_EXCLUDED_GEOIDS
]

df = candidates.merge(
    pareto[["geoid", "on_nsga2_pareto", "on_both_fronts"]],
    on="geoid",
    how="left",
)
df["on_nsga2_pareto"] = df["on_nsga2_pareto"].fillna(False).astype(bool)
df["on_both_fronts"]  = df["on_both_fronts"].fillna(False).astype(bool)
df["state"]           = df["geoid"].map(lambda g: geo_lookup.get(g, {}).get("state_abbr", ""))

# Coal-to-Nuclear: flag candidate counties that contain coal plant infrastructure
df["has_coal_plant"]  = df["geoid"].isin(coal_lookup).astype(bool)
df["coal_capacity_mw"] = df["geoid"].map(lambda g: coal_lookup.get(g, {}).get("coal_capacity_mw", 0.0))

# Geoids of ALL coal counties (for full-map overlay, not just candidates)
_all_coal_geoids = [g for g in coal_lookup if g in set(all_geoids)]

_pct_cols = [
    "pga_max", "pct_sfha", "population_density",
    "dist_to_lakes_km", "distance_to_rivers_km",
    "max_voltage", "total_energy_consumption_mwh",
]
for _col in _pct_cols:
    if _col in df.columns:
        df[f"{_col}_pct"] = df[_col].rank(pct=True)

_rank_max  = int(df["rank"].max())
_score_min = df["mcda_score"].min()
_score_max = df["mcda_score"].max()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Display")

    theme = st.selectbox(
        "Theme",
        list(_THEMES.keys()),
        index=0,
        help="Choose a color scheme. High Contrast meets WCAG AAA.",
    )

    accessible = st.toggle(
        "Enhanced Accessibility",
        value=False,
        help="Increases text sizes and adds visible map descriptions for low-vision users (WCAG 2.1 AA).",
    )

    st.divider()
    st.markdown("## Reactor Mode")

    reactor_mode = st.selectbox(
        "Select reactor type",
        ["LWR", "SMR - NuScale VOYGR", "SMR - General"],
        index=0,
        help=(
            "Changes scoring thresholds based on each reactor type's regulatory profile. "
            "LWR uses the standard NRC framework. SMR modes apply relaxed population, "
            "grid, and water thresholds reflecting smaller footprints and advanced safety features."
        ),
    )

    if reactor_mode != "LWR":
        st.info(
            "**Why SMR thresholds differ from LWR:**\n\n"
            "**Smaller exclusion zone:** SMRs have significantly smaller Emergency Planning Zones. "
            "NuScale VOYGR has an NRC-approved site-boundary EPZ of roughly 400m, versus the "
            "10-mile standard for LWRs (NRC approval ML22287A155). This allows siting in areas "
            "with higher population density.\n\n"
            "**Dry cooling capability:** NuScale and many SMR designs can operate without a "
            "large nearby water source, reducing water proximity requirements.\n\n"
            "**Smaller footprint:** SMRs require less land and infrastructure, making them "
            "viable in locations that would be impractical for a full-scale LWR."
        )

    st.divider()
    st.markdown("## Filters")
    st.caption("Adjust the sliders below to limit the map and table to counties meeting all three thresholds.")
    st.divider()

    st.markdown("**Earthquake Risk**")
    # SMR mode extends the allowable seismic range to 0.50g (vs 0.30g for LWR)
    _pga_max = 0.50 if reactor_mode != "LWR" else 0.30
    st.caption(
        f"Peak ground shaking intensity in g-force. "
        f"SMR designs tolerate up to 0.50 g; LWR sites are capped at 0.30 g by the NRC. "
        f"Slide left for a stricter cutoff."
        if reactor_mode != "LWR" else
        "Peak ground shaking intensity in g-force. "
        "The NRC caps reactor sites at 0.30 g. "
        "Slide left for a stricter cutoff."
    )
    pga_filter = st.slider(
        "Earthquake risk cutoff (g)",
        0.0, _pga_max, _pga_max, 0.01,
        format="%.2f g",
        key=f"pga_slider_{reactor_mode}",
        help=(
            "Peak Ground Acceleration (PGA) measures how hard the ground shakes in an earthquake. "
            "0.05 g is barely perceptible; 0.30 g is the NRC limit for LWR siting; "
            "SMR designs can tolerate up to 0.50 g before disqualification."
        ),
    )

    st.markdown("**Flood Risk**")
    st.caption(
        "Share of the county inside FEMA's 100-year flood zone. "
        "Reactor sites require stable, dry ground. "
        "Slide left to filter out flood-prone areas."
    )
    sfha_filter = st.slider(
        "Flood zone coverage cutoff (%)",
        0.0, 20.0, 20.0, 1.0,
        format="%.0f%%",
        help=(
            "Special Flood Hazard Area (SFHA): the percentage of the county within FEMA's "
            "100-year flood boundary. 0% means no flood exposure; 20% is highly flood-prone."
        ),
    )
    sfha_filter = sfha_filter / 100.0  # convert back to fraction for filtering

    st.markdown("**Population Density**")
    _pop_cap = 10000
    st.caption(
        "Residents per square kilometer. "
        "The NRC requires large, low-population exclusion zones around reactor sites. "
        "Rural counties are generally easier to permit."
    )
    pop_filter = st.slider(
        "Max residents per km²",
        0, _pop_cap, _pop_cap, 50,
        format="%d / km²",
        help=(
            "Population density of the county. "
            "Under 10/km² is very rural and ideal for siting. "
            "Above 100/km² is suburban or urban, making NRC exclusion zone requirements difficult to meet."
        ),
    )

    st.divider()
    if reactor_mode == "LWR":
        pareto_only = st.toggle(
            "Show only top-tier sites (Pareto-optimal)",
            value=False,
            help=(
                "Pareto-optimal counties are those where improving one criterion (e.g., lower seismic risk) "
                "would require giving up ground on another (e.g., water access). "
                "These 111 sites represent the best possible trade-offs across all 6 scoring factors."
            ),
        )
    else:
        pareto_only = False
        st.caption(
            "Pareto-optimal filtering is not available in SMR mode. "
            "The Pareto front was computed under LWR weights and does not apply to SMR scoring."
        )

    st.divider()
    show_coal = st.toggle(
        "Coal-to-Nuclear Opportunity layer",
        value=False,
        help=(
            "Highlights counties that contain retired or operating coal power plants "
            "(EIA Form 860, 2022). These sites already have transmission infrastructure, "
            "grid interconnections, and a workforce familiar with large power generation — "
            "making them stronger candidates for nuclear conversion. "
            "Enabling this layer adds a +0.05 bonus to the displayed score for coal counties "
            "and draws a purple outline on the map."
        ),
    )

    st.divider()
    st.markdown("**About**")
    st.caption(
        "Team Nuclear Family | IDSC Data Dive Spring 2026 \U0001f947  \n"
        "Scores use NRC-guided, safety-first weighting via the Rank Order Centroid method. "
        "18 of the top 20 counties held their ranking across all sensitivity tests."
    )

    st.divider()
    with st.expander("How We Compare to OR-SAGE"):
        st.markdown(
            "**OR-SAGE** (Oak Ridge Siting Analysis for Power Generation Expansion) "
            "is the NRC-sponsored benchmark for nuclear plant siting in the US. "
            "Here is how our approach differs:\n\n"
            "- **Spatial resolution.** OR-SAGE evaluates 100-meter grid cells across the "
            "continental US. We aggregate to the county level, which is coarser but aligns "
            "directly with the regulatory, census, and energy datasets that report at county "
            "boundaries.\n\n"
            "- **Reactor scope.** OR-SAGE was built for large Light Water Reactors. We extend "
            "the framework with dedicated SMR scoring modes — NuScale VOYGR and a general "
            "advanced reactor profile — each with thresholds tuned to their design basis.\n\n"
            "- **SMR-specific criteria.** OR-SAGE does not include an SMR scoring mode with "
            "NRC Regulatory Guide 4.7 Rev. 4 Appendix A criteria. Our SMR modes apply "
            "current NRC guidance for Emergency Planning Zone size, grid access, and water "
            "proximity for advanced reactor designs.\n\n"
            "- **Coal-to-Nuclear transition.** We explicitly reward counties with existing "
            "coal plant infrastructure as higher-opportunity sites for nuclear conversion, "
            "reflecting DOE and NRC transition planning guidance. OR-SAGE does not include "
            "this criterion."
        )


# ── Apply theme CSS ───────────────────────────────────────────────────────────

st.markdown(_build_css(theme, accessible), unsafe_allow_html=True)


# ── Apply filters ─────────────────────────────────────────────────────────────

mask = (
    (df["pga_max"] <= pga_filter)
    & (df["pct_sfha"] <= sfha_filter)
    & (df["population_density"].fillna(0) <= pop_filter)
)
if pareto_only:
    mask &= df["on_nsga2_pareto"]

filtered_df     = df[mask].copy()
filtered_geoids = set(filtered_df["geoid"])

# ── Reactor-mode scoring ──────────────────────────────────────────────────────
if reactor_mode != "LWR":
    filtered_df  = compute_smr_scores(filtered_df, reactor_mode)
    active_df    = filtered_df[filtered_df["smr_score"].notna()].copy()
    _score_col   = "smr_score"
    _score_label = "SMR Score"
else:
    active_df    = filtered_df
    _score_col   = "mcda_score"
    _score_label = "MCDA Score"

# ── Coal-to-Nuclear score bonus (+0.05 for counties with coal infrastructure) ──
_disp_col = _score_col   # column actually used for map coloring and ranking
if show_coal and len(active_df) > 0:
    active_df  = active_df.copy()
    _disp_col  = f"_disp_{_score_col}"
    active_df[_disp_col] = active_df[_score_col].copy()
    _coal_mask = active_df["has_coal_plant"].fillna(False)
    active_df.loc[_coal_mask, _disp_col] = (
        active_df.loc[_coal_mask, _score_col] + 0.05
    ).clip(upper=1.0)
    _score_label = _score_label + " (+coal bonus)"

_active_min = float(active_df[_disp_col].min()) if len(active_df) else 0.0
_active_max = float(active_df[_disp_col].max()) if len(active_df) else 1.0


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">Nuclear Reactor Siting Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    "Explore 2,161 US counties scored through our MCDA siting framework. "
    "Darker green means higher suitability. "
    "Amber outlines mark Pareto-optimal sites. "
    "Filter by safety thresholds in the sidebar, then click any county for a full breakdown."
    "</div>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Showing", f"{len(active_df):,}", f"of 2,161 candidates")
if reactor_mode == "LWR":
    m2.metric("Pareto-Optimal Shown", f"{int(active_df['on_nsga2_pareto'].sum()):,}", "NSGA-II front")
    m3.metric(
        "Best MCDA Score",
        f"{active_df['mcda_score'].max():.3f}" if len(active_df) else "—",
    )
    _best_rank = active_df["rank"].min()
    m4.metric("Highest Rank Shown", f"#{int(_best_rank)}" if pd.notna(_best_rank) else "—")
else:
    smr_mode_short = "NuScale VOYGR" if "NuScale" in reactor_mode else "General SMR"
    m2.metric("Mode", smr_mode_short)
    m3.metric(
        "Best SMR Score",
        f"{active_df['smr_score'].max():.3f}" if len(active_df) else "—",
    )
    m4.metric("Qualifying Counties", f"{len(active_df):,}", "passed SMR criteria")


# ── Map ───────────────────────────────────────────────────────────────────────

_map_title = {
    "LWR":                "County Suitability Map",
    "SMR - NuScale VOYGR": "County Suitability Map — SMR NuScale VOYGR Mode",
    "SMR - General":       "County Suitability Map — SMR General Mode",
}[reactor_mode]
st.markdown(f'<div class="section-header">{_map_title}</div>', unsafe_allow_html=True)

if reactor_mode == "SMR - NuScale VOYGR":
    st.info(
        "**NuScale VOYGR mode active.** Scoring reflects NRC-approved thresholds for the VOYGR design: "
        "site-boundary EPZ (~400m per ML22287A155), islanded grid operation viable at 115 kV+, "
        "and dry cooling support reduces water proximity requirements. "
        "Population disqualification raised to 1,000/mi². Seismic tolerance extended to 0.50 g."
    )
elif reactor_mode == "SMR - General":
    st.info(
        "**SMR General mode active** (covers designs including Xe-100 and KP-FHR). "
        "Scoring uses relaxed population thresholds (disqualification at 1,000/mi²), "
        "reduced grid requirements (230 kV for full score), "
        "and improved water proximity scoring. Seismic tolerance extended to 0.50 g."
    )

# Screen-reader description (hidden visually)
st.markdown(
    '<p class="sr-only" role="img">'
    "Map of the United States showing nuclear reactor siting scores by county. "
    "Darker green indicates higher suitability. State borders are shown as distinct lines. "
    "Pareto-optimal counties have an amber outline. "
    "Click a county to see its full profile. "
    "The Top 20 Candidates table below provides the same data in a keyboard-accessible format."
    "</p>",
    unsafe_allow_html=True,
)

if accessible:
    st.caption(
        "Counties are shaded green by suitability score (darker = higher). "
        "State lines are shown as distinct borders. "
        "Pareto-optimal counties have an amber outline and a star in the tooltip. "
        "Click a county or use the table below."
    )

tc = _THEMES[theme]
fig = go.Figure()

# Layer 1: all counties — neutral background (opacity<1 lets base tiles show through,
# making water bodies like Lake Michigan visible underneath)
fig.add_trace(go.Choroplethmap(
    geojson=geojson,
    locations=all_geoids,
    z=[0.0] * len(all_geoids),
    featureidkey="id",
    colorscale=[[0, tc["county_fill"]], [1, tc["county_fill"]]],
    showscale=False,
    marker=dict(opacity=0.20, line=dict(width=0.2, color=tc["county_border"])),
    hoverinfo="skip",
    name="",
))

# Layer 2: qualifying candidates — colored by MCDA or SMR score
if len(active_df) > 0:
    if reactor_mode == "LWR":
        _rank_str  = active_df["rank"].apply(lambda x: f"#{int(x)}" if pd.notna(x) else "unranked")
        _score_str = active_df["mcda_score"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        _pareto_str = active_df["on_nsga2_pareto"].map({True: "<br><b>★ Pareto-Optimal</b>", False: ""})
        _coal_hover = active_df.apply(
            lambda r: f"<br>Coal capacity: {r['coal_capacity_mw']:.0f} MW" if r.get("has_coal_plant") else "",
            axis=1,
        )
        hover_text = (
            "<b>" + active_df["county_name"] + ", " + active_df["state"].fillna("") + "</b><br>"
            + "Rank: " + _rank_str + "<br>"
            + "MCDA Score: " + _score_str + "<br>"
            + "Seismic: " + active_df["pga_max"].map("{:.3f}".format) + " g<br>"
            + "Flood: " + (active_df["pct_sfha"] * 100).map("{:.1f}".format) + "% SFHA<br>"
            + "Pop Density: " + active_df["population_density"].map("{:.1f}".format) + " / km²"
            + _pareto_str
            + _coal_hover
        )
    else:
        _score_str = active_df["smr_score"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        _coal_hover = active_df.apply(
            lambda r: f"<br>Coal capacity: {r['coal_capacity_mw']:.0f} MW" if r.get("has_coal_plant") else "",
            axis=1,
        )
        hover_text = (
            "<b>" + active_df["county_name"] + ", " + active_df["state"].fillna("") + "</b><br>"
            + "SMR Score: " + _score_str + "<br>"
            + "Seismic: " + active_df["pga_max"].map("{:.3f}".format) + " g<br>"
            + "Flood: " + (active_df["pct_sfha"] * 100).map("{:.1f}".format) + "% SFHA<br>"
            + "Pop Density: " + active_df["population_density"].map("{:.1f}".format) + " / km²<br>"
            + "Grid: " + active_df["max_voltage"].fillna(0).map("{:.0f}".format) + " kV"
            + _coal_hover
        )

    fig.add_trace(go.Choroplethmap(
        geojson=geojson,
        locations=active_df["geoid"].tolist(),
        z=active_df[_disp_col].tolist(),
        featureidkey="id",
        colorscale=tc["colorscale"],
        zmin=_active_min,
        zmax=_active_max,
        colorbar=dict(
            title=dict(text=_score_label, font=dict(size=12, color=tc["font_color"])),
            thickness=14,
            len=0.55,
            x=1.01,
            tickformat=".2f",
            tickfont=dict(color=tc["font_color"]),
        ),
        marker=dict(opacity=0.82, line=dict(width=0.2, color=tc["county_border"])),
        text=hover_text.tolist(),
        hovertemplate="%{text}<extra></extra>",
        name="Candidates",
    ))

    # Layer 3: Pareto outline — LWR mode only
    if reactor_mode == "LWR":
        pareto_geoids = active_df.loc[active_df["on_nsga2_pareto"], "geoid"].tolist()
        if pareto_geoids:
            fig.add_trace(go.Choroplethmap(
                geojson=geojson,
                locations=pareto_geoids,
                z=[1.0] * len(pareto_geoids),
                featureidkey="id",
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker=dict(line=dict(width=2.5, color=tc["pareto_border"])),
                hoverinfo="skip",
                name="★ Pareto-Optimal",
            ))

    # Layer 4: Coal-to-Nuclear overlay — purple border on all coal counties
    if show_coal and _all_coal_geoids:
        fig.add_trace(go.Choroplethmap(
            geojson=geojson,
            locations=_all_coal_geoids,
            z=[1.0] * len(_all_coal_geoids),
            featureidkey="id",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            showscale=False,
            marker=dict(opacity=0.0, line=dict(width=2.0, color=tc["coal_border"])),
            hoverinfo="skip",
            name="Coal Infrastructure",
        ))

fig.update_layout(
    map=dict(
        style=tc["map_style"],
        center={"lat": 39.5, "lon": -96},
        zoom=3.5,
        layers=[{
            "source": state_geojson,
            "type": "line",
            "color": tc["state_border"],
            "line": {"width": 1},
            "opacity": 0.8,
        }],
    ),
    margin=dict(r=0, t=0, l=0, b=0),
    height=560,
    paper_bgcolor=tc["paper_bg"],
    font=dict(color=tc["font_color"]),
)

selection = st.plotly_chart(
    fig,
    width="stretch",
    on_select="rerun",
    key="county_map",
    config={"displayModeBar": True, "displaylogo": False},
)

# Resolve click → session state
_active_geoids = set(active_df["geoid"])
if selection and hasattr(selection, "selection") and selection.selection.points:
    pt  = selection.selection.points[0]
    loc = pt.get("location")
    if loc and loc in _active_geoids:
        st.session_state["selected_geoid"] = loc
    elif loc:
        st.session_state.pop("selected_geoid", None)


# ── Bottom section ─────────────────────────────────────────────────────────────

table_col, detail_col = st.columns([3, 2], gap="large")


# ── Top 20 table ──────────────────────────────────────────────────────────────

with table_col:
    st.markdown('<div class="section-header">Top 20 Candidates</div>', unsafe_allow_html=True)

    if len(active_df) == 0:
        st.info("No counties match the current filters.")
    elif reactor_mode == "LWR":
        _lwr_cols = ["rank", "county_name", "state", _disp_col,
                     "pga_max", "pct_sfha", "population_density",
                     "max_voltage", "total_energy_consumption_mwh", "on_nsga2_pareto"]
        if show_coal:
            _lwr_cols.append("coal_capacity_mw")
        _sort_lwr = _disp_col if _disp_col != _score_col else "rank"
        if _sort_lwr == "rank":
            top20 = active_df.dropna(subset=["rank"]).nsmallest(20, "rank")[_lwr_cols].copy()
        else:
            top20 = active_df.nlargest(20, _disp_col)[_lwr_cols].copy()

        if "rank" in top20.columns:
            top20["rank"] = top20["rank"].astype(int)
        top20[_disp_col] = top20[_disp_col].map("{:.3f}".format)
        top20["pga_max"]    = top20["pga_max"].map("{:.3f}".format)
        top20["pct_sfha"]   = top20["pct_sfha"].map("{:.1%}".format)
        top20["population_density"] = top20["population_density"].map("{:.1f}".format)
        top20["max_voltage"] = top20["max_voltage"].fillna(0).map("{:.0f}".format)
        top20["total_energy_consumption_mwh"] = top20["total_energy_consumption_mwh"].map("{:,.0f}".format)
        top20["on_nsga2_pareto"] = top20["on_nsga2_pareto"].map({True: "★", False: ""})
        if show_coal:
            top20["coal_capacity_mw"] = top20["coal_capacity_mw"].fillna(0).map("{:.0f}".format)

        _lwr_display_cols = ["Rank", "County", "State", _score_label,
                             "Seismic (g)", "Flood Risk", "Pop / km²",
                             "Max kV", "Energy (MWh)", "Pareto"]
        if show_coal:
            _lwr_display_cols.append("Coal MW")
        top20.columns = _lwr_display_cols

        st.dataframe(
            top20,
            width="stretch",
            hide_index=True,
            column_config={
                "Rank":       st.column_config.TextColumn(width="small"),
                "Pareto":     st.column_config.TextColumn(
                                  "★", width="small",
                                  help="★ = county is on the NSGA-II Pareto front"),
                _score_label: st.column_config.TextColumn(width="small"),
            },
        )
        st.caption(
            f"★ marks the {int(df['on_nsga2_pareto'].sum())} counties on the NSGA-II Pareto front. "
            "These sites also appear with an amber outline on the map. "
            "Click any county on the map to view its full profile."
        )
    else:
        # SMR mode: rank by _disp_col descending
        _smr_cols = ["county_name", "state", _disp_col,
                     "pga_max", "pct_sfha", "population_density", "max_voltage"]
        if show_coal:
            _smr_cols.append("coal_capacity_mw")
        top20 = active_df.nlargest(20, _disp_col)[_smr_cols].copy()

        top20[_disp_col] = top20[_disp_col].map("{:.3f}".format)
        top20["pga_max"]   = top20["pga_max"].map("{:.3f}".format)
        top20["pct_sfha"]  = top20["pct_sfha"].map("{:.1%}".format)
        top20["population_density"] = top20["population_density"].map("{:.1f}".format)
        top20["max_voltage"] = top20["max_voltage"].fillna(0).map("{:.0f}".format)
        if show_coal:
            top20["coal_capacity_mw"] = top20["coal_capacity_mw"].fillna(0).map("{:.0f}".format)

        _smr_display_cols = ["County", "State", _score_label,
                             "Seismic (g)", "Flood Risk", "Pop / km²", "Max kV"]
        if show_coal:
            _smr_display_cols.append("Coal MW")
        top20.columns = _smr_display_cols

        st.dataframe(
            top20,
            width="stretch",
            hide_index=True,
            column_config={
                "SMR Score": st.column_config.TextColumn(width="small"),
            },
        )
        mode_label = "NuScale VOYGR" if "NuScale" in reactor_mode else "General SMR"
        st.caption(
            f"Ranked by composite SMR score under {mode_label} thresholds. "
            "Counties disqualified by seismic or population criteria are excluded. "
            "Click any county on the map to view its full profile."
        )


# ── County detail panel ────────────────────────────────────────────────────────

with detail_col:
    st.markdown('<div class="section-header">County Detail</div>', unsafe_allow_html=True)

    selected_geoid = st.session_state.get("selected_geoid")
    # Look up in active_df so SMR-scored row is available; fall back to full df for context
    row_matches    = active_df[active_df["geoid"] == selected_geoid] if selected_geoid else pd.DataFrame()
    if len(row_matches) == 0 and selected_geoid:
        row_matches = df[df["geoid"] == selected_geoid]

    if selected_geoid and len(row_matches) > 0:
        row        = row_matches.iloc[0]
        state_abbr = geo_lookup.get(selected_geoid, {}).get("state_abbr", "")
        is_pareto  = bool(row.get("on_nsga2_pareto", False)) if reactor_mode == "LWR" else False

        badge = (
            ' <span class="pareto-badge"'
            ' aria-label="Pareto Tier 1: non-dominated across all 6 optimization criteria">'
            "★ Pareto Tier 1</span>"
            if is_pareto else ""
        )
        st.markdown(
            f"### {row['county_name']}, {state_abbr}{badge}",
            unsafe_allow_html=True,
        )

        rc1, rc2 = st.columns(2)
        if reactor_mode == "LWR":
            rank_raw = row["rank"]
            score    = row["mcda_score"]
            if pd.notna(rank_raw) and pd.notna(score):
                rank   = int(rank_raw)
                pctile = round(100 * (1 - rank / _rank_max))
                if rank <= 50:
                    rank_interp = "Top 50 nationally"
                elif rank <= 200:
                    rank_interp = f"Top {pctile}%, strong site"
                elif rank <= 600:
                    rank_interp = f"Top {pctile}%"
                else:
                    rank_interp = f"#{rank} of {_rank_max}"
                rc1.metric("LWR Rank", f"#{rank}", rank_interp)
                rc2.metric("MCDA Score", f"{score:.4f}")
            else:
                rc1.metric("LWR Rank", "N/A", "Masked due to insufficient data")
                rc2.metric("MCDA Score", "N/A")
        else:
            smr_score = row.get("smr_score", np.nan)
            mode_label = "NuScale VOYGR" if "NuScale" in reactor_mode else "General SMR"
            if pd.notna(smr_score):
                rc1.metric("SMR Score", f"{smr_score:.4f}")
                rc2.metric("Mode", mode_label)
            else:
                rc1.metric("SMR Score", "Disqualified")
                rc2.metric("Mode", mode_label)

        if selected_geoid not in _active_geoids:
            st.warning("This county is outside the current filter settings or disqualified under the active reactor mode.")

        st.divider()
        st.markdown("**Safety Criteria**")

        pga = row["pga_max"]
        seismic_txt = (
            "minimal seismic hazard" if pga < 0.05 else
            "low seismic hazard" if pga < 0.10 else
            "moderate seismic hazard" if pga < 0.15 else
            "elevated, approaching NRC review threshold" if pga < 0.20 else
            "high, near the 0.30 g regulatory limit"
        )
        st.markdown(f"- **Seismic Risk:** {pga:.3f} g ({seismic_txt})")

        sfha = row["pct_sfha"]
        flood_txt = (
            "negligible flood hazard" if sfha < 0.03 else
            "low flood exposure" if sfha < 0.08 else
            "moderate flood exposure" if sfha < 0.14 else
            "high flood exposure, near the 0.20 threshold"
        )
        st.markdown(f"- **Flood Risk:** {sfha:.1%} SFHA ({flood_txt})")

        pop_d = row["population_density"]
        pop_txt = (
            "very sparse, well-suited for NRC exclusion zones" if pop_d < 10 else
            "low density, good buffer potential" if pop_d < 30 else
            "moderate density" if pop_d < 100 else
            "dense, proximity review likely required"
        )
        st.markdown(f"- **Population Density:** {pop_d:.1f} / km² ({pop_txt})")

        mil = row.get("pct_military", 0.0)
        if pd.notna(mil) and mil > 0:
            st.markdown(f"- **Military Coverage:** {mil:.1%} of county")

        st.divider()
        st.markdown("**Infrastructure & Demand**")

        lake_d  = row.get("dist_to_lakes_km", np.nan)
        river_d = row.get("distance_to_rivers_km", np.nan)
        if pd.notna(lake_d) and pd.notna(river_d):
            nearest_water = min(lake_d, river_d)
            water_src     = "lake" if lake_d <= river_d else "river"
            water_txt = (
                f"excellent cooling access via nearby {water_src}" if nearest_water < 5 else
                f"good water access, nearest {water_src} within range" if nearest_water < 15 else
                f"adequate water access from {water_src}" if nearest_water < 30 else
                "limited water access, cooling infrastructure would be required"
            )
            st.markdown(f"- **Nearest Water Body:** {nearest_water:.1f} km ({water_txt})")

        voltage = row.get("max_voltage", np.nan)
        if pd.notna(voltage) and voltage > 0:
            grid_txt = (
                "high-voltage transmission line present (345+ kV)" if voltage >= 345 else
                "strong grid connectivity" if voltage >= 230 else
                "moderate grid access" if voltage >= 138 else
                "low voltage, grid upgrade would be required"
            )
            st.markdown(f"- **Max Transmission Line:** {voltage:.0f} kV ({grid_txt})")

        energy     = row.get("total_energy_consumption_mwh", np.nan)
        energy_pct = row.get("total_energy_consumption_mwh_pct", np.nan)
        if pd.notna(energy):
            energy_txt = (
                "high local demand, strong market for reactor output"
                if pd.notna(energy_pct) and energy_pct >= 0.75 else
                "moderate local demand"
                if pd.notna(energy_pct) and energy_pct >= 0.5 else
                "low local demand, output would likely be exported to the regional grid"
            )
            st.markdown(f"- **Energy Consumption:** {energy:,.0f} MWh ({energy_txt})")

        dc = row.get("data_centers_count", 0)
        if pd.notna(dc) and dc > 0:
            st.markdown(f"- **Data Centers:** {int(dc)} (indicative of high baseload electricity demand)")

        if row.get("has_plant", False):
            st.info("This county hosts or hosted a nuclear plant in NRC records.")

        if row.get("has_coal_plant", False):
            coal_mw = row.get("coal_capacity_mw", 0.0) or 0.0
            coal_detail = coal_lookup.get(row["geoid"], {})
            coal_status = []
            if coal_detail.get("has_operating_coal"):
                coal_status.append("operating")
            if coal_detail.get("has_retired_coal"):
                coal_status.append("retired")
            status_str = " and ".join(coal_status) if coal_status else "existing"
            st.info(
                f"This county has {status_str} coal plant infrastructure "
                f"({coal_mw:.0f} MW capacity). Coal-to-nuclear conversion may be "
                "feasible here, with potential benefits for workforce continuity and "
                "transmission interconnection."
            )

        if is_pareto:
            st.divider()
            st.success(
                "★ This county is on the NSGA-II Pareto front. It scores well enough across all "
                "6 criteria (seismic risk, flood risk, population density, water access, grid "
                "connectivity, and energy demand) that no other county strictly outperforms it "
                "on every dimension."
            )

    else:
        st.info("Click a highlighted county on the map to view its full scoring profile.")
        st.caption(
            "Darker green means higher suitability. "
            "Gray counties did not pass the safety filters. "
            "Amber outlines mark Pareto-optimal counties (★)."
        )
