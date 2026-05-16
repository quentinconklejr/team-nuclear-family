import json

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
    with open("processed_data/county_boundaries.geojson") as f:
        return json.load(f)


@st.cache_data
def load_state_geojson():
    with open("processed_data/state_boundaries.geojson") as f:
        return json.load(f)


@st.cache_data
def build_state_lines(_geojson):
    """Flatten dissolved state polygon exterior rings into lat/lon arrays for Scattermap."""
    lats: list = []
    lons: list = []
    for feat in _geojson["features"]:
        geom = feat["geometry"]
        rings = (
            [geom["coordinates"][0]]
            if geom["type"] == "Polygon"
            else [p[0] for p in geom["coordinates"]]
        )
        for ring in rings:
            lons.extend(pt[0] for pt in ring)
            lats.extend(pt[1] for pt in ring)
            lons.append(None)
            lats.append(None)
    return lats, lons


@st.cache_data
def build_geo_lookup(_geojson):
    result = {}
    for feat in _geojson["features"]:
        p = feat["properties"]
        # Always derive from state_fips — GeoJSON state_abbr field is systematically corrupted
        state_abbr = _FIPS_TO_STATE.get(p.get("state_fips", ""), "")
        result[p["geoid"]] = {
            "state_abbr": state_abbr,
            "county_name_full": p["county_name_full"],
        }
    return result



candidates = load_candidates()
pareto     = load_pareto()
geojson    = load_geojson()
geo_lookup = build_geo_lookup(geojson)
state_geojson = load_state_geojson()
state_lats, state_lons = build_state_lines(state_geojson)

all_geoids = [f["properties"]["geoid"] for f in geojson["features"]]

df = candidates.merge(
    pareto[["geoid", "on_nsga2_pareto", "on_both_fronts"]],
    on="geoid",
    how="left",
)
df["on_nsga2_pareto"] = df["on_nsga2_pareto"].fillna(False).astype(bool)
df["on_both_fronts"]  = df["on_both_fronts"].fillna(False).astype(bool)
df["state"] = df["geoid"].map(lambda g: geo_lookup.get(g, {}).get("state_abbr", ""))

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
        help="Choose a color scheme. High Contrast meets WCAG AAA requirements.",
    )

    accessible = st.toggle(
        "Enhanced Accessibility",
        value=False,
        help=(
            "Scales up all text sizes and adds visible map descriptions "
            "for low-vision users (WCAG 2.1 AA)."
        ),
    )

    st.divider()
    st.markdown("## Filters")
    st.caption("Narrow the map and table to counties that meet all three safety thresholds below.")
    st.divider()

    st.markdown("**Earthquake Risk**")
    st.caption(
        "Maximum ground shaking strength (g-force). "
        "The NRC requires sites to stay below 0.30 g. "
        "Lower = safer. Move the slider left to be more strict."
    )
    pga_filter = st.slider(
        "Earthquake risk cutoff (g)",
        0.0, 0.30, 0.30, 0.01,
        format="%.2f g",
        help=(
            "Peak Ground Acceleration (PGA): how strongly the ground shakes during an earthquake. "
            "0.05 g = barely felt; 0.30 g = NRC regulatory threshold for reactor siting."
        ),
    )

    st.markdown("**Flood Risk**")
    st.caption(
        "Percent of the county in a FEMA high-risk flood zone. "
        "Reactors need dry, stable ground. "
        "Lower = safer. Move the slider left to exclude flood-prone counties."
    )
    sfha_filter = st.slider(
        "Flood zone coverage cutoff (%)",
        0.0, 20.0, 20.0, 1.0,
        format="%.0f%%",
        help=(
            "Special Flood Hazard Area (SFHA): the percentage of the county inside FEMA's "
            "100-year flood boundary. 0% = no flood risk; 20% = very flood-prone."
        ),
    )
    sfha_filter = sfha_filter / 100.0  # convert back to fraction for filtering

    st.markdown("**Population Density**")
    st.caption(
        "How many people live per square kilometer. "
        "Reactors need large exclusion zones with few nearby residents. "
        "Lower = easier to site safely."
    )
    pop_filter = st.slider(
        "Max residents per km²",
        0, 10000, 10000, 50,
        format="%d / km²",
        help=(
            "Population density of the county. "
            "Under 10/km² = very rural (ideal). "
            "Over 100/km² = suburban or urban (challenging for NRC exclusion zones)."
        ),
    )

    st.divider()
    pareto_only = st.toggle(
        "Show only top-tier sites (Pareto-optimal)",
        value=False,
        help=(
            "A Pareto-optimal county is one where you can't improve any single criterion "
            "(e.g., earthquake safety) without making another worse (e.g., water access). "
            "These 111 counties represent the best achievable trade-offs across all 6 factors."
        ),
    )

    st.divider()
    st.markdown("**About**")
    st.caption(
        "Team Nuclear Family — IDSC Data Dive Spring 2026 \U0001f947  \n"
        "MCDA scores reflect NRC-guided safety-first weighting with "
        "Rank Order Centroid method. Sensitivity CI: 18.8 / 20 top counties stable."
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


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">Nuclear Reactor Siting Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    "Interactive explorer of 2,161 candidate US counties scored under our MCDA siting framework. "
    "Counties are colored by overall suitability score — darker green indicates higher suitability. "
    "Pareto-optimal counties are additionally outlined in amber. "
    "Use the sidebar to filter by safety thresholds, then click any county for a full profile."
    "</div>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Showing", f"{len(filtered_df):,}", f"of 2,161 candidates")
m2.metric("Pareto-Optimal Shown", f"{int(filtered_df['on_nsga2_pareto'].sum()):,}", "NSGA-II front")
m3.metric(
    "Best MCDA Score",
    f"{filtered_df['mcda_score'].max():.3f}" if len(filtered_df) else "—",
)
_best_rank = filtered_df["rank"].min()
m4.metric("Highest Rank Shown", f"#{int(_best_rank)}" if pd.notna(_best_rank) else "—")


# ── Map ───────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">County Suitability Map</div>', unsafe_allow_html=True)

# Screen-reader description (hidden visually)
st.markdown(
    '<p class="sr-only" role="img">'
    "Choropleth map of the United States showing nuclear reactor siting suitability "
    "scores by county. Darker green indicates higher MCDA suitability. State borders "
    "are drawn as distinct outlines. Pareto-optimal counties are additionally outlined "
    "in amber. Click any highlighted county to load its full profile. "
    "The Top 20 Candidates table below is a keyboard-accessible tabular alternative."
    "</p>",
    unsafe_allow_html=True,
)

if accessible:
    st.caption(
        "Map: counties colored green by MCDA suitability score (darker = higher). "
        "State boundaries shown as distinct outlines. "
        "Pareto-optimal counties have an amber outline and ★ in the tooltip. "
        "Click a county to load its full profile, or use the table below."
    )

tc = _THEMES[theme]
fig = go.Figure()

# Layer 1: all counties — neutral background (opacity<1 lets base tiles show through,
# making water bodies like Lake Michigan visible underneath)
fig.add_trace(go.Choroplethmap(
    geojson=geojson,
    locations=all_geoids,
    z=[0.0] * len(all_geoids),
    featureidkey="properties.geoid",
    colorscale=[[0, tc["county_fill"]], [1, tc["county_fill"]]],
    showscale=False,
    marker=dict(opacity=0.20, line=dict(width=0.2, color=tc["county_border"])),
    hoverinfo="skip",
    name="",
))

# Layer 2: filtered candidates — colored by MCDA score
if len(filtered_df) > 0:
    _rank_str  = filtered_df["rank"].apply(lambda x: f"#{int(x)}" if pd.notna(x) else "unranked")
    _score_str = filtered_df["mcda_score"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    hover_text = (
        "<b>" + filtered_df["county_name"] + ", " + filtered_df["state"].fillna("") + "</b><br>"
        + "Rank: " + _rank_str + "<br>"
        + "MCDA Score: " + _score_str + "<br>"
        + "Seismic: " + filtered_df["pga_max"].map("{:.3f}".format) + " g<br>"
        + "Flood: " + (filtered_df["pct_sfha"] * 100).map("{:.1f}".format) + "% SFHA<br>"
        + "Pop Density: " + filtered_df["population_density"].map("{:.1f}".format) + " / km²"
        + filtered_df["on_nsga2_pareto"].map({True: "<br><b>★ Pareto-Optimal</b>", False: ""})
    )

    fig.add_trace(go.Choroplethmap(
        geojson=geojson,
        locations=filtered_df["geoid"].tolist(),
        z=filtered_df["mcda_score"].tolist(),
        featureidkey="properties.geoid",
        colorscale=tc["colorscale"],
        zmin=_score_min,
        zmax=_score_max,
        colorbar=dict(
            title=dict(text="MCDA Score", font=dict(size=12, color=tc["font_color"])),
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

    # Layer 3: Pareto counties — amber outline as non-color visual indicator
    pareto_geoids = filtered_df.loc[filtered_df["on_nsga2_pareto"], "geoid"].tolist()
    if pareto_geoids:
        fig.add_trace(go.Choroplethmap(
            geojson=geojson,
            locations=pareto_geoids,
            z=[1.0] * len(pareto_geoids),
            featureidkey="properties.geoid",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            showscale=False,
            marker=dict(line=dict(width=2.5, color=tc["pareto_border"])),
            hoverinfo="skip",
            name="★ Pareto-Optimal",
        ))

# Layer 4: state boundaries — drawn on top of all county fills
fig.add_trace(go.Scattermap(
    lat=state_lats,
    lon=state_lons,
    mode="lines",
    line=dict(width=2.0, color=tc["state_border"]),
    hoverinfo="skip",
    showlegend=False,
    name="State Boundaries",
))

fig.update_layout(
    map=dict(
        style=tc["map_style"],
        center={"lat": 39.5, "lon": -96},
        zoom=3.5,
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
if selection and hasattr(selection, "selection") and selection.selection.points:
    pt  = selection.selection.points[0]
    loc = pt.get("location")
    if loc and loc in filtered_geoids:
        st.session_state["selected_geoid"] = loc
    elif loc:
        st.session_state.pop("selected_geoid", None)


# ── Bottom section ─────────────────────────────────────────────────────────────

table_col, detail_col = st.columns([3, 2], gap="large")


# ── Top 20 table ──────────────────────────────────────────────────────────────

with table_col:
    st.markdown('<div class="section-header">Top 20 Candidates</div>', unsafe_allow_html=True)

    if len(filtered_df) == 0:
        st.info("No counties match the current filters.")
    else:
        top20 = filtered_df.dropna(subset=["rank"]).nsmallest(20, "rank")[[
            "rank", "county_name", "state", "mcda_score",
            "pga_max", "pct_sfha", "population_density",
            "max_voltage", "total_energy_consumption_mwh", "on_nsga2_pareto",
        ]].copy()

        top20["rank"]       = top20["rank"].astype(int)
        top20["mcda_score"] = top20["mcda_score"].map("{:.3f}".format)
        top20["pga_max"]    = top20["pga_max"].map("{:.3f}".format)
        top20["pct_sfha"]   = top20["pct_sfha"].map("{:.1%}".format)
        top20["population_density"] = top20["population_density"].map("{:.1f}".format)
        top20["max_voltage"] = top20["max_voltage"].map("{:.0f}".format)
        top20["total_energy_consumption_mwh"] = top20["total_energy_consumption_mwh"].map("{:,.0f}".format)
        top20["on_nsga2_pareto"] = top20["on_nsga2_pareto"].map({True: "★", False: ""})

        top20.columns = [
            "Rank", "County", "State", "MCDA Score",
            "Seismic (g)", "Flood Risk", "Pop / km²",
            "Max kV", "Energy (MWh)", "Pareto",
        ]

        st.dataframe(
            top20,
            width="stretch",
            hide_index=True,
            column_config={
                "Rank":       st.column_config.TextColumn(width="small"),
                "Pareto":     st.column_config.TextColumn(
                                  "★", width="small",
                                  help="★ = county is on the NSGA-II Pareto front"),
                "MCDA Score": st.column_config.TextColumn(width="small"),
            },
        )
        st.caption(
            f"★ = on NSGA-II Pareto front ({int(df['on_nsga2_pareto'].sum())} counties total). "
            "Pareto counties also appear with an amber outline on the map. "
            "Click a county on the map to see its full profile."
        )


# ── County detail panel ────────────────────────────────────────────────────────

with detail_col:
    st.markdown('<div class="section-header">County Detail</div>', unsafe_allow_html=True)

    selected_geoid = st.session_state.get("selected_geoid")
    row_matches    = df[df["geoid"] == selected_geoid] if selected_geoid else pd.DataFrame()

    if selected_geoid and len(row_matches) > 0:
        row        = row_matches.iloc[0]
        state_abbr = geo_lookup.get(selected_geoid, {}).get("state_abbr", "")
        is_pareto  = bool(row.get("on_nsga2_pareto", False))

        badge = (
            ' <span class="pareto-badge"'
            ' aria-label="Pareto Tier 1 — non-dominated across all optimization criteria">'
            "★ Pareto Tier 1</span>"
            if is_pareto else ""
        )
        st.markdown(
            f"### {row['county_name']}, {state_abbr}{badge}",
            unsafe_allow_html=True,
        )

        rank_raw = row["rank"]
        score    = row["mcda_score"]

        rc1, rc2 = st.columns(2)
        if pd.notna(rank_raw) and pd.notna(score):
            rank   = int(rank_raw)
            pctile = round(100 * (1 - rank / _rank_max))
            if rank <= 50:
                rank_interp = "Top 50 nationally — elite candidate"
            elif rank <= 200:
                rank_interp = f"Top {pctile}% — strong candidate"
            elif rank <= 600:
                rank_interp = f"Top {pctile}% — competitive candidate"
            else:
                rank_interp = f"Ranked #{rank} of {_rank_max}"
            rc1.metric("Overall Rank", f"#{rank}", rank_interp)
            rc2.metric("MCDA Score", f"{score:.4f}")
        else:
            rc1.metric("Overall Rank", "N/A", "Masked — insufficient data")
            rc2.metric("MCDA Score", "N/A")

        if selected_geoid not in filtered_geoids:
            st.warning("This county is outside the current filter settings.")

        st.divider()
        st.markdown("**Safety Criteria**")

        pga = row["pga_max"]
        seismic_txt = (
            "Minimal seismic hazard — ideal" if pga < 0.05 else
            "Low seismic hazard" if pga < 0.10 else
            "Moderate seismic hazard" if pga < 0.15 else
            "Elevated — approach NRC review threshold" if pga < 0.20 else
            "High — near regulatory limit of 0.30 g"
        )
        st.markdown(f"- **Seismic Risk:** {pga:.3f} g — {seismic_txt}")

        sfha = row["pct_sfha"]
        flood_txt = (
            "Negligible flood hazard area" if sfha < 0.03 else
            "Low flood exposure" if sfha < 0.08 else
            "Moderate flood exposure" if sfha < 0.14 else
            "High flood exposure — near 0.20 limit"
        )
        st.markdown(f"- **Flood Risk:** {sfha:.1%} SFHA — {flood_txt}")

        pop_d = row["population_density"]
        pop_txt = (
            "Very sparse — ideal NRC exclusion zone" if pop_d < 10 else
            "Low density — good buffer zone" if pop_d < 30 else
            "Moderate density" if pop_d < 100 else
            "Dense — proximity review recommended"
        )
        st.markdown(f"- **Population Density:** {pop_d:.1f} / km² — {pop_txt}")

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
                f"Excellent cooling access (nearest {water_src})" if nearest_water < 5 else
                f"Good water access (nearest {water_src})" if nearest_water < 15 else
                f"Adequate access (nearest {water_src})" if nearest_water < 30 else
                "Limited water — cooling infrastructure needed"
            )
            st.markdown(f"- **Nearest Water Body:** {nearest_water:.1f} km — {water_txt}")

        voltage = row.get("max_voltage", np.nan)
        if pd.notna(voltage) and voltage > 0:
            grid_txt = (
                "High-voltage transmission ready (345+ kV)" if voltage >= 345 else
                "Strong grid connectivity" if voltage >= 230 else
                "Moderate grid access" if voltage >= 138 else
                "Low voltage — grid upgrade required"
            )
            st.markdown(f"- **Max Transmission Line:** {voltage:.0f} kV — {grid_txt}")

        energy     = row.get("total_energy_consumption_mwh", np.nan)
        energy_pct = row.get("total_energy_consumption_mwh_pct", np.nan)
        if pd.notna(energy):
            energy_txt = (
                "High local demand — strong market for output"
                if pd.notna(energy_pct) and energy_pct >= 0.75 else
                "Moderate local demand"
                if pd.notna(energy_pct) and energy_pct >= 0.5 else
                "Lower local demand — export to grid likely"
            )
            st.markdown(f"- **Energy Consumption:** {energy:,.0f} MWh — {energy_txt}")

        dc = row.get("data_centers_count", 0)
        if pd.notna(dc) and dc > 0:
            st.markdown(f"- **Data Centers:** {int(dc)} — high baseload demand proxy")

        if row.get("has_plant", False):
            st.info("This county hosts or hosted a nuclear plant in NRC records.")

        if is_pareto:
            st.divider()
            st.success(
                "★ This county is **non-dominated** across all 6 multi-objective optimization "
                "criteria (seismic risk, flood risk, population density, water access, grid "
                "connectivity, energy demand) — it is on the NSGA-II Pareto front."
            )

    else:
        st.info(
            "Click any highlighted county on the map to see its full scoring profile "
            "and a plain-English interpretation of why it ranked where it did."
        )
        st.caption(
            "Tip: darker green = higher MCDA suitability score. "
            "Gray counties did not pass the safety masking step. "
            "Amber-outlined counties are Pareto-optimal (★). "
            "State borders are shown as distinct outlines."
        )
