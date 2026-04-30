# Team Nuclear Family - Nuclear Reactor Siting in the United States

## Project Overview
AI data centers are consuming more electricity than ever, and demand keeps climbing as new ones come online. Nuclear is one of the few clean energy sources that can actually keep up. But where should new reactors go?

That's what this project answers.

We built a geospatial framework that scores every realistic location in the US based on two things: how suitable it is to physically build a reactor there, and how much energy demand exists nearby (since nuclear energy can only be used locally). The final deliverable is a ranked map of optimal reactor sites framed as an actual policy recommendation.

## Team Members
- Brian Lin
- Khue Nguyen
- Millie Chu
- Nina Schreiber
- Quentin Conkle

## Data Description
This project integrates 13 datasets from various open sources that represent different factors that make a county a good or bad candidate for a nuclear reactor, things like population, energy consumption, and seismic risk. All original datasets are in `raw_data`, and all processed data goes in `processed_data`.

| Dataset | Source | Description |
|---|---|---|
| County boundaries | [U.S. Census Bureau TIGER/Line (2025)](https://www2.census.gov/geo/tiger/TIGER2025/COUNTY/) | Geographic boundaries for all US counties, used for spatial joins |
| US population in 2025 by county | [Census.gov](Census.gov) | Population of each county at the end of 2025 |
| Housing units by county | [Census.gov](Census.gov) | Number of housing units and median household income in each county in 2024 |
| Energy consumption by county | [Find Energy](https://findenergy.com/) | Total energy consumption of each county in 2024 |
| Data centers in the US | [Office of Scientific and Technical Information](https://www.osti.gov/biblio/2550666) and [OpenStreetMap](https://www.openstreetmap.org/about) | All data centers in the US, used as a proxy for energy demand |
| Flood hazard | [National Flood Hazard Layer (NFHL) Database](https://msc.fema.gov/portal/advanceSearch) | Regions with high risk of flooding, by state |
| Lakes in the US | [HydroSHEDS](https://www.hydrosheds.org/products/hydrolakes) | All lakes in the US |
| Rivers | [ArcGIS](https://hub.arcgis.com/datasets/esri::usa-rivers-and-streams/explore?location=44.592478%2C-119.086063%2C3) and [National Weather Service](https://www.weather.gov/gis/Rivers) | All rivers and streams in the US |
| Military areas | [U.S. Department of Transportation](https://geodata.bts.gov/datasets/usdot::military-bases/explore?location=42.326716%2C3.071733%2C1) | Military installations across the US |
| Existing nuclear plants in the US | [U.S. Nuclear Regulatory Commission](https://www.nrc.gov/) | All nuclear power plants currently in the US, including ones no longer in service |
| Seismic hazard by county | [United States Geological Survey (USGS)](https://www.usgs.gov/programs/earthquake-hazards/hazards) | County-level seismic hazard |
| Transmission lines in the US | [ArcGIS](https://www.arcgis.com/home/item.html?id=d4090758322c4d32a4cd002ffaa0aa12) | All transmission lines in the US, each represented as a point |
| Wetlands in the US | [U.S. Fish & Wildlife Service](https://www.fws.gov/program/national-wetlands-inventory/data-download) | Wetland areas in the US (mangrove, mud, marsh, etc.) |

### Data Cleaning and Preprocessing

Some of the major cleaning and preprocessing steps:

- Drop inaccurate or invalid observations in some datasets:
    - Military installations not currently in service
    - Transmission lines not in service or with invalid voltage values
    - Intermittent streams or rivers (they cannot support nuclear cooling because they run dry seasonally)
- Spatial join datasets not originally organized at the county level, using county boundaries as a backbone
- For some datasets we used `geopandas.overlay()` to find the specific area intersecting with each county (for example, with military data we used the actual overlapping area rather than the full base footprint)
- Drop unused variables from the original datasets, keeping 2 to 4 per source that are likely to matter for scoring
- Feature engineering: we created new variables from existing ones, like the number of nuclear plants in each county and the proportion of a county that has high flood risk
- Merge all datasets into a single county-level file. All merging steps are in `notebooks/cleaning/merge.ipynb`

### Key Variables
Our final dataset `processed_data/final_dataset.csv` has **34 columns** and **3,235 observations**, one per county. Some key variables:

| Variable Name | Description | Unit |
| :-- | :-- | :-- |
| `population` | County population in 2025 | people |
| `total_energy_consumption_mwh` | County total energy consumption in 2024 | MWh |
| `pct_sfha` | Proportion of the county marked as a severe flood hazard area | |
| `total_lake_area` | Total lake area in the county | km² |
| `dist_to_lakes_km` | Distance from the county centroid to the nearest lake | km |
| `dist_to_rivers_km` | Distance from the county centroid to the nearest river | km |
| `pct_military` | Proportion of the county covered by military installations | |
| `pga_max` | Maximum peak ground acceleration in the county (seismic risk indicator) | g (fraction of gravitational acceleration, 9.8 m/s²) |
| `max_voltage` | Maximum voltage of a transmission line in the county | kV |


## Methodology

### Exploratory Data Analysis
Our main EDA notebook is `notebooks/eda/eda.ipynb`. Some of the techniques we used:

- Correlation heatmap across all numerical variables. We found high correlation between:
    - `housing_units` and `median_household_income`
    - `distance_to_lines_km` and `distance_to_rivers_km`
    - `total_rivers_mile` and `rivers_count`
    - `total_energy_consumption_mwh` and `population`

- Boxplots to explore distributions of key numerical variables

- Comparison of counties with and without an existing nuclear reactor. Key findings:
    - Counties with nuclear plants consistently have lower seismic and flood risk, and stronger access to rivers and water bodies
    - Plant counties tend to cluster around lower population density but consume more energy overall compared to counties without a plant
    - Plant counties average a maximum voltage around 300 to 500 kV with a moderate number of lines, suggesting that higher-voltage grid access is preferred over a dense but lower-voltage grid

### Modeling
All modeling notebooks are in `notebooks/modeling`. The process runs in this order:

1. `mask.ipynb`
2. `scoring.ipynb`
3. `logistic_regression.ipynb`
4. `validation.ipynb`
5. `random_forest.ipynb`
6. `moo.ipynb`

### Step 1: Masking

We start by hard-excluding any counties that don't meet the minimum safety requirements for a nuclear reactor site. A county is excluded if:

- `pga_max` exceeds 0.3
- `pct_sfha` exceeds 0.2
- `pct_military` exceeds 0.1
- `pct_protected` exceeds 0.1

These thresholds are based on regulatory guidelines from the [Nuclear Regulatory Commission](https://www.nrc.gov/docs/ml1218/ml12188a053.pdf), the [Next Generation Nuclear Plant (NGNP)](https://inldigitallibrary.inl.gov/sites/sti/sti/5144360.pdf), and [Executive Orders from the Office of the Federal Register](https://www.archives.gov/federal-register/codification/executive-order/11988.html).

After masking, the dataset went from 3,235 counties down to **2,161 candidates**.

### Step 2: Scoring

Our scoring framework uses [Multi-Criteria Decision Analysis (MCDA)](https://www.1000minds.com/decision-making/what-is-mcdm-mcda), assigning weights to 12 variables to compute a suitability score per county.

Because most existing plants were built in the 1970s and 80s under older criteria, we defined a modern framework driven by expert knowledge and NRC guidelines. We consulted with professor [Caleb Brooks](https://npre.illinois.edu/people/profile/csbrooks) from Grainger Engineering to figure out which variables matter most.

We ranked variables by importance, with the top three being `pga_max`, `pct_sfha`, and `population_density` (all safety criteria), then applied the [Rank Order Centroid (ROC)](https://www.nature.com/articles/s41598-024-61945-z) method to convert that ranking into weights.

To check robustness, we ran a sensitivity analysis with random +/-20% weight perturbations across 1,000 simulations. On average, **18.8 counties** in the top 20 remained the same, with a **95% CI of [18.744, 18.847]**. This confirms that the recommendations hold up even if the weights aren't perfect.

### Step 3: Validation and Comparison

The top-scoring counties from our MCDA framework mostly don't have existing nuclear plants. We expected this, since older siting criteria don't necessarily match our modern framework.

To understand the difference, we fit supervised ML models on the dataset (which includes a flag for historical plant locations) and extracted feature importance to see what priorities actually drove siting in the past. The logistic regression work is in `logistic_regression.ipynb`, and the tree-based models and SHAP analysis are in `validation.ipynb`.

**Key results:**

- Models used: Logistic Regression, Decision Tree, XGBoost (with class imbalance handling via SMOTE, `scale_pos_weight`, and `class_weight='balanced'`)

- Because of heavy class imbalance (41 counties with plants vs. about 2,100 without), minority class F1 stayed low across all models:
    - Logistic Regression: 0.11
    - Decision Tree: 0.08 to 0.15
    - XGBoost: 0.10 to 0.18

- This suggests the difficulty isn't just class imbalance. Historical siting decisions were probably shaped by things our dataset doesn't capture, like politics, public consent, and economics.

- Despite the low predictive performance, feature importance was pretty consistent across all models (and confirmed by SHAP analysis): `total_energy_consumption_mwh`, `max_voltage`, and `data_centers_count` rank highest. This partly supports a shift from infrastructure- and demand-driven siting toward a safety-first framework.

We also ran a Random Forest in `random_forest.ipynb` as an additional check. With `class_weight='balanced'`, it achieved 98% accuracy overall but predicted zero plant counties correctly, which reflects how severe the class imbalance is and reinforces the same interpretation above.

### Step 4: Recommendations

We used non-dominated sorting (`pymoo`'s `NonDominatedSorting`) across 6 key objectives.

We ran MOO twice: once on the full dataset and once on the MCDA top 10%, to see how much our weights narrow the candidate space. Each run returns a Pareto front, a subset of counties that no other county dominates on all criteria simultaneously.

Counties are assigned to tiers based on their Pareto front membership:

- **Tier 1** (175 counties): appear on both fronts
- **Tier 2** (1 county): MCDA Pareto only
- **Tier 3** (569 counties): global Pareto only

**Key metrics:**

- Our MCDA scoring is fairly aggressive: it overlooked 569 non-dominated counties that the global Pareto front found
- **Hypervolume ratio**: the MCDA top 10% still captures **91.6%** of the global Pareto front's trade-off coverage
- The top 20 candidates from MCDA alone and from MOO have 18 in common, confirming the consistency of our framework

Most optimal counties are in the Great Lakes area, e.g. Michigan or Wisconsin.


## Key Findings
- Counties with nuclear plants tend to have lower flood and seismic risks, lower population (density), closer water access, and higher-voltage grid access, compared to counties without
- Policies for siting a nuclear reactor have shifted over time: the past prioritized energy demand and other factors (politics, economy, etc.) while today's criteria (from the [Nuclear Regulatory Commission](https://www.nrc.gov/docs/ml1218/ml12188a053.pdf) and our framework) focuses more on safety. 
- Most of the most suitable counties to have a nuclear reactor are in the Great Lakes Area (Wisconsin, Michigan, etc.) - found by both **Multi-objective Optimization** and our **MCDA scoring framework**.


## Installation and Setup

Instructions for setting up the project environment:
### 1. Clone the repository
```bash
git clone https://github.com/UIUC-DSC/team-nuclear-family.git
cd team-nuclear-family
```
### 2. Installations
```bash
pip install -r requirements.txt
```
### 3. Run the pipeline
Notebooks in `notebooks` should be run in the following order: 
1. `notebooks/cleaning` ←  data cleaning/preprocessing
2. `notebooks/eda` ←  exploratory data analysis
3. `notebooks/modeling` ←  modeling


## Project Structure
```
team-nuclear-family/
├── notebooks/
│   ├── cleaning/            # data cleaning and feature engineering
│   ├── eda/                 # exploratory data analysis 
│   └── modeling/
│       ├── mask.ipynb       # hard exclusion
│       ├── scoring.ipynb    # MCDA scoring 
│       ├── logistic_regression.ipynb    # supervised ML validation
│       ├── validation.ipynb # supervised ML validation
│       └── moo.ipynb        # multi-objective optimization
│
├── processed_data/          # cleaned and merged datasets
├── raw_data/                # original raw datasets
├── results/                 # data visualizations
├── requirements.txt         # package requirement
└── README.md                # this file
```

## Future Work 
- Include more variables (politics, economics, county policies, etc.) in the scoring framework to have a more comprehensive model
- Apply more class imbalance handling methods (SMOTE, RandomOverSampler, etc.) in training supervised ML models
- Apply sentiment analysis techniques to analyze residents' consensus of building a nuclear reactor in a region
- Compare our results with existing models (e.g. [OR-SAGE](https://orsage.ornl.gov/)) used by the [Department of Energy](https://www.energy.gov/) to validate our results

## Acknowledgement 
- Professor [Caleb Brooks](https://npre.illinois.edu/people/profile/csbrooks) (UIUC Nuclear Engineering) - expert-driven feature rankings.
- [Nuclear Regulatory Committee (NRC) Guidelines](https://www.ecfr.gov/current/title-10/chapter-I/part-100) on nuclear sites criteria
- **Erdem et al. (2025)** — methodological reference for multi-objective nuclear siting ([paper](https://www.sciencedirect.com/science/article/pii/S2590174525000558))
- **Data Sources**: US Census Bureau (TIGER/ACS), FEMA NFHL, HIFLD, USGS, EIA, NRC, HydroLAKES, NOAA, FWS NWI, OpenStreetMap (see **Data Description** above)
- **Tools**: pandas, GeoPandas, sklearn, pymoo, XGBoost, SHAP, Matplotlib, seaborn
- **Project Lead**: Grant Magnabosco

## Contact Information

We can be contacted for questions and collaboration via email:
- Brian: brianl11@illinois.edu
- Khue: khueln2@illinois.edu
- Millie: milliec2@illinois.edu
- Nina: ninass2@illinois.edu
- Quentin: qconkle2@illinois.edu