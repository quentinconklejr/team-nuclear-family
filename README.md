# Team Nuclear Family - Nuclear Reactor Siting in the United States

##  Project Overview
AI data centers are consuming more electricity than ever, and it is only increasing with new data centers being built often. Nuclear is one of the few clean energy sources that can actually keep up. But where should new reactors go?

That’s what this project answers.

We’re building a geospatial framework that scores every realistic location in the US based on two things: how suitable it is to physically build a reactor there, and how much energy demand exists in that area because nuclear energy can only be used locally. The final deliverable is a ranked map of optimal reactor sites framed as an actual policy recommendation.

## Team Members 
- Brian Lin
- Khue Nguyen
- Millie Chu 
- Nina Schreiber 
- Quentin Conkle 

## Data Description
This project integrates 13 datasets from various open sources that represent different factors constituting a good site for a nuclear reactor, such as population, energy consumption, seismic risks, etc. All of the original datasets are stored in `raw_data`, and all processed data go in `processed_data`.

| Dataset | Source | Description |
|---|---|---|
| County boundaries | [U.S. Census Bureau TIGER/Line (2025)](https://www2.census.gov/geo/tiger/TIGER2025/COUNTY/) | Geographic boundaries for all counties in the US, used for spatial joins |
| US population in 2025 by county | [Census.gov](Census.gov) | Population of each county by the end of 2025 |
| Housing units by county | [Census.gov](Census.gov) | Number of housing units and median household income in each county in 2024|
| Energy consumption by county | [Find Energy](https://findenergy.com/) | Total energy consumption of each county in 2024 |
| Data centers in the US | [Office of Scientific and Technical Information](https://www.osti.gov/biblio/2550666) and [OpenStreetMap](https://www.openstreetmap.org/about) | All data centers in the US, used as a proxy for energy demand|
| Flood hazard | [National Flood Hazard Layer (NFHL) Database](https://msc.fema.gov/portal/advanceSearch) | **Regions** with high risk of flood by state|
| Lakes in the US | [HydroSHEDS](https://www.hydrosheds.org/products/hydrolakes) | All lakes in the US |
| Rivers | [ArcGIS](https://hub.arcgis.com/datasets/esri::usa-rivers-and-streams/explore?location=44.592478%2C-119.086063%2C3) and [National Weather Service](https://www.weather.gov/gis/Rivers)| All rivers and streams in the US |
| Military areas | [U.S. Department of Transportation](https://geodata.bts.gov/datasets/usdot::military-bases/explore?location=42.326716%2C3.071733%2C1) | Military installations across the US|
| Existing nuclear plants in the US | [U.S. Nuclear Regulatory Commission](https://www.nrc.gov/) | All nuclear power plants currently existing in the US (including ones not in-service) |
| Seismic hazard by county | [United States Geological Survey (USGS)](https://www.usgs.gov/programs/earthquake-hazards/hazards) | County-level seismic hazard |
| Transmission lines in the US | [ArcGIS](https://www.arcgis.com/home/item.html?id=d4090758322c4d32a4cd002ffaa0aa12) | All transmission lines in the US, each represented by a **point**|
| Wetlands in the US | [U.S. Fish & Wildlife Service](https://www.fws.gov/program/national-wetlands-inventory/data-download) | Wetland areas in the US (mangrove, mud, marsh, etc.) |

### Data Cleaning/Preprocessing

Some of the major cleaning and preprocessing steps taken: 
- Drop inaccurate/invalid observations in some datasets: 
    - Military installations not currently in service
    - Transmission lines not in service or with invalid voltage values
    - Intermittent streams or rivers (they cannot support nuclear cooling due to being seasonal)
- Spatial join datasets not originally in county unit using the county boundaries data as a backbone

- For some datasets we used `geopandas.overlay()` to find the specific areas intersecting with the county (e.g. for the military dataset, we found the area of the installations intersecting with the county, not the whole military base)

- Drop unused variables in the original datasets. We kept about 2 - 4 variables for each that might contribute meaningfully to our scoring framework later on

- Feature engineering: we created new variables based on the existing ones, such as **number of nuclear plants in each county**, **proportion of a county that has high flood risks**, etc.

- Merging all datasets into a final one that contains all necessary variables on a county level. All of our merging steps go in `notebooks/merge.ipynb`.


### Key Variables
Our final dataset `processed_data/final_dataset.csv` has **28 columns** and **3235 observations** corresponding to 3235 counties named by [Census.gov](Census.gov). Some key variables (not all of them): 

| Variable Name | Description | Unit |
| :-- | :-- | :-- |
| `population` | County population in 2025 | people |
| `total_energy_consumption_mwh` | County total consumption in 2024 | $MWh$ |
| `pct_sfha` | Proportion of the county marked as severe flood hazard area |
| `total_lake_area` | Total lake areas in the county | ${km^2}$ |
| `dist_to_lakes_km` | Distance to the county's centroid to the nearest lake | $km$ |
| `dist_to_rivers_km` | Distance to the county's centroid to the nearest river | $km$ |
| `pct_military` | Proportion of the county with military installations |
| `pga_max` | Maximum peak ground acceleration of the county (seismic risk indicator) | $g$ (fraction of gravitational acceleration, 9.8 $m/s^2$)
| `max_voltage` |  Maximum voltage of a transmission line in the county | $kV$ |


## Methodology 
### Exploratory Data Analysis
Our main notebook for EDA is `notebooks/eda/eda.ipynb`. Some of the EDA techniques we used were: 
- Explore the correlation between all numerical variables with a correlation heatmap. We found high correlation between 
    - `housing_units` and `median_household_income`
    - `distance_to_lines_km` and `distance_to_rivers_km`
    - `total_rivers_mile` and `rivers_count`
    - `total_energy_consumption_mwh` and `population`

- Explore the distributions of some numerical variables using boxplots

- Compare counties with and without a nuclear reactor to see the differences. Some key findings: 
    - Counties with nuclear plants consistently have lower seismic and flood risk, stronger access to rivers and water bodies
    - Counties with plants tend to cluster around lower population density but consume more energy in general compared to counties without one.
    - Counties with plants have an average maximum voltage around 300 - 500kV with a moderate number of lines, which suggests that higher-voltage grid access is preferred rather than dense grid with lower voltage. 

### Modeling
Our modeling notebooks go in `notebooks/modeling`. The modeling process follows this order:
1. `mask.ipynb`
2. `scoring.ipynb`
3. `logistic_regression.ipynb`
4. `validation.ipynb`
5. `moo.ipynb`
#### Step 1: Masking - `mask.ipynb`
We start by hard-excluding any counties that do not meet the minimum safety regulations to be a nuclear reactor site. We exclude a county if:
- `pga_max` exceeds 0.3 
- `pct_sfha` exceeds 0.2 
- `pct_military` exceeds 0.1 
- `pct_protected` exceeds 0.1 

All of these thresholds are based on regulatory guidelines of the [Nuclear Regulatory Commission](https://www.nrc.gov/docs/ml1218/ml12188a053.pdf), [Next Generation Nuclear Plant (NGNP)](https://inldigitallibrary.inl.gov/sites/sti/sti/5144360.pdf), and [Executive Orders from the Office of the Federal Register](https://www.archives.gov/federal-register/codification/executive-order/11988.html)

Our dataset was reduced from 3235 to **2161 counties** after this hard exclusion.
#### Step 2: Scoring - `scoring.ipynb`
Our scoring framework uses the [Multi-Criteria Decision Analysis (MCDA)](https://www.1000minds.com/decision-making/what-is-mcdm-mcda) technique, assigning weights to 12 variables to compute a suitability score per county.

Since most existing plants were built in the 70-80s under outdated criteria, we want to define a modern framework driven by expert knowledge and the NRC guidelines. We talked with professor [Caleb Brooks](https://npre.illinois.edu/people/profile/csbrooks) from Grainger Engineering to figure out the variables' importance

We ranked our variables by importance - top 3 being `pga_max`, `pct_sfha`, and `population_density` (all safety criteria), and applied the [Rank Order Centroid (ROC)](https://www.nature.com/articles/s41598-024-61945-z) method to convert our discrete ranking into weights. 

To validate robustness, we ran a **sensitivity analysis** with random ±20% weight perturbations across 1,000 simulations to see if our ranking remains stable. On average, **18.8 counties** in the top 20 remain the same, with a **95% CI of [18.744, 18.847]**. 

This confirms that our recommendations are reliable even if the weights aren't perfect. 

#### Step 3: Validation and Comparison
When ranking our MCDA scores, the highest-scoring counties are ones that currently don't have any nuclear reactors. We expected this to happen, as the old criteria might not match our modern scoring frameworks

We want to investigate this difference to see how policies have changed, by fitting **supervised ML models** on our dataset (which consist of "old" plants) and extracting feature importance to see priorities in the past. 

Key results:


#### Step 4: Recommendations
