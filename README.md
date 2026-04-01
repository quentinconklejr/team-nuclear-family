## Team Nuclear Family

# Nuclear Reactor Siting in the United States

AI data centers are consuming more electricity than ever, and it is onlt increasing with new data centers being built often. Nuclear is one of the few clean energy sources that can actually keep up. But where should new reactors go?

That’s what this project answers.



## What we’re doing

We’re building a geospatial framework that scores every realistic location in the US based on two things: how suitable it is to physically build a reactor there, and how much energy demand exists in that area because nuclear energy can only be used locally. The output is a ranked map of optimal reactor sites framed as an actual policy recommendation.

## Methodology

**Step 1: Land suitability mask**

Before scoring anything, we filter out land that’s never going to work. National parks, military zones, wetlands, floodplains, high seismic risk areas, dense urban cores. Whatever’s left is the eligible candidate set.

**Step 2: Suitability scoring**

Each eligible location gets scored on the factors the NRC actually uses to evaluate reactor sites: seismic stability, distance to cooling water, proximity to transmission infrastructure, and population density in the surrounding area.

**Step 3: Demand scoring**

Each location also gets scored on how much energy demand exists nearby, using county level consumption data and data center concentration as a proxy for AI infrastructure load.

**Step 4: Composite ranking**

Suitability and demand scores are combined into a final score. The map shows the top ranked sites across the country.

**Step 5: Validation (if we have time)**

We run existing NRC licensed reactor sites through our framework to see how they score. If they score well, great. If they don’t, that’s an interesting finding too since most were built in the 70s and 80s under very different data and standards.



## Data we are using 

All data is county level or finer and covers the contiguous US.

|File                           |What it is                                       |
|-------------------------------|-------------------------------------------------|
|`nuclear_plants_cleaned.csv`   |Existing NRC licensed reactor locations          |
|`seismic_hazard.csv`           |USGS ASCE 7-22 peak ground acceleration by county|
|`transmission_lines_coords.csv`|HIFLD electric power transmission lines          |
|`demand_counties.csv`          |EIA energy consumption by county                 |
|`data_centers.csv`             |US data center locations (AI demand proxy)       |
|`population_2024.csv`          |Census 2024 population by county                 |
|`county_boundaries.geojson`    |Census TIGER county boundaries                   |
|`protected_areas.csv`          |USGS PAD-US 4.0 protected areas for land mask    |
|`military_boundaries.geojson`  |Military zone boundaries for land mask           |
|`fema_claims_county.csv`       |FEMA NFIP flood claims by county                 |
|`fema_community_status.csv`    |FEMA community flood status                      |
|`flood_by_counties.csv`        |Flood risk by county                             |
|`rivers_usa.geojson`           |US river network (cooling water access)          |
|`lakes_usa.geojson`            |US lakes (cooling water access)                  |
|`lakes_by_county.csv`          |Lake coverage by county                          |
|`wetlands_cleaned.csv`         |Wetland polygons for land mask                   |
|`land_use.csv`                 |Land use proxy by county                         |
|`census_income_housing.csv`    |Census ACS income and housing data               |
|`decommissioned_plants.csv`    |Decommissioned nuclear plant locations           |
|`wiki_decommissioned_coal.csv` |Decommissioned coal plant locations              |

The seismic hazard data tells us where the ground is stable enough to build on. Rivers, lakes, and water coverage tell us where reactors can actually get the cooling water they need to operate. Transmission lines tell us where grid infrastructure already exists so a new reactor can connect without needing to build out entirely new power lines. Population data tells us two things where demand is high, and where we need "buffer" zones since you can’t put a reactor in someone’s backyard or in the middle of a big city like in downtown Chicago. The protected areas, military boundaries, wetlands, and flood data are all about ruling places out.
The demand and data center files are about finding where the energy is actually needed, since reactors only serve the local area. The decommissioned nuclear and coal plant locations are interesting because those sites already have transmission hookups, cooling infrastructure, and community familiarity with industrial energy facilities, so they might actually be strong candidate locations even if the original plant is gone. 

**limitation:** County energy consumption data is from 2016, we could not find anything newer.



