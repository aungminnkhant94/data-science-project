# Forest Fire Burned-Area Analysis (Montesinho Natural Park)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Forest%20Fires%20(CC%20BY%204.0)-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Can weather and fire-danger indices explain how much a forest fire burns?
> Spoiler: only up to a point - and the failure itself is the finding.**

End-to-end data science project on 517 forest fires in Portugal's Montesinho
Natural Park: exploratory analysis, publication-quality visualizations, and a
Random Forest benchmark. Built as a portfolio project for CISC7204 Data
Visualization (University of Macau).

## Table of Contents

- [Research Question](#research-question)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Model Benchmark](#model-benchmark)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [How to Run](#how-to-run)
- [References](#references)

## Research Question

When and under what weather conditions do the most damaging forest fires occur
in Montesinho Natural Park, and how well do the Fire Weather Index (FWI)
components (FFMC, DMC, DC, ISI) predict the burned area?

## Dataset

| | |
|---|---|
| **Source** | [UCI Machine Learning Repository - Forest Fires](https://archive.ics.uci.edu/dataset/162/forest+fires) (Cortez & Morais, 2007) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) - free to share and adapt with attribution |
| **Size** | 517 fires, 13 attributes, Montesinho Natural Park, Portugal, 2000-2003 |
| **Features** | FWI indices (FFMC, DMC, DC, ISI), weather (`temp`, `RH`, `wind`, `rain`), calendar (`month`, `day`), park grid coordinates (`X`, `Y`) |
| **Target** | `area` - burned area in hectares |
| **File** | `forestfires.csv` (included in this repository) |

## Project Structure

```
data-science-project/
|-- analysis.py              # full pipeline: EDA figures + model benchmark
|-- forestfires.csv          # dataset (UCI, CC BY 4.0)
|-- requirements.txt         # pinned dependencies
|-- figures/                 # all generated visualizations
|   |-- monthly_burned_area.png
|   |-- area_distribution.png
|   |-- temp_vs_area.png
|   |-- correlation_heatmap.png
|   |-- feature_importance.png
|   |-- predicted_vs_actual.png
|-- README.md
```

## Key Findings

### 1. Fire damage is concentrated in late summer

August and September account for ~5,384 ha of the ~6,639 ha total burned area -
over 80% of all damage in two months, matching the peak in fire counts.

![Monthly burned area](figures/monthly_burned_area.png)

### 2. The target is zero-inflated and extremely skewed

Nearly half of all fires (47.8%) burn less than 1 ha, while the largest single
fire burned 1,090.8 ha. Mean = 12.8 ha vs. median = 0.5 ha. Even after a
`log1p` transform the distribution stays right-skewed - a warning sign for any
regression model.

![Area distribution](figures/area_distribution.png)

### 3. Large fires cluster under hot, dry conditions

Fires burning more than ~100 ha occur almost exclusively above 20 deg C with
relative humidity below 45%.

![Temperature vs area](figures/temp_vs_area.png)

### 4. But no single variable explains fire size

All pairwise correlations between FWI indices / weather and `log(area)` are
weak (|r| < 0.1). Temperature correlates with the drought codes (as expected
physically), but not with the outcome itself.

![Correlation heatmap](figures/correlation_heatmap.png)

## Model Benchmark

A Random Forest regressor (`n_estimators=400`, `min_samples_leaf=3`) predicts
`log1p(area)` from the 4 FWI indices and 4 weather variables, evaluated on a
held-out 20% test split (`random_state=42`):

| Model | MAE (log-ha) | R2 |
|---|---|---|
| Baseline (predict train median) | 1.133 | 0.000 |
| Random Forest | 1.193 | -0.011 |

![Feature importance](figures/feature_importance.png)

![Predicted vs actual](figures/predicted_vs_actual.png)

**The honest result: the Random Forest cannot beat the median baseline.**
Feature importances confirm the EDA story - temperature, humidity and the
drought codes (DMC, DC) carry the most signal - but that signal is not enough
to predict how much a given fire will burn. Burned area appears to be driven
largely by factors *outside* this dataset: ignition cause, firefighting
response time, terrain, and fuel load. This mirrors the original paper, which
also found burned area very hard to predict (Cortez & Morais, 2007).

## Limitations

- **Weak predictive signal:** weather and FWI features alone are insufficient;
  the best model achieves R2 ~= 0 on held-out data.
- **Zero inflation:** 47.8% of records have `area = 0`, breaking the assumptions
  of plain regression; a hurdle/two-part model would be more appropriate.
- **Small, old, single-park dataset:** 517 records from one park (2000-2003)
  limit generalization to other regions or today's climate.

## Future Work

- Reformulate as a **two-part (hurdle) model**: first classify "will the fire
  burn > 1 ha?", then regress size only on positive cases.
- Try gradient boosting (XGBoost/LightGBM) with spatial features (`X`, `Y`
  grid coordinates) and month as a categorical feature.
- Add external data sources (firefighting response times, land cover) to test
  whether the missing signal is recoverable.

## How to Run

```bash
git clone https://github.com/aungminnkhant94/data-science-project.git
cd data-science-project
pip install -r requirements.txt
python analysis.py
```

The script prints summary statistics and model metrics, and regenerates all
six figures in `figures/`. Total runtime: under 15 seconds.

## References

- Cortez, P. & Morais, A. (2007). *A Data Mining Approach to Predict Forest
  Fires using Meteorological Data.* Proceedings of EPIA 2007.
- UCI Machine Learning Repository: Forest Fires dataset,
  https://archive.ics.uci.edu/dataset/162/forest+fires
