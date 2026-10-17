# Forest Fire Burned-Area Analysis (Montesinho Natural Park)

Exploratory analysis of forest fire data from Portugal's Montesinho Natural Park,
built as a portfolio project for CISC7204 Data Visualization.

## Problem / Research Question

When and under what weather conditions do the most damaging forest fires occur in
Montesinho Natural Park, and how well do the Fire Weather Index (FWI) components
relate to the size of the burned area?

## Dataset

- **Source:** [UCI Machine Learning Repository - Forest Fires](https://archive.ics.uci.edu/dataset/162/forest+fires)
  (Cortez & Morais, 2007), 517 records from Montesinho Natural Park, Portugal
  (2000-2003).
- **Usage permissions:** Licensed under Creative Commons Attribution 4.0
  International (CC BY 4.0); free to share and adapt with attribution.
- **File:** `forestfires.csv` (included in this repository).

## Methods and Tools

- Python 3 with `pandas` for data wrangling and aggregation.
- `matplotlib` for visualization; `numpy` for the log-transform of the
  zero-inflated `area` target (`log1p`).
- Analysis steps: monthly aggregation of burned area and fire counts,
  weather-vs-area scatter analysis, and a correlation study of FWI indices
  (FFMC, DMC, DC, ISI) and weather variables against `log(area)`.

## Key Findings

1. **Fire damage is concentrated in late summer.** August and September account
   for ~5,384 ha of the ~6,639 ha total burned area, and also have the highest
   fire counts (see `figures/monthly_burned_area.png`).
2. **Large fires cluster under hot, dry conditions.** The biggest burns occur at
   high temperature with low relative humidity, though small fires happen in all
   conditions (see `figures/temp_vs_area.png`).
3. **The burned-area distribution is extremely skewed.** Mean = 12.8 ha,
   median = 0.5 ha, max = 1,090.8 ha; nearly half of all fires burn under 1 ha.

## Limitation

Individual FWI indices and weather variables correlate only weakly with
`log(area)` (all |r| < 0.1 in `figures/correlation_heatmap.png`), and the target
is zero-inflated. A simple linear model on these features would predict burned
area poorly; non-linear models and spatial/temporal factors would be needed.

## How to Run

```bash
pip install -r requirements.txt
python analysis.py
```

The script prints summary statistics and regenerates the three figures in
`figures/`.
