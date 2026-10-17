"""Forest Fire Burned-Area Analysis - Montesinho Natural Park.

Reproducible end-to-end pipeline for the data-science-project portfolio repo:
exploratory analysis, visualization, and a Random Forest benchmark that
predicts log-transformed burned area from Fire Weather Index (FWI) components
and weather variables.

Reads forestfires.csv, prints summary findings and model metrics, and writes
all figures into the figures/ directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = Path(__file__).parent / "forestfires.csv"
FIG_DIR = Path(__file__).parent / "figures"

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
FWI_COLS = ["FFMC", "DMC", "DC", "ISI"]
WEATHER_COLS = ["temp", "RH", "wind", "rain"]
FEATURE_COLS = FWI_COLS + WEATHER_COLS


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["log_area"] = np.log1p(df["area"])
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)
    return df


# ---------------------------------------------------------------------------
# Exploratory figures
# ---------------------------------------------------------------------------

def figure_monthly_burned_area(df: pd.DataFrame) -> None:
    monthly = df.groupby("month", observed=True)["area"].agg(["sum", "count"])
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(monthly.index.astype(str), monthly["sum"], color="#d95f02", alpha=0.85)
    ax1.set_ylabel("Total burned area (ha)")
    ax1.set_xlabel("Month")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index.astype(str), monthly["count"], color="#1b9e77",
             marker="o", linewidth=2, label="Number of fires")
    ax2.set_ylabel("Number of fires")
    ax1.set_title("Burned area and fire count peak in late summer (Aug-Sep)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_burned_area.png", dpi=150)
    plt.close(fig)


def figure_area_distribution(df: pd.DataFrame) -> None:
    zero_share = (df["area"] == 0).mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(df["area"], bins=60, color="#7570b3", edgecolor="white")
    axes[0].set_xlabel("Burned area (ha)")
    axes[0].set_ylabel("Number of fires")
    axes[0].set_title(f"Raw area: {zero_share:.0%} of fires burn < 1 ha")
    axes[1].hist(df["log_area"], bins=40, color="#1b9e77", edgecolor="white")
    axes[1].set_xlabel("log(1 + burned area)")
    axes[1].set_ylabel("Number of fires")
    axes[1].set_title("After log1p transform: still right-skewed")
    fig.suptitle("The target variable is zero-inflated and extremely skewed",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "area_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_temp_vs_area(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(df["temp"], df["area"], c=df["RH"], cmap="coolwarm_r",
                    alpha=0.65, edgecolors="none", s=28)
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("Burned area (ha, log scale)")
    ax.set_title("Larger fires cluster at high temperature and low humidity")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Relative humidity (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "temp_vs_area.png", dpi=150)
    plt.close(fig)


def figure_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = FWI_COLS + WEATHER_COLS + ["log_area"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("Correlation of FWI indices and weather with log(area)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model benchmark
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df["log_area"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    baseline_pred = np.full_like(y_test, y_train.median(), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    model = RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                  random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "baseline_mae": baseline_mae,
        "rf_mae": mean_absolute_error(y_test, pred),
        "rf_r2": r2_score(y_test, pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return model, X_test, y_test, pred, metrics


def figure_feature_importance(model) -> None:
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d95f02" if c in FWI_COLS else "#1b9e77" for c in imp.index]
    ax.barh(imp.index, imp.values, color=colors, alpha=0.85)
    ax.set_xlabel("Mean decrease in impurity (feature importance)")
    ax.set_title("Random Forest: temperature and drought codes matter most")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#d95f02", label="FWI index"),
                       Patch(color="#1b9e77", label="Weather")],
              loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def figure_predicted_vs_actual(y_test, pred, metrics) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, pred, alpha=0.6, color="#7570b3", edgecolors="none", s=30)
    lim = max(y_test.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=1)
    ax.set_xlabel("Actual log(1 + area)")
    ax.set_ylabel("Predicted log(1 + area)")
    ax.set_title(f"Predicted vs actual (test set, R2 = {metrics['rf_r2']:.2f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    df = load_data()

    print(f"Records: {len(df)}")
    print(f"Burned-area stats (ha): mean={df['area'].mean():.1f}, "
          f"median={df['area'].median():.1f}, max={df['area'].max():.1f}")
    print(f"Fires burning < 1 ha: {(df['area'] == 0).mean():.1%} of records")
    top_months = df.groupby("month", observed=True)["area"].sum().nlargest(3)
    print("Top months by burned area:")
    print(top_months.to_string())

    figure_monthly_burned_area(df)
    figure_area_distribution(df)
    figure_temp_vs_area(df)
    figure_correlation_heatmap(df)

    model, X_test, y_test, pred, metrics = train_model(df)
    print("\nModel benchmark (RandomForestRegressor, target = log1p(area)):")
    print(f"  train/test split: {metrics['n_train']}/{metrics['n_test']}")
    print(f"  baseline MAE (predict train median): {metrics['baseline_mae']:.3f}")
    print(f"  Random Forest MAE:                   {metrics['rf_mae']:.3f}")
    print(f"  Random Forest R^2:                   {metrics['rf_r2']:.3f}")
    figure_feature_importance(model)
    figure_predicted_vs_actual(y_test, pred, metrics)

    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
