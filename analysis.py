"""Forest Fire Burned-Area Analysis - Montesinho Natural Park.

Reproducible analysis script for the data-science-project portfolio repo.
Reads forestfires.csv, prints summary findings, and writes three figures
into the figures/ directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).parent / "forestfires.csv"
FIG_DIR = Path(__file__).parent / "figures"

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["log_area"] = np.log1p(df["area"])
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)
    return df


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
    cols = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "log_area"]
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


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    df = load_data()
    print(f"Records: {len(df)}")
    print(f"Burned-area stats (ha): mean={df['area'].mean():.1f}, "
          f"median={df['area'].median():.1f}, max={df['area'].max():.1f}")
    top_months = df.groupby("month", observed=True)["area"].sum().nlargest(3)
    print("Top months by burned area:")
    print(top_months.to_string())
    figure_monthly_burned_area(df)
    figure_temp_vs_area(df)
    figure_correlation_heatmap(df)
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
