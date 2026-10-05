"""Reusable descriptive statistics and exploratory figures for Phase 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TRAJECTORY_COUNTRIES = ["JPN", "CHN", "KOR", "IND", "NGA"]
DESCRIPTIVE_VARIABLES = [
    "working_age_share", "working_age_change_10y", "old_age_dependency",
    "old_age_dependency_change_10y", "fertility", "real_gdp_pc_growth",
    "real_gdp_pc_growth_5y_forward", "investment_gdp", "inflation",
]


def descriptive_statistics(panel: pd.DataFrame) -> pd.DataFrame:
    statistics = panel[DESCRIPTIVE_VARIABLES].describe(percentiles=[0.25, 0.5, 0.75]).T
    statistics = statistics.rename(columns={"50%": "median"})
    return statistics[["count", "mean", "std", "min", "25%", "median", "75%", "max"]]


def _save_trajectory_figure(panel: pd.DataFrame, variable: str, ylabel: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for country in TRAJECTORY_COUNTRIES:
        subset = panel[panel["country_code"] == country].sort_values("year")
        axis.plot(subset["year"], subset[variable], linewidth=1.8, label=country)
    axis.set_xlabel("Year")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel}: selected countries")
    axis.grid(alpha=0.25)
    axis.legend(title="Country", ncol=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def trajectory_figures(panel: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_trajectory_figure(panel, "working_age_share", "Working-age population share (%)", output_dir / "working_age_trajectories.png")
    _save_trajectory_figure(panel, "old_age_dependency", "Old-age dependency ratio", output_dir / "old_age_dependency_trajectories.png")


def scatter_figure(panel: pd.DataFrame, predictor: str, output_path: Path, xlabel: str) -> None:
    sample = panel[[predictor, "real_gdp_pc_growth_5y_forward"]].dropna()
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(sample[predictor], sample["real_gdp_pc_growth_5y_forward"], alpha=0.35, s=18)
    if len(sample) >= 2:
        slope, intercept = __import__("numpy").polyfit(sample[predictor], sample["real_gdp_pc_growth_5y_forward"], 1)
        x_values = __import__("numpy").linspace(sample[predictor].min(), sample[predictor].max(), 100)
        axis.plot(x_values, intercept + slope * x_values, color="black", linewidth=1.5, label="Linear fit")
        axis.legend()
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Mean real GDP per-capita growth, next five years (%)")
    axis.set_title(f"{xlabel} and subsequent growth")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def make_country_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country_code, group in panel.groupby("country_code", sort=True):
        group = group.sort_values("year")
        first_working = group["working_age_share"].dropna()
        first_dependency = group["old_age_dependency"].dropna()
        rows.append({
            "country_code": country_code,
            "first_year": group["year"].min(),
            "last_year": group["year"].max(),
            "mean_real_gdp_pc_growth": group["real_gdp_pc_growth"].mean(),
            "mean_forward_5y_gdp_pc_growth": group["real_gdp_pc_growth_5y_forward"].mean(),
            "working_age_share_first": first_working.iloc[0] if len(first_working) else pd.NA,
            "working_age_share_last": first_working.iloc[-1] if len(first_working) else pd.NA,
            "working_age_share_total_change": first_working.iloc[-1] - first_working.iloc[0] if len(first_working) else pd.NA,
            "old_age_dependency_first": first_dependency.iloc[0] if len(first_dependency) else pd.NA,
            "old_age_dependency_last": first_dependency.iloc[-1] if len(first_dependency) else pd.NA,
            "old_age_dependency_total_change": first_dependency.iloc[-1] - first_dependency.iloc[0] if len(first_dependency) else pd.NA,
            "latest_working_age_change_10y": group["working_age_change_10y"].dropna().iloc[-1] if group["working_age_change_10y"].notna().any() else pd.NA,
            "latest_old_age_dependency_change_10y": group["old_age_dependency_change_10y"].dropna().iloc[-1] if group["old_age_dependency_change_10y"].notna().any() else pd.NA,
        })
    return pd.DataFrame(rows)