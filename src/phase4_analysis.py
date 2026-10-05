"""Phase 4 descriptive structural-financial analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .phase4 import phase4_coverage

DESCRIPTIVE_VARIABLES = [
    "government_debt_gdp", "fiscal_balance_gdp", "private_credit_gdp", "credit_gap",
    "debt_service_ratio", "dsr_deviation_10y", "real_house_price_growth_5y", "working_age_change_10y",
]
CORRELATION_VARIABLES = DESCRIPTIVE_VARIABLES[:2] + [
    "government_debt_change_5y", "private_credit_gdp", "private_credit_change_5y", "credit_gap",
    "dsr_deviation_10y", "real_house_price_growth_5y", "working_age_change_10y",
]
TRAJECTORY_COUNTRIES = ["JPN", "USA", "GBR", "CHN", "KOR", "DEU"]


def descriptive_statistics(panel: pd.DataFrame) -> pd.DataFrame:
    stats = panel[DESCRIPTIVE_VARIABLES].describe(percentiles=[0.25, 0.5, 0.75]).T.rename(columns={"50%": "median"})
    return stats[["count", "mean", "std", "min", "25%", "median", "75%", "max"]]


def correlations(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left in CORRELATION_VARIABLES:
        for right in CORRELATION_VARIABLES:
            if left >= right:
                continue
            sample = panel[[left, right]].dropna()
            rows.extend([
                {"left": left, "right": right, "method": "pearson", "value": sample[left].corr(sample[right]), "N": len(sample)},
                {"left": left, "right": right, "method": "spearman", "value": sample[left].corr(sample[right], method="spearman"), "N": len(sample)},
            ])
    return pd.DataFrame(rows)


def _future_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.sort_values(["country_code", "year"]).copy()
    for years, name in [(1, "future_1y_real_gdp_pc_growth")]:
        result[name] = pd.NA
        for _, group in result.groupby("country_code", sort=False):
            values = group.set_index("year")["real_gdp_pc_growth"]
            result.loc[group.index, name] = [values.get(year + years) for year in group["year"]]
    for window, name in [(3, "future_3y_real_gdp_pc_growth_avg"), (5, "future_5y_real_gdp_pc_growth_avg")]:
        result[name] = pd.NA
        for _, group in result.groupby("country_code", sort=False):
            values = group.set_index("year")["real_gdp_pc_growth"]
            output = []
            for year in group["year"]:
                window_values = [values.get(year + offset) for offset in range(1, window + 1)]
                output.append(sum(window_values) / window if all(pd.notna(value) for value in window_values) else pd.NA)
            result.loc[group.index, name] = pd.Series(output, index=group.index, dtype="Float64").to_numpy()
    return result


def _point_in_time_top_quintile(group: pd.DataFrame, column: str, minimum_history: int = 5) -> pd.Series:
    values = group[column].tolist()
    flags = []
    for index, value in enumerate(values):
        history = pd.Series(values[:index]).dropna()
        flags.append(bool(pd.notna(value) and len(history) >= minimum_history and value >= history.quantile(0.8)))
    return pd.Series(flags, index=group.index)


def add_point_in_time_flags(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.sort_values(["country_code", "year"]).copy()
    result["demographic_deterioration"] = False
    for _, group in result.groupby("country_code", sort=False):
        working = group["working_age_change_10y"].tolist()
        demographic = []
        for index, value in enumerate(working):
            history = pd.Series(working[:index]).dropna()
            demographic.append(bool(pd.notna(value) and len(history) >= 5 and value < history.median()))
        result.loc[group.index, "demographic_deterioration"] = demographic
        for variable, flag in [("credit_gap", "credit_gap_elevated"), ("dsr_deviation_10y", "dsr_elevated"), ("real_house_price_growth_5y", "property_elevated")]:
            result.loc[group.index, flag] = _point_in_time_top_quintile(group, variable).to_numpy()
    return result


def structural_quadrants(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specifications = [("demographic_fiscal", "government_debt_gdp", "debt"), ("demographic_credit", "private_credit_gdp", "credit"), ("demographic_credit_gap", "credit_gap", "credit_gap")]
    for comparison, financial, label in specifications:
        sample = panel[["country_code", "year", "working_age_change_10y", financial, "real_gdp_pc_growth_5y_forward", "inflation", "investment_gdp"]].dropna()
        demo_median, financial_median = sample["working_age_change_10y"].median(), sample[financial].median()
        sample["demographic_group"] = np.where(sample.working_age_change_10y >= demo_median, "above_median", "below_median")
        sample["financial_group"] = np.where(sample[financial] >= financial_median, "above_median", "below_median")
        for (demo_group, financial_group), group in sample.groupby(["demographic_group", "financial_group"]):
            rows.append({"comparison": comparison, "financial_variable": label, "demographic_group": demo_group, "financial_group": financial_group, "sample_median_demographic": demo_median, "sample_median_financial": financial_median, "N": len(group), "countries": group.country_code.nunique(), "first_year": group.year.min(), "last_year": group.year.max(), "mean_future_growth": group.real_gdp_pc_growth_5y_forward.mean(), "median_future_growth": group.real_gdp_pc_growth_5y_forward.median(), "mean_inflation": group.inflation.mean(), "mean_investment_gdp": group.investment_gdp.mean()})
    return pd.DataFrame(rows)


def vulnerability_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    data = add_point_in_time_flags(_future_outcomes(panel))
    rows = []
    for event, flag in [("credit_gap", "credit_gap_elevated"), ("dsr_deviation_10y", "dsr_elevated"), ("property", "property_elevated")]:
        for elevated, group in data.groupby(flag):
            sample = group.dropna(subset=["future_1y_real_gdp_pc_growth", "future_3y_real_gdp_pc_growth_avg", "future_5y_real_gdp_pc_growth_avg"])
            rows.append({"episode": event, "classification": "elevated" if elevated else "non_elevated", "N": len(sample), "countries": sample.country_code.nunique(), "first_year": sample.year.min(), "last_year": sample.year.max(), "mean_future_1y_growth": sample.future_1y_real_gdp_pc_growth.mean(), "mean_future_3y_growth": sample.future_3y_real_gdp_pc_growth_avg.mean(), "mean_future_5y_growth": sample.future_5y_real_gdp_pc_growth_avg.mean()})
    return pd.DataFrame(rows)


def cooccurrence(panel: pd.DataFrame) -> pd.DataFrame:
    data = add_point_in_time_flags(_future_outcomes(panel))
    rows = []
    for financial_flag, label in [("credit_gap_elevated", "credit_gap"), ("dsr_elevated", "dsr_deviation_10y")]:
        sample = data.dropna(subset=["working_age_change_10y", "future_1y_real_gdp_pc_growth", "future_3y_real_gdp_pc_growth_avg", "future_5y_real_gdp_pc_growth_avg", "inflation"]).copy()
        financial = sample[financial_flag].fillna(False).astype(bool)
        demographic = sample["demographic_deterioration"].fillna(False).astype(bool)
        sample["category"] = np.select([demographic & financial, demographic, financial], ["both", "demographic_only", "financial_only"], default="neither")
        for category, group in sample.groupby("category"):
            rows.append({"financial_condition": label, "category": category, "N": len(group), "countries": group.country_code.nunique(), "first_year": group.year.min(), "last_year": group.year.max(), "mean_future_1y_growth": group.future_1y_real_gdp_pc_growth.mean(), "mean_future_3y_growth": group.future_3y_real_gdp_pc_growth_avg.mean(), "mean_future_5y_growth": group.future_5y_real_gdp_pc_growth_avg.mean(), "mean_inflation": group.inflation.mean()})
    return pd.DataFrame(rows)


def trajectory_figures(panel: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for variable, filename, ylabel in [("government_debt_gdp", "government_debt_trajectories.png", "Government debt (% GDP)"), ("private_credit_gdp", "private_credit_trajectories.png", "Private credit (% GDP)"), ("credit_gap", "credit_gap_trajectories.png", "Credit gap (percentage points)"), ("dsr_deviation_10y", "dsr_deviation_trajectories.png", "DSR deviation from prior 10-year mean")]:
        figure, axis = plt.subplots(figsize=(9, 5.5))
        for country in TRAJECTORY_COUNTRIES:
            subset = panel[panel.country_code == country].sort_values("year")
            axis.plot(subset.year, subset[variable], label=country, linewidth=1.7)
        axis.set_title(ylabel)
        axis.set_xlabel("Year")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)


def run_phase4(panel_path: Path, tables_dir: Path, figures_dir: Path, coverage_dir: Path) -> dict[str, pd.DataFrame]:
    panel = pd.read_csv(panel_path)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)
    coverage = phase4_coverage(panel)
    coverage.to_csv(coverage_dir / "phase4_source_coverage.csv", index=False)
    country_rows = coverage[coverage["country_code"] != "ALL"]
    country_summary = country_rows.groupby("country_code", as_index=False).agg(variable_count=("variable", "count"), available_variables=("non_null_observations", lambda values: int((values > 0).sum())), mean_coverage_pct=("coverage_pct", "mean"))
    country_summary.to_csv(coverage_dir / "phase4_country_summary.csv", index=False)
    descriptive_statistics(panel).to_csv(tables_dir / "phase4_descriptive_statistics.csv")
    correlations(panel).to_csv(tables_dir / "phase4_correlations.csv", index=False)
    structural_quadrants(panel).to_csv(tables_dir / "phase4_structural_quadrants.csv", index=False)
    vulnerability_episodes(panel).to_csv(tables_dir / "phase4_financial_vulnerability_episodes.csv", index=False)
    cooccurrence(panel).to_csv(tables_dir / "phase4_demographic_financial_cooccurrence.csv", index=False)
    trajectory_figures(panel, figures_dir)
    return {"panel": panel, "coverage": coverage, "quadrants": structural_quadrants(panel), "episodes": vulnerability_episodes(panel), "cooccurrence": cooccurrence(panel)}


if __name__ == "__main__":
    from .config import PROJECT_ROOT, STRUCTURAL_FINANCIAL_PANEL_FILE

    run_phase4(STRUCTURAL_FINANCIAL_PANEL_FILE, PROJECT_ROOT / "outputs" / "tables", PROJECT_ROOT / "outputs" / "figures", PROJECT_ROOT / "outputs" / "coverage")
    print("Phase 4 analysis outputs generated")