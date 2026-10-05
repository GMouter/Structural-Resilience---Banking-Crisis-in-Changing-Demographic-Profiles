"""Phase 2 exploratory panel analysis helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

OUTCOME = "real_gdp_pc_growth_5y_forward"
MODEL_PREDICTORS = {
    "Model 1": ["working_age_change_10y", "old_age_dependency_change_10y"],
    "Model 2": ["working_age_change_10y", "old_age_dependency_change_10y", "investment_gdp_5y_backward_avg", "inflation_5y_backward_avg"],
    "Model 3A": ["working_age_change_10y", "investment_gdp_5y_backward_avg", "inflation_5y_backward_avg"],
    "Model 3B": ["old_age_dependency_change_10y", "investment_gdp_5y_backward_avg", "inflation_5y_backward_avg"],
}


def complete_case_data(panel: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    if any("forward" in column for column in predictors):
        raise ValueError("Forward variables may only be used as the dependent outcome")
    required = ["country_code", "year", OUTCOME, *predictors]
    return panel.dropna(subset=required).copy()


def fit_fixed_effects_regression(panel: pd.DataFrame, model_name: str, predictors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = complete_case_data(panel, predictors)
    if data["country_code"].nunique() < 2:
        raise ValueError("At least two country clusters are required")
    country_effects = pd.get_dummies(data["country_code"], prefix="country", drop_first=True, dtype=float)
    year_effects = pd.get_dummies(data["year"], prefix="year", drop_first=True, dtype=float)
    design = pd.concat([data[predictors].astype(float), country_effects, year_effects], axis=1)
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(data[OUTCOME].astype(float), design).fit(cov_type="cluster", cov_kwds={"groups": data["country_code"]})
    confidence = model.conf_int()
    country_means = data.groupby("country_code")[OUTCOME].transform("mean")
    within_r_squared = 1 - ((data[OUTCOME] - model.fittedvalues) ** 2).sum() / ((data[OUTCOME] - country_means) ** 2).sum()
    metadata = {
        "N": len(data), "number_of_countries": data["country_code"].nunique(), "r_squared": model.rsquared,
        "within_r_squared": within_r_squared, "fixed_effects": "country, year", "se_method": "country-clustered",
    }
    rows = []
    for predictor in predictors:
        rows.append({"model": model_name, "term": predictor, "estimate": model.params[predictor], "clustered_se": model.bse[predictor], "t_stat": model.tvalues[predictor], "p_value": model.pvalues[predictor], "ci_lower": confidence.loc[predictor, 0], "ci_upper": confidence.loc[predictor, 1], **metadata})
    summary = pd.DataFrame([{ "model": model_name, "term": "__model__", **{key: value for key, value in metadata.items()}, "estimate": np.nan, "clustered_se": np.nan, "t_stat": np.nan, "p_value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan }])
    return pd.DataFrame(rows), summary


def correlation_table(panel: pd.DataFrame) -> pd.DataFrame:
    pairs = [("working_age_change_10y", "real_gdp_pc_growth_5y_forward"), ("old_age_dependency_change_10y", "real_gdp_pc_growth_5y_forward")]
    rows = []
    for predictor, outcome in pairs:
        sample = panel[[predictor, outcome]].dropna()
        rows.extend([
            {"relationship": f"{predictor} vs {outcome}", "method": "pearson", "value": sample[predictor].corr(sample[outcome], method="pearson"), "N": len(sample)},
            {"relationship": f"{predictor} vs {outcome}", "method": "spearman", "value": sample[predictor].corr(sample[outcome], method="spearman"), "N": len(sample)},
        ])
    return pd.DataFrame(rows)


def decade_summary(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["decade"] = (data["year"] // 10 * 10).astype(str) + "s"
    return data.groupby("decade", sort=False).agg(
        mean_working_age_change_10y=("working_age_change_10y", "mean"),
        mean_old_age_dependency_change_10y=("old_age_dependency_change_10y", "mean"),
        mean_real_gdp_pc_growth=("real_gdp_pc_growth", "mean"),
        mean_real_gdp_pc_growth_5y_forward=(OUTCOME, "mean"),
        observation_count=("country_code", "count"),
    ).reindex([f"{year}s" for year in range(1960, 2030, 10)]).reset_index()


def multicollinearity_table(panel: pd.DataFrame) -> pd.DataFrame:
    predictors = MODEL_PREDICTORS["Model 2"]
    sample = panel[predictors].dropna()
    rows = [{"metric": "predictor_correlation", "variable": "working_age_change_10y__old_age_dependency_change_10y", "value": sample[predictors[0]].corr(sample[predictors[1]])}]
    matrix = sm.add_constant(sample.astype(float))
    for index, variable in enumerate(matrix.columns[1:], start=1):
        rows.append({"metric": "vif_model_2", "variable": variable, "value": variance_inflation_factor(matrix.values, index)})
    return pd.DataFrame(rows)


def nonoverlap_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep calendar years divisible by five, a deterministic one-in-five rule."""
    return panel[panel["year"].astype(int).mod(5).eq(0)].copy()


def run_regressions(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_tables = []
    summary_tables = []
    for model_name, predictors in MODEL_PREDICTORS.items():
        coefficients, summary = fit_fixed_effects_regression(panel, model_name, predictors)
        coefficient_tables.append(coefficients)
        summary_tables.append(summary)
    return pd.concat(coefficient_tables + summary_tables, ignore_index=True), pd.concat(summary_tables, ignore_index=True)


def run_phase2(panel_path: Path, tables_dir: Path, figures_dir: Path) -> dict[str, pd.DataFrame]:
    from .eda import descriptive_statistics, make_country_summary, scatter_figure, trajectory_figures

    panel = pd.read_csv(panel_path)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    descriptive_statistics(panel).to_csv(tables_dir / "phase2_descriptive_statistics.csv")
    trajectory_figures(panel, figures_dir)
    scatter_figure(panel, "working_age_change_10y", figures_dir / "working_age_vs_future_growth.png", "10-year change in working-age share")
    scatter_figure(panel, "old_age_dependency_change_10y", figures_dir / "dependency_vs_future_growth.png", "10-year change in old-age dependency")
    correlations = correlation_table(panel)
    correlations.to_csv(tables_dir / "demographic_growth_correlations.csv", index=False)
    decades = decade_summary(panel)
    decades.to_csv(tables_dir / "decade_summary.csv", index=False)
    country_summary = make_country_summary(panel)
    country_summary.to_csv(tables_dir / "country_demographic_growth_summary.csv", index=False)
    multicollinearity = multicollinearity_table(panel)
    multicollinearity.to_csv(tables_dir / "phase2_multicollinearity.csv", index=False)
    regression_results, _ = run_regressions(panel)
    regression_results.to_csv(tables_dir / "phase2_regression_results.csv", index=False)
    sensitivity_rows = []
    reduced = nonoverlap_sample(panel)
    for model_name in ["Model 3A", "Model 3B"]:
        coefficients, summary = fit_fixed_effects_regression(reduced, model_name, MODEL_PREDICTORS[model_name])
        sensitivity_rows.append(pd.concat([coefficients, summary], ignore_index=True))
    sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
    sensitivity.to_csv(tables_dir / "phase2_nonoverlap_sensitivity.csv", index=False)
    return {"panel": panel, "correlations": correlations, "regressions": regression_results, "sensitivity": sensitivity}


if __name__ == "__main__":
    from .config import DERIVED_PANEL_FILE, PROJECT_ROOT

    run_phase2(DERIVED_PANEL_FILE, PROJECT_ROOT / "outputs" / "tables", PROJECT_ROOT / "outputs" / "figures")
    print("Phase 2 analysis outputs generated")