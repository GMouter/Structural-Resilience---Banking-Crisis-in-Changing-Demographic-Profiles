"""Phase 3 exploratory panel models and descriptive outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .phase3 import phase3_coverage

OUTCOME = "real_gdp_pc_growth_5y_forward"
WORKING_AGE = "working_age_change_10y"
PRODUCTIVITY = "productivity_growth_5y_backward"
LABOUR = "labour_force_participation_5y_backward_avg"
BASE_CONTROLS = ["investment_gdp_5y_backward_avg", "inflation_5y_backward_avg"]
MODEL_PREDICTORS = {
    "P3-0": [WORKING_AGE, *BASE_CONTROLS],
    "P3-1": [WORKING_AGE, "log_real_gdp_pc", *BASE_CONTROLS],
    "P3-2": [WORKING_AGE, "log_real_gdp_pc", *BASE_CONTROLS, LABOUR],
    "P3-3": [WORKING_AGE, PRODUCTIVITY, "log_real_gdp_pc", *BASE_CONTROLS, LABOUR],
}


def complete_case_data(panel: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    if any("forward" in column for column in predictors):
        raise ValueError("Forward variables may only be used as the dependent outcome")
    return panel.dropna(subset=["country_code", "year", OUTCOME, *predictors]).copy()


def add_interaction(panel: pd.DataFrame, sample: pd.DataFrame | None = None) -> pd.DataFrame:
    result = panel.copy()
    reference = sample if sample is not None else result.dropna(subset=[WORKING_AGE, PRODUCTIVITY])
    working_mean = reference[WORKING_AGE].mean()
    productivity_mean = reference[PRODUCTIVITY].mean()
    result["centred_working_age_change_10y"] = result[WORKING_AGE] - working_mean
    result["centred_productivity_growth_5y_backward"] = result[PRODUCTIVITY] - productivity_mean
    result["demographic_productivity_interaction"] = result["centred_working_age_change_10y"] * result["centred_productivity_growth_5y_backward"]
    return result


def _design(data: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    country_effects = pd.get_dummies(data["country_code"], prefix="country", drop_first=True, dtype=float)
    year_effects = pd.get_dummies(data["year"], prefix="year", drop_first=True, dtype=float)
    return sm.add_constant(pd.concat([data[predictors].astype(float), country_effects, year_effects], axis=1), has_constant="add")


def _fit(data: pd.DataFrame, model_name: str, predictors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, object, pd.DataFrame]:
    data = complete_case_data(data, predictors)
    if data["country_code"].nunique() < 2:
        raise ValueError("At least two country clusters are required")
    design = _design(data, predictors)
    model = sm.OLS(data[OUTCOME].astype(float), design).fit(cov_type="cluster", cov_kwds={"groups": data["country_code"]})
    confidence = model.conf_int()
    country_means = data.groupby("country_code")[OUTCOME].transform("mean")
    denominator = ((data[OUTCOME] - country_means) ** 2).sum()
    within_r_squared = 1 - ((data[OUTCOME] - model.fittedvalues) ** 2).sum() / denominator if denominator else np.nan
    metadata = {"N": len(data), "countries": data["country_code"].nunique(), "r_squared": model.rsquared, "within_r_squared": within_r_squared, "fixed_effects": "country, year", "se_method": "country-clustered"}
    rows = []
    for predictor in predictors:
        rows.append({"model": model_name, "term": predictor, "estimate": model.params[predictor], "clustered_se": model.bse[predictor], "t_stat": model.tvalues[predictor], "p_value": model.pvalues[predictor], "ci_lower": confidence.loc[predictor, 0], "ci_upper": confidence.loc[predictor, 1], **metadata})
    summary = pd.DataFrame([{ "model": model_name, "term": "__model__", "estimate": np.nan, "clustered_se": np.nan, "t_stat": np.nan, "p_value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, **metadata}])
    return pd.DataFrame(rows), summary, model, data


def fit_phase3_model(panel: pd.DataFrame, model_name: str, predictors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients, summary, _, _ = _fit(panel, model_name, predictors)
    return coefficients, summary


def _model_on_sample(sample: pd.DataFrame, model_name: str, predictors: list[str]) -> pd.DataFrame:
    coefficients, summary, _, _ = _fit(sample, model_name, predictors)
    return pd.concat([coefficients, summary], ignore_index=True)


def regression_results(panel: pd.DataFrame) -> pd.DataFrame:
    results = []
    for model_name, predictors in MODEL_PREDICTORS.items():
        results.append(_model_on_sample(panel, model_name, predictors))
    interaction_sample = complete_case_data(panel, MODEL_PREDICTORS["P3-3"])
    interaction_panel = add_interaction(panel, interaction_sample)
    interaction_predictors = [WORKING_AGE, PRODUCTIVITY, "demographic_productivity_interaction", "log_real_gdp_pc", *BASE_CONTROLS, LABOUR]
    results.append(_model_on_sample(interaction_panel, "P3-4", interaction_predictors))
    return pd.concat(results, ignore_index=True)


def marginal_productivity_effects(panel: pd.DataFrame) -> pd.DataFrame:
    sample = complete_case_data(panel, MODEL_PREDICTORS["P3-3"])
    interaction_panel = add_interaction(panel, sample)
    predictors = [WORKING_AGE, PRODUCTIVITY, "demographic_productivity_interaction", "log_real_gdp_pc", *BASE_CONTROLS, LABOUR]
    _, _, model, data = _fit(interaction_panel, "P3-4", predictors)
    mean_productivity = sample[PRODUCTIVITY].mean()
    covariance = model.cov_params()
    levels = [("low", sample[PRODUCTIVITY].quantile(0.25)), ("median", sample[PRODUCTIVITY].quantile(0.5)), ("high", sample[PRODUCTIVITY].quantile(0.75))]
    rows = []
    for label, level in levels:
        effect = model.params[WORKING_AGE] + model.params["demographic_productivity_interaction"] * (level - mean_productivity)
        variance = covariance.loc[WORKING_AGE, WORKING_AGE] + (level - mean_productivity) ** 2 * covariance.loc["demographic_productivity_interaction", "demographic_productivity_interaction"] + 2 * (level - mean_productivity) * covariance.loc[WORKING_AGE, "demographic_productivity_interaction"]
        standard_error = np.sqrt(max(variance, 0))
        rows.append({"productivity_level": label, "productivity_value": level, "marginal_working_age_association": effect, "clustered_se": standard_error, "ci_lower": effect - 1.96 * standard_error, "ci_upper": effect + 1.96 * standard_error, "N": len(data), "countries": data["country_code"].nunique()})
    return pd.DataFrame(rows)


def common_sample_results(panel: pd.DataFrame) -> pd.DataFrame:
    common = complete_case_data(panel, MODEL_PREDICTORS["P3-3"])
    return pd.concat([_model_on_sample(common, name, predictors) for name, predictors in MODEL_PREDICTORS.items()], ignore_index=True)


def nonoverlap_sample(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["year"].astype(int).mod(5).eq(0)].copy()


def nonoverlap_results(panel: pd.DataFrame) -> pd.DataFrame:
    reduced = nonoverlap_sample(panel)
    sample = complete_case_data(reduced, MODEL_PREDICTORS["P3-3"])
    interaction_panel = add_interaction(reduced, sample)
    interaction_predictors = [WORKING_AGE, PRODUCTIVITY, "demographic_productivity_interaction", "log_real_gdp_pc", *BASE_CONTROLS, LABOUR]
    return pd.concat([_model_on_sample(reduced, "P3-3", MODEL_PREDICTORS["P3-3"]), _model_on_sample(interaction_panel, "P3-4", interaction_predictors)], ignore_index=True)


def multicollinearity(panel: pd.DataFrame) -> pd.DataFrame:
    variables = MODEL_PREDICTORS["P3-3"]
    sample = panel[variables].dropna()
    rows = []
    for left in variables:
        for right in variables:
            if left < right:
                rows.append({"metric": "correlation", "variable": f"{left}__{right}", "value": sample[left].corr(sample[right])})
    matrix = sm.add_constant(sample.astype(float))
    for index, variable in enumerate(matrix.columns[1:], start=1):
        rows.append({"metric": "vif_p3_3", "variable": variable, "value": variance_inflation_factor(matrix.values, index)})
    return pd.DataFrame(rows)


def productivity_tercile_figure(panel: pd.DataFrame, output_path: Path) -> None:
    sample = complete_case_data(panel, [WORKING_AGE, PRODUCTIVITY])
    sample = sample.copy()
    sample["productivity_tercile"] = pd.qcut(sample[PRODUCTIVITY], 3, labels=["low", "middle", "high"])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for axis, label in zip(axes, ["low", "middle", "high"]):
        group = sample[sample["productivity_tercile"] == label]
        axis.scatter(group[WORKING_AGE], group[OUTCOME], alpha=0.35, s=16)
        if len(group) >= 2:
            slope, intercept = np.polyfit(group[WORKING_AGE], group[OUTCOME], 1)
            x_values = np.linspace(group[WORKING_AGE].min(), group[WORKING_AGE].max(), 100)
            axis.plot(x_values, intercept + slope * x_values, color="black", linewidth=1.4)
        axis.set_title(f"{label.capitalize()} productivity")
        axis.set_xlabel("10-year working-age-share change")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean real GDP per-capita growth, next five years")
    figure.suptitle("Demography and subsequent growth by productivity tercile")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_phase3(panel_path: Path, tables_dir: Path, figures_dir: Path, coverage_path: Path) -> dict[str, pd.DataFrame]:
    panel = pd.read_csv(panel_path)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    coverage = phase3_coverage(panel)
    coverage.to_csv(coverage_path, index=False)
    results = regression_results(panel)
    results.to_csv(tables_dir / "phase3_regression_results.csv", index=False)
    marginal = marginal_productivity_effects(panel)
    marginal.to_csv(tables_dir / "phase3_productivity_interaction.csv", index=False)
    common = common_sample_results(panel)
    common.to_csv(tables_dir / "phase3_common_sample_results.csv", index=False)
    sensitivity = nonoverlap_results(panel)
    sensitivity.to_csv(tables_dir / "phase3_nonoverlap_sensitivity.csv", index=False)
    multicollinearity(panel).to_csv(tables_dir / "phase3_multicollinearity.csv", index=False)
    productivity_tercile_figure(panel, figures_dir / "demography_productivity_relationship.png")
    return {"panel": panel, "coverage": coverage, "results": results, "marginal": marginal, "common": common, "sensitivity": sensitivity}


if __name__ == "__main__":
    from .config import PHASE3_PANEL_FILE, PROJECT_ROOT

    run_phase3(PHASE3_PANEL_FILE, PROJECT_ROOT / "outputs" / "tables", PROJECT_ROOT / "outputs" / "figures", PROJECT_ROOT / "outputs" / "coverage" / "phase3_variable_coverage.csv")
    print("Phase 3 analysis outputs generated")