"""Phase 5 formal tests of structural and financial vulnerability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import PROJECT_ROOT, STRUCTURAL_FINANCIAL_PANEL_FILE
from .panel_analysis import nonoverlap_sample

OUTCOME = "real_gdp_pc_growth_5y_forward"
ANNUAL_GROWTH = "real_gdp_pc_growth"
WORKING_AGE = "working_age_change_10y"
BASE_CONTROLS = [
    WORKING_AGE,
    "log_real_gdp_pc",
    "investment_gdp_5y_backward_avg",
    "inflation_5y_backward_avg",
]
SENSITIVITY_CONTROLS = [
    "labour_force_participation_5y_backward_avg",
    "productivity_growth_5y_backward",
]
FINANCIAL_VARIABLES = {
    "credit_gap": {"tag": "CG", "unit": "percentage points of GDP"},
    "dsr_deviation_10y": {"tag": "DSR", "unit": "percentage points of DSR"},
    "real_house_price_growth_5y": {"tag": "HP", "unit": "percent change over five years"},
}
HORIZON_OUTCOMES = {
    1: "real_gdp_pc_growth_1y_forward",
    3: "real_gdp_pc_growth_3y_forward",
    5: OUTCOME,
}


def complete_case_data(
    panel: pd.DataFrame,
    predictors: list[str],
    outcome: str = OUTCOME,
) -> pd.DataFrame:
    """Return model-complete rows and prohibit future information as predictors."""
    if any("forward" in column for column in predictors):
        raise ValueError("Forward variables may only be used as the dependent outcome")
    return panel.dropna(subset=["country_code", "year", outcome, *predictors]).copy()


def add_forward_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    """Construct exact-calendar future growth windows without changing stored 5y data."""
    if panel.duplicated(["country_code", "year"]).any():
        raise ValueError("Forward outcomes require unique country-year rows")
    result = panel.sort_values(["country_code", "year"]).copy()
    for horizon, column in [(1, HORIZON_OUTCOMES[1]), (3, HORIZON_OUTCOMES[3])]:
        result[column] = np.nan
        for _, group in result.groupby("country_code", sort=False):
            annual = group.set_index("year")[ANNUAL_GROWTH]
            values = []
            for year in group["year"]:
                window = [annual.get(year + offset, np.nan) for offset in range(1, horizon + 1)]
                values.append(float(np.mean(window)) if all(pd.notna(value) for value in window) else np.nan)
            result.loc[group.index, column] = values
    return result


def _design(data: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    countries = pd.get_dummies(data["country_code"], prefix="country", drop_first=True, dtype=float)
    years = pd.get_dummies(data["year"], prefix="year", drop_first=True, dtype=float)
    regressors = pd.concat([data[predictors].astype(float), countries, years], axis=1)
    return sm.add_constant(regressors, has_constant="add")


def _fit(
    panel: pd.DataFrame,
    model_name: str,
    predictors: list[str],
    outcome: str = OUTCOME,
) -> tuple[pd.DataFrame, pd.DataFrame, object, pd.DataFrame]:
    data = complete_case_data(panel, predictors, outcome)
    if data["country_code"].nunique() < 2:
        raise ValueError("At least two country clusters are required")
    design = _design(data, predictors)
    model = sm.OLS(data[outcome].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": data["country_code"]}
    )
    confidence = model.conf_int()
    metadata = {
        "N": len(data),
        "countries": data["country_code"].nunique(),
        "first_year": int(data["year"].min()),
        "last_year": int(data["year"].max()),
        "fixed_effects": "country, year",
        "se_method": "country-clustered",
        "outcome": outcome,
    }
    coefficients = pd.DataFrame(
        [
            {
                "model": model_name,
                "term": variable,
                "estimate": model.params[variable],
                "clustered_se": model.bse[variable],
                "t_stat": model.tvalues[variable],
                "p_value": model.pvalues[variable],
                "ci_lower": confidence.loc[variable, 0],
                "ci_upper": confidence.loc[variable, 1],
                **metadata,
            }
            for variable in predictors
        ]
    )
    model_row = pd.DataFrame([{"model": model_name, "term": "__model__", **metadata}])
    return coefficients, model_row, model, data


def _centered_interaction(
    panel: pd.DataFrame,
    financial: str,
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    result = panel.copy()
    working_mean = float(sample[WORKING_AGE].mean())
    financial_mean = float(sample[financial].mean())
    result["centred_working_age_change_10y"] = result[WORKING_AGE] - working_mean
    result[f"centred_{financial}"] = result[financial] - financial_mean
    result[f"working_age_x_{financial}"] = (
        result["centred_working_age_change_10y"] * result[f"centred_{financial}"]
    )
    return result, working_mean, financial_mean


def _interaction_effects(
    model: object,
    sample: pd.DataFrame,
    financial: str,
    working_mean: float,
    financial_mean: float,
    family: str,
) -> pd.DataFrame:
    interaction = f"working_age_x_{financial}"
    covariance = model.cov_params()
    rows = []
    for label, value in [
        ("p25", sample[WORKING_AGE].quantile(0.25)),
        ("median", sample[WORKING_AGE].median()),
        ("p75", sample[WORKING_AGE].quantile(0.75)),
    ]:
        centered = value - working_mean
        estimate = model.params[f"centred_{financial}"] + model.params[interaction] * centered
        variance = (
            covariance.loc[f"centred_{financial}", f"centred_{financial}"]
            + centered**2 * covariance.loc[interaction, interaction]
            + 2 * centered * covariance.loc[f"centred_{financial}", interaction]
        )
        se = float(np.sqrt(max(variance, 0)))
        rows.append({
            "family": family,
            "marginal_association": "financial_at_working_age_change",
            "condition": label,
            "condition_value": value,
            "estimate": estimate,
            "clustered_se": se,
            "ci_lower": estimate - 1.96 * se,
            "ci_upper": estimate + 1.96 * se,
            "N": len(sample),
            "countries": sample["country_code"].nunique(),
            "financial_mean_used_for_centering": financial_mean,
        })
    for label, value in [
        ("p25", sample[financial].quantile(0.25)),
        ("median", sample[financial].median()),
        ("p75", sample[financial].quantile(0.75)),
    ]:
        centered = value - financial_mean
        estimate = model.params["centred_working_age_change_10y"] + model.params[interaction] * centered
        variance = (
            covariance.loc["centred_working_age_change_10y", "centred_working_age_change_10y"]
            + centered**2 * covariance.loc[interaction, interaction]
            + 2 * centered * covariance.loc["centred_working_age_change_10y", interaction]
        )
        se = float(np.sqrt(max(variance, 0)))
        rows.append({
            "family": family,
            "marginal_association": "working_age_change_at_financial",
            "condition": label,
            "condition_value": value,
            "estimate": estimate,
            "clustered_se": se,
            "ci_lower": estimate - 1.96 * se,
            "ci_upper": estimate + 1.96 * se,
            "N": len(sample),
            "countries": sample["country_code"].nunique(),
            "working_age_mean_used_for_centering": working_mean,
        })
    return pd.DataFrame(rows)


def _family_interaction(
    panel: pd.DataFrame,
    financial: str,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, object, pd.DataFrame, pd.DataFrame]:
    predictors = [WORKING_AGE, financial, *BASE_CONTROLS[1:]]
    sample = complete_case_data(panel, predictors)
    interacted, working_mean, financial_mean = _centered_interaction(panel, financial, sample)
    interaction = f"working_age_x_{financial}"
    model_predictors = [
        "centred_working_age_change_10y",
        f"centred_{financial}",
        interaction,
        *BASE_CONTROLS[1:],
    ]
    coefficients, summary, model, data = _fit(interacted, model_name, model_predictors)
    marginal = _interaction_effects(
        model, data, financial, working_mean, financial_mean, model_name
    )
    return coefficients, summary, model, data, marginal


def _continuous_family(
    panel: pd.DataFrame,
    financial: str,
    family: str,
) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictors = [*BASE_CONTROLS, financial]
    sample = complete_case_data(panel, predictors)
    baseline_name = f"P5-{family}0"
    main_name = f"P5-{family}1"
    baseline, baseline_summary, _, baseline_data = _fit(sample, baseline_name, BASE_CONTROLS)
    main, main_summary, _, data = _fit(sample, main_name, predictors)
    sd = float(data[financial].std(ddof=1))
    mean = float(data[financial].mean())
    standardized = data.copy()
    standardized[f"{financial}_standardized"] = (standardized[financial] - mean) / sd
    standardized_result, standardized_summary, _, _ = _fit(
        standardized,
        f"{main_name}-1SD",
        [*BASE_CONTROLS, f"{financial}_standardized"],
    )
    main.loc[main["term"] == financial, "standard_deviation"] = sd
    main.loc[main["term"] == financial, "unit"] = FINANCIAL_VARIABLES[financial]["unit"]
    standardized_result.loc[
        standardized_result["term"] == f"{financial}_standardized", "unit"
    ] = "one sample standard deviation"
    baseline_table = pd.concat([baseline, baseline_summary], ignore_index=True)
    results = [
        baseline_table,
        pd.concat([main, main_summary], ignore_index=True),
        pd.concat([standardized_result, standardized_summary], ignore_index=True),
    ]
    baseline_sample = baseline_data[["country_code", "year"]].reset_index(drop=True)
    financial_sample = data[["country_code", "year"]].reset_index(drop=True)
    if not baseline_sample.equals(financial_sample):
        raise AssertionError(f"{baseline_name} and {main_name} samples differ")
    return results, main, standardized_result, pd.DataFrame([{
        "family": family,
        "financial_variable": financial,
        "unit": FINANCIAL_VARIABLES[financial]["unit"],
        "mean": mean,
        "standard_deviation": sd,
        "N": len(data),
        "countries": data["country_code"].nunique(),
        "first_year": int(data.year.min()),
        "last_year": int(data.year.max()),
        "baseline_sample_identity_verified": True,
    }])


def financial_nonlinearity(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for financial, metadata in FINANCIAL_VARIABLES.items():
        sample = complete_case_data(panel, [*BASE_CONTROLS, financial])
        quartiles = pd.qcut(sample[financial], 4, labels=False, duplicates="drop") + 1
        sample = sample.assign(financial_quartile=quartiles)
        for quartile, group in sample.groupby("financial_quartile", observed=True):
            rows.append({
                "record_type": "quartile_descriptive",
                "family": metadata["tag"],
                "financial_variable": financial,
                "term": f"Q{int(quartile)}",
                "estimate": group[OUTCOME].mean(),
                "N": len(group),
                "countries": group.country_code.nunique(),
                "first_year": int(group.year.min()),
                "last_year": int(group.year.max()),
                "unit": metadata["unit"],
            })
        centered = sample.copy()
        centered_value = f"centred_{financial}"
        centered_square = f"centred_{financial}_squared"
        centered[centered_value] = centered[financial] - centered[financial].mean()
        centered[centered_square] = centered[centered_value] ** 2
        coefficients, summary, _, _ = _fit(
            centered,
            f"P5-{metadata['tag']}-quadratic",
            [*BASE_CONTROLS, centered_value, centered_square],
        )
        for _, coefficient in coefficients.iterrows():
            rows.append({
                "record_type": "quadratic_model",
                "family": metadata["tag"],
                "financial_variable": financial,
                "term": coefficient["term"],
                "estimate": coefficient["estimate"],
                "clustered_se": coefficient["clustered_se"],
                "t_stat": coefficient["t_stat"],
                "p_value": coefficient["p_value"],
                "ci_lower": coefficient["ci_lower"],
                "ci_upper": coefficient["ci_upper"],
                "N": coefficient["N"],
                "countries": coefficient["countries"],
                "first_year": coefficient["first_year"],
                "last_year": coefficient["last_year"],
                "unit": metadata["unit"],
            })
        rows.extend(
            {
                "record_type": "quadratic_model_summary",
                "family": metadata["tag"],
                "financial_variable": financial,
                "term": "__model__",
                **summary.iloc[0].to_dict(),
                "unit": metadata["unit"],
            }
            for _, model_summary in summary.iterrows()
        )
    return pd.DataFrame(rows)


def common_sample_results(panel: pd.DataFrame) -> pd.DataFrame:
    records = []
    for financial, metadata in FINANCIAL_VARIABLES.items():
        base = [*BASE_CONTROLS, financial]
        common = complete_case_data(panel, base)
        interacted, _, _ = _centered_interaction(panel, financial, common)
        interaction = f"working_age_x_{financial}"
        interaction_predictors = [
            "centred_working_age_change_10y",
            f"centred_{financial}",
            interaction,
            *BASE_CONTROLS[1:],
        ]
        specifications = [
            ("A_controls", BASE_CONTROLS, common),
            ("B_financial", base, common),
            ("C_interaction", interaction_predictors, interacted),
        ]
        identities = []
        for label, predictors, source in specifications:
            coefficients, summary, _, sample = _fit(
                source, f"P5-{metadata['tag']}-common-{label}", predictors
            )
            identity = sample[["country_code", "year"]].sort_values(
                ["country_code", "year"]
            ).reset_index(drop=True)
            identities.append(identity)
            for frame in [coefficients, summary]:
                frame = frame.copy()
                frame["family"] = metadata["tag"]
                frame["common_specification"] = label
                frame["common_sample_identity_verified"] = True
                records.append(frame)
        if not all(identities[0].equals(identity) for identity in identities[1:]):
            raise AssertionError(f"Common-sample identity failed for {metadata['tag']}")
    return pd.concat(records, ignore_index=True)


def phase3_control_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for financial, family in [("credit_gap", "CG"), ("dsr_deviation_10y", "DSR")]:
        base_predictors = [*BASE_CONTROLS, financial]
        extra_predictors = [*base_predictors, *SENSITIVITY_CONTROLS]
        base_result, _, _, _ = _fit(
            panel, f"P5-{family}1", base_predictors
        )
        base_coefficient = base_result.loc[base_result.term == financial].iloc[0]
        sensitivity, _, _, _ = _fit(
            panel, f"P5-{family}1-extra-controls", extra_predictors
        )
        sensitivity_coefficient = sensitivity.loc[sensitivity.term == financial].iloc[0]
        base_interaction, _, _, _, _ = _family_interaction(
            panel, financial, f"P5-{family}2"
        )
        interaction_predictors = [
            "centred_working_age_change_10y", f"centred_{financial}",
            f"working_age_x_{financial}", *BASE_CONTROLS[1:],
        ]
        interacted_extra, _, _ = _centered_interaction(
            panel, financial, complete_case_data(panel, [*extra_predictors])
        )
        extra_interaction_predictors = [
            *interaction_predictors, *SENSITIVITY_CONTROLS
        ]
        extra_interaction, _, _, _ = _fit(
            interacted_extra,
            f"P5-{family}2-extra-controls",
            extra_interaction_predictors,
        )
        base_interaction_coefficient = base_interaction.loc[
            base_interaction.term == f"centred_{financial}"
        ].iloc[0]
        extra_interaction_coefficient = extra_interaction.loc[
            extra_interaction.term == f"centred_{financial}"
        ].iloc[0]
        for model_type, base_value, extra_value, base_n, extra_n in [
            ("continuous", base_coefficient.estimate, sensitivity_coefficient.estimate, base_coefficient.N, sensitivity_coefficient.N),
            ("interaction_financial_main_effect", base_interaction_coefficient.estimate, extra_interaction_coefficient.estimate, base_interaction_coefficient.N, extra_interaction_coefficient.N),
            ("interaction_term", base_interaction.loc[base_interaction.term == f"working_age_x_{financial}", "estimate"].iloc[0], extra_interaction.loc[extra_interaction.term == f"working_age_x_{financial}", "estimate"].iloc[0], base_interaction_coefficient.N, extra_interaction_coefficient.N),
        ]:
            rows.append({
                "family": family,
                "model_type": model_type,
                "financial_variable": financial,
                "base_estimate": base_value,
                "extra_control_estimate": extra_value,
                "change_in_estimate": extra_value - base_value,
                "base_N": base_n,
                "extra_control_N": extra_n,
                "change_in_N": extra_n - base_n,
                "extra_controls": ", ".join(SENSITIVITY_CONTROLS),
            })
    return pd.DataFrame(rows)


def nonoverlap_results(panel: pd.DataFrame) -> pd.DataFrame:
    reduced = nonoverlap_sample(panel)
    rows = []
    for financial, metadata in FINANCIAL_VARIABLES.items():
        family = metadata["tag"]
        predictors = [*BASE_CONTROLS, financial]
        coefficients, summary, _, _ = _fit(reduced, f"P5-{family}1-nonoverlap", predictors)
        rows.extend([coefficients, summary])
        if family != "HP":
            interaction, interaction_summary, _, _, _ = _family_interaction(
                reduced, financial, f"P5-{family}2-nonoverlap"
            )
            rows.extend([interaction, interaction_summary])
    result = pd.concat(rows, ignore_index=True)
    if not result.empty:
        if (result["first_year"].dropna().astype(int) % 5 != 0).any() or (result["last_year"].dropna().astype(int) % 5 != 0).any():
            raise AssertionError("Non-overlapping models include years outside year % 5 == 0")
    return result


def horizon_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    data = add_forward_outcomes(panel)
    rows = []
    for horizon, outcome in HORIZON_OUTCOMES.items():
        for financial, metadata in FINANCIAL_VARIABLES.items():
            coefficients, summary, _, _ = _fit(
                data,
                f"P5-{metadata['tag']}{horizon}y",
                [*BASE_CONTROLS, financial],
                outcome,
            )
            coefficients = coefficients.loc[coefficients.term == financial].copy()
            coefficients["horizon_years"] = horizon
            summary["horizon_years"] = horizon
            rows.extend([coefficients, summary])
    return pd.concat(rows, ignore_index=True)


def _point_in_time_flags(panel: pd.DataFrame, minimum_history: int = 5) -> pd.DataFrame:
    data = panel.sort_values(["country_code", "year"]).copy()
    for column in ["credit_gap_elevated_pit", "dsr_elevated_pit", "demographic_deterioration_pit"]:
        data[column] = pd.Series(pd.NA, index=data.index, dtype="boolean")
    for _, group in data.groupby("country_code", sort=False):
        for index, (_, row) in enumerate(group.iterrows()):
            history = group.iloc[:index]
            if len(history["credit_gap"].dropna()) >= minimum_history and pd.notna(row["credit_gap"]):
                threshold = history["credit_gap"].dropna().quantile(0.8)
                data.loc[row.name, "credit_gap_elevated_pit"] = row["credit_gap"] >= threshold
            if len(history["dsr_deviation_10y"].dropna()) >= minimum_history and pd.notna(row["dsr_deviation_10y"]):
                threshold = history["dsr_deviation_10y"].dropna().quantile(0.8)
                data.loc[row.name, "dsr_elevated_pit"] = row["dsr_deviation_10y"] >= threshold
            historical_demographics = history[WORKING_AGE].dropna()
            if len(historical_demographics) >= minimum_history and pd.notna(row[WORKING_AGE]):
                data.loc[row.name, "demographic_deterioration_pit"] = (
                    row[WORKING_AGE] < historical_demographics.median()
                )
    return data


def point_in_time_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    data = _point_in_time_flags(panel)
    rows = []
    for financial, flag, family in [
        ("credit_gap", "credit_gap_elevated_pit", "CG"),
        ("dsr_deviation_10y", "dsr_elevated_pit", "DSR"),
    ]:
        sample = data.dropna(subset=[flag]).copy()
        sample[flag] = sample[flag].astype(float)
        base_predictors = [flag, *BASE_CONTROLS]
        coefficients, summary, _, _ = _fit(
            sample, f"P5-{family}-PIT1", base_predictors
        )
        coefficients["classification"] = "point_in_time_elevated"
        summary["classification"] = "point_in_time_elevated"
        rows.extend([coefficients, summary])
        interaction_sample = sample.dropna(subset=["demographic_deterioration_pit"]).copy()
        interaction_sample["demographic_deterioration_pit"] = interaction_sample[
            "demographic_deterioration_pit"
        ].astype(float)
        interaction_sample["elevated_x_demographic_deterioration"] = (
            interaction_sample[flag]
            * interaction_sample["demographic_deterioration_pit"]
        )
        predictors = [
            flag,
            "demographic_deterioration_pit",
            "elevated_x_demographic_deterioration",
            *BASE_CONTROLS,
        ]
        coefficients, summary, _, _ = _fit(
            interaction_sample, f"P5-{family}-PIT2", predictors
        )
        coefficients["classification"] = "point_in_time_elevated_with_demographic_interaction"
        summary["classification"] = "point_in_time_elevated_with_demographic_interaction"
        rows.extend([coefficients, summary])
    return pd.concat(rows, ignore_index=True)


def multicollinearity_and_scale(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for financial, metadata in FINANCIAL_VARIABLES.items():
        sample = complete_case_data(panel, [*BASE_CONTROLS, financial])
        for variable in [*BASE_CONTROLS, financial]:
            rows.append({
                "family": metadata["tag"],
                "metric": "standard_deviation" if variable == financial else "predictor_correlation",
                "variable": variable,
                "other_variable": "",
                "value": sample[financial].std(ddof=1) if variable == financial else np.nan,
                "unit": metadata["unit"] if variable == financial else "",
                "N": len(sample),
            })
        for left_index, left in enumerate([*BASE_CONTROLS, financial]):
            for right in [*BASE_CONTROLS, financial][left_index + 1:]:
                rows.append({
                    "family": metadata["tag"],
                    "metric": "predictor_correlation",
                    "variable": left,
                    "other_variable": right,
                    "value": sample[left].corr(sample[right]),
                    "unit": "",
                    "N": len(sample),
                })
        matrix = sm.add_constant(sample[[*BASE_CONTROLS, financial]].astype(float))
        for index, variable in enumerate(matrix.columns[1:], start=1):
            rows.append({
                "family": metadata["tag"],
                "metric": "VIF",
                "variable": variable,
                "other_variable": "",
                "value": variance_inflation_factor(matrix.values, index),
                "unit": "",
                "N": len(sample),
            })
        if "private_credit_gdp" in sample.columns:
            rows.append({
                "family": metadata["tag"],
                "metric": "not_included_scale_reference",
                "variable": "private_credit_gdp",
                "other_variable": "",
                "value": sample["private_credit_gdp"].std(ddof=1),
                "unit": "percent of GDP",
                "N": len(sample),
            })
    return pd.DataFrame(rows)


def _scatter_figure(panel: pd.DataFrame, financial: str, output_path: Path) -> None:
    sample = panel[[financial, OUTCOME]].dropna()
    figure, axis = plt.subplots(figsize=(8.2, 5.5))
    axis.scatter(sample[financial], sample[OUTCOME], alpha=0.35, s=18, color="#247b78")
    design = sm.add_constant(sample[financial].astype(float))
    model = sm.OLS(sample[OUTCOME].astype(float), design).fit()
    x_values = np.linspace(sample[financial].min(), sample[financial].max(), 120)
    prediction = model.get_prediction(sm.add_constant(pd.Series(x_values, name=financial), has_constant="add")).summary_frame()
    axis.plot(x_values, prediction["mean"], color="#b84a3a", linewidth=1.8, label="Linear fit")
    axis.fill_between(x_values, prediction["mean_ci_lower"], prediction["mean_ci_upper"], color="#b84a3a", alpha=0.16, label="95% CI for mean")
    axis.set_xlabel(financial.replace("_", " "))
    axis.set_ylabel("Mean real GDP per-capita growth, next five years (%)")
    axis.set_title(f"{financial.replace('_', ' ')} and subsequent growth")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _credit_interaction_figure(
    model: object,
    sample: pd.DataFrame,
    working_mean: float,
    output_path: Path,
) -> None:
    values = np.linspace(sample[WORKING_AGE].min(), sample[WORKING_AGE].max(), 140)
    centered = values - working_mean
    interaction = model.params["working_age_x_credit_gap"]
    estimates = model.params["centred_credit_gap"] + interaction * centered
    covariance = model.cov_params()
    variance = (
        covariance.loc["centred_credit_gap", "centred_credit_gap"]
        + centered**2 * covariance.loc["working_age_x_credit_gap", "working_age_x_credit_gap"]
        + 2 * centered * covariance.loc["centred_credit_gap", "working_age_x_credit_gap"]
    )
    errors = np.sqrt(np.maximum(variance, 0))
    figure, axis = plt.subplots(figsize=(8.2, 5.5))
    axis.plot(values, estimates, color="#b84a3a", linewidth=2)
    axis.fill_between(values, estimates - 1.96 * errors, estimates + 1.96 * errors, color="#b84a3a", alpha=0.18, label="95% CI")
    axis.axhline(0, color="#555555", linewidth=1, linestyle="--")
    axis.set_xlabel("10-year working-age-share change")
    axis.set_ylabel("Marginal credit-gap association with future growth")
    axis.set_title("Credit-gap association across demographic conditions")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_phase5(
    panel_path: Path = STRUCTURAL_FINANCIAL_PANEL_FILE,
    tables_dir: Path = PROJECT_ROOT / "outputs" / "tables",
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures",
) -> dict[str, pd.DataFrame]:
    panel = pd.read_csv(panel_path)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[pd.DataFrame] = []
    main_terms: list[pd.DataFrame] = []
    standardized_terms: list[pd.DataFrame] = []
    sample_metadata: list[pd.DataFrame] = []
    interaction_tables: dict[str, pd.DataFrame] = {}
    interaction_models: dict[str, tuple[object, pd.DataFrame, float, float]] = {}

    for financial, metadata in FINANCIAL_VARIABLES.items():
        family = metadata["tag"]
        family_results, main, standardized, sample_info = _continuous_family(
            panel, financial, family
        )
        all_results.extend(family_results)
        main_terms.append(main.loc[main.term == financial].copy())
        standardized_terms.append(
            standardized.loc[standardized.term == f"{financial}_standardized"].copy()
        )
        sample_metadata.append(sample_info)
        coefficients, summary, model, data, marginal = _family_interaction(
            panel, financial, f"P5-{family}2"
        )
        all_results.extend([coefficients, summary])
        interaction_tables[family] = pd.concat([coefficients, summary], ignore_index=True)
        interaction_tables[f"{family}_marginal"] = marginal
        interaction_models[family] = (
            model,
            data,
            float(data[WORKING_AGE].mean()),
            float(data[financial].mean()),
        )

    primary = pd.concat(main_terms + [
        table.loc[table.term.isin(["centred_credit_gap", "working_age_x_credit_gap"])]
        for table in [interaction_tables["CG"]]
    ] + [
        table.loc[table.term.isin(["centred_dsr_deviation_10y", "working_age_x_dsr_deviation_10y"])]
        for table in [interaction_tables["DSR"]]
    ] + [
        table.loc[table.term.isin(["centred_real_house_price_growth_5y", "working_age_x_real_house_price_growth_5y"])]
        for table in [interaction_tables["HP"]]
    ], ignore_index=True)
    primary["priority"] = "PRIMARY"
    standardized_table = pd.concat(standardized_terms, ignore_index=True)
    baseline_and_continuous = pd.concat(all_results, ignore_index=True)

    nonlinear = financial_nonlinearity(panel)
    common = common_sample_results(panel)
    sensitivities = phase3_control_sensitivity(panel)
    nonoverlap = nonoverlap_results(panel)
    horizons = horizon_sensitivity(panel)
    point_in_time = point_in_time_episodes(panel)
    scales = multicollinearity_and_scale(panel)
    marginals = pd.concat(
        [interaction_tables[f"{family}_marginal"] for family in ["CG", "DSR", "HP"]],
        ignore_index=True,
    )

    outputs = {
        "phase5_regression_results.csv": baseline_and_continuous,
        "phase5_standardized_financial_coefficients.csv": standardized_table,
        "phase5_financial_sample_summary.csv": pd.concat(sample_metadata, ignore_index=True),
        "phase5_financial_nonlinearity.csv": nonlinear,
        "phase5_credit_gap_interaction.csv": pd.concat([
            interaction_tables["CG"].assign(record_type="model_coefficient"),
            interaction_tables["CG_marginal"].assign(record_type="marginal_association"),
        ], ignore_index=True),
        "phase5_dsr_interaction.csv": pd.concat([
            interaction_tables["DSR"].assign(record_type="model_coefficient"),
            interaction_tables["DSR_marginal"].assign(record_type="marginal_association"),
        ], ignore_index=True),
        "phase5_property_interaction.csv": pd.concat([
            interaction_tables["HP"].assign(record_type="model_coefficient"),
            interaction_tables["HP_marginal"].assign(record_type="marginal_association"),
        ], ignore_index=True),
        "phase5_common_sample_results.csv": common,
        "phase5_control_sensitivity.csv": sensitivities,
        "phase5_nonoverlap_results.csv": nonoverlap,
        "phase5_horizon_sensitivity.csv": horizons,
        "phase5_point_in_time_episodes.csv": point_in_time,
        "phase5_multicollinearity.csv": scales,
        "phase5_prespecified_summary.csv": primary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(tables_dir / filename, index=False)

    _scatter_figure(panel, "credit_gap", figures_dir / "phase5_credit_gap_growth.png")
    _scatter_figure(panel, "dsr_deviation_10y", figures_dir / "phase5_dsr_growth.png")
    credit_model, credit_data, working_mean, financial_mean = interaction_models["CG"]
    _credit_interaction_figure(
        credit_model,
        credit_data,
        working_mean,
        figures_dir / "phase5_credit_gap_demographic_interaction.png",
    )
    return {
        "panel": panel,
        "primary": primary,
        "standardized": standardized_table,
        "nonlinearity": nonlinear,
        "common": common,
        "sensitivity": sensitivities,
        "nonoverlap": nonoverlap,
        "horizons": horizons,
        "point_in_time": point_in_time,
        "multicollinearity": scales,
        "marginals": marginals,
    }


if __name__ == "__main__":
    generated = run_phase5()
    print(f"Generated Phase 5 outputs from {len(generated['panel']):,} country-years")