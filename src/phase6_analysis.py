"""Phase 6 macro-financial vulnerability and asset-outcome validation."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import COUNTRIES, PROJECT_ROOT, STRUCTURAL_FINANCIAL_PANEL_FILE
from .panel_analysis import nonoverlap_sample

JST_RELEASE = "Release 6 (2022)"
JST_SOURCE_URL = "https://www.macrohistory.net/database/"
JST_DOWNLOAD_URL = "https://www.macrohistory.net/app/download/9834512569/JSTdatasetR6.xlsx?t=1763503850"
JST_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)"
JST_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "jst" / "JSTdatasetR6.xlsx"
MARKET_PANEL_FILE = PROJECT_ROOT / "data" / "processed" / "market_validation_panel.csv"
ASSET_NOMINAL = {
    "equity": "eq_tr",
    "bond": "bond_tr",
    "bill": "bill_rate",
    "housing": "housing_tr",
}
ASSET_REAL = {asset: f"{asset}_real_return" for asset in ASSET_NOMINAL}
ASSET_FORWARD = {
    asset: {
        1: f"{asset}_real_return_1y_forward",
        3: f"{asset}_real_return_3y_forward_annualized",
        5: f"{asset}_real_return_5y_forward_annualized",
    }
    for asset in ASSET_NOMINAL
}
BASE_CONTROLS = [
    "working_age_change_10y",
    "log_real_gdp_pc",
    "investment_gdp_5y_backward_avg",
    "inflation_5y_backward_avg",
]


def country_mapping() -> pd.DataFrame:
    """Map JST ISO codes to the project universe without fuzzy country matching."""
    rows = []
    for country, iso in sorted(
        {
            "Australia": "AUS", "Belgium": "BEL", "Canada": "CAN",
            "Switzerland": "CHE", "Germany": "DEU", "Denmark": "DNK",
            "Spain": "ESP", "Finland": "FIN", "France": "FRA",
            "UK": "GBR", "Ireland": "IRL", "Italy": "ITA",
            "Japan": "JPN", "Netherlands": "NLD", "Norway": "NOR",
            "Portugal": "PRT", "Sweden": "SWE", "USA": "USA",
        }.items(), key=lambda item: item[1]
    ):
        included = iso in COUNTRIES
        rows.append({
            "jst_country": country,
            "iso3": iso,
            "included": included,
            "reason": "ISO3 matches project country universe" if included else "JST country is outside the project country universe",
        })
    return pd.DataFrame(rows)


def _source_metadata(raw: pd.DataFrame) -> dict[str, object]:
    return {
        "database": "Jorda-Schularick-Taylor Macrohistory Database",
        "release": JST_RELEASE,
        "source_url": JST_SOURCE_URL,
        "download_url": JST_DOWNLOAD_URL,
        "download_date": date.today().isoformat(),
        "license": JST_LICENSE,
        "use_restriction": "Non-commercial research only; commercial integration or resale is prohibited by the provider's terms.",
        "citation": "Jorda, Schularick, and Taylor (2017), Macrofinancial History and the New Business Cycle Facts; return data: Jorda, Knoll, Kuvshinov, Schularick, and Taylor (2019), The Rate of Return on Everything, 1870-2015.",
        "source_rows": int(len(raw)),
        "source_countries": int(raw["iso"].nunique()),
        "first_year": int(raw["year"].min()),
        "last_year": int(raw["year"].max()),
        "variables_used": {
            "eq_tr": "Nominal equity total return, local currency, decimal ratio.",
            "bond_tr": "Nominal government bond total return, local currency, decimal ratio.",
            "bill_rate": "Nominal government bill return coupon[t]/price[t-1], local currency, decimal ratio.",
            "housing_tr": "Nominal housing total return, local currency, decimal ratio.",
            "cpi": "Consumer price index, 1990=100; annual inflation is derived from the exact-calendar-year ratio.",
            "crisisJST": "Systemic financial crisis 0-1 indicator; one in the documented onset year.",
        },
    }


def load_jst(raw_path: Path = JST_RAW_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and validate the official JST workbook, producing real annual returns."""
    raw = pd.read_excel(raw_path)
    required = ["year", "country", "iso", "cpi", "crisisJST", *ASSET_NOMINAL.values()]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"JST workbook is missing required fields: {missing}")
    raw = raw[["year", "country", "iso", "cpi", "crisisJST", *ASSET_NOMINAL.values()]].copy()
    raw["year"] = pd.to_numeric(raw["year"], errors="raise").astype(int)
    raw["iso"] = raw["iso"].astype(str).str.upper()
    if raw.duplicated(["iso", "year"]).any():
        raise ValueError("JST workbook contains duplicate country-years")
    raw["cpi"] = pd.to_numeric(raw["cpi"], errors="coerce")
    raw["inflation_jst"] = np.nan
    for _, group in raw.groupby("iso", sort=False):
        indexed = group.set_index("year")["cpi"]
        inflation = [
            indexed.get(year) / indexed.get(year - 1) - 1
            if pd.notna(indexed.get(year))
            and pd.notna(indexed.get(year - 1))
            and indexed.get(year - 1) > 0
            else np.nan
            for year in group["year"]
        ]
        raw.loc[group.index, "inflation_jst"] = inflation
    for asset, nominal in ASSET_NOMINAL.items():
        raw[nominal] = pd.to_numeric(raw[nominal], errors="coerce")
        raw[ASSET_REAL[asset]] = (
            (1 + raw[nominal]) / (1 + raw["inflation_jst"]) - 1
        )
        invalid = raw[nominal].notna() & raw["inflation_jst"].le(-1)
        raw.loc[invalid, ASSET_REAL[asset]] = np.nan
    result = raw.rename(columns={"country": "jst_country", "iso": "country_code"})
    metadata = _source_metadata(raw)
    return result, pd.DataFrame([metadata])


def build_market_panel(
    structural_panel: pd.DataFrame,
    jst_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Use the structural-financial panel as the time-t predictor spine."""
    if structural_panel.duplicated(["country_code", "year"]).any():
        raise ValueError("Structural-financial input contains duplicate country-years")
    matched_codes = set(country_mapping().loc[lambda frame: frame.included, "iso3"])
    jst = jst_panel.loc[jst_panel.country_code.isin(matched_codes)].copy()
    if jst.duplicated(["country_code", "year"]).any():
        raise ValueError("Mapped JST data contains duplicate country-years")
    predictors = structural_panel.loc[
        structural_panel.country_code.isin(matched_codes)
    ].copy()
    asset_columns = [
        "country_code", "year", "jst_country", "cpi", "inflation_jst",
        "crisisJST", *ASSET_NOMINAL.values(), *ASSET_REAL.values(),
    ]
    market = predictors.merge(
        jst[asset_columns],
        on=["country_code", "year"],
        how="left",
        validate="one_to_one",
    )
    if market.duplicated(["country_code", "year"]).any():
        raise AssertionError("Market validation panel has duplicate country-years")
    return add_forward_outcomes(market)


def add_forward_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    """Build exact-calendar compounded forward returns and downside/crisis events."""
    if panel.duplicated(["country_code", "year"]).any():
        raise ValueError("Forward outcomes require unique country-year rows")
    result = panel.sort_values(["country_code", "year"]).copy()
    for asset, column in ASSET_REAL.items():
        for horizon in [1, 3, 5]:
            output = ASSET_FORWARD[asset][horizon]
            result[output] = np.nan
            for _, group in result.groupby("country_code", sort=False):
                returns = group.set_index("year")[column]
                outcomes = []
                for year in group["year"]:
                    values = [returns.get(year + offset, np.nan) for offset in range(1, horizon + 1)]
                    if all(pd.notna(value) for value in values) and all(1 + value > 0 for value in values):
                        outcomes.append(float(np.prod([1 + value for value in values]) ** (1 / horizon) - 1))
                    else:
                        outcomes.append(np.nan)
                result.loc[group.index, output] = outcomes
    for horizon in [3, 5]:
        output = f"equity_worst_annual_return_next_{horizon}y"
        result[output] = np.nan
        for _, group in result.groupby("country_code", sort=False):
            series = group.set_index("year")[ASSET_REAL["equity"]]
            values_out = []
            for year in group["year"]:
                values = [series.get(year + offset, np.nan) for offset in range(1, horizon + 1)]
                values_out.append(float(min(values)) if all(pd.notna(value) for value in values) else np.nan)
            result.loc[group.index, output] = values_out
    result["equity_negative_year_next_3y"] = np.nan
    result["systemic_crisis_next_1y"] = np.nan
    result["systemic_crisis_next_3y"] = np.nan
    result["systemic_crisis_next_5y"] = np.nan
    for _, group in result.groupby("country_code", sort=False):
        indexed_returns = group.set_index("year")[ASSET_REAL["equity"]]
        indexed_crisis = group.set_index("year")["crisisJST"]
        for index, row in group.iterrows():
            year = int(row.year)
            equity3 = [indexed_returns.get(year + step, np.nan) for step in range(1, 4)]
            if all(pd.notna(value) for value in equity3):
                result.loc[index, "equity_negative_year_next_3y"] = int(any(value < 0 for value in equity3))
            for horizon in [1, 3, 5]:
                crisis = [indexed_crisis.get(year + step, np.nan) for step in range(1, horizon + 1)]
                if all(pd.notna(value) for value in crisis):
                    result.loc[index, f"systemic_crisis_next_{horizon}y"] = int(any(value == 1 for value in crisis))
    return result


def _sample(panel: pd.DataFrame, outcome: str, predictors: list[str]) -> pd.DataFrame:
    if any("forward" in predictor or predictor.startswith("systemic_crisis_next_") for predictor in predictors):
        raise ValueError("Future outcomes cannot be predictors in predictive models")
    return panel.dropna(subset=["country_code", "year", outcome, *predictors]).copy()


def _fit(
    panel: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    model_name: str,
    *,
    allow_ex_post_crisis: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, object, pd.DataFrame]:
    if not allow_ex_post_crisis and any(predictor.startswith("systemic_crisis_next_") for predictor in predictors):
        raise ValueError("Future crisis indicators are prohibited in predictive models")
    if any("forward" in predictor for predictor in predictors):
        raise ValueError("Forward return outcomes cannot be predictors")
    data = panel.dropna(subset=["country_code", "year", outcome, *predictors]).copy()
    if data.country_code.nunique() < 2:
        raise ValueError("At least two country clusters are required")
    country_effects = pd.get_dummies(data.country_code, prefix="country", drop_first=True, dtype=float)
    year_effects = pd.get_dummies(data.year, prefix="year", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[predictors].astype(float), country_effects, year_effects], axis=1),
        has_constant="add",
    )
    fitted = sm.OLS(data[outcome].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": data.country_code}
    )
    confidence = fitted.conf_int()
    metadata = {
        "N": len(data),
        "countries": int(data.country_code.nunique()),
        "first_year": int(data.year.min()),
        "last_year": int(data.year.max()),
        "fixed_effects": "country, year",
        "se_method": "country-clustered",
        "outcome": outcome,
        "analysis_type": "ex-post decomposition" if allow_ex_post_crisis else "predictive, signal at t and outcome after t",
    }
    coefficients = pd.DataFrame([
        {
            "model": model_name,
            "term": predictor,
            "estimate": fitted.params[predictor],
            "clustered_se": fitted.bse[predictor],
            "t_stat": fitted.tvalues[predictor],
            "p_value": fitted.pvalues[predictor],
            "ci_lower": confidence.loc[predictor, 0],
            "ci_upper": confidence.loc[predictor, 1],
            **metadata,
        }
        for predictor in predictors
    ])
    summary = pd.DataFrame([{"model": model_name, "term": "__model__", **metadata}])
    return coefficients, summary, fitted, data


def asset_descriptives(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for asset, variable in ASSET_REAL.items():
        sample = panel[["country_code", "year", variable]].dropna()
        values = sample[variable]
        rows.append({
            "asset": asset,
            "variable": variable,
            "count": int(values.count()),
            "countries": int(sample.country_code.nunique()),
            "first_year": int(sample.year.min()),
            "last_year": int(sample.year.max()),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "min": values.min(),
            "25%": values.quantile(0.25),
            "75%": values.quantile(0.75),
            "max": values.max(),
        })
    correlations = panel[list(ASSET_REAL.values())].corr(min_periods=2)
    corr_rows = []
    for index, left in enumerate(ASSET_REAL):
        for right in list(ASSET_REAL)[index + 1:]:
            pair = panel[[ASSET_REAL[left], ASSET_REAL[right]]].dropna()
            corr_rows.append({"asset_left": left, "asset_right": right, "correlation": pair.iloc[:, 0].corr(pair.iloc[:, 1]), "N": len(pair)})
    return pd.DataFrame(rows), pd.DataFrame(corr_rows)


def _asset_models(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, outcomes in ASSET_FORWARD.items():
        for horizon, outcome in outcomes.items():
            parsimonious = ["credit_gap"]
            controlled = ["credit_gap", *BASE_CONTROLS]
            parsimonious_sample = _sample(panel, outcome, parsimonious)
            controlled_sample = _sample(panel, outcome, controlled)
            primary_id = (
                "P6-EQ" if asset == "equity" and horizon == 5
                else "P6-BOND" if asset == "bond" and horizon == 5
                else f"P6-{asset.upper()}-{horizon}y"
            )
            model_specs = [
                (primary_id, parsimonious, parsimonious_sample, "PRIMARY" if asset in {"equity", "bond"} and horizon == 5 else "SECONDARY"),
                (f"{primary_id}-controlled", controlled, controlled_sample, "SENSITIVITY"),
            ]
            for model_name, predictors, source, priority in model_specs:
                coefficients, summary, _, _ = _fit(source, outcome, predictors, model_name)
                coefficients["priority"] = priority
                summary["priority"] = priority
                rows.extend([coefficients, summary])
    return pd.concat(rows, ignore_index=True)


def _downside_models(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in [3, 5]:
        outcome = f"equity_worst_annual_return_next_{horizon}y"
        for predictors, suffix, priority in [
            (["credit_gap"], "primary" if horizon == 5 else "secondary", "PRIMARY" if horizon == 5 else "SECONDARY"),
            (["credit_gap", *BASE_CONTROLS], "controlled", "SENSITIVITY"),
        ]:
            coefficients, summary, _, _ = _fit(
                panel, outcome, predictors,
                "P6-EQ-DOWN" if horizon == 5 and suffix == "primary" else f"P6-EQ-DOWN-{horizon}y-{suffix}",
            )
            coefficients["priority"] = priority
            summary["priority"] = priority
            rows.extend([coefficients, summary])
    lpm, lpm_summary, _, _ = _fit(
        panel,
        "equity_negative_year_next_3y",
        ["credit_gap", *BASE_CONTROLS],
        "P6-EQ-negative-year-LPM",
    )
    lpm["priority"] = "SENSITIVITY: linear-probability model, not calibrated"
    lpm_summary["priority"] = "SENSITIVITY: linear-probability model, not calibrated"
    rows.extend([lpm, lpm_summary])
    return pd.concat(rows, ignore_index=True)


def _quartile_and_quadratic(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, outcome in [("equity", ASSET_FORWARD["equity"][5]), ("bond", ASSET_FORWARD["bond"][5]), ("bill", ASSET_FORWARD["bill"][5]), ("housing", ASSET_FORWARD["housing"][5])]:
        sample = _sample(panel, outcome, ["credit_gap"])
        sample = sample.assign(credit_gap_quartile=pd.qcut(sample.credit_gap, 4, labels=False, duplicates="drop") + 1)
        for quartile, group in sample.groupby("credit_gap_quartile", observed=True):
            rows.append({"record_type": "quartile_descriptive", "asset": asset, "term": f"Q{int(quartile)}", "mean_forward_return": group[outcome].mean(), "N": len(group), "countries": group.country_code.nunique(), "first_year": int(group.year.min()), "last_year": int(group.year.max())})
    outcome = ASSET_FORWARD["equity"][5]
    sample = _sample(panel, outcome, ["credit_gap", *BASE_CONTROLS])
    sample["centred_credit_gap"] = sample.credit_gap - sample.credit_gap.mean()
    sample["centred_credit_gap_squared"] = sample.centred_credit_gap**2
    coefficients, summary, _, _ = _fit(
        sample,
        outcome,
        ["centred_credit_gap", "centred_credit_gap_squared", *BASE_CONTROLS],
        "P6-EQ-credit-gap-quadratic",
    )
    for frame, record_type in [(coefficients, "quadratic_model"), (summary, "quadratic_model_summary")]:
        frame = frame.copy()
        frame["record_type"] = record_type
        rows.extend(frame.to_dict("records"))
    return pd.DataFrame(rows)


def _point_in_time_credit_flag(panel: pd.DataFrame, minimum_history: int = 5) -> pd.DataFrame:
    result = panel.sort_values(["country_code", "year"]).copy()
    result["credit_gap_elevated_pit"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    for _, group in result.groupby("country_code", sort=False):
        for index, (row_index, row) in enumerate(group.iterrows()):
            history = group.iloc[:index]["credit_gap"].dropna()
            if len(history) >= minimum_history and pd.notna(row.credit_gap):
                result.loc[row_index, "credit_gap_elevated_pit"] = row.credit_gap >= history.quantile(0.8)
    return result


def _point_in_time_analysis(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _point_in_time_credit_flag(panel)
    rows = []
    for asset, outcomes in ASSET_FORWARD.items():
        for horizon, outcome in outcomes.items():
            sample = data.dropna(subset=["credit_gap_elevated_pit", outcome])
            for elevated, group in sample.groupby("credit_gap_elevated_pit", observed=True):
                rows.append({"asset": asset, "horizon_years": horizon, "classification": "elevated" if elevated else "not_elevated", "outcome": outcome, "mean_forward_real_return": group[outcome].mean(), "N": len(group), "countries": group.country_code.nunique(), "first_year": int(group.year.min()), "last_year": int(group.year.max())})
    comparisons = pd.DataFrame(rows)
    regressions = []
    for asset, outcome in [(asset, outcomes[5]) for asset, outcomes in ASSET_FORWARD.items()]:
        sample = data.dropna(subset=["credit_gap_elevated_pit", outcome])
        sample = sample.copy()
        sample["credit_gap_elevated_pit"] = sample.credit_gap_elevated_pit.astype(float)
        predictors = ["credit_gap_elevated_pit", *BASE_CONTROLS]
        coefficients, summary, _, _ = _fit(
            sample, outcome, predictors, f"P6-{asset.upper()}-PIT-controlled"
        )
        coefficients["priority"] = "SENSITIVITY"
        summary["priority"] = "SENSITIVITY"
        regressions.extend([coefficients, summary])
    return comparisons, pd.concat(regressions, ignore_index=True)


def _crisis_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome = "systemic_crisis_next_3y"
    common = _sample(panel, outcome, ["credit_gap", "dsr_deviation_10y"])
    model_a, model_b = [], []
    identities = []
    for name, predictors in [
        ("P6-CRISIS-A-credit-gap", ["credit_gap"]),
        ("P6-CRISIS-B-credit-gap-DSR", ["credit_gap", "dsr_deviation_10y"]),
    ]:
        coefficients, summary, _, sample = _fit(common, outcome, predictors, name)
        coefficients["priority"] = "PRIMARY" if name == "P6-CRISIS-A-credit-gap" else "SENSITIVITY"
        summary["priority"] = coefficients.iloc[0]["priority"]
        (model_a if name.endswith("credit-gap") else model_b).extend([coefficients, summary])
        identities.append(sample[["country_code", "year"]].sort_values(["country_code", "year"]).reset_index(drop=True))
    if not identities[0].equals(identities[1]):
        raise AssertionError("P6 crisis Model A and B do not have common observations")
    return pd.concat(model_a + model_b, ignore_index=True), common


def _crisis_asset_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = [(asset, horizon, outcome) for asset, horizons in ASSET_FORWARD.items() for horizon, outcome in horizons.items()]
    outcomes.append(("equity", 5, "equity_worst_annual_return_next_5y"))
    for asset, horizon, outcome in outcomes:
        sample = panel.dropna(subset=["systemic_crisis_next_3y", outcome])
        for followed, group in sample.groupby("systemic_crisis_next_3y"):
            rows.append({"asset": asset, "horizon_years": horizon, "outcome": outcome, "classification": "followed_by_crisis_within_3y" if followed else "not_followed_by_crisis_within_3y", "mean": group[outcome].mean(), "median": group[outcome].median(), "N": len(group), "countries": group.country_code.nunique(), "first_year": int(group.year.min()), "last_year": int(group.year.max())})
    return pd.DataFrame(rows)


def _crisis_signal_comparison(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in ["credit_gap", "dsr_deviation_10y"]:
        sample = panel.dropna(subset=["systemic_crisis_next_3y", variable])
        for followed, group in sample.groupby("systemic_crisis_next_3y"):
            rows.append({
                "signal": variable,
                "condition": "followed_by_crisis_within_3y" if followed else "not_followed_by_crisis_within_3y",
                "mean": group[variable].mean(),
                "median": group[variable].median(),
                "std": group[variable].std(),
                "N": len(group),
                "countries": group.country_code.nunique(),
                "first_year": int(group.year.min()),
                "last_year": int(group.year.max()),
            })
    return pd.DataFrame(rows)


def _dsr_secondary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = [
        ("equity", ASSET_FORWARD["equity"][5]),
        ("bond", ASSET_FORWARD["bond"][5]),
        ("housing", ASSET_FORWARD["housing"][5]),
        ("equity_worst_5y", "equity_worst_annual_return_next_5y"),
    ]
    for label, outcome in selected:
        for predictors, model_suffix, priority in [
            (["dsr_deviation_10y"], "parsimonious", "SECONDARY"),
            (["dsr_deviation_10y", *BASE_CONTROLS], "controlled", "SENSITIVITY"),
        ]:
            coefficients, summary, _, _ = _fit(
                panel, outcome, predictors, f"P6-DSR-{label}-5y-{model_suffix}"
            )
            coefficients["priority"] = priority
            summary["priority"] = priority
            rows.extend([coefficients, summary])
    return pd.concat(rows, ignore_index=True)


def _nonoverlap_models(panel: pd.DataFrame) -> pd.DataFrame:
    reduced = nonoverlap_sample(panel)
    rows = []
    specifications = [
        ("equity", ASSET_FORWARD["equity"][5], "P6-EQ-5y-nonoverlap"),
        ("bond", ASSET_FORWARD["bond"][5], "P6-BOND-5y-nonoverlap"),
        ("housing", ASSET_FORWARD["housing"][5], "P6-HOUSING-5y-nonoverlap"),
        ("equity_downside", "equity_worst_annual_return_next_5y", "P6-EQ-DOWN-5y-nonoverlap"),
    ]
    for _, outcome, name in specifications:
        coefficients, summary, _, _ = _fit(reduced, outcome, ["credit_gap"], name)
        coefficients["priority"] = "SENSITIVITY"
        summary["priority"] = "SENSITIVITY"
        rows.extend([coefficients, summary])
    result = pd.concat(rows, ignore_index=True)
    if (result.first_year.dropna().astype(int) % 5 != 0).any() or (result.last_year.dropna().astype(int) % 5 != 0).any():
        raise AssertionError("Non-overlap results contain years outside year % 5 == 0")
    return result


def _credit_gap_figures(panel: pd.DataFrame, point_in_time: pd.DataFrame, figure_dir: Path) -> None:
    scatter_specs = [
        (ASSET_FORWARD["equity"][5], "phase6_credit_gap_equity_5y.png", "Credit gap and subsequent 5-year annualized real equity return"),
        ("equity_worst_annual_return_next_5y", "phase6_credit_gap_equity_worst_5y.png", "Credit gap and worst subsequent annual real equity return"),
    ]
    for outcome, filename, title in scatter_specs:
        sample = panel[["credit_gap", outcome]].dropna()
        design = sm.add_constant(sample.credit_gap.astype(float))
        model = sm.OLS(sample[outcome].astype(float), design).fit()
        x = np.linspace(sample.credit_gap.min(), sample.credit_gap.max(), 120)
        prediction = model.get_prediction(sm.add_constant(pd.Series(x, name="credit_gap"), has_constant="add")).summary_frame()
        figure, axis = plt.subplots(figsize=(8, 5.4))
        axis.scatter(sample.credit_gap, sample[outcome], alpha=0.32, s=18, color="#277b78")
        axis.plot(x, prediction["mean"], color="#b74a3a", linewidth=1.8, label="Linear fit")
        axis.fill_between(x, prediction["mean_ci_lower"], prediction["mean_ci_upper"], color="#b74a3a", alpha=0.16, label="95% CI for mean")
        axis.set_xlabel("BIS credit gap (percentage points of GDP)")
        axis.set_ylabel("Real local-currency return")
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(figure_dir / filename, dpi=160)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    crisis = point_in_time.dropna(subset=["systemic_crisis_next_3y", "credit_gap"])
    groups = [crisis.loc[crisis.systemic_crisis_next_3y.eq(value), "credit_gap"] for value in [0, 1]]
    axis.boxplot(groups, tick_labels=["No crisis in next 3y", "Crisis in next 3y"], showfliers=False)
    axis.set_ylabel("BIS credit gap (percentage points of GDP)")
    axis.set_title("Credit-gap distribution before documented crises")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(figure_dir / "phase6_credit_gap_by_crisis.png", dpi=160)
    plt.close(figure)

    mean_rows = []
    for asset, horizons in ASSET_FORWARD.items():
        for horizon, outcome in horizons.items():
            grouped = point_in_time.dropna(subset=["credit_gap_elevated_pit", outcome]).groupby("credit_gap_elevated_pit", observed=True)[outcome].mean()
            for elevated, value in grouped.items():
                mean_rows.append({"asset": asset, "horizon": f"{horizon}y", "condition": "elevated" if elevated else "not elevated", "mean": value})
    means = pd.DataFrame(mean_rows)
    pivot = means.pivot(index=["asset", "horizon"], columns="condition", values="mean")
    figure, axis = plt.subplots(figsize=(9, 5.6))
    positions = np.arange(len(pivot))
    width = 0.38
    axis.bar(positions - width / 2, pivot.get("not elevated"), width, label="Not elevated", color="#277b78")
    axis.bar(positions + width / 2, pivot.get("elevated"), width, label="Point-in-time elevated", color="#c66b42")
    axis.set_xticks(positions, [f"{asset} {horizon}" for asset, horizon in pivot.index], rotation=35, ha="right")
    axis.set_ylabel("Mean annualized real return")
    axis.set_title("Forward asset performance by point-in-time credit-gap status")
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "phase6_asset_performance_by_credit_gap.png", dpi=160)
    plt.close(figure)


def _coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [*ASSET_REAL.values(), "credit_gap", "dsr_deviation_10y", "crisisJST"]
    rows = []
    for variable in columns:
        available = panel.dropna(subset=[variable])
        rows.append({"variable": variable, "observations": len(available), "countries": available.country_code.nunique(), "first_year": available.year.min(), "last_year": available.year.max(), "panel_observations": len(panel), "coverage_pct": 100 * len(available) / len(panel) if len(panel) else np.nan})
    return pd.DataFrame(rows)


def run_phase6(
    raw_path: Path = JST_RAW_FILE,
    structural_path: Path = STRUCTURAL_FINANCIAL_PANEL_FILE,
    processed_path: Path = MARKET_PANEL_FILE,
    tables_dir: Path = PROJECT_ROOT / "outputs" / "tables",
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures",
    coverage_dir: Path = PROJECT_ROOT / "outputs" / "coverage",
) -> dict[str, pd.DataFrame]:
    jst, metadata = load_jst(raw_path)
    metadata_path = raw_path.parent / "source_metadata.json"
    metadata_path.write_text(json.dumps(metadata.iloc[0].to_dict(), indent=2), encoding="utf-8")
    mapping = country_mapping()
    structural = pd.read_csv(structural_path)
    market = build_market_panel(structural, jst)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    market.to_csv(processed_path, index=False)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(coverage_dir / "phase6_jst_country_mapping.csv", index=False)
    coverage = _coverage_table(market)
    coverage.to_csv(coverage_dir / "phase6_market_coverage.csv", index=False)
    stats, correlations = asset_descriptives(market)
    stats.to_csv(tables_dir / "phase6_asset_descriptive_statistics.csv", index=False)
    correlations.to_csv(tables_dir / "phase6_asset_correlations.csv", index=False)

    asset_models = _asset_models(market)
    asset_models.to_csv(tables_dir / "phase6_credit_gap_asset_models.csv", index=False)
    downside = _downside_models(market)
    downside.to_csv(tables_dir / "phase6_equity_downside_models.csv", index=False)
    nonlinear = _quartile_and_quadratic(market)
    nonlinear.to_csv(tables_dir / "phase6_credit_gap_nonlinearity.csv", index=False)
    point_in_time = _point_in_time_credit_flag(market)
    pit_comparison, pit_models = _point_in_time_analysis(market)
    pit_comparison.to_csv(tables_dir / "phase6_point_in_time_asset_comparison.csv", index=False)
    pit_models.to_csv(tables_dir / "phase6_point_in_time_models.csv", index=False)
    crisis_models, crisis_common = _crisis_models(market)
    crisis_models.to_csv(tables_dir / "phase6_crisis_models.csv", index=False)
    crisis_asset = _crisis_asset_table(market)
    crisis_asset.to_csv(tables_dir / "phase6_crisis_asset_performance.csv", index=False)
    crisis_signals = _crisis_signal_comparison(market)
    crisis_signals.to_csv(tables_dir / "phase6_crisis_signal_comparison.csv", index=False)
    dsr = _dsr_secondary(market)
    dsr.to_csv(tables_dir / "phase6_dsr_secondary_models.csv", index=False)
    nonoverlap = _nonoverlap_models(market)
    nonoverlap.to_csv(tables_dir / "phase6_nonoverlap_results.csv", index=False)

    # Ex-post only: future crisis status is conditioned on to locate the credit-gap association.
    expost, expost_summary, _, _ = _fit(
        market,
        ASSET_FORWARD["equity"][5],
        ["credit_gap", "systemic_crisis_next_3y"],
        "P6-EXPOST-equity5y-credit-gap-crisis",
        allow_ex_post_crisis=True,
    )
    expost["label"] = "EX-POST DECOMPOSITION; NOT POINT-IN-TIME; NOT A TRADING OR BACKTEST MODEL"
    expost_summary["label"] = "EX-POST DECOMPOSITION; NOT POINT-IN-TIME; NOT A TRADING OR BACKTEST MODEL"
    pd.concat([expost, expost_summary], ignore_index=True).to_csv(tables_dir / "phase6_ex_post_decomposition.csv", index=False)

    common_rows = []
    for asset in ["equity", "bond"]:
        outcome = ASSET_FORWARD[asset][5]
        sample = _sample(market, outcome, ["credit_gap", *BASE_CONTROLS])
        identities = []
        for name, predictors in [("A-credit-only", ["credit_gap"]), ("B-controls", ["credit_gap", *BASE_CONTROLS])]:
            coefficients, summary, _, fitted = _fit(sample, outcome, predictors, f"P6-{asset.upper()}-common-{name}")
            identities.append(fitted[["country_code", "year"]].sort_values(["country_code", "year"]).reset_index(drop=True))
            coefficients["sample_identity_verified"] = True
            summary["sample_identity_verified"] = True
            common_rows.extend([coefficients, summary])
        if not identities[0].equals(identities[1]):
            raise AssertionError(f"{asset} common-sample models differ")
    for name, predictors in [("A-credit-gap", ["credit_gap"]), ("B-credit-gap-DSR", ["credit_gap", "dsr_deviation_10y"])]:
        coefficients, summary, _, _ = _fit(crisis_common, "systemic_crisis_next_3y", predictors, f"P6-CRISIS-common-{name}")
        coefficients["sample_identity_verified"] = True
        summary["sample_identity_verified"] = True
        common_rows.extend([coefficients, summary])
    pd.concat(common_rows, ignore_index=True).to_csv(tables_dir / "phase6_common_sample_results.csv", index=False)

    primary_terms = []
    for frame, model, term in [
        (asset_models, "P6-EQ", "credit_gap"),
        (downside, "P6-EQ-DOWN", "credit_gap"),
        (asset_models, "P6-BOND", "credit_gap"),
        (crisis_models, "P6-CRISIS-A-credit-gap", "credit_gap"),
    ]:
        selected = frame.loc[(frame.model == model) & (frame.term == term)].copy()
        primary_terms.append(selected)
    primary = pd.concat(primary_terms, ignore_index=True)
    primary["priority"] = "PRIMARY"
    primary.to_csv(tables_dir / "phase6_prespecified_summary.csv", index=False)
    _credit_gap_figures(market, point_in_time, figures_dir)
    return {
        "jst": jst,
        "mapping": mapping,
        "market_panel": market,
        "coverage": coverage,
        "asset_statistics": stats,
        "asset_models": asset_models,
        "downside": downside,
        "nonlinearity": nonlinear,
        "point_in_time": pit_comparison,
        "crisis_models": crisis_models,
        "crisis_asset_performance": crisis_asset,
        "crisis_signal_comparison": crisis_signals,
        "dsr": dsr,
        "nonoverlap": nonoverlap,
        "primary": primary,
    }


if __name__ == "__main__":
    results = run_phase6()
    print(f"Generated Phase 6 outputs: {len(results['market_panel']):,} market-panel country-years")