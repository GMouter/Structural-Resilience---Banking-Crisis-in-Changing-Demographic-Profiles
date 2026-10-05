"""Calendar-safe within-country derived variables."""

from __future__ import annotations

import pandas as pd


def _calendar_difference(group: pd.DataFrame, column: str, years: int) -> pd.Series:
    values = group.set_index("year")[column]
    return pd.Series(
        [values.get(year) - values.get(year - years) if year - years in values.index and pd.notna(values.get(year)) and pd.notna(values.get(year - years)) else pd.NA for year in group["year"]],
        index=group.index,
        dtype="Float64",
    )


def _window_mean(group: pd.DataFrame, column: str, offsets: range) -> pd.Series:
    values = group.set_index("year")[column]
    result = []
    for year in group["year"]:
        window = [values.get(year + offset) for offset in offsets]
        result.append(sum(window) / len(window) if len(window) == len([value for value in window if pd.notna(value)]) else pd.NA)
    return pd.Series(result, index=group.index, dtype="Float64")


def add_transformations(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"country_code", "year", "working_age_share", "old_age_dependency", "fertility", "real_gdp_pc_growth", "investment_gdp", "inflation", "current_account_gdp"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing columns: {sorted(missing)}")
    result = panel.sort_values(["country_code", "year"]).copy()
    for column in ["working_age_share", "old_age_dependency", "fertility", "real_gdp_pc_growth", "investment_gdp", "inflation", "current_account_gdp"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    transformations = {
        "working_age_change_5y": lambda group: _calendar_difference(group, "working_age_share", 5),
        "working_age_change_10y": lambda group: _calendar_difference(group, "working_age_share", 10),
        "old_age_dependency_change_5y": lambda group: _calendar_difference(group, "old_age_dependency", 5),
        "old_age_dependency_change_10y": lambda group: _calendar_difference(group, "old_age_dependency", 10),
        "fertility_change_5y": lambda group: _calendar_difference(group, "fertility", 5),
        "real_gdp_pc_growth_5y_backward": lambda group: _window_mean(group, "real_gdp_pc_growth", range(-4, 1)),
        "real_gdp_pc_growth_5y_forward": lambda group: _window_mean(group, "real_gdp_pc_growth", range(1, 6)),
        "investment_gdp_5y_backward_avg": lambda group: _window_mean(group, "investment_gdp", range(-4, 1)),
        "inflation_5y_backward_avg": lambda group: _window_mean(group, "inflation", range(-4, 1)),
        "current_account_gdp_5y_backward_avg": lambda group: _window_mean(group, "current_account_gdp", range(-4, 1)),
    }
    for name, calculation in transformations.items():
        result[name] = pd.Series(pd.NA, index=result.index, dtype="Float64")
        for _, group in result.groupby("country_code", sort=False):
            result.loc[group.index, name] = calculation(group).to_numpy()
    return result
