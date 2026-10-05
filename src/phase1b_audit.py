"""Small descriptive Phase 1B persistence and coverage audit."""

from __future__ import annotations

import pandas as pd

from .build_panel import build_derived_panel
from .config import COVERAGE_DIR, DERIVED_PANEL_FILE, PANEL_FILE

DERIVED_COLUMNS = [
    "working_age_change_5y",
    "working_age_change_10y",
    "old_age_dependency_change_5y",
    "old_age_dependency_change_10y",
    "fertility_change_5y",
    "real_gdp_pc_growth_5y_backward",
    "real_gdp_pc_growth_5y_forward",
    "investment_gdp_5y_backward_avg",
    "inflation_5y_backward_avg",
    "current_account_gdp_5y_backward_avg",
]

TRAJECTORY_YEARS = [1960, 1990, 2010, 2025]
TRAJECTORY_COUNTRIES = ["JPN", "CHN", "KOR", "IND", "NGA"]


def make_trajectory_table(derived: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country_code in TRAJECTORY_COUNTRIES:
        country = derived[derived["country_code"] == country_code].set_index("year")
        for year in TRAJECTORY_YEARS:
            row = {"country_code": country_code, "year": year}
            for column in ["working_age_share", "old_age_dependency", "fertility"]:
                row[column] = country.at[year, column] if year in country.index else pd.NA
            rows.append(row)
        latest = country.loc[2025] if 2025 in country.index else pd.Series(dtype="object")
        rows[-1]["working_age_change_10y"] = latest.get("working_age_change_10y", pd.NA)
        rows[-1]["old_age_dependency_change_10y"] = latest.get("old_age_dependency_change_10y", pd.NA)
    return pd.DataFrame(rows)


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_columns = pd.read_csv(PANEL_FILE, nrows=0).columns
    previously_persisted = set(DERIVED_COLUMNS).issubset(base_columns)
    if previously_persisted:
        raise ValueError("Base panel unexpectedly already contains derived variables")
    derived = build_derived_panel()
    trajectory = make_trajectory_table(derived)
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(COVERAGE_DIR / "demographic_trajectory_examples.csv", index=False)
    return derived, trajectory


if __name__ == "__main__":
    derived, _ = run_audit()
    print(f"Derived panel: {len(derived):,} rows x {len(derived.columns):,} columns")
    print(f"Saved {DERIVED_PANEL_FILE}")