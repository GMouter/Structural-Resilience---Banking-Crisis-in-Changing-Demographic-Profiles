"""Build the Phase 4 structural-financial panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BIS_FINANCIAL_FILE, IMF_FISCAL_FILE, PHASE3_PANEL_FILE, STRUCTURAL_FINANCIAL_PANEL_FILE


def _calendar_lag(group: pd.DataFrame, column: str, years: int) -> pd.Series:
    values = group.set_index("year")[column]
    return pd.Series([values.get(year) - values.get(year - years) if year - years in values.index and pd.notna(values.get(year)) and pd.notna(values.get(year - years)) else pd.NA for year in group["year"]], index=group.index, dtype="Float64")


def _calendar_window(group: pd.DataFrame, column: str, offsets: range, minimum: int) -> pd.Series:
    values = group.set_index("year")[column]
    results = []
    for year in group["year"]:
        window = [values.get(year + offset) for offset in offsets]
        observed = [value for value in window if pd.notna(value)]
        results.append(sum(observed) / len(observed) if len(observed) >= minimum else pd.NA)
    return pd.Series(results, index=group.index, dtype="Float64")


def build_structural_financial_panel(output_path=STRUCTURAL_FINANCIAL_PANEL_FILE):
    panel = pd.read_csv(PHASE3_PANEL_FILE)
    imf = pd.read_csv(IMF_FISCAL_FILE).pivot(index=["country_code", "year"], columns="indicator_code", values="value").reset_index().rename(columns={"GGXWDG_NGDP": "government_debt_gdp", "GGXCNL_NGDP": "fiscal_balance_gdp"})
    bis = pd.read_csv(BIS_FINANCIAL_FILE).pivot(index=["country_code", "year"], columns="variable", values="value").reset_index()
    merged = panel.merge(imf, on=["country_code", "year"], how="left", validate="one_to_one").merge(bis, on=["country_code", "year"], how="left", validate="one_to_one")
    merged = merged.sort_values(["country_code", "year"])
    for variable, output in [("government_debt_gdp", "government_debt_change_5y"), ("private_credit_gdp", "private_credit_change_5y")]:
        merged[output] = pd.NA
        for _, group in merged.groupby("country_code", sort=False):
            merged.loc[group.index, output] = _calendar_lag(group, variable, 5).to_numpy()
    merged["credit_gap_3y_avg"] = pd.NA
    merged["real_house_price_growth_5y"] = pd.NA
    merged["dsr_deviation_10y"] = pd.NA
    for _, group in merged.groupby("country_code", sort=False):
        merged.loc[group.index, "credit_gap_3y_avg"] = _calendar_window(group, "credit_gap", range(-2, 1), 3).to_numpy()
        prices = group.set_index("year")["real_house_price_index"]
        growth = [((prices.get(year) / prices.get(year - 5)) - 1) * 100 if year - 5 in prices.index and pd.notna(prices.get(year)) and pd.notna(prices.get(year - 5)) else pd.NA for year in group["year"]]
        merged.loc[group.index, "real_house_price_growth_5y"] = pd.Series(growth, index=group.index, dtype="Float64").to_numpy()
        merged.loc[group.index, "dsr_deviation_10y"] = (group["debt_service_ratio"] - _calendar_window(group, "debt_service_ratio", range(-10, 0), 5)).to_numpy()
    if merged.duplicated(["country_code", "year"]).any():
        raise ValueError("Structural-financial panel contains duplicate country-years")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def phase4_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    excluded = {"country_code", "country_name", "year", "pwt_country_name"}
    rows = []
    for variable in [column for column in panel.columns if column not in excluded]:
        usable = panel.loc[panel[variable].notna(), ["country_code", "year"]]
        rows.append({
            "variable": variable,
            "country_code": "ALL",
            "first_year": usable["year"].min(),
            "last_year": usable["year"].max(),
            "non_null_observations": len(usable),
            "coverage_pct": 100 * len(usable) / len(panel),
        })
        for country_code, group in panel.groupby("country_code"):
            country_usable = group.loc[group[variable].notna(), "year"]
            rows.append({
                "variable": variable,
                "country_code": country_code,
                "first_year": country_usable.min(),
                "last_year": country_usable.max(),
                "non_null_observations": int(country_usable.notna().sum()),
                "coverage_pct": 100 * country_usable.notna().sum() / len(group),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    panel = build_structural_financial_panel()
    print(f"Built structural-financial panel with {len(panel):,} rows and {len(panel.columns):,} columns")