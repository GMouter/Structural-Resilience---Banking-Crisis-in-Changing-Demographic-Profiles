"""Penn World Table 10.0 ingestion for the 29-country research universe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import COUNTRIES, PWT_FILE, PWT_DIR

PWT_VERSION = "10.0"
PWT_SOURCE_URL = "https://www.rug.nl/ggdc/docs/pwt100.xlsx"
PWT_DOWNLOAD_DATE = "2026-09-23"
PWT_VARIABLES = {
    "rgdpo": "Output-side real GDP at chained PPPs (in mil. 2017US$)",
    "emp": "Number of persons engaged (in millions)",
    "cn": "Capital stock at current PPPs (in mil. 2017US$)",
    "ctfp": "TFP level at current PPPs (USA=1)",
    "hc": "Human capital index, based on years of schooling and returns to education",
}


def read_pwt(path: Path = PWT_FILE) -> pd.DataFrame:
    columns = ["countrycode", "country", "year", *PWT_VARIABLES]
    data = pd.read_excel(path, sheet_name="Data", usecols=columns)
    data = data[data["countrycode"].isin(COUNTRIES)].copy()
    data = data.rename(columns={"countrycode": "country_code", "country": "pwt_country_name"})
    data["year"] = pd.to_numeric(data["year"], errors="raise").astype(int)
    if data.duplicated(["country_code", "year"]).any():
        raise ValueError("PWT data contains duplicate country-year rows")
    return data.sort_values(["country_code", "year"]).reset_index(drop=True)


def add_pwt_derived(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["pwt_output_per_worker"] = result["rgdpo"] / result["emp"]
    result["pwt_capital_stock_per_worker"] = result["cn"] / result["emp"]
    result["pwt_output_per_worker_growth"] = pd.NA
    for _, group in result.groupby("country_code", sort=False):
        current = group.set_index("year")["pwt_output_per_worker"]
        growth = pd.Series(
            [((current.get(year) / current.get(year - 1)) - 1) * 100 if year - 1 in current.index and pd.notna(current.get(year)) and pd.notna(current.get(year - 1)) else pd.NA for year in group["year"]],
            index=group.index,
            dtype="Float64",
        )
        result.loc[group.index, "pwt_output_per_worker_growth"] = growth.to_numpy()
    result["productivity_growth_5y_backward"] = pd.NA
    for _, group in result.groupby("country_code", sort=False):
        growth = group.set_index("year")["pwt_output_per_worker_growth"]
        values = []
        for year in group["year"]:
            window = [growth.get(year - offset) for offset in range(5)]
            values.append(sum(window) / 5 if all(pd.notna(value) for value in window) else pd.NA)
        result.loc[group.index, "productivity_growth_5y_backward"] = pd.Series(values, index=group.index, dtype="Float64").to_numpy()
    return result


def load_pwt(path: Path = PWT_FILE) -> pd.DataFrame:
    return add_pwt_derived(read_pwt(path))