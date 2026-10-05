"""Phase 3 source ingestion, merge, derived variables, and coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import COUNTRIES, DERIVED_PANEL_FILE, END_YEAR, PHASE3_PANEL_FILE, PHASE3_WB_RAW_FILE, START_YEAR
from .pwt import PWT_VARIABLES, load_pwt
from .world_bank import fetch_indicator

WORLD_BANK_PHASE3_INDICATORS = {
    "real_gdp_pc_level": ("NY.GDP.PCAP.KD", "GDP per capita (constant 2015 US$)"),
    "labour_force_participation": ("SL.TLF.CACT.ZS", "Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)"),
    "employment_to_population": ("SL.EMP.TOTL.SP.ZS", "Employment to population ratio, 15+, total (%) (modeled ILO estimate)"),
}


def download_phase3_world_bank(output_path: Path = PHASE3_WB_RAW_FILE) -> pd.DataFrame:
    rows = []
    for _, (code, _) in WORLD_BANK_PHASE3_INDICATORS.items():
        rows.extend(fetch_indicator(list(COUNTRIES), code, START_YEAR, END_YEAR, timeout=120))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No Phase 3 World Bank observations were downloaded")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def _pivot_phase3_world_bank(raw: pd.DataFrame) -> pd.DataFrame:
    code_to_name = {code: name for name, (code, _) in WORLD_BANK_PHASE3_INDICATORS.items()}
    selected = raw[raw["indicator_code"].isin(code_to_name)].copy()
    selected["variable"] = selected["indicator_code"].map(code_to_name)
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    if selected.duplicated(["country_code", "year", "variable"]).any():
        raise ValueError("Phase 3 World Bank data contains duplicate country-year-variable rows")
    return selected.pivot(index=["country_code", "year"], columns="variable", values="value").reset_index()


def _add_labour_averages(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.sort_values(["country_code", "year"]).copy()
    for column in ["labour_force_participation", "employment_to_population"]:
        result[f"{column}_5y_backward_avg"] = pd.NA
        for _, group in result.groupby("country_code", sort=False):
            values = group.set_index("year")[column]
            averages = []
            for year in group["year"]:
                window = [values.get(year - offset) for offset in range(5)]
                averages.append(sum(window) / 5 if all(pd.notna(value) for value in window) else pd.NA)
            result.loc[group.index, f"{column}_5y_backward_avg"] = pd.Series(averages, index=group.index, dtype="Float64").to_numpy()
    return result


def build_phase3_panel(
    derived_path: Path = DERIVED_PANEL_FILE,
    world_bank_raw_path: Path = PHASE3_WB_RAW_FILE,
    pwt_path: Path | None = None,
    output_path: Path = PHASE3_PANEL_FILE,
) -> pd.DataFrame:
    base = pd.read_csv(derived_path)
    wb_raw = pd.read_csv(world_bank_raw_path)
    wb = _pivot_phase3_world_bank(wb_raw)
    pwt = load_pwt(pwt_path) if pwt_path else load_pwt()
    pwt = pwt.drop(columns=["pwt_country_name"])
    merged = base.merge(wb, on=["country_code", "year"], how="left", validate="one_to_one")
    merged = merged.merge(pwt, on=["country_code", "year"], how="left", validate="one_to_one")
    merged["real_gdp_pc_level"] = pd.to_numeric(merged["real_gdp_pc_level"], errors="coerce")
    merged["log_real_gdp_pc"] = np.where(merged["real_gdp_pc_level"] > 0, np.log(merged["real_gdp_pc_level"]), np.nan)
    merged = _add_labour_averages(merged)
    if merged.duplicated(["country_code", "year"]).any():
        raise ValueError("Phase 3 panel contains duplicate country-year rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(["country_code", "year"]).to_csv(output_path, index=False)
    return merged.sort_values(["country_code", "year"]).reset_index(drop=True)


def phase3_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    variables = [column for column in panel.columns if column not in {"country_code", "country_name", "year", "pwt_country_name"}]
    rows = []
    for variable in variables:
        usable = panel.loc[panel[variable].notna(), ["country_code", "year"]]
        rows.append({"variable": variable, "observations": len(usable), "countries": usable["country_code"].nunique(), "first_year": usable["year"].min(), "last_year": usable["year"].max(), "missing_count": int(panel[variable].isna().sum()), "coverage_pct": 100 * len(usable) / len(panel)})
    return pd.DataFrame(rows)