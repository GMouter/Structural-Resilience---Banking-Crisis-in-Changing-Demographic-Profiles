"""Clean selected official BIS bulk datasets and annualise quarterly series."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .config import BIS_DIR, BIS_FINANCIAL_FILE, COUNTRIES

BIS_SOURCE_FILES = {
    "private_credit_gdp": "WS_TC_csv_col.zip",
    "credit_gap": "WS_CREDIT_GAP_csv_col.zip",
    "debt_service_ratio": "WS_DSR_csv_col.zip",
    "real_house_price_index": "WS_SPP_csv_col.zip",
}
BIS_CODE_MAP = {"JP": "JPN", "US": "USA", "CA": "CAN", "AU": "AUS", "NZ": "NZL", "DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP", "NL": "NLD", "SE": "SWE", "NO": "NOR", "DK": "DNK", "CH": "CHE", "PL": "POL", "KR": "KOR", "CN": "CHN", "IN": "IND", "ID": "IDN", "SG": "SGP", "BR": "BRA", "MX": "MEX", "CL": "CHL", "ZA": "ZAF", "TR": "TUR", "IL": "ISR", "SA": "SAU"}


def _read_zip(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        return pd.read_csv(archive.open(archive.namelist()[0]), low_memory=False)


def _melt_quarterly(frame: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    period_columns = [column for column in frame.columns if isinstance(column, str) and "-Q" in column]
    result = frame.melt(id_vars=id_columns, value_vars=period_columns, var_name="period", value_name="value")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result["year"] = result["period"].str[:4].astype(int)
    result["quarter"] = result["period"].str[-1].astype(int)
    return result.dropna(subset=["value"])


def clean_bis(output_path: Path = BIS_FINANCIAL_FILE, raw_dir: Path = BIS_DIR) -> pd.DataFrame:
    rows = []
    total = _read_zip(raw_dir / BIS_SOURCE_FILES["private_credit_gdp"])
    total = total[(total.TC_BORROWERS == "P") & (total.TC_LENDERS == "A") & (total.VALUATION == "M") & (total.UNIT_TYPE == "770") & (total.TC_ADJUST == "A")]
    rows.append(_melt_quarterly(total, ["BORROWERS_CTY"]).assign(variable="private_credit_gdp"))
    gap = _read_zip(raw_dir / BIS_SOURCE_FILES["credit_gap"])
    gap = gap[(gap.TC_BORROWERS == "P") & (gap.TC_LENDERS == "A") & (gap.CG_DTYPE == "C")]
    rows.append(_melt_quarterly(gap, ["BORROWERS_CTY"]).assign(variable="credit_gap"))
    dsr = _read_zip(raw_dir / BIS_SOURCE_FILES["debt_service_ratio"])
    dsr = dsr[dsr.DSR_BORROWERS == "P"]
    rows.append(_melt_quarterly(dsr, ["BORROWERS_CTY"]).assign(variable="debt_service_ratio"))
    spp = _read_zip(raw_dir / BIS_SOURCE_FILES["real_house_price_index"])
    spp = spp[(spp.VALUE == "R") & (spp.UNIT_MEASURE == 628)]
    rows.append(_melt_quarterly(spp, ["REF_AREA"]).rename(columns={"REF_AREA": "BORROWERS_CTY"}).assign(variable="real_house_price_index"))
    result = pd.concat(rows, ignore_index=True)
    result["country_code"] = result["BORROWERS_CTY"].map(BIS_CODE_MAP)
    result = result[result.country_code.isin(COUNTRIES)].copy()
    annual = []
    for (country_code, year, variable), group in result.groupby(["country_code", "year", "variable"]):
        if variable in {"private_credit_gdp", "credit_gap"}:
            selected = group[group.quarter == 4]
            value = selected.value.iloc[-1] if not selected.empty else pd.NA
        else:
            value = group.value.mean() if group.quarter.nunique() >= 2 else pd.NA
        annual.append({"country_code": country_code, "year": year, "variable": variable, "value": value, "annualisation": "Q4" if variable in {"private_credit_gdp", "credit_gap"} else "quarterly_mean_min_2"})
    result = pd.DataFrame(annual)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    frame = clean_bis()
    print(f"Saved {len(frame):,} annual BIS observations")