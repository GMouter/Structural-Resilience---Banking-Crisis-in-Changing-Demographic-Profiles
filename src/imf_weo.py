"""IMF WEO DataMapper historical fiscal ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .config import COUNTRIES, END_YEAR, IMF_DIR, IMF_FISCAL_FILE, START_YEAR

WEO_RELEASE = "WEO October 2025 DataMapper API release"
WEO_SOURCE_URL = "https://www.imf.org/external/datamapper/api/v1/"
WEO_HISTORICAL_LAST_YEAR = 2024
WEO_INDICATORS = {
    "government_debt_gdp": ("GGXWDG_NGDP", "General government gross debt", "% of GDP"),
    "fiscal_balance_gdp": ("GGXCNL_NGDP", "General government net lending/borrowing", "% of GDP"),
}


def fetch_weo_indicator(code: str, session: requests.Session | None = None) -> pd.DataFrame:
    response = (session or requests).get(WEO_SOURCE_URL + code, timeout=60)
    response.raise_for_status()
    payload = response.json()
    values = payload.get("values", {}).get(code, {})
    rows = []
    for country_code, observations in values.items():
        if country_code not in COUNTRIES:
            continue
        for year, value in observations.items():
            year = int(year)
            if START_YEAR <= year <= min(END_YEAR, WEO_HISTORICAL_LAST_YEAR):
                rows.append({"country_code": country_code, "year": year, "indicator_code": code, "value": value, "weo_release": WEO_RELEASE, "is_forecast": False})
    return pd.DataFrame(rows)


def download_fiscal(output_path: Path = IMF_FISCAL_FILE) -> pd.DataFrame:
    frames = []
    IMF_DIR.mkdir(parents=True, exist_ok=True)
    for name, (code, _, _) in WEO_INDICATORS.items():
        response = requests.get(WEO_SOURCE_URL + code, timeout=60)
        response.raise_for_status()
        (IMF_DIR / f"weo_oct2025_{name}_{code}.json").write_bytes(response.content)
        frames.append(fetch_weo_indicator(code))
    result = pd.concat(frames, ignore_index=True)
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    frame = download_fiscal()
    print(f"Saved {len(frame):,} historical IMF WEO observations")