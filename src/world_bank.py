"""Small World Bank Indicators API V2 client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import COUNTRIES, END_YEAR, INDICATORS, RAW_FILE, RAW_DIR, START_YEAR

API_URL = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
USER_AGENT = "structural-resilience-research/0.1 (data engineering)"


def parse_response(payload: Any, indicator_code: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("World Bank response is missing metadata or observations")
    metadata, observations = payload[0], payload[1]
    if not isinstance(metadata, dict) or not isinstance(observations, list):
        raise ValueError("World Bank response has an unexpected JSON shape")

    parsed = []
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("World Bank observation is not an object")
        country = observation.get("country") or {}
        country_code = observation.get("countryiso3code")
        year = observation.get("date")
        if not country_code or year is None:
            continue
        try:
            year = int(year)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid World Bank year: {year!r}") from exc
        parsed.append(
            {
                "country_code": country_code,
                "country_name": country.get("value"),
                "year": year,
                "indicator_code": indicator_code,
                "indicator_name": observation.get("indicator", {}).get("value"),
                "value": observation.get("value"),
                "observation_status": observation.get("obs_status"),
                "source_id": metadata.get("sourceid"),
            }
        )
    return parsed


def fetch_indicator(
    country_codes: list[str],
    indicator_code: str,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    if not country_codes:
        return []
    client = session or requests.Session()
    params = {"format": "json", "per_page": 1000, "date": f"{start_year}:{end_year}"}
    url = API_URL.format(countries=";".join(country_codes), indicator=indicator_code)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        params["page"] = page
        response = client.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"World Bank returned malformed JSON for {indicator_code}") from exc
        page_rows = parse_response(payload, indicator_code)
        rows.extend(page_rows)
        metadata = payload[0]
        pages = int(metadata.get("pages", 1))
        if page >= pages:
            break
        page += 1
    return rows


def download_raw(
    output_path: Path = RAW_FILE,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    session: requests.Session | None = None,
    timeout: int = 120,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    country_codes = list(COUNTRIES)
    for indicator in INDICATORS.values():
        rows.extend(fetch_indicator(country_codes, indicator["code"], start_year, end_year, session=session, timeout=timeout))
    frame = pd.DataFrame(rows, columns=[
        "country_code", "country_name", "year", "indicator_code", "indicator_name", "value",
        "observation_status", "source_id",
    ])
    if frame.empty:
        raise ValueError("World Bank returned no observations")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download World Bank raw observations")
    parser.add_argument("--output", type=Path, default=RAW_FILE)
    args = parser.parse_args()
    downloaded = download_raw(args.output)
    print(f"Downloaded {len(downloaded):,} raw observations to {args.output}")
