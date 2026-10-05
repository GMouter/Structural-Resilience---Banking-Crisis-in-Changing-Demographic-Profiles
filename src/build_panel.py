"""Build the country-year panel from long World Bank observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DERIVED_PANEL_FILE, INDICATORS, PANEL_COLUMNS, PANEL_FILE, PROCESSED_DIR, RAW_FILE
from .transformations import add_transformations


def build_panel(raw_path: Path = RAW_FILE, output_path: Path = PANEL_FILE) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    required = {"country_code", "country_name", "year", "indicator_code", "value"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")
    code_to_name = {details["code"]: name for name, details in INDICATORS.items()}
    raw = raw[raw["indicator_code"].isin(code_to_name)].copy()
    raw["indicator"] = raw["indicator_code"].map(code_to_name)
    if raw.duplicated(["country_code", "year", "indicator"]).any():
        raise ValueError("Raw data contains duplicate country-year-indicator observations")
    panel = raw.pivot(index=["country_code", "country_name", "year"], columns="indicator", values="value").reset_index()
    panel.columns.name = None
    for column in INDICATORS:
        if column not in panel:
            panel[column] = pd.NA
    panel = panel[PANEL_COLUMNS].sort_values(["country_code", "year"]).reset_index(drop=True)
    if panel.duplicated(["country_code", "year"]).any():
        raise ValueError("Panel country_code + year is not unique")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    return panel


def build_derived_panel(panel_path: Path = PANEL_FILE, output_path: Path = DERIVED_PANEL_FILE) -> pd.DataFrame:
    panel = pd.read_csv(panel_path)
    derived = add_transformations(panel)
    if derived.duplicated(["country_code", "year"]).any():
        raise ValueError("Derived panel country_code + year is not unique")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(output_path, index=False)
    return derived


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the processed structural panel")
    parser.add_argument("--raw", type=Path, default=RAW_FILE)
    parser.add_argument("--output", type=Path, default=PANEL_FILE)
    args = parser.parse_args()
    panel = build_panel(args.raw, args.output)
    print(f"Built panel with {len(panel):,} rows and {len(panel.columns):,} columns")
