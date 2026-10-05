"""Coverage reports for the raw long data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import COVERAGE_DIR, COUNTRIES, END_YEAR, INDICATORS, RAW_FILE, START_YEAR


def generate_coverage(raw_path: Path = RAW_FILE, output_dir: Path = COVERAGE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(raw_path)
    rows = []
    for country_code in COUNTRIES:
        for indicator_name, details in INDICATORS.items():
            values = raw[(raw.country_code == country_code) & (raw.indicator_code == details["code"])]
            years = pd.to_numeric(values.loc[values.value.notna(), "year"], errors="coerce").dropna()
            first = int(years.min()) if not years.empty else pd.NA
            last = int(years.max()) if not years.empty else pd.NA
            expected = int(last - first + 1) if not years.empty else 0
            rows.append({"country_code": country_code, "indicator": indicator_name, "first_year": first, "last_year": last, "non_null_observations": int(years.nunique()), "expected_years_between_first_last": expected, "coverage_pct": (100 * years.nunique() / expected if expected else 0.0)})
    detail = pd.DataFrame(rows)
    summary = detail.groupby("country_code", as_index=False).agg(non_null_observations=("non_null_observations", "sum"), indicator_count=("indicator", "count"), mean_coverage_pct=("coverage_pct", "mean"))
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "indicator_coverage.csv", index=False)
    summary.to_csv(output_dir / "country_coverage_summary.csv", index=False)
    weakest_countries = summary.sort_values("mean_coverage_pct").head(3).country_code.tolist()
    weakest_indicators = detail.groupby("indicator")["coverage_pct"].mean().sort_values().head(3).index.tolist()
    late_starts = detail.groupby("indicator")["first_year"].min().sort_values(na_position="last").tail(3).index.tolist()
    print(f"Best-covered countries: {', '.join(summary.sort_values('mean_coverage_pct', ascending=False).head(3).country_code)}")
    print(f"Weakest-covered countries: {', '.join(weakest_countries)}")
    print(f"Weakest indicators: {', '.join(weakest_indicators)}")
    print(f"Indicators with major late starts: {', '.join(late_starts)}")
    return detail, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate World Bank coverage reports")
    parser.add_argument("--raw", type=Path, default=RAW_FILE)
    parser.add_argument("--output-dir", type=Path, default=COVERAGE_DIR)
    args = parser.parse_args()
    generate_coverage(args.raw, args.output_dir)
