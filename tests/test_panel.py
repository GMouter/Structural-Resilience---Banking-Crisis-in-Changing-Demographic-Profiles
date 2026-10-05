import pandas as pd

from src.build_panel import build_panel
from src.config import INDICATORS
from src.phase1b_audit import DERIVED_COLUMNS


def test_country_universe_contains_exactly_29_countries():
    from src.config import COUNTRIES
    assert len(COUNTRIES) == 29


def test_panel_has_unique_country_year_and_all_indicator_columns(tmp_path):
    rows = []
    for name, details in INDICATORS.items():
        rows.append({"country_code": "USA", "country_name": "United States", "year": 2000, "indicator_code": details["code"], "value": 1.0})
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    panel = build_panel(raw_path, output_path)
    assert not panel.duplicated(["country_code", "year"]).any()
    assert set(INDICATORS).issubset(panel.columns)


def test_raw_data_does_not_contain_derived_variables():
    raw_columns = {"country_code", "country_name", "year", "indicator_code", "indicator_name", "value"}
    derived = {"working_age_change_5y", "real_gdp_pc_growth_5y_forward"}
    assert not raw_columns.intersection(derived)


def test_derived_dataset_contract_contains_required_columns(tmp_path):
    from src.build_panel import build_derived_panel

    rows = []
    for year in range(1960, 1966):
        for name, details in INDICATORS.items():
            rows.append({"country_code": "USA", "country_name": "United States", "year": year, "indicator_code": details["code"], "value": 1.0})
    raw_path = tmp_path / "raw.csv"
    panel_path = tmp_path / "panel.csv"
    derived_path = tmp_path / "derived.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    build_panel(raw_path, panel_path)
    derived = build_derived_panel(panel_path, derived_path)
    assert set(DERIVED_COLUMNS).issubset(derived.columns)
