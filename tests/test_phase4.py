import pandas as pd
import pytest
from pathlib import Path

from src.phase4 import _calendar_lag, _calendar_window
from src.phase4_analysis import _future_outcomes, add_point_in_time_flags


def test_exact_calendar_change_does_not_use_row_position():
    group = pd.DataFrame({"year": [2000, 2002, 2005], "value": [1.0, 2.0, 6.0]})
    group["value"] = group.value
    assert pd.isna(_calendar_lag(group, "value", 5).iloc[1])
    assert _calendar_lag(group, "value", 5).iloc[2] == 5


def test_dsr_reference_window_excludes_current_year():
    group = pd.DataFrame({"year": range(2000, 2012), "dsr": [10.0] * 11 + [30.0]})
    reference = _calendar_window(group.rename(columns={"dsr": "value"}), "value", range(-10, 0), 5)
    assert reference.iloc[-1] == 10


def test_future_outcomes_are_calendar_keyed():
    panel = pd.DataFrame({"country_code": ["A", "A", "A"], "year": [2000, 2002, 2003], "real_gdp_pc_growth": [1.0, 2.0, 3.0]})
    result = _future_outcomes(panel)
    assert pd.isna(result.loc[result.year == 2000, "future_1y_real_gdp_pc_growth"]).all()


def test_point_in_time_flags_do_not_use_future_observations():
    panel = pd.DataFrame({"country_code": ["A"] * 7, "year": range(2000, 2007), "working_age_change_10y": [0, 1, 2, 3, 4, 5, -5], "credit_gap": [0, 1, 2, 3, 4, 5, 100], "dsr_deviation_10y": [0, 1, 2, 3, 4, 5, 100], "real_house_price_growth_5y": [0, 1, 2, 3, 4, 5, 100]})
    result = add_point_in_time_flags(panel)
    assert not result.loc[result.year == 2000, "credit_gap_elevated"].iloc[0]
    assert result.loc[result.year == 2006, "credit_gap_elevated"].iloc[0]


def test_phase4_panel_has_unique_country_years_and_preserves_prior_panels():
    panel = pd.read_csv("data/processed/structural_financial_panel.csv")
    assert not panel.duplicated(["country_code", "year"]).any()
    assert panel.country_code.nunique() == 29
    assert Path("data/processed/structural_panel.csv").exists()
    assert Path("data/processed/structural_panel_derived.csv").exists()
    assert Path("data/processed/structural_panel_phase3.csv").exists()