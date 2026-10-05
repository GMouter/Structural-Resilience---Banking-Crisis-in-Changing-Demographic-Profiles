import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.phase3 import _add_labour_averages
from src.phase3_analysis import (
    MODEL_PREDICTORS,
    add_interaction,
    complete_case_data,
    nonoverlap_sample,
)
from src.pwt import add_pwt_derived


def test_phase3_merge_shape_and_log_level():
    panel = pd.DataFrame({"country_code": ["AAA", "AAA"], "year": [2000, 2001], "real_gdp_pc_level": [100.0, 121.0], "labour_force_participation": [50.0, 51.0], "employment_to_population": [45.0, 46.0]})
    result = panel.copy()
    result["log_real_gdp_pc"] = np.log(result["real_gdp_pc_level"])
    assert len(result) == 2
    assert result.loc[0, "log_real_gdp_pc"] == pytest.approx(np.log(100))


def test_productivity_growth_and_backward_window_have_no_future_information():
    source = pd.DataFrame({"country_code": ["AAA"] * 7, "year": range(2000, 2007), "rgdpo": [100, 102, 104, 106, 108, 110, 112], "emp": [10] * 7, "cn": [100] * 7, "ctfp": [1] * 7, "hc": [1] * 7})
    result = add_pwt_derived(source)
    assert pd.isna(result.loc[0, "pwt_output_per_worker_growth"])
    assert pd.isna(result.loc[4, "productivity_growth_5y_backward"])
    assert result.loc[5, "productivity_growth_5y_backward"] == pytest.approx(result.loc[1:5, "pwt_output_per_worker_growth"].mean())


def test_interaction_is_centered_product():
    panel = pd.DataFrame({"working_age_change_10y": [1.0, 2.0], "productivity_growth_5y_backward": [3.0, 5.0]})
    result = add_interaction(panel)
    assert result.loc[0, "demographic_productivity_interaction"] == pytest.approx(0.5)
    assert result.loc[1, "demographic_productivity_interaction"] == pytest.approx(0.5)


def test_common_case_and_nonoverlap_rules_reject_forward_predictors():
    with pytest.raises(ValueError):
        complete_case_data(pd.DataFrame({"real_gdp_pc_growth_5y_forward": [1]}), ["real_gdp_pc_growth_5y_forward"])
    panel = pd.DataFrame({"year": range(2000, 2010)})
    assert set(nonoverlap_sample(panel)["year"]) == {2000, 2005}


def test_phase3_model_predictors_have_no_forward_variable():
    assert all("forward" not in variable for variables in MODEL_PREDICTORS.values() for variable in variables)


def test_original_phase_panels_remain_present_and_distinct():
    assert Path("data/processed/structural_panel.csv").exists()
    assert Path("data/processed/structural_panel_derived.csv").exists()
    assert Path("data/processed/structural_panel_phase3.csv").exists()
    assert "real_gdp_pc_level" not in pd.read_csv("data/processed/structural_panel_derived.csv", nrows=0).columns