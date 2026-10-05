import pandas as pd
import pytest

from src.panel_analysis import (
    MODEL_PREDICTORS,
    complete_case_data,
    fit_fixed_effects_regression,
    nonoverlap_sample,
)


def sample_panel():
    rows = []
    for country_index, country in enumerate(["AAA", "BBB", "CCC"]):
        for year in range(2000, 2015):
            rows.append({
                "country_code": country, "year": year,
                "real_gdp_pc_growth_5y_forward": year / 10 + country_index,
                "working_age_change_10y": year / 100,
                "old_age_dependency_change_10y": year / 200,
                "investment_gdp_5y_backward_avg": year / 20,
                "inflation_5y_backward_avg": year / 30,
            })
    return pd.DataFrame(rows)


def test_regression_rejects_forward_predictor():
    with pytest.raises(ValueError, match="dependent outcome"):
        complete_case_data(sample_panel(), ["real_gdp_pc_growth_5y_forward"])


def test_fixed_effects_result_contains_expected_fields():
    coefficients, summary = fit_fixed_effects_regression(sample_panel(), "Model 3A", MODEL_PREDICTORS["Model 3A"])
    assert {"estimate", "clustered_se", "t_stat", "p_value", "ci_lower", "ci_upper"}.issubset(coefficients.columns)
    assert summary.iloc[0]["fixed_effects"] == "country, year"
    assert summary.iloc[0]["se_method"] == "country-clustered"


def test_missing_values_use_model_specific_complete_cases():
    panel = sample_panel()
    panel.loc[0, "inflation_5y_backward_avg"] = pd.NA
    data = complete_case_data(panel, MODEL_PREDICTORS["Model 1"])
    assert len(data) == len(panel)
    data_with_controls = complete_case_data(panel, MODEL_PREDICTORS["Model 2"])
    assert len(data_with_controls) == len(panel) - 1


def test_nonoverlap_sample_uses_deterministic_calendar_rule():
    reduced = nonoverlap_sample(sample_panel())
    assert set(reduced["year"] % 5) == {0}
    assert len(reduced) == 3 * 3


def test_source_panel_columns_remain_unchanged():
    source = pd.read_csv("data/processed/structural_panel.csv", nrows=0)
    assert "real_gdp_pc_growth_5y_forward" not in source.columns