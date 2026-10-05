import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.phase5_analysis import (
    BASE_CONTROLS,
    FINANCIAL_VARIABLES,
    _centered_interaction,
    _fit,
    _point_in_time_flags,
    add_forward_outcomes,
    common_sample_results,
    complete_case_data,
    run_phase5,
)
from src.panel_analysis import nonoverlap_sample


def test_forward_outcomes_use_exact_calendar_windows_and_leave_five_year_outcome_unchanged():
    panel = pd.DataFrame({
        "country_code": ["AAA"] * 7,
        "year": range(2000, 2007),
        "real_gdp_pc_growth": [1, 2, 3, 4, 5, 6, 7],
        "real_gdp_pc_growth_5y_forward": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    })
    original_five_year = panel["real_gdp_pc_growth_5y_forward"].copy()

    result = add_forward_outcomes(panel)

    at_2000 = result.loc[result.year == 2000].iloc[0]
    assert at_2000["real_gdp_pc_growth_1y_forward"] == pytest.approx(2.0)
    assert at_2000["real_gdp_pc_growth_3y_forward"] == pytest.approx(3.0)
    pd.testing.assert_series_equal(result["real_gdp_pc_growth_5y_forward"], original_five_year)


def test_missing_calendar_year_invalidates_forward_windows():
    panel = pd.DataFrame({
        "country_code": ["AAA"] * 6,
        "year": [2000, 2001, 2003, 2004, 2005, 2006],
        "real_gdp_pc_growth": [1, 2, 3, 4, 5, 6],
    })

    result = add_forward_outcomes(panel).set_index("year")

    assert pd.isna(result.loc[2000, "real_gdp_pc_growth_3y_forward"])
    assert pd.isna(result.loc[2001, "real_gdp_pc_growth_1y_forward"])
    assert pd.isna(result.loc[2001, "real_gdp_pc_growth_3y_forward"])


def test_forward_outcomes_are_never_predictors():
    with pytest.raises(ValueError, match="dependent outcome"):
        complete_case_data(
            pd.DataFrame({"real_gdp_pc_growth_1y_forward": [1.0]}),
            ["real_gdp_pc_growth_1y_forward"],
        )


def test_interaction_is_centered_on_the_supplied_analytical_sample():
    panel = pd.DataFrame({
        "working_age_change_10y": [1.0, 2.0, 100.0],
        "credit_gap": [20.0, 10.0, 1000.0],
    })
    sample = panel.iloc[:2]

    result, working_mean, financial_mean = _centered_interaction(
        panel, "credit_gap", sample
    )

    assert working_mean == pytest.approx(1.5)
    assert financial_mean == pytest.approx(15.0)
    assert result.loc[0, "working_age_x_credit_gap"] == pytest.approx(-2.5)
    assert result.loc[1, "working_age_x_credit_gap"] == pytest.approx(-2.5)
    assert result.loc[2, "working_age_x_credit_gap"] != pytest.approx(0.0)


def test_point_in_time_thresholds_ignore_future_observations():
    panel = pd.DataFrame({
        "country_code": ["AAA"] * 7,
        "year": range(2000, 2007),
        "working_age_change_10y": [0, 1, 2, 3, 4, 5, -10],
        "credit_gap": [0, 1, 2, 3, 4, 5, 1000],
        "dsr_deviation_10y": [0, 1, 2, 3, 4, 5, 1000],
    })
    extended = panel.copy()
    extended.loc[6, ["working_age_change_10y", "credit_gap", "dsr_deviation_10y"]] = [
        -1000,
        100000,
        100000,
    ]

    original_flags = _point_in_time_flags(panel).set_index("year")
    extended_flags = _point_in_time_flags(extended).set_index("year")

    for flag in [
        "credit_gap_elevated_pit",
        "dsr_elevated_pit",
        "demographic_deterioration_pit",
    ]:
        assert original_flags.loc[2005, flag] == extended_flags.loc[2005, flag]
    assert original_flags.loc[2005, "credit_gap_elevated_pit"]


def test_fitted_models_include_country_year_effects_and_country_clustered_se():
    rows = []
    for country_index, country in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        for year in range(2000, 2010):
            rows.append({
                "country_code": country,
                "year": year,
                "real_gdp_pc_growth_5y_forward": (year - 2000) * 0.1 + country_index,
                "x": (year - 1999) * (country_index + 1) / 10,
            })
    coefficients, summary, _, _ = _fit(pd.DataFrame(rows), "test", ["x"])

    assert "x" in coefficients.term.tolist()
    assert summary.iloc[0]["fixed_effects"] == "country, year"
    assert summary.iloc[0]["se_method"] == "country-clustered"


def test_common_sample_specs_use_identical_observations():
    panel = pd.read_csv("data/processed/structural_financial_panel.csv")

    results = common_sample_results(panel)
    model_rows = results.loc[results.term == "__model__"]

    assert model_rows["common_sample_identity_verified"].all()
    assert model_rows.groupby("family")["N"].nunique().eq(1).all()


def test_nonoverlap_rule_and_financial_units_are_explicit():
    panel = pd.DataFrame({"year": range(2000, 2010)})

    assert set(nonoverlap_sample(panel)["year"] % 5) == {0}
    assert FINANCIAL_VARIABLES["credit_gap"]["unit"] == "percentage points of GDP"
    assert FINANCIAL_VARIABLES["dsr_deviation_10y"]["unit"] == "percentage points of DSR"
    assert FINANCIAL_VARIABLES["real_house_price_growth_5y"]["unit"] == "percent change over five years"
    assert BASE_CONTROLS[0] == "working_age_change_10y"


def test_pipeline_writes_only_phase5_outputs_and_preserves_processed_panels(tmp_path):
    processed = Path("data/processed")
    source_files = sorted(processed.glob("*.csv"))
    hashes_before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
    }

    run_phase5(
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
    )

    hashes_after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
    }
    assert hashes_after == hashes_before
    assert (tmp_path / "tables" / "phase5_prespecified_summary.csv").exists()
    assert (tmp_path / "tables" / "phase5_credit_gap_interaction.csv").exists()
    assert (tmp_path / "tables" / "phase5_dsr_interaction.csv").exists()
    assert (tmp_path / "tables" / "phase5_property_interaction.csv").exists()
    assert (tmp_path / "tables" / "phase5_nonoverlap_results.csv").exists()
    assert (tmp_path / "tables" / "phase5_horizon_sensitivity.csv").exists()
    assert (tmp_path / "tables" / "phase5_point_in_time_episodes.csv").exists()
    assert (tmp_path / "tables" / "phase5_prespecified_summary.csv").exists()
    assert len(list((tmp_path / "figures").glob("phase5_*.png"))) == 3

    interaction = pd.read_csv(tmp_path / "tables" / "phase5_credit_gap_interaction.csv")
    assert "marginal_association" in interaction.columns
    assert set(interaction.loc[interaction.record_type == "marginal_association", "condition"]) == {
        "p25", "median", "p75"
    }
    regression = pd.read_csv(tmp_path / "tables" / "phase5_regression_results.csv")
    assert not regression.loc[regression.term != "__model__", "term"].str.contains(
        "forward", case=False
    ).any()
    nonoverlap = pd.read_csv(tmp_path / "tables" / "phase5_nonoverlap_results.csv")
    assert (nonoverlap["first_year"].dropna().astype(int) % 5 == 0).all()
    assert (nonoverlap["last_year"].dropna().astype(int) % 5 == 0).all()