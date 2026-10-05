import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase6_analysis import (
    ASSET_FORWARD,
    ASSET_REAL,
    _fit,
    _point_in_time_credit_flag,
    add_forward_outcomes,
    build_market_panel,
    country_mapping,
    load_jst,
    run_phase6,
)


def _fixture_panel(years=range(2000, 2008)):
    return pd.DataFrame({
        "country_code": ["AAA"] * len(years),
        "year": list(years),
        "equity_real_return": [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.05, -0.03][:len(years)],
        "bond_real_return": [0.02] * len(years),
        "bill_real_return": [0.01] * len(years),
        "housing_real_return": [0.04] * len(years),
        "crisisJST": [0, 0, 1, 0, 0, 0, 0, 0][:len(years)],
        "credit_gap": list(range(len(years))),
        "dsr_deviation_10y": list(range(len(years))),
        "working_age_change_10y": list(range(len(years))),
        "log_real_gdp_pc": list(range(len(years))),
        "investment_gdp_5y_backward_avg": list(range(len(years))),
        "inflation_5y_backward_avg": list(range(len(years))),
    })


def test_nominal_to_real_uses_multiplicative_deflation_and_cpi_index_change(tmp_path):
    rows = []
    for year, cpi, nominal in [(2000, 100, 0.10), (2001, 110, 0.21), (2002, 121, -0.10)]:
        rows.append({
            "year": year, "country": "A", "iso": "AUS", "cpi": cpi,
            "crisisJST": 0, "eq_tr": nominal, "bond_tr": nominal,
            "bill_rate": nominal, "housing_tr": nominal,
        })
    path = tmp_path / "source.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)

    result, _ = load_jst(path)

    row = result.loc[result.year == 2001].iloc[0]
    assert row.inflation_jst == pytest.approx(0.1)
    assert row.equity_real_return == pytest.approx((1.21 / 1.1) - 1)
    assert row.equity_real_return != pytest.approx(0.21 - 0.10)


def test_compounded_forward_returns_and_worst_year_exclude_signal_year():
    panel = _fixture_panel()

    result = add_forward_outcomes(panel).set_index("year")

    expected_3y = ((1 - 0.2) * (1 + 0.3) * (1 + 0.4)) ** (1 / 3) - 1
    expected_5y = ((1 - 0.2) * (1 + 0.3) * (1 + 0.4) * (1 - 0.1) * (1 + 0.2)) ** (1 / 5) - 1
    assert result.loc[2000, ASSET_FORWARD["equity"][1]] == pytest.approx(-0.2)
    assert result.loc[2000, ASSET_FORWARD["equity"][3]] == pytest.approx(expected_3y)
    assert result.loc[2000, ASSET_FORWARD["equity"][5]] == pytest.approx(expected_5y)
    assert result.loc[2000, "equity_worst_annual_return_next_3y"] == pytest.approx(-0.2)
    assert result.loc[2000, "equity_worst_annual_return_next_5y"] == pytest.approx(-0.2)
    assert result.loc[2000, "equity_negative_year_next_3y"] == 1


def test_missing_calendar_year_invalidates_all_crossing_forward_windows():
    panel = _fixture_panel([2000, 2001, 2003, 2004, 2005, 2006, 2007])

    result = add_forward_outcomes(panel).set_index("year")

    assert pd.isna(result.loc[2001, ASSET_FORWARD["equity"][3]])
    assert pd.isna(result.loc[2001, ASSET_FORWARD["equity"][5]])
    assert pd.isna(result.loc[2001, "equity_worst_annual_return_next_3y"])


def test_crisis_windows_are_strictly_future_and_use_jst_onset_years():
    result = add_forward_outcomes(_fixture_panel()).set_index("year")

    assert result.loc[2001, "systemic_crisis_next_1y"] == 1
    assert result.loc[2000, "systemic_crisis_next_3y"] == 1
    assert result.loc[2002, "systemic_crisis_next_3y"] == 0
    assert result.loc[2002, "systemic_crisis_next_5y"] == 0


def test_missing_calendar_year_invalidates_crisis_horizon():
    panel = _fixture_panel([2000, 2001, 2003, 2004, 2005, 2006, 2007])

    result = add_forward_outcomes(panel).set_index("year")

    assert pd.isna(result.loc[2000, "systemic_crisis_next_3y"])


def test_country_mapping_includes_only_iso3_members_of_project_universe():
    mapping = country_mapping()

    assert mapping.iso3.is_unique
    assert set(mapping.loc[mapping.included, "iso3"]) == {
        "AUS", "CAN", "CHE", "DEU", "DNK", "ESP", "FRA", "GBR", "ITA", "JPN", "NLD", "NOR", "SWE", "USA"
    }
    assert set(mapping.loc[~mapping.included, "iso3"]) == {"BEL", "FIN", "IRL", "PRT"}


def test_market_panel_merge_is_unique_and_does_not_force_unmatched_countries():
    structural = pd.DataFrame({"country_code": ["AUS", "BEL"], "year": [2000, 2000], "credit_gap": [1.0, 2.0]})
    jst = pd.DataFrame({"country_code": ["AUS"], "year": [2000], "jst_country": ["Australia"], "cpi": [10.0], "inflation_jst": [0.1], "crisisJST": [0], "eq_tr": [0.1], "bond_tr": [0.1], "bill_rate": [0.1], "housing_tr": [0.1], **{column: [0.0] for column in ASSET_REAL.values()}})

    result = build_market_panel(structural, jst)

    assert result.country_code.tolist() == ["AUS"]
    assert not result.duplicated(["country_code", "year"]).any()


def test_future_crisis_and_forward_return_cannot_enter_predictive_models():
    panel = pd.concat([_fixture_panel().assign(country_code=country) for country in ["AAA", "BBB", "CCC"]], ignore_index=True)
    panel = add_forward_outcomes(panel)

    with pytest.raises(ValueError, match="Future crisis"):
        _fit(panel, ASSET_FORWARD["equity"][5], ["credit_gap", "systemic_crisis_next_3y"], "bad")
    with pytest.raises(ValueError, match="Forward return"):
        _fit(panel, ASSET_FORWARD["equity"][5], ["credit_gap", ASSET_FORWARD["bond"][3]], "bad")


def test_ex_post_crisis_decomposition_is_explicitly_labelled():
    panel = pd.concat([_fixture_panel().assign(country_code=country) for country in ["AAA", "BBB", "CCC"]], ignore_index=True)
    for country, onset_year, scale in [("AAA", 2002, 1), ("BBB", 2004, 2), ("CCC", 2006, 3)]:
        rows = panel.country_code.eq(country)
        panel.loc[rows, "crisisJST"] = 0
        panel.loc[rows & panel.year.eq(onset_year), "crisisJST"] = 1
        panel.loc[rows, "credit_gap"] = [
            (year - 2000) * scale for year in panel.loc[rows, "year"]
        ]
    panel = add_forward_outcomes(panel)
    coefficients, _, _, _ = _fit(
        panel,
        ASSET_FORWARD["equity"][5],
        ["credit_gap", "systemic_crisis_next_3y"],
        "P6-EXPOST-test",
        allow_ex_post_crisis=True,
    )

    assert coefficients.analysis_type.eq("ex-post decomposition").all()


def test_point_in_time_credit_flag_is_not_changed_by_future_values():
    panel = _fixture_panel(range(2000, 2008))
    extended = panel.copy()
    extended.loc[extended.year == 2007, "credit_gap"] = 100000

    original = _point_in_time_credit_flag(panel).set_index("year")
    altered = _point_in_time_credit_flag(extended).set_index("year")

    assert original.loc[2006, "credit_gap_elevated_pit"] == altered.loc[2006, "credit_gap_elevated_pit"]


def test_jst_fixture_crisis_values_are_not_relabelled_by_loader(tmp_path):
    source = pd.DataFrame({
        "year": [2000, 2001], "country": ["A", "A"], "iso": ["AUS", "AUS"],
        "cpi": [100, 110], "crisisJST": [0, 1], "eq_tr": [0.1, 0.2],
        "bond_tr": [0.01, 0.02], "bill_rate": [0.03, 0.04], "housing_tr": [0.05, 0.06],
    })
    path = tmp_path / "source.xlsx"
    source.to_excel(path, index=False)

    result, _ = load_jst(path)

    assert result.loc[result.year == 2001, "crisisJST"].iloc[0] == 1


def test_full_pipeline_preserves_phase1_to5_files_and_writes_separate_outputs(tmp_path):
    processed = Path("data/processed")
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in processed.glob("*.csv")}
    market_path = tmp_path / "processed" / "market_validation_panel.csv"

    results = run_phase6(
        processed_path=market_path,
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
        coverage_dir=tmp_path / "coverage",
    )

    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in processed.glob("*.csv")}
    assert before == after
    assert market_path.exists()
    assert (tmp_path / "coverage" / "phase6_jst_country_mapping.csv").exists()
    assert (tmp_path / "tables" / "phase6_prespecified_summary.csv").exists()
    assert (tmp_path / "tables" / "phase6_crisis_asset_performance.csv").exists()
    assert (tmp_path / "tables" / "phase6_crisis_signal_comparison.csv").exists()
    assert len(list((tmp_path / "figures").glob("phase6_*.png"))) == 4
    assert not results["market_panel"].duplicated(["country_code", "year"]).any()
    common = pd.read_csv(tmp_path / "tables" / "phase6_common_sample_results.csv")
    model_rows = common.loc[common.term == "__model__"]
    assert model_rows.sample_identity_verified.all()
    assert model_rows.groupby("model").N.nunique().eq(1).all()
    nonoverlap = pd.read_csv(tmp_path / "tables" / "phase6_nonoverlap_results.csv")
    assert (nonoverlap.first_year.dropna().astype(int) % 5 == 0).all()
    assert (nonoverlap.last_year.dropna().astype(int) % 5 == 0).all()