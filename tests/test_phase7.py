import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase7_analysis import (
    ALL_SIGNALS,
    MINIMUM_TRAINING_EVENTS,
    MINIMUM_PRIOR_HISTORY,
    OUTCOME,
    SIGNALS,
    _assign_alert_bands,
    _fit_logistic,
    add_cooldown_flag,
    add_past_crisis_outcome,
    crisis_start_audit,
    event_lead_times,
    expanding_predictions,
    historical_credit_gap_ranks,
    run_phase7,
)


def _panel(start=1980, stop=2021):
    rows = []
    crisis_years = {
        "AAA": [1987, 1991, 1995, 1999, 2015],
        "BBB": [1988, 1992, 1996, 2000, 2008],
        "CCC": [1989, 1993, 1997, 2001, 2012],
    }
    for country_index, country in enumerate(crisis_years):
        for year in range(start, stop):
            crisis = int(year in crisis_years[country])
            credit = (year - start) * (country_index + 1) / 10 + (year % 4) * 0.1
            dsr = (year - start) / 15 + country_index * 0.2
            house = (year % 7) * 0.1 + country_index * 0.05
            next_crisis = int(any(year < event <= year + 3 for event in crisis_years[country]))
            rows.append({
                "country_code": country,
                "year": year,
                "crisisJST": crisis,
                "credit_gap": credit,
                "dsr_deviation_10y": dsr,
                "real_house_price_growth_5y": house,
                OUTCOME: next_crisis,
            })
    return pd.DataFrame(rows)


def test_jst_crisis_start_audit_uses_source_onsets_without_reconstruction():
    panel = _panel()

    events, summary = crisis_start_audit(panel)

    starts = events.loc[events.record_type == "crisis_start"]
    assert len(starts) == 15
    assert starts.groupby("country_code").size().to_dict() == {"AAA": 5, "BBB": 5, "CCC": 5}
    assert set(starts.jst_variable) == {"crisisJST"}
    overall = summary.iloc[0]
    assert overall.crisis_start_count == 15
    assert overall.countries_with_crises == 3
    assert overall.positive_next_3y_country_years > 0
    assert overall.negative_next_3y_country_years > 0


def test_past_and_future_crisis_windows_exclude_current_year():
    panel = _panel()
    labeled = add_past_crisis_outcome(panel).set_index(["country_code", "year"])

    assert labeled.loc[("AAA", 2012), OUTCOME] == 1
    assert labeled.loc[("AAA", 2015), OUTCOME] == 0
    assert labeled.loc[("AAA", 2002), "past_crisis_previous_3y"] == 1
    assert labeled.loc[("AAA", 2012), "past_crisis_previous_3y"] == 0


def test_expanding_window_predictions_are_chronological_with_three_year_label_embargo():
    panel = _panel()

    predictions = expanding_predictions(panel, "S1", SIGNALS["S1"])

    assert not predictions.empty
    assert (predictions.train_last_year <= predictions.year - 4).all()
    assert (predictions.train_unique_crisis_events >= MINIMUM_TRAINING_EVENTS).all()
    assert predictions.year.min() >= 1990
    assert (predictions.train_first_year < predictions.year).all()


def test_scaler_and_fitted_parameters_use_training_rows_only():
    training = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0], OUTCOME: [0, 0, 1, 1], "country_code": ["A", "A", "B", "B"]})
    evaluation = pd.DataFrame({"x": [0.0, 1.0], OUTCOME: [0, 1], "country_code": ["A", "B"]})
    changed_evaluation = evaluation.assign(x=[100000.0, -50000.0], **{OUTCOME: [1, 0]})

    _, first = _fit_logistic(training, evaluation, ["x"])
    _, changed = _fit_logistic(training, changed_evaluation, ["x"])

    np.testing.assert_array_equal(first["scaler_mean"], changed["scaler_mean"])
    np.testing.assert_array_equal(first["scaler_scale"], changed["scaler_scale"])
    np.testing.assert_array_equal(first["fit_parameters"], changed["fit_parameters"])
    assert first["fit_method"].startswith("L2_fallback_C_")


def test_logistic_predictors_reject_future_outcomes():
    panel = _panel()

    with pytest.raises(ValueError, match="Future or placebo outcomes"):
        expanding_predictions(panel, "bad", [OUTCOME])
    with pytest.raises(ValueError, match="Future or placebo outcomes"):
        expanding_predictions(panel, "bad", ["past_crisis_previous_3y"])


def test_common_sample_predictions_have_identical_country_years():
    panel = _panel()
    for _, indices in panel.groupby("country_code").groups.items():
        panel.loc[list(indices)[:4], "dsr_deviation_10y"] = np.nan
    common = panel.dropna(subset=[*SIGNALS["S3"], OUTCOME])
    frames = [expanding_predictions(common, family, predictors) for family, predictors in SIGNALS.items()]
    key_sets = [set(zip(frame.country_code, frame.year)) for frame in frames]
    shared = set.intersection(*key_sets)
    assert shared
    for frame in frames:
        assert set(zip(frame.country_code, frame.year)) == shared


def test_alert_bands_are_ranked_within_each_evaluation_year():
    predictions = pd.DataFrame({
        "family": ["S1"] * 10,
        "year": [2000] * 5 + [2001] * 5,
        "country_code": list("ABCDE") * 2,
        "actual": [0, 1, 0, 0, 1] * 2,
        "probability": [0.1, 0.9, 0.3, 0.4, 0.2, 0.9, 0.1, 0.2, 0.3, 0.4],
    })

    result = _assign_alert_bands(predictions)

    assert result.groupby("year").top_10pct.sum().eq(1).all()
    assert result.groupby("year").top_20pct.sum().eq(1).all()
    assert result.groupby("year").top_30pct.sum().eq(2).all()
    assert result.loc[result.year.eq(2000) & result.country_code.eq("B"), "top_10pct"].iloc[0]


def test_historical_credit_gap_percentile_uses_only_prior_country_observations():
    panel = _panel(1980, 2000)
    altered = panel.copy()
    altered.loc[altered.year.eq(1999), "credit_gap"] = 1e9

    original, _ = historical_credit_gap_ranks(panel)
    changed, _ = historical_credit_gap_ranks(altered)

    assert MINIMUM_PRIOR_HISTORY == 5
    pd.testing.assert_series_equal(
        original.loc[original.year.le(1998), "credit_gap_prior_percentile"].reset_index(drop=True),
        changed.loc[changed.year.le(1998), "credit_gap_prior_percentile"].reset_index(drop=True),
    )
    assert original.loc[original.year.lt(1985), "credit_gap_prior_percentile"].isna().all()


def test_event_level_lead_matching_uses_only_preceding_three_year_predictions():
    panel = _panel()
    predictions = expanding_predictions(panel, "S1", SIGNALS["S1"])
    predictions = _assign_alert_bands(predictions)
    starts = panel.loc[panel.crisisJST.eq(1), ["country_code", "year"]]

    leads, events = event_lead_times(predictions, starts)

    assert len(leads) == len(starts) * 3
    assert leads.loc[leads.prediction_available, "origin_year"].lt(leads.loc[leads.prediction_available, "crisis_start_year"]).all()
    assert set(leads.lead_years.unique()) == {1, 2, 3}
    assert events.loc[events.record_type.eq("crisis_event"), "unique_crisis_event"].all()
    summaries = events.loc[events.record_type.eq("event_level_summary")]
    assert {"event_level_recall_all_documented", "event_level_recall_when_prediction_available"}.issubset(summaries.columns)


def test_cooldown_excludes_crisis_start_and_following_two_years():
    panel = _panel()

    result = add_cooldown_flag(panel)

    event_map = {"AAA": [1987, 1991, 1995, 1999, 2015], "BBB": [1988, 1992, 1996, 2000, 2008], "CCC": [1989, 1993, 1997, 2001, 2012]}
    for country, starts in event_map.items():
        for start in starts:
            for year in [start, start + 1, start + 2]:
                assert result.loc[result.country_code.eq(country) & result.year.eq(year), "crisis_cooldown"].all()
    assert not result.loc[result.year.eq(2004), "crisis_cooldown"].any()


def test_pipeline_preserves_phase1_to6_processed_files_and_writes_phase7_outputs(tmp_path):
    processed = Path("data/processed")
    originals = [path for path in processed.glob("*.csv") if path.name != "phase7_analytical_panel.csv"]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}

    result = run_phase7(
        analytical_path=tmp_path / "processed" / "phase7_analytical_panel.csv",
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
    )

    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}
    assert before == after
    assert (tmp_path / "tables" / "phase7_crisis_audit.csv").exists()
    assert (tmp_path / "tables" / "phase7_oos_predictions.csv").exists()
    assert (tmp_path / "tables" / "phase7_common_sample_comparison.csv").exists()
    assert (tmp_path / "tables" / "phase7_crisis_lead_times.csv").exists()
    assert (tmp_path / "tables" / "phase7_calibration.csv").exists()
    assert (tmp_path / "tables" / "phase7_crisis_cooldown_sensitivity.csv").exists()
    assert (tmp_path / "tables" / "phase7_prespecified_summary.csv").exists()
    assert len(list((tmp_path / "figures").glob("phase7_*.png"))) == 4
    wide = pd.read_csv(tmp_path / "tables" / "phase7_oos_predictions.csv")
    assert {"s1_probability", "s2_probability", "s3_probability"}.issubset(wide.columns)
    assert not result["panel"].duplicated(["country_code", "year"]).any()