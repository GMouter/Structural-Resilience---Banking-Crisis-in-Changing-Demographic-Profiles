import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase8_analysis import (
    BOOTSTRAP_SEED,
    MINIMUM_PRIOR_HISTORY,
    R1_WEIGHTS,
    R2_WEIGHTS,
    STATIC_WEIGHTS,
    TRANSACTION_COST_RATE,
    add_point_in_time_credit_gap_state,
    bootstrap_country_differences,
    build_strategy_returns,
    performance_metrics,
    run_phase8,
    state_from_percentile,
    turnover,
)


def _panel(years=range(2000, 2020), countries=("AAA", "BBB", "CCC")):
    rows = []
    for country_index, country in enumerate(countries):
        for offset, year in enumerate(years):
            rows.append({
                "country_code": country,
                "year": year,
                "credit_gap": float(offset + country_index),
                "equity_real_return": [-0.25, 0.12, 0.08, -0.04, 0.15][offset % 5],
                "bond_real_return": [0.08, -0.02, 0.04, 0.05, -0.01][offset % 5],
                "bill_real_return": [0.01, 0.01, 0.015, 0.02, 0.01][offset % 5],
                "crisisJST": int(country == "AAA" and year in {2012, 2017}),
                "systemic_crisis_next_3y": int(country == "AAA" and year in {2009, 2014}),
            })
    return pd.DataFrame(rows)


def test_historical_percentiles_use_prior_observations_only_and_minimum_history():
    panel = _panel(range(2000, 2010), countries=("AAA",))
    changed = panel.copy()
    changed.loc[changed.year.eq(2009), "credit_gap"] = 1e9

    first = add_point_in_time_credit_gap_state(panel).set_index("year")
    second = add_point_in_time_credit_gap_state(changed).set_index("year")

    assert MINIMUM_PRIOR_HISTORY == 5
    assert pd.isna(first.loc[2004, "vulnerability_state"])
    assert first.loc[2005, "vulnerability_state"] == "HIGH"
    pd.testing.assert_series_equal(
        first.loc[first.index <= 2008, "credit_gap_prior_percentile"],
        second.loc[second.index <= 2008, "credit_gap_prior_percentile"],
    )
    assert first.loc[first.index < 2005, "credit_gap_prior_percentile"].isna().all()


def test_signal_at_t_changes_only_t_plus_one_weights_and_crisis_does_not_drive_weights():
    panel = _panel(range(2000, 2018), countries=("AAA",))
    first = build_strategy_returns(panel)
    changed_crises = panel.copy()
    changed_crises["crisisJST"] = 1 - changed_crises.crisisJST
    changed_crises["systemic_crisis_next_3y"] = 1 - changed_crises.systemic_crisis_next_3y
    second = build_strategy_returns(changed_crises)

    assert (first.return_year == first.signal_year + 1).all()
    columns = [f"{strategy}_{asset}_weight" for strategy in ["R1", "R2"] for asset in ["equity", "bond", "bill"]]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert (first.R1_equity_weight <= 0.6).all()


def test_frozen_weights_sum_to_one_are_long_only_and_benchmarks_are_fixed():
    for state, weights in R1_WEIGHTS.items():
        assert weights == {
            "NORMAL": {"equity": 0.6, "bond": 0.2, "bill": 0.2},
            "ELEVATED": {"equity": 0.45, "bond": 0.30, "bill": 0.25},
            "HIGH": {"equity": 0.30, "bond": 0.40, "bill": 0.30},
        }[state]
        assert sum(weights.values()) == pytest.approx(1.0)
        assert min(weights.values()) >= 0
    assert R2_WEIGHTS == {
        "NORMAL": {"equity": 0.6, "bond": 0.2, "bill": 0.2},
        "HIGH": {"equity": 0.30, "bond": 0.40, "bill": 0.30},
    }
    assert STATIC_WEIGHTS == {
        "B1": {"equity": 1.0, "bond": 0.0, "bill": 0.0},
        "B2": {"equity": 0.6, "bond": 0.4, "bill": 0.0},
        "B3": {"equity": 0.6, "bond": 0.2, "bill": 0.2},
    }


def test_frozen_state_cutoffs_are_not_tuned():
    assert state_from_percentile(0.799999) == "NORMAL"
    assert state_from_percentile(0.80) == "ELEVATED"
    assert state_from_percentile(0.899999) == "ELEVATED"
    assert state_from_percentile(0.90) == "HIGH"
    assert state_from_percentile(None) is None


def test_turnover_and_transaction_cost_formula():
    old = {"equity": 0.6, "bond": 0.2, "bill": 0.2}
    new = {"equity": 0.3, "bond": 0.4, "bill": 0.3}

    traded = turnover(old, new)

    assert traded == pytest.approx(0.3)
    assert TRANSACTION_COST_RATE == pytest.approx(0.001)
    assert traded * TRANSACTION_COST_RATE == pytest.approx(0.0003)


def test_dynamic_returns_are_next_year_and_static_weights_rebalance_annually():
    panel = _panel(range(2000, 2020), countries=("AAA", "BBB"))
    result = build_strategy_returns(panel)

    assert (result.return_year == result.signal_year + 1).all()
    for strategy in ["B1", "B2", "B3", "R1", "R2", "B4"]:
        weights = result[[f"{strategy}_{asset}_weight" for asset in ["equity", "bond", "bill"]]]
        assert np.allclose(weights.sum(axis=1), 1.0)
        assert (weights >= 0).all().all()
    for strategy, expected in STATIC_WEIGHTS.items():
        for asset in ["equity", "bond", "bill"]:
            assert result[f"{strategy}_{asset}_weight"].eq(expected[asset]).all()
    assert result.groupby("country_code").B1_turnover.sum().eq(0).all()
    assert result.groupby("country_code").B3_turnover.sum().gt(0).all()
    assert result.groupby("country_code").B3_transaction_cost.sum().gt(0).all()
    assert result.groupby("country_code").B4_turnover.sum().gt(0).all()


def test_transaction_costs_reduce_dynamic_net_return_and_first_entry_is_not_charged():
    panel = _panel(range(2000, 2018), countries=("AAA",))
    result = build_strategy_returns(panel)

    assert result.iloc[0].R1_turnover == 0
    assert result.iloc[0].R1_transaction_cost == 0
    assert np.allclose(result.R1_net_return, result.R1_gross_return - result.R1_transaction_cost)
    assert (result.R1_transaction_cost == result.R1_turnover * TRANSACTION_COST_RATE).all()


def test_b4_static_weights_equal_realized_average_r1_weights_on_matched_sample():
    result = build_strategy_returns(_panel())

    for asset in ["equity", "bond", "bill"]:
        assert result[f"B4_{asset}_weight"].iloc[0] == pytest.approx(result[f"R1_{asset}_weight"].mean())
        assert result[f"B4_{asset}_weight"].nunique() == 1
    assert result.attrs["b4_weights"] == pytest.approx({
        asset: result[f"R1_{asset}_weight"].mean() for asset in ["equity", "bond", "bill"]
    })


def test_missing_asset_return_year_is_not_filled_and_resets_turnover_history():
    panel = _panel(range(2000, 2010), countries=("AAA",))
    panel.loc[panel.year.eq(2007), "bond_real_return"] = np.nan
    result = build_strategy_returns(panel)

    assert 2007 not in set(result.return_year)
    assert result.loc[result.return_year.eq(2008), "R1_turnover"].iloc[0] == 0


def test_annual_frequency_drawdown_uses_compounded_wealth():
    returns = pd.Series([0.10, -0.20, 0.10])
    years = pd.Series([2000, 2001, 2002])

    metrics = performance_metrics(returns, years)

    wealth = np.cumprod([1.1, 0.8, 1.1])
    expected_drawdown = min(wealth[1] / max(1.1, wealth[0]) - 1, wealth[2] / max(1.1, wealth[0], wealth[1]) - 1)
    assert metrics["max_drawdown_annual_frequency"] == pytest.approx(expected_drawdown)
    assert metrics["worst_year"] == pytest.approx(-0.20)
    assert metrics["p05_annual_return"] == pytest.approx(np.quantile(returns, 0.05))


def test_missing_year_breaks_drawdown_path_instead_of_forward_filling():
    metrics = performance_metrics(pd.Series([-0.5, 0.2]), pd.Series([2000, 2002]))

    assert metrics["max_drawdown_annual_frequency"] == pytest.approx(-0.5)
    assert pd.isna(metrics["real_cagr"])


def test_country_bootstrap_is_reproducible_and_resamples_country_rows():
    rows = []
    for index, country in enumerate(["AAA", "BBB", "CCC"]):
        rows.append({"country_code": country, "strategy": "R1", "annualized_return": index / 100, "worst_year": -index / 100, "p05_annual_return": -index / 200, "max_drawdown_annual_frequency": -index / 10})
        rows.append({"country_code": country, "strategy": "B3", "annualized_return": 0.01, "worst_year": -0.05, "p05_annual_return": -0.03, "max_drawdown_annual_frequency": -0.15})
    metrics = pd.DataFrame(rows)

    first = bootstrap_country_differences(metrics, replicates=100, seed=BOOTSTRAP_SEED)
    second = bootstrap_country_differences(metrics, replicates=100, seed=BOOTSTRAP_SEED)

    pd.testing.assert_frame_equal(first, second)
    assert set(first.unit_of_resampling) == {"country"}
    assert set(first.seed) == {BOOTSTRAP_SEED}


def test_phase8_source_contains_no_optimization_or_future_signal_construction():
    source = Path("src/phase8_analysis.py").read_text(encoding="utf-8").lower()

    assert "scipy.optimize" not in source
    assert "minimize(" not in source
    assert "gridsearch" not in source
    assert "credit_gap_prior_percentile" in source
    assert "year + 1" in source


def test_pipeline_preserves_phase1_to7_processed_files_and_writes_phase8_outputs(tmp_path):
    processed = Path("data/processed")
    originals = [path for path in processed.glob("*.csv") if path.name != "phase8_portfolio_panel.csv"]
    hashes_before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}

    source = tmp_path / "market_validation_panel.csv"
    _panel().to_csv(source, index=False)
    result = run_phase8(
        panel_path=source,
        output_panel_path=tmp_path / "processed" / "phase8_portfolio_panel.csv",
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
        bootstrap_replicates=50,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

    assert hashes_before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}
    for filename in [
        "phase8_country_results.csv", "phase8_strategy_summary.csv",
        "phase8_crisis_performance.csv", "phase8_false_alert_cost.csv",
        "phase8_missed_crises.csv", "phase8_correct_alert_crises.csv",
        "phase8_matched_returns.csv", "phase8_sample_identity.csv",
        "phase8_country_bootstrap.csv", "phase8_leave_one_country_out.csv",
        "phase8_time_period_sensitivity.csv", "phase8_zero_cost_sensitivity.csv",
    ]:
        assert (tmp_path / "tables" / filename).exists()
    assert len(list((tmp_path / "figures").glob("phase8_*.png"))) == 5
    identity = pd.read_csv(tmp_path / "tables" / "phase8_sample_identity.csv")
    assert identity.same_sample_as_B3.all()
    assert not result["strategy_returns"].duplicated(["country_code", "return_year"]).any()