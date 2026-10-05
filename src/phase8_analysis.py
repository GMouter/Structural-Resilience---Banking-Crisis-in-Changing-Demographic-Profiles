"""Phase 8 preregistered experimental resilience portfolio falsification test."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT

MARKET_PANEL = PROJECT_ROOT / "data" / "processed" / "market_validation_panel.csv"
PHASE8_PANEL = PROJECT_ROOT / "data" / "processed" / "phase8_portfolio_panel.csv"
ASSETS = ["equity", "bond", "bill"]
RETURN_COLUMNS = {asset: f"{asset}_real_return" for asset in ASSETS}
MINIMUM_PRIOR_HISTORY = 5
STATE_THRESHOLDS = {"normal_upper": 0.80, "elevated_upper": 0.90}
BOOTSTRAP_SEED = 8042026
BOOTSTRAP_REPLICATES = 2000
TRANSACTION_COST_RATE = 0.001
STATIC_WEIGHTS = {
    "B1": {"equity": 1.0, "bond": 0.0, "bill": 0.0},
    "B2": {"equity": 0.6, "bond": 0.4, "bill": 0.0},
    "B3": {"equity": 0.6, "bond": 0.2, "bill": 0.2},
}
R1_WEIGHTS = {
    "NORMAL": {"equity": 0.6, "bond": 0.2, "bill": 0.2},
    "ELEVATED": {"equity": 0.45, "bond": 0.30, "bill": 0.25},
    "HIGH": {"equity": 0.30, "bond": 0.40, "bill": 0.30},
}
R2_WEIGHTS = {
    "NORMAL": {"equity": 0.6, "bond": 0.2, "bill": 0.2},
    "HIGH": {"equity": 0.30, "bond": 0.40, "bill": 0.30},
}


def add_point_in_time_credit_gap_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank t's credit gap against only earlier observations for that country."""
    if panel.duplicated(["country_code", "year"]).any():
        raise ValueError("Portfolio signal requires unique country-years")
    result = panel.sort_values(["country_code", "year"]).copy()
    result["credit_gap_prior_percentile"] = np.nan
    for _, group in result.groupby("country_code", sort=False):
        history: list[float] = []
        for row_index, row in group.iterrows():
            value = row["credit_gap"]
            if pd.notna(value) and len(history) >= MINIMUM_PRIOR_HISTORY:
                result.loc[row_index, "credit_gap_prior_percentile"] = float(
                    np.mean(np.asarray(history) <= float(value))
                )
            if pd.notna(value):
                history.append(float(value))
    result["vulnerability_state"] = result["credit_gap_prior_percentile"].map(state_from_percentile).astype("string")
    return result


def state_from_percentile(percentile: float | None) -> str | None:
    """Map fixed historical-rank thresholds to the preregistered states."""
    if percentile is None or pd.isna(percentile):
        return None
    if percentile < STATE_THRESHOLDS["normal_upper"]:
        return "NORMAL"
    if percentile < STATE_THRESHOLDS["elevated_upper"]:
        return "ELEVATED"
    return "HIGH"


def turnover(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    """One-way portfolio turnover for fully invested long-only allocations."""
    return 0.5 * sum(abs(new_weights[asset] - old_weights[asset]) for asset in ASSETS)


def _weights_for_state(state: str, strategy: str) -> dict[str, float]:
    if strategy == "R1":
        return R1_WEIGHTS[state].copy()
    if strategy == "R2":
        return R2_WEIGHTS["HIGH" if state == "HIGH" else "NORMAL"].copy()
    raise ValueError(f"Unknown dynamic strategy: {strategy}")


def _assert_portfolio_weights(weights: dict[str, float]) -> None:
    if set(weights) != set(ASSETS):
        raise AssertionError("Portfolio weights must cover equity, bond, and bill")
    if not np.isclose(sum(weights.values()), 1.0):
        raise AssertionError("Portfolio weights must sum to one")
    if any(weight < 0 for weight in weights.values()):
        raise AssertionError("Short positions are prohibited")


def build_strategy_returns(panel: pd.DataFrame, *, transaction_cost_rate: float = TRANSACTION_COST_RATE) -> pd.DataFrame:
    """Apply year-t signal weights to complete real asset returns at t+1."""
    if transaction_cost_rate < 0:
        raise ValueError("Transaction-cost rate must be non-negative")
    data = add_point_in_time_credit_gap_state(panel)
    data = data.sort_values(["country_code", "year"]).copy()
    signal_lookup = data.set_index(["country_code", "year"])["vulnerability_state"]
    return_lookup = data.set_index(["country_code", "year"])[list(RETURN_COLUMNS.values())]
    crisis_lookup = data.set_index(["country_code", "year"])["crisisJST"]
    forward_crisis_lookup = data.set_index(["country_code", "year"])["systemic_crisis_next_3y"]
    rows = []
    for country, group in data.groupby("country_code", sort=False):
        previous_weights: dict[str, dict[str, float] | None] = {name: None for name in [*STATIC_WEIGHTS, "B4", "R1", "R2"]}
        previous_return_year: int | None = None
        for signal_year in group.year.astype(int):
            state = signal_lookup.get((country, signal_year), pd.NA)
            if pd.isna(state):
                continue
            return_year = signal_year + 1
            key = (country, return_year)
            if key not in return_lookup.index:
                continue
            returns = return_lookup.loc[key]
            if returns.isna().any():
                continue
            if previous_return_year is not None and return_year != previous_return_year + 1:
                previous_weights = {name: None for name in previous_weights}
            crisis = crisis_lookup.get(key, np.nan)
            row: dict[str, object] = {
                "country_code": country,
                "signal_year": signal_year,
                "return_year": return_year,
                "credit_gap_prior_percentile": data.loc[
                    data.country_code.eq(country) & data.year.eq(signal_year),
                    "credit_gap_prior_percentile",
                ].iloc[0],
                "vulnerability_state": str(state),
                "crisis_start_return_year": int(crisis == 1) if pd.notna(crisis) else np.nan,
                "systemic_crisis_next_3y": forward_crisis_lookup.get((country, signal_year), np.nan),
                "equity_event_year_return": float(returns[RETURN_COLUMNS["equity"]]),
                **{RETURN_COLUMNS[asset]: float(returns[RETURN_COLUMNS[asset]]) for asset in ASSETS},
            }
            weights_by_strategy = {name: weights.copy() for name, weights in STATIC_WEIGHTS.items()}
            # B4 is set after R1's realized average exposure has been computed.
            weights_by_strategy["R1"] = _weights_for_state(str(state), "R1")
            weights_by_strategy["R2"] = _weights_for_state(str(state), "R2")
            for strategy, weights in weights_by_strategy.items():
                _assert_portfolio_weights(weights)
                previous = previous_weights[strategy]
                traded = turnover(previous, weights) if previous is not None else 0.0
                gross = sum(weights[asset] * float(returns[RETURN_COLUMNS[asset]]) for asset in ASSETS)
                cost = traded * transaction_cost_rate
                row[f"{strategy}_equity_weight"] = weights["equity"]
                row[f"{strategy}_bond_weight"] = weights["bond"]
                row[f"{strategy}_bill_weight"] = weights["bill"]
                row[f"{strategy}_turnover"] = traded
                row[f"{strategy}_transaction_cost"] = cost
                row[f"{strategy}_gross_return"] = gross
                row[f"{strategy}_net_return"] = gross - cost
                ending_value = 1 + gross
                if ending_value <= 0:
                    raise ValueError("Portfolio wealth must remain positive to compute drifted weights")
                previous_weights[strategy] = {
                    asset: weights[asset] * (1 + float(returns[RETURN_COLUMNS[asset]])) / ending_value
                    for asset in ASSETS
                }
            rows.append(row)
            previous_return_year = return_year
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # B4 is the frozen full-sample average realized R1 exposure, a retrospective
    # benchmark that diagnoses static equity exposure rather than an implementable rule.
    average_weights = {
        asset: float(result[f"R1_{asset}_weight"].mean())
        for asset in ASSETS
    }
    total = sum(average_weights.values())
    average_weights = {asset: weight / total for asset, weight in average_weights.items()}
    for asset in ASSETS:
        result[f"B4_{asset}_weight"] = average_weights[asset]
    result["B4_turnover"] = 0.0
    result["B4_transaction_cost"] = 0.0
    result["B4_gross_return"] = sum(
        average_weights[asset] * result[RETURN_COLUMNS[asset]] for asset in ASSETS
    )
    result["B4_turnover"] = 0.0
    result["B4_transaction_cost"] = 0.0
    result["B4_net_return"] = result["B4_gross_return"]
    for _, group in result.groupby("country_code", sort=False):
        prior_drifted_weights: dict[str, float] | None = None
        prior_year: int | None = None
        for row_index, row in group.sort_values("return_year").iterrows():
            year = int(row.return_year)
            if prior_year is not None and year != prior_year + 1:
                prior_drifted_weights = None
            traded = turnover(prior_drifted_weights, average_weights) if prior_drifted_weights is not None else 0.0
            gross = float(row["B4_gross_return"])
            cost = traded * transaction_cost_rate
            result.loc[row_index, "B4_turnover"] = traded
            result.loc[row_index, "B4_transaction_cost"] = cost
            result.loc[row_index, "B4_net_return"] = gross - cost
            if 1 + gross <= 0:
                raise ValueError("B4 wealth must remain positive to compute drifted weights")
            prior_drifted_weights = {
                asset: average_weights[asset] * (1 + float(row[RETURN_COLUMNS[asset]])) / (1 + gross)
                for asset in ASSETS
            }
            prior_year = year
    result.attrs["b4_weights"] = average_weights
    return result


def _contiguous_segments(group: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = group.sort_values("return_year").copy()
    segment_id = ordered.return_year.diff().ne(1).cumsum()
    return [segment.copy() for _, segment in ordered.groupby(segment_id, sort=False)]


def performance_metrics(returns: pd.Series, years: pd.Series) -> dict[str, float]:
    """Compute annual metrics without treating missing calendar years as flat."""
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {key: np.nan for key in ["annualized_return", "real_cagr", "annualized_volatility", "worst_year", "max_drawdown_annual_frequency", "downside_deviation", "sortino_ratio", "p05_annual_return"]}
    wealth_growth = np.prod(1 + values)
    annualized = wealth_growth ** (1 / len(values)) - 1 if np.all(1 + values > 0) else np.nan
    downside = np.sqrt(np.mean(np.minimum(values, 0.0) ** 2))
    sortino = (values.mean() / downside) if downside > 0 else np.nan
    max_drawdown = 0.0
    valid = pd.DataFrame({"return": pd.to_numeric(returns, errors="coerce"), "year": years}).dropna().sort_values("year")
    contiguous = len(valid) <= 1 or valid.year.astype(int).diff().dropna().eq(1).all()
    cagr = annualized if contiguous else np.nan
    for segment in _contiguous_segments(valid.rename(columns={"year": "return_year"})):
        wealth = np.cumprod(1 + segment["return"].to_numpy(dtype=float))
        peaks = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
        drawdowns = wealth / peaks - 1
        if len(drawdowns):
            max_drawdown = min(max_drawdown, float(drawdowns.min()))
    return {
        "annualized_return": annualized,
        "real_cagr": cagr,
        "annualized_volatility": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "worst_year": float(np.min(values)),
        "max_drawdown_annual_frequency": max_drawdown,
        "downside_deviation": float(downside),
        "sortino_ratio": float(sortino),
        "p05_annual_return": float(np.quantile(values, 0.05)),
    }


def country_strategy_metrics(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strategies = [*STATIC_WEIGHTS, "B4", "R1", "R2"]
    for country, group in strategy_returns.groupby("country_code", sort=True):
        for strategy in strategies:
            net = performance_metrics(group[f"{strategy}_net_return"], group.return_year)
            gross = performance_metrics(group[f"{strategy}_gross_return"], group.return_year)
            rows.append({
                "country_code": country,
                "strategy": strategy,
                "N_years": group[f"{strategy}_net_return"].notna().sum(),
                "first_return_year": int(group.loc[group[f"{strategy}_net_return"].notna(), "return_year"].min()),
                "last_return_year": int(group.loc[group[f"{strategy}_net_return"].notna(), "return_year"].max()),
                **net,
                "gross_annualized_return": gross["annualized_return"],
                "gross_worst_year": gross["worst_year"],
                "gross_max_drawdown_annual_frequency": gross["max_drawdown_annual_frequency"],
                "mean_equity_weight": group[f"{strategy}_equity_weight"].mean(),
                "mean_bond_weight": group[f"{strategy}_bond_weight"].mean(),
                "mean_bill_weight": group[f"{strategy}_bill_weight"].mean(),
                "total_turnover": group[f"{strategy}_turnover"].sum(),
                "total_transaction_cost": group[f"{strategy}_transaction_cost"].sum(),
                "elevated_signal_years": int(group.vulnerability_state.eq("ELEVATED").sum()),
                "high_signal_years": int(group.vulnerability_state.eq("HIGH").sum()),
                "documented_crisis_starts": int(group.crisis_start_return_year.eq(1).sum()),
                "annual_frequency_drawdown_note": "Annualized return is the geometric mean per observed return year. CAGR is reported only for contiguous calendar years. Drawdown is computed within contiguous annual-return segments; missing years are not carried forward.",
            })
    return pd.DataFrame(rows)


def _paired_country_differences(country_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for baseline in ["B3", "B4"]:
        left = country_metrics.loc[country_metrics.strategy.eq("R1")].set_index("country_code")
        right = country_metrics.loc[country_metrics.strategy.eq(baseline)].set_index("country_code")
        countries = left.index.intersection(right.index)
        for country in countries:
            rows.append({
                "country_code": country,
                "comparison": f"R1_minus_{baseline}",
                "annualized_return_difference": left.loc[country, "annualized_return"] - right.loc[country, "annualized_return"],
                "worst_year_difference": left.loc[country, "worst_year"] - right.loc[country, "worst_year"],
                "p05_difference": left.loc[country, "p05_annual_return"] - right.loc[country, "p05_annual_return"],
                "max_drawdown_difference": left.loc[country, "max_drawdown_annual_frequency"] - right.loc[country, "max_drawdown_annual_frequency"],
                "matched_country_years": min(left.loc[country, "N_years"], right.loc[country, "N_years"]) if "N_years" in left.columns and "N_years" in right.columns else np.nan,
            })
    return pd.DataFrame(rows)


def pooled_country_summary(country_metrics: pd.DataFrame) -> pd.DataFrame:
    paired = _paired_country_differences(country_metrics)
    rows = []
    for comparison, group in paired.groupby("comparison"):
        rows.append({
            "comparison": comparison,
            "countries": len(group),
            "mean_country_annualized_return_difference": group.annualized_return_difference.mean(),
            "median_country_annualized_return_difference": group.annualized_return_difference.median(),
            "mean_country_worst_year_difference": group.worst_year_difference.mean(),
            "median_country_worst_year_difference": group.worst_year_difference.median(),
            "mean_country_p05_difference": group.p05_difference.mean(),
            "median_country_p05_difference": group.p05_difference.median(),
            "mean_country_max_drawdown_difference": group.max_drawdown_difference.mean(),
            "median_country_max_drawdown_difference": group.max_drawdown_difference.median(),
            "countries_improved_worst_year": int(group.worst_year_difference.gt(0).sum()),
            "countries_improved_p05": int(group.p05_difference.gt(0).sum()),
            "countries_improved_max_drawdown": int(group.max_drawdown_difference.gt(0).sum()),
        })
    return pd.DataFrame(rows)


def bootstrap_country_differences(
    country_metrics: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    paired = _paired_country_differences(country_metrics)
    generator = np.random.default_rng(seed)
    rows = []
    metric_columns = {
        "annualized_return_difference": "annualized_return_difference",
        "worst_year_difference": "worst_year_difference",
        "p05_difference": "p05_difference",
        "max_drawdown_difference": "max_drawdown_difference",
    }
    for comparison, group in paired.groupby("comparison"):
        countries = group.country_code.to_numpy()
        for metric, column in metric_columns.items():
            values = group.set_index("country_code")[column]
            draws = np.empty(replicates, dtype=float)
            for index in range(replicates):
                sampled = generator.choice(countries, size=len(countries), replace=True)
                draws[index] = values.reindex(sampled).mean()
            rows.append({
                "comparison": comparison,
                "metric": metric,
                "estimate_mean_country_difference": values.mean(),
                "ci_2_5": float(np.quantile(draws, 0.025)),
                "ci_97_5": float(np.quantile(draws, 0.975)),
                "bootstrap_replicates": replicates,
                "unit_of_resampling": "country",
                "countries": len(countries),
                "seed": seed,
            })
    return pd.DataFrame(rows)


def leave_one_country_out(country_metrics: pd.DataFrame) -> pd.DataFrame:
    paired = _paired_country_differences(country_metrics)
    rows = []
    for comparison, group in paired.groupby("comparison"):
        for excluded in group.country_code:
            reduced = group.loc[group.country_code.ne(excluded)]
            rows.append({
                "comparison": comparison,
                "excluded_country": excluded,
                "countries_remaining": len(reduced),
                "median_annualized_return_difference": reduced.annualized_return_difference.median(),
                "median_worst_year_difference": reduced.worst_year_difference.median(),
                "median_p05_difference": reduced.p05_difference.median(),
                "median_max_drawdown_difference": reduced.max_drawdown_difference.median(),
                "countries_improved_drawdown": int(reduced.max_drawdown_difference.gt(0).sum()),
                "countries_improved_worst_year": int(reduced.worst_year_difference.gt(0).sum()),
            })
    return pd.DataFrame(rows)


def time_period_sensitivity(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, condition in [("pre_2000", strategy_returns.return_year.lt(2000)), ("2000_onward", strategy_returns.return_year.ge(2000))]:
        data = strategy_returns.loc[condition]
        for country, group in data.groupby("country_code", sort=True):
            for strategy in ["B3", "B4", "R1"]:
                metrics = performance_metrics(group[f"{strategy}_net_return"], group.return_year)
                rows.append({"period": period, "country_code": country, "strategy": strategy, "N_years": len(group), **metrics})
    frame = pd.DataFrame(rows)
    summaries = []
    for period, group in frame.groupby("period"):
        wide = group.pivot(index="country_code", columns="strategy", values=["annualized_return", "worst_year", "p05_annual_return", "max_drawdown_annual_frequency"])
        for baseline in ["B3", "B4"]:
            for metric, column in [("annualized_return", "annualized_return"), ("worst_year", "worst_year"), ("p05_annual_return", "p05_annual_return"), ("max_drawdown_annual_frequency", "max_drawdown_annual_frequency")]:
                delta = wide[(column, "R1")] - wide[(column, baseline)]
                summaries.append({"period": period, "comparison": f"R1_minus_{baseline}", "metric": metric, "countries": int(delta.notna().sum()), "mean_country_difference": delta.mean(), "median_country_difference": delta.median()})
    return pd.concat([frame.assign(record_type="country_result"), pd.DataFrame(summaries).assign(record_type="summary")], ignore_index=True, sort=False)


def crisis_performance(strategy_returns: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    starts = panel.loc[panel.crisisJST.eq(1), ["country_code", "year"]].rename(columns={"year": "crisis_start_year"})
    lookup = strategy_returns.set_index(["country_code", "return_year"])
    rows = []
    for event in starts.itertuples(index=False):
        values = {}
        for offset, label in [(0, "crisis_start_year"), (1, "year_after_crisis_start")]:
            key = (event.country_code, int(event.crisis_start_year) + offset)
            values[label] = lookup.loc[key] if key in lookup.index else None
        for strategy in ["B1", "B2", "B3", "B4", "R1", "R2"]:
            start_row = values["crisis_start_year"]
            after_row = values["year_after_crisis_start"]
            first = start_row[f"{strategy}_net_return"] if start_row is not None else np.nan
            second = after_row[f"{strategy}_net_return"] if after_row is not None else np.nan
            rows.append({
                "country_code": event.country_code,
                "crisis_start_year": int(event.crisis_start_year),
                "strategy": strategy,
                "crisis_start_year_return": first,
                "year_after_crisis_start_return": second,
                "two_year_compounded_return": (1 + first) * (1 + second) - 1 if pd.notna(first) and pd.notna(second) else np.nan,
                "both_years_available": bool(pd.notna(first) and pd.notna(second)),
                "crisis_labels_used_for_weights": False,
            })
    return pd.DataFrame(rows)


def _signal_event_classification(panel: pd.DataFrame, strategy_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_lookup = panel.set_index(["country_code", "year"])["vulnerability_state"]
    return_lookup = strategy_returns.set_index(["country_code", "return_year"])
    starts = panel.loc[panel.crisisJST.eq(1), ["country_code", "year"]]
    missed, correctly_alerted = [], []
    for event in starts.itertuples(index=False):
        prior_states = [state_lookup.get((event.country_code, int(event.year) - lead), pd.NA) for lead in (1, 2, 3)]
        available_states = [state for state in prior_states if pd.notna(state)]
        fully_observed = len(available_states) == 3
        alert = any(state in {"ELEVATED", "HIGH"} for state in available_states)
        record = {
            "country_code": event.country_code,
            "crisis_start_year": int(event.year),
            "prior_three_states": ",".join("NA" if pd.isna(state) else str(state) for state in prior_states),
            "all_three_signal_years_available": fully_observed,
            "correctly_alerted_within_prior_three_years": alert,
            "classification": "missed_normal_all_three" if fully_observed and all(state == "NORMAL" for state in prior_states) else "alerted" if alert else "not_classifiable_or_mixed",
        }
        for strategy in ["B3", "R1"]:
            start_row = return_lookup.loc[(event.country_code, int(event.year))] if (event.country_code, int(event.year)) in return_lookup.index else None
            after_row = return_lookup.loc[(event.country_code, int(event.year) + 1)] if (event.country_code, int(event.year) + 1) in return_lookup.index else None
            first = start_row[f"{strategy}_net_return"] if start_row is not None else np.nan
            second = after_row[f"{strategy}_net_return"] if after_row is not None else np.nan
            record[f"{strategy}_start_year_return"] = first
            record[f"{strategy}_year_after_return"] = second
            record[f"{strategy}_two_year_return"] = (1 + first) * (1 + second) - 1 if pd.notna(first) and pd.notna(second) else np.nan
        event_return = return_lookup.loc[(event.country_code, int(event.year))] if (event.country_code, int(event.year)) in return_lookup.index else None
        record["equity_crisis_start_year_return"] = event_return[RETURN_COLUMNS["equity"]] if event_return is not None else np.nan
        record["B3_performance"] = record["B3_two_year_return"]
        record["R1_performance"] = record["R1_two_year_return"]
        event_return = return_lookup.loc[(event.country_code, int(event.year))] if (event.country_code, int(event.year)) in return_lookup.index else None
        record["equity_crisis_start_year_return"] = event_return[RETURN_COLUMNS["equity"]] if event_return is not None else np.nan
        record["B3_performance"] = record["B3_two_year_return"]
        record["R1_performance"] = record["R1_two_year_return"]
        if record["classification"] == "missed_normal_all_three":
            missed.append(record)
        if alert:
            correctly_alerted.append(record)
    false_alerts = strategy_returns.loc[
        strategy_returns.vulnerability_state.isin(["ELEVATED", "HIGH"])
        & strategy_returns.systemic_crisis_next_3y.eq(0)
    ].copy()
    false_alerts["R1_minus_B3"] = false_alerts.R1_net_return - false_alerts.B3_net_return
    false_alerts["alert_state"] = false_alerts.vulnerability_state
    false_alerts["strategy_return_year"] = false_alerts.return_year
    false_alerts["decision_signal_year"] = false_alerts.signal_year
    return pd.DataFrame(missed), pd.DataFrame(correctly_alerted), false_alerts


def _figure_outputs(strategy_returns: pd.DataFrame, country_metrics: pd.DataFrame, crisis_table: pd.DataFrame, false_alerts: pd.DataFrame, figures_dir: Path) -> None:
    wealth_strategies = ["B1", "B3", "B4", "R1"]
    history = strategy_returns.groupby("country_code").size().sort_values(ascending=False)
    example = history.index[0]
    group = strategy_returns.loc[strategy_returns.country_code.eq(example)].sort_values("return_year")
    figure, axis = plt.subplots(figsize=(9, 5.7))
    for strategy in wealth_strategies:
        values = group[f"{strategy}_net_return"].to_numpy()
        wealth = np.cumprod(1 + values)
        axis.plot(group.return_year, wealth, label=strategy, linewidth=1.7)
    axis.set_title(f"Annual real wealth paths: {example} (longest matched history)")
    axis.set_xlabel("Return year")
    axis.set_ylabel("Growth of 1 unit")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures_dir / "phase8_example_wealth_paths.png", dpi=160)
    plt.close(figure)

    differences = _paired_country_differences(country_metrics)
    for baseline, filename, metric in [
        ("B3", "phase8_r1_vs_b3_downside_by_country.png", "max_drawdown_difference"),
        ("B4", "phase8_r1_vs_b4_downside_by_country.png", "max_drawdown_difference"),
    ]:
        data = differences.loc[differences.comparison.eq(f"R1_minus_{baseline}")].sort_values(metric)
        figure, axis = plt.subplots(figsize=(9, 5.4))
        axis.bar(data.country_code, data[metric], color=np.where(data[metric] > 0, "#287c78", "#bd6046"))
        axis.axhline(0, color="#555555", linewidth=0.9)
        axis.set_ylabel("R1 minus benchmark annual-frequency max drawdown")
        axis.set_title(f"Country downside differences: R1 vs {baseline}")
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        figure.savefig(figures_dir / filename, dpi=160)
        plt.close(figure)

    event = crisis_table.dropna(subset=["two_year_compounded_return"])
    pivot = event.loc[event.strategy.isin(["B3", "R1"])].pivot_table(index=["country_code", "crisis_start_year"], columns="strategy", values="two_year_compounded_return")
    figure, axis = plt.subplots(figsize=(8.4, 5.5))
    if not pivot.empty:
        axis.scatter(pivot.B3, pivot.R1, color="#287c78", alpha=0.8)
        low = float(np.nanmin(pivot[ ["B3", "R1"] ].to_numpy()))
        high = float(np.nanmax(pivot[ ["B3", "R1"] ].to_numpy()))
        axis.plot([low, high], [low, high], linestyle="--", color="#555555")
    axis.set(xlabel="B3 crisis start + following year return", ylabel="R1 crisis start + following year return", title="Crisis-conditional two-year real returns")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures_dir / "phase8_crisis_event_returns.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.6, 5.5))
    if not false_alerts.empty:
        axis.hist(false_alerts["R1_minus_B3"], bins=16, color="#bd6046", alpha=0.85)
        axis.axvline(0, color="#555555", linestyle="--")
    axis.set_xlabel("False-alert R1 minus B3 next-year real return")
    axis.set_ylabel("Country-signal years")
    axis.set_title("False-alert opportunity costs")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures_dir / "phase8_false_alert_costs.png", dpi=160)
    plt.close(figure)


def run_phase8(
    panel_path: Path = MARKET_PANEL,
    output_panel_path: Path = PHASE8_PANEL,
    tables_dir: Path = PROJECT_ROOT / "outputs" / "tables",
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures",
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, pd.DataFrame]:
    panel = pd.read_csv(panel_path)
    panel["year"] = panel.year.astype(int)
    panel = add_point_in_time_credit_gap_state(panel)
    strategy_returns = build_strategy_returns(panel)
    if strategy_returns.empty:
        raise ValueError("No country-years have point-in-time states and complete next-year investable returns")
    output_panel_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_returns.to_csv(output_panel_path, index=False)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    country_metrics = country_strategy_metrics(strategy_returns)
    country_metrics.to_csv(tables_dir / "phase8_country_results.csv", index=False)
    # Match strategy/benchmark observations before summarizing performance differences.
    matched_keys = strategy_returns[["country_code", "return_year"]].drop_duplicates()
    matched = strategy_returns.merge(matched_keys, on=["country_code", "return_year"], how="inner", validate="one_to_one")
    strategy_keys = {
        strategy: set(zip(
            matched.loc[matched[f"{strategy}_net_return"].notna(), "country_code"],
            matched.loc[matched[f"{strategy}_net_return"].notna(), "return_year"],
        ))
        for strategy in ["B1", "B2", "B3", "B4", "R1", "R2"]
    }
    if not all(strategy_keys["B3"] == strategy_keys[strategy] for strategy in strategy_keys):
        raise AssertionError("Portfolio strategies do not share identical country-years")
    matched.to_csv(tables_dir / "phase8_matched_returns.csv", index=False)
    pd.DataFrame([
        {
            "strategy": strategy,
            "matched_observations": len(keys),
            "countries": len({country for country, _ in keys}),
            "same_sample_as_B3": keys == strategy_keys["B3"],
            "first_return_year": min(year for _, year in keys),
            "last_return_year": max(year for _, year in keys),
        }
        for strategy, keys in strategy_keys.items()
    ]).to_csv(tables_dir / "phase8_sample_identity.csv", index=False)
    expected_average_weights = strategy_returns[[f"R1_{asset}_weight" for asset in ASSETS]].mean().to_numpy(dtype=float)
    b4_weights = strategy_returns[[f"B4_{asset}_weight" for asset in ASSETS]].iloc[0].to_numpy(dtype=float)
    if not np.allclose(expected_average_weights, b4_weights):
        raise AssertionError("B4 weights do not equal full matched-sample average R1 weights")
    strategy_means = country_metrics.groupby("strategy").agg(
        countries=("country_code", "nunique"),
        median_annualized_return=("annualized_return", "median"),
        mean_annualized_return=("annualized_return", "mean"),
        median_real_cagr=("real_cagr", "median"),
        mean_real_cagr=("real_cagr", "mean"),
        median_volatility=("annualized_volatility", "median"),
        mean_volatility=("annualized_volatility", "mean"),
        median_sortino=("sortino_ratio", "median"),
        mean_sortino=("sortino_ratio", "mean"),
        median_worst_year=("worst_year", "median"),
        mean_worst_year=("worst_year", "mean"),
        median_max_drawdown=("max_drawdown_annual_frequency", "median"),
        mean_max_drawdown=("max_drawdown_annual_frequency", "mean"),
        median_p05=("p05_annual_return", "median"),
        mean_p05=("p05_annual_return", "mean"),
    ).reset_index()
    strategy_means.to_csv(tables_dir / "phase8_strategy_summary.csv", index=False)
    paired = _paired_country_differences(country_metrics)
    pooled = pooled_country_summary(country_metrics)
    pooled.to_csv(tables_dir / "phase8_pooled_country_evidence.csv", index=False)
    paired.to_csv(tables_dir / "phase8_country_differences.csv", index=False)
    bootstrap = bootstrap_country_differences(country_metrics, bootstrap_replicates, bootstrap_seed)
    bootstrap.to_csv(tables_dir / "phase8_country_bootstrap.csv", index=False)
    loco = leave_one_country_out(country_metrics)
    loco.to_csv(tables_dir / "phase8_leave_one_country_out.csv", index=False)
    periods = time_period_sensitivity(strategy_returns)
    periods.to_csv(tables_dir / "phase8_time_period_sensitivity.csv", index=False)

    crises = crisis_performance(strategy_returns, panel)
    crises.to_csv(tables_dir / "phase8_crisis_performance.csv", index=False)
    missed, alerted, false_alerts = _signal_event_classification(panel, strategy_returns)
    missed.to_csv(tables_dir / "phase8_missed_crises.csv", index=False)
    alerted.to_csv(tables_dir / "phase8_correct_alert_crises.csv", index=False)
    false_alerts.to_csv(tables_dir / "phase8_false_alert_cost.csv", index=False)
    states = panel.loc[panel.vulnerability_state.notna(), "vulnerability_state"].value_counts().rename_axis("state").reset_index(name="signal_country_years")
    states.to_csv(tables_dir / "phase8_signal_state_counts.csv", index=False)
    weight_summary = pd.DataFrame([
        {"asset": asset, "R1_average_realized_weight": expected_average_weights[index], "B4_static_weight": b4_weights[index]}
        for index, asset in enumerate(ASSETS)
    ])
    weight_summary.to_csv(tables_dir / "phase8_average_weights.csv", index=False)
    cost_rows = []
    for strategy in ["B1", "B2", "B3", "B4", "R1", "R2"]:
        cost_rows.append({
            "strategy": strategy,
            "mean_annual_turnover": strategy_returns[f"{strategy}_turnover"].mean(),
            "total_turnover": strategy_returns[f"{strategy}_turnover"].sum(),
            "total_transaction_cost": strategy_returns[f"{strategy}_transaction_cost"].sum(),
            "transaction_cost_bps_per_one_way_turnover": TRANSACTION_COST_RATE * 10000,
            "initial_entry_cost_included": False,
            "turnover_basis": "Target weights versus prior year's realized drifted weights; reset after data gaps.",
        })
    pd.DataFrame(cost_rows).to_csv(tables_dir / "phase8_transaction_costs.csv", index=False)
    zero_cost_returns = build_strategy_returns(panel, transaction_cost_rate=0.0)
    zero_cost_metrics = country_strategy_metrics(zero_cost_returns)
    zero_cost_metrics.to_csv(tables_dir / "phase8_zero_cost_sensitivity.csv", index=False)
    _figure_outputs(strategy_returns, country_metrics, crises, false_alerts, figures_dir)
    return {
        "panel": panel,
        "strategy_returns": strategy_returns,
        "country_metrics": country_metrics,
        "zero_cost_metrics": zero_cost_metrics,
        "strategy_summary": strategy_means,
        "paired_differences": paired,
        "pooled_summary": pooled,
        "bootstrap": bootstrap,
        "leave_one_out": loco,
        "time_periods": periods,
        "crisis_performance": crises,
        "missed_crises": missed,
        "correctly_alerted_crises": alerted,
        "false_alert_cost": false_alerts,
        "average_weights": weight_summary,
    }


if __name__ == "__main__":
    results = run_phase8()
    print(f"Generated Phase 8 country-level tests across {results['strategy_returns'].country_code.nunique()} countries")