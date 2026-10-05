"""Phase 7 point-in-time early-warning validation for JST systemic crises."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler

from .config import PROJECT_ROOT

MARKET_PANEL = PROJECT_ROOT / "data" / "processed" / "market_validation_panel.csv"
PHASE7_PANEL = PROJECT_ROOT / "data" / "processed" / "phase7_analytical_panel.csv"
SIGNALS = {
    "S1": ["credit_gap"],
    "S2": ["credit_gap", "dsr_deviation_10y"],
    "S3": ["credit_gap", "dsr_deviation_10y", "real_house_price_growth_5y"],
}
SECONDARY_SIGNALS = {
    "DSR_only": ["dsr_deviation_10y"],
    "Property_only": ["real_house_price_growth_5y"],
}
ALL_SIGNALS = {**SIGNALS, **SECONDARY_SIGNALS}
OUTCOME = "systemic_crisis_next_3y"
CRISIS_START = "crisisJST"
MINIMUM_TRAINING_YEARS = 5
MINIMUM_TRAINING_EVENTS = 3
MINIMUM_PRIOR_HISTORY = 5
TRAINING_LABEL_EMBARGO_YEARS = 3
LOGISTIC_C = 1.0
ALERT_BANDS = (0.10, 0.20, 0.30)


def crisis_start_audit(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize JST onset years and three-year forward label prevalence."""
    starts = panel.loc[panel[CRISIS_START].eq(1), ["country_code", "year"]].copy()
    starts = starts.sort_values(["country_code", "year"])
    rows = [{
        "record_type": "crisis_start",
        "country_code": row.country_code,
        "year": int(row.year),
        "jst_variable": CRISIS_START,
        "definition": "JST marks the first year of each systemic banking crisis; used directly, without episode reconstruction.",
    } for row in starts.itertuples(index=False)]
    country_counts = starts.groupby("country_code").size()
    for country in sorted(set(panel.country_code)):
        country_starts = starts.loc[starts.country_code.eq(country), "year"].astype(int).tolist()
        rows.append({
            "record_type": "country_summary",
            "country_code": country,
            "year": np.nan,
            "jst_variable": CRISIS_START,
            "crisis_start_count": len(country_starts),
            "crisis_start_years": ",".join(map(str, country_starts)),
            "has_at_least_one_jst_crisis_start": country in set(starts.country_code),
        })
    observed_labels = panel[OUTCOME].dropna()
    counts = pd.DataFrame([
        {"record_type": "overall_summary", "country_code": "ALL", "year": np.nan,
         "jst_variable": CRISIS_START, "crisis_start_count": len(starts),
         "countries_with_crises": int(starts.country_code.nunique()),
         "positive_next_3y_country_years": int(observed_labels.eq(1).sum()),
         "negative_next_3y_country_years": int(observed_labels.eq(0).sum()),
         "unique_crisis_events": len(starts),
         "definition": "systemic_crisis_next_3y is any JST crisisJST onset in exact years t+1 through t+3."}
    ])
    audit = pd.DataFrame(rows)
    audit.attrs["country_counts"] = country_counts.to_dict()
    return audit, counts


def add_past_crisis_outcome(panel: pd.DataFrame) -> pd.DataFrame:
    """Add a negative-control outcome from exact crisis-onset years t-3:t-1."""
    result = panel.sort_values(["country_code", "year"]).copy()
    result["past_crisis_previous_3y"] = np.nan
    for _, group in result.groupby("country_code", sort=False):
        onset = group.set_index("year")[CRISIS_START]
        for index, row in group.iterrows():
            years = [int(row.year) - offset for offset in (1, 2, 3)]
            values = [onset.get(year, np.nan) for year in years]
            if all(pd.notna(value) for value in values):
                result.loc[index, "past_crisis_previous_3y"] = int(any(value == 1 for value in values))
    return result


def _unique_events_for_training(panel: pd.DataFrame, training: pd.DataFrame) -> int:
    event_years = {
        int(origin_year + offset)
        for origin_year in training.year.astype(int)
        for offset in (1, 2, 3)
    }
    starts = panel.loc[panel[CRISIS_START].eq(1), ["country_code", "year"]]
    eligible_keys = set(zip(training.country_code, training.year.astype(int)))
    represented: set[tuple[str, int]] = set()
    for event in starts.itertuples(index=False):
        event_year = int(event.year)
        for offset in (1, 2, 3):
            origin = (event.country_code, event_year - offset)
            if origin in eligible_keys:
                represented.add((event.country_code, event_year))
                break
    return len(represented)


def _fit_logistic(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    predictors: list[str],
    *,
    country_controls: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(training[predictors].astype(float))
    x_eval = scaler.transform(evaluation[predictors].astype(float))
    if country_controls:
        train_countries = pd.get_dummies(training.country_code, prefix="country", dtype=float)
        eval_countries = pd.get_dummies(evaluation.country_code, prefix="country", dtype=float)
        eval_countries = eval_countries.reindex(columns=train_countries.columns, fill_value=0.0)
        x_train = np.column_stack([x_train, train_countries.to_numpy()])
        x_eval = np.column_stack([x_eval, eval_countries.to_numpy()])
    if training[OUTCOME].nunique() < 2:
        raise ValueError("Training fold must contain both crisis outcomes")
    train_design = sm.add_constant(pd.DataFrame(x_train), has_constant="add")
    evaluation_design = sm.add_constant(pd.DataFrame(x_eval), has_constant="add")
    fit_method = "unpenalized_logit"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(training[OUTCOME].astype(int), train_design).fit(disp=False, maxiter=300)
        converged = bool(model.mle_retvals.get("converged", True))
        if not converged or not np.isfinite(model.params.to_numpy()).all():
            raise ArithmeticError("Unpenalized logistic fit did not converge to finite coefficients")
        probabilities = model.predict(evaluation_design).to_numpy()
        fit_parameters = model.params.to_numpy()
    except Exception:
        model = LogisticRegression(C=LOGISTIC_C, solver="lbfgs", max_iter=2000, random_state=0)
        model.fit(x_train, training[OUTCOME].astype(int))
        probabilities = model.predict_proba(x_eval)[:, 1]
        fit_parameters = np.concatenate([model.intercept_, model.coef_[0]])
        fit_method = f"L2_fallback_C_{LOGISTIC_C:g}"
    return probabilities, {
        "scaler_mean": scaler.mean_.copy(),
        "scaler_scale": scaler.scale_.copy(),
        "model": model,
        "fit_parameters": fit_parameters,
        "fit_method": fit_method,
    }


def expanding_predictions(
    panel: pd.DataFrame,
    family: str,
    predictors: list[str],
    *,
    evaluation_years: list[int] | None = None,
    cooldown: bool = False,
    country_controls: bool = False,
) -> pd.DataFrame:
    """Generate strictly chronological OOS predictions with a 3-year label embargo."""
    prohibited = [name for name in predictors if "forward" in name or name.startswith("systemic_crisis_next_") or name.startswith("past_crisis_")]
    if prohibited:
        raise ValueError(f"Future or placebo outcomes cannot be predictors: {prohibited}")
    data = panel.copy()
    evaluation_data = data
    if cooldown:
        data = add_cooldown_flag(data)
        evaluation_data = data.loc[~data["crisis_cooldown"]].copy()
    years = sorted(data.loc[data[OUTCOME].notna(), "year"].astype(int).unique())
    crisis_lookup: dict[tuple[str, int], list[int]] = {}
    for row in panel.loc[panel[CRISIS_START].eq(1), ["country_code", "year"]].itertuples(index=False):
        crisis_lookup.setdefault((row.country_code, int(row.year)), []).append(int(row.year))
    if evaluation_years is not None:
        years = [year for year in years if year in evaluation_years]
    predictions = []
    for evaluation_year in years:
        train_end = evaluation_year - TRAINING_LABEL_EMBARGO_YEARS - 1
        evaluation = evaluation_data.loc[
            evaluation_data.year.eq(evaluation_year)
            & evaluation_data[[*predictors, OUTCOME]].notna().all(axis=1)
        ].copy()
        training = data.loc[
            data.year.le(train_end)
            & data[[*predictors, OUTCOME]].notna().all(axis=1)
        ].copy()
        if evaluation.empty or training.empty:
            continue
        if training.year.nunique() < MINIMUM_TRAINING_YEARS:
            continue
        unique_events = _unique_events_for_training(panel, training)
        if unique_events < MINIMUM_TRAINING_EVENTS or training[OUTCOME].nunique() < 2:
            continue
        probabilities, fit_metadata = _fit_logistic(
            training, evaluation, predictors, country_controls=country_controls
        )
        for (_, row), probability in zip(evaluation.iterrows(), probabilities):
            predictions.append({
                "family": family,
                "country_code": row.country_code,
                "year": int(row.year),
                "actual": int(row[OUTCOME]),
                "training_base_rate": float(training[OUTCOME].mean()),
                "linked_crisis_events": ";".join(
                    str(event_year)
                    for offset in (1, 2, 3)
                    for event_year in crisis_lookup.get((row.country_code, int(row.year) + offset), [])
                ),
                "probability": float(probability),
                "fit_method": fit_metadata["fit_method"],
                "train_first_year": int(training.year.min()),
                "train_last_year": int(training.year.max()),
                "train_N": len(training),
                "train_countries": training.country_code.nunique(),
                "train_unique_crisis_events": unique_events,
                "evaluation_N": len(evaluation),
                "standardization_mean": ";".join(map(str, fit_metadata["scaler_mean"])),
                "standardization_scale": ";".join(map(str, fit_metadata["scaler_scale"])),
                "country_controls": country_controls,
                "cooldown_excluded": cooldown,
                "label_embargo_years": TRAINING_LABEL_EMBARGO_YEARS,
            })
    return pd.DataFrame(predictions)


def point_in_time_equal_weight_index(panel: pd.DataFrame, *, cooldown: bool = False) -> pd.DataFrame:
    """Compute equal-weight training-standardized signals and OOS index scores."""
    predictors = SIGNALS["S3"]
    data = panel.copy()
    if cooldown:
        data = add_cooldown_flag(data)
        data = data.loc[~data.crisis_cooldown].copy()
    output = []
    for evaluation_year in sorted(data.loc[data[OUTCOME].notna(), "year"].astype(int).unique()):
        train_end = evaluation_year - TRAINING_LABEL_EMBARGO_YEARS - 1
        training = data.loc[
            data.year.le(train_end)
            & data[predictors].notna().all(axis=1)
            & data[OUTCOME].notna()
        ]
        evaluation = data.loc[data.year.eq(evaluation_year) & data[predictors].notna().all(axis=1) & data[OUTCOME].notna()]
        if training.year.nunique() < MINIMUM_TRAINING_YEARS or _unique_events_for_training(panel, training) < MINIMUM_TRAINING_EVENTS or evaluation.empty:
            continue
        means = training[predictors].mean()
        scales = training[predictors].std(ddof=0).replace(0, np.nan)
        if scales.isna().any():
            continue
        for _, row in evaluation.iterrows():
            z = (row[predictors] - means) / scales
            output.append({
                "family": "TRANSPARENT SENSITIVITY INDEX",
                "country_code": row.country_code,
                "year": int(row.year),
                "actual": int(row[OUTCOME]),
                "training_base_rate": float(training[OUTCOME].mean()),
                "probability": float(expit(z.mean())),
                "fit_method": "transparent_equal_weight_index",
                "linked_crisis_events": ";".join(
                    str(event_year)
                    for offset in (1, 2, 3)
                    for event_year in panel.loc[
                        panel.country_code.eq(row.country_code)
                        & panel.year.eq(int(row.year) + offset)
                        & panel[CRISIS_START].eq(1), "year"
                    ].astype(int).tolist()
                ),
                "train_first_year": int(training.year.min()),
                "train_last_year": int(training.year.max()),
                "train_N": len(training),
                "train_unique_crisis_events": _unique_events_for_training(panel, training),
                "evaluation_N": len(evaluation),
                "standardization_mean": ";".join(map(str, means.to_numpy())),
                "standardization_scale": ";".join(map(str, scales.to_numpy())),
                "cooldown_excluded": cooldown,
                "label_embargo_years": TRAINING_LABEL_EMBARGO_YEARS,
            })
    return pd.DataFrame(output)


def _assign_alert_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    if result.empty:
        for band in ALERT_BANDS:
            result[f"top_{int(band * 100)}pct"] = pd.Series(dtype=bool)
        result["cross_section_rank_pct"] = pd.Series(dtype=float)
        return result
    result["cross_section_rank_pct"] = np.nan
    for band in ALERT_BANDS:
        result[f"top_{int(band * 100)}pct"] = False
    for _, indices in result.groupby(["family", "year"]).groups.items():
        group = result.loc[indices]
        ranks = group.probability.rank(method="first", ascending=False, pct=True)
        result.loc[indices, "cross_section_rank_pct"] = ranks.to_numpy()
        for band in ALERT_BANDS:
            count = max(1, int(np.ceil(len(group) * band)))
            ordered = group.assign(_rank=ranks).sort_values(["_rank", "country_code"])
            result.loc[ordered.head(count).index, f"top_{int(band * 100)}pct"] = True
    return result


def _metrics(predictions: pd.DataFrame, label: str) -> dict[str, object]:
    data = predictions.dropna(subset=["actual", "probability"])
    actual = data.actual.astype(int)
    scores = data.probability.astype(float)
    prevalence = float(actual.mean()) if len(data) else np.nan
    roc = roc_auc_score(actual, scores) if actual.nunique() == 2 else np.nan
    pr = average_precision_score(actual, scores) if actual.nunique() == 2 else np.nan
    brier = brier_score_loss(actual, scores.clip(0, 1)) if len(data) else np.nan
    # Use each prediction's historical training prevalence as the no-skill Brier benchmark.
    naive = np.mean((actual.to_numpy() - data["training_base_rate"].to_numpy()) ** 2) if len(data) and "training_base_rate" in data else np.nan
    event_keys = set()
    if "linked_crisis_events" in data:
        for row in data.itertuples(index=False):
            if row.linked_crisis_events:
                event_keys.update((row.country_code, int(year)) for year in row.linked_crisis_events.split(";"))
    unique_events = len(event_keys) if "linked_crisis_events" in data else np.nan
    return {
        "sample": label,
        "N": len(data),
        "countries": data.country_code.nunique(),
        "first_year": data.year.min() if len(data) else np.nan,
        "last_year": data.year.max() if len(data) else np.nan,
        "positive_country_years": int(actual.sum()) if len(data) else 0,
        "negative_country_years": int((actual == 0).sum()) if len(data) else 0,
        "event_prevalence": prevalence,
        "roc_auc": roc,
        "pr_auc_average_precision": pr,
        "pr_prevalence_baseline": prevalence,
        "discrimination_estimable": bool(actual.nunique() == 2),
        "metric_note": "" if actual.nunique() == 2 else "Only one outcome class is present in these OOS predictions; ROC-AUC, PR-AUC, and recall are undefined.",
        "brier_score": brier,
        "historical_base_rate_brier": naive,
        "unique_crisis_events_represented": unique_events,
    }


def _with_training_prevalence(predictions: pd.DataFrame, panel: pd.DataFrame, predictors_by_family: dict[str, list[str]]) -> pd.DataFrame:
    result = predictions.copy()
    if "training_base_rate" in result:
        return result
    result["training_base_rate"] = np.nan
    for index, row in result.iterrows():
        predictors = predictors_by_family.get(row.family, SIGNALS["S3"])
        train_end = int(row.train_last_year)
        train = panel.loc[panel.year.le(train_end) & panel[predictors + [OUTCOME]].notna().all(axis=1)]
        result.loc[index, "training_base_rate"] = train[OUTCOME].mean() if len(train) else np.nan
    return result


def _metric_table(predictions: pd.DataFrame, sample_name: str, *, by_family: bool = True) -> pd.DataFrame:
    predictions = _assign_alert_bands(predictions)
    if by_family:
        families = sorted(predictions.family.unique())
    else:
        families = [sample_name]
    rows = []
    for family in families:
        data = predictions.loc[predictions.family.eq(family)].copy() if by_family else predictions.copy()
        metric = _metrics(data, sample_name)
        metric["family"] = family
        for band in ALERT_BANDS:
            column = f"top_{int(band * 100)}pct"
            flagged = data[column].astype(bool)
            tp = int((flagged & data.actual.eq(1)).sum())
            fp = int((flagged & data.actual.eq(0)).sum())
            positives = int(data.actual.eq(1).sum())
            negatives = int(data.actual.eq(0).sum())
            metric[f"top_{int(band*100)}_flagged"] = int(flagged.sum())
            metric[f"top_{int(band*100)}_events_captured"] = tp
            metric[f"top_{int(band*100)}_precision"] = tp / (tp + fp) if tp + fp else np.nan
            metric[f"top_{int(band*100)}_recall"] = tp / positives if positives else np.nan
            metric[f"top_{int(band*100)}_false_positive_rate"] = fp / negatives if negatives else np.nan
        rows.append(metric)
    return pd.DataFrame(rows), predictions


def _common_sample_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    common = panel.dropna(subset=[*SIGNALS["S3"], OUTCOME])
    frames = [expanding_predictions(common, family, predictors) for family, predictors in SIGNALS.items()]
    if not frames:
        return pd.DataFrame()
    # Only compare a year-country tuple if every signal family has an OOS prediction.
    keys = [set(zip(frame.country_code, frame.year)) for frame in frames]
    shared = set.intersection(*keys) if keys else set()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[[ (country, year) in shared for country, year in zip(combined.country_code, combined.year) ]]
    if not combined.empty:
        identities = [set(zip(combined.loc[combined.family.eq(family), "country_code"], combined.loc[combined.family.eq(family), "year"])) for family in SIGNALS]
        if not all(identities[0] == identity for identity in identities[1:]):
            raise AssertionError("Common OOS sample differs across S1-S3")
    return combined


def event_lead_times(predictions: pd.DataFrame, crisis_starts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _assign_alert_bands(predictions)
    leads = []
    event_rows = []
    for family, family_data in data.groupby("family"):
        event_capture = []
        for event in crisis_starts.itertuples(index=False):
            lead_records = []
            for lead in (1, 2, 3):
                origin = int(event.year) - lead
                matched = family_data.loc[
                    family_data.country_code.eq(event.country_code)
                    & family_data.year.eq(origin)
                ]
                row = matched.iloc[0] if not matched.empty else None
                record = {
                    "family": family,
                    "country_code": event.country_code,
                    "crisis_start_year": int(event.year),
                    "lead_years": lead,
                    "origin_year": origin,
                    "prediction_available": row is not None,
                    "predicted_probability_or_score": row.probability if row is not None else np.nan,
                    "cross_section_rank_pct": row.cross_section_rank_pct if row is not None else np.nan,
                }
                for band in ALERT_BANDS:
                    record[f"top_{int(band*100)}pct"] = bool(row[f"top_{int(band*100)}pct"]) if row is not None else False
                leads.append(record)
                if row is not None:
                    lead_records.append(row)
            event_record = {
                "family": family,
                "country_code": event.country_code,
                "crisis_start_year": int(event.year),
                "unique_crisis_event": True,
                "predicted_at_any_prior_lead": bool(lead_records),
            }
            for band in ALERT_BANDS:
                flagged_leads = [
                    int(event.year) - int(row.year)
                    for row in lead_records
                    if bool(row[f"top_{int(band*100)}pct"])
                ]
                event_record[f"captured_top_{int(band*100)}pct"] = bool(flagged_leads)
                event_record[f"earliest_warning_lead_top_{int(band*100)}pct"] = max(flagged_leads) if flagged_leads else np.nan
                event_record[f"latest_warning_lead_top_{int(band*100)}pct"] = min(flagged_leads) if flagged_leads else np.nan
            event_capture.append(event_record)
        event_rows.extend(event_capture)
    lead_table = pd.DataFrame(leads)
    event_table = pd.DataFrame(event_rows)
    if not event_table.empty:
        summary_rows = []
        for family, group in event_table.groupby("family"):
            for band in ALERT_BANDS:
                column = f"captured_top_{int(band*100)}pct"
                summary_rows.append({
                    "record_type": "event_level_summary",
                    "family": family,
                    "alert_band": int(band * 100),
                    "unique_crisis_events": len(group),
                    "events_with_any_prediction": int(group.predicted_at_any_prior_lead.sum()),
                    "events_without_any_prediction": int((~group.predicted_at_any_prior_lead).sum()),
                    "events_captured": int(group[column].sum()),
                    "event_level_recall_all_documented": group[column].mean(),
                    "event_level_recall_when_prediction_available": (
                        group.loc[group.predicted_at_any_prior_lead, column].mean()
                        if group.predicted_at_any_prior_lead.any()
                        else np.nan
                    ),
                    "median_earliest_warning_lead_years": group.loc[group[column], f"earliest_warning_lead_top_{int(band*100)}pct"].median(),
                })
        event_table["record_type"] = "crisis_event"
        event_table = pd.concat([event_table, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)
    return lead_table, event_table


def add_cooldown_flag(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    start_keys = set(zip(
        result.loc[result[CRISIS_START].eq(1), "country_code"],
        result.loc[result[CRISIS_START].eq(1), "year"].astype(int),
    ))
    result["crisis_cooldown"] = [
        any((country, int(year) - offset) in start_keys for offset in (0, 1, 2))
        for country, year in zip(result.country_code, result.year)
    ]
    return result


def historical_credit_gap_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.sort_values(["country_code", "year"]).copy()
    result["credit_gap_prior_percentile"] = np.nan
    for _, group in result.groupby("country_code", sort=False):
        history: list[float] = []
        for index, row in group.iterrows():
            value = row.credit_gap
            if pd.notna(value) and len(history) >= MINIMUM_PRIOR_HISTORY:
                result.loc[index, "credit_gap_prior_percentile"] = np.mean(np.asarray(history) <= value)
            if pd.notna(value):
                history.append(float(value))
    ranked = result.dropna(subset=["credit_gap_prior_percentile", OUTCOME]).copy()
    ranked["historical_rank_band"] = pd.cut(
        ranked.credit_gap_prior_percentile,
        bins=[0, 0.5, 0.8, 0.9, 1.000001],
        labels=["0-50", "50-80", "80-90", "90-100"],
        include_lowest=True,
        right=True,
    )
    rows = []
    for band, group in ranked.groupby("historical_rank_band", observed=True):
        rows.append({
            "historical_rank_band": str(band),
            "N": len(group),
            "positive_country_years": int(group[OUTCOME].sum()),
            "crisis_incidence": group[OUTCOME].mean(),
            "countries": group.country_code.nunique(),
            "unique_crisis_events": _unique_events_for_training(result, group),
        })
    return result, pd.DataFrame(rows)


def calibration_table(predictions: pd.DataFrame, bins: int = 5, minimum_bin_n: int = 20) -> pd.DataFrame:
    rows = []
    for family, group in predictions.groupby("family"):
        if len(group) < minimum_bin_n:
            continue
        group = group.copy()
        group["calibration_bin"] = pd.qcut(group.probability, q=bins, duplicates="drop")
        for interval, subset in group.groupby("calibration_bin", observed=True):
            rows.append({
                "family": family,
                "bin": str(interval),
                "mean_predicted_probability": subset.probability.mean(),
                "observed_event_rate": subset.actual.mean(),
                "N": len(subset),
                "positive_country_years": int(subset.actual.sum()),
                "interpretation_note": "Descriptive OOS calibration; small bins should not be overinterpreted.",
            })
    return pd.DataFrame(rows)


def _placebo_models(panel: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression as LR
    rows = []
    data = add_past_crisis_outcome(panel)
    for label, predictors in SIGNALS.items():
        sample = data.dropna(subset=["past_crisis_previous_3y", OUTCOME, *predictors])
        if sample.empty:
            continue
        scaler = StandardScaler()
        x = scaler.fit_transform(sample[predictors].astype(float))
        for outcome, outcome_label, analysis_type in [
            ("past_crisis_previous_3y", "past_crisis_previous_3y", "FULL-SAMPLE NEGATIVE-CONTROL DIAGNOSTIC; NOT PREDICTIVE"),
            (OUTCOME, "systemic_crisis_next_3y", "FULL-SAMPLE ASSOCIATION BENCHMARK; NOT OOS PREDICTIVE EVIDENCE"),
        ]:
            if sample[outcome].nunique() < 2:
                continue
            model = LR(C=LOGISTIC_C, solver="lbfgs", max_iter=2000, random_state=0)
            model.fit(x, sample[outcome].astype(int))
            for predictor, coefficient in zip(predictors, model.coef_[0]):
                rows.append({
                    "family": label,
                    "outcome": outcome_label,
                    "term": predictor,
                    "standardized_log_odds_coefficient": coefficient,
                    "odds_ratio_per_full_sample_sd": np.exp(coefficient),
                    "N": len(sample),
                    "countries": sample.country_code.nunique(),
                    "positive_country_years": int(sample[outcome].sum()),
                    "same_sample_past_future": True,
                    "analysis_type": analysis_type,
                })
    return pd.DataFrame(rows)


def _country_control_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    predictions = expanding_predictions(
        panel, "S1_country_controls", SIGNALS["S1"], country_controls=True
    )
    if not predictions.empty:
        predictions["priority"] = "COUNTRY-CONTROL SENSITIVITY"
    return predictions


def initial_split_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic 2007 holdout; three-year label maturity leaves origins through 2003."""
    family = "S1_initial_split"
    predictors = SIGNALS["S1"]
    training = panel.loc[
        panel.year.le(2003)
        & panel[[*predictors, OUTCOME]].notna().all(axis=1)
    ].copy()
    evaluation = panel.loc[
        panel.year.ge(2007)
        & panel[[*predictors, OUTCOME]].notna().all(axis=1)
    ].copy()
    if training.year.nunique() < MINIMUM_TRAINING_YEARS or training[OUTCOME].nunique() < 2 or evaluation.empty:
        return pd.DataFrame()
    unique_events = _unique_events_for_training(panel, training)
    probabilities, fit_metadata = _fit_logistic(training, evaluation, predictors)
    return pd.DataFrame([
        {
            "family": family,
            "country_code": row.country_code,
            "year": int(row.year),
            "actual": int(row[OUTCOME]),
            "training_base_rate": float(training[OUTCOME].mean()),
            "probability": float(probability),
            "linked_crisis_events": ";".join(
                str(event_year)
                for offset in (1, 2, 3)
                for event_year in panel.loc[
                    panel.country_code.eq(row.country_code)
                    & panel.year.eq(int(row.year) + offset)
                    & panel[CRISIS_START].eq(1), "year"
                ].astype(int).tolist()
            ),
            "train_first_year": int(training.year.min()),
            "train_last_year": int(training.year.max()),
            "train_N": len(training),
            "train_countries": training.country_code.nunique(),
            "train_unique_crisis_events": unique_events,
            "evaluation_N": len(evaluation),
            "standardization_mean": ";".join(map(str, fit_metadata["scaler_mean"])),
            "standardization_scale": ";".join(map(str, fit_metadata["scaler_scale"])),
            "fit_method": fit_metadata["fit_method"],
            "label_embargo_years": TRAINING_LABEL_EMBARGO_YEARS,
            "note": "S1-only initial-split diagnostic; S2/S3 lack usable pre-test DSR coverage.",
        }
        for (_, row), probability in zip(evaluation.iterrows(), probabilities)
    ])


def _summary_rows(
    natural_predictions: pd.DataFrame,
    common_predictions: pd.DataFrame,
    index_predictions: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for label, predictions in [
        ("natural_coverage", natural_predictions),
        ("common_complete_sample", common_predictions),
        ("transparent_sensitivity_index", index_predictions),
    ]:
        if predictions.empty:
            continue
        for family, group in predictions.groupby("family"):
            group = _with_training_prevalence(group, panel, ALL_SIGNALS)
            metrics, _ = _metric_table(group, label)
            records.extend(metrics.to_dict("records"))
    return pd.DataFrame(records)


def _plot_figures(
    common_predictions: pd.DataFrame,
    natural_predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    timeline_predictions: pd.DataFrame,
    crisis_starts: pd.DataFrame,
    figures_dir: Path,
) -> None:
    families = [family for family in SIGNALS if family in set(common_predictions.family)]
    figure, axis = plt.subplots(figsize=(7.6, 5.8))
    for family in families:
        group = common_predictions.loc[common_predictions.family.eq(family)]
        if group.actual.nunique() == 2:
            fpr, tpr, _ = roc_curve(group.actual, group.probability)
            axis.plot(fpr, tpr, label=f"{family} (AUC {roc_auc_score(group.actual, group.probability):.2f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Expanding-window OOS ROC (common sample)")
    if any(common_predictions.loc[common_predictions.family.eq(family), "actual"].nunique() == 2 for family in families):
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "Insufficient common-sample OOS coverage", ha="center", va="center", transform=axis.transAxes)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures_dir / "phase7_oos_roc.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.6, 5.8))
    for family in families:
        group = common_predictions.loc[common_predictions.family.eq(family)]
        if group.actual.nunique() == 2:
            precision, recall, _ = precision_recall_curve(group.actual, group.probability)
            prevalence = group.actual.mean()
            axis.plot(recall, precision, label=f"{family} (AP {average_precision_score(group.actual, group.probability):.2f})")
            if family == families[0]:
                axis.axhline(prevalence, color="#666666", linestyle="--", label=f"Prevalence {prevalence:.2f}")
    axis.set(xlabel="Recall", ylabel="Precision", title="Expanding-window OOS precision-recall (common sample)")
    if any(common_predictions.loc[common_predictions.family.eq(family), "actual"].nunique() == 2 for family in families):
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "Insufficient common-sample OOS coverage", ha="center", va="center", transform=axis.transAxes)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures_dir / "phase7_oos_precision_recall.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 5.7))
    for family, group in calibration.groupby("family"):
        axis.plot(group.mean_predicted_probability, group.observed_event_rate, marker="o", label=family)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
    axis.set(xlabel="Mean predicted probability", ylabel="Observed event rate", title="Expanding-window OOS calibration")
    if calibration.family.nunique() > 0 and axis.lines:
        axis.legend(frameon=False)
    elif calibration.empty:
        axis.text(0.5, 0.5, "No predictions available for calibration", ha="center", va="center", transform=axis.transAxes)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures_dir / "phase7_calibration.png", dpi=160)
    plt.close(figure)

    # Pick the three most complete countries by the count of available S1 signals,
    # without using model performance or crisis outcomes in selection.
    completeness = timeline_predictions.groupby("country_code").size().sort_values(ascending=False)
    selected = completeness.head(3).index.tolist()
    figure, axes = plt.subplots(len(selected), 1, figsize=(10, max(3.0, 2.6 * len(selected))), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for axis, country in zip(axes, selected):
        group = timeline_predictions.loc[timeline_predictions.country_code.eq(country)].sort_values("year")
        axis.plot(group.year, group.probability, color="#287c78", linewidth=1.5)
        starts = crisis_starts.loc[crisis_starts.country_code.eq(country), "year"]
        for onset in starts:
            axis.axvline(onset, color="#ba533d", linewidth=1.0, alpha=0.8)
        axis.set_ylabel(f"{country}\nOOS risk")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Year (red vertical lines: JST crisis starts)")
    figure.suptitle("Point-in-time S1 predictions; countries selected by signal completeness")
    figure.tight_layout()
    figure.savefig(figures_dir / "phase7_oos_timeline_examples.png", dpi=160)
    plt.close(figure)


def run_phase7(
    panel_path: Path = MARKET_PANEL,
    analytical_path: Path = PHASE7_PANEL,
    tables_dir: Path = PROJECT_ROOT / "outputs" / "tables",
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures",
) -> dict[str, pd.DataFrame]:
    panel = pd.read_csv(panel_path)
    if panel.duplicated(["country_code", "year"]).any():
        raise ValueError("Phase 6 market panel must have unique country-years")
    panel["year"] = panel.year.astype(int)
    audit, audit_summary = crisis_start_audit(panel)
    panel = add_past_crisis_outcome(panel)
    analytical_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(analytical_path, index=False)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([audit, audit_summary], ignore_index=True, sort=False).to_csv(tables_dir / "phase7_crisis_audit.csv", index=False)

    crisis_starts = panel.loc[panel[CRISIS_START].eq(1), ["country_code", "year"]].copy()
    split_counts = pd.DataFrame([
        {"period": "TRAIN origins through 2006; raw preferred split", "unique_crisis_starts": int(crisis_starts.year.le(2006).sum()), "crisis_starts": int(crisis_starts.year.le(2006).sum()), "positive_outcome_country_years": int(panel.loc[panel.year.le(2006), OUTCOME].eq(1).sum()), "negative_outcome_country_years": int(panel.loc[panel.year.le(2006), OUTCOME].eq(0).sum()), "countries": int(panel.loc[panel.year.le(2006), "country_code"].nunique())},
        {"period": "TEST origins from 2007", "unique_crisis_starts": int(crisis_starts.year.ge(2007).sum()), "crisis_starts": int(crisis_starts.year.ge(2007).sum()), "positive_outcome_country_years": int(panel.loc[panel.year.ge(2007), OUTCOME].eq(1).sum()), "negative_outcome_country_years": int(panel.loc[panel.year.ge(2007), OUTCOME].eq(0).sum()), "countries": int(panel.loc[panel.year.ge(2007), "country_code"].nunique())},
        {"period": "TRAIN labels mature by test start (origin through 2003)", "unique_crisis_starts": int(crisis_starts.year.le(2006).sum()), "crisis_starts": int(crisis_starts.year.le(2006).sum()), "positive_outcome_country_years": int(panel.loc[panel.year.le(2003), OUTCOME].eq(1).sum()), "negative_outcome_country_years": int(panel.loc[panel.year.le(2003), OUTCOME].eq(0).sum()), "countries": int(panel.loc[panel.year.le(2003), "country_code"].nunique())},
    ])
    split_counts["note"] = [
        "The naive cutoff would use overlapping three-year training labels that extend into the test period.",
        "JST-documented starts; test starts are not the same as positive prediction-origin country-years.",
        "Three-year outcome-label maturity cutoff used for leakage-safe initial-split sensitivity.",
    ]
    split_counts.to_csv(tables_dir / "phase7_chronological_split_audit.csv", index=False)

    natural_frames = [expanding_predictions(panel, family, predictors) for family, predictors in SIGNALS.items()]
    for family, predictors in SECONDARY_SIGNALS.items():
        natural_frames.append(expanding_predictions(panel, family, predictors))
    natural = pd.concat([frame for frame in natural_frames if not frame.empty], ignore_index=True)
    natural = _with_training_prevalence(natural, panel, ALL_SIGNALS)
    common = _common_sample_predictions(panel)
    common = _with_training_prevalence(common, panel, SIGNALS)
    index_predictions = point_in_time_equal_weight_index(panel)
    index_predictions = _with_training_prevalence(index_predictions, panel, ALL_SIGNALS)

    natural_metrics, natural = _metric_table(natural, "natural_coverage")
    common_metrics, common = _metric_table(common, "common_complete_sample")
    index_metrics, index_predictions = _metric_table(index_predictions, "transparent_sensitivity_index")
    combined_metrics = pd.concat([natural_metrics, common_metrics, index_metrics], ignore_index=True)
    combined_metrics.to_csv(tables_dir / "phase7_oos_performance_metrics.csv", index=False)
    common_metrics.to_csv(tables_dir / "phase7_common_sample_comparison.csv", index=False)

    country_sensitivity = _country_control_predictions(panel)
    country_sensitivity = _with_training_prevalence(country_sensitivity, panel, {"S1_country_controls": SIGNALS["S1"]})
    if not country_sensitivity.empty:
        country_metrics, country_sensitivity = _metric_table(country_sensitivity, "country_controls_sensitivity")
        country_metrics.to_csv(tables_dir / "phase7_country_control_sensitivity.csv", index=False)
    else:
        country_metrics = pd.DataFrame()

    initial_split = initial_split_predictions(panel)
    if not initial_split.empty:
        initial_split = _with_training_prevalence(initial_split, panel, {"S1_initial_split": SIGNALS["S1"]})
        initial_metrics, initial_split = _metric_table(initial_split, "initial_split_S1_only")
    else:
        initial_metrics = pd.DataFrame()
    initial_metrics.to_csv(tables_dir / "phase7_initial_split_diagnostic.csv", index=False)
    initial_split.to_csv(tables_dir / "phase7_initial_split_predictions.csv", index=False)

    all_predictions = pd.concat([natural, index_predictions], ignore_index=True, sort=False)
    wide = natural.loc[natural.family.isin(SIGNALS)].pivot(
        index=["country_code", "year", "actual"], columns="family", values="probability"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={family: f"{family.lower()}_probability" for family in SIGNALS})
    wide.to_csv(tables_dir / "phase7_oos_predictions.csv", index=False)
    all_predictions.to_csv(tables_dir / "phase7_oos_predictions_long.csv", index=False)
    natural.to_csv(tables_dir / "phase7_oos_predictions_natural.csv", index=False)
    common.to_csv(tables_dir / "phase7_oos_predictions_common.csv", index=False)

    lead_times, event_capture = event_lead_times(natural, crisis_starts)
    lead_times.to_csv(tables_dir / "phase7_crisis_lead_times.csv", index=False)
    event_capture.to_csv(tables_dir / "phase7_event_level_recall.csv", index=False)

    calibration = calibration_table(natural)
    calibration.to_csv(tables_dir / "phase7_calibration.csv", index=False)
    ranked_panel, historical_bins = historical_credit_gap_ranks(panel)
    historical_bins.to_csv(tables_dir / "phase7_historical_credit_gap_incidence.csv", index=False)
    ranked_panel.to_csv(tables_dir / "phase7_historical_credit_gap_ranks.csv", index=False)

    placebo = _placebo_models(panel)
    placebo.to_csv(tables_dir / "phase7_past_crisis_placebo.csv", index=False)
    cooldown_frames = [
        expanding_predictions(panel, family, predictors, cooldown=True)
        for family, predictors in [("S1", SIGNALS["S1"]), ("S2", SIGNALS["S2"])]
    ]
    cooldown = pd.concat([frame for frame in cooldown_frames if not frame.empty], ignore_index=True)
    if not cooldown.empty:
        cooldown = _with_training_prevalence(cooldown, panel, SIGNALS)
        cooldown_metrics, cooldown = _metric_table(cooldown, "crisis_cooldown_sensitivity")
        cooldown_metrics.to_csv(tables_dir / "phase7_crisis_cooldown_sensitivity.csv", index=False)
        cooldown.to_csv(tables_dir / "phase7_crisis_cooldown_predictions.csv", index=False)
    else:
        cooldown_metrics = pd.DataFrame()
        cooldown_metrics.to_csv(tables_dir / "phase7_crisis_cooldown_sensitivity.csv", index=False)
        cooldown.to_csv(tables_dir / "phase7_crisis_cooldown_predictions.csv", index=False)

    timeline = natural.loc[natural.family.eq("S1")].copy()
    timeline["credit_gap_nonmissing"] = 1
    _plot_figures(common, natural, calibration, timeline, crisis_starts, figures_dir)

    summary = _summary_rows(natural, common, index_predictions, panel)
    summary.to_csv(tables_dir / "phase7_prespecified_summary.csv", index=False)
    coverage = pd.DataFrame([
        {"family": family, "predictors": ", ".join(predictors), "available_N": int(panel.dropna(subset=[*predictors, OUTCOME]).shape[0]), "available_countries": int(panel.dropna(subset=[*predictors, OUTCOME]).country_code.nunique()), "available_first_year": panel.dropna(subset=[*predictors, OUTCOME]).year.min(), "available_last_year": panel.dropna(subset=[*predictors, OUTCOME]).year.max(), "oos_N": int(natural.loc[natural.family.eq(family)].shape[0]) if not natural.empty else 0, "oos_countries": int(natural.loc[natural.family.eq(family), "country_code"].nunique()) if not natural.empty else 0}
        for family, predictors in ALL_SIGNALS.items()
    ])
    coverage.to_csv(tables_dir / "phase7_signal_coverage.csv", index=False)
    summary_note = pd.DataFrame([{
        "minimum_prior_evaluation_years": MINIMUM_TRAINING_YEARS,
        "minimum_unique_crisis_events_in_training_labels": MINIMUM_TRAINING_EVENTS,
        "label_embargo_years": TRAINING_LABEL_EMBARGO_YEARS,
        "initial_split_train_origin_cutoff": 2003,
        "initial_split_test_start": 2007,
        "initial_split_reason": "Embargoed so each three-year training label is fully realized before the first test year; DSR natural coverage is not sufficient before 2007, so expanding window is primary.",
        "logistic_regularization": "Unpenalized logistic first; fixed L2 C=1.0 fallback only on convergence/separation failure; not tuned.",
        "scaling": "StandardScaler fitted on training observations separately at each evaluation year.",
        "timeline_country_selection": "Three countries with the largest number of non-missing credit-gap observations among OOS prediction countries; not selected on model performance.",
    }])
    summary_note.to_csv(tables_dir / "phase7_methodology_audit.csv", index=False)
    return {
        "panel": panel,
        "audit": audit,
        "audit_summary": audit_summary,
        "natural_predictions": natural,
        "common_predictions": common,
        "index_predictions": index_predictions,
        "metrics": combined_metrics,
        "lead_times": lead_times,
        "event_capture": event_capture,
        "calibration": calibration,
        "historical_bins": historical_bins,
        "placebo": placebo,
        "cooldown_metrics": cooldown_metrics,
        "initial_split_metrics": initial_metrics,
        "summary": summary,
    }


if __name__ == "__main__":
    output = run_phase7()
    print(f"Generated Phase 7 validation outputs; {len(output['natural_predictions']):,} natural-coverage OOS predictions")