# Structural Resilience Research

## Research question

Can demographic, fiscal and macro-financial indicators identify structural economic regimes in which conventional portfolios become vulnerable, and can a diversified resilience strategy improve real downside protection without sacrificing excessive long-run growth?

This repository contains Phase 1 data engineering and Phase 2 exploratory economic analysis. It is exploratory research. No investment conclusions have yet been drawn.

## Scope

The panel covers 29 ISO3 countries from 1960 through requested upper bound 2025, using nine World Development Indicators from the World Bank Indicators API V2. The country universe is defined in `src/config.py`; no regional aggregates are included.

Raw observations are saved in long format at `data/raw/world_bank/world_bank_raw.csv`. The processed country-year panel is `data/processed/structural_panel.csv`. Raw downloads are ignored by Git because they are generated data.

## Transformations

All derived variables are calculated within country and use calendar-year keys. Changes compare t with t-5 or t-10. Backward five-year averages include t-4 through t and require all five annual observations. The forward GDP-per-capita growth outcome averages t+1 through t+5, excludes t, and is an outcome only: it must never be used as a predictor at time t.

Missing observations are not interpolated. Missing intermediate years prevent a change or rolling average from being calculated.

## Setup and commands

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.world_bank
python -m src.build_panel
python -m src.coverage
pytest
```

The download client requests each indicator independently so an individual failed indicator can be retried. Network/API failures are not masked by tests; tests use local data and mocked response-shaped payloads.

The project does not build portfolios, systemic-risk scores, forecasts, machine-learning models, dashboards, APIs, or investment recommendations.

## Phase 2 exploratory analysis

Phase 2 examines historical relationships between demographic deterioration and subsequent real GDP-per-capita growth using `data/processed/structural_panel_derived.csv`. It produces descriptive statistics, selected-country trajectories, demographic-growth scatter figures, correlations, decade and country summaries, and fixed-effects panel regressions.

The regression outcome is the overlapping five-year forward growth measure. Country and year fixed effects are included, and standard errors are clustered by country. Overlapping outcomes induce serial correlation; country clustering is a pragmatic V1 treatment, not a perfect solution. Predictors are restricted to contemporaneous or backward-looking variables. Results are associations conditional on the sample and specification, not causal estimates.

Run Phase 2 with:

```powershell
python -m src.panel_analysis
```

Phase 2 results are written to `outputs/tables/` and `outputs/figures/`. The deterministic non-overlap sensitivity sample keeps observations where `year % 5 == 0` within each country.

The Phase 2 descriptive correlations are positive for working-age change and near zero for dependency change. In the fixed-effects specifications, the working-age coefficient is positive and fairly stable across Models 1, 2, and 3A, while the dependency coefficient changes direction between the joint and separate specifications and has wide confidence intervals. These are conditional associations within this sample, not causal findings.

## Phase 4 structural-financial data layer

Phase 4 adds official IMF WEO and BIS data for descriptive structural-financial comparisons. IMF WEO October 2025 DataMapper historical observations retain `GGXWDG_NGDP` (general government gross debt, percent of GDP) and `GGXCNL_NGDP` (general government net lending/borrowing, percent of GDP). Forecast-labelled years are excluded by retaining observations only through 2024; primary balance, revenue, and expenditure are not substituted because the selected official API did not provide validated series for them.

BIS bulk datasets are the official `WS_TC` credit-to-the-non-financial-sector file, `WS_CREDIT_GAP`, `WS_DSR`, and selected residential property prices `WS_SPP`. The retained concepts remain distinct: private credit/GDP, the published credit-to-GDP gap, private non-financial-sector DSR, and the real selected residential property-price index. Credit gap is taken from the BIS-published actual-minus-trend series and is not recomputed. DSR is retained as a level and as a within-country deviation from the prior ten-year mean, because BIS cautions against simplistic cross-country level comparisons.

Quarterly BIS stock/gap series use Q4; DSR and property-index series use the annual mean of at least two available quarters. Exact calendar-year lookups are used for five-year changes, three-year credit-gap averages, five-year property growth, and the DSR historical reference window. Phase 4 uses descriptive sample-median quadrants and point-in-time, expanding historical quintile flags. These are exploratory groupings, not investment thresholds or a composite risk score. No causal interpretation is intended.

Run Phase 4 with:

```powershell
python -m src.imf_weo
python -m src.bis
python -m src.phase4
python -m src.phase4_analysis
```

The new panel is `data/processed/structural_financial_panel.csv`; Phase 1-3 processed files are preserved unchanged.

## Phase 3 demographic robustness and productivity offset

Phase 3 uses World Bank `NY.GDP.PCAP.KD` (GDP per capita, constant 2015 US$), `SL.TLF.CACT.ZS` (total labour-force participation, modeled ILO estimate), and `SL.EMP.TOTL.SP.ZS` (total employment-to-population ratio, modeled ILO estimate). Labour-force participation is the primary utilisation control; employment-to-population is retained descriptively because its coverage is slightly weaker.

Productivity uses Penn World Table 10.0, downloaded from the University of Groningen on 2026-09-23. Retained variables are `rgdpo` (output-side real GDP at chained PPPs), `emp` (persons engaged), `cn` (capital stock at current PPPs), `ctfp` (TFP level at current PPPs), and `hc` (human capital index). The main productivity measure is annual growth in PWT output per worker (`rgdpo / emp`), averaged over t-4 through t. This is labour productivity, not TFP; the concepts are kept distinct.

The Phase 3 panel is a separate `data/processed/structural_panel_phase3.csv`; Phase 1 and Phase 2 processed files remain unchanged. Models use country and year fixed effects with country-clustered standard errors. The five-year forward outcome overlaps, so clustering is a pragmatic treatment of serial dependence rather than a complete solution. The productivity interaction is mean-centred and summarized at the 25th, 50th, and 75th productivity percentiles. Results are exploratory conditional associations, not causal estimates or forecasts. Limited labour-series history and PWT ending in 2019 reduce complete-case samples.

## Phase 5 structural-financial tests

Phase 5 formally tests whether three separate financial measures are associated with subsequent real GDP-per-capita growth after observable structural and macroeconomic controls: the published BIS credit gap, the deviation of the debt-service ratio from its prior ten-year mean, and five-year real house-price growth. No composite risk score is constructed. The primary outcome is the existing mean annual growth measure over t+1 through t+5. One- and three-year outcomes are additionally constructed from exact calendar years; a missing year invalidates the corresponding forward window. Forward outcomes are never predictors.

The pre-specified baseline controls are the ten-year working-age-share change, log real GDP per capita, backward five-year average investment share, and backward five-year average inflation. Each financial family has a controls-only baseline on the identical complete sample, a raw continuous-variable model, and a separate one-standard-deviation interpretation. Country and year fixed effects are included throughout, with standard errors clustered by country. The overlapping five-year outcomes induce serial dependence; country clustering is a pragmatic treatment, not perfect identification.

The primary demographic interactions mean-centre working-age change and the financial measure on that model's analytical sample. The interaction coefficient is not interpreted alone: marginal associations and 95% confidence intervals are reported at the 25th percentile, median, and 75th percentile of each interacting variable. These are conditional associations and do not establish that demographic deterioration causes vulnerability or changes its effects.

For nonlinearity, Phase 5 reports descriptive growth by within-sample financial quartile and fits one quadratic specification using the centred financial measure and its square. It does not search higher-order polynomials or use top-quintile classifications as the starting model. Common-sample comparisons fit controls-only, controls plus the financial measure, and the interaction specification on identical observations within each financial family; sample identity is checked programmatically.

Robustness analyses add the Phase 3 labour-force-utilisation and productivity controls for credit gap and DSR, repeat key models on the deterministic `year % 5 == 0` non-overlapping sample, compare one-, three-, and five-year continuous-variable models, and use expanding point-in-time quintiles for credit-gap and DSR episode indicators. Point-in-time thresholds use only earlier observed values for that country, require at least five prior observations, and classify demographic deterioration against the prior-history median. These sensitivity samples differ in size and coverage, so coefficient changes must be read together with changes in N and country coverage.

The prespecified summary keeps the six primary financial and interaction specifications distinct from sensitivity results. Phase 5 contains multiple tests; estimates should be assessed by magnitude, uncertainty, sample composition, and stability across pre-specified checks rather than ranked or selected by p-value. The panel is observational, financial measures have different units and country coverage, five-year windows overlap, and fixed effects do not remove unobserved time-varying confounding. No causal interpretation or investment recommendation is intended. No credible systemic-banking-crisis dataset was already available for a clean local country-year audit, so crisis validation is deferred rather than assigned manually.

Run Phase 5 with:

```powershell
python -m src.phase5_analysis
```

The analysis reads `data/processed/structural_financial_panel.csv` without modifying it. Phase 5 tables are written to `outputs/tables/` and its three figures to `outputs/figures/`; Phase 1-4 processed datasets are preserved.

## Phase 6 macro-financial vulnerability and asset outcomes

Phase 6 uses the latest accessible official JST Macrohistory Database, Release 6 (2022), retrieved from the provider on 2026-09-29. The workbook is downloaded from the [official JST database page](https://www.macrohistory.net/database/) and saved separately under `data/raw/jst/`; exact source URL, release, retrieval date, coverage, licence, and selected fields are recorded in `data/raw/jst/source_metadata.json`. Official documentation includes the [Release 6 notes](https://www.macrohistory.net/app/download/9834516169/JST_documentationR6.pdf?t=1676279836), [returns documentation](https://www.macrohistory.net/app/download/9918957869/JST_RORE_Documentation_R6.pdf?t=1658254098), and [crisis chronology](https://www.macrohistory.net/app/download/9844625569/JSTcrisis_chronology.pdf?t=1616702593). The data consulted covers 18 advanced economies, 1870-2020. Four JST countries (Belgium, Finland, Ireland, Portugal) are outside the existing 29-country project universe; they are excluded by ISO3 mapping, not fuzzy name matching.

**Licence restriction:** JST is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. The provider expressly prohibits commercial data providers from integrating or reselling the database. This use is exploratory research validation only. JST data and derived outputs cannot simply become the data backend of a commercial Structural Risk Analytics product; any commercial use would require separate permission or a different appropriately licensed source.

JST fields used are limited to `eq_tr` (nominal local-currency equity total return), `bond_tr` (nominal government-bond total return), `bill_rate` (nominal government-bill return, coupon divided by prior-year price), `housing_tr` (nominal local-currency housing total return), `cpi` (consumer price index, 1990=100), and `crisisJST` (systemic-financial-crisis indicator, equal to one in a documented onset year). The return fields are decimal ratios, not percentage points. CPI inflation is derived as the exact-calendar-year CPI ratio minus one. Real return is `(1 + nominal return) / (1 + CPI inflation) - 1`; simple subtraction is not used. Total-return series, rather than price changes alone, are used for the primary asset outcomes.

The market validation panel is separate at `data/processed/market_validation_panel.csv`. It starts from the structural-financial panel as the signal-time spine and merges JST fields only for matched ISO3/year keys; prior Phase 1-5 data is not overwritten. A signal observed at year t predicts only returns beginning at t+1. Forward 3- and 5-year returns are compounded and annualized, and every forward window requires exact consecutive calendar years. Worst annual equity return is the minimum annual real return in the stated future window, not a maximum drawdown. Crisis-next-horizon indicators use only JST's documented `crisisJST` onset years in t+1 through t+horizon.

Primary outcomes are credit gap versus five-year annualized real equity return (P6-EQ), worst subsequent annual equity return (P6-EQ-DOWN), five-year annualized real bond return (P6-BOND), and JST crisis onset within three years (P6-CRISIS). Parsimonious models are primary; adding the specified macro controls is a sample-sensitive robustness check. DSR, bills, housing, shorter horizons, nonlinear terms, quartile summaries, and point-in-time elevated credit-gap classifications are secondary or sensitivity analyses. Country and year fixed effects and country-clustered standard errors are used where models are estimated. Overlapping forward-return windows imply serial dependence; country clustering is pragmatic, not perfect identification.

The future-crisis-conditioned equity model is labelled **ex-post decomposition** because its crisis indicator is not known at t. It is not point-in-time predictive evidence and must never be treated as a trading or backtest model. These are historical, local-currency, observational associations; JST's advanced-economy coverage is narrower than the project universe and asset histories are unbalanced. No causal claim or investment strategy is produced in Phase 6, and no portfolio is optimized or recommended. An experimental multi-asset strategy would be a separate future research question, subject to licensing, out-of-sample validation, transaction costs, and investability checks.

Run Phase 6 with:

```powershell
python -m src.phase6_analysis
```

The pipeline writes `data/processed/market_validation_panel.csv`, a country mapping and coverage tables under `outputs/coverage/`, four figures and Phase 6 tables under `outputs/figures/` and `outputs/tables/`. Do not redistribute the JST source or derivative data for commercial purposes under its current licence.

## Phase 7 point-in-time crisis early-warning validation

Phase 7 asks whether three pre-specified macro-financial signal families contain chronological out-of-sample information about a JST systemic banking crisis starting within the next three years. It is an early-warning validation only: not an investment strategy, backtest, production predictor, machine-learning project, or systemic-risk score. No asset returns, future GDP, future crisis labels, or demographic interactions are predictors.

The event source is JST Release 6's `crisisJST` variable, which the official crisis chronology defines as one in the first year of a systemic banking crisis and zero otherwise. The source onset years are used directly; Phase 7 does not reconstruct or manually revise multi-year episodes. `systemic_crisis_next_3y` is any documented onset in exact years t+1 through t+3. `past_crisis_previous_3y` uses t-3 through t-1 solely as a reverse-timing placebo.

The primary signal families are fixed in advance: S1 credit gap; S2 credit gap plus DSR deviation; and S3 those two plus five-year real house-price growth. DSR-only and property-only fits are secondary. The transparent equal-weight index standardizes each input using only earlier training observations, averages the z-scores, and is a sensitivity index, not a systemic-risk score. Its probability-like score is the fixed logistic transform of that mean; it is not calibrated or optimized.

Evaluation is chronological. Each expanding-window evaluation year is scored using earlier data only. For an evaluation year y, the latest training origin is y-4: this three-year label-maturity embargo ensures each t+1:t+3 outcome in training is fully realized by y-1. Training additionally requires five distinct prior calendar years and three distinct JST crisis starts represented in mature training labels. This fixed minimum preserves an expanding evaluation for the sparse combined-signal sample; the resulting estimates remain highly event-limited. Each fold first fits unpenalized logistic regression. Only a non-convergent, singular, or non-finite fit falls back to L2 logistic regression with fixed C=1.0; the fallback is not tuned. The chosen method is recorded per prediction. Continuous predictors are standardized separately within every training window and that scaler is applied to that year's evaluation observations only.

The preferred 2007 test split is audited, but an unembargoed training set through 2006 would include outcome windows extending into test years. The leakage-safe initial split therefore caps training outcome origins at 2003, allowing their three-year labels to mature by 2006. DSR has no adequate usable training coverage by that cutoff, so the expanding window is the primary evaluation. Country controls are a sensitivity model; pooled models retain countries even when they have no crisis event.

Results report ROC-AUC, average-precision PR-AUC, Brier score, the evaluation event prevalence, and a Brier benchmark using each fold's historical training event rate. PR-AUC is compared with the positive prevalence because this is a rare-event problem. Threshold-free measures come first. Research-only top 10%, 20%, and 30% alert bands are then assigned by risk rank within each evaluation year; their cutoffs are not optimized. OOS event-level recall tests whether each unique crisis appears in any alert band at least once in its preceding three years, alongside country-year metrics and t-1/t-2/t-3 lead tables. Event-level summaries include the earliest successful warning lead per alert band.

S1-S3 are evaluated on natural coverage and an exactly common complete sample. The historical credit-gap percentile signal uses only each country's preceding credit-gap observations and requires five prior values. Calibration uses broad prediction-rank bins where counts permit. Cooldown sensitivity retains earlier training history but excludes evaluation candidates in a crisis-start year and the following two years before re-running S1 and S2. The past-crisis placebo is a full-sample diagnostic, not a predictive target; similarity to future association may indicate persistence or reverse timing.

Rare events, overlapping three-year outcome windows, few independent crisis starts, country-specific event concentration, incomplete DSR/property coverage, historical data revisions, and changing crisis base rates limit interpretation. Observation-level counts overstate independent event information, so unique-event recall is reported separately. No causal or production-prediction claim is made. Phase 7 does not recommend or optimize a portfolio; any later experimental resilience-portfolio test would remain a distinct research exercise, not a validated investment strategy.

Run Phase 7 with:

```powershell
python -m src.phase7_analysis
```

The analysis reads `data/processed/market_validation_panel.csv` and writes a separate `data/processed/phase7_analytical_panel.csv`, Phase 7 tables in `outputs/tables/`, and four figures in `outputs/figures/`. Phase 1-6 processed files remain unchanged.

## Phase 8 preregistered experimental resilience portfolio test

Phase 8 is a frozen falsification experiment: test whether the modest point-in-time credit-gap information from Phase 7 improves country-level downside resilience over simple static allocations. It is not a strategy recommendation, commercial backtest, or claim of a usable crisis predictor. It uses only the validated JST real local-currency total returns for equities, government bonds, and bills. Housing is excluded from the primary portfolio because its national total-return series is not treated as a liquid annual-rebalanced instrument. JST remains research-only under CC BY-NC-SA 4.0; these results are not a commercial backtest.

The state uses the credit gap's empirical percentile against that country's prior observations only, requires at least five prior observations, and classifies states as NORMAL below p80, ELEVATED from p80 to below p90, and HIGH at or above p90. The signal at year t changes weights only for complete real equity/bond/bill returns in t+1. Missing returns are not filled. Only country-years with a valid prior-only state and all three next-year asset returns enter the matched strategy panel.

Weights are frozen: R1 uses (60/20/20) NORMAL, (45/30/25) ELEVATED, and (30/40/30) HIGH for equity/bond/bill. R2 merges NORMAL and ELEVATED into a 60/20/20 state and uses 30/40/30 for HIGH. Static benchmarks are B1 100% equity, B2 60/40 equity/bonds, and B3 60/20/20 equity/bonds/bills. No strategy is optimized, leveraged, shorted, volatility-targeted, or selected after seeing returns. The mandatory B4 uses R1's full matched-sample average realized asset weights as a retrospective static exposure comparator; B4 is not implementable point-in-time and separates average equity exposure from timing, rather than serving as a tradable benchmark.

All portfolios are rebalanced annually. Turnover is one-half the absolute difference between new target weights and prior-year realized, drifted weights; it costs 10 bps per 100% one-way turnover. Initial establishment is not charged; turnover history resets after gaps in complete return observations and that convention is recorded. Zero-cost country results are included as sensitivity. Benchmarks and dynamic portfolios use identical country-years, verified in the output. Country histories are reported separately; no artificial pooled wealth path is formed. Annualized return is the geometric mean per observed return year, while CAGR is reported only for contiguous calendar histories. Maximum drawdown is annual-frequency and computed within contiguous observed segments, not monthly or daily precision.

Primary resilience comparisons are annual-frequency maximum drawdown, worst year, and fifth-percentile annual return, with annualized return, volatility, and Sortino as secondary. Retrospective JST crisis starts are evaluation labels only; they never determine weights. Outputs separately show crisis-start and following-year returns, false alerts (elevated/high states without a JST start in the next three years), missed events (all three available preceding states NORMAL), and correctly alerted events. Country-level bootstrap resampling, leave-one-country-out results, pre/post-2000 sensitivities, and a zero-cost comparison describe uncertainty and sensitivity; small country/event counts remain a major limitation.

Run Phase 8 with:

```powershell
python -m src.phase8_analysis
```

The pipeline reads the Phase 6 market-validation panel and writes `data/processed/phase8_portfolio_panel.csv`, Phase 8 tables under `outputs/tables/`, and five figures under `outputs/figures/`; Phase 1-7 processed files remain unchanged. No investment recommendation is made. A positive finding would only motivate a later independent-market validation phase, not establish an investable strategy.
