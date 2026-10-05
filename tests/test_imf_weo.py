from src.imf_weo import WEO_HISTORICAL_LAST_YEAR, WEO_INDICATORS


def test_weo_forecasts_are_excluded_by_historical_cutoff():
    assert WEO_HISTORICAL_LAST_YEAR == 2024
    assert all("government" in definition.lower() for _, definition, _ in WEO_INDICATORS.values())


def test_general_government_concepts_are_explicit():
    assert WEO_INDICATORS["government_debt_gdp"][0] == "GGXWDG_NGDP"
    assert WEO_INDICATORS["fiscal_balance_gdp"][0] == "GGXCNL_NGDP"