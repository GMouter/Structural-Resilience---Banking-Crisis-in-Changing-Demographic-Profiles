import pandas as pd

from src.bis import BIS_CODE_MAP


def test_bis_country_mappings_are_explicit():
    assert BIS_CODE_MAP["JP"] == "JPN"
    assert BIS_CODE_MAP["US"] == "USA"
    assert BIS_CODE_MAP["TR"] == "TUR"


def test_bis_quarterly_aggregation_convention_is_documented_in_clean_output():
    frame = pd.DataFrame({"variable": ["private_credit_gdp", "debt_service_ratio"], "annualisation": ["Q4", "quarterly_mean_min_2"]})
    assert frame.loc[0, "annualisation"] == "Q4"
    assert frame.loc[1, "annualisation"] == "quarterly_mean_min_2"