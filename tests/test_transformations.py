import pandas as pd

from src.transformations import add_transformations


def make_panel(years_a=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010), years_b=(2005, 2006, 2007, 2008, 2009, 2010)):
    rows = []
    for country, years, base in [("AAA", years_a, 0), ("BBB", years_b, 100)]:
        for year in years:
            rows.append({"country_code": country, "year": year, "working_age_share": base + year, "old_age_dependency": base + year / 10, "fertility": base + year / 100, "real_gdp_pc_growth": year - 1999, "investment_gdp": year - 1990, "inflation": year - 1980, "current_account_gdp": year - 1970})
    return pd.DataFrame(rows)


def test_transformations_do_not_cross_country_boundaries():
    result = add_transformations(make_panel())
    b_2010 = result[(result.country_code == "BBB") & (result.year == 2010)].iloc[0]
    assert b_2010.working_age_change_5y == 5
    assert b_2010.working_age_change_10y is pd.NA


def test_five_and_ten_year_changes_use_calendar_years():
    result = add_transformations(make_panel())
    row = result[(result.country_code == "AAA") & (result.year == 2010)].iloc[0]
    assert row.working_age_change_5y == 5
    assert row.working_age_change_10y == 10


def test_rolling_windows_and_forward_exclude_or_include_t_as_defined():
    result = add_transformations(make_panel())
    row = result[(result.country_code == "AAA") & (result.year == 2005)].iloc[0]
    assert row.real_gdp_pc_growth_5y_backward == 4
    assert row.real_gdp_pc_growth_5y_forward == 9
    assert row.real_gdp_pc_growth_5y_forward != 8


def test_missing_calendar_year_does_not_create_lag_or_rolling_value():
    years = (2000, 2002, 2004, 2005, 2006, 2007, 2008, 2009, 2010)
    result = add_transformations(make_panel(years_a=years, years_b=()))
    change_row = result[(result.country_code == "AAA") & (result.year == 2006)].iloc[0]
    average_row = result[(result.country_code == "AAA") & (result.year == 2005)].iloc[0]
    assert pd.isna(change_row.working_age_change_5y)
    assert pd.isna(average_row.real_gdp_pc_growth_5y_backward)
