import pytest

from src.config import INDICATORS
from src.world_bank import parse_response


def test_indicator_mapping_contains_exactly_nine_indicators():
    assert set(INDICATORS) == {
        "working_age_share", "old_age_dependency", "fertility", "population_growth",
        "real_gdp_pc_growth", "real_gdp_growth", "investment_gdp", "inflation", "current_account_gdp",
    }


def test_parse_response_preserves_observation_fields():
    payload = [{"page": 1, "pages": 1, "sourceid": "2"}, [{
        "countryiso3code": "JPN", "date": "2000", "value": 60.5,
        "country": {"value": "Japan"}, "indicator": {"value": "Test"}, "obs_status": "",
    }]]
    assert parse_response(payload, "TEST.CODE")[0]["country_code"] == "JPN"
    assert parse_response(payload, "TEST.CODE")[0]["year"] == 2000
    assert parse_response(payload, "TEST.CODE")[0]["indicator_code"] == "TEST.CODE"


def test_parse_response_rejects_malformed_response():
    with pytest.raises(ValueError):
        parse_response([], "TEST.CODE")
