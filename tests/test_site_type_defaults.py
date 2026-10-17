"""The site type assumed for a site whose type nobody recorded.

Before 2026-10-13 this guess was made by three different rules in nine places:
Climate TRACE's pipeline used income, region and named countries, while SWEET's
site and city paths used one of two region lists (one with Eastern Europe, one
with Australia and New Zealand instead). A Kazakh site with no type was a
sanitary landfill in Climate TRACE and an open dump in WasteMAP's site tool.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

import SWEET_python.city_params as city_params
import SWEET_python.site_type_defaults as std
from SWEET_python.country_income import INCOME_ID_BY_ISO3


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "iso3, why",
    [
        ("AUS", "high income; its region is not in the list"),
        ("ABW", "high income in the Caribbean"),
        ("CHL", "high income in South America"),
        ("POL", "Eastern Europe"),
        ("KAZ", "Central Asia"),
        ("TUR", "Western Asia"),
        ("JPN", "Eastern Asia"),
        ("MEX", "named: open-dumps 3% of its waste"),
        ("BRA", "named"),
        ("CRI", "named"),
        ("CHN", "named, and Eastern Asia"),
        ("RUS", "named, and Eastern Europe"),
    ],
)
def test_engineered_disposal_systems_default_to_landfill(iso3, why):
    assert std.default_site_type(iso3) == "Sanitary Landfill", why


@pytest.mark.parametrize("iso3", ["NGA", "IND", "IDN", "PAK", "EGY", "PER", "ZAF"])
def test_other_countries_default_to_an_open_dump(iso3):
    assert std.default_site_type(iso3) == "Dumpsite"


@pytest.mark.parametrize("iso3", ["MNG", "YEM"])
def test_mongolia_and_yemen_are_dumps_whatever_their_region_says(iso3):
    """Central Asia and Western Asia would otherwise sweep them in."""
    assert std.default_site_type(iso3) == "Dumpsite"
    assert std.default_site_type(iso3, income_id="HIC") == "Dumpsite"


@pytest.mark.parametrize("iso3", [None, np.nan, pd.NA, "", "XXX"])
def test_an_unknown_country_is_an_open_dump(iso3):
    assert std.default_site_type(iso3) == "Dumpsite"


def test_a_lower_case_code_is_read_the_same():
    assert std.default_site_type("aus") == "Sanitary Landfill"
    assert std.default_site_type("mng") == "Dumpsite"


def test_a_callers_own_income_is_used_when_it_has_one():
    """Climate TRACE passes the income on its site rows."""
    assert std.default_site_type("NGA", income_id="HIC") == "Sanitary Landfill"
    assert std.default_site_type("ARG", income_id="UMC") == "Dumpsite"


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, ""])
def test_a_missing_income_falls_back_to_sweets_table(missing):
    assert std.default_site_type("AUS", income_id=missing) == "Sanitary Landfill"
    assert std.default_site_type("NGA", income_id=missing) == "Dumpsite"


def test_landfill_is_default_agrees_with_default_site_type():
    for iso3 in INCOME_ID_BY_ISO3:
        landfill = std.landfill_is_default(iso3)
        assert std.default_site_type(iso3) == ("Sanitary Landfill" if landfill else "Dumpsite")


# --------------------------------------------------------------------------- #
# The income table
# --------------------------------------------------------------------------- #

def test_the_income_table_is_the_one_climate_trace_reads():
    """217 countries, the What a Waste 2.0 classification (2018).

    Pinned on three countries the World Bank has reclassified since, so a swap to
    the current list fails here instead of quietly disagreeing with Climate TRACE.
    """
    assert len(INCOME_ID_BY_ISO3) == 217
    assert set(INCOME_ID_BY_ISO3.values()) == {"HIC", "UMC", "LMC", "LIC"}
    assert INCOME_ID_BY_ISO3["ARG"] == "HIC"
    assert INCOME_ID_BY_ISO3["BGR"] == "UMC"
    assert INCOME_ID_BY_ISO3["CRI"] == "UMC"


def test_every_listed_country_has_a_sweet_region_except_the_channel_islands():
    from SWEET_python import defaults_2019

    unplaced = set(INCOME_ID_BY_ISO3) - set(defaults_2019.region_lookup_iso3)
    assert unplaced == {"CHI"}


# --------------------------------------------------------------------------- #
# Every caller uses it
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "method, calls",
    [
        ("load_andre_params", 3),  # the city split's last fallback, three branches
        ("site_only_estimate", 1),
        ("site_only_estimate_trace", 1),
        ("citysite_estimate_trace", 1),
        ("sdst_prepopulate", 2),  # by location, and a listed site with no type
    ],
)
def test_each_path_that_guesses_a_type_calls_the_shared_rule(method, calls):
    source = inspect.getsource(getattr(city_params.City, method))
    assert source.count("site_type_defaults.") == calls
    assert "landfill_default_regions" not in source
    assert '"Australia and New Zealand"' not in source


def test_a_site_with_no_type_is_typed_by_the_rule_end_to_end():
    """A Kazakh site with no type models as a sanitary landfill, MCF 1.0.

    Its region list used to make it an open dump (0.6) in this path while Climate
    TRACE typed it a landfill.
    """
    row = pd.Series(
        {
            "asset_identifier": 990001,
            "iso3_country": "KAZ",
            "type": np.nan,
            "income_id": np.nan,
            "annual_incoming_waste": 100_000.0,
            "incoming_waste_year": 2023.0,
            "site_open_year": 2000.0,
            "site_close_year": 2050.0,
            "waste_depth": np.nan,
            "landfill_gas_collection": False,
            "gas_collection_efficiency": 0.0,
            "latitude": 51.0,
            "longitude": 71.0,
            "precipitation": 300.0,
            "temperature": 5.0,
        }
    )
    pop_data = pd.DataFrame(
        {"growth_rate_historic": [1.01], "growth_rate_future": [1.01]}, index=["KAZ"]
    )
    city = city_params.City(city_name="990001")
    city.site_only_estimate_trace(row, pop_data=pop_data)
    landfill = city.baseline_parameters.landfills[0]
    assert landfill.site_type == "Sanitary Landfill"
    assert float(pd.Series(landfill.mcf).iloc[-1]) == 1.0
