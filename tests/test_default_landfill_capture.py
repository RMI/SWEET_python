"""A defaulted landfill share captures gas where its country's landfills do.

A city with no landfill shares of its own, and every Custom Location, takes its country's
or region's disposal split (IPCC 2019 Table 2.1). That table says nothing about gas
capture, so every defaulted landfill used to be modelled without it. ``landfill_capture``
gives the share of a country's landfilled waste that goes to landfills with gas
collection, and the defaulted split moves that share into "landfill with capture".

A city that reports its own landfill shares keeps them.
"""

import numpy as np
import pytest

from SWEET_python import defaults_2019, landfill_capture
from SWEET_python.city_params import City
from SWEET_python.landfill_capture import CaptureShare

from tests.test_custom_location_disposal_split import _NoDataRow, _cities_table_split

HALF = CaptureShare(0.5, 2020, "test", "test", "https://example.org")


@pytest.fixture
def half_of_usa_captures(monkeypatch):
    monkeypatch.setitem(landfill_capture.CAPTURE_SHARE_COUNTRY, "USA", HALF)


def test_the_capture_share_moves_landfilled_waste_and_nothing_else():
    split = {"landfill_w_capture": 0.0, "landfill_wo_capture": 0.8, "dumpsite": 0.2}
    out = landfill_capture.with_default_capture(split, "XXX", shares={"XXX": HALF})
    assert out == pytest.approx({"landfill_w_capture": 0.4, "landfill_wo_capture": 0.4, "dumpsite": 0.2})
    assert split["landfill_wo_capture"] == 0.8  # not changed in place


def test_a_country_with_no_row_captures_nothing_unless_it_is_high_income():
    split = {"landfill_w_capture": 0.0, "landfill_wo_capture": 1.0, "dumpsite": 0.0}
    assert landfill_capture.with_default_capture(split, "XXX", shares={}) == split
    assert landfill_capture.with_default_capture(split, "XXX", "UMC", shares={}) == split
    assert landfill_capture.with_default_capture(split, None, shares={"XXX": HALF}) == split
    high = landfill_capture.with_default_capture(split, "XXX", "HIC", shares={})
    assert high["landfill_w_capture"] == pytest.approx(landfill_capture.HIGH_INCOME_SHARE)


def test_a_researched_share_beats_the_high_income_default():
    assert landfill_capture.capture_share("XXX", "HIC", shares={"XXX": HALF._replace(share=0.9)}) == 0.9
    assert landfill_capture.capture_share("CAN", "HIC") == landfill_capture.CAPTURE_SHARE_COUNTRY["CAN"].share


def test_a_high_income_custom_location_takes_the_default_share():
    region = defaults_2019.region_lookup_iso3["JPN"]
    without = defaults_2019.disposal_split_for("JPN", region)
    high = defaults_2019.disposal_split_for("JPN", region, "HIC")
    assert without["landfill_w_capture"] == 0
    assert high["landfill_w_capture"] == pytest.approx(0.5 * without["landfill_wo_capture"])


def test_a_chinese_city_with_no_figures_takes_chinas_2020_split():
    assert defaults_2019.fraction_landfilled_country["CHN"] == 0.33
    assert defaults_2019.fraction_incinerated_country["CHN"] == 0.62
    assert defaults_2019.fraction_composted_country["CHN"] == 0.05
    split = defaults_2019.disposal_split_for("CHN", defaults_2019.region_lookup_iso3["CHN"], "UMC")
    assert split == {"landfill_w_capture": 0.0, "landfill_wo_capture": 1.0, "dumpsite": 0.0}


def test_a_custom_location_takes_its_countrys_capture_share(half_of_usa_captures):
    split = defaults_2019.disposal_split_for("USA", defaults_2019.region_lookup_iso3["USA"])
    assert split["landfill_w_capture"] == pytest.approx(split["landfill_wo_capture"])
    assert sum(split.values()) == pytest.approx(1.0)


def test_a_city_with_no_landfill_shares_takes_its_countrys_capture_share(half_of_usa_captures):
    table = _cities_table_split("USA")
    custom = defaults_2019.disposal_split_for("USA", defaults_2019.region_lookup_iso3["USA"])
    assert table.landfill_w_capture == pytest.approx(0.5 * (table.landfill_w_capture + table.landfill_wo_capture))
    assert table.landfill_w_capture == pytest.approx(custom["landfill_w_capture"])


def test_a_city_that_reports_its_landfill_shares_keeps_them(half_of_usa_captures):
    row = _NoDataRow(
        {
            "country": "United States",
            "iso": "USA",
            "population_count": 1_000_000,
            "population_year": 2022.0,
            "population_data_source": "test",
            "msw_generated_metric_tons_per_year": 500_000.0,
            "msw_generated_year": 2022.0,
            "mean_yearly_precip_2000_2021": 500.0,
            "mean_yearly_temp_2000_2021": 25.0,
            "latitude": 0.0,
            "longitude": 0.0,
            "historic_growth_rate": 1.0,
            "future_growth_rate": 1.0,
            "waste_treatment_controlled_landfill_percent": 70.0,
            "waste_treatment_recycling_percent": 30.0,
        }
    )
    city = City("Reports its landfills")
    city.load_andre_params(row)
    split = city.baseline_parameters.split_fractions
    assert split.landfill_w_capture == 0
    assert split.landfill_wo_capture == pytest.approx(1.0)


# The table itself.

ROWS = sorted(landfill_capture.CAPTURE_SHARE_COUNTRY.items())


@pytest.mark.parametrize("iso3,row", ROWS, ids=[iso3 for iso3, _ in ROWS])
def test_each_capture_share_is_a_sourced_share_of_a_known_country(iso3, row):
    assert iso3 in defaults_2019.region_lookup_iso3
    assert 0 < row.share <= 1
    assert 2005 <= row.year <= 2026
    assert row.basis and row.source and row.url.startswith("http")


@pytest.mark.parametrize("iso3", [iso3 for iso3, _ in ROWS])
def test_custom_location_and_the_cities_table_agree_on_capture(iso3):
    custom = defaults_2019.disposal_split_for(iso3, defaults_2019.region_lookup_iso3[iso3])
    table = _cities_table_split(iso3)
    assert table.model_dump() == pytest.approx({**table.model_dump(), **custom})
