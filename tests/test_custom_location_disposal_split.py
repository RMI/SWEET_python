"""A Custom Location buries or dumps its waste in every country.

`City.dst_baseline_blank` (WasteMAP's Custom Location) looked up a country's
disposal split with `fraction_open_dumped_country.get(iso3,
fraction_open_dumped.get(region, 0))`, and the same for `fraction_landfilled`.
Two gaps in those tables made the split (0, 0, 0) for 31 countries:

- Southern Asia, South-Eastern Asia and Southern Africa have no regional row.
  This is not a label mismatch: every other regional table carries these three
  regions under the same names, as `0.0  # np.nan`. None of their 28 countries
  here has a row of its own.
- Canada's, Germany's and Switzerland's country rows are all zeros.

With a zero split every landfill took 0% of the waste, so the city modelled no
landfill methane at all. Pakistan and the Philippines at 2M people gave 0 t CH4
in 2040, against 28,745 t for Mexico. The `except KeyError` fallback written for
this never ran, because `.get` never raises.

The cities table (`City.load_andre_params`) handles both gaps. A region with no
row falls back to all landfill or all dumpsite by `landfill_default_regions`, and
the three zero-row countries are landfilled. Custom Location now takes the split
the cities table gives a city with no landfill data
(`defaults_2019.disposal_split_for`).
"""

import numpy as np
import pytest

from SWEET_python import defaults_2019
from SWEET_python.city_params import City

POPULATION = 2_000_000
PRECIPITATION = 500.0
TEMPERATURE = 25.0
YEAR = 2040

# Every country whose Custom Location split was (0, 0, 0).
NO_ROW = {
    "Southern Asia": ["BGD", "BTN", "IND", "IOT", "IRN", "LKA", "MDV", "NPL", "PAK"],
    "South-Eastern Asia": [
        "BRN", "IDN", "KHM", "LAO", "MMR", "MYS", "PHL", "THA", "TLS", "VNM",
    ],
    "Southern Africa": ["ATF", "BWA", "LSO", "MOZ", "NAM", "SHN", "SWZ", "ZAF", "ZWE"],
}
ZERO_COUNTRY_ROW = ["CAN", "CHE", "DEU"]
ZEROED = sorted([iso3 for codes in NO_ROW.values() for iso3 in codes] + ZERO_COUNTRY_ROW)


def _custom_location(iso3):
    city = City("Custom Location")
    city.dst_baseline_blank(iso3, POPULATION, PRECIPITATION, TEMPERATURE)
    return city.baseline_parameters


class _NoDataRow(dict):
    """A cities-table row in which every field not given is missing (NaN)."""

    def __missing__(self, key):
        return np.nan


def _cities_table_split(iso3):
    """The split `load_andre_params` gives a city with no landfill data in `iso3`.

    The row says where the city is and nothing about its waste: every landfill and
    diversion share is NaN, so the loader takes its defaults for both, which is
    the cities-table counterpart of a Custom Location.
    """
    country = next(
        name
        for name, code in defaults_2019.country_to_iso3.items()
        if code == iso3 and name in defaults_2019.region_lookup
    )
    row = _NoDataRow(
        {
            "country": country,
            "iso": iso3,
            "population_count": POPULATION,
            "population_year": 2022.0,
            "population_data_source": "test",
            "msw_generated_metric_tons_per_year": 500_000.0,
            "msw_generated_year": 2022.0,
            "mean_yearly_precip_2000_2021": PRECIPITATION,
            "mean_yearly_temp_2000_2021": TEMPERATURE,
            "Temperature (C)": TEMPERATURE,
            "latitude": 0.0,
            "longitude": 0.0,
            "historic_growth_rate": 1.0,
            "future_growth_rate": 1.0,
        }
    )
    city = City("Cities table")
    city.load_andre_params(row)
    return city.baseline_parameters.split_fractions


def _how_the_split_is_decided(iso3, region):
    if iso3 in defaults_2019.fraction_open_dumped_country:
        row = (
            defaults_2019.fraction_open_dumped_country[iso3]
            + defaults_2019.fraction_landfilled_country[iso3]
        )
        return "country row" if row > 0 else "all-zero country row"
    if region in defaults_2019.fraction_open_dumped:
        return "regional row"
    return "no row"


def _one_country_per_region_and_rule():
    """The first country of each (region, which row decides) pair, plus Germany.

    Comparing every country with the cities table takes ten seconds; one per
    pair runs every branch of both lookups for every region in a fifth of that.
    Germany is added so all three zero-row countries are covered.
    """
    first = {}
    for iso3, region in sorted(defaults_2019.region_lookup_iso3.items()):
        first.setdefault((region, _how_the_split_is_decided(iso3, region)), iso3)
    return sorted(set(first.values()) | set(ZERO_COUNTRY_ROW))


@pytest.mark.parametrize("iso3", ZEROED)
def test_a_custom_location_models_methane_from_the_waste_it_disposes_of(iso3):
    parameters = _custom_location(iso3)

    assert sum(landfill.fraction_of_waste for landfill in parameters.landfills) == (
        pytest.approx(1.0)
    )
    # Landfill methane only. Germany's total was already 26 t from composting
    # while its landfills took nothing.
    assert parameters.landfill_emissions.loc[YEAR, "total"] > 0


@pytest.mark.parametrize("iso3", ["PAK", "PHL", "ZAF"])
def test_a_country_in_a_region_with_no_row_dumps_its_waste(iso3):
    # None of the three regions is in `landfill_default_regions`.
    split = _custom_location(iso3).split_fractions

    assert (split.landfill_w_capture, split.landfill_wo_capture, split.dumpsite) == (
        0.0,
        0.0,
        1.0,
    )


@pytest.mark.parametrize("iso3", ZERO_COUNTRY_ROW)
def test_a_country_whose_row_is_all_zeros_landfills_its_waste(iso3):
    split = _custom_location(iso3).split_fractions

    assert (split.landfill_w_capture, split.landfill_wo_capture, split.dumpsite) == (
        0.0,
        1.0,
        0.0,
    )


@pytest.mark.parametrize("iso3", _one_country_per_region_and_rule())
def test_a_custom_location_starts_from_the_cities_tables_split(iso3):
    custom = _custom_location(iso3).split_fractions
    table = _cities_table_split(iso3)

    assert custom.model_dump() == pytest.approx(table.model_dump())


def test_every_country_has_a_disposal_split():
    splits = {
        iso3: defaults_2019.disposal_split_for(iso3, region)
        for iso3, region in defaults_2019.region_lookup_iso3.items()
    }

    assert {
        iso3: split
        for iso3, split in splits.items()
        if set(split) != {"landfill_w_capture", "landfill_wo_capture", "dumpsite"}
        or min(split.values()) < 0
        or sum(split.values()) != pytest.approx(1.0)
    } == {}
