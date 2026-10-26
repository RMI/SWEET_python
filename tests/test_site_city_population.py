"""A Climate TRACE site can grow with its nearby city instead of its country.

Climate TRACE hands in the population of the UN-listed city near a site, as JSON text on
the site row. SWEET uses the city's own numbers where it has them and the country's shape
outside them (`population.city_population_series`). A row without a city grows with the
country, as before.
"""

import json

import numpy as np
import pandas as pd
import pytest

from SWEET_python.city_params import City
from SWEET_python.population import (
    city_population_series,
    country_population_series,
    parse_city_population,
)

WINDOW = list(range(1970, 2051))
USA = country_population_series("USA").loc[WINDOW]


# --------------------------------------------------------------------------- #
# Reading the city
# --------------------------------------------------------------------------- #

def test_json_text_with_string_years_reads_as_int_years():
    assert parse_city_population('{"1975": 50000, "2025": 80000}') == {1975: 50000.0, 2025: 80000.0}


@pytest.mark.parametrize("blank", [None, np.nan, "", "{}"])
def test_no_city_reads_as_none(blank):
    assert parse_city_population(blank) is None


def test_years_under_the_threshold_are_dropped_not_read_as_zero():
    """The UN leaves a city blank in the years it was under 50,000 people."""
    assert parse_city_population({1975: None, 1980: 0, 1985: 51000}) == {1985: 51000.0}


# --------------------------------------------------------------------------- #
# Filling the city's gaps
# --------------------------------------------------------------------------- #

def test_the_citys_own_years_are_kept_exactly():
    city = {year: 100_000.0 + 1_000 * (year - 1975) for year in range(1975, 2051)}
    series = city_population_series(city, USA)
    assert series.loc[1975:2050].tolist() == [city[y] for y in range(1975, 2051)]


def test_years_before_the_city_follow_the_countrys_shape_and_join_smoothly():
    """A city first listed in 2005 at 50,000: earlier years move with the country."""
    series = city_population_series({2005: 50_000.0, 2025: 200_000.0}, USA)
    assert series[2005] == 50_000.0
    for year in (1970, 1990, 2004):
        assert series[year] == pytest.approx(50_000.0 * USA[year] / USA[2005])


def test_years_after_the_city_follow_the_countrys_shape():
    series = city_population_series({1975: 60_000.0, 2025: 90_000.0}, USA)
    assert series[2050] == pytest.approx(90_000.0 * USA[2050] / USA[2025])


def test_between_two_reported_years_growth_is_constant():
    """For a source that reports every five years."""
    series = city_population_series({2000: 100_000.0, 2005: 200_000.0}, USA)
    rate = 2 ** (1 / 5)
    assert series[2001] == pytest.approx(100_000.0 * rate)
    assert series[2004] == pytest.approx(100_000.0 * rate ** 4)


def test_a_fast_city_never_reaches_zero_going_back():
    """A straight line back from 50,000 in 2005 to 200,000 in 2025 hits zero by 1998."""
    series = city_population_series({2005: 50_000.0, 2025: 200_000.0}, USA)
    assert (series > 0).all()
    assert series.index.tolist() == WINDOW


@pytest.mark.parametrize("city, country", [(None, USA), ({2005: 1.0}, None), ({1960: 1.0}, USA)])
def test_without_a_city_or_country_or_overlap_there_is_no_series(city, country):
    assert city_population_series(city, country) is None


# --------------------------------------------------------------------------- #
# Through a site
# --------------------------------------------------------------------------- #

def _site(city_population):
    row = pd.Series({
        "asset_identifier": 999802, "iso3_country": "USA", "type": "Sanitary Landfill",
        "income_id": "HIC", "annual_incoming_waste": 100_000.0, "incoming_waste_year": 2023.0,
        "site_open_year": 1980.0, "site_close_year": np.nan, "waste_depth": 10.0,
        "landfill_gas_collection": False, "gas_collection_efficiency": np.nan,
        "latitude": 39.0, "longitude": -95.0, "precipitation": 800.0, "temperature": 15.0,
        "city_population": city_population,
    })
    pop_data = pd.DataFrame(
        {"growth_rate_historic": [1.02], "growth_rate_future": [1.01],
         **{f"pop_{y}": [USA[y]] for y in WINDOW}},
        index=["USA"],
    )
    city = City("999802")
    city.site_only_estimate_trace(row, pop_data=pop_data)
    return city.baseline_parameters


def _shape(series, anchor=2023):
    total = series.sum(axis=1) if isinstance(series, pd.DataFrame) else series
    return (total / total.loc[anchor]).loc[WINDOW].to_numpy()


def test_a_site_with_a_city_grows_with_the_city():
    city = {2005: 50_000.0, 2025: 200_000.0}
    baseline = _site(json.dumps(city))
    expected = city_population_series(city, USA)
    assert np.allclose(_shape(baseline.waste_generated_df.df), _shape(expected))
    assert np.allclose(baseline.population_series.loc[WINDOW], expected)


def test_a_site_with_no_city_still_grows_with_its_country():
    baseline = _site(None)
    assert np.allclose(_shape(baseline.waste_generated_df.df), _shape(USA))
