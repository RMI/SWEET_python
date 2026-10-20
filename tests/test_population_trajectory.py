"""A place's population by year, worked out once and stored, grows its waste as SWEET would.

WasteMAP's map build stores every city's and country's population trajectory with the
build, and its city DST hands a city's back to `load_csv_new` as `population_series`.
Before that the build and the city DST each re-ran `_city_growth`, so they could grow a
city differently whenever WasteMAP's SWEET pin moved. These tests hold
`city_population_trajectory` to `_city_growth`'s choice for both kinds of city, the
country trajectories to what a Custom Location grows by, and `load_csv_new` to the
series it is handed.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python import city_params
from SWEET_python.city_params import (
    City,
    city_population_trajectory,
    country_population_trajectory,
)
from SWEET_python.population import country_population_series
from test_load_csv_composition import CITIES, _cities_table_row

YEARS = list(range(1970, 2051))


def test_a_big_city_follows_its_own_two_rates():
    trajectory, source = city_population_trajectory("NGA", 2020, 15_000_000.0, 1.0556, 1.036)

    assert source == city_params.UN_CITY_RATES == "UN city rates"
    assert list(trajectory.index) == YEARS
    assert trajectory.loc[2020] == 15_000_000.0
    assert trajectory.loc[2030] == pytest.approx(15_000_000 * 1.036**10, rel=1e-13)
    assert trajectory.loc[2000] == pytest.approx(15_000_000 * 1.0556**-20, rel=1e-13)


@pytest.mark.parametrize(
    "population, historic, future",
    [(250_000.0, 1.0556, 1.036), (1_800_000.0, 1.0022, 1.1687)],
    ids=["a town under 300k", "implausible rates"],
)
def test_any_other_city_follows_its_countrys_un_population(population, historic, future):
    trajectory, source = city_population_trajectory("BRA", 2022, population, historic, future)
    brazil = country_population_series("BRA")

    assert source == city_params.COUNTRY_CURVE == "country curve"
    assert trajectory.loc[2022] == population
    np.testing.assert_allclose(
        trajectory / population, (brazil / brazil.loc[2022]).loc[YEARS], rtol=1e-14
    )


def test_a_citys_own_un_points_replace_the_shape():
    points = {2000: 1_000_000, 2010: 1_100_000, 2020: 1_210_000}  # +10% a decade

    trajectory, source = city_population_trajectory(
        "BRA", 2015, 2_000_000.0, 1.05, 1.05, city_points=points
    )

    assert source == city_params.UN_CITY_TRAJECTORY == "UN city trajectory"
    assert trajectory.loc[2015] == pytest.approx(2_000_000.0, rel=1e-14)
    decade = 1.1
    assert trajectory.loc[2025] / trajectory.loc[2015] == pytest.approx(decade, rel=1e-12)
    # Beyond the last point it keeps the last pair's rate, and before the first the first's.
    assert trajectory.loc[2050] / trajectory.loc[2020] == pytest.approx(decade**3, rel=1e-12)
    assert trajectory.loc[1970] / trajectory.loc[2000] == pytest.approx(decade**-3, rel=1e-12)


def test_one_un_point_is_not_a_trajectory():
    with pytest.raises(ValueError, match="at least two"):
        city_population_trajectory("BRA", 2015, 2e6, 1.05, 1.05, city_points={2020: 1e6, 2025: np.nan})


def test_a_country_trajectory_is_its_un_population_in_people():
    trajectory, source = country_population_trajectory("USA")

    assert source == "country curve"
    assert list(trajectory.index) == YEARS
    assert trajectory.loc[2023] == 342_475_098.0
    # Kosovo is in the UN's table under XKX.
    assert country_population_trajectory("XKX")[1] == "country curve"


def test_a_small_countrys_trajectory_keeps_the_uns_fractions():
    """Svalbard and Jan Mayen's row carries interpolated fractions of a thousand
    (2,591.78 people in 2022). Rounded to whole people its curve moved by 0.03%."""
    trajectory, _ = country_population_trajectory("SJM")
    svalbard = country_population_series("SJM")

    np.testing.assert_allclose(
        (trajectory / trajectory.loc[2022]).to_numpy(),
        (svalbard / svalbard.loc[2022]).loc[YEARS].to_numpy(),
        rtol=1e-14,
    )


def test_a_country_without_a_un_population_grows_as_a_custom_location_there_does():
    trajectory, source = country_population_trajectory("ATA")
    city = City("custom")
    city.dst_baseline_blank("Antarctica", 100_000, 800.0, -20.0)
    waste = city.baseline_parameters.waste_generated_df.sum(axis=1)

    assert source == city_params.COUNTRY_FALLBACK_RATES == "country fallback rates"
    assert trajectory.loc[city_params.CUSTOM_LOCATION_POPULATION_YEAR] == 1.0
    np.testing.assert_allclose(trajectory.loc[YEARS], (waste / waste.loc[2022]).loc[YEARS], rtol=1e-13)


def _load(population_series=None):
    """São Paulo's published row, as WasteMAP's city DST loads it."""
    city = City("São Paulo")
    city.load_csv_new(
        _cities_table_row("São Paulo", **CITIES["São Paulo"]),
        dst=True,
        population_series=population_series,
    )
    city.baseline_parameters.sites_method = False
    city._calculate_divs()
    return city.baseline_parameters


def test_load_csv_new_grows_a_city_along_the_trajectory_it_is_handed():
    inputs = CITIES["São Paulo"]
    stored, _ = city_population_trajectory(
        "BRA", 2022, inputs["population"], inputs["historic"] / 100 + 1, inputs["future"] / 100 + 1
    )
    flat = pd.Series(inputs["population"], index=YEARS)

    worked_out, along_stored, along_flat = _load(), _load(stored), _load(flat)

    waste = lambda p: p.waste_generated_df.sum(axis=1)  # noqa: E731
    np.testing.assert_allclose(waste(along_stored), waste(worked_out), rtol=1e-14)
    assert waste(along_flat).loc[2050] == pytest.approx(waste(along_flat).loc[2022], rel=1e-12)
    assert waste(along_flat).loc[2050] != pytest.approx(waste(worked_out).loc[2050], rel=1e-3)
    # The two rates a person reads are still _city_growth's.
    assert (along_flat.growth_rate_historic, along_flat.growth_rate_future) == (
        worked_out.growth_rate_historic,
        worked_out.growth_rate_future,
    )


def _custom_location(country, population_series=None):
    """A Custom Location, as WasteMAP's city DST builds one."""
    city = City("custom")
    city.dst_baseline_blank(country, 100_000, 800.0, 15.0, population_series=population_series)
    return city


def test_a_custom_location_grows_along_the_country_trajectory_it_is_handed():
    stored, _ = country_population_trajectory("USA")
    flat = pd.Series(331_000_000.0, index=YEARS)

    worked_out, along_stored, along_flat = (
        _custom_location("United States", series).baseline_parameters
        for series in (None, stored, flat)
    )

    waste = lambda p: p.waste_generated_df.sum(axis=1)  # noqa: E731
    np.testing.assert_allclose(waste(along_stored), waste(worked_out), rtol=1e-14)
    np.testing.assert_allclose(
        (waste(along_stored) / waste(along_stored).loc[2022]).loc[YEARS],
        stored / stored.loc[city_params.CUSTOM_LOCATION_POPULATION_YEAR],
        rtol=1e-14,
    )
    # Only the shape counts: the waste is the caller's population's, in every case.
    assert waste(along_flat).loc[2022] == waste(worked_out).loc[2022]
    assert waste(along_flat).loc[2050] == pytest.approx(waste(along_flat).loc[2022], rel=1e-12)
    assert waste(along_flat).loc[2050] != pytest.approx(waste(worked_out).loc[2050], rel=1e-3)
    # The two rates a person reads are still the country's.
    assert (along_flat.growth_rate_historic, along_flat.growth_rate_future) == (
        worked_out.growth_rate_historic,
        worked_out.growth_rate_future,
    )


def test_a_custom_location_with_its_fallback_trajectory_is_unchanged():
    stored, source = country_population_trajectory("ATA")
    assert source == city_params.COUNTRY_FALLBACK_RATES

    worked_out = _custom_location("Antarctica").baseline_parameters
    along_stored = _custom_location("Antarctica", stored).baseline_parameters

    pd.testing.assert_frame_equal(along_stored.waste_generated_df, worked_out.waste_generated_df)
    assert (along_stored.growth_rate_historic, along_stored.growth_rate_future) == (
        city_params.FALLBACK_GROWTH_RATE_HISTORIC,
        city_params.FALLBACK_GROWTH_RATE_FUTURE,
    )


def test_a_custom_location_scenario_grows_along_the_trajectory_it_is_handed():
    flat = pd.Series(331_000_000.0, index=YEARS)
    city = _custom_location("United States", flat)
    for landfill in city.baseline_parameters.landfills:
        landfill.estimate_emissions()

    divs = city_params.DiversionFractions(compost=0.1, anaerobic=0.0, combustion=0.0, recycling=0.1)
    city.implement_dst_changes_simple_v1_5(divs, False, False, 0.0, 0.0, 2030, 1, 0.0)
    waste = city.scenario_parameters[0].waste_generated_df.df.sum(axis=1)

    assert waste.loc[2050] == pytest.approx(waste.loc[2030], rel=1e-12)
