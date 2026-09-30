"""WasteMAP's tools grow waste with the country's UN population, year by year.

Climate TRACE's pipeline hands SWEET the UN WPP2024 population for every year and
SWEET grows a site's waste by P(year) / P(anchor year). The WasteMAP tools compounded
two flat rates instead: the site tool's form value, the city tool's cities-table
columns, and for Custom Location one constant for every country. These tests hold the
WasteMAP paths to the population series, and hold the paths that still take a rate
(a typed site-tool rate, a big city's own UN city rates) to what they did before.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python import city_params
from SWEET_python.city_params import City
from SWEET_python.class_defs import DivMasses, DivsDF, DiversionFractions, Variant, WasteGeneratedDF, WasteMasses
from SWEET_python.population import average_growth_rates, country_population_series
from test_sdst_v1_5_preimplement_composition import (
    BASELINE_FRACTIONS,
    CLOSE_YEAR,
    COMPONENT_ORDER,
    MODEL_YEAR_MAX,
    OPEN_YEAR,
)

BRA = country_population_series("BRA")
USA = country_population_series("USA")
WINDOW = range(1970, 2051)


def _shape(frame_or_series, anchor):
    total = frame_or_series.sum(axis=1) if isinstance(frame_or_series, pd.DataFrame) else frame_or_series
    return (total / total.loc[anchor]).loc[list(WINDOW)].to_numpy()


def _population_shape(series, anchor):
    return (series / series.loc[anchor]).loc[list(WINDOW)].to_numpy()


# --- the table ---------------------------------------------------------------


def test_a_country_has_a_population_for_every_year_the_site_tool_models():
    assert list(USA.index) == list(range(1950, 2051))
    assert (USA > 0).all()
    assert USA.loc[2023] == pytest.approx(342475.098)  # WPP2024, thousands


def test_an_unknown_country_has_no_series():
    assert country_population_series("XXX") is None
    assert country_population_series(None) is None


def test_a_caller_cannot_edit_the_table():
    mine = country_population_series("USA")
    mine.loc[2023] = 0.0
    assert country_population_series("USA").loc[2023] == pytest.approx(342475.098)


def test_the_average_rates_reach_the_series_at_both_ends():
    historic, future = average_growth_rates(USA, 2022)
    assert USA.loc[2022] * historic ** (1970 - 2022) == pytest.approx(USA.loc[1970])
    assert USA.loc[2022] * future ** (2050 - 2022) == pytest.approx(USA.loc[2050])


# --- the growth constructors ---------------------------------------------------


def _masses(years):
    return pd.DataFrame({"food": 10.0, "paper_cardboard": 5.0}, index=years)


@pytest.mark.parametrize("builder", ["create_advanced", "create_advanced_2"])
def test_the_site_tool_constructors_follow_the_series_when_given_one(builder):
    years = np.arange(1970, 2051)
    args = (_masses(years), 1970, 2050, 2025) + (() if builder == "create_advanced" else (2025,))
    grown = getattr(WasteGeneratedDF, builder)(*args, 1.5, 1.5, population_series=USA).df
    assert np.allclose(_shape(grown, 2025), _population_shape(USA, 2025))


@pytest.mark.parametrize("builder", ["create_advanced", "create_advanced_2"])
def test_the_site_tool_constructors_compound_a_rate_as_before(builder):
    years = np.arange(1970, 2051)
    args = (_masses(years), 1970, 2050, 2025) + (() if builder == "create_advanced" else (2025,))
    grown = getattr(WasteGeneratedDF, builder)(*args, 1.02, 1.03).df
    expected = np.where(years < 2025, 1.02, 1.03) ** (years - 2025)
    assert np.array_equal(grown["food"].to_numpy(), 10.0 * expected)


def test_scenario_diversions_follow_the_series_when_given_one():
    zero = WasteMasses(**{w: 0.0 for w in COMPONENT_ORDER})
    compost = WasteMasses(**{**{w: 0.0 for w in COMPONENT_ORDER}, "food": 100.0})
    divs = DivMasses(compost=compost, anaerobic=zero, combustion=zero, recycling=zero)
    grown = DivsDF.create_simple(divs, divs, 1970, 2050, 2030, 2025, 1.5, 1.5, population_series=BRA)
    assert np.allclose(_shape(grown.compost["food"], 2025), _population_shape(BRA, 2025))


# --- the city tool -------------------------------------------------------------


def test_a_big_city_with_plausible_rates_keeps_its_own():
    series, historic, future = city_params._city_growth("NGA", 2020, 15_000_000, 1.0556, 1.036)
    assert series is None and (historic, future) == (1.0556, 1.036)


@pytest.mark.parametrize(
    "population, historic, future, why",
    [
        (781, 0.9144, 1.8303, "a town under 300k borrows the nearest big city's rate"),
        (1_800_000, 1.0022, 1.1687, "Manila City: its own population mixed with the agglomeration's"),
        (np.nan, 1.02, 1.01, "no population to tell"),
        (2_000_000, np.nan, 1.01, "no rate"),
    ],
)
def test_any_other_city_grows_with_its_country(population, historic, future, why):
    series, got_historic, got_future = city_params._city_growth("BRA", 2022, population, historic, future)
    assert series is not None, why
    assert (got_historic, got_future) == pytest.approx(average_growth_rates(BRA, 2022))


def test_custom_location_grows_with_its_country_not_one_constant():
    city = City("custom")
    city.dst_baseline_blank("United States", 100_000, 800.0, 15.0)
    baseline = city.baseline_parameters

    assert np.allclose(_shape(baseline.waste_generated_df, 2022), _population_shape(USA, 2022))
    # Was (4.3e9 / 751e6) ** (1/70), 2.5% a year, for every country.
    assert baseline.growth_rate_historic == pytest.approx(average_growth_rates(USA, 2022)[0])


def test_the_city_scenario_grows_as_its_baseline_does():
    city = City("custom")
    city.dst_baseline_blank("United States", 100_000, 800.0, 15.0)
    for landfill in city.baseline_parameters.landfills:
        landfill.estimate_emissions()

    divs = DiversionFractions(compost=0.1, anaerobic=0.0, combustion=0.0, recycling=0.1)
    city.implement_dst_changes_simple_v1_5(divs, False, False, 0.0, 0.0, 2030, 1, 0.0)
    scenario = city.scenario_parameters[0]

    after = list(range(2030, 2051))
    waste = scenario.waste_generated_df.df.sum(axis=1)
    compost = scenario.divs_df.compost.sum(axis=1)
    usa = (USA / USA.loc[2035]).loc[after].to_numpy()
    assert np.allclose((waste / waste.loc[2035]).loc[after].to_numpy(), usa)
    assert np.allclose((compost / compost.loc[2035]).loc[after].to_numpy(), usa)


# --- the site tool -------------------------------------------------------------


def _custom_site(growth_rate_override):
    years = pd.Index(range(OPEN_YEAR, MODEL_YEAR_MAX + 1))
    fractions = pd.DataFrame([BASELINE_FRACTIONS] * len(years), index=years, columns=COMPONENT_ORDER, dtype=float)
    year = Variant[int](baseline=2025, scenario=2025)
    city = City("custom site")
    city.cityparams_obj_for_blank_site(
        country="BRA",
        population=None,
        precipitation=500.0,
        temperature=10.0,
        waste_fractions=Variant(baseline=list(BASELINE_FRACTIONS), scenario=list(BASELINE_FRACTIONS)),
        waste_mass_year=year,
        growth_rate_override=growth_rate_override,
    )
    city.sdst_v1_5(
        precipitation=500.0,
        new_waste_fractions={"baseline": fractions, "scenario": fractions},
        new_landfill_types=Variant(baseline=[2], scenario=[2]),
        new_gas_efficiency=Variant(baseline=[0.0], scenario=[0.0]),
        new_landfill_open_close_dates=Variant(baseline=[(OPEN_YEAR, CLOSE_YEAR)], scenario=[(OPEN_YEAR, CLOSE_YEAR)]),
        scenario=1,
        landfill_split_timeline=Variant(baseline={y: [1.0] for y in years}, scenario={y: [1.0] for y in years}),
        new_landfill_latlons=None,
        new_landfill_areas=None,
        new_covertypes=None,
        new_coverthicknesses=None,
        waste_burning=Variant(baseline=0.0, scenario=0.0),
        new_landfill_flaring=Variant(baseline=[0.98], scenario=[0.98]),
        fancy_ox=None,
        new_waste_mass=Variant(baseline=10000.0, scenario=10000.0),
        waste_mass_year=year,
        depths=Variant(baseline=[3.0], scenario=[3.0]),
        ks_overrides=Variant(baseline=0.2, scenario=0.2),
        biocover={"baseline": 0.0, "scenario": 0.0},
        oxidation_override=None,
        baseline_data=None,
        implement_year=2030,
        growth_rate_override=growth_rate_override,
        country_growth_defaults=[1.0, 1.0],
    )
    deposits = city.baseline_parameters.landfills[0].waste_mass_df
    return deposits.sum(axis=1)


def test_a_custom_site_with_no_rate_grows_with_its_country():
    deposits = _custom_site(None)
    open_years = list(range(OPEN_YEAR, CLOSE_YEAR))
    assert np.allclose(
        (deposits / deposits.loc[2025]).loc[open_years].to_numpy(),
        (BRA / BRA.loc[2025]).loc[open_years].to_numpy(),
    )


def test_a_typed_rate_is_compounded_as_before():
    deposits = _custom_site(0.02)
    assert deposits.loc[2035] == pytest.approx(10000.0 * 1.02**10)


# --- review fixes --------------------------------------------------------------


@pytest.mark.parametrize(
    "given, iso3",
    [("MUS", "MUS"), ("Mauritius", "MUS"), ("Niger", "NER"), ("Nigeria", "NGA"), ("United States", "USA")],
)
def test_a_country_is_found_by_its_code_before_the_fuzzy_search(given, iso3):
    """search_fuzzy('MUS') is Turkey (province Mus), and 'Niger' was Nigeria."""
    assert city_params._iso3_for(given) == iso3


def test_mauritius_grows_with_mauritius():
    city = City("custom")
    city.dst_baseline_blank("MUS", 100_000, 1500.0, 24.0)
    assert city.baseline_parameters.population_series.equals(country_population_series("MUS"))


def test_republishing_a_citys_own_rates_keeps_its_growth_the_same():
    """The map build writes a city's rates back into the cities table and the city tool
    re-runs _city_growth on them. Republishing the rates the loader READ keeps the
    choice; republishing the averages it applied flipped Manila City to compounding."""
    own = (1.0022, 1.1687)  # Manila City: its own population mixed with the agglomeration's
    series, historic, future = city_params._city_growth("PHL", 2020, 1_800_000, *own)
    assert series is not None

    reread, *_ = city_params._city_growth("PHL", 2020, 1_800_000, *own)
    assert reread is not None
    flipped, *_ = city_params._city_growth("PHL", 2020, 1_800_000, historic, future)
    assert flipped is None  # why the applied averages must not be republished
