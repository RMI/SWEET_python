"""A city with no waste figure of its own generates waste at the per-capita default.

When a cities-table row has neither a generated nor a collected tonnage,
`City.load_andre_params` sizes the city's waste from the IPCC per-capita default:
the rate times the city's population, which is the population of the row's
population year. That mass already describes the population year. The loader
nonetheless labelled it 2019 and then ran the adjustment that moves a surveyed
tonnage from its survey year to the population year, so the default was moved a
second time and the city got default x P(population year) / P(2019).

In the June 2026 cities table, Kuwait City published 7.92 kg/person/day against
Kuwait's default of 3.05 t/yr (8.36 kg/day), and Eldoret (population year 2005)
got 57% of Kenya's rate. 39 cities take this path. A default now reaches the model
unchanged, and is published with the year it describes: the population year.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python import defaults_2019
from SWEET_python.city_params import City
from SWEET_python.population import country_population_series

NOT_REPORTED = [
    "msw_generated_metric_tons_per_year",
    "msw_collected_metric_tons_per_year",
    "msw_collected_year",
    "msw_generated_year",
    "composition_food_organic_waste_percent",
    "composition_yard_garden_green_waste_percent",
    "composition_wood_percent",
    "composition_paper_cardboard_percent",
    "composition_textiles_percent",
    "composition_plastic_percent",
    "composition_metal_percent",
    "composition_glass_percent",
    "composition_rubber_leather_percent",
    "composition_other_percent",
    "waste_treatment_compost_percent",
    "waste_treatment_anaerobic_digestion_percent",
    "waste_treatment_incineration_percent",
    "waste_treatment_advanced_thermal_treatment_percent",
    "waste_treatment_recycling_percent",
    "waste_treatment_sanitary_landfill_landfill_gas_system_percent",
    "waste_treatment_controlled_landfill_percent",
    "waste_treatment_landfill_unspecified_percent",
    "waste_treatment_open_dump_percent",
]

# (city, country, ISO3, population, population year, the city's own UN rates), as
# the June 2026 table published them. Kuwait City and Namangan are big enough to
# keep their own rates; Eldoret and Kadoma City grow with their country's population
# series. Uzbekistan has no country row, so Namangan takes Central Asia's rate.
# Kadoma City's population year is after 2019, so the old code moved it up, not down.
CITIES = [
    ("Kuwait City", "Kuwait", "KWT", 2_779_000, 2015, 1.0574, 1.0134),
    ("Eldoret", "Kenya", "KEN", 167_016, 2005, 1.0551, 1.0405),
    ("Kadoma City", "Zimbabwe", "ZWE", 116_300, 2022, 1.0334, 1.0241),
    ("Namangan", "Uzbekistan", "UZB", 521_000, 2015, 1.0237, 1.0134),
]
IDS = [city[0] for city in CITIES]


def _row(country, iso3, population, population_year, historic, future, **reported):
    """A cities-query row with nothing measured but the population, plus ``reported``."""
    row = pd.Series(
        {
            "population_data_source": "World Bank",
            "country": country,
            "iso": iso3,
            "population_year": float(population_year),
            "population_count": float(population),
            "historic_growth_rate": historic,
            "future_growth_rate": future,
            "latitude": 0.0,
            "longitude": 0.0,
            "mean_yearly_precip_2000_2021": 600.0,
            "mean_yearly_temp_2000_2021": 20.0,
            **{column: np.nan for column in NOT_REPORTED},
            **reported,
        }
    )
    # The cities query selects data_collection_year twice (composition, then
    # treatment), and the loader falls back to the first when no MSW year is given.
    return pd.concat([row, pd.Series([2015.0, 2015.0], index=["data_collection_year"] * 2)])


def _default_rate(country, iso3):
    """kg/person/day: the country's IPCC row, else its region's."""
    if iso3 in defaults_2019.msw_per_capita_country:
        return defaults_2019.msw_per_capita_country[iso3]
    return defaults_2019.msw_per_capita_defaults[defaults_2019.region_lookup[country]]


def _loaded(name, row):
    city = City(name)
    city.load_andre_params(row)
    return city


def _emissions(city):
    """Total CH4 by year, the way the cities pipeline runs a loaded city."""
    for landfill in city.baseline_parameters.landfills:
        landfill.estimate_emissions()
    city.estimate_diversion_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=0, simple=True)
    return city.baseline_parameters.total_emissions["total"]


@pytest.mark.parametrize("name, country, iso3, population, year, historic, future", CITIES, ids=IDS)
def test_a_city_with_no_waste_figure_generates_the_default_rate(
    name, country, iso3, population, year, historic, future
):
    city = _loaded(name, _row(country, iso3, population, year, historic, future))
    params = city.baseline_parameters
    rate = _default_rate(country, iso3)

    assert city.waste_mass_defaults is True
    assert params.waste_per_capita == pytest.approx(rate, rel=1e-12)
    assert params.waste_mass.iloc[0] == pytest.approx(rate * population * 365 / 1000, rel=1e-12)


@pytest.mark.parametrize("name, country, iso3, population, year, historic, future", CITIES, ids=IDS)
def test_a_default_rate_is_published_with_the_population_year(
    name, country, iso3, population, year, historic, future
):
    params = _loaded(name, _row(country, iso3, population, year, historic, future)).baseline_parameters

    assert params.year_of_data_msw == params.year_of_data_pop == year


@pytest.mark.parametrize("name, country, iso3, population, year, historic, future", CITIES, ids=IDS)
def test_a_defaulted_city_is_modelled_as_if_it_reported_the_default_rate(
    name, country, iso3, population, year, historic, future
):
    tonnage = _default_rate(country, iso3) * population * 365 / 1000
    defaulted = _loaded(name, _row(country, iso3, population, year, historic, future))
    reported = _loaded(
        name,
        _row(
            country, iso3, population, year, historic, future,
            msw_generated_metric_tons_per_year=tonnage,
            msw_generated_year=float(year),
        ),
    )

    assert np.allclose(_emissions(defaulted), _emissions(reported), rtol=1e-12, atol=0)


def test_a_surveyed_tonnage_still_moves_from_its_survey_year_to_the_population_year():
    """The adjustment the default skipped is still right for a measured figure."""
    # A big city with its own rates: compounded from the 2012 survey to 2015.
    kuwait = _loaded(
        "Kuwait City",
        _row(
            "Kuwait", "KWT", 2_779_000, 2015, 1.0574, 1.0134,
            msw_generated_metric_tons_per_year=1_500_000.0,
            msw_generated_year=2012.0,
        ),
    ).baseline_parameters
    assert kuwait.waste_mass.iloc[0] == pytest.approx(1_500_000 * 1.0574**3, rel=1e-12)
    assert kuwait.year_of_data_msw == 2012

    # A small city: back from the 2010 survey to 2005 along Kenya's population.
    kenya = country_population_series("KEN")
    eldoret = _loaded(
        "Eldoret",
        _row(
            "Kenya", "KEN", 167_016, 2005, 1.0551, 1.0405,
            msw_generated_metric_tons_per_year=30_000.0,
            msw_generated_year=2010.0,
        ),
    ).baseline_parameters
    assert eldoret.waste_mass.iloc[0] == pytest.approx(
        30_000 * kenya.loc[2005] / kenya.loc[2010], rel=1e-12
    )
    assert eldoret.year_of_data_msw == 2010


def test_the_legacy_sinir_loader_generates_the_default_rate_too():
    """`import_basics` (reached only from `model_city_via_sites`) has the same rule."""
    basics = City("Kuwait City").import_basics(
        _row("Kuwait", "KWT", 2_779_000, 2015, 1.0574, 1.0134)
    )
    rate = defaults_2019.msw_per_capita_country["KWT"]

    assert basics["waste_mass_defaults"] is True
    assert basics["waste_per_capita"] == pytest.approx(rate, rel=1e-12)
    assert basics["waste_mass"] == pytest.approx(rate * 2_779_000 * 365 / 1000, rel=1e-12)
    assert basics["year_of_data_msw"] == basics["year_of_data_pop"] == 2015
