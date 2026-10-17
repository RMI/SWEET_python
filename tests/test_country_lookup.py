"""A Custom Location in Kosovo builds, and a site in Kosovo is modelled as Kosovo.

pycountry has no Kosovo. ``_iso3_for`` asked pycountry first, so Kosovo's code XKX,
which SWEET's defaults and WasteMAP's country list both use, raised LookupError:
``City.dst_baseline_blank`` turned it into "Country 'XKX' not found." and WasteMAP's
city tool answered 500. By name it was worse: the fuzzy search resolved "Kosovo" to
Serbia, whose province Kosovo-Metohija matches, so the site tool's prefill for a
point in Kosovo returned Serbia's defaults and code. SWEET's own country table now
comes first.
"""

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from SWEET_python import city_params, defaults_2019
from SWEET_python.city_params import City
from SWEET_python.class_defs import Variant
from SWEET_python.population import country_population_series

KOSOVO = country_population_series("XKX")


def test_a_custom_location_in_kosovo_builds():
    city = City("*")
    city.dst_baseline_blank("XKX", 2_000_000, 500.0, 25.0)

    assert city.baseline_parameters.total_emissions["total"].loc[2040] > 0
    assert city.baseline_parameters.population_series.equals(KOSOVO)


def test_a_custom_site_in_kosovo_builds():
    city = City("custom site")
    city.cityparams_obj_for_blank_site(
        country="XKX",
        population=None,
        precipitation=500.0,
        temperature=10.0,
        waste_fractions=Variant(baseline=[0.1] * 10, scenario=[0.1] * 10),
        waste_mass_year=Variant[int](baseline=2025, scenario=2025),
    )

    assert city.baseline_parameters.population_series.equals(KOSOVO)


def test_the_site_tool_prefills_a_point_in_kosovo_as_kosovo(monkeypatch):
    """The prefill names the country with the geocoder's English name, "Kosovo"."""
    place = SimpleNamespace(raw={"address": {"country": "Kosovo"}})
    monkeypatch.setattr(
        city_params, "create_geolocator", lambda: SimpleNamespace(reverse=lambda *a, **k: place)
    )

    def no_database(*args, **kwargs):
        raise OSError("no database in this test")

    async def no_async_database(*args, **kwargs):
        no_database()

    # The weather lookup falls back to its defaults when the database is unreachable.
    monkeypatch.setattr(city_params.socket, "create_connection", no_database)
    monkeypatch.setattr(city_params.asyncpg, "connect", no_async_database)

    prefill = asyncio.run(
        City("latlon_lookup").sdst_prepopulate(
            "db", 5432, "user", "password", "wastemap", "disable",
            sites_list=pd.DataFrame(),
            latlon=(42.66, 21.17),  # Pristina
        )
    )

    assert prefill["iso3"] == "XKX"


@pytest.mark.parametrize(
    "given, iso3",
    [
        ("XKX", "XKX"),
        ("xkx", "XKX"),
        ("Kosovo", "XKX"),  # was Serbia
        ("kosovo", "XKX"),
        ("Curacao", "CUW"),  # was the Netherlands, whose subdivision Curaçao matches
        ("Macau", "MAC"),  # these four raised LookupError
        ("Pitcairn Islands", "PCN"),
        ("Svalbard and Jan Mayen Islands", "SJM"),
        ("US Minor Islands", "UMI"),
        ("Curaçao", "CUW"),  # pycountry still answers for names SWEET spells differently
        ("United States", "USA"),
    ],
)
def test_sweets_own_country_table_comes_before_pycountry(given, iso3):
    assert city_params._iso3_for(given) == iso3


def _resolve(country):
    try:
        return city_params._iso3_for(country)
    except LookupError:
        return None


def test_every_country_sweet_has_defaults_for_is_found_by_its_code():
    wrong = {code: _resolve(code) for code in defaults_2019.region_lookup_iso3}
    assert {code: got for code, got in wrong.items() if got != code} == {}


def test_every_country_sweet_names_is_found_by_its_name():
    wrong = {name: _resolve(name) for name in defaults_2019.country_to_iso3}
    assert {
        name: got for name, got in wrong.items() if got != defaults_2019.country_to_iso3[name]
    } == {}


def test_a_country_that_does_not_exist_is_still_not_found():
    assert _resolve("ZZZ") is None
    assert _resolve("Narnia") is None
    with pytest.raises(ValueError, match="Country 'ZZZ' not found"):
        City("*").dst_baseline_blank("ZZZ", 2_000_000, 500.0, 25.0)
