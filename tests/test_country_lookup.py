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


def _prefill(monkeypatch, place, latlon):
    """The site tool's prefill for ``latlon``, where the geocoder answers ``place``.

    ``place`` is what geopy's ``reverse`` returns: an object with Nominatim's JSON as
    ``raw``, or None where Nominatim can't place the point.
    """
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

    return asyncio.run(
        City("latlon_lookup").sdst_prepopulate(
            "db", 5432, "user", "password", "wastemap", "disable",
            sites_list=pd.DataFrame(),
            latlon=latlon,
        )
    )


def test_the_site_tool_prefills_a_point_in_kosovo_as_kosovo(monkeypatch):
    """The prefill names the country with the geocoder's English name, "Kosovo"."""
    place = SimpleNamespace(raw={"address": {"country": "Kosovo"}})

    prefill = _prefill(monkeypatch, place, (42.66, 21.17))  # Pristina

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


# --- no country, and a name that isn't a country's ----------------------------
#
# pycountry's fuzzy search scores every country for an empty name, and scores a
# country for any of its subdivisions that match. So "" was the United Kingdom
# (the most subdivisions), "None" Italy (Frosinone) and "nan" Thailand (Nan
# province), and the caller got that country's model without a word.


@pytest.mark.parametrize("given", ["", " ", "\t\n", None, float("nan"), 0, b"GBR"])
def test_no_country_is_not_found(given):
    with pytest.raises(LookupError):
        city_params._iso3_for(given)


@pytest.mark.parametrize(
    "given",
    [
        "None",  # was Italy: its province Frosinone contains "none"
        "nan",  # was Thailand: Nan province
        "-",  # was France, whose subdivisions are full of hyphens
        "0",  # was Sweden
        "England",  # was right, but a province's name is not a country's
        "Kurdistan",  # was Iraq
    ],
)
def test_a_name_only_a_subdivision_matches_is_not_a_country(given):
    with pytest.raises(LookupError):
        city_params._iso3_for(given)


@pytest.mark.parametrize(
    "given, iso3",
    [
        (" Kosovo ", "XKX"),  # was Serbia: SWEET's table missed the padded name
        ("GBR\n", "GBR"),
        ("Brunei", "BRN"),  # whole words of "Brunei Darussalam"
        ("Vatican", "VAT"),  # of "Holy See (Vatican City State)"
        ("Sint Maarten", "SXM"),  # was the Netherlands, through its subdivision Sint Maarten
        ("UK", "GBR"),  # initials of "United Kingdom"
        ("UAE", "ARE"),
        ("826", "GBR"),  # ISO 3166-1 numeric, an exact match
    ],
)
def test_a_country_is_still_found_by_its_own_name(given, iso3):
    assert city_params._iso3_for(given) == iso3


def test_a_custom_location_with_no_country_is_not_the_united_kingdom():
    with pytest.raises(ValueError, match="Country '' not found"):
        City("*").dst_baseline_blank("", 2_000_000, 500.0, 25.0)


def test_a_custom_site_with_no_country_is_not_the_united_kingdom():
    with pytest.raises(ValueError, match="Country ' ' not found"):
        City("custom site").cityparams_obj_for_blank_site(
            country=" ",
            population=None,
            precipitation=500.0,
            temperature=10.0,
            waste_fractions=Variant(baseline=[0.1] * 10, scenario=[0.1] * 10),
            waste_mass_year=Variant[int](baseline=2025, scenario=2025),
        )


# --- the site tool's prefill ---------------------------------------------------


def test_the_prefill_for_a_point_at_sea_is_not_found(monkeypatch):
    """Nominatim can't place it, so geopy returns None. That raised AttributeError."""
    with pytest.raises(ValueError, match="No country at 0.0, 0.0"):
        _prefill(monkeypatch, None, (0.0, 0.0))


def test_the_prefill_for_a_point_in_no_country_is_not_found(monkeypatch):
    """Nominatim places McMurdo Station in Antarctica with no country. That raised AttributeError."""
    place = SimpleNamespace(raw={"address": {"road": "Scott Base Road"}})

    with pytest.raises(ValueError, match="No country at -77.85, 166.68"):
        _prefill(monkeypatch, place, (-77.85, 166.68))


@pytest.mark.parametrize(
    "country, code, iso3",
    [
        # Nominatim's English names, which SWEET's table and pycountry spell otherwise.
        # Each raised "Country '...' not found.", so the prefill failed.
        ("Turkey", "tr", "TUR"),
        ("Ivory Coast", "ci", "CIV"),
        ("Cape Verde", "cv", "CPV"),
        ("Democratic Republic of the Congo", "cd", "COD"),
        ("Congo-Brazzaville", "cg", "COG"),
        ("East Timor", "tl", "TLS"),
        ("Palestinian Territories", "ps", "PSE"),
        ("Sahrawi Arab Democratic Republic", "eh", "ESH"),
        ("Northern Cyprus", "cy", "CYP"),
        ("Somaliland", "so", "SOM"),
        ("South Ossetia", "ge", "GEO"),
        ("Abkhazia", "ge", "GEO"),  # was Georgia through its subdivision; now through the code
        # The name wins where SWEET has the country's own code and Nominatim the Netherlands'.
        ("Aruba", "nl", "ABW"),
        ("Curaçao", "nl", "CUW"),
        ("Sint Maarten", "nl", "SXM"),
    ],
)
def test_the_prefill_falls_back_to_the_geocoders_country_code(monkeypatch, country, code, iso3):
    place = SimpleNamespace(raw={"address": {"country": country, "country_code": code}})

    assert _prefill(monkeypatch, place, (0.0, 0.0))["iso3"] == iso3


def test_the_prefill_names_a_country_it_cannot_find(monkeypatch):
    place = SimpleNamespace(raw={"address": {"country": "Narnia"}})

    with pytest.raises(ValueError, match="Country 'Narnia' not found"):
        _prefill(monkeypatch, place, (0.0, 0.0))
