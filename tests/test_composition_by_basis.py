"""Generated and disposed country mixes are kept apart, each row with a source.

A generated mix describes all the municipal waste a country produces; a disposed mix
describes what reaches a landfill or dump. The two tables sit beside the IPCC table that
waste_composition_for() reads, and nothing reads them yet, so adding them must not
change any default.
"""

import pandas as pd
import pytest

import SWEET_python.defaults_2019 as d
from SWEET_python.constants import WASTE_TYPES

EU28 = {
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL",
    "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "GBR",
}


def test_reference_model_rows_are_complete_as_printed():
    # Each printed row has 16 shares that add to 100 within rounding (99.7-100.3 on p.125).
    assert set(d.ERM_2015_COMPOSITION) == EU28
    for iso3, shares in d.ERM_2015_COMPOSITION.items():
        assert len(shares) == len(d.ERM_2015_COLUMNS), iso3
        assert sum(shares) == pytest.approx(100, abs=0.35), iso3
    assert set(d.ERM_TO_WASTE_TYPE) == set(d.ERM_2015_COLUMNS)
    assert set(d.ERM_TO_WASTE_TYPE.values()) <= set(WASTE_TYPES)


def test_generated_table_maps_the_reference_model_onto_the_ten_types():
    g = d.WASTE_FRACTIONS_GENERATED_COUNTRY
    assert list(g.columns) == list(WASTE_TYPES)
    assert set(g.index) == EU28 | set(d.GENERATED_ROWS_FROM_IPCC_TABLE)
    # Austria as printed: food 18.2, garden 18.6, no other biowaste.
    assert g.loc["AUT", "food"] == pytest.approx(18.2)
    assert g.loc["AUT", "green"] == pytest.approx(18.6)
    # Other biowaste goes to green: the Netherlands' 19.4 garden + 11.4 other biowaste.
    assert g.loc["NLD", "green"] == pytest.approx(30.8)
    # WEEE, rubble, batteries, hazardous, fines, inerts and other all go to other.
    assert g.loc["GBR", "other"] == pytest.approx(2.4 + 2.1 + 0.1 + 0.6 + 1.3 + 0.0 + 11.5)
    assert (g.loc[sorted(EU28), "rubber"] == 0).all()  # the Reference Model has no rubber column
    pd.testing.assert_series_equal(
        g.loc[sorted(EU28)].sum(axis=1),
        pd.Series({k: sum(v) for k, v in d.ERM_2015_COMPOSITION.items()}).sort_index(),
        check_names=False,
    )


@pytest.mark.parametrize(
    "table, rows",
    [
        ("WASTE_FRACTIONS_GENERATED_COUNTRY", "GENERATED_ROWS_FROM_IPCC_TABLE"),
        ("WASTE_FRACTIONS_DISPOSED_COUNTRY", "DISPOSED_ROWS_FROM_IPCC_TABLE"),
    ],
)
def test_rows_from_the_ipcc_table_are_read_not_copied(table, rows):
    for iso3 in getattr(d, rows):
        pd.testing.assert_series_equal(
            getattr(d, table).loc[iso3, list(WASTE_TYPES)],
            d.WASTE_FRACTIONS_COUNTRY.loc[iso3, list(WASTE_TYPES)],
            check_names=False,
        )


def test_an_ipcc_row_is_filed_under_one_kind_at_most():
    # Inventory rows that their own inventory calls generated are not disposed mixes.
    assert not set(d.GENERATED_ROWS_FROM_IPCC_TABLE) & set(d.DISPOSED_ROWS_FROM_IPCC_TABLE)
    for iso3 in ("BGR", "MDA", "RUS", "SWE", "ZWE", "DEU", "NLD", "LVA"):
        assert iso3 not in d.DISPOSED_ROWS_FROM_IPCC_TABLE
    assert "USA" in d.DISPOSED_ROWS_FROM_IPCC_TABLE  # EPA 2014 landfilled, to 0.1 point


def test_disposed_rows_of_their_own_add_up_and_are_not_in_the_ipcc_set():
    for iso3, (shares, source) in d.WASTE_FRACTIONS_DISPOSED_OWN.items():
        assert iso3 not in d.DISPOSED_ROWS_FROM_IPCC_TABLE
        assert len(shares) == len(WASTE_TYPES)
        assert 98.5 <= sum(shares) <= 100.5, iso3  # Austria's printed column sums to 99
        assert source
    # Spain's inventory row, not the IPCC table's Spain row (a campus study).
    assert d.WASTE_FRACTIONS_DISPOSED_COUNTRY.loc["ESP", "food"] == pytest.approx(37.27)
    assert d.WASTE_FRACTIONS_COUNTRY.loc["ESP", "food"] == pytest.approx(56.2)


def test_every_row_has_a_source():
    assert set(d.WASTE_FRACTIONS_GENERATED_SOURCES) == set(d.WASTE_FRACTIONS_GENERATED_COUNTRY.index)
    assert set(d.WASTE_FRACTIONS_DISPOSED_SOURCES) == set(d.WASTE_FRACTIONS_DISPOSED_COUNTRY.index)
    for sources in (d.WASTE_FRACTIONS_GENERATED_SOURCES, d.WASTE_FRACTIONS_DISPOSED_SOURCES):
        assert all(isinstance(s, str) and s for s in sources.values())


def test_waste_composition_by_basis():
    for basis in ("generated", "disposed"):
        row = d.waste_composition_by_basis("GBR", basis)
        assert list(row.index) == list(WASTE_TYPES)
        assert row.sum() == pytest.approx(1)
    assert d.waste_composition_by_basis("USA", "generated") is None
    assert d.waste_composition_by_basis("AUT", "disposed")["food"] == pytest.approx(35.2 / 99)
    with pytest.raises(ValueError):
        d.waste_composition_by_basis("GBR", "landfilled")


@pytest.mark.parametrize(
    "iso3, food",
    [
        ("GBR", 0.213),  # the IPCC table's UK row, not the generated 16.8
        ("ESP", 0.562),  # the IPCC table's Spain row, not the inventory's 37.27
        ("ITA", 0.126),  # the 2019 printing's Italy row, not the 2023 reissue's 21.9
        ("AUT", 0.332),  # no country row: Western Europe
        ("IRL", 0.332),  # country row has no plastic: Western Europe
    ],
)
def test_the_default_every_path_reads_is_unchanged(iso3, food):
    got = d.waste_composition_for(iso3, d.REGION_LOOKUP_ISO3[iso3])
    assert got["food"] == pytest.approx(food)
    assert got.sum() == pytest.approx(1)
