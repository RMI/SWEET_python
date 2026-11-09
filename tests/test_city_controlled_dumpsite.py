"""The generic city model sends a city's controlled-dumpsite share to a controlled dumpsite.

The model had three disposal buckets: a landfill with gas capture, one without, and a
dumpsite. A controlled-dumpsite share, such as SINIR's "aterro controlado", which the
cities table carries as ``waste_treatment_controlled_dumpsite_percent``, had nowhere to
go. The loader never read it, and the split was rescaled so the other shares filled its
place, so a city sending 40% to controlled dumps and 60% to open dumps was modelled as
100% open dump.

The controlled dumpsite takes the values SWEET already gives one at site level: the
uncategorised MCF (0.6, the same as an open dump's) and an oxidation factor of 0.05
without gas capture, or 0.10 with it. It is the last of the city's landfills, so the
first three keep the places the DST levers index them by. A city with no such share
sends it nothing and is modelled as before.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from SWEET_python import mcf as mcf_defaults
from SWEET_python.city_params import (
    CONTROLLED_DUMPSITE_LANDFILL,
    CONTROLLED_DUMPSITE_PERCENT,
    PUBLISHED_CONTROLLED_DUMPSITE_PERCENT,
    City,
    DiversionFractions,
    _controlled_dumpsite_percent,
    _move_to_existing_gas_capture,
    _published_controlled_dumpsite_share,
)
from SWEET_python.class_defs import SplitFractions
from SWEET_python.constants import DEFAULT_FLARE_EFFICIENCY

_COUNTRY, _POP, _PRECIP, _TEMP = "Brazil", 200_000, 1_200.0, 24.0
_IMPLEMENT_YEAR, _SCENARIO = 2026, 1
_BEFORE = slice(None, _IMPLEMENT_YEAR)
_AFTER = slice(_IMPLEMENT_YEAR + 1, None)
_NO_DIVERSION = DiversionFractions(compost=0.0, anaerobic=0.0, combustion=0.0, recycling=0.0)

TREATMENTS = [
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
COMPOSITION = [
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
]


# --- the cities-table row ----------------------------------------------------------


def _brazilian_city(**reported):
    """A cities-query row for a Brazilian city with no treatment data, plus ``reported``.

    Without ``waste_treatment_controlled_dumpsite_percent`` it is a row from a query
    that does not select the column, as WasteMAP's did before it read it.
    """
    row = pd.Series(
        {
            "population_data_source": "SINIR",
            "country": "Brazil",
            "iso": "BRA",
            "population_year": 2019.0,
            "population_count": 200_000.0,
            "historic_growth_rate": 1.01,
            "future_growth_rate": 1.005,
            "latitude": -7.2,
            "longitude": -39.3,
            "mean_yearly_precip_2000_2021": _PRECIP,
            "mean_yearly_temp_2000_2021": _TEMP,
            "msw_generated_metric_tons_per_year": 60_000.0,
            "msw_generated_year": 2019.0,
            "msw_collected_metric_tons_per_year": np.nan,
            "msw_collected_year": np.nan,
            **{column: np.nan for column in TREATMENTS + COMPOSITION},
            **reported,
        }
    )
    return pd.concat([row, pd.Series([2019.0, 2019.0], index=["data_collection_year"] * 2)])


def _loaded(row) -> City:
    city = City("Juazeiro do Norte")
    city.load_andre_params(row)
    return city


def _diverts_nothing(**reported):
    """The row with every diversion reported as 0, so the split is used as given."""
    zeros = {column: 0.0 for column in TREATMENTS[:5]}
    return _brazilian_city(**zeros, **reported)


def test_a_reported_controlled_dumpsite_share_reaches_the_split_and_its_landfill():
    """Before, this city came out 100% open dump."""
    city = _loaded(_diverts_nothing(waste_treatment_open_dump_percent=60.0, **{CONTROLLED_DUMPSITE_PERCENT: 40.0}))

    split = city.baseline_parameters.split_fractions
    assert (split.dumpsite, split.controlled_dumpsite) == pytest.approx((0.6, 0.4), abs=1e-12)
    landfill = city.baseline_parameters.landfills[CONTROLLED_DUMPSITE_LANDFILL]
    assert landfill.site_type == "controlled_dumpsite"
    assert landfill.fraction_of_waste == pytest.approx(0.4, abs=1e-12)


def test_a_controlled_dumpsite_share_counts_toward_the_reported_disposal():
    """A city reporting no diversion and disposal adding up to 100% diverts nothing.
    Before, the controlled dumpsite share went uncounted, the disposal came to 60%,
    and the city took its country's default diversion as if its data were partial."""
    city = _loaded(_brazilian_city(waste_treatment_open_dump_percent=60.0, **{CONTROLLED_DUMPSITE_PERCENT: 40.0}))

    assert city.diversion_defaults is False
    assert city.baseline_parameters.split_fractions.controlled_dumpsite == pytest.approx(0.4, abs=1e-12)


@pytest.mark.parametrize(
    "reported",
    [{}, {CONTROLLED_DUMPSITE_PERCENT: np.nan}],
    ids=["no-column", "blank"],
)
def test_a_row_without_the_share_sends_the_controlled_dumpsite_nothing(reported):
    city = _loaded(_diverts_nothing(waste_treatment_open_dump_percent=100.0, **reported))

    split = city.baseline_parameters.split_fractions
    assert (split.dumpsite, split.controlled_dumpsite) == (1.0, 0.0)


@pytest.mark.parametrize("blank", [None, np.nan, "", "  "], ids=["none", "nan", "empty", "spaces"])
def test_the_row_share_reads_a_blank_as_nan(blank):
    assert np.isnan(_controlled_dumpsite_percent(pd.Series({CONTROLLED_DUMPSITE_PERCENT: blank})))


def test_the_row_share_reads_a_missing_column_as_nan_and_a_value_as_itself():
    assert np.isnan(_controlled_dumpsite_percent(pd.Series({"other": 1.0})))
    assert _controlled_dumpsite_percent(pd.Series({CONTROLLED_DUMPSITE_PERCENT: 40})) == 40.0


# --- the published cities table ----------------------------------------------------


@pytest.mark.parametrize(
    "columns, expected",
    [
        ({}, 0.0),
        ({PUBLISHED_CONTROLLED_DUMPSITE_PERCENT: [np.nan]}, 0.0),
        ({PUBLISHED_CONTROLLED_DUMPSITE_PERCENT: [""]}, 0.0),
        ({PUBLISHED_CONTROLLED_DUMPSITE_PERCENT: [25.0]}, 0.25),
    ],
    ids=["published-before-the-column", "blank", "empty-string", "reported"],
)
def test_the_published_share_is_a_fraction_and_zero_when_absent(columns, expected):
    city_data = pd.DataFrame({"City": ["Juazeiro do Norte"], **columns})
    assert _published_controlled_dumpsite_share(city_data) == expected


# --- the model ------------------------------------------------------------------------


def _city_with_split(**split) -> City:
    """A city whose baseline sends its disposal to the generic split given, as
    WasteMAP's ``load_city`` builds one from the cities table."""
    city = City("controlled_dumpsite_regression")
    city.dst_baseline_blank(_COUNTRY, _POP, _PRECIP, _TEMP)
    baseline = city.baseline_parameters
    baseline.split_fractions = SplitFractions(**split)
    city._calculate_divs()
    baseline.repopulate_attr_dicts()
    for landfill in baseline.landfills:
        landfill.estimate_emissions()
    city.estimate_diversion_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=0)
    return city


@pytest.fixture(scope="module")
def half_controlled() -> City:
    """Half its disposal to an open dump, half to a controlled one."""
    return _city_with_split(landfill_w_capture=0.0, landfill_wo_capture=0.0, dumpsite=0.5, controlled_dumpsite=0.5)


def _implement(city: City, add_gas=0, move_gas=0, new_gas_pct=0.0, existing_gas_pct=0.0):
    """Run the city DST on a copy of ``city`` with no diversion."""
    city = copy.deepcopy(city)
    city.implement_dst_changes_simple_v1_5(
        _NO_DIVERSION, add_gas, move_gas, new_gas_pct, existing_gas_pct, _IMPLEMENT_YEAR, _SCENARIO
    )
    return city.scenario_parameters[_SCENARIO - 1]


def test_the_controlled_dumpsite_is_last_and_carries_a_controlled_dumps_values(half_controlled):
    landfills = half_controlled.baseline_parameters.landfills
    assert [lf.site_type for lf in landfills] == ["landfill", "landfill", "dumpsite", "controlled_dumpsite"]

    controlled = landfills[CONTROLLED_DUMPSITE_LANDFILL]
    assert (controlled.mcf == mcf_defaults.MCF_UNCATEGORISED).all()
    assert controlled.oxidation_factor == 0.05
    assert controlled.gas_capture_efficiency == 0.0


def test_a_controlled_dump_emits_what_an_open_dump_does_less_its_oxidation(half_controlled):
    """Same MCF, same waste: only the 5% of its methane the cover oxidises differs."""
    dumpsite, controlled = half_controlled.baseline_parameters.landfills[2:]

    pd.testing.assert_series_equal(
        controlled.emissions["total"], dumpsite.emissions["total"] * 0.95, rtol=1e-12, check_names=False
    )


def test_a_city_with_no_controlled_dumpsite_share_gets_nothing_from_it():
    """Its other three landfills are built as before, so its emissions are too."""
    city = _city_with_split(landfill_w_capture=0.2, landfill_wo_capture=0.5, dumpsite=0.3)

    controlled = city.baseline_parameters.landfills[CONTROLLED_DUMPSITE_LANDFILL]
    assert (controlled.emissions["total"] == 0).all()
    assert controlled not in city.baseline_parameters.non_zero_landfills


def test_add_gas_gives_the_controlled_dumpsite_a_controlled_dumps_capture(half_controlled):
    """45% capture and 0.10 oxidation after the implementation year, nothing before.
    It stays a controlled dumpsite, so its MCF does not move."""
    no_change = _implement(half_controlled).landfills[CONTROLLED_DUMPSITE_LANDFILL]
    added = _implement(half_controlled, add_gas=1).landfills[CONTROLLED_DUMPSITE_LANDFILL]

    assert (added.gas_capture_efficiency.loc[_BEFORE] == 0.0).all()
    assert (added.gas_capture_efficiency.loc[_AFTER] == 0.45).all()
    assert (added.oxidation_factor.loc[_BEFORE] == 0.05).all()
    assert (added.oxidation_factor.loc[_AFTER] == 0.10).all()
    assert (added.mcf == mcf_defaults.MCF_UNCATEGORISED).all()

    total, before = added.emissions["total"], no_change.emissions["total"]
    pd.testing.assert_series_equal(total.loc[_BEFORE], before.loc[_BEFORE])
    factor = ((1 - 0.45) * (1 - 0.10) + 0.45 * (1 - DEFAULT_FLARE_EFFICIENCY)) / (1 - 0.05)
    pd.testing.assert_series_equal(total.loc[_AFTER], before.loc[_AFTER] * factor, rtol=1e-12)


def test_moving_to_existing_gas_capture_scales_the_controlled_dumpsite_too():
    """It is a site without capture, so it gives up waste in proportion with the others."""
    split = SplitFractions(landfill_w_capture=0.2, landfill_wo_capture=0.4, dumpsite=0.2, controlled_dumpsite=0.2)

    moved = _move_to_existing_gas_capture(split, 0.6)

    assert (moved.landfill_wo_capture, moved.dumpsite, moved.controlled_dumpsite) == pytest.approx(
        (0.2, 0.1, 0.1), abs=1e-12
    )
    assert sum(moved.model_dump().values()) == pytest.approx(1.0, abs=1e-12)


def test_moved_waste_reaches_the_controlled_dumpsite():
    city = _city_with_split(landfill_w_capture=0.2, landfill_wo_capture=0.4, dumpsite=0.2, controlled_dumpsite=0.2)

    scenario = _implement(city, move_gas=1, existing_gas_pct=0.6)

    fractions = [lf.fraction_of_waste for lf in scenario.landfills]
    assert fractions == pytest.approx([0.6, 0.2, 0.1, 0.1], abs=1e-12)


def test_a_new_landfill_with_capture_comes_after_the_controlled_dumpsite(half_controlled):
    """Each existing share gives up its part of the new landfill's 20%, and the new
    landfill, last, takes no waste before it opens."""
    scenario = _implement(half_controlled, new_gas_pct=0.2)

    fractions = [lf.fraction_of_waste for lf in scenario.landfills]
    assert fractions == pytest.approx([0.0, 0.0, 0.4, 0.4, 0.2], abs=1e-12)
    new_landfill = scenario.landfills[-1]
    assert (new_landfill.waste_mass_df.loc[: _IMPLEMENT_YEAR - 1] == 0).all().all()
    assert (new_landfill.waste_mass_df.loc[_IMPLEMENT_YEAR:].sum(axis=1) > 0).all()
    controlled = scenario.landfills[CONTROLLED_DUMPSITE_LANDFILL]
    assert (controlled.waste_mass_df.loc[: _IMPLEMENT_YEAR - 1].sum(axis=1) > 0).all()

