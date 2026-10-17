"""Regression tests for the City DST's "move waste to existing gas capture" lever
(``move_gas`` and ``existing_gas_pct`` in ``City.implement_dst_changes_simple_v1_5``).

Bug (fixed): for a city on the generic landfill split, the lever rewrote
``split_fractions`` but never passed the new split to the landfills. Each
landfill's waste comes from its own ``fraction_of_waste``, which only the
"new landfill with gas capture" block updated, and the frontend never sends that
lever together with this one. So the slider changed nothing: Lahore (34% of its
waste to gas capture) moved to 90% returned exactly the no-change scenario, as did
every other city that has a live slider.

The arithmetic was wrong too. It set the capture share and then multiplied the
shares without capture by ``existing / original``, scaling them up: Lahore's
landfill without capture came out at 171% of the city's waste, Mexico City's at
3,848%. With no capture to start from, it zeroed them. Combined with a new
landfill with gas capture, the inflated split failed its sum check and raised a
``TypeError`` (``CustomError`` was built with one argument), which the API
returned as a 500.

The fix gives the capture share ``existing_gas_pct`` and the landfill and dumpsite
without capture the rest, in their original proportions, then applies that split
to the landfills.

These tests build a city with gas capture from country defaults
(``dst_baseline_blank``, no database), give it a split, and run the real
``implement_dst_changes_simple_v1_5``.
"""

import copy
import math

import pandas as pd
import pytest

from SWEET_python.city_params import City, CustomError, DiversionFractions
from SWEET_python.class_defs import SplitFractions

_COUNTRY, _POP, _PRECIP, _TEMP = "Mexico", 2_000_000, 600.0, 17.0
_IMPLEMENT_YEAR, _SCENARIO = 2026, 1
# 30% of disposal to a landfill with gas capture, 50% to one without, 20% to a
# dumpsite, so the move has two sites without capture to scale.
_WITH_CAPTURE, _WITHOUT_CAPTURE, _DUMPSITE = 0.3, 0.5, 0.2


def _city_with_split(with_capture, without_capture, dumpsite) -> City:
    """A city whose baseline sends its disposal to the generic split given.

    ``dst_baseline_blank`` builds every Custom Location with no gas capture, so the
    split is replaced and the landfills rebuilt from it, which is what WasteMAP's
    ``load_city`` does after reading a city's split from the cities table.
    """
    city = City("move_gas_regression")
    city.dst_baseline_blank(_COUNTRY, _POP, _PRECIP, _TEMP)
    baseline = city.baseline_parameters
    baseline.split_fractions = SplitFractions(
        landfill_w_capture=with_capture,
        landfill_wo_capture=without_capture,
        dumpsite=dumpsite,
    )
    city._calculate_divs()
    baseline.repopulate_attr_dicts()
    for landfill in baseline.landfills:
        landfill.estimate_emissions()
    city.estimate_diversion_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=0)
    return city


@pytest.fixture(scope="module")
def city_with_capture() -> City:
    return _city_with_split(_WITH_CAPTURE, _WITHOUT_CAPTURE, _DUMPSITE)


def _implement(city: City, move_gas: int, existing_gas_pct: float, new_gas_pct: float = 0.0):
    """Run the city DST on a copy of ``city``, keeping its diversions."""
    city = copy.deepcopy(city)
    div = city.baseline_parameters.div_fractions.iloc[0]
    div_fractions = DiversionFractions(
        compost=float(div["compost"]),
        anaerobic=float(div["anaerobic"]),
        combustion=float(div["combustion"]),
        recycling=float(div["recycling"]),
    )
    city.implement_dst_changes_simple_v1_5(
        div_fractions,
        0,
        move_gas,
        new_gas_pct,
        existing_gas_pct,
        _IMPLEMENT_YEAR,
        _SCENARIO,
    )
    return city.scenario_parameters[_SCENARIO - 1]


def _total(scenario_parameters) -> pd.Series:
    return scenario_parameters.total_emissions["total"]


def _moved_split(existing_gas_pct):
    rest = 1 - existing_gas_pct
    without_capture_total = _WITHOUT_CAPTURE + _DUMPSITE
    return (
        existing_gas_pct,
        _WITHOUT_CAPTURE * rest / without_capture_total,
        _DUMPSITE * rest / without_capture_total,
    )


def test_moving_waste_to_gas_capture_lowers_emissions(city_with_capture):
    """Before the fix this returned the no-change scenario exactly."""
    no_change = _total(_implement(city_with_capture, 0, 0.0))
    moved = _total(_implement(city_with_capture, 1, 0.9))

    pd.testing.assert_series_equal(
        moved.loc[: _IMPLEMENT_YEAR - 1], no_change.loc[: _IMPLEMENT_YEAR - 1]
    )
    after = slice(_IMPLEMENT_YEAR + 1, None)
    assert (moved.loc[after] < no_change.loc[after]).all(), (
        "moving waste to gas capture did not lower emissions after "
        f"{_IMPLEMENT_YEAR}: 2040 {moved.loc[2040]:.1f} vs {no_change.loc[2040]:.1f}"
    )


def test_moving_waste_off_gas_capture_raises_emissions(city_with_capture):
    no_change = _total(_implement(city_with_capture, 0, 0.0))
    moved = _total(_implement(city_with_capture, 1, 0.0))

    after = slice(_IMPLEMENT_YEAR + 1, None)
    assert (moved.loc[after] > no_change.loc[after]).all(), (
        "moving all waste off gas capture did not raise emissions after "
        f"{_IMPLEMENT_YEAR}: 2040 {moved.loc[2040]:.1f} vs {no_change.loc[2040]:.1f}"
    )


@pytest.mark.parametrize("existing_gas_pct", [0.0, 0.6, 0.9, 1.0])
def test_landfills_receive_the_moved_split(city_with_capture, existing_gas_pct):
    """The landfill and dumpsite without capture share what capture does not
    take, in their original 5:2 proportion, and the split still sums to 1."""
    scenario = _implement(city_with_capture, 1, existing_gas_pct)
    expected = _moved_split(existing_gas_pct)

    split = scenario.split_fractions
    assert (split.landfill_w_capture, split.landfill_wo_capture, split.dumpsite) == pytest.approx(
        expected, abs=1e-12
    )
    fractions = [landfill.fraction_of_waste for landfill in scenario.landfills]
    assert fractions == pytest.approx(list(expected), abs=1e-12)
    assert sum(fractions) == pytest.approx(1.0, abs=1e-12)


def test_moved_waste_is_reallocated_not_created_or_lost(city_with_capture):
    """From the implementation year each landfill's waste is the no-change
    scenario's, scaled by its new share over its old one, and the total
    landfilled each year is unchanged. Before it, nothing moves."""
    no_change = _implement(city_with_capture, 0, 0.0)
    moved = _implement(city_with_capture, 1, 0.9)

    after = slice(_IMPLEMENT_YEAR, None)
    before = slice(None, _IMPLEMENT_YEAR - 1)
    old_split = (_WITH_CAPTURE, _WITHOUT_CAPTURE, _DUMPSITE)
    for old_share, new_share, base_lf, moved_lf in zip(
        old_split, _moved_split(0.9), no_change.landfills, moved.landfills
    ):
        pd.testing.assert_frame_equal(
            moved_lf.waste_mass_df.loc[before], base_lf.waste_mass_df.loc[before]
        )
        pd.testing.assert_frame_equal(
            moved_lf.waste_mass_df.loc[after],
            base_lf.waste_mass_df.loc[after] * (new_share / old_share),
            rtol=1e-12,
        )

    def landfilled(scenario):
        return sum(lf.waste_mass_df.sum(axis=1) for lf in scenario.landfills)

    pd.testing.assert_series_equal(landfilled(moved), landfilled(no_change), rtol=1e-12)


def test_the_city_own_share_is_the_no_change_scenario(city_with_capture):
    no_change = _total(_implement(city_with_capture, 0, 0.0))
    unmoved = _total(_implement(city_with_capture, 1, _WITH_CAPTURE))

    pd.testing.assert_series_equal(unmoved, no_change, rtol=1e-12)


def test_moving_and_adding_a_new_landfill_with_gas_capture(city_with_capture):
    """The new landfill takes its share off the top of the moved split. The UI
    never sends both levers, but the API accepts them; before the fix the
    inflated split failed the sum check with a TypeError (an API 500)."""
    scenario = _implement(city_with_capture, 1, 0.9, new_gas_pct=0.2)

    expected = [share * 0.8 for share in _moved_split(0.9)] + [0.2]
    fractions = [landfill.fraction_of_waste for landfill in scenario.landfills]
    assert fractions == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("existing_gas_pct", [-0.1, 1.1, math.nan])
def test_a_share_outside_zero_to_one_is_rejected(city_with_capture, existing_gas_pct):
    with pytest.raises(CustomError) as excinfo:
        _implement(city_with_capture, 1, existing_gas_pct)
    assert excinfo.value.code == "INVALID_PARAMETERS"


@pytest.mark.parametrize(
    "split",
    [
        (0.0, 0.7, 0.3),
        # No disposal at all, the split a Custom Location had in 31 countries
        # before RMI/SWEET_python#74.
        (0.0, 0.0, 0.0),
    ],
)
def test_a_city_without_gas_capture_has_no_landfill_to_move_waste_to(split):
    """The UI only offers the lever to a city with gas capture. Before the fix
    the API returned the no-change scenario, as if the move had been made. A
    share of 0 asks for no change, which is what it gets."""
    city = _city_with_split(*split)
    with pytest.raises(CustomError) as excinfo:
        _implement(city, 1, 0.5)
    assert excinfo.value.code == "INVALID_PARAMETERS"

    pd.testing.assert_series_equal(
        _total(_implement(city, 1, 0.0)), _total(_implement(city, 0, 0.0)), rtol=1e-12
    )


def test_a_city_all_to_gas_capture_has_no_site_to_move_waste_back_to():
    """41 cities in the June 2026 cities table send all their disposal to gas
    capture. The UI disables the slider for them and sends 100%."""
    city = _city_with_split(1.0, 0.0, 0.0)
    with pytest.raises(CustomError) as excinfo:
        _implement(city, 1, 0.5)
    assert excinfo.value.code == "INVALID_PARAMETERS"

    pd.testing.assert_series_equal(
        _total(_implement(city, 1, 1.0)), _total(_implement(city, 0, 0.0)), rtol=1e-12
    )


def test_lahore_moved_to_ninety_percent():
    """Lahore's split in the June 2026 cities table. Before the fix the split came
    out as 90% / 171% / 0% and never reached the landfills."""
    city = _city_with_split(0.34430946, 0.65569054, 0.0)
    scenario = _implement(city, 1, 0.9)

    split = scenario.split_fractions
    assert (split.landfill_w_capture, split.landfill_wo_capture, split.dumpsite) == pytest.approx(
        (0.9, 0.1, 0.0), abs=1e-12
    )
    fractions = [landfill.fraction_of_waste for landfill in scenario.landfills]
    assert fractions == pytest.approx([0.9, 0.1, 0.0], abs=1e-12)
