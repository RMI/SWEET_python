"""The city-level advanced DST reports where its methane goes, and how much is already buried.

``with_methane_budget`` splits each year's landfill methane into what is
generated, captured, oxidised in the cover and emitted, and ``committed_before``
reports the methane of the waste buried before a year. Both are read from the
run the headline comes from, so the tests that matter are the reconciliations:
the budget adds up to the headline total, and the buried-before series matches
a run in which the city simply stops sending waste.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python.advanced_dst_city import AdvancedDSTCityRequest, run_advanced_dst_city
from SWEET_python.city_params import CustomError
from SWEET_python.dst_common import OX_CAP, OX_NOCAP

YEARS = range(2000, 2051)
FRACTIONS = [0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.02, 0.03, 0.0, 0.05]  # sums to 1.0
IMPLEMENT_YEAR = 2025
TOTAL_WASTE = 20_000.0

LANDFILL, CONTROLLED_DUMP, OPEN_DUMP = 0, 1, 2


def _constant(value):
    return {y: value for y in YEARS}


def _landfill(site_type, gas=0.0, scenario_gas=None, flaring=None):
    scenario_gas = gas if scenario_gas is None else scenario_gas
    spec = dict(
        landfill_type={"baseline": site_type, "scenario": site_type},
        landfill_open_close={"baseline": [2000, 2050], "scenario": [2000, 2050]},
        gas_capture_efficiency={
            "baseline": _constant(gas),
            "scenario": {y: (scenario_gas if y >= IMPLEMENT_YEAR else gas) for y in YEARS},
        },
    )
    if flaring is not None:
        spec["flaring"] = {"baseline": _constant(flaring), "scenario": _constant(flaring)}
    return spec


def _request(landfills, *, compost=0.0, waste_mass=None) -> AdvancedDSTCityRequest:
    share = 1.0 / len(landfills)
    mass = waste_mass or _constant(TOTAL_WASTE)
    body = dict(
        city_name="Budgetville",
        precipitation=1200.0,
        temperature=20.0,
        implement_year=IMPLEMENT_YEAR,
        waste_mass={"baseline": mass, "scenario": mass},
        waste_fractions={"baseline": _constant(FRACTIONS), "scenario": _constant(FRACTIONS)},
        landfills=landfills,
        landfill_split_timeline={
            "baseline": _constant([share] * len(landfills)),
            "scenario": _constant([share] * len(landfills)),
        },
        country="USA",
    )
    if compost:
        body["diversion_fractions"] = {
            "baseline": {"compost": _constant(compost)},
            "scenario": {"compost": _constant(compost)},
        }
    return AdvancedDSTCityRequest(**body)


MIXED = [_landfill(LANDFILL, gas=0.6), _landfill(CONTROLLED_DUMP), _landfill(OPEN_DUMP)]


# --------------------------------------------------------------------------- #
# The methane budget
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("half", ["baseline", "scenario"])
def test_budget_adds_up_to_the_headline(half):
    result = run_advanced_dst_city(_request(MIXED, compost=0.1), with_methane_budget=True)
    budget = result["methane_budget"][half]

    assert list(budget.columns) == [
        "generated", "captured", "oxidised", "emitted", "diversion", "total",
    ]
    parts = budget.captured + budget.oxidised + budget.emitted
    np.testing.assert_allclose(budget.generated, parts, rtol=0, atol=1e-9)
    np.testing.assert_allclose(budget.emitted + budget.diversion, budget.total, rtol=1e-12)
    pd.testing.assert_series_equal(budget.total, result[half]["total"], check_names=False)
    # Composting emits, so the split is not just the landfills.
    assert budget.diversion.max() > 0
    assert (budget[["generated", "captured", "emitted"]] >= 0).all().all()
    assert budget.oxidised.min() > -1e-9


def test_asking_for_the_budget_changes_nothing():
    plain = run_advanced_dst_city(_request(MIXED, compost=0.1), with_site_emissions=True)
    asked = run_advanced_dst_city(
        _request(MIXED, compost=0.1),
        with_site_emissions=True,
        with_methane_budget=True,
        committed_before=2026,
    )
    for half in ("baseline", "scenario"):
        pd.testing.assert_frame_equal(plain[half], asked[half])
        for before, after in zip(plain["site_emissions"][half], asked["site_emissions"][half]):
            pd.testing.assert_frame_equal(before.city, after.city)


def test_captured_and_oxidised_follow_the_sites_rates():
    """One capturing landfill: captured is generation x capture x flare, and the
    cover oxidises its share of whatever is not captured."""
    result = run_advanced_dst_city(
        _request([_landfill(LANDFILL, gas=0.6, flaring=0.9)]), with_methane_budget=True
    )
    budget = result["methane_budget"]["baseline"]
    live = budget.generated > 0

    np.testing.assert_allclose(budget.captured[live], budget.generated[live] * 0.6 * 0.9)
    np.testing.assert_allclose(
        budget.oxidised[live],
        (budget.generated[live] - budget.captured[live]) * OX_CAP["landfill"],
    )


def test_a_site_without_capture_only_oxidises():
    result = run_advanced_dst_city(_request([_landfill(LANDFILL)]), with_methane_budget=True)
    budget = result["methane_budget"]["baseline"]

    assert (budget.captured == 0).all()
    np.testing.assert_allclose(budget.oxidised, budget.generated * OX_NOCAP["landfill"])


def test_an_open_dump_emits_all_it_generates():
    result = run_advanced_dst_city(_request([_landfill(OPEN_DUMP)]), with_methane_budget=True)
    budget = result["methane_budget"]["baseline"]

    np.testing.assert_allclose(budget.emitted, budget.generated)
    np.testing.assert_allclose(budget.oxidised, 0.0, atol=1e-9)


def test_capture_added_in_the_scenario_shows_up_as_captured():
    result = run_advanced_dst_city(
        _request([_landfill(LANDFILL, scenario_gas=0.6)]), with_methane_budget=True
    )
    baseline = result["methane_budget"]["baseline"]
    scenario = result["methane_budget"]["scenario"]

    assert (baseline.captured == 0).all()
    assert (scenario.captured.loc[: IMPLEMENT_YEAR - 1] == 0).all()
    assert (scenario.captured.loc[IMPLEMENT_YEAR:] > 0).all()
    # Same waste, so the same methane is made; only where it goes changes.
    np.testing.assert_allclose(scenario.generated, baseline.generated)


# --------------------------------------------------------------------------- #
# Methane from waste already buried
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("before", [2010, 2026, 2040])
def test_committed_is_a_city_that_stops_sending_waste(before):
    """The buried-before series equals the landfill emissions of a run whose city
    generates nothing from `before` on -- the definition, run the long way."""
    stopped = {y: (TOTAL_WASTE if y < before else 0.0) for y in YEARS}
    result = run_advanced_dst_city(
        _request(MIXED, compost=0.1), with_methane_budget=True, committed_before=before
    )
    reference = run_advanced_dst_city(
        _request(MIXED, compost=0.1, waste_mass=stopped), with_methane_budget=True
    )

    assert result["committed"]["before"] == before
    for half in ("baseline", "scenario"):
        np.testing.assert_allclose(
            result["committed"][half]["total"],
            reference["methane_budget"][half]["emitted"],
            rtol=1e-10,
        )


def test_committed_is_all_of_it_before_the_year_and_shrinks_after():
    result = run_advanced_dst_city(
        _request(MIXED), with_methane_budget=True, committed_before=2026
    )
    committed = result["committed"]["baseline"]["total"]
    landfills = result["methane_budget"]["baseline"]["emitted"]

    # A year's deposits start emitting the year after, so 2026 is still all old waste.
    np.testing.assert_allclose(committed.loc[:2026], landfills.loc[:2026])
    assert (committed.loc[2027:] < landfills.loc[2027:]).all()
    assert committed.loc[2027:].is_monotonic_decreasing


def test_capture_added_later_reaches_waste_already_buried():
    result = run_advanced_dst_city(
        _request([_landfill(LANDFILL, scenario_gas=0.6)]), committed_before=2020
    )
    baseline = result["committed"]["baseline"]["total"]
    scenario = result["committed"]["scenario"]["total"]

    np.testing.assert_allclose(scenario.loc[: IMPLEMENT_YEAR - 1], baseline.loc[: IMPLEMENT_YEAR - 1])
    assert (scenario.loc[IMPLEMENT_YEAR:] < baseline.loc[IMPLEMENT_YEAR:]).all()


def test_committed_before_the_first_year_is_nothing():
    result = run_advanced_dst_city(_request(MIXED), committed_before=2000)
    assert (result["committed"]["baseline"]["total"] == 0).all()


@pytest.mark.parametrize("before", [1999, 2052])
def test_committed_before_outside_the_model_is_refused(before):
    with pytest.raises(CustomError) as raised:
        run_advanced_dst_city(_request(MIXED), committed_before=before)
    assert raised.value.code == "invalid_committed_before"
