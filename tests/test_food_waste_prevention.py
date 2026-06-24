"""Regression tests for food-waste-prevention fraction renormalization in
`City.implement_dst_changes_simple_v1_5`.

Bug (fixed): food prevention removes food mass and shrinks the total, but the
non-food waste fractions were rescaled by ``old_nonfood_total / new_nonfood_total``
instead of by the actual total-reduction factor. That over-inflated every
non-food share, so the diversion allocator believed more metal / glass / other /
textiles existed than the (unchanged) masses actually hold. The downstream
mass check then reported a spurious ``Negative mass for <type>`` and the city
DST chart errored — e.g. Algiers with composting + recycling + food prevention.

The fix rescales *every* fraction by ``1 - food_waste_prevention * food_fraction``
(the same factor ``waste_mass`` is reduced by), preserving the invariant

    waste_fractions[w] * waste_mass == waste_masses[w]   for every waste type w.

These tests build a synthetic, food-heavy city from country defaults
(``dst_baseline_blank`` — no database required) and exercise the real
``implement_dst_changes_simple_v1_5`` code path. On the pre-fix code the
food-prevention tests fail (negative ``textiles`` for this city at fwp >= 0.25,
and the renormalization formula is wrong); they pass on the fixed code.
"""

import pandas as pd
import pytest

from SWEET_python.city_params import City, CustomError, DiversionFractions

# Food-heavy synthetic city (food ~50%) built purely from country defaults.
_COUNTRY, _POP, _PRECIP, _TEMP = "Algeria", 2_594_000, 716.81, 18.38
# A diversion feasible with no food prevention that triggered the bug once food
# prevention was applied (composting + recycling contend, pushing recycling onto
# the over-inflated non-food pools). Kept within the food-prevention range where
# it stays genuinely feasible (at very high prevention the compostable pool
# shrinks below the compost target, which is a *correct* rejection, not the bug).
_COMPOST, _RECYCLING = 0.40, 0.40
_IMPLEMENT_YEAR, _SCENARIO = 2026, 1


def _fresh_city() -> City:
    city = City("fwp_regression")
    city.dst_baseline_blank(_COUNTRY, _POP, _PRECIP, _TEMP)
    return city


def _implement(city: City, food_waste_prevention: float, compost: float, recycling: float):
    div_fractions = DiversionFractions(
        compost=compost, anaerobic=0.0, combustion=0.0, recycling=recycling
    )
    city.implement_dst_changes_simple_v1_5(
        div_fractions, 0, 0, 0.0, 0.0, _IMPLEMENT_YEAR, _SCENARIO, food_waste_prevention
    )


def _net_masses(scenario_parameters) -> dict:
    nm = scenario_parameters.net_masses
    if isinstance(nm, pd.DataFrame):
        return {col: float(nm[col].min()) for col in nm.columns}
    return {key: float(value) for key, value in nm.items()}


def test_diversion_is_feasible_without_food_prevention():
    """Sanity: the chosen diversion is feasible at fwp=0, so the food-prevention
    assertions below are not vacuously true."""
    city = _fresh_city()
    _implement(city, 0.0, _COMPOST, _RECYCLING)  # must not raise
    negatives = {w: v for w, v in _net_masses(city.scenario_parameters[0]).items() if v < -1e-6}
    assert not negatives, f"baseline diversion already infeasible: {negatives}"


@pytest.mark.parametrize("fwp", [0.0, 0.25, 0.5])
def test_food_prevention_does_not_create_negative_mass(fwp):
    """A diversion feasible at fwp=0 stays feasible as food prevention rises
    (prevention removes waste, so it can only make diversion easier). Pre-fix
    this raised 'Negative mass for textiles' for this city at fwp >= 0.25."""
    city = _fresh_city()
    try:
        _implement(city, fwp, _COMPOST, _RECYCLING)
    except CustomError as exc:
        pytest.fail(
            f"food prevention {fwp:.0%} wrongly rejected a feasible diversion: {exc.message}"
        )
    negatives = {w: v for w, v in _net_masses(city.scenario_parameters[0]).items() if v < -1e-6}
    assert not negatives, f"negative net masses at fwp={fwp:.0%}: {negatives}"


@pytest.mark.parametrize("fwp", [0.25, 0.5, 0.75, 0.9])
def test_food_prevention_preserves_fraction_renormalization(fwp):
    """After food prevention every fraction is the original divided by the
    total-reduction factor (food additionally scaled by ``1 - fwp``), and the
    row still sums to 1. This is the exact invariant the bug broke for non-food
    types (it scaled them by old_nonfood/new_nonfood instead). Uses zero
    diversion so the renormalization is isolated from any feasibility limit."""
    city = _fresh_city()
    base = city.baseline_parameters.waste_fractions.iloc[0, :].copy()
    food0 = float(base["food"])
    total_scale = 1 - fwp * food0  # reduced_total / original_total

    _implement(city, fwp, compost=0.0, recycling=0.0)
    scenario = city.scenario_parameters[0].waste_fractions.iloc[0, :]

    for waste_type, base_frac in base.items():
        if waste_type == "food":
            expected = food0 * (1 - fwp) / total_scale
        else:
            expected = float(base_frac) / total_scale
        assert float(scenario[waste_type]) == pytest.approx(expected, rel=1e-9, abs=1e-12), (
            f"renormalization wrong for {waste_type} at fwp={fwp:.0%}: "
            f"got {float(scenario[waste_type])}, expected {expected}"
        )

    assert float(scenario.sum()) == pytest.approx(1.0, abs=1e-9)
