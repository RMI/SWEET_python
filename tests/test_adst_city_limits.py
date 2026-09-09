"""``run_advanced_dst_city_limits`` bounds both variants the model runs.

The bounds exist so a caller enforcing them cannot build a request the model
refuses on mass balance. That promise is only kept if the bounds are computed
from the same inputs the model uses — and the model runs *two* variants,
enforcing ``over_diversion`` on each, with the scenario's composition, tonnage,
prevention and diversion all taking effect from ``implement_year``. Bounds taken
from the baseline alone leave every scenario year unbounded, which is what these
tests pin.
"""

import pandas as pd
import pytest

from SWEET_python import dst_common as common
from SWEET_python.advanced_dst_city import (
    DIVERSION_PATHWAYS,
    AdvancedDSTCityRequest,
    run_advanced_dst_city,
    run_advanced_dst_city_limits,
)
from SWEET_python.city_params import CustomError

YEARS = range(2000, 2051)
IMPLEMENT_YEAR = 2025
TOTAL_TONS = 100_000.0

# food 20, plastic 50, metal 30
FOOD_PLASTIC_METAL = [0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0]
# the same city with most of its food gone
LEAN_ON_FOOD = [0.05, 0.0, 0.0, 0.0, 0.0, 0.65, 0.30, 0.0, 0.0, 0.0]
NO_ORGANICS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.0]


def _landfill():
    return dict(
        landfill_type={"baseline": 2, "scenario": 2},
        landfill_open_close={
            "baseline": [2000, common.NEVER_CLOSES],
            "scenario": [2000, common.NEVER_CLOSES],
        },
        gas_capture_efficiency={
            "baseline": {y: 0.0 for y in YEARS},
            "scenario": {y: 0.0 for y in YEARS},
        },
    )


def _request(baseline_fractions=None, scenario_fractions=None,
             baseline_diversion=None, scenario_diversion=None):
    baseline_fractions = baseline_fractions or FOOD_PLASTIC_METAL
    payload = dict(
        city_name="Limitsville",
        country="USA",
        precipitation=1200.0,
        temperature=20.0,
        implement_year=IMPLEMENT_YEAR,
        waste_mass={
            "baseline": {y: TOTAL_TONS for y in YEARS},
            "scenario": {y: TOTAL_TONS for y in YEARS},
        },
        waste_fractions={
            "baseline": {y: baseline_fractions for y in YEARS},
            "scenario": {y: (scenario_fractions or baseline_fractions) for y in YEARS},
        },
        landfills=[_landfill()],
        landfill_split_timeline={
            "baseline": {y: [1.0] for y in YEARS},
            "scenario": {y: [1.0] for y in YEARS},
        },
    )
    if baseline_diversion or scenario_diversion:
        payload["diversion_fractions"] = {
            "baseline": {
                p: {y: v for y in YEARS} for p, v in (baseline_diversion or {}).items()
            },
            "scenario": {
                p: {y: v for y in YEARS}
                for p, v in (scenario_diversion or baseline_diversion or {}).items()
            },
        }
    return AdvancedDSTCityRequest(**payload)


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #
def test_bounds_are_reported_per_variant():
    limits = run_advanced_dst_city_limits(_request(baseline_diversion={"compost": 0.1}))

    assert set(limits) == {"baseline", "scenario"}
    for variant in ("baseline", "scenario"):
        assert set(limits[variant]) == {
            "component_use",
            "limiting_component",
            "max_diversion",
            "max_food_waste_prevention",
            "starved_pathways",
        }


def test_the_variants_agree_when_the_request_states_no_scenario_of_its_own():
    limits = run_advanced_dst_city_limits(_request(baseline_diversion={"compost": 0.1}))

    for key in ("component_use", "max_diversion"):
        pd.testing.assert_frame_equal(limits["baseline"][key], limits["scenario"][key])


# --------------------------------------------------------------------------- #
# The scenario is bounded on its own inputs
# --------------------------------------------------------------------------- #
def test_a_scenario_composition_moves_the_scenario_bounds_only():
    """Baseline is 20% food; the scenario drops it to 5%.

    Organics are food alone here, so compost can take 20% of the baseline stream
    and 5% of the scenario's. Reporting the baseline's 20% for the scenario years
    is what let a caller build a request the model then refused.
    """
    limits = run_advanced_dst_city_limits(
        _request(baseline_fractions=FOOD_PLASTIC_METAL, scenario_fractions=LEAN_ON_FOOD)
    )

    baseline = limits["baseline"]["max_diversion"]
    scenario = limits["scenario"]["max_diversion"]

    assert float(baseline.loc[2030, "compost"]) == pytest.approx(0.2)
    assert float(scenario.loc[2030, "compost"]) == pytest.approx(0.05)


def test_the_scenario_bounds_track_baseline_before_the_implement_year():
    """Scenario inputs take effect at `implement_year`, and so do its bounds."""
    limits = run_advanced_dst_city_limits(
        _request(baseline_fractions=FOOD_PLASTIC_METAL, scenario_fractions=LEAN_ON_FOOD)
    )
    scenario = limits["scenario"]["max_diversion"]

    assert float(scenario.loc[IMPLEMENT_YEAR - 1, "compost"]) == pytest.approx(0.2)
    assert float(scenario.loc[IMPLEMENT_YEAR, "compost"]) == pytest.approx(0.05)


def test_a_scenario_only_diversion_is_bounded_too():
    """The pathway the caller is actually editing lives in the scenario half."""
    limits = run_advanced_dst_city_limits(
        _request(
            baseline_diversion={"compost": 0.04},
            scenario_diversion={"compost": 0.04, "recycling": 0.5},
        )
    )
    baseline = limits["baseline"]["component_use"]
    scenario = limits["scenario"]["component_use"]

    # Recycling only exists in the scenario, so only its rows claim recyclables.
    assert float(baseline.loc[2030, "plastic"]) == pytest.approx(0.0)
    assert float(scenario.loc[2030, "plastic"]) > 0.0


def test_a_request_the_model_refuses_is_visible_in_the_scenario_bounds():
    """The parity promise, on the half that used to be unbounded.

    The scenario composts 15% of a stream whose scenario food share is 5%, which
    `run_advanced_dst_city` refuses. Before the bounds were per variant they
    described the baseline's roomier 20% food and reported this as fine.
    """
    request = _request(
        baseline_fractions=FOOD_PLASTIC_METAL,
        scenario_fractions=LEAN_ON_FOOD,
        baseline_diversion={"compost": 0.04},
        scenario_diversion={"compost": 0.15},
    )

    limits = run_advanced_dst_city_limits(request)
    assert float(limits["baseline"]["component_use"].max(axis=1).loc[2030]) <= 1.0
    assert float(limits["scenario"]["component_use"].max(axis=1).loc[2030]) > 1.0

    with pytest.raises(CustomError) as excinfo:
        run_advanced_dst_city(request)
    assert excinfo.value.code == "over_diversion"


# --------------------------------------------------------------------------- #
# An empty pool is infeasible, not free
# --------------------------------------------------------------------------- #
def test_a_pathway_with_no_material_is_named_rather_than_read_as_zero():
    """`component_use` cannot express this case, so it is reported beside it.

    With no organics at all, compost's denominator is zero and its claim is
    0/0 — which `_component_use` reads as zero, so the row looks comfortably
    under 1. A caller reading only `component_use` would call this request fine,
    when in fact the pathway can process none of the stream: `_diverted_masses`
    splits its mass by 0/0 and diverts nothing at all.
    """
    request = _request(baseline_fractions=NO_ORGANICS, baseline_diversion={"compost": 0.15})
    limits = run_advanced_dst_city_limits(request)

    assert float(limits["baseline"]["component_use"].max(axis=1).loc[2030]) == pytest.approx(0.0)
    assert bool(limits["baseline"]["starved_pathways"].loc[2030, "compost"]) is True
    # And the headroom is nil, so a caller enforcing `max_diversion` never
    # arrives here in the first place.
    assert float(limits["baseline"]["max_diversion"].loc[2030, "compost"]) == pytest.approx(0.0)


def test_a_pathway_with_material_is_not_flagged():
    limits = run_advanced_dst_city_limits(
        _request(baseline_diversion={"compost": 0.1, "recycling": 0.2})
    )
    starved = limits["baseline"]["starved_pathways"]

    assert set(starved.columns) == set(DIVERSION_PATHWAYS)
    assert not bool(starved.loc[2030].any())


def test_an_unrequested_pathway_is_not_starved():
    """Starvation is about a demand that cannot be met, not an empty pool."""
    limits = run_advanced_dst_city_limits(
        _request(baseline_fractions=NO_ORGANICS, baseline_diversion={"recycling": 0.2})
    )
    # Compost draws on nothing, but nothing is asked of it either.
    assert not bool(limits["baseline"]["starved_pathways"].loc[2030, "compost"])


def test_the_model_refuses_the_starved_request_the_bounds_flag():
    """`starved_pathways` and `diversion_without_material` are one boundary.

    The mask exists so a caller can see, before it runs anything, the request
    the model will refuse — the same reason the diversion bounds and
    `over_diversion` are measured on one basis.
    """
    request = _request(baseline_fractions=NO_ORGANICS, baseline_diversion={"compost": 0.15})

    assert bool(run_advanced_dst_city_limits(request)["baseline"]["starved_pathways"].loc[2030, "compost"])

    with pytest.raises(CustomError) as excinfo:
        run_advanced_dst_city(request)
    assert excinfo.value.code == "diversion_without_material"
    assert "compost" in excinfo.value.message
