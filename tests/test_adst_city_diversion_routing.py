"""Diversion pathways share material by rerouting, not by a fixed proportional split.

Compost and recycling both draw on wood and paper. Split in proportion to the
composition, a recycling share of 18.9% holds a fixed claim on the city's paper,
so compost used to be capped at 39% of this mix even though recycling could have
taken plastic, metal and glass instead. ``_pathway_splits`` now moves the split
toward a routing that fits whenever one exists, and the bounds say how far a
pathway can go with that freedom (Hall's condition).
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python import dst_common as common
from SWEET_python.advanced_dst_city import (
    DIVERSION_PATHWAYS,
    ROUTING_GRID_TOLERANCE,
    AdvancedDSTCityRequest,
    _diverted_masses,
    _max_pathway_fraction,
    _pathway_splits,
    _routing_slack,
    run_advanced_dst_city,
    run_advanced_dst_city_limits,
)
from SWEET_python.city_params import City, CustomError

YEARS = range(2000, 2051)
YEAR = 2030
TOTAL_TONS = 1_000_000.0

# food, green, wood, paper, textiles, plastic, metal, glass, rubber, other.
# Organics 59%, recyclables 55%, wood + paper 14% of both.
MIX = [0.40, 0.05, 0.02, 0.12, 0.03, 0.12, 0.03, 0.04, 0.01, 0.18]
RECYCLING = 0.189
# Recycling's proportional claim on wood and paper leaves compost this much.
PROPORTIONAL_COMPOST_CAP = (1 - RECYCLING / 0.55) * 0.59


def _fractions(mix=MIX, years=(YEAR,)):
    return pd.DataFrame([mix] * len(years), index=list(years), columns=common.WASTE_COMPONENTS)


def _masses(diversion, mix=MIX, years=(YEAR,)):
    """Gross diverted tonnes per pathway, years x components."""
    fractions = _fractions(mix, years)
    total = pd.Series(TOTAL_TONS, index=fractions.index)
    div = {p: {y: v for y in years} for p, v in diversion.items()}
    _net, gross = _diverted_masses(fractions, total, div, City("Routing"), fractions.index)
    return fractions, total, gross


def _assert_fits(diversion, mix=MIX):
    fractions, total, gross = _masses(diversion, mix)
    generated = fractions.mul(total, axis=0)
    claimed = sum(getattr(gross, p) .reindex(columns=generated.columns, fill_value=0.0) for p in DIVERSION_PATHWAYS)
    assert ((generated - claimed) >= -1e-6).to_numpy().all(), (generated - claimed).min()
    for pathway, share in diversion.items():
        sent = float(getattr(gross, pathway).sum(axis=1).loc[YEAR])
        # A request at its exact bound may come up short by the solver's grid.
        assert sent == pytest.approx(share * TOTAL_TONS, abs=ROUTING_GRID_TOLERANCE * TOTAL_TONS), pathway
    return gross


def _request(diversion, mix=MIX):
    return AdvancedDSTCityRequest(
        city_name="Routing",
        country="USA",
        precipitation=1200.0,
        temperature=20.0,
        implement_year=2025,
        waste_mass={
            "baseline": {y: TOTAL_TONS for y in YEARS},
            "scenario": {y: TOTAL_TONS for y in YEARS},
        },
        waste_fractions={
            "baseline": {y: mix for y in YEARS},
            "scenario": {y: mix for y in YEARS},
        },
        landfills=[
            dict(
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
        ],
        landfill_split_timeline={
            "baseline": {y: [1.0] for y in YEARS},
            "scenario": {y: [1.0] for y in YEARS},
        },
        diversion_fractions={
            "baseline": {p: {y: v for y in YEARS} for p, v in diversion.items()},
            "scenario": {p: {y: v for y in YEARS} for p, v in diversion.items()},
        },
    )


# --------------------------------------------------------------------------- #
# Past the proportional cap
# --------------------------------------------------------------------------- #
def test_compost_runs_past_where_the_proportional_split_stopped():
    assert PROPORTIONAL_COMPOST_CAP == pytest.approx(0.387, abs=1e-3)

    result = run_advanced_dst_city(_request({"compost": 0.5, "recycling": RECYCLING}))

    assert result is not None


def test_recycling_leans_on_what_only_it_can_take():
    """Compost needs the paper, so recycling takes more plastic and less paper."""
    proportional = _assert_fits({"compost": 0.30, "recycling": RECYCLING})
    rerouted = _assert_fits({"compost": 0.50, "recycling": RECYCLING})

    def share(gross, component):
        recycling = gross.recycling.loc[YEAR]
        return float(recycling[component] / recycling.sum())

    assert share(rerouted, "paper_cardboard") < share(proportional, "paper_cardboard")
    assert share(rerouted, "plastic") > share(proportional, "plastic")
    # Wood and paper are not left in the landfill: the two pathways take all of it.
    for component in ("wood", "paper_cardboard"):
        taken = float(rerouted.compost.loc[YEAR, component] + rerouted.recycling.loc[YEAR, component])
        assert taken == pytest.approx(MIX[common.WASTE_COMPONENTS.index(component)] * TOTAL_TONS)


def test_a_split_that_fits_is_left_exactly_proportional():
    fractions = _fractions()
    total = pd.Series(TOTAL_TONS, index=fractions.index)
    shares = {p: pd.Series(0.0, index=fractions.index) for p in DIVERSION_PATHWAYS}
    shares["compost"][:] = 0.30
    shares["recycling"][:] = RECYCLING
    city = City("Routing")

    splits = _pathway_splits(fractions, total, shares, city)

    for pathway in DIVERSION_PATHWAYS:
        components = list(city.div_components[pathway])
        sub = fractions[components]
        expected = sub.div(sub.sum(axis=1), axis=0)
        pd.testing.assert_frame_equal(splits[pathway], expected)


def test_just_past_the_cap_the_split_is_just_past_proportional():
    """The split moves by the smallest step that fits, so it has no jump at the cap."""
    below = _assert_fits({"compost": PROPORTIONAL_COMPOST_CAP - 1e-6, "recycling": RECYCLING})
    above = _assert_fits({"compost": PROPORTIONAL_COMPOST_CAP + 1e-6, "recycling": RECYCLING})

    for pathway in ("compost", "recycling"):
        moved = (getattr(above, pathway) - getattr(below, pathway)).abs().to_numpy().max()
        assert moved < 10.0  # tonnes, of a million


# --------------------------------------------------------------------------- #
# Still refused when no routing fits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "diversion",
    [
        {"compost": 0.60},  # more than the 59% of organics
        {"compost": 0.55, "recycling": 0.50},  # 105% between them
        {"recycling": 0.56},  # more than the 55% of recyclables
    ],
)
def test_a_request_no_routing_fits_is_still_refused(diversion):
    with pytest.raises(CustomError) as excinfo:
        run_advanced_dst_city(_request(diversion))
    assert excinfo.value.code == "over_diversion"


# --------------------------------------------------------------------------- #
# The bounds
# --------------------------------------------------------------------------- #
def test_the_bounds_allow_what_the_routing_can_reach():
    limits = run_advanced_dst_city_limits(_request({"compost": 0.5, "recycling": RECYCLING}))
    maximum = limits["baseline"]["max_diversion"].loc[YEAR]
    use = limits["baseline"]["component_use"].loc[YEAR]

    # Compost: all the organics, since recycling has 41% elsewhere to take its 18.9% from.
    assert float(maximum["compost"]) == pytest.approx(0.59)
    # Recycling: whatever compost leaves of everything the two can take.
    assert float(maximum["recycling"]) == pytest.approx(0.50)
    # Rerouted, no material is claimed past what exists.
    assert float(use.max()) <= 1.0 + 1e-9


def test_the_bounds_and_the_model_agree_on_random_cities():
    """At each pathway's bound the model runs; a little past it, it refuses."""
    rng = np.random.default_rng(20261028)
    city = City("Routing")
    for _ in range(150):
        mix = list(rng.dirichlet(np.ones(10) * rng.uniform(0.3, 3.0)))
        others = {
            p: float(v)
            for p, v in zip(("compost", "anaerobic", "recycling"), rng.dirichlet(np.ones(4))[:3] * 0.6)
        }
        others["combustion"] = 0.0
        pathway = str(rng.choice(["compost", "anaerobic", "recycling"]))
        fractions = _fractions(mix)
        div = {p: pd.Series(v, index=fractions.index) for p, v in others.items()}
        div[pathway][:] = 0.0
        if float(_routing_slack(fractions, div, city).loc[YEAR]) < 0:
            continue  # the others overreach before this pathway takes anything

        bound = float(_max_pathway_fraction(fractions, div, city, pathway).loc[YEAR])
        others[pathway] = bound
        _assert_fits({p: v for p, v in others.items() if v > 0}, mix)

        others[pathway] = bound + 1e-3
        if others[pathway] > 1:
            continue
        _f, _t, gross = _masses({p: v for p, v in others.items() if v > 0}, mix)
        generated = _f.mul(_t, axis=0)
        claimed = sum(getattr(gross, p).reindex(columns=generated.columns, fill_value=0.0) for p in DIVERSION_PATHWAYS)
        assert ((generated - claimed) < -1e-6).to_numpy().any()
