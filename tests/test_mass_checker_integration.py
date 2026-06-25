"""
Integration tests for City.mass_checker_math after the min-cost-max-flow swap-in.

These exercise the REAL method (feasibility / accept-reject + no crashes),
stubbing only `_divs_from_component_fractions` (the DB/mass-dependent output
builder) so no database or city data is needed. They lock in:
  - the documented false-rejection fix (full feasible band reachable),
  - the under-delivery / empty-pool "silent accept" fix,
  - the legacy AssertionError-crash inputs now solving cleanly,
  - combustion remainder feasibility + divide-by-zero guard,
  - the no-conflict happy path still being taken.

Requires the SWEET_python runtime deps (pydantic, numpy, geopy, ...).
"""

import pytest

from SWEET_python.city_params import (
    City, CustomError, WasteFractions, DiversionFractions, DivComponentFractions,
)

WT = ["food", "green", "wood", "paper_cardboard", "textiles",
      "plastic", "metal", "glass", "rubber", "other"]
ELIG = {
    "compost": ["food", "green", "wood", "paper_cardboard"],
    "anaerobic": ["food", "green", "wood", "paper_cardboard"],
    "combustion": ["food", "green", "wood", "paper_cardboard", "textiles", "plastic", "rubber"],
    "recycling": ["wood", "paper_cardboard", "textiles", "plastic", "rubber", "metal", "glass", "other"],
}


def _wf(**kw):
    b = {w: 0.0 for w in WT}
    b.update(kw)
    return b


def _norm(waste, comps):
    s = sum(waste[c] for c in comps)
    return _wf(**{c: (waste[c] / s if s > 0 else 0.0) for c in comps})


def _run(waste, targets):
    """Return True if mass_checker_math accepts, False if it raises CustomError."""
    city = City("test")
    city._divs_from_component_fractions = lambda *a, **k: "OK"  # stub DB/mass builder
    dcf = DivComponentFractions(**{d: WasteFractions(**_norm(waste, ELIG[d])) for d in ELIG})
    df = DiversionFractions(
        compost=targets.get("compost", 0.0),
        anaerobic=targets.get("anaerobic", 0.0),
        combustion=targets.get("combustion", 0.0),
        recycling=targets.get("recycling", 0.0),
    )
    try:
        city.mass_checker_math(div_fractions=df, div_component_fractions=dcf,
                               waste_fractions=WasteFractions(**waste), scenario=1)
        return True
    except CustomError:
        return False


CASES = [
    ("repro_c40_r50", _wf(food=.10, paper_cardboard=.45, plastic=.45),
     {"compost": .40, "recycling": .50}, True),
    ("repro_c40_r60_ceiling", _wf(food=.10, paper_cardboard=.45, plastic=.45),
     {"compost": .40, "recycling": .60}, True),
    ("repro_c40_r62_over", _wf(food=.10, paper_cardboard=.45, plastic=.45),
     {"compost": .40, "recycling": .62}, False),
    ("all_food_compost100", _wf(food=1.0), {"compost": 1.0}, True),
    ("all_food_recycling1_impossible", _wf(food=1.0), {"recycling": .01}, False),
    ("subtol_compost_on_textiles", _wf(textiles=1.0), {"compost": .0003}, False),
    ("metal_compost_ineligible", _wf(metal=1.0), {"compost": .01}, False),
    ("only_paper_60_40", _wf(paper_cardboard=1.0), {"compost": .60, "recycling": .40}, True),
    ("only_paper_60_50_over", _wf(paper_cardboard=1.0), {"compost": .60, "recycling": .50}, False),
    ("modest_no_conflict_happy_path",
     _wf(food=.40, green=.10, wood=.05, paper_cardboard=.15, plastic=.10,
         metal=.05, glass=.05, textiles=.05, other=.05),
     {"compost": .20, "recycling": .20}, True),
    ("legacy_crash_A",
     _wf(wood=.18, paper_cardboard=.064, textiles=.284, glass=.369, rubber=.104),
     {"anaerobic": .1979, "recycling": .4188}, True),
    ("legacy_crash_B", _wf(paper_cardboard=.433, plastic=.385, glass=.182),
     {"compost": .3585, "recycling": .3527}, True),
    ("C4_combustible_sparing", _wf(food=.30, paper_cardboard=.30, metal=.30, plastic=.10),
     {"recycling": .30, "combustion": .55}, True),
    ("C5_combustion_over", _wf(food=.30, paper_cardboard=.30, metal=.30, plastic=.10),
     {"recycling": .30, "combustion": .71}, False),
    ("C_divzero_no_combustible", _wf(paper_cardboard=1.0),
     {"compost": .60, "recycling": .40, "combustion": .05}, False),
    ("C_divzero_combustion0_ok", _wf(paper_cardboard=1.0),
     {"compost": .60, "recycling": .40, "combustion": 0.0}, True),
    ("full_4slider_boundary",
     _wf(food=.25, green=.10, wood=.05, paper_cardboard=.15, plastic=.15,
         metal=.10, glass=.10, textiles=.05),
     {"compost": .30, "recycling": .25, "combustion": .40}, True),
]


@pytest.mark.parametrize("name,waste,targets,expected", CASES, ids=[c[0] for c in CASES])
def test_mass_checker_accept_reject(name, waste, targets, expected):
    assert _run(waste, targets) is expected


def test_repro_full_band_no_false_rejection():
    """compost=40%: every recycling 0..60% must be accepted, 61..100% rejected."""
    waste = _wf(food=.10, paper_cardboard=.45, plastic=.45)
    for r in range(0, 101):
        accepted = _run(waste, {"compost": .40, "recycling": r / 100})
        assert accepted is (r <= 60), f"recycling {r}% -> {accepted}"
