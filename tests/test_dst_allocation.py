"""
Unit tests for the DST min-cost-max-flow allocator (SWEET_python.dst_allocation).

These are pure (no City / DB / geopy needed). They:
  - lock in the documented bug fix and the audit regression suite;
  - cross-check the solver against an INDEPENDENT exact-rational Hall/Gale
    feasibility oracle (different math, no shared quantization) on a fuzz sweep;
  - verify combustion's leftover-remainder feasibility + the divide-by-zero guard.

The full standalone harness (~400k-case fuzz) is SWEET_python/dst_allocation_prototype.py.
"""

import itertools
import random
from fractions import Fraction

import pytest

from SWEET_python.dst_allocation import (
    DEFAULT_ELIGIBILITY,
    NON_COMBUSTION,
    SCALE,
    solve_allocation,
    max_feasible_target,
)

ELIG = DEFAULT_ELIGIBILITY
WASTE_TYPES = ["food", "green", "wood", "paper_cardboard", "textiles",
               "plastic", "metal", "glass", "rubber", "other"]
COMBUSTIBLE = ELIG["combustion"]
NON_COMBUSTIBLE = set(WASTE_TYPES) - COMBUSTIBLE


def wf(**kw):
    return {w: kw.get(w, 0.0) for w in WASTE_TYPES}


# --------------------------------------------------------------------------- #
# Independent oracles (exact rational; share no code with the flow solver)
# --------------------------------------------------------------------------- #
def hall_feasible_exact(waste, targets, treatments=NON_COMBUSTION):
    tgt = {t: Fraction(targets.get(t, 0.0)) for t in treatments}
    av = {w: Fraction(waste.get(w, 0.0)) for w in WASTE_TYPES}
    for r in range(1, len(treatments) + 1):
        for sub in itertools.combinations(treatments, r):
            demand = sum(tgt[t] for t in sub)
            reach = set().union(*(ELIG[t] for t in sub))
            if demand > sum(av[w] for w in reach):
                return False
    return True


def max_remainder_exact(waste, targets):
    av = {w: Fraction(waste.get(w, 0.0)) for w in WASTE_TYPES}
    total_comb = sum(av[w] for w in COMBUSTIBLE)
    sum3 = sum(Fraction(targets.get(t, 0.0)) for t in NON_COMBUSTION)
    mgo = sum(av[w] for w in NON_COMBUSTIBLE)
    absorbed = min(Fraction(targets.get("recycling", 0.0)), mgo)
    return total_comb - max(Fraction(0), sum3 - absorbed)


def combustion_feasible(waste, targets):
    return (hall_feasible_exact(waste, targets)
            and Fraction(targets.get("combustion", 0.0)) <= max_remainder_exact(waste, targets))


def allocation_is_valid(waste, targets, allocation, treatments=NON_COMBUSTION, tol=1e-6):
    for t in treatments:
        if abs(sum(allocation.get(t, {}).values()) - targets.get(t, 0.0)) > tol:
            return False
    for t in treatments:
        for w, v in allocation.get(t, {}).items():
            if w not in ELIG[t] or v < -tol:
                return False
    for w in WASTE_TYPES:
        if sum(allocation.get(t, {}).get(w, 0.0) for t in treatments) > waste.get(w, 0.0) + tol:
            return False
    if sum(sum(allocation.get(t, {}).values()) for t in treatments) > sum(waste.values()) + tol:
        return False
    return True


# --------------------------------------------------------------------------- #
# Curated + audit regression scenarios (no combustion)
#   (name, waste, targets, expected_feasible)
# --------------------------------------------------------------------------- #
NONCOMBUSTION_CASES = [
    ("repro_compost40_recycling50",
     wf(food=0.10, paper_cardboard=0.45, plastic=0.45),
     {"compost": 0.40, "recycling": 0.50}, True),
    ("repro_at_ceiling_recycling60",
     wf(food=0.10, paper_cardboard=0.45, plastic=0.45),
     {"compost": 0.40, "recycling": 0.60}, True),
    ("repro_over_ceiling_recycling62",
     wf(food=0.10, paper_cardboard=0.45, plastic=0.45),
     {"compost": 0.40, "recycling": 0.62}, False),
    ("all_food_compost100", wf(food=1.0), {"compost": 1.0}, True),
    ("all_food_recycling1_impossible", wf(food=1.0), {"recycling": 0.01}, False),
    ("all_plastic_recycling100", wf(plastic=1.0), {"recycling": 1.0}, True),
    ("all_plastic_compost1_impossible", wf(plastic=1.0), {"compost": 0.01}, False),
    ("only_paper_60_40_fits", wf(paper_cardboard=1.0),
     {"compost": 0.60, "recycling": 0.40}, True),
    ("only_paper_60_50_over", wf(paper_cardboard=1.0),
     {"compost": 0.60, "recycling": 0.50}, False),
    ("food_heavy_50_45", wf(food=0.55, paper_cardboard=0.20, plastic=0.15, metal=0.10),
     {"compost": 0.50, "recycling": 0.45}, True),
    ("food_heavy_60_45_over", wf(food=0.55, paper_cardboard=0.20, plastic=0.15, metal=0.10),
     {"compost": 0.60, "recycling": 0.45}, False),
    ("compost_anaerobic_over_organics", wf(food=0.50, paper_cardboard=0.10, plastic=0.40),
     {"compost": 0.35, "anaerobic": 0.35}, False),
    ("all_three_mixed", wf(food=0.30, green=0.10, wood=0.10, paper_cardboard=0.10,
     plastic=0.20, metal=0.10, glass=0.10),
     {"compost": 0.30, "anaerobic": 0.20, "recycling": 0.40}, True),
    # audit: empty-pool / silent-drop (legacy wrongly "accepted" with 0 tons)
    ("audit_empty_pool_recycling_on_food", wf(food=1.0), {"recycling": 0.01}, False),
    ("audit_subtolerance_compost_on_textiles", wf(textiles=1.0), {"compost": 0.0003}, False),
    ("audit_metal_compost_ineligible", wf(metal=1.0), {"compost": 0.01}, False),
    # audit: inputs that crashed legacy with AssertionError(city_params.py:6144)
    ("audit_legacy_crash_A",
     wf(wood=0.18, paper_cardboard=0.064, textiles=0.284, glass=0.369, rubber=0.104),
     {"anaerobic": 0.1979, "recycling": 0.4188}, True),
    ("audit_legacy_crash_B",
     wf(paper_cardboard=0.433, plastic=0.385, glass=0.182),
     {"compost": 0.3585, "recycling": 0.3527}, True),
    # boundary / numeric
    ("one_step_over_shared_pool", wf(wood=1.0),
     {"compost": 0.5, "recycling": 0.500001}, False),
    ("one_step_under_shared_pool", wf(wood=1.0),
     {"compost": 0.5, "recycling": 0.499999}, True),
    ("float_dust_0p1_plus_0p2", wf(paper_cardboard=1.0), {"compost": 0.1 + 0.2}, True),
    ("not_sum_to_one_097", wf(food=0.55, paper_cardboard=0.20, plastic=0.22),
     {"recycling": 0.42}, True),
    ("not_sum_to_one_097_over", wf(food=0.55, paper_cardboard=0.20, plastic=0.22),
     {"recycling": 0.43}, False),
]


@pytest.mark.parametrize("name,waste,targets,expected",
                         NONCOMBUSTION_CASES, ids=[c[0] for c in NONCOMBUSTION_CASES])
def test_noncombustion_feasibility(name, waste, targets, expected):
    res = solve_allocation(waste, targets, ELIG)
    assert res["feasible"] is expected
    if expected:
        assert allocation_is_valid(waste, targets, res["allocation"])


@pytest.mark.parametrize("name,waste,targets,expected",
                         NONCOMBUSTION_CASES, ids=[c[0] for c in NONCOMBUSTION_CASES])
def test_noncombustion_matches_exact_oracle(name, waste, targets, expected):
    assert solve_allocation(waste, targets, ELIG)["feasible"] is hall_feasible_exact(waste, targets)


def test_repro_full_band_reachable():
    """The documented bug: old code accepted only recycling 0..32%; the true
    feasible ceiling with compost=40% is 60%."""
    waste = wf(food=0.10, paper_cardboard=0.45, plastic=0.45)
    accepted = [r for r in range(0, 101)
                if solve_allocation(waste, {"compost": 0.40, "recycling": r / 100}, ELIG)["feasible"]]
    assert accepted == list(range(0, 61))  # 0..60 inclusive
    assert max_feasible_target("recycling", waste, {"compost": 0.40}, ELIG) == pytest.approx(0.60, abs=1e-6)


# --------------------------------------------------------------------------- #
# Combustion (caller-side remainder step) — validated via the prototype's logic
#   (name, waste, targets WITH combustion, expected_feasible)
# --------------------------------------------------------------------------- #
def solve_with_combustion(waste, targets):
    """Mirror of City.mass_checker_math's intended combustion handling, for
    test purposes: solve the 3 with combustible-sparing, then check the
    leftover-remainder ceiling."""
    three = {t: targets.get(t, 0.0) for t in NON_COMBUSTION}
    tc = targets.get("combustion", 0.0)
    res = solve_allocation(waste, three, ELIG, spare_combustibles=(tc > 0))
    if not res["feasible"]:
        return False
    alloc = res["allocation"]
    leftover_comb = sum(
        waste.get(w, 0.0) - sum(alloc[t].get(w, 0.0) for t in NON_COMBUSTION)
        for w in COMBUSTIBLE
    )
    return tc <= leftover_comb + 1e-9


COMBUSTION_CASES = [
    ("C4_combustible_sparing_coupling",
     wf(food=0.30, paper_cardboard=0.30, metal=0.30, plastic=0.10),
     {"recycling": 0.30, "combustion": 0.55}, True),
    ("C5_combustion_just_over",
     wf(food=0.30, paper_cardboard=0.30, metal=0.30, plastic=0.10),
     {"recycling": 0.30, "combustion": 0.71}, False),
    ("C6_organics_eaten_ok", wf(food=0.50, paper_cardboard=0.20, plastic=0.30),
     {"compost": 0.40, "combustion": 0.55}, True),
    ("C7_organics_eaten_over", wf(food=0.50, paper_cardboard=0.20, plastic=0.30),
     {"compost": 0.40, "combustion": 0.65}, False),
    ("C9_max_remainder_spare_paper_via_metal", wf(paper_cardboard=0.60, metal=0.40),
     {"compost": 0.30, "recycling": 0.30, "combustion": 0.01}, True),
    ("C10_recycling_starves_combustion",
     wf(food=0.20, paper_cardboard=0.40, plastic=0.40),
     {"compost": 0.20, "recycling": 0.40, "combustion": 0.45}, False),
    ("C_divzero_no_combustible_left", wf(paper_cardboard=1.0),
     {"compost": 0.60, "recycling": 0.40, "combustion": 0.05}, False),
    ("C_divzero_combustion0_ok", wf(paper_cardboard=1.0),
     {"compost": 0.60, "recycling": 0.40, "combustion": 0.0}, True),
    ("C_combustion_noncombustible_only", wf(metal=1.0), {"combustion": 0.10}, False),
    ("C_full_4slider_tight_boundary",
     wf(food=0.25, green=0.10, wood=0.05, paper_cardboard=0.15, plastic=0.15,
        metal=0.10, glass=0.10, textiles=0.05),
     {"compost": 0.30, "recycling": 0.25, "combustion": 0.40}, True),
]


@pytest.mark.parametrize("name,waste,targets,expected",
                         COMBUSTION_CASES, ids=[c[0] for c in COMBUSTION_CASES])
def test_combustion_feasibility(name, waste, targets, expected):
    assert solve_with_combustion(waste, targets) is expected
    assert combustion_feasible(waste, targets) is expected  # vs closed-form oracle


# --------------------------------------------------------------------------- #
# Fuzz: solver vs independent exact oracle (band-aware)
# --------------------------------------------------------------------------- #
def _random_waste(rng):
    k = rng.choice([1, 2, 3, 3, 4, 5, 7, 10])
    ts = rng.sample(WASTE_TYPES, k)
    raw = [rng.random() for _ in ts]
    s = sum(raw) or 1.0
    return wf(**{t: r / s for t, r in zip(ts, raw)})


def _boundary_slack(waste, targets, treatments=NON_COMBUSTION):
    tgt = {t: Fraction(targets.get(t, 0.0)) for t in treatments}
    av = {w: Fraction(waste.get(w, 0.0)) for w in WASTE_TYPES}
    best = Fraction(1)
    for r in range(1, len(treatments) + 1):
        for sub in itertools.combinations(treatments, r):
            demand = sum(tgt[t] for t in sub)
            cap = sum(av[w] for w in set().union(*(ELIG[t] for t in sub)))
            best = min(best, abs(demand - cap))
    return best


def test_fuzz_solver_matches_exact_oracle():
    rng = random.Random(20260622)
    genuine = 0
    invalid = 0
    for _ in range(20000):
        waste = _random_waste(rng)
        targets = {t: round(rng.random() * rng.choice([0.3, 0.6, 1.0]), 4)
                   for t in NON_COMBUSTION if rng.random() < 0.75}
        res = solve_allocation(waste, targets, ELIG)
        if res["feasible"] != hall_feasible_exact(waste, targets):
            if _boundary_slack(waste, targets) > Fraction(1, SCALE):
                genuine += 1  # only count disagreements outside the grid band
        elif res["feasible"] and not allocation_is_valid(waste, targets, res["allocation"]):
            invalid += 1
    assert genuine == 0
    assert invalid == 0
