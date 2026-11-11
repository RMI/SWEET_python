"""
Unit tests for the DST diversion allocator (SWEET_python.dst_allocation).

These are pure (no City / DB / geopy needed). They:
  - lock in the documented bug fix and the audit regression suite;
  - cross-check the max-flow feasibility test against an INDEPENDENT
    exact-rational Hall/Gale oracle (different math, no shared quantization) on
    a fuzz sweep;
  - check incineration's feasibility as the fourth treatment;
  - check the re-split (`proportional_allocation`): it keeps each diversion's
    mix where nothing is over-drawn, shares an over-drawn type in the same
    proportion for every diversion that takes it, doesn't jump where a type
    runs out, and never diverts less of a type when a slider rises.

The standalone feasibility harness (~400k-case fuzz) is dst_allocation_prototype.py.
"""

import itertools
import random
from fractions import Fraction

import pytest

from SWEET_python.dst_allocation import (
    DEFAULT_ELIGIBILITY,
    NON_COMBUSTION,
    SCALE,
    TREATMENTS,
    max_feasible_target,
    proportional_allocation,
    solve_allocation,
    treatment_mixes,
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
# Combustion: the fourth treatment in the same max-flow
#   (name, waste, targets WITH combustion, expected_feasible)
# --------------------------------------------------------------------------- #
def solve_with_combustion(waste, targets):
    """City.mass_checker_math's feasibility test: all four treatments in one
    max-flow, incineration eligible for every type."""
    return solve_allocation(waste, targets, ELIG, treatments=TREATMENTS)["feasible"]


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
    # metal/glass/other are now combustion-eligible: a 100%-metal city CAN combust.
    ("C_combustion_metal_now_combustible", wf(metal=1.0), {"combustion": 0.10}, True),
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


# --------------------------------------------------------------------------- #
# The re-split: proportional_allocation keeps each diversion's own mix
# --------------------------------------------------------------------------- #
def _taken(allocation):
    """How much of each type all the diversions take together."""
    return {w: sum(allocation.get(t, {}).get(w, 0.0) for t in TREATMENTS) for w in WASTE_TYPES}


def _check_resplit(waste, targets, mixes, tol=1e-8):
    res = proportional_allocation(waste, targets, mixes)
    assert res["feasible"]
    alloc = res["allocation"]
    for t in TREATMENTS:
        assert sum(alloc.get(t, {}).values()) == pytest.approx(targets.get(t, 0.0), abs=tol)
    taken = _taken(alloc)
    assert all(taken[w] <= waste[w] + tol for w in WASTE_TYPES)
    return alloc


# London in the cities table: incineration 47% and recycling 30%, each published
# in proportion to the waste it can take.
LONDON = wf(food=0.34, paper_cardboard=0.36, plastic=0.06, metal=0.04, glass=0.12, other=0.08)
# Navi Mumbai: composting 82%, recycling 0.
NAVI_MUMBAI = wf(food=0.64276, green=0.06064, paper_cardboard=0.25953, other=0.03708)


def test_a_mix_that_fits_is_used_as_it_is():
    mixes = treatment_mixes(LONDON)
    alloc = _check_resplit(LONDON, {"combustion": 0.47, "recycling": 0.30}, mixes)
    for t, share in (("combustion", 0.47), ("recycling", 0.30)):
        for w, p in mixes[t].items():
            assert alloc[t][w] == pytest.approx(share * p, abs=1e-12)


def test_an_over_drawn_type_is_held_back_alike_for_every_diversion():
    """In proportion to the mix: every diversion that takes two types takes them in
    its own ratio times the same factor, so no diversion is steered to a type."""
    rng = random.Random(11)
    checked = 0
    for _ in range(300):
        waste = _random_waste(rng)
        mixes = treatment_mixes(waste)
        targets = {t: round(rng.random() * 0.5, 3) for t in TREATMENTS if mixes[t] and rng.random() < 0.8}
        res = proportional_allocation(waste, targets, mixes)
        if not res["feasible"] or not res["converged"]:
            continue
        alloc = res["allocation"]
        # held[t][w] = x[t][w] / (target * mix): the hold-back on w, times t's own factor.
        held = {t: {w: alloc[t].get(w, 0.0) / (targets[t] * p) for w, p in mixes[t].items()}
                for t in targets if targets[t] > 0}
        for t, s in itertools.combinations(held, 2):
            for w, u in itertools.combinations(sorted(set(held[t]) & set(held[s])), 2):
                if min(held[t][w], held[t][u], held[s][w], held[s][u]) < 1e-6:
                    continue  # a type held back to nothing; the ratio is noise
                assert held[t][w] / held[t][u] == pytest.approx(held[s][w] / held[s][u], rel=1e-6)
                checked += 1
    assert checked > 100


def test_crossing_into_the_re_split_does_not_jump():
    """London's recycling at 35% uses up paper, plastic, metal, glass and other
    exactly. Just past it, incineration gives way and burns food instead."""
    mixes = treatment_mixes(LONDON)
    at = _check_resplit(LONDON, {"combustion": 0.47, "recycling": 0.35}, mixes)
    past = _check_resplit(LONDON, {"combustion": 0.47, "recycling": 0.35 + 1e-6}, mixes)
    for t in TREATMENTS:
        for w in WASTE_TYPES:
            assert abs(past.get(t, {}).get(w, 0.0) - at.get(t, {}).get(w, 0.0)) < 1e-5
    assert past["combustion"]["food"] > at["combustion"]["food"]


def test_more_recycling_in_navi_mumbai_landfills_less_of_everything():
    """The old solver composted all food and green first, so 5% recycling
    landfilled 13 points of paper instead of 4: methane +24% in 2040."""
    mixes = treatment_mixes(NAVI_MUMBAI)
    before = _taken(_check_resplit(NAVI_MUMBAI, {"compost": 0.82}, mixes))
    after = _taken(_check_resplit(NAVI_MUMBAI, {"compost": 0.82, "recycling": 0.05}, mixes))
    for w in WASTE_TYPES:
        assert after[w] >= before[w] - 1e-12
    assert NAVI_MUMBAI["paper_cardboard"] - after["paper_cardboard"] < 1e-9


def test_raising_a_slider_never_diverts_less_of_a_type():
    """Fuzz, mixes proportional or not (an older table's mix may name fewer types)."""
    rng = random.Random(20261110)
    compared = 0
    for _ in range(1500):
        waste = _random_waste(rng)
        mixes = treatment_mixes(waste)
        for t in TREATMENTS:
            if mixes[t] and rng.random() < 0.4:
                keep = [w for w in mixes[t] if rng.random() < 0.7] or list(mixes[t])[:1]
                mixes[t] = {w: rng.random() for w in keep}
        targets = {t: round(rng.random() * 0.5, 3) for t in TREATMENTS if rng.random() < 0.7}
        res = proportional_allocation(waste, targets, mixes)
        if not res["feasible"] or not targets:
            continue
        raised = dict(targets)
        t = rng.choice(list(targets))
        raised[t] += rng.choice([1e-4, 0.01, 0.05])
        res2 = proportional_allocation(waste, raised, mixes)
        if not res2["feasible"]:
            continue
        before, after = _taken(res["allocation"]), _taken(res2["allocation"])
        assert all(after[w] >= before[w] - 1e-8 for w in WASTE_TYPES), (waste, targets, t)
        compared += 1
    assert compared > 500


def test_a_slider_at_its_maximum_is_split_validly():
    """At the maximum some diversion must take none of a type its mix names, which
    the re-split only approaches; it still hits every target within 1e-8."""
    rng = random.Random(7)
    done = 0
    for _ in range(400):
        waste = _random_waste(rng)
        mixes = treatment_mixes(waste)
        targets = {t: round(rng.random() * 0.4, 3) for t in TREATMENTS if mixes[t] and rng.random() < 0.7}
        if not targets:
            continue
        t = rng.choice(list(targets))
        others = {k: v for k, v in targets.items() if k != t}
        if not solve_allocation(waste, others, ELIG, TREATMENTS)["feasible"]:
            continue
        targets[t] = max_feasible_target(t, waste, others, ELIG, TREATMENTS)
        if targets[t] <= 0:
            continue
        _check_resplit(waste, targets, mixes)
        done += 1
    assert done > 100


def test_a_type_the_mix_leaves_out_is_used_once_the_rest_runs_out():
    """Coimbatore's published composting is all food, from the solver this replaced.
    Composting more takes food first, then paper, up to the same maximum as before."""
    waste = wf(food=0.6, paper_cardboard=0.4)
    mixes = treatment_mixes(waste, {"compost": {"food": 1.0}})
    assert mixes["compost"]["paper_cardboard"] == pytest.approx(1e-6)
    alloc = _check_resplit(waste, {"compost": 0.5}, mixes)
    assert alloc["compost"]["paper_cardboard"] < 1e-6
    alloc = _check_resplit(waste, {"compost": 0.7}, mixes)
    assert alloc["compost"]["food"] == pytest.approx(0.6, abs=1e-8)
    assert alloc["compost"]["paper_cardboard"] == pytest.approx(0.1, abs=1e-8)
    assert max_feasible_target("compost", waste, {}, ELIG, TREATMENTS) == pytest.approx(1.0)


def test_a_missing_mix_takes_the_waste_mix():
    mixes = treatment_mixes(LONDON, {"recycling": {"food": 0.5}})
    assert mixes["recycling"] == pytest.approx(
        {w: LONDON[w] / 0.66 for w in ("paper_cardboard", "plastic", "metal", "glass", "other")}
    )


def test_diverting_all_the_waste_fits_though_the_grid_rounds_it_short():
    """Kitakyushu incinerates 74.95% of its waste and recycles the other 25.05%. On the
    1e-9 grid the max-flow can come out a step short, which the old incineration check
    allowed, so its baseline stopped modelling until this was accepted."""
    waste = wf(food=1 / 3, paper_cardboard=1 / 3, plastic=1 / 3)
    targets = {"combustion": 1 / 3, "recycling": 2 / 3}
    assert solve_allocation(waste, targets, ELIG, TREATMENTS)["shortfall"] == pytest.approx(1e-9)
    alloc = _check_resplit(waste, targets, treatment_mixes(waste))
    assert alloc["combustion"]["food"] == pytest.approx(1 / 3, abs=1e-8)
