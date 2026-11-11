"""
Diversion allocation for the city model and the city DST.

Composting, anaerobic digestion and recycling each take a share of a city's waste.
Each draws its share from the waste types it can take, in the proportions of its own
mix: the component fractions the caller passes (a city's published mix in the DST),
else the waste mix. `City.mass_checker_math` uses those proportions as they are
whenever they fit.

Incineration burns what those three leave: the same share of every type left
(`incineration_mix`). An incinerator takes the waste left after separate collection
for recycling and composting, so when composting grows, the incinerator and the
landfill each give up food in proportion to what they had.

When the three's mixes don't fit, because together they ask for more of a type than
the city has, `proportional_allocation` re-splits them. A type that is over-drawn is
shared among the diversions that take it, and each makes up the rest from its other
types, still in its own proportions. Formally it is the allocation closest to the
asked-for one (in relative entropy) that hits every target and over-draws nothing.
Two properties follow, and the tests hold the code to both:

- It continues the proportional split. At the point where a type first runs out, the
  re-split equals the proportional one, so moving a slider across that point doesn't
  jump.
- Raising any of the three never diverts less of any type. With incineration burning
  a fixed share of what is left, raising any slider never landfills more of any type.

The previous solver here split over-drawn waste by min-cost max-flow, steering each
diversion to the types fewest diversions take. Composting took all the food and green
waste first and recycling took plastic, metal and glass, so paper was landfilled
instead. Raising Navi Mumbai's recycling from 0 to 5% raised its 2040 methane 24%.
And only that solver burned what the others left; a split that fit burned the
published mix, so a scenario jumped as a slider crossed into the solver.

The exact max-flow test below still decides whether a request fits at all, and
supplies the numbers in the error messages (`max_feasible_target`).
"""

from __future__ import annotations

import math
from collections import deque
from fractions import Fraction

import numpy as np

from SWEET_python.constants import DIVERSION_ELIGIBILITY

# constants.DIVERSION_ELIGIBILITY, which City.div_components copies, as plain
# sets. Callers should pass their own `eligibility` (e.g. self.div_components);
# this is the default / drift reference for tests.
DEFAULT_ELIGIBILITY = {
    pathway: set(types) for pathway, types in DIVERSION_ELIGIBILITY.items()
}
TREATMENTS = ("compost", "anaerobic", "combustion", "recycling")
NON_COMBUSTION = ("compost", "anaerobic", "recycling")

# 1e9 integer grid: feasibility exact to ~1e-9 (far below slider/display
# precision) while exactly-fitting boundary cases (e.g. divert 100% of a fully
# divertible city) stay feasible. Verified against an exact rational oracle.
SCALE = 10 ** 9

# The re-split stops when no type is over-drawn, and no type it held back is left
# unused, by more than this share of the city's waste.
RESPLIT_TOLERANCE = 1e-12
# A request set exactly at a slider's maximum can fit only on the 1e-9 grid, or need
# some diversion to take none of a type its mix names; the re-split then closes in
# without arriving. After this many steps, a result over-drawing no type by more than
# ACCEPT_TOLERANCE is used (the mass check allows 1e-5), else the max-flow allocation.
RESPLIT_MAX_STEPS = 100
ACCEPT_TOLERANCE = 1e-8
# A cities table built with the solver this replaced can publish a mix that leaves
# out a type the diversion takes: Coimbatore's composting is 100% food, though the
# city has paper. Such a type keeps this weight, relative to the named ones, so the
# diversion turns to it only once those run out, and a slider reaches the same
# maximum as before.
LEFT_OUT_WEIGHT = 1e-6


class _MinCostMaxFlow:
    def __init__(self, n: int):
        self.n = n
        self.g: list[list[list]] = [[] for _ in range(n)]  # [to, cap, cost, rev]

    def add_edge(self, u: int, v: int, cap: int, cost: int) -> None:
        self.g[u].append([v, cap, cost, len(self.g[v])])
        self.g[v].append([u, 0, -cost, len(self.g[u]) - 1])

    def run(self, s: int, t: int) -> int:
        INF = float("inf")
        total_flow = 0
        while True:
            dist = [INF] * self.n
            in_q = [False] * self.n
            pv = [-1] * self.n
            pe = [-1] * self.n
            dist[s] = 0
            dq = deque([s])
            in_q[s] = True
            while dq:
                u = dq.popleft()
                in_q[u] = False
                du = dist[u]
                for i, e in enumerate(self.g[u]):
                    v, cap, cost, _ = e
                    if cap > 0 and du + cost < dist[v]:
                        dist[v] = du + cost
                        pv[v], pe[v] = u, i
                        if not in_q[v]:
                            dq.append(v)
                            in_q[v] = True
            if dist[t] == INF:
                break
            push = INF
            v = t
            while v != s:
                push = min(push, self.g[pv[v]][pe[v]][1])
                v = pv[v]
            v = t
            while v != s:
                e = self.g[pv[v]][pe[v]]
                e[1] -= push
                self.g[e[0]][e[3]][1] += push
                v = pv[v]
            total_flow += push
        return total_flow


def solve_allocation(waste: dict, targets: dict, eligibility: dict = None,
                     treatments: tuple = NON_COMBUSTION,
                     scale: int = SCALE) -> dict:
    """
    Decide exactly whether the targets fit the waste, by max-flow.

    Args:
        waste: {waste_type: fraction_of_total} availability.
        targets: {treatment: fraction_of_total} desired per treatment.
        eligibility: {treatment: set(waste_types)} (defaults to DEFAULT_ELIGIBILITY).
        treatments: which treatments to allocate (default the 3 non-combustion).

    Returns dict: {feasible, allocation {t: {w: fraction}}, flow, need, shortfall}.
    Guarantees, when feasible: sum_w alloc[t][w] == targets[t] (to ~1/scale) and
    sum_t alloc[t][w] <= waste[w]. The allocation is one feasible split, not a
    proportional one: the model uses `proportional_allocation`, which falls back on
    this split only when its own re-split fails to converge.
    """
    if eligibility is None:
        eligibility = DEFAULT_ELIGIBILITY
    tlist = [t for t in treatments]
    types = list(waste.keys())

    T, W = len(tlist), len(types)
    S, SINK = 0, 1 + T + W
    mc = _MinCostMaxFlow(SINK + 1)

    # Symmetric nearest rounding onto the integer grid (keeps exact-fit feasible).
    avail_i = {w: int(round(Fraction(waste.get(w, 0.0)) * scale)) for w in types}
    tgt_i = {t: int(round(Fraction(targets.get(t, 0.0)) * scale)) for t in tlist}

    for i, t in enumerate(tlist):
        mc.add_edge(S, 1 + i, tgt_i[t], 0)
    for i, t in enumerate(tlist):
        for j, w in enumerate(types):
            if w in eligibility.get(t, ()) and avail_i[w] > 0:
                mc.add_edge(1 + i, 1 + T + j, avail_i[w], 0)
    for j, w in enumerate(types):
        mc.add_edge(1 + T + j, SINK, avail_i[w], 0)

    need = sum(tgt_i.values())
    flow = mc.run(S, SINK)
    feasible = (flow == need)

    allocation = {t: {} for t in tlist}
    for i, t in enumerate(tlist):
        for e in mc.g[1 + i]:
            v, cap, _c, _r = e
            if 1 + T <= v < 1 + T + W:
                w = types[v - (1 + T)]
                used = avail_i[w] - cap
                if used > 0:
                    allocation[t][w] = used / scale

    return {"feasible": feasible, "allocation": allocation,
            "flow": flow / scale, "need": need / scale,
            "shortfall": (need - flow) / scale}


def max_feasible_target(treatment: str, waste: dict, other_targets: dict,
                        eligibility: dict = None,
                        treatments: tuple = NON_COMBUSTION,
                        scale: int = SCALE) -> float:
    """Largest value `treatment`'s slider can reach with the others fixed
    (binary search on the grid using solve_allocation as the feasibility test).
    For actionable error messages / the future clamp-to-max UX."""
    lo, hi = 0, scale
    while lo < hi:
        mid = (lo + hi + 1) // 2
        tg = dict(other_targets)
        tg[treatment] = mid / scale
        if solve_allocation(waste, tg, eligibility, treatments, scale=scale)["feasible"]:
            lo = mid
        else:
            hi = mid - 1
    return lo / scale


def _positive(value) -> bool:
    """A usable weight: a number above zero (None and NaN are not)."""
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def treatment_mixes(waste: dict, mixes: dict = None, eligibility: dict = None,
                    treatments: tuple = TREATMENTS) -> dict:
    """
    The proportions each treatment draws its types in, summing to 1.

    A treatment keeps the weights `mixes` gives it, on the types it is eligible for
    and the city has. A treatment with no such weight takes the waste mix of its
    eligible types. An eligible type the given mix leaves out gets LEFT_OUT_WEIGHT of
    the mix, shared by waste share, so it is used only once the named types run out.

    Returns {treatment: {waste_type: weight}}; a treatment with nothing it can take
    maps to {}.
    """
    if eligibility is None:
        eligibility = DEFAULT_ELIGIBILITY
    mixes = mixes or {}
    out = {}
    for t in treatments:
        eligible = [w for w in waste if w in eligibility.get(t, ()) and _positive(waste[w])]
        if not eligible:
            out[t] = {}
            continue
        given = mixes.get(t) or {}
        named = {w: float(given[w]) for w in eligible if _positive(given.get(w))}
        if not named:
            named = {w: float(waste[w]) for w in eligible}
        total = sum(named.values())
        weights = {w: v / total for w, v in named.items()}
        left_out = [w for w in eligible if w not in weights]
        if left_out:
            share = sum(float(waste[w]) for w in left_out)
            weights = {w: v * (1 - LEFT_OUT_WEIGHT) for w, v in weights.items()}
            weights.update({w: LEFT_OUT_WEIGHT * float(waste[w]) / share for w in left_out})
        out[t] = weights
    return out


def left_after(waste: dict, taken: dict) -> dict:
    """What composting, digestion and recycling leave of each type, as fractions of
    the city's waste (never below 0). `taken` is {treatment: {waste_type: fraction}}.

    Every caller goes through here, in the same order, so a city DST baseline and its
    no-change scenario burn exactly the same."""
    left = {}
    for w, available in waste.items():
        used = sum(_number(taken.get(t, {}).get(w)) for t in NON_COMBUSTION)
        left[w] = max(_number(available) - used, 0.0)
    return left


def _number(value) -> float:
    """A float, with a blank (None or NaN) as 0: a table cell left empty."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


# decision: diversion-split
def incineration_mix(waste: dict, taken: dict) -> dict:
    """What incineration burns, as shares summing to 1: what composting, digestion and
    recycling leave (`left_after`), in its own proportions. {} when nothing is left."""
    left = left_after(waste, taken)
    total = sum(left.values())
    if total <= 0:
        return {}
    return {w: v / total for w, v in left.items() if v > 0}


# decision: diversion-split
def proportional_allocation(waste: dict, targets: dict, mixes: dict = None,
                            eligibility: dict = None,
                            treatments: tuple = NON_COMBUSTION) -> dict:
    """
    Split each treatment's target across its types, keeping its own mix where the
    waste allows (see the module docstring).

    Args:
        waste: {waste_type: fraction_of_total} availability.
        targets: {treatment: fraction_of_total} desired per treatment.
        mixes: {treatment: {waste_type: weight}}, each treatment's own proportions
            (e.g. a city's published component fractions). Missing: the waste mix.
        eligibility: {treatment: set(waste_types)} (defaults to DEFAULT_ELIGIBILITY).
        treatments: the treatments to allocate (default composting, digestion and
            recycling; incineration burns what they leave, `incineration_mix`).

    Returns dict: {feasible, allocation {t: {w: fraction_of_total}}, support
    {t: set(waste_types)}, converged}. `support` is the types each treatment can take
    that the city has; feasibility is decided on it, so pass it to
    `max_feasible_target` for an error message. When feasible, sum_w alloc[t][w] ==
    targets[t] and sum_t alloc[t][w] <= waste[w], each to within ACCEPT_TOLERANCE.
    """
    if eligibility is None:
        eligibility = DEFAULT_ELIGIBILITY
    weights = treatment_mixes(waste, mixes, eligibility, treatments)
    support = {t: set(weights[t]) for t in treatments}
    asked = {t: float(targets.get(t, 0.0) or 0.0) for t in treatments}
    active = [t for t in treatments if asked[t] > 0]

    exact = solve_allocation(waste, asked, support, treatments)
    # Short by no more than rounding still fits: diverting all of a city's waste can
    # come out a grid step short (Kitakyushu's incineration plus recycling, 100%).
    if not exact["feasible"] and exact["shortfall"] > ACCEPT_TOLERANCE:
        return {"feasible": False, "allocation": {t: {} for t in treatments},
                "support": support, "converged": False}
    if not active:
        return {"feasible": True, "allocation": {t: {} for t in treatments},
                "support": support, "converged": True}

    types = [w for w in waste if any(w in weights[t] for t in active)]
    split, worst = _resplit(
        np.array([[weights[t].get(w, 0.0) for w in types] for t in active]),
        np.array([asked[t] for t in active]),
        np.array([float(waste[w]) for w in types]),
    )
    converged = worst <= ACCEPT_TOLERANCE
    if converged:
        allocation = {t: {} for t in treatments}
        for i, t in enumerate(active):
            allocation[t] = {w: float(split[i, j]) for j, w in enumerate(types) if split[i, j] > 0}
    else:
        allocation = exact["allocation"]
    return {"feasible": True, "allocation": allocation, "support": support,
            "converged": converged}


def _resplit(mix: np.ndarray, asked: np.ndarray, waste: np.ndarray):
    """The allocation closest to `asked * mix` that over-draws no type.

    `mix` is treatments x types, each row summing to 1; `waste` is each type's share.
    Each type w gets a hold-back v[w] >= 0, and treatment t takes
        x[t, w] = asked[t] * mix[t, w] * exp(-v[w]) / sum_u mix[t, u] * exp(-v[u]),
    so every row is its target. A type with v > 0 is over-drawn at v = 0 and is used
    up exactly; one with v = 0 is not over-drawn. These v minimise the convex
        psi(v) = sum_t asked[t] * log(sum_w mix[t, w] exp(-v[w])) + sum_w waste[w] v[w],
    whose gradient is each type's waste minus what is taken of it. Newton's method on
    psi (projected onto v >= 0, regularised where psi is flat, with a line search) gets
    there in a few steps even where one diversion must take nearly all of a type.

    Returns (x, worst): the allocation, and the largest over-draw or unused hold-back.
    """
    v = np.zeros(len(waste))

    def evaluate(v):
        held = np.exp(-v)
        totals = mix @ held
        shares = mix * held / totals[:, None]
        x = asked[:, None] * shares
        taken = x.sum(axis=0)
        psi = float(asked @ np.log(totals) + waste @ v)
        return psi, waste - taken, x, shares, taken

    psi, grad, x, shares, taken = evaluate(v)
    for _ in range(RESPLIT_MAX_STEPS):
        # How far from the optimum: over-drawn types, and held-back types left unused.
        projected = np.where(v > 0, grad, np.minimum(grad, 0.0))
        if np.max(np.abs(projected)) <= RESPLIT_TOLERANCE:
            break
        free = ~((v <= 0) & (grad > 0))
        hessian = np.diag(taken) - shares.T @ (asked[:, None] * shares)
        g = grad[free]
        step = np.zeros(len(waste))
        try:
            step[free] = np.linalg.solve(
                hessian[np.ix_(free, free)] + np.linalg.norm(g) * np.eye(len(g)), -g
            )
        except np.linalg.LinAlgError:
            break  # the caller falls back on the max-flow split
        alpha = 1.0
        for _ in range(60):
            trial = np.minimum(np.maximum(v + alpha * step, 0.0), 700.0)
            candidate = evaluate(trial)
            if candidate[0] <= psi + 1e-4 * grad @ (trial - v):
                break
            alpha *= 0.5
        else:
            break
        v = trial
        psi, grad, x, shares, taken = candidate
    worst = float(np.max(np.where(v > 0, np.abs(grad), np.maximum(-grad, 0.0))))
    return x, worst
