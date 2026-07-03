"""
Exact diversion allocation for the city DST, via min-cost max-flow.

Splitting user diversion sliders (compost / anaerobic / recycling, each a
fraction of total waste) across waste types is a transportation/assignment
problem: each treatment accepts only some waste types, and no type can be
diverted more than it exists. The legacy hand-rolled redistribution in
`City.mass_checker_math` solved this with local moves and FALSELY REJECTED many
feasible slider combinations (and silently mis-handled / crashed on others).

This module solves it exactly with a small pure-Python min-cost max-flow
(no scipy / networkx). Feasibility is decided correctly for the entire true
feasible region; among feasible allocations a contention-aware cost keeps the
split sensible (and, when combustion is in play, spares combustible mass so the
leftover-remainder available to combustion is maximized).

Combustion itself is NOT allocated here: the DST model burns a uniform fraction
of whatever combustible mass is left after the other three treatments, so the
caller handles it as a remainder step (see City.mass_checker_math).

Validation harness (independent exact-rational oracle + ~400k-case fuzz) lives
at SWEET_python/dst_allocation_prototype.py.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction

# Mirrors City.div_components (SWEET_python/city_params.py:195). Callers should
# pass their own `eligibility` (e.g. self.div_components); this is the default /
# drift reference for tests.
DEFAULT_ELIGIBILITY = {
    "compost": {"food", "green", "wood", "paper_cardboard"},
    "anaerobic": {"food", "green", "wood", "paper_cardboard"},
    "combustion": {"food", "green", "wood", "paper_cardboard",
                   "textiles", "plastic", "rubber", "metal", "glass", "other"},
    "recycling": {"wood", "paper_cardboard", "textiles", "plastic",
                  "rubber", "metal", "glass", "other"},
}
NON_COMBUSTION = ("compost", "anaerobic", "recycling")

# 1e9 integer grid: feasibility exact to ~1e-9 (far below slider/display
# precision) while exactly-fitting boundary cases (e.g. divert 100% of a fully
# divertible city) stay feasible. Verified against an exact rational oracle.
SCALE = 10 ** 9
# Two-tier cost: any combustible edge costs more than any non-combustible edge,
# so when combustion > 0 the three treatments spare combustibles for it. The
# per-type contention degree breaks ties within a tier (protects shared
# wood/paper). Max degree (3) << penalty, so the tiers never cross.
# NOTE: with the current config every waste type is combustion-eligible, so this
# penalty is added uniformly to all edges and is a no-op -- there are no
# non-combustible types to steer the other treatments onto, and the combustion
# remainder (total - compost - anaerobic - recycling) is independent of the
# allocation. Kept for the general case (if a non-combustible type is reintroduced).
_COMBUSTIBLE_PENALTY = 1000


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
                     spare_combustibles: bool = False,
                     scale: int = SCALE) -> dict:
    """
    Allocate each treatment's target across eligible waste types.

    Args:
        waste: {waste_type: fraction_of_total} availability.
        targets: {treatment: fraction_of_total} desired per treatment.
        eligibility: {treatment: set(waste_types)} (defaults to DEFAULT_ELIGIBILITY).
        treatments: which treatments to allocate (default the 3 non-combustion).
        spare_combustibles: raise the cost of combustible edges so the leftover
            combustible remainder is maximized (use when a combustion slider > 0).

    Returns dict: {feasible, allocation {t: {w: fraction}}, flow, need, shortfall}.
    Guarantees, when feasible: sum_w alloc[t][w] == targets[t] (to ~1/scale) and
    sum_t alloc[t][w] <= waste[w].
    """
    if eligibility is None:
        eligibility = DEFAULT_ELIGIBILITY
    tlist = [t for t in treatments]
    types = list(waste.keys())
    combustible = eligibility.get("combustion", set())

    T, W = len(tlist), len(types)
    S, SINK = 0, 1 + T + W
    mc = _MinCostMaxFlow(SINK + 1)

    # Symmetric nearest rounding onto the integer grid (keeps exact-fit feasible).
    avail_i = {w: int(round(Fraction(waste.get(w, 0.0)) * scale)) for w in types}
    tgt_i = {t: int(round(Fraction(targets.get(t, 0.0)) * scale)) for t in tlist}
    degree = {w: sum(1 for t in tlist if w in eligibility[t]) for w in types}

    def edge_cost(w):
        c = degree[w]
        if spare_combustibles and w in combustible:
            c += _COMBUSTIBLE_PENALTY
        return c

    for i, t in enumerate(tlist):
        mc.add_edge(S, 1 + i, tgt_i[t], 0)
    for i, t in enumerate(tlist):
        for j, w in enumerate(types):
            if w in eligibility[t] and avail_i[w] > 0:
                mc.add_edge(1 + i, 1 + T + j, avail_i[w], edge_cost(w))
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
