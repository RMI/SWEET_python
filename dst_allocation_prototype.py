"""
Prototype: robust DST diversion allocation via min-cost max-flow.

WHY THIS EXISTS
---------------
`City.mass_checker_math` (SWEET_python/city_params.py) tries to split user-
requested diversion fractions across waste types using a hand-rolled iterative
redistribution loop. That loop is a local heuristic on what is really a
*transportation / assignment problem*, and it:
  - FALSELY REJECTS feasible combinations (the documented bug);
  - SILENTLY ACCEPTS some impossible ones, diverting 0 tons (dead guard at
    city_params.py:5772-5783, after the early return at 5732);
  - CRASHES with a bare AssertionError at city_params.py:6144.

This file prototypes the correct approach as a standalone, dependency-free
module (plain `python3`, no scipy/networkx/SWEET imports), so we can prove
correctness in isolation before touching city_params.py.

DESIGN
------
The 3 "non-combustion" treatments (compost, anaerobic, recycling) contend over
wood + paper_cardboard. We allocate them with min-cost max-flow:
    sum_w x[t][w] == target[t]            (hit every slider)
    sum_t x[t][w] <= waste[w]             (don't over-draw any type)
    x[t][w] == 0 unless w eligible for t
Combustion is NOT a free consumer: the real model burns a *uniform* fraction of
whatever combustible mass is left (city_params.py:6128-6131). So it stays a
separate post-step: feasible iff target_combustion <= achievable remainder.
To maximize that remainder when combustion > 0, the flow uses a two-tier cost
that steers recycling onto NON-combustible types (metal/glass/other) first.

VERIFICATION (run: python3 dst_allocation_prototype.py)
  - solver vs an INDEPENDENT exact (rational) Hall/Gale oracle: 0 disagreements
  - per-feasible-case allocation validity incl. aggregate-mass check
  - combustion remainder vs an independent closed-form max-remainder oracle

Eligibility / waste types mirror SWEET_python/city_params.py:195 (source of
truth). Keep them in sync if that ever changes.
"""

from __future__ import annotations

import itertools
import random
from collections import deque
from fractions import Fraction

# --- mirror of city_params.py:195 (div_components) + waste_types -------------
WASTE_TYPES = [
    "food", "green", "wood", "paper_cardboard", "textiles",
    "plastic", "metal", "glass", "rubber", "other",
]
ELIGIBILITY = {
    "compost":    {"food", "green", "wood", "paper_cardboard"},
    "anaerobic":  {"food", "green", "wood", "paper_cardboard"},
    "combustion": {"food", "green", "wood", "paper_cardboard",
                   "textiles", "plastic", "rubber"},
    "recycling":  {"wood", "paper_cardboard", "textiles", "plastic",
                   "rubber", "metal", "glass", "other"},
}
NON_COMBUSTION = ("compost", "anaerobic", "recycling")
COMBUSTIBLE = ELIGIBILITY["combustion"]            # types combustion can take
NON_COMBUSTIBLE = set(WASTE_TYPES) - COMBUSTIBLE    # metal, glass, other

# 1e9 grid: feasibility decided to ~1e-9, far below slider/display precision.
SCALE = 10 ** 9
# Two-tier cost: any combustible edge costs more than any non-combustible edge,
# so when combustion > 0 the 3 treatments spare combustibles. Degree breaks ties
# within each tier (protects contended wood/paper). Max degree is 3 << penalty.
COMBUSTIBLE_PENALTY = 1000
TOL = 1e-7


# ---------------------------------------------------------------------------
# 1. Generic integer min-cost max-flow (successive shortest paths / SPFA)
# ---------------------------------------------------------------------------
class MinCostMaxFlow:
    def __init__(self, n: int):
        self.n = n
        self.g: list[list[list]] = [[] for _ in range(n)]  # [to,cap,cost,rev]

    def add_edge(self, u: int, v: int, cap: int, cost: int) -> None:
        self.g[u].append([v, cap, cost, len(self.g[v])])
        self.g[v].append([u, 0, -cost, len(self.g[u]) - 1])

    def run(self, s: int, t: int) -> tuple[int, int]:
        INF = float("inf")
        total_flow = total_cost = 0
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
            d = INF
            v = t
            while v != s:
                d = min(d, self.g[pv[v]][pe[v]][1])
                v = pv[v]
            v = t
            while v != s:
                e = self.g[pv[v]][pe[v]]
                e[1] -= d
                self.g[e[0]][e[3]][1] += d
                v = pv[v]
            total_flow += d
            total_cost += d * dist[t]
        return total_flow, total_cost


# ---------------------------------------------------------------------------
# 2. Non-combustion allocation solver (compost / anaerobic / recycling)
# ---------------------------------------------------------------------------
def solve_allocation(waste: dict, targets: dict,
                     treatments: tuple = NON_COMBUSTION,
                     spare_combustibles: bool = False,
                     scale: int = SCALE) -> dict:
    """Min-cost max-flow allocation. `spare_combustibles` raises the cost of
    combustible edges so the leftover combustible remainder is maximized (only
    matters when a combustion slider is in play)."""
    tlist = list(treatments)
    T, W = len(tlist), len(WASTE_TYPES)
    S, SINK = 0, 1 + T + W
    mc = MinCostMaxFlow(SINK + 1)

    # Symmetric nearest rounding onto the SCALE grid. At SCALE=1e9 the decision
    # is exact to ~1e-9 (verified against the exact rational oracle), and crucially
    # exactly-fitting boundary cases (demand == capacity, e.g. divert 100% of a
    # fully-divertible city) round identically and stay FEASIBLE -- which is the
    # whole point. The tiny possible over-allocation (<=0.5/SCALE/type) is far
    # below float noise and is caught in aggregate by allocation_is_valid.
    avail_i = {w: int(round(Fraction(waste.get(w, 0.0)) * scale)) for w in WASTE_TYPES}
    tgt_i = {t: int(round(Fraction(targets.get(t, 0.0)) * scale)) for t in tlist}

    degree = {w: sum(1 for t in tlist if w in ELIGIBILITY[t]) for w in WASTE_TYPES}

    def edge_cost(w):
        c = degree[w]
        if spare_combustibles and w in COMBUSTIBLE:
            c += COMBUSTIBLE_PENALTY
        return c

    for i, t in enumerate(tlist):
        mc.add_edge(S, 1 + i, tgt_i[t], 0)
    for i, t in enumerate(tlist):
        for j, w in enumerate(WASTE_TYPES):
            if w in ELIGIBILITY[t] and avail_i[w] > 0:
                mc.add_edge(1 + i, 1 + T + j, avail_i[w], edge_cost(w))
    for j, w in enumerate(WASTE_TYPES):
        mc.add_edge(1 + T + j, SINK, avail_i[w], 0)

    need = sum(tgt_i.values())
    flow, _ = mc.run(S, SINK)
    feasible = (flow == need)

    allocation = {t: {} for t in tlist}
    for i, t in enumerate(tlist):
        for e in mc.g[1 + i]:
            v, cap, _c, _r = e
            if 1 + T <= v < 1 + T + W:
                w = WASTE_TYPES[v - (1 + T)]
                used = avail_i[w] - cap
                if used > 0:
                    allocation[t][w] = used / scale
    return {"feasible": feasible, "allocation": allocation,
            "flow": flow / scale, "need": need / scale,
            "shortfall": (need - flow) / scale}


# ---------------------------------------------------------------------------
# 3. Full 4-slider solve: 3 via flow, combustion via leftover remainder
# ---------------------------------------------------------------------------
def solve_with_combustion(waste: dict, targets: dict) -> dict:
    """Feasibility + allocation for all four sliders.
    Returns {feasible, stage, allocation, remainder, max_remainder, reason}."""
    three = {t: targets.get(t, 0.0) for t in NON_COMBUSTION}
    tc = targets.get("combustion", 0.0)
    res3 = solve_allocation(waste, three, spare_combustibles=(tc > 0))
    if not res3["feasible"]:
        return {"feasible": False, "stage": "non_combustion",
                "allocation": res3["allocation"], "remainder": None,
                "reason": "compost/anaerobic/recycling targets exceed available "
                          "waste of their eligible types"}

    alloc = {t: dict(res3["allocation"][t]) for t in NON_COMBUSTION}
    leftover = {w: waste.get(w, 0.0) - sum(alloc[t].get(w, 0.0)
                                           for t in NON_COMBUSTION)
                for w in WASTE_TYPES}
    remainder = sum(leftover[w] for w in COMBUSTIBLE)

    if tc > remainder + TOL:
        return {"feasible": False, "stage": "combustion",
                "allocation": alloc, "remainder": remainder,
                "reason": f"combustion {tc:.4f} exceeds leftover combustible "
                          f"mass {remainder:.4f}"}

    frac = (tc / remainder) if remainder > TOL else 0.0
    alloc["combustion"] = {w: frac * leftover[w] for w in COMBUSTIBLE
                           if leftover[w] > 0}
    return {"feasible": True, "stage": "ok", "allocation": alloc,
            "remainder": remainder, "reason": "ok"}


def max_feasible_target(treatment: str, waste: dict, other_targets: dict,
                        treatments: tuple = NON_COMBUSTION) -> float:
    """Largest value `treatment`'s slider can reach with others fixed."""
    lo, hi = 0, SCALE
    while lo < hi:
        mid = (lo + hi + 1) // 2
        tg = dict(other_targets)
        tg[treatment] = mid / SCALE
        if solve_allocation(waste, tg, treatments)["feasible"]:
            lo = mid
        else:
            hi = mid - 1
    return lo / SCALE


# ---------------------------------------------------------------------------
# 4. INDEPENDENT oracles (no shared code / no quantization with the solver)
# ---------------------------------------------------------------------------
def hall_feasible_exact(waste: dict, targets: dict,
                        treatments: tuple = NON_COMBUSTION) -> bool:
    """Exact rational Gale/Hall subset condition. Different math from the flow
    solver AND no SCALE rounding, so it can observe the quantization band."""
    tlist = list(treatments)
    tgt = {t: Fraction(targets.get(t, 0.0)) for t in tlist}
    av = {w: Fraction(waste.get(w, 0.0)) for w in WASTE_TYPES}
    for r in range(1, len(tlist) + 1):
        for sub in itertools.combinations(tlist, r):
            demand = sum(tgt[t] for t in sub)
            reach = set().union(*(ELIGIBILITY[t] for t in sub))
            if demand > sum(av[w] for w in reach):
                return False
    return True


def max_remainder_exact(waste: dict, targets: dict) -> Fraction:
    """Closed-form max combustible remainder after the 3 (exact). The only
    non-combustible mass the 3 can absorb is recycling drawing metal/glass/other,
    up to min(t_recycling, metal+glass+other)."""
    av = {w: Fraction(waste.get(w, 0.0)) for w in WASTE_TYPES}
    total_comb = sum(av[w] for w in COMBUSTIBLE)
    sum3 = sum(Fraction(targets.get(t, 0.0)) for t in NON_COMBUSTION)
    mgo = sum(av[w] for w in NON_COMBUSTIBLE)
    absorbed_noncomb = min(Fraction(targets.get("recycling", 0.0)), mgo)
    min_comb_consumed = max(Fraction(0), sum3 - absorbed_noncomb)
    return total_comb - min_comb_consumed


def combustion_feasible_oracle(waste: dict, targets: dict) -> bool:
    if not hall_feasible_exact(waste, targets, NON_COMBUSTION):
        return False
    return Fraction(targets.get("combustion", 0.0)) <= max_remainder_exact(waste, targets)


# ---------------------------------------------------------------------------
# 5. Allocation validity (incl. the aggregate-mass check the fuzz needs)
# ---------------------------------------------------------------------------
def allocation_is_valid(waste: dict, targets: dict, allocation: dict,
                        treatments: tuple = NON_COMBUSTION,
                        tol: float = 1e-6) -> tuple[bool, str]:
    for t in treatments:
        got = sum(allocation.get(t, {}).values())
        if abs(got - targets.get(t, 0.0)) > tol:
            return False, f"target {t}: want {targets.get(t,0):.7f} got {got:.7f}"
    for t in treatments:
        for w, v in allocation.get(t, {}).items():
            if w not in ELIGIBILITY[t]:
                return False, f"{t} used ineligible {w}"
            if v < -tol:
                return False, f"{t}/{w} negative {v}"
    for w in WASTE_TYPES:
        used = sum(allocation.get(t, {}).get(w, 0.0) for t in treatments)
        if used > waste.get(w, 0.0) + tol:
            return False, f"over-drew {w}: {used:.7f} > {waste.get(w,0):.7f}"
    # aggregate: total drawn cannot exceed total waste (catches stacked rounding)
    total_used = sum(sum(allocation.get(t, {}).values()) for t in treatments)
    if total_used > sum(waste.values()) + tol:
        return False, f"aggregate over-draw {total_used:.7f} > {sum(waste.values()):.7f}"
    return True, "ok"


# ---------------------------------------------------------------------------
# 6. Curated scenarios + audit regression suite
# ---------------------------------------------------------------------------
def wf(**kw):
    return {w: kw.get(w, 0.0) for w in WASTE_TYPES}


# (name, waste, targets[no combustion], expected_feasible, note)
SCENARIOS = [
    ("REPRO compost40+recycling50 (100% divertible)",
     wf(food=0.10, paper_cardboard=0.45, plastic=0.45),
     {"compost": 0.40, "recycling": 0.50}, True, "old code raised"),
    ("REPRO at ceiling compost40+recycling60",
     wf(food=0.10, paper_cardboard=0.45, plastic=0.45),
     {"compost": 0.40, "recycling": 0.60}, True, "exactly fits"),
    ("REPRO over ceiling compost40+recycling62",
     wf(food=0.10, paper_cardboard=0.45, plastic=0.45),
     {"compost": 0.40, "recycling": 0.62}, False, "only paper.15 free"),
    ("all food compost100", wf(food=1.0), {"compost": 1.0}, True, ""),
    ("all food recycling1 (impossible)", wf(food=1.0), {"recycling": 0.01},
     False, "food not recyclable"),
    ("all plastic recycling100", wf(plastic=1.0), {"recycling": 1.0}, True, ""),
    ("all plastic compost1 (impossible)", wf(plastic=1.0), {"compost": 0.01},
     False, "plastic not compostable"),
    ("only paper compost60+recycling40", wf(paper_cardboard=1.0),
     {"compost": 0.60, "recycling": 0.40}, True, "shared, sums to 1"),
    ("only paper compost60+recycling50", wf(paper_cardboard=1.0),
     {"compost": 0.60, "recycling": 0.50}, False, "1.1>1"),
    ("food-heavy compost50 recycling45",
     wf(food=0.55, paper_cardboard=0.20, plastic=0.15, metal=0.10),
     {"compost": 0.50, "recycling": 0.45}, True, ""),
    ("food-heavy infeasible compost60 recycling45",
     wf(food=0.55, paper_cardboard=0.20, plastic=0.15, metal=0.10),
     {"compost": 0.60, "recycling": 0.45}, False, "1.05>1"),
    ("compost+anaerobic over organics each35",
     wf(food=0.50, paper_cardboard=0.10, plastic=0.40),
     {"compost": 0.35, "anaerobic": 0.35}, False, "0.70>organics 0.60"),
    ("all three on mixed", wf(food=0.30, green=0.10, wood=0.10,
     paper_cardboard=0.10, plastic=0.20, metal=0.10, glass=0.10),
     {"compost": 0.30, "anaerobic": 0.20, "recycling": 0.40}, True, ""),
    # --- audit regression: empty-pool / silent-drop (legacy wrongly accepted) -
    ("AUDIT empty-pool recycling0.01 on food",
     wf(food=1.0), {"recycling": 0.01}, False, "silent-drop guard"),
    ("AUDIT sub-tolerance compost0.0003 on textiles",
     wf(textiles=1.0), {"compost": 0.0003}, False, "sub-tol must reject"),
    ("AUDIT metal compost0.01 (ineligible)", wf(metal=1.0),
     {"compost": 0.01}, False, ""),
    # --- audit regression: legacy AssertionError(6144) inputs, must solve ----
    ("AUDIT legacy-crash A",
     wf(wood=0.18, paper_cardboard=0.064, textiles=0.284, glass=0.369, rubber=0.104),
     {"anaerobic": 0.1979, "recycling": 0.4188}, True, "legacy asserted"),
    ("AUDIT legacy-crash B",
     wf(paper_cardboard=0.433, plastic=0.385, glass=0.182),
     {"compost": 0.3585, "recycling": 0.3527}, True, "legacy asserted"),
    # --- shared-pool one-grid-step boundary ---------------------------------
    ("AUDIT one-step-over shared pool", wf(wood=1.0),
     {"compost": 0.5, "recycling": 0.500001}, False, "1 grid step over"),
    ("AUDIT one-step-under shared pool", wf(wood=1.0),
     {"compost": 0.5, "recycling": 0.499999}, True, ""),
    # --- numeric / float dust -----------------------------------------------
    ("AUDIT float dust 0.1+0.2", wf(paper_cardboard=1.0),
     {"compost": 0.1 + 0.2}, True, "0.30000000000000004"),
    ("AUDIT does-not-sum-to-one 0.97", wf(food=0.55, paper_cardboard=0.20,
     plastic=0.22), {"recycling": 0.42}, True, "recyclables .42"),
    ("AUDIT does-not-sum-to-one 0.97 over", wf(food=0.55, paper_cardboard=0.20,
     plastic=0.22), {"recycling": 0.43}, False, "recyclables only .42"),
]

# (name, waste, targets WITH combustion, expected_feasible, note)
COMBUSTION_SCENARIOS = [
    ("C4 combustible-sparing (THE coupling case)",
     wf(food=0.30, paper_cardboard=0.30, metal=0.30, plastic=0.10),
     {"recycling": 0.30, "combustion": 0.55}, True,
     "feasible only if recycling takes metal -> remainder 0.70"),
    ("C5 combustion just over",
     wf(food=0.30, paper_cardboard=0.30, metal=0.30, plastic=0.10),
     {"recycling": 0.30, "combustion": 0.71}, False, "0.71>max_remainder 0.70"),
    ("C6 organics-eaten ok",
     wf(food=0.50, paper_cardboard=0.20, plastic=0.30),
     {"compost": 0.40, "combustion": 0.55}, True, ""),
    ("C7 organics-eaten over",
     wf(food=0.50, paper_cardboard=0.20, plastic=0.30),
     {"compost": 0.40, "combustion": 0.65}, False, ""),
    ("C9 max-remainder (spare paper via metal)",
     wf(paper_cardboard=0.60, metal=0.40),
     {"compost": 0.30, "recycling": 0.30, "combustion": 0.01}, True, ""),
    ("C10 recycling starves combustion (no metal/glass/other)",
     wf(food=0.20, paper_cardboard=0.40, plastic=0.40),
     {"compost": 0.20, "recycling": 0.40, "combustion": 0.45}, False,
     "max_remainder 0.40 < 0.45"),
    ("C-divzero guard (no combustible left)",
     wf(paper_cardboard=1.0),
     {"compost": 0.60, "recycling": 0.40, "combustion": 0.05}, False,
     "remainder 0 -> infeasible, must not ZeroDivisionError"),
    ("C-divzero combustion0 ok", wf(paper_cardboard=1.0),
     {"compost": 0.60, "recycling": 0.40, "combustion": 0.0}, True, ""),
    ("C combustion on non-combustible only", wf(metal=1.0),
     {"combustion": 0.10}, False, "remainder 0"),
    ("C full 4-slider tight boundary",
     wf(food=0.25, green=0.10, wood=0.05, paper_cardboard=0.15, plastic=0.15,
        metal=0.10, glass=0.10, textiles=0.05),
     {"compost": 0.30, "recycling": 0.25, "combustion": 0.40}, True,
     "t_comb == max_remainder"),
]


# ---------------------------------------------------------------------------
# 7. Runners + fuzz
# ---------------------------------------------------------------------------
def run_curated() -> list:
    print("=" * 78)
    print("NON-COMBUSTION SCENARIOS (solver vs exact oracle vs expectation)")
    print("=" * 78)
    tricky = []
    for name, waste, targets, expected, note in SCENARIOS:
        res = solve_allocation(waste, targets)
        orc = hall_feasible_exact(waste, targets)
        valid, why = (allocation_is_valid(waste, targets, res["allocation"])
                      if res["feasible"] else (True, "n/a"))
        ok = (res["feasible"] == orc == expected) and valid
        print(f"[{'OK ' if ok else '!!!'}] {name}")
        if not ok:
            print(f"      solver={res['feasible']} oracle={orc} "
                  f"expected={expected} valid={valid} ({why}) :: {note}")
            tricky.append((name, waste, targets, expected, res, orc))
    return tricky


def run_combustion() -> list:
    print()
    print("=" * 78)
    print("COMBUSTION SCENARIOS (solve_with_combustion vs closed-form oracle)")
    print("=" * 78)
    tricky = []
    for name, waste, targets, expected, note in COMBUSTION_SCENARIOS:
        try:
            res = solve_with_combustion(waste, targets)
            feas = res["feasible"]
            crashed = None
        except Exception as e:
            feas, crashed = None, f"{type(e).__name__}: {e}"
        orc = combustion_feasible_oracle(waste, targets)
        valid, why = (allocation_is_valid(waste, targets, res["allocation"], NON_COMBUSTION)
                      if (crashed is None and feas) else (True, "n/a"))
        ok = (crashed is None) and (feas == orc == expected) and valid
        print(f"[{'OK ' if ok else '!!!'}] {name}")
        if not ok:
            print(f"      solver={feas} oracle={orc} expected={expected} "
                  f"valid={valid} crash={crashed} ({why}) :: {note}")
            tricky.append((name, waste, targets, expected))
    return tricky


def run_repro_sweep() -> None:
    print()
    print("=" * 78)
    print("REPRO SWEEP")
    print("=" * 78)
    waste = wf(food=0.10, paper_cardboard=0.45, plastic=0.45)
    accepted = [r for r in range(0, 64, 2)
                if solve_allocation(waste, {"compost": 0.40, "recycling": r / 100})["feasible"]]
    print(f"waste food.10/paper.45/plastic.45, compost40%: new solver accepts "
          f"recycling {accepted[0]}%..{accepted[-1]}% (old code: 0..32%)")
    print(f"max_feasible_target(recycling|compost40) = "
          f"{max_feasible_target('recycling', waste, {'compost': 0.40})*100:.4f}%")


def random_waste(rng):
    k = rng.choice([1, 2, 3, 3, 4, 5, 7, 10])
    ts = rng.sample(WASTE_TYPES, k)
    raw = [rng.random() for _ in ts]
    s = sum(raw) or 1.0
    return wf(**{t: r / s for t, r in zip(ts, raw)})


def fuzz(n=200_000, seed=12345):
    print()
    print("=" * 78)
    print(f"FUZZ {n:,} cases (solver vs EXACT rational oracle; band-aware)")
    print("=" * 78)
    rng = random.Random(seed)
    mis, invalid, band = [], [], 0
    for _ in range(n):
        waste = random_waste(rng)
        targets = {t: round(rng.random() * rng.choice([0.3, 0.6, 1.0]), 4)
                   for t in NON_COMBUSTION if rng.random() < 0.75}
        res = solve_allocation(waste, targets)
        orc = hall_feasible_exact(waste, targets)
        if res["feasible"] != orc:
            # disagreement only acceptable within one grid step of a boundary
            slack = _boundary_slack(waste, targets)
            if slack <= Fraction(1, SCALE):
                band += 1
            else:
                mis.append((waste, targets, res["feasible"], orc, float(slack)))
        elif res["feasible"]:
            okv, why = allocation_is_valid(waste, targets, res["allocation"])
            if not okv:
                invalid.append((waste, targets, why))
    print(f"genuine feasibility mismatches : {len(mis)}")
    print(f"within-grid-band disagreements : {band}  (expected, |slack|<=1/SCALE)")
    print(f"invalid allocations            : {len(invalid)}")
    for waste, targets, sv, ov, sl in mis[:3]:
        nz = {k: round(v, 5) for k, v in waste.items() if v}
        print(f"   MISMATCH waste={nz} t={targets} solver={sv} oracle={ov} slack={sl:.2e}")
    for waste, targets, why in invalid[:3]:
        print(f"   INVALID {why}")
    return mis + invalid


def fuzz_combustion(n=200_000, seed=999):
    print()
    print("=" * 78)
    print(f"FUZZ COMBUSTION {n:,} cases (solve_with_combustion vs oracle)")
    print("=" * 78)
    rng = random.Random(seed)
    mis, invalid, crashes, band = [], [], [], 0
    for _ in range(n):
        waste = random_waste(rng)
        targets = {t: round(rng.random() * rng.choice([0.3, 0.6, 1.0]), 4)
                   for t in ("compost", "anaerobic", "recycling", "combustion")
                   if rng.random() < 0.7}
        try:
            res = solve_with_combustion(waste, targets)
        except Exception as e:
            crashes.append((waste, targets, f"{type(e).__name__}: {e}"))
            continue
        orc = combustion_feasible_oracle(waste, targets)
        if res["feasible"] != orc:
            slack = _boundary_slack_comb(waste, targets)
            if slack <= Fraction(2, SCALE):
                band += 1
            else:
                mis.append((waste, targets, res["feasible"], orc, float(slack)))
        elif res["feasible"]:
            okv, why = allocation_is_valid(waste, targets, res["allocation"], NON_COMBUSTION)
            if not okv:
                invalid.append((waste, targets, why))
    print(f"genuine feasibility mismatches : {len(mis)}")
    print(f"within-grid-band disagreements : {band}")
    print(f"invalid allocations            : {len(invalid)}")
    print(f"crashes                        : {len(crashes)}")
    for waste, targets, sv, ov, sl in mis[:3]:
        nz = {k: round(v, 5) for k, v in waste.items() if v}
        print(f"   MISMATCH waste={nz} t={targets} solver={sv} oracle={ov} slack={sl:.2e}")
    for waste, targets, msg in crashes[:3]:
        print(f"   CRASH {msg}")
    return mis + invalid + crashes


def _boundary_slack(waste, targets, treatments=NON_COMBUSTION):
    """Smallest absolute Hall-constraint margin (how close to a boundary)."""
    tgt = {t: Fraction(targets.get(t, 0.0)) for t in treatments}
    av = {w: Fraction(waste.get(w, 0.0)) for w in WASTE_TYPES}
    best = None
    for r in range(1, len(treatments) + 1):
        for sub in itertools.combinations(treatments, r):
            demand = sum(tgt[t] for t in sub)
            cap = sum(av[w] for w in set().union(*(ELIGIBILITY[t] for t in sub)))
            m = abs(demand - cap)
            best = m if best is None else min(best, m)
    return best if best is not None else Fraction(1)


def _boundary_slack_comb(waste, targets):
    s1 = _boundary_slack(waste, targets, NON_COMBUSTION)
    s2 = abs(Fraction(targets.get("combustion", 0.0)) - max_remainder_exact(waste, targets))
    return min(s1, s2)


if __name__ == "__main__":
    assert {k: set(v) for k, v in ELIGIBILITY.items()}["compost"] == \
        {"food", "green", "wood", "paper_cardboard"}, "eligibility drift!"
    t1 = run_curated()
    t2 = run_combustion()
    run_repro_sweep()
    p1 = fuzz()
    p2 = fuzz_combustion()
    print()
    print("=" * 78)
    print(f"SUMMARY  noncomb_tricky={len(t1)} comb_tricky={len(t2)}  "
          f"fuzz_problems={len(p1)} comb_fuzz_problems={len(p2)}")
    print("=" * 78)
