"""What a city generated, from a composition surveyed after diversion.

SWEET reads a city's composition as the waste the city generates. It takes each
diversion's share of that waste out of it, then landfills what is left. Some surveys
weighed what reached the landfill or dump instead, after recycling, composting and
incineration had taken theirs. SWEET then takes those materials out a second time, and a
city that diverts more of a material than its surveyed waste still holds can't be
modelled at all: Seattle recycles 58% of its waste, and 46% of its surveyed composition
is material SWEET can recycle.

`composition_before_diversion` adds the diverted waste back. It undoes SWEET's own
diversion, which takes each pathway's share of the waste from the types that pathway can
take, in proportion to how much of each the city generates (`diversion-eligibility`).
Its answer is the composition that, once SWEET diverts from it, leaves the surveyed
composition behind. The city's tonnage is not touched.

That is the waste sent straight to disposal. SWEET then landfills each pathway's rejects
as well (`diversion-rejects`), as it does for every city: the recycling and compost
residue a survey of garbage trucks or bins does not see. So what SWEET landfills is the
survey plus those rejects; for Seattle, 51% of its waste against the survey's 42%.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np

from SWEET_python.constants import DIVERSION_ELIGIBILITY

#: The answer is final when adding each pathway's take back onto what was left moves no
#: share by more than this, which bounds the round trip's error.
TOLERANCE = 1e-12
#: Where some type is almost all diverted, rounding stops the answer short of TOLERANCE:
#: its share is known to about 1e-16 over the fraction of it left. An answer is still
#: used within this, a thousandth of what the city model's mass check allows (1e-5).
PRECISE_ENOUGH = 1e-8
#: Newton steps allowed. The slowest of 14,000 random mixes, up to 99.99999% diverted,
#: took 38.
MAX_STEPS = 200
#: Newton's decrement under which the cost no longer tells steps apart from rounding, so
#: they are taken whole.
NEAR = 1e-13
#: Below this share of the waste left undiverted there is too little to survey, and
#: rounding, not the survey, would decide the answer.
LEAST_LEFT = 1e-7


# decision: composition-after-diversion
def composition_before_diversion(
    surveyed: Mapping[str, float],
    diverted: Mapping[str, float],
    eligibility: Mapping[str, Collection[str]] = DIVERSION_ELIGIBILITY,
) -> dict[str, float]:
    """The composition a city generates, from the one surveyed after its diversion.

    Args:
        surveyed: each waste type's share of the waste left after diversion. Scaled to
            sum to 1.
        diverted: each pathway's share of all the waste the city generates, keyed like
            ``eligibility`` (``compost``, ``anaerobic``, ``combustion``, ``recycling``).
        eligibility: the waste types each pathway takes.

    Returns:
        Each waste type's share of the waste the city generates, keyed like ``surveyed``
        and summing to 1.

    Raises:
        ValueError: if the pathways divert all of the waste, or all but LEAST_LEFT of
            it, so none was left to survey; or a pathway diverts some of it, but the
            surveyed waste has none of the types that pathway takes, so nothing says
            which types it took.
        RuntimeError: if rounding keeps the answer further than PRECISE_ENOUGH from
            the round trip, which happens only where a type is almost all diverted
            (one in about 50 random mixes diverting 99.99999%, none below 99.99%).
    """
    total = sum(surveyed.values())
    if not total > 0:
        raise ValueError("The surveyed composition is empty.")
    left = {waste: share / total for waste, share in surveyed.items()}

    shares = {path: float(share) for path, share in diverted.items() if share > 0}
    taken = sum(shares.values())
    # Shares scaled down to 100% can add up to a hair under it (0.9999999999999999).
    if 1 - taken < LEAST_LEFT:
        raise ValueError(
            f"The diversion shares add up to {taken:.1%} of the waste, so none was "
            "left to survey."
        )
    pools = {path: [w for w in left if w in eligibility[path]] for path in shares}
    for path, types in pools.items():
        if sum(left[w] for w in types) <= 0:
            raise ValueError(
                f"{path} takes {shares[path]:.1%} of the waste, but the surveyed waste "
                f"has none of the types it takes ({', '.join(sorted(eligibility[path]))})."
            )

    # Each pathway takes the same fraction of every type it takes, its rate: its share
    # over the generated size of its pool. A type's generated share is then what was
    # left of it over what all its pathways' rates leave of it, kept*left / (1 - rates).
    # The rates that make every pathway's take come to its share are the minimum of a
    # strictly convex function, so there is one answer, and Newton's method finds it.
    if not shares:
        return dict(left)
    kept = 1 - taken
    paths = list(shares)
    types = list(left)
    weight = np.array([kept * left[w] for w in types])
    share = np.array([shares[p] for p in paths])
    takes = np.array([[w in eligibility[p] for w in types] for p in paths], dtype=float)
    present = weight > 0

    def room(rates):
        return np.where(present, 1 - takes.T @ rates, 1.0)

    def cost(rates):
        if (rates <= 0).any() or (room(rates)[present] <= 0).any():
            return np.inf
        return -(weight[present] @ np.log(room(rates)[present])) - share @ np.log(rates)

    def answer(rates):
        # Add what each pathway takes back onto what was left. This needs no
        # subtraction, so it keeps its precision where almost all of a type is diverted.
        generated = weight / room(rates)
        pool_sizes = takes @ generated
        adjusted = kept * np.array([left[w] for w in types]) + generated * (
            takes.T @ (share / pool_sizes)
        )
        return adjusted, np.abs(adjusted - generated).max()

    # Each pathway's share as its rate keeps every type's rates under 1: they sum to
    # less than all the diversion. A step is shortened until it stays in bounds and,
    # far from the answer, lowers the cost; near it, whole steps converge quadratically
    # until rounding stops them improving.
    rates = share.copy()
    best, best_error, stalled = None, np.inf, 0
    for _ in range(MAX_STEPS):
        generated, error = answer(rates)
        gradient = takes @ (weight / room(rates)) - share / rates
        hessian = (takes * (weight / room(rates) ** 2)) @ takes.T + np.diag(share / rates**2)
        step = np.linalg.solve(hessian, gradient)
        decrement = gradient @ step
        if error < best_error:
            best, best_error, stalled = generated, error, 0
        elif decrement <= NEAR:
            stalled += 1
        if best_error < TOLERANCE or stalled >= 3:
            break
        size, before = 1.0, cost(rates)
        while cost(rates - size * step) == np.inf or (
            decrement > NEAR and cost(rates - size * step) > before - size * decrement / 4
        ):
            size /= 2
        rates = rates - size * step
    if best_error > PRECISE_ENOUGH:
        raise RuntimeError(
            "The composition before diversion could not be found to within "
            f"{PRECISE_ENOUGH:g} of the waste (best {best_error:.1e}): some type is "
            "almost all diverted."
        )
    return dict(zip(types, best.tolist()))
