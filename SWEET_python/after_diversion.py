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

from SWEET_python.constants import DIVERSION_ELIGIBILITY

#: The answer is final when no share moves by more than this between two passes. A pass
#: moves each share by exactly its round-trip error, so this also bounds that error.
TOLERANCE = 1e-12
#: A single pathway, or several whose types don't overlap, is exact after one pass:
#: within one pathway's types the add-back keeps their proportions. Overlapping ones
#: iterate. Bristol (recycling with incineration) takes 20 passes; the slowest of 4,000
#: adversarial mixes took 4,970.
MAX_PASSES = 100_000
#: Below this share of the waste left undiverted there is too little to survey, and
#: floating-point noise, not the survey, would decide the answer.
LEAST_LEFT = 1e-9


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
        RuntimeError: if the answer has not settled after MAX_PASSES passes.
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

    # Each type's generated share is what was left of it plus what each pathway took of
    # it. A pathway takes its share in proportion to the generated shares, the unknowns,
    # so iterate. Every pass keeps the shares summing to 1.
    kept = 1 - taken
    generated = dict(left)
    for _ in range(MAX_PASSES):
        pool_sizes = {
            path: sum(generated[w] for w in types) for path, types in pools.items()
        }
        updated = {
            waste: kept * left[waste]
            + sum(
                shares[path] * generated[waste] / pool_sizes[path]
                for path in shares
                if waste in eligibility[path]
            )
            for waste in left
        }
        if max(abs(updated[w] - generated[w]) for w in left) < TOLERANCE:
            return updated
        generated = updated
    raise RuntimeError(
        f"The composition before diversion did not settle in {MAX_PASSES} passes."
    )
