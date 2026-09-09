"""Canonical definitions shared across SWEET_python, the Climate TRACE waste
methane pipeline, and the WasteMAP backend: the modeling window, and the order
of the waste types.

Waste deposited before MODEL_START_YEAR is assumed to be zero everywhere:
sites with earlier reported opening years keep their true opening year in
source data, but models treat deposition as starting in MODEL_START_YEAR.
Export and display windows (Climate TRACE submissions, WasteMAP charts) are
filters over this window, never separate modeling horizons.

MODEL_START_YEAR moved 1990 -> 1970 on 2026-08-26. The cutoff is a truncation of
the decay tail, not a neutral choice: methane emitted today comes from decades of
accumulated stock, so zeroing deposition before the cutoff understates every site
with a long landfilling history -- most severely in cold/dry climates, where the
IPCC k values are lowest and the tail is longest. Measured on the 08_24_26 run,
Russia deposited as much municipal waste before 1990 as it did 1990-2021, and
restoring the earlier stock raises its national FOD by ~20%. 1970 is chosen over
1950 because the population series backing the waste projection (WPP2024, via
pops_yearly.csv) is credible per-year that far back while per-capita generation
before ~1970 is not, and because the residual tail before 1970 is small at every
k in defaults_2019.

CHANGING THIS CONSTANT REQUIRES A MATCHING pops_yearly.csv. The waste series is
population-driven, and city_params._population_series_from_pop_data returns None
unless the table carries EVERY column from MODEL_START_YEAR onward -- which
silently drops every country back to the frozen-CAGR growth scalars. Regenerate
with diagnostic_scripts/generate_pops_yearly.py and upload to blob
static_data/pops_yearly.csv BEFORE the constant lands in a run.
"""

MODEL_START_YEAR: int = 1970
MODEL_END_YEAR: int = 2050


# ---------------------------------------------------------------------------
# Canonical waste-type ordering
# ---------------------------------------------------------------------------

WASTE_TYPES: tuple[str, ...] = (
    "food",
    "green",
    "wood",
    "paper_cardboard",
    "textiles",
    "plastic",
    "metal",
    "glass",
    "rubber",
    "other",
)
"""The one true order of the ten waste types.

This is not an arbitrary choice: it is the field order of ``WasteFractions`` /
``WasteMasses`` in ``class_defs.py``. Those pydantic models already impose it on
every frame built from a ``model_dump()`` (the ``divs_df`` frames, for one), so
adopting the same sequence here makes the whole model agree on one column order
instead of two. ``tests/test_waste_type_order.py`` pins the two together.

Waste-type collections in the model are *sets* -- eligibility answers the
question "can this type be composted?", which is membership, not sequence. But
they are also iterated to build DataFrame columns, and a ``set`` of ``str``
iterates in hash order, which Python randomizes per process (PEP 456). Column
order therefore differed between two runs of the same code on the same inputs.
``WasteTypeSet`` below keeps the set semantics and fixes the iteration order.
"""

_WASTE_TYPE_INDEX: dict[str, int] = {w: i for i, w in enumerate(WASTE_TYPES)}


class WasteTypeSet(frozenset):
    """A set of waste types that always iterates in ``WASTE_TYPES`` order.

    Every ``list(city.components)``, ``for waste in city.div_components[div]``
    and ``{w: ... for w in components}`` in this package -- and in the Climate
    TRACE pipeline and the WasteMAP backend, which reach into the same
    attributes -- becomes deterministic by construction, with nothing to
    remember at the call site. That is the point of doing it in the type rather
    than sprinkling ``sorted()`` around: a ``sorted()`` that someone forgets to
    add is silent, and it would also order columns alphabetically rather than in
    the model's own order.

    A member outside ``WASTE_TYPES`` is not an error -- it sorts alphabetically
    after the known types, so a caller experimenting with a new stream still
    gets a stable order.

    Set operations return a ``WasteTypeSet`` rather than the base ``frozenset``,
    so the guarantee survives ``eligible & combustible`` -- as long as the
    ``WasteTypeSet`` is the left operand, since Python resolves the left
    operand's ``__and__`` first and a plain ``set``'s wins.
    """

    __slots__ = ()

    def __iter__(self):
        known = [w for w in WASTE_TYPES if frozenset.__contains__(self, w)]
        unknown = sorted(
            w for w in frozenset.__iter__(self) if w not in _WASTE_TYPE_INDEX
        )
        return iter(known + unknown)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self)!r})"

    # frozenset's operators return the base class; re-wrap so a derived set is
    # still ordered. Note the asymmetry Python imposes: in `plain_set & ordered`
    # the LEFT operand's __and__ runs first and returns a plain set, so keep the
    # WasteTypeSet on the left (or use the named methods below).
    def __and__(self, other):
        return type(self)(frozenset.__and__(self, other))

    def __or__(self, other):
        return type(self)(frozenset.__or__(self, other))

    def __sub__(self, other):
        return type(self)(frozenset.__sub__(self, other))

    def __xor__(self, other):
        return type(self)(frozenset.__xor__(self, other))

    def union(self, *others):
        return type(self)(frozenset.union(self, *others))

    def intersection(self, *others):
        return type(self)(frozenset.intersection(self, *others))

    def difference(self, *others):
        return type(self)(frozenset.difference(self, *others))

    def symmetric_difference(self, other):
        return type(self)(frozenset.symmetric_difference(self, other))

    def copy(self):
        return self
