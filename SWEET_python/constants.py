"""Canonical definitions shared across SWEET_python, the Climate TRACE waste
methane pipeline, and the WasteMAP backend: the modeling window, the order of the
waste types, which types each diversion pathway takes and what it rejects, unit
conversions, and flaring.

Each number here has one home. A copy of one elsewhere can be changed alone and
silently disagree with the rest, which is how the oxidation and MCF tables drifted;
import the name instead of writing the number. Site-type tables live beside this
module: ``mcf``, ``oxidation`` and ``gas_capture``.

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
SWEET_python/pops_yearly.csv with Climate TRACE's
diagnostic_scripts/generate_pops_yearly.py BEFORE the constant lands in a run.

THERE IS ONE COPY OF THAT TABLE: SWEET_python/pops_yearly.csv (SWEET_python.population),
from 1950, because the site tool models from a landfill's real opening year. Climate
TRACE's pipeline reads this file too, MODEL_START_YEAR onward (its
util/population.py), and so does its offline country-data build; the blob copy it used
to read, static_data/pops_yearly.csv, is no longer read by anything. So which
population a run used is fixed by the SWEET commit the run records.
"""

MODEL_START_YEAR: int = 1970
MODEL_END_YEAR: int = 2050

#: A close year meaning "still open when the model ends". A site takes no waste in
#: its close year, so MODEL_END_YEAR itself would close it a year early; this is
#: what an unknown close year means on every path.
# decision: intake-ends-before-close-year
NEVER_CLOSES: int = MODEL_END_YEAR + 1

#: The earliest open year the site tool and the advanced DST accept. An input
#: floor against accidental giant simulations, not the modelling start: those two
#: paths model a site from its real open year, back to here, because
#: pops_yearly.csv starts in 1950.
MODEL_YEAR_MIN: int = 1950


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

#: Global warming potential of methane over 100 years, IPCC AR5 (2013), without
#: climate-carbon feedbacks. SWEET's compost factor is in CO2e and is divided by
#: this to give methane; every DST endpoint reports t CH4.
GWP100_CH4: int = 28

#: Density of methane at 0 C and 1 atm, in kg/m3. landfill.py once carried two
#: values for it -- this and 0.657 in the `doing_fancy_ox` branch, 9.1% apart,
#: which is CH4 at 0 C versus at roughly 25 C. Both conversions are between the
#: same two units on the same quantity, so they cannot legitimately differ. 0.7168
#: wins because it is the value the rest of SWEET uses and the basis L_0 is
#: denominated in.
CH4_DENSITY_KG_PER_M3: float = 0.7168

#: Short tons in a metric tonne. EPA factors are per short ton.
SHORT_TONS_PER_TONNE: float = 1.1023


# ---------------------------------------------------------------------------
# Flaring
# ---------------------------------------------------------------------------
# Here rather than in dst_common, which defines the advanced DST's other
# defaults, because the annual engine (model_v2) needs one too and dst_common
# imports city_params -> landfill -> model_v2.

#: Share of captured gas a flare destroys when no one states it, on every path.
#: Both engines use it for a landfill with no flaring series, which is every
#: city-model landfill.
DEFAULT_FLARE_EFFICIENCY: float = 0.98


# ---------------------------------------------------------------------------
# Unknown climate
# ---------------------------------------------------------------------------

#: The climate a site or city gets when its caller sends none: the advanced DST's
#: default temperature, and the site tool's prefill when its weather lookup finds
#: nothing. One pair, so the two tools agree on an unknown climate.
# decision: missing-climate
UNKNOWN_CLIMATE_TEMPERATURE_C: float = 15.0
UNKNOWN_CLIMATE_PRECIPITATION_MM: float = 1000.0


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


# ---------------------------------------------------------------------------
# Which waste types decay, and which each diversion pathway takes
# ---------------------------------------------------------------------------

DEGRADABLE_TYPES: tuple[str, ...] = (
    "food",
    "green",
    "wood",
    "paper_cardboard",
    "textiles",
)
"""The waste types that make methane: the ones with a decay rate and an L_0.
In WASTE_TYPES order, so a frame built from it has the model's column order."""

# decision: diversion-eligibility
DIVERSION_ELIGIBILITY: dict[str, WasteTypeSet] = {
    "compost": WasteTypeSet({"food", "green", "wood", "paper_cardboard"}),
    "anaerobic": WasteTypeSet({"food", "green", "wood", "paper_cardboard"}),
    "combustion": WasteTypeSet(WASTE_TYPES),
    "recycling": WasteTypeSet(
        {
            "wood",
            "paper_cardboard",
            "textiles",
            "plastic",
            "rubber",
            "metal",
            "glass",
            "other",
        }
    ),
}
"""The waste types each diversion pathway can take. ``City.div_components`` is a
copy of this, and ``dst_allocation.DEFAULT_ELIGIBILITY`` is built from it."""


# ---------------------------------------------------------------------------
# What each diversion pathway rejects
# ---------------------------------------------------------------------------
# A rejected share is landfilled as the material it was. These come from SWEET's
# Excel, Assumptions sheet; City.__init__ copies them onto each city.

#: Share of each type sent to composting or anaerobic digestion that the process
#: cannot handle.
# decision: diversion-rejects
UNPROCESSABLE_SHARE: dict[str, float] = {
    "food": 0.0192,
    "green": 0.042522,
    "wood": 0.07896,
    "paper_cardboard": 0.12,
}

#: Contamination: the share of each type collected for composting that is not
#: compostable feedstock.
NON_COMPOSTABLE_NOT_TARGETED: dict[str, float] = {
    "food": 0.1,
    "green": 0.05,
    "wood": 0.05,
    "paper_cardboard": 0.1,
}

# The share of the waste sent to incineration that is landfilled unburned,
# as the material it was. None: everything sent to an incinerator stops
# producing methane. It was 0.1, SWEET's Excel "Reject Rate for
# Combustibles", sourced there as best professional judgement, and that
# 10% of food and paper was most of the modelled methane of a city that
# incinerates heavily. What an incinerator leaves is ash, nearly all of it
# mineral: the EU's Industrial Emissions Directive (Art. 50) holds its
# organic carbon under 3%, EPA's WARM counts no methane from it, and Japan
# landfills under 1% of the waste it processes unburned.
# decision: combustion-reject
COMBUSTION_REJECT_RATE: float = 0.0

#: Share of each type sent to recycling that IS recycled; the rest is rejected
#: and landfilled. A yield, not a reject rate, although City carries it as
#: ``recycling_reject_rates``.
RECYCLING_YIELD: dict[str, float] = {
    "wood": 0.8,
    "paper_cardboard": 0.775,
    "textiles": 0.99,
    "plastic": 0.875,
    "metal": 0.955,
    "glass": 0.88,
    "rubber": 0.78,
    "other": 0.87,
}
