"""
Advanced DST (adst) — city-level, multi-landfill modeling.

Backend for the ``/v1/site_emissions/adst_city_level`` endpoint. It is the
city-level sibling of :mod:`SWEET_python.advanced_dst`: same "caller supplies
time series directly" philosophy, but it models a whole city at once —
diversion (compost / anaerobic / combustion / recycling) plus an arbitrary
number of landfills that share the city's landfilled waste.

Input shape
-----------
City-level quantities live at the top of the request; per-landfill quantities
live in a list, one dict per landfill:

* ``waste_mass`` here is total **generated** city waste per year (tons), BEFORE
  diversion (unlike single-site adst where it is the landfilled mass).
* ``diversion_fractions`` is, per pathway, a {year: fraction-of-generated} series.
  The within-pathway component split is derived from the city composition
  (so the frontend only sends overall pathway fractions, not per-component splits);
  see ``_pathway_splits``.
* ``food_waste_prevention`` is a {year: fraction} series applied to ``waste_mass``
  and ``waste_fractions`` before anything else reads them: it is generation-side,
  so it lands upstream of diversion. Callers send the composition and the total
  they measured and let this module remove the food, rather than pre-shrinking
  the two themselves -- pairing a total scaled on one composition with fractions
  taken from another is the failure mode that pre-shrinking invites.
* each landfill carries its own type / open-close / gas capture / flaring /
  biocover. The split of the city's *landfilled* (net-of-diversion) waste across
  landfills is a top-level time series, ``landfill_split_timeline``:
  ``{year: [frac per landfill]}`` ordered to match the ``landfills`` list, with
  each year's fractions summing to ~1.
* a facility may set ``combusts``: it burns the waste routed to it instead of
  depositing it, and only the unburnable reject is deposited. See
  :class:`CityLandfillSpec`.
* ``other_landfills`` (optional) is where the rest goes. With it, each year's
  fractions may sum to less than 1, a listed landfill's share goes to other
  landfills in the years it is shut, and ``landfills`` may be empty. See
  :class:`OtherLandfillsSpec`.

Diversion math (without ``City``'s reject rates: see ``_diverted_masses``), the
per-landfill split, and emissions aggregation mirror the tested ``City`` machinery (``_calculate_diverted_masses``,
``LandfillWasteMassDF.create_advanced``, ``estimate_diversion_emissions``,
``sum_landfill_emissions``); the leaf primitives are reused directly.
"""

import copy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

import SWEET_python.defaults_2019 as defaults_2019
from SWEET_python.city_params import City, CityParameters, CustomError
from SWEET_python.class_defs import DivsDF, LandfillType, LandfillWasteMassDF, Variant
from SWEET_python.constants import MODEL_START_YEAR
from SWEET_python import dst_allocation
from SWEET_python import dst_common as common
from SWEET_python import site_inflow
from SWEET_python.dst_common import YearlyFloat, YearlyFractions, YearlyOptionalFloat

__all__ = [
    "AdvancedDSTCityRequest",
    "CityLandfillSpec",
    "OtherLandfillsSpec",
    "run_advanced_dst_city",
]

#: The diversion pathways `DivsDF` carries, in the order the mass flow reports.
DIVERSION_PATHWAYS: tuple[str, ...] = ("compost", "anaerobic", "combustion", "recycling")
SHARE_SUM_TOLERANCE = 0.02
#: How far past a material's generated mass the diversion pathways may reach
#: before ``over_diversion`` -- float noise, not a quantity of waste.
OVER_DIVERSION_TOLERANCE_TONS = 1e-6
#: Shortfall, as a share of generated waste, below which a routing from
#: ``dst_allocation`` (which rounds to a 1e-9 grid) is taken to fit.
ROUTING_GRID_TOLERANCE = 1e-8


class CityLandfillSpec(BaseModel):
    """One landfill within the city. All per-scenario fields are Variants."""

    landfill_type: Variant[LandfillType] = Field(
        ..., description="Site type: 0 landfill, 1 controlled dump, 2 open dump."
    )
    depth: Optional[Variant[float]] = Field(
        None,
        description=(
            "Site depth in metres. For a controlled/open dump (type 1 or 2) the "
            "depth selects the IPCC unmanaged category: deeper than 5 m raises "
            "MCF to 0.8, at or below 5 m lowers it to 0.4. Omit (the default) "
            "when the depth is unknown, which keeps the IPCC uncategorised 0.6 "
            "\u2014 an omitted depth is not read as shallow. Never applies to an "
            "engineered landfill (type 0), which is 1.0 regardless."
        ),
    )
    landfill_open_close: Variant[tuple[int, int]] = Field(
        ..., description="(open_year, close_year) of this site."
    )
    gas_capture_efficiency: Variant[YearlyOptionalFloat] = Field(
        ...,
        description=(
            "Fraction of the site's methane its collection system captures, per "
            "year; 0 is a site with no collection system. Null is a year the "
            "site collects gas at an efficiency nobody stated: it captures at "
            "its type's default for that year -- 60% for a landfill, 45% for a "
            "controlled dump, 0 for an open dump, the defaults SWEET's city "
            "model gives a site with a collection system and no measured "
            "efficiency. `run_advanced_dst_city_defaults` reports it."
        ),
    )
    flaring: Optional[Variant[YearlyFloat]] = Field(
        None, description="Flare destruction efficiency per year (defaults to 0.98)."
    )
    biocover: Optional[Variant[YearlyFloat]] = Field(
        None, description="Biocover oxidation floor per year (a fraction; defaults to 0)."
    )
    k_override: Optional[Variant[YearlyFloat]] = Field(
        None,
        description=(
            "Decomposition rate k for this site, per year (1/yr, 0 or more). "
            "Applied to every biodegradable component in place of the k the "
            "model derives from the city's temperature, precipitation and "
            "composition -- the site DST's `ks_overrides`, per landfill. Omit "
            "(the default) to use the derived k, which "
            "`run_advanced_dst_city_defaults` reports."
        ),
    )
    oxidation_override: Optional[Variant[YearlyOptionalFloat]] = Field(
        None,
        description=(
            "Oxidation rate for this site, per year: a fraction (0-1), or null "
            "for a year that states none. A number is the site's oxidation that "
            "year as stated -- it replaces both the factor the model looks up "
            "from the site's type and gas capture and the `biocover` floor. A "
            "null year, like an omitted field, gets that factor raised to the "
            "biocover as usual, which `run_advanced_dst_city_defaults` reports. "
            "Null is how a caller hands a year back to the model, e.g. once an "
            "intervention has changed the site the rate was stated for."
        ),
    )
    combusts: Optional[Variant[bool]] = Field(
        None,
        description=(
            "Whether this facility burns the waste routed to it -- incineration, "
            "with or without energy recovery -- instead of depositing it. The "
            "share of the city's disposed waste sent here still arrives here, "
            "but only the unburnable reject fraction "
            "(``City.combustion_reject_rate``, 10%) is deposited; the rest is "
            "destroyed. Combustion produces no methane in this model, so the "
            "burnt share simply leaves the deposited stream -- it is not "
            "re-routed to another site. The remaining per-site fields then "
            "describe the residue: ``landfill_type`` is the residue's disposal "
            "category, and the gas/biocover fields are the residue pile's, not "
            "the incinerator's. Omit (the default) for a depositing site.\n\n"
            "This is the city-level counterpart of ``City.sdst_v1_5``'s "
            "incineration handling, where a waste-to-energy site combusts 100% "
            "of its intake for every year it is open and the 10% reject is what "
            "decays."
        ),
    )
    accepted_waste_mass: Optional[Variant[YearlyFloat]] = Field(
        None,
        description=(
            "Total waste crossing this site's weighbridge each year, in tons, "
            "from every source -- a fact measured at the gate rather than a "
            "share of anything. Omit (the default) and the site receives "
            "exactly the city's allocation, which is the existing behaviour.\n\n"
            "This does not replace ``landfill_split_timeline``, which still "
            "divides the city's own landfilled waste across the city's sites "
            "and must still sum to ~1 per year. Whatever this figure exceeds "
            "the city's allocation is modeled as a second stream arriving from "
            "outside the baseline: it carries the city's post-diversion "
            "residual composition, and is combusted and windowed on exactly the "
            "same terms as the city's stream, because it is the same hole in "
            "the ground.\n\n"
            "It never changes the city's own emissions -- a city answers for "
            "the waste it generates, wherever that waste goes. A figure *below* "
            "the city's allocation is read as 'no waste from outside', not as a "
            "reduction of the city's; see ``site_inflow.surplus_intake``. Ask "
            "for the split with ``with_site_emissions=True``."
        ),
    )


class OtherLandfillsSpec(CityLandfillSpec):
    """Where the city's landfilled waste goes when no listed landfill takes it.

    Each year, other landfills take whatever share of the city's landfilled
    waste the listed landfills leave: what their shares fall short of 1 by, and
    the share of any listed landfill that is shut that year. So a caller can
    state the sites it knows about and leave the rest here, without balancing
    the shares to 1 first.

    They are modelled as one more landfill, with every per-site field a listed
    one has. Two fields are not taken. Other landfills accept waste in every
    model year, so they have no open or close years. They take only the city's
    own waste, so they have no gate total.
    """

    landfill_open_close: Optional[Variant[tuple[int, int]]] = Field(
        None,
        description=(
            "Not taken: other landfills accept waste from the model's first year "
            "and never close, so every year's leftover has somewhere to go."
        ),
    )

    @model_validator(mode="after")
    def _take_the_city_s_leftover_every_year(self) -> "OtherLandfillsSpec":
        if self.landfill_open_close is not None:
            raise ValueError(
                "other_landfills take no landfill_open_close: they accept waste "
                "in every model year."
            )
        if self.accepted_waste_mass is not None:
            raise ValueError(
                "other_landfills take no accepted_waste_mass: they take only the "
                "city's own waste."
            )
        return self


class AdvancedDSTCityRequest(BaseModel):
    """Request body for the city-level advanced DST endpoint."""

    city_name: str = Field(..., description="City name (identity).")
    precipitation: float = Field(..., ge=0, description="Average annual precipitation, mm/year.")
    implement_year: int = Field(..., description="Year scenario changes take effect.")
    waste_mass: Variant[YearlyFloat] = Field(
        ..., description="Total GENERATED city waste per year (tons), before diversion. {year: tons}."
    )
    waste_fractions: Variant[YearlyFractions] = Field(
        ...,
        description=(
            "City waste composition per year, {year: [10 fractions]} in the order "
            "food, green, wood, paper_cardboard, textiles, plastic, metal, glass, "
            "rubber, other."
        ),
    )
    landfills: List[CityLandfillSpec] = Field(
        default_factory=list,
        description=(
            "One entry per landfill in the city. It may be empty when "
            "`other_landfills` is given, which then takes all of the city's "
            "landfilled waste."
        ),
    )
    landfill_split_timeline: Optional[Variant[Dict[int, List[float]]]] = Field(
        None,
        description=(
            "Fraction of the city's landfilled (net-of-diversion) waste sent to "
            "each landfill per year: {year: [frac_landfill_0, frac_landfill_1, ...]}. "
            "The list order matches the `landfills` list. Required whenever "
            "`landfills` lists any. Without `other_landfills`, each year's "
            "fractions should sum to ~1. With it, they may sum to less, and "
            "other landfills take the rest."
        ),
    )
    diversion_fractions: Optional[Variant[Dict[str, YearlyFloat]]] = Field(
        None,
        description=(
            "Per-pathway fraction of total generated waste diverted each year: "
            "{compost|anaerobic|combustion|recycling: {year: fraction}}. Omitted "
            "pathways are treated as zero."
        ),
    )
    food_waste_prevention: Optional[Variant[YearlyFloat]] = Field(
        None,
        description=(
            "Fraction (0-1) of the city's FOOD waste that is never generated, "
            "per year: {year: fraction}. Food mass falls by that share, no other "
            "material's tonnage moves, and the total generated stream shrinks by "
            "exactly the food removed -- so it applies upstream of diversion, "
            "which then takes its share of the smaller stream. Omitted (the "
            "default) means none. Values outside 0-1 are clamped."
        ),
    )
    temperature: float = Field(10.0, description="Average annual temperature, deg C.")
    country: Optional[str] = Field(None, description="ISO3 country code (identity).")
    rmi_id: Optional[int] = Field(None, description="City/site identifier (identity).")
    other_landfills: Optional[OtherLandfillsSpec] = Field(
        None,
        description=(
            "Where the city's landfilled waste goes when no listed landfill takes "
            "it. Each year, other landfills take 1 minus the shares of the listed "
            "landfills open that year. That includes the share of a listed "
            "landfill that is shut that year, which without this field is spread "
            "over the open ones instead. A year whose open listed landfills "
            "claim more than 1 is scaled down to 1, and other landfills take "
            "nothing that year. They are reported after the listed landfills, "
            "as the last entry of `site_emissions` and of the mass flow's "
            "`sites`. Omit (the default) and the shares must sum to ~1, as "
            "before."
        ),
    )

    @model_validator(mode="after")
    def _the_waste_has_somewhere_to_go(self) -> "AdvancedDSTCityRequest":
        if not self.landfills and self.other_landfills is None:
            raise ValueError(
                "landfills must list at least one landfill unless other_landfills "
                "is given."
            )
        if self.landfills and self.landfill_split_timeline is None:
            raise ValueError(
                "landfill_split_timeline is required when landfills lists any."
            )
        return self


# --------------------------------------------------------------------------- #
# Food waste prevention
# --------------------------------------------------------------------------- #
def _prevent_food_waste(
    fractions: pd.DataFrame,
    total: pd.Series,
    prevented: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove a share of generated food waste, leaving every other material alone.

    ``prevented`` is the per-year fraction (0-1) of food waste that is never
    generated. Food mass falls by that share, no other component's tonnage
    moves, and the total shrinks by exactly the food removed -- the composition
    renormalizes around the smaller stream, so non-food *shares* rise while
    their tonnages do not. This is the same lever as the city DST's
    ``food_waste_prevention`` (``City.implement_dst_changes_simple_v1_5``).

    Deliberately done in mass space rather than by rescaling the fractions
    directly. The returned fractions and total are consumed as
    ``fraction x total``, and that only conserves non-food mass if both sides
    were derived from *one* composition -- multiplying out first, then dividing
    back, makes it impossible to pair a total scaled on one composition with
    fractions taken from another.

    Returns the inputs untouched when no prevention applies, so an omitted
    ``food_waste_prevention`` is bit-identical to the pre-existing behaviour.
    """
    share = prevented.reindex(fractions.index).fillna(0.0).clip(0.0, 1.0)
    if float(share.max()) <= 0.0:
        return fractions, total

    masses = fractions.mul(total, axis=0)
    masses["food"] = masses["food"] * (1.0 - share)
    new_total = masses.sum(axis=1)

    # A year whose stream is prevented away entirely has no composition left to
    # speak of; keep the original fractions there so the k-values and the compost
    # emission factor stay well defined against a zero mass.
    new_fractions = fractions.copy()
    positive = new_total > 0
    if bool(positive.any()):
        new_fractions.loc[positive] = masses.loc[positive].div(
            new_total[positive], axis=0
        )
    return new_fractions, new_total


# --------------------------------------------------------------------------- #
# Mass flow
# --------------------------------------------------------------------------- #
def _mass_flow(
    generated: pd.DataFrame,
    wgen: pd.DataFrame,
    divs: DivsDF,
    net: pd.DataFrame,
    site_masses: List[pd.DataFrame],
) -> dict:
    """Where one variant's tonnage went, per year.

    Per waste type throughout, with one exception: ``sites`` is a per-landfill
    total, summed across the components, because the split timeline it comes
    from is stated per landfill and not per material.

    Every frame here is one the emissions were computed from, so a caller
    rendering this is showing the model rather than a parallel estimate of it.
    That matters most for ``diverted``: compost and anaerobic digestion draw only
    on organics and recycling only on recyclables, so the split is per waste type and approximating it as one scalar on
    the total quietly composts metal and glass.

    ``sites`` is per landfill in request order, already windowed by each one's
    open/close years, with other landfills last when the request has them. It
    sums to ``landfilled`` in every year that has at least one open landfill,
    because the split is renormalized over whichever ones are accepting waste
    (see ``_renormalize_over_open``) -- and other landfills are open in every
    year. The one case where the two
    differ is a year with *no* landfill open at all: that city's post-diversion
    waste has nowhere to go, and the gap is reported rather than hidden —
    ``landfilled`` is what the city disposed of, ``sites`` is what arrived
    somewhere.
    """
    def shaped(frame: pd.DataFrame) -> pd.DataFrame:
        """All ten components, in one order, zero-filled.

        Each pathway frame carries only the components that pathway can draw on
        (compost has four columns, not ten), and subtracting frames reorders
        them alphabetically. Callers get one stable shape instead.
        """
        return frame.reindex(columns=list(common.WASTE_COMPONENTS), fill_value=0.0).fillna(0.0)

    # A pathway the request never uses is an all-zero frame; omitting it roughly
    # halves this payload in ordinary use, since most cities run two of the four.
    # A missing pathway reads as zero.
    diverted = {}
    for pathway in DIVERSION_PATHWAYS:
        frame = shaped(getattr(divs, pathway))
        if frame.to_numpy().any():
            diverted[pathway] = frame

    return {
        "generated": shaped(generated),
        "prevented": shaped(generated.sub(wgen, fill_value=0.0)),
        "diverted": diverted,
        "landfilled": shaped(net),
        "sites": [frame.sum(axis=1) for frame in site_masses],
    }


# --------------------------------------------------------------------------- #
# Diversion
# --------------------------------------------------------------------------- #
def _mef_compost(fractions_df: pd.DataFrame, ref_year: int) -> float:
    """Per-city compost emission factor, from food/green share at the reference year.

    Mirrors City.cityparams_obj_for_blank_site: weighted CH4 factor for the
    food/green split, scaled to CO2e (the *1.1023*0.7 is baked in).
    """
    food = float(fractions_df.loc[ref_year, "food"])
    green = float(fractions_df.loc[ref_year, "green"])
    denom = food + green
    if denom <= 0:
        return 0.0
    return (0.0055 * food / denom + 0.0139 * green / denom) * 1.1023 * 0.7


def _pathway_splits(
    fractions_df: pd.DataFrame,
    total_generated: pd.Series,
    pathway_fractions: Dict[str, pd.Series],
    city: City,
) -> Dict[str, pd.DataFrame]:
    """How each pathway's mass divides across its components, per year.

    One frame per pathway, years x that pathway's components; a row sums to 1, or
    to 0 where the city has none of the pathway's components.

    A pathway takes its components in proportion to the composition, as
    ``City._calculate_diverted_masses`` does. That can ask for more of a material
    than the city generates even when the request as a whole fits: compost and
    recycling both draw on wood and paper, so a large compost share plus a
    recycling share, each split in proportion, can overrun the paper while the
    city still has plastic and glass that only recycling wants. In those years the
    split moves toward a routing that fits (``dst_allocation.solve_allocation``,
    the city DST's solver) by the smallest step that takes every material back
    within what the city generates. Recycling then leans on plastic, metal and
    glass and leaves the paper to compost.

    The smallest step keeps the answer continuous: a request just past where the
    proportional split stops fitting gets a split just past proportional, not the
    solver's routing outright. Years the proportional split fits are untouched,
    measured on the same tolerance ``over_diversion`` uses. A year no routing fits
    keeps its proportional split, and ``over_diversion`` names what overflows.
    """
    splits: Dict[str, pd.DataFrame] = {}
    for pathway in DIVERSION_PATHWAYS:
        # div_components values are WasteTypeSet, so iteration is already in
        # canonical WASTE_TYPES order -- the same order every other frame in the
        # package uses. This used to be sorted() (alphabetical), which was
        # deterministic but disagreed with the rest of the model.
        components = list(city.div_components[pathway])
        sub = fractions_df[components]
        denom = sub.sum(axis=1)
        splits[pathway] = sub.div(denom, axis=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    claimed = pd.DataFrame(0.0, index=fractions_df.index, columns=fractions_df.columns)
    for pathway, split in splits.items():
        mass_into_pathway = pathway_fractions[pathway] * total_generated
        claimed = claimed.add(split.mul(mass_into_pathway, axis=0), fill_value=0.0)
    generated = fractions_df.mul(total_generated, axis=0)
    overrun = (claimed - generated > OVER_DIVERSION_TOLERANCE_TONS).any(axis=1)

    for year in fractions_df.index[overrun]:
        rerouted = _rerouted_split(
            fractions_df.loc[year],
            {p: float(pathway_fractions[p].loc[year]) for p in DIVERSION_PATHWAYS},
            {p: splits[p].loc[year] for p in DIVERSION_PATHWAYS},
            city,
        )
        if rerouted is None:
            continue
        for pathway, row in rerouted.items():
            splits[pathway].loc[year] = row
    return splits


def _rerouted_split(
    composition: pd.Series,
    targets: Dict[str, float],
    proportional: Dict[str, pd.Series],
    city: City,
) -> Optional[Dict[str, pd.Series]]:
    """One year's split, moved from proportional just far enough to fit.

    ``None`` when no routing fits. All three allocations are fractions of generated
    waste, pathways x components: ``prop`` (each pathway in proportion, which
    overruns), ``routed`` (the solver's, which fits) and the blend
    ``(1 - t) * prop + t * routed``. Every blend gives each pathway its full share,
    and each material's total is linear in ``t``, so the least ``t`` that brings
    every overrun material back to what the city generates is a closed form.
    """
    eligibility = {p: set(city.div_components[p]) for p in DIVERSION_PATHWAYS}
    result = dst_allocation.solve_allocation(
        {c: float(composition[c]) for c in composition.index},
        targets,
        eligibility=eligibility,
        treatments=DIVERSION_PATHWAYS,
    )
    # A request exactly at its bound fits with nothing to spare, and the solver's
    # grid can round that to a shortfall of a few 1e-9. Refusing it would refuse
    # the very value the bounds offer.
    if result["shortfall"] > ROUTING_GRID_TOLERANCE:
        return None

    available = composition.astype(float)
    prop = pd.DataFrame(
        {p: proportional[p].mul(targets[p]) for p in DIVERSION_PATHWAYS}
    ).T.reindex(columns=available.index, fill_value=0.0).fillna(0.0)
    routed = pd.DataFrame(
        {p: pd.Series(result["allocation"][p], dtype=float) for p in DIVERSION_PATHWAYS}
    ).T.reindex(index=list(DIVERSION_PATHWAYS), columns=available.index, fill_value=0.0)
    routed = routed.fillna(0.0)

    # Materials the same pathways can take are interchangeable to the solver, so
    # it fills them in arbitrary order -- all of the wood, say, and little of the
    # paper. Spread each pathway's take of such a group over it in proportion to
    # the mix, as the proportional split does. Without this the blend below sees
    # the solver's routing as already full on a material and steps all the way to
    # it, and a request just past the cap jumps to a different split.
    groups: Dict[frozenset, List[str]] = {}
    for component in available.index:
        takers = frozenset(p for p in DIVERSION_PATHWAYS if component in eligibility[p])
        groups.setdefault(takers, []).append(component)
    for members in groups.values():
        group_size = float(available[members].sum())
        if group_size <= 0:
            continue
        taken = routed[members].sum(axis=1)
        routed[members] = np.outer(taken.to_numpy(), (available[members] / group_size).to_numpy())

    # The solver works on a 1e-9 grid. Give each pathway exactly its share, then
    # hold each material to what exists; the second step can leave a pathway short
    # by grid noise, never over.
    row_totals = routed.sum(axis=1)
    target_series = pd.Series(targets, dtype=float)
    routed = routed.mul((target_series / row_totals).where(row_totals > 0, 0.0), axis=0)
    routed_use = routed.sum(axis=0)
    routed = routed.mul((available / routed_use).where(routed_use > available, 1.0), axis=1)

    prop_use = prop.sum(axis=0)
    routed_use = routed.sum(axis=0)
    over = prop_use > available
    step = ((prop_use - available) / (prop_use - routed_use))[over]
    t = float(step.max()) if not step.empty else 0.0
    t = min(max(t, 0.0), 1.0)
    blended = prop.mul(1.0 - t).add(routed.mul(t))

    return {
        p: (blended.loc[p, proportional[p].index] / targets[p]) if targets[p] > 0 else proportional[p]
        for p in DIVERSION_PATHWAYS
    }


def _diverted_masses(
    fractions_df: pd.DataFrame,
    total_generated: pd.Series,
    div_fracs: Dict[str, Dict[int, float]],
    city: City,
    years: pd.Index,
) -> DivsDF:
    """Per-pathway, per-component diverted mass (tons/yr).

    For each pathway: mass into the pathway = pathway_fraction * total_generated;
    that mass is split across the pathway's eligible components (``_pathway_splits``:
    in proportion to the city composition, rerouted where that overruns a material).

    A pathway keeps everything it is sent. Unlike ``City._calculate_diverted_masses``,
    no reject or yield rate sends part of it back to the landfills: a city that
    composts 10% of its waste composts 10%, and every share a user states in the
    advanced DST is a share of what actually happens. The city DST and Climate TRACE
    keep their rates.
    """
    div_dfs: Dict[str, pd.DataFrame] = {}
    pathway_fractions = {
        pathway: common.yearly_to_series(div_fracs.get(pathway), years, default=0.0)
        for pathway in DIVERSION_PATHWAYS
    }
    splits = _pathway_splits(fractions_df, total_generated, pathway_fractions, city)
    for pathway in DIVERSION_PATHWAYS:
        components = list(city.div_components[pathway])
        denom = fractions_df[components].sum(axis=1)
        pathway_fraction = pathway_fractions[pathway]

        # A pathway with none of its eligible components present cannot receive
        # the mass asked of it. The split below would divide 0 by 0, and zeroing
        # that quietly diverted nothing at all -- so a request to compost a
        # sixth of a stream with no organics in it composted none of it and said
        # nothing. It is the degenerate end of over-diversion: a positive share
        # of a pool of size zero.
        starved = (pathway_fraction > 0) & (denom <= 0)
        if bool(starved.any()):
            year = int(pathway_fraction.index[starved][0])
            raise CustomError(
                "diversion_without_material",
                f"{pathway} is sent {float(pathway_fraction.loc[year]):.4g} of the "
                f"waste stream in {year}, but the city generates none of the "
                f"material it can process ({', '.join(components)}).",
            )

        mass_into_pathway = pathway_fraction * total_generated
        div_dfs[pathway] = splits[pathway].mul(mass_into_pathway, axis=0)  # years x components

    return DivsDF(
        compost=div_dfs["compost"],
        anaerobic=div_dfs["anaerobic"],
        combustion=div_dfs["combustion"],
        recycling=div_dfs["recycling"],
    )


def _splice_divs(baseline: DivsDF, scenario: DivsDF, implement_year: int) -> DivsDF:
    """Make the scenario diversion equal the baseline for years before implement_year."""
    def splice(b: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
        out = s.copy()
        out.loc[: implement_year - 1, :] = b.loc[: implement_year - 1, :]
        return out

    return DivsDF(
        compost=splice(baseline.compost, scenario.compost),
        anaerobic=splice(baseline.anaerobic, scenario.anaerobic),
        combustion=splice(baseline.combustion, scenario.combustion),
        recycling=splice(baseline.recycling, scenario.recycling),
    )


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
def _make_city_parameters(
    request: AdvancedDSTCityRequest,
    fractions_df: pd.DataFrame,
    ks,
    mef_compost: float,
    divs_df: DivsDF,
    city_instance_attrs: dict,
    implement_year: int,
    scenario: int,
) -> CityParameters:
    parameters = CityParameters(
        precip=request.precipitation,
        precip_zone=defaults_2019.get_precipitation_zone(request.precipitation),
        temperature=request.temperature,
        growth_rate_historic=1.0,
        growth_rate_future=1.0,
        year_of_data_pop={"baseline": implement_year, "scenario": implement_year},
        mef_compost=mef_compost,
        scenario=scenario,
        implement_year=implement_year,
        city_instance_attrs=city_instance_attrs,
        ks=ks,
        waste_fractions=fractions_df,
        rmi_id=request.rmi_id,
    )
    # divs_df is typed as a plain DataFrame on CityParameters, but the engine
    # (estimate_diversion_emissions) expects a DivsDF. Assign it post-construction
    # so we bypass the field validator, matching how the existing City code does it.
    parameters.divs_df = divs_df
    return parameters


def _split_timeline_to_df(
    timeline: Dict[int, List[float]], years: pd.Index, n_landfills: int
) -> pd.DataFrame:
    """Parse a {year: [frac per landfill]} timeline into a years x landfill DataFrame.

    Columns are 0..n_landfills-1 (matching the ``landfills`` list order); missing
    years are forward/back filled.
    """
    if not timeline:
        raise CustomError("invalid_parameters", "landfill_split_timeline is required.")
    rows: Dict[int, List[float]] = {}
    for year, fracs in timeline.items():
        if len(fracs) != n_landfills:
            raise CustomError(
                "invalid_parameters",
                f"landfill_split_timeline for year {year} must have {n_landfills} "
                f"fractions (one per landfill, matching the landfills list).",
            )
        rows[int(year)] = [float(x) for x in fracs]
    df = pd.DataFrame.from_dict(rows, orient="index", columns=list(range(n_landfills)))
    return df.sort_index().reindex(years).ffill().bfill()


def _deposited_share(combusts: bool, city: City) -> float:
    """Fraction of the waste arriving at a facility that is actually deposited.

    A depositing site keeps all of it. A combusting facility keeps only the
    unburnable reject, ``combustion_reject_rate``. The combustion diversion
    pathway, like every pathway here, keeps everything it is sent
    (``_diverted_masses``), so the residue is the facility's alone.
    """
    return float(city.combustion_reject_rate) if combusts else 1.0


def _accepting_windows(
    request: "AdvancedDSTCityRequest", variant: str
) -> List[tuple[int, int]]:
    """Each landfill's (open, close) pair for one variant, in request order."""
    windows = []
    for spec in request.landfills:
        dates = spec.landfill_open_close["baseline"]
        if variant == "scenario":
            dates = spec.landfill_open_close["scenario"] or dates
        windows.append((int(dates[0]), int(dates[1])))
    return windows


def _accepting_mask(
    windows: List[tuple[int, int]], columns: pd.Index, years: pd.Index
) -> pd.DataFrame:
    """years x landfill booleans: is this landfill accepting waste this year?

    Intake runs ``[open, close)`` — a closure year is not an intake year, see
    ``common.apply_window`` — clipped into the modeled window. A landfill that
    closes the year it opens never accepts anything.
    """
    accepting = pd.DataFrame(False, index=years, columns=columns)
    first, last = int(years.min()), int(years.max())
    for column, (open_year, close_year) in zip(columns, windows):
        lower, upper = max(int(open_year), first), min(int(close_year) - 1, last)
        if lower <= upper:
            accepting.loc[lower:upper, column] = True
    return accepting


def _renormalize_over_open(
    split: pd.DataFrame, accepting: pd.DataFrame
) -> pd.DataFrame:
    """Redistribute each year's shares across the landfills actually accepting waste.

    The split says how the landfilled stream divides across landfills, but a
    landfill only accepts waste inside its own window — and a closure year is
    not an intake year (see ``common.apply_window``). A share pointed at a
    landfill that is shut is not a share of anything: the mass used to be scaled
    onto it and then zeroed by the window, so it left the model altogether. The
    shares are meant to account for *all* the post-diversion waste, so they are
    gated to the open landfills and renormalized over them.

    A year in which no landfill is open renormalizes to all zeros rather than
    raising. That city has nowhere to put its waste, which is a real thing for a
    single-site city past its site's closure, and the mass flow reports it as
    ``sites`` falling short of ``landfilled`` — the honest description of waste
    with no destination, and now the only way that gap can arise.

    A caller that already zeroes shut landfills and normalizes over the rest
    gets its own numbers back untouched.
    """
    gated = split.where(accepting, 0.0)
    totals = gated.sum(axis=1)
    somewhere_open = totals > 0

    renormalized = gated.copy()
    renormalized.loc[somewhere_open] = gated.loc[somewhere_open].div(
        totals[somewhere_open], axis=0
    )
    renormalized.loc[~somewhere_open] = 0.0
    return renormalized


def _validate_share_range(checked: pd.DataFrame, label: str) -> None:
    """Every share in ``checked`` must be a fraction, within the share tolerance."""
    out_of_range = (checked < -SHARE_SUM_TOLERANCE) | (
        checked > 1.0 + SHARE_SUM_TOLERANCE
    )
    if bool(out_of_range.to_numpy().any()):
        rows = out_of_range.any(axis=1)
        first_year = int(rows[rows].index[0])
        column = int(out_of_range.loc[first_year].idxmax())
        raise CustomError(
            "invalid_parameters",
            f"{label} landfill waste_share fractions must each be between 0 and 1 "
            f"(year {first_year}, landfill {column} is "
            f"{float(checked.loc[first_year, column]):.3f}).",
        )


def _with_other_landfills(request: AdvancedDSTCityRequest) -> AdvancedDSTCityRequest:
    """The request with its other landfills listed as one more landfill.

    Every entry point calls this first, so the rest of the module only ever
    sees listed landfills whose shares sum to 1. Other landfills become the
    last landfill: open from the model's first year, never closed, and holding
    each year's leftover. The run is the same, bit for bit, as one where the
    caller listed that landfill itself at those shares, because it *is* that
    run.

    For each variant and each year:

    * a listed landfill keeps the share it was given in the years it accepts
      waste. In the years it is shut (not yet open, or closed by its own dates
      or by a scenario's), that share goes to other landfills instead of being
      spread over the listed landfills still open;
    * other landfills take 1 minus what the open listed landfills claim;
    * a year whose open listed landfills claim more than 1 is scaled down to 1,
      and other landfills take nothing that year.

    Each listed share is checked to be a fraction as submitted, so the scaling
    can't hide a negative share or one above 1.
    """
    other = request.other_landfills
    if other is None:
        return request

    # Other landfills open with the model, so that every modelled year's
    # leftover has a destination: at MODEL_START_YEAR, or with a listed
    # landfill that states an earlier opening. `validate_years` checks the
    # listed dates and returns that earliest open year.
    pairs = [(MODEL_START_YEAR, common.NEVER_CLOSES)]
    for spec in request.landfills:
        baseline_dates = spec.landfill_open_close["baseline"]
        scenario_dates = spec.landfill_open_close["scenario"] or baseline_dates
        pairs += [tuple(baseline_dates), tuple(scenario_dates)]
    open_year = common.validate_years(pairs, int(request.implement_year))
    years = pd.Index(range(open_year, common.MODEL_YEAR_MAX + 1), name="year")

    n_listed = len(request.landfills)
    variants = ("baseline", "scenario")
    splits: Dict[str, pd.DataFrame] = {}
    accepting: Dict[str, pd.DataFrame] = {}
    for variant in variants:
        if n_listed:
            # An empty scenario split means the baseline's, as in
            # `run_advanced_dst_city`; `variant_get` falls back only on None.
            timeline = (
                request.landfill_split_timeline[variant]
                or request.landfill_split_timeline["baseline"]
            )
            splits[variant] = _split_timeline_to_df(timeline, years, n_listed)
            accepting[variant] = _accepting_mask(
                _accepting_windows(request, variant), splits[variant].columns, years
            )
        else:
            splits[variant] = pd.DataFrame(index=years)
            accepting[variant] = pd.DataFrame(index=years)
    # The scenario tracks the baseline before changes take effect. Spliced
    # before the check, as `run_advanced_dst_city` does, so a pre-implement
    # scenario row the model discards can't get the request refused.
    implement_year = int(request.implement_year)
    splits["scenario"].loc[: implement_year - 1, :] = splits["baseline"].loc[
        : implement_year - 1, :
    ]
    accepting["scenario"].loc[: implement_year - 1, :] = accepting["baseline"].loc[
        : implement_year - 1, :
    ]

    rows: Dict[str, Dict[int, List[float]]] = {}
    for variant in variants:
        _validate_share_range(splits[variant], variant)
        # The range check tolerates a share a rounding error outside [0, 1]. Used
        # as given, -0.01 would bury negative waste at that site and hand other
        # landfills 101% of the rest.
        listed = splits[variant].where(accepting[variant], 0.0).clip(lower=0.0, upper=1.0)
        claimed = listed.sum(axis=1)
        over = claimed > 1.0
        listed.loc[over] = listed.loc[over].div(claimed[over], axis=0)
        leftover = (1.0 - listed.sum(axis=1)).clip(lower=0.0)
        rows[variant] = {
            int(year): [float(x) for x in listed.loc[year]] + [float(leftover.loc[year])]
            for year in years
        }

    always_open = (open_year, common.NEVER_CLOSES)
    listed_other = CityLandfillSpec.model_validate(
        {
            **other.model_dump(exclude={"landfill_open_close", "accepted_waste_mass"}),
            "landfill_open_close": {"baseline": always_open, "scenario": always_open},
        }
    )
    return request.model_copy(
        update={
            "landfills": [*request.landfills, listed_other],
            "landfill_split_timeline": Variant[Dict[int, List[float]]](
                baseline=rows["baseline"], scenario=rows["scenario"]
            ),
            "other_landfills": None,
        }
    )


def _validate_shares(split: pd.DataFrame, accepting: pd.DataFrame, label: str) -> None:
    """Each year's landfill shares must be fractions, and must sum to ~1.

    Checked on what the caller submitted, before the shares are renormalized over
    the open landfills — afterwards every year sums to exactly 1 or exactly 0 by
    construction, so checking it there would only confirm our own arithmetic.

    Summing to one is not on its own enough to make a row a split. Each share is
    a fraction of the landfilled stream, so it has to lie in [0, 1]: the row
    ``[-0.5, 1.5]`` sums to exactly 1 and passes any sum check, then scales a
    *negative* mass onto the first landfill and buries it there. Nothing
    downstream rejects that — ``LandfillWasteMassDF.create_advanced``
    multiplies straight through, and the ``over_diversion`` guard measures the
    city's residual before the split — so it is checked here or not at all.

    A year in which no landfill is open at all is exempt: there is nowhere for
    that waste to go, so no set of shares can account for it, and whatever was
    submitted is discarded either way.
    """
    _validate_share_range(split.loc[accepting.any(axis=1)], label)

    total = split.sum(axis=1)
    somewhere_open = accepting.any(axis=1)
    off = (total < 1.0 - SHARE_SUM_TOLERANCE) | (total > 1.0 + SHARE_SUM_TOLERANCE)
    bad = total[off & somewhere_open]
    if not bad.empty:
        first_year = int(bad.index[0])
        raise CustomError(
            "invalid_parameters",
            f"{label} landfill waste_share fractions must sum to ~1 per year "
            f"(year {first_year} sums to {float(bad.iloc[0]):.3f}).",
        )


# --------------------------------------------------------------------------- #
# Diversion limits
# --------------------------------------------------------------------------- #
#
# A pathway can only divert material it actually receives: you cannot compost
# more organics than a city generates. ``_pathway_splits`` reroutes a pathway's
# mass away from a material others need whenever some routing fits, so the bound
# is whether *any* routing fits, not whether the proportional one does.
#
# Whether one fits is Hall's condition (the supply-demand theorem for a
# transportation problem): for every set ``S`` of pathways, what they are asked
# for together must fit in the material any of them can take,
#
#     sum(divfrac_p for p in S)  <=  sum(f_c for c in union(components(p) for p in S))
#
# With four pathways that is fifteen sets, so the bound stays closed-form. The
# sets that bind in practice: compost and digestion together against the
# organics; recycling against the recyclables; and all of them against
# everything either can take. Wood and paper sit in both of the first two, so
# they count once in the third.
#
# This bounds diverted mass, which is what ``run_advanced_dst_city``'s
# ``over_diversion`` guard measures. The two are
# the same boundary on purpose: a caller respecting these limits cannot produce a
# request the model rejects on mass balance.
SHARE_LIMIT_TOLERANCE = 1e-9


def _pathway_sets() -> List[tuple[str, ...]]:
    """Every non-empty set of diversion pathways: the sets Hall's condition checks."""
    return [
        tuple(p for i, p in enumerate(DIVERSION_PATHWAYS) if mask >> i & 1)
        for mask in range(1, 2 ** len(DIVERSION_PATHWAYS))
    ]


def _material_for(fractions: pd.DataFrame, city: City, pathways: tuple[str, ...]) -> pd.Series:
    """Share of the stream any of ``pathways`` can draw on, per year."""
    components = set().union(*(city.div_components[p] for p in pathways))
    return fractions[[c for c in fractions.columns if c in components]].sum(axis=1)


def _routing_slack(
    fractions: pd.DataFrame,
    div_fracs: Dict[str, pd.Series],
    city: City,
    pathway_sets: Optional[List[tuple[str, ...]]] = None,
) -> pd.Series:
    """Least spare material over every set of pathways, per year; negative = no routing fits.

    ``pathway_sets`` checks only those sets instead of all of them.
    """
    slack = pd.Series(np.inf, index=fractions.index)
    for pathways in pathway_sets if pathway_sets is not None else _pathway_sets():
        demand = sum(div_fracs[p] for p in pathways)
        slack = np.minimum(slack, _material_for(fractions, city, pathways) - demand)
    return slack


def _component_denominators(
    fractions: pd.DataFrame, city: City
) -> Dict[str, pd.Series]:
    """``D_p`` per pathway per year: the share of the stream that pathway can draw on."""
    return {
        pathway: fractions[sorted(city.div_components[pathway])].sum(axis=1)
        for pathway in DIVERSION_PATHWAYS
    }


def _component_use(
    fractions: pd.DataFrame,
    total: pd.Series,
    div_fracs: Dict[str, pd.Series],
    city: City,
) -> pd.DataFrame:
    """Fraction of each component's own mass that the pathways claim, per year.

    Read off the split the model would use (``_pathway_splits``), so a year where
    the pathways are rerouted shows the materials they share at most full. A year
    no routing fits keeps its proportional split, and shows above 1 on what
    overflows. A pathway with none of its components present draws on nothing.
    """
    splits = _pathway_splits(fractions, total, div_fracs, city)
    claimed = pd.DataFrame(0.0, index=fractions.index, columns=fractions.columns)
    for pathway, split in splits.items():
        claimed = claimed.add(split.mul(div_fracs[pathway], axis=0), fill_value=0.0)
    return claimed.divide(fractions).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _max_pathway_fraction(
    fractions: pd.DataFrame,
    div_fracs: Dict[str, pd.Series],
    city: City,
    pathway: str,
) -> pd.Series:
    """Largest fraction of generated waste ``pathway`` may take, per year.

    Every other pathway is held at its own value for that year. Each set of
    pathways containing this one bounds it by the material the set can draw on,
    less what the rest of the set is asked for; the tightest set wins. A year
    whose other pathways already overreach without this one has no headroom:
    no value of this pathway makes that request run.
    """
    bound = pd.Series(1.0, index=fractions.index)
    for pathways in _pathway_sets():
        if pathway not in pathways:
            continue
        others = sum((div_fracs[p] for p in pathways if p != pathway), pd.Series(0.0, index=fractions.index))
        bound = np.minimum(bound, _material_for(fractions, city, pathways) - others)
    without = {p: (s if p != pathway else s * 0.0) for p, s in div_fracs.items()}
    others_fit = _routing_slack(fractions, without, city) >= -SHARE_LIMIT_TOLERANCE
    return bound.where(others_fit, 0.0).clip(lower=0.0, upper=1.0)


def _max_food_waste_prevention(
    fractions: pd.DataFrame,
    total: pd.Series,
    div_fracs: Dict[str, pd.Series],
    city: City,
    tolerance: float = 1e-4,
) -> pd.Series:
    """Largest food-waste-prevention fraction each year's diversion demand permits.

    Prevention shrinks the organic pool — it removes food — so a compost fraction
    that fitted before it takes effect can stop fitting after. It also grows
    every other material's share of the smaller stream, so it can make a request
    fit that did not: 80% food and 20% plastic, with 30% compost and 50%
    recycling, fits only between about 75% and 89% prevention.

    So the feasible shares are an interval that need not start at zero, and the
    answer is its top. Each set of pathways (Hall's condition) moves one way: a
    set that draws on food loses material as more is prevented, and one that
    does not gains share. The food sets alone are therefore true then false in
    the prevented share, which is what bisection needs, and their crossing is
    the top of the interval. The other sets only get easier toward it, so they
    are checked once, there; if they fail, no prevention fits and the answer is
    zero.

    Per year, and deliberately so. Folding dated interventions into a per-year
    series is the caller's concern — a cap for one intervention is the minimum of
    these over the years it is in force — and the model has no notion of one.
    """
    sets = _pathway_sets()
    drawing_on_food = [
        s for s in sets if "food" in set().union(*(city.div_components[p] for p in s))
    ]
    not_drawing_on_food = [s for s in sets if s not in drawing_on_food]

    def holds(shares: pd.Series, pathway_sets: List[tuple[str, ...]]) -> pd.Series:
        shrunk_fractions, _ = _prevent_food_waste(fractions, total, shares)
        return _routing_slack(shrunk_fractions, div_fracs, city, pathway_sets) >= -SHARE_LIMIT_TOLERANCE

    def fits(shares: pd.Series) -> pd.Series:
        """Whether each year's food sets survive that year's prevention.

        Takes a share *per year* rather than one scalar, because every step
        below is row-wise — ``_prevent_food_waste`` renormalizes within a year,
        the denominators are per-year sums, and ``starved`` is a per-year mask.
        So one call answers every year at its own probe, and the bisection needs
        one per halving instead of one per year per halving.
        """
        shrunk_fractions, _ = _prevent_food_waste(fractions, total, shares)
        within = holds(shares, drawing_on_food)

        # A pathway asked for a share of a pool that prevention has emptied is
        # refused by the model (`diversion_without_material`) however small the
        # share, and a share inside the tolerance above would pass the slack
        # test. So an empty pool is checked on its own.
        denominators = _component_denominators(shrunk_fractions, city)
        starved = pd.Series(False, index=fractions.index)
        for pathway in DIVERSION_PATHWAYS:
            starved = starved | (
                (div_fracs[pathway] > 0) & (denominators[pathway] <= 0.0)
            )
        return within & ~starved

    unconstrained = fits(pd.Series(1.0, index=fractions.index))
    low = pd.Series(0.0, index=fractions.index)
    high = pd.Series(1.0, index=fractions.index)
    # ~14 halvings clears the 1e-4 tolerance, and each is one pass over all
    # years -- so the whole bisection is ~14 frame operations, not 14 per year.
    while float((high - low).max()) > tolerance:
        mid = (low + high) / 2.0
        ok = fits(mid)
        low = low.where(~ok, mid)
        high = high.where(ok, mid)
    top = low.where(~unconstrained, 1.0)
    return top.where(holds(top, not_drawing_on_food), 0.0)


def _limits_for_variant(
    fractions: pd.DataFrame,
    total: pd.Series,
    prevented: pd.Series,
    div_fracs: Dict[str, pd.Series],
    city: City,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """One variant's bounds, from that variant's own inputs."""
    prevented_fractions, prevented_total = _prevent_food_waste(fractions, total, prevented)
    use = _component_use(prevented_fractions, prevented_total, div_fracs, city)

    # A pathway asked for a share of a pool with nothing in it is infeasible,
    # and `component_use` cannot say so: the pathway draws on nothing, so its
    # row looks comfortably under 1. `run_advanced_dst_city` raises
    # `diversion_without_material` for exactly this request, so the bounds have
    # to name it or a caller reading only `component_use` would call the request
    # fine right up to the error.
    denominators = _component_denominators(prevented_fractions, city)
    starved = pd.DataFrame(
        {
            pathway: (div_fracs[pathway] > 0) & (denominators[pathway] <= 0.0)
            for pathway in DIVERSION_PATHWAYS
        },
        index=fractions.index,
    )

    return {
        "component_use": use,
        "limiting_component": use.idxmax(axis=1),
        "max_diversion": pd.DataFrame(
            {
                pathway: _max_pathway_fraction(
                    prevented_fractions, div_fracs, city, pathway
                )
                for pathway in DIVERSION_PATHWAYS
            }
        ),
        "max_food_waste_prevention": _max_food_waste_prevention(
            fractions, total, div_fracs, city
        ),
        "starved_pathways": starved,
    }


def run_advanced_dst_city_limits(
    request: "AdvancedDSTCityRequest",
) -> Dict[str, Dict[str, pd.DataFrame | pd.Series]]:
    """What this city's composition allows, per year — the caller's input bounds.

    Returns ``{"baseline": {...}, "scenario": {...}}``, each variant's bounds
    computed from that variant's own inputs. Per variant, indexed by year:

    * ``component_use`` — years x components, the fraction of each component's
      own mass the pathways claim. Anything above 1 is over-diversion.
    * ``limiting_component`` — which component that maximum falls on.
    * ``max_diversion`` — per pathway, the largest fraction of generated waste it
      may take with every other pathway held where it is.
    * ``max_food_waste_prevention`` — the largest prevented share of food the
      year's diversion demand leaves room for.
    * ``starved_pathways`` — years x pathways, true where a pathway is asked for
      material the city does not generate any of. ``component_use`` reads that
      case as zero rather than as too much (the claim is 0/0), so it is reported
      separately; ``run_advanced_dst_city`` raises ``diversion_without_material``
      for the same request.

    Both variants are returned because the model runs both, and it enforces
    ``over_diversion`` on each: the scenario's composition, tonnage, prevention
    and diversion all apply from ``implement_year`` onward, so bounds taken from
    the baseline alone leave the scenario years unbounded. Every scenario series
    here is spliced at ``implement_year`` the same way the model splices it, so
    a variant's pre-implement bounds are its baseline's by construction.

    That is the point of computing this here rather than in a caller: the
    component sets come from ``city.div_components``, the same ones
    ``_diverted_masses`` splits on, and the basis is diverted mass, the
    same one the ``over_diversion`` guard measures. A caller respecting these
    bounds cannot produce a request the model rejects on mass balance.

    All of it reflects the prevention already in the request; a caller asking
    "how much further could I go" gets an answer consistent with where it is.
    """
    request = _with_other_landfills(request)
    implement_year = int(request.implement_year)
    years = _model_years(request)

    baseline_fractions = common.fractions_to_df(request.waste_fractions["baseline"], years)
    scenario_fractions = common.fractions_to_df(
        request.waste_fractions["scenario"] or request.waste_fractions["baseline"], years
    )
    # Scenario tracks baseline until changes take effect, exactly as the model
    # splices it -- `variant_series` already does this for the series inputs.
    scenario_fractions.loc[: implement_year - 1, :] = baseline_fractions.loc[
        : implement_year - 1, :
    ]

    baseline_total, scenario_total = common.variant_series(
        request.waste_mass, years, implement_year, default=None
    )
    baseline_prevented, scenario_prevented = common.variant_series(
        request.food_waste_prevention, years, implement_year, default=0.0
    )

    div_variant = request.diversion_fractions
    baseline_raw = (
        (common.variant_get(div_variant, "baseline") or {}) if div_variant is not None else {}
    )
    scenario_raw = div_variant["scenario"] if div_variant is not None else None
    if scenario_raw is None:
        scenario_raw = baseline_raw
    baseline_div = {
        pathway: common.yearly_to_series(baseline_raw.get(pathway), years, default=0.0)
        for pathway in DIVERSION_PATHWAYS
    }
    scenario_div = {}
    for pathway in DIVERSION_PATHWAYS:
        series = baseline_div[pathway].copy()
        own = common.yearly_to_series(scenario_raw.get(pathway), years, default=0.0)
        series.loc[implement_year:] = own.loc[implement_year:]
        scenario_div[pathway] = series

    city = City(request.city_name)
    common.city_instance_attrs(city, request.country)

    return {
        "baseline": _limits_for_variant(
            baseline_fractions, baseline_total, baseline_prevented, baseline_div, city
        ),
        "scenario": _limits_for_variant(
            scenario_fractions, scenario_total, scenario_prevented, scenario_div, city
        ),
    }


# --------------------------------------------------------------------------- #
# Per-site decomposition and oxidation rates
# --------------------------------------------------------------------------- #
def _model_years(request: "AdvancedDSTCityRequest") -> pd.Index:
    """The years every series is aligned to, after validating each site's window."""
    open_close_pairs = []
    for spec in request.landfills:
        b_open, b_close = (int(x) for x in spec.landfill_open_close["baseline"])
        s_dates = spec.landfill_open_close["scenario"] or spec.landfill_open_close["baseline"]
        s_open, s_close = (int(x) for x in s_dates)
        open_close_pairs.append((b_open, b_close))
        open_close_pairs.append((s_open, s_close))
    model_start = common.validate_years(open_close_pairs, int(request.implement_year))
    return pd.Index(range(model_start, common.MODEL_YEAR_MAX + 1), name="year")


def _reference_year(implement_year: int, years: pd.Index) -> int:
    """The year whose composition stands for a variant's k and compost factor."""
    return min(max(implement_year, int(years.min())), int(years.max()))


def _city_decomposition_rates(
    request: "AdvancedDSTCityRequest",
    years: pd.Index,
    baseline_fractions: pd.DataFrame,
    scenario_fractions: pd.DataFrame,
):
    """The k every site decays at unless it states its own, per variant.

    Derived once for the city, from its climate and the composition left after
    food waste prevention, because that is what every site receives.
    """
    implement_year = int(request.implement_year)
    ref_year = _reference_year(implement_year, years)
    return common.decomposition_rates(
        request.temperature, request.precipitation, implement_year, years,
        common.representative_vector(baseline_fractions, ref_year),
        common.representative_vector(scenario_fractions, ref_year),
    )


def _site_types(spec: CityLandfillSpec) -> tuple[int, int]:
    base_type = int(spec.landfill_type["baseline"])
    scen_type = (
        int(spec.landfill_type["scenario"])
        if spec.landfill_type["scenario"] is not None
        else base_type
    )
    return base_type, scen_type


def _gas_capture_defaults(
    spec: CityLandfillSpec, years: pd.Index, implement_year: int
) -> tuple[pd.Series, pd.Series]:
    """The (baseline, scenario) capture a null year of the site's efficiency gets.

    Its type's default (``common.GAS_CAPTURE_DEFAULT``) for the type it is that
    year, as ``mcf_series`` and ``oxidation_series`` switch it.
    """
    base_type, scen_type = _site_types(spec)
    return (
        common.gas_capture_default_series(base_type, base_type, implement_year, years),
        common.gas_capture_default_series(base_type, scen_type, implement_year, years),
    )


def _gas_capture(
    spec: CityLandfillSpec, years: pd.Index, implement_year: int
) -> tuple[pd.Series, pd.Series]:
    """A site's (baseline, scenario) capture efficiency per year, defaults filled in.

    A null year is a site with a collection system and no stated efficiency.
    It captures at its type's default rather than at nothing, the rule
    ``city_params._resolve_gas_capture`` applies to a site with a system and no
    measured efficiency. Before this, a caller that knew a site collects gas but
    not how well had to send 0, which modelled a site that captures nothing.
    """
    gas_base, gas_scen = common.optional_variant_series(
        spec.gas_capture_efficiency, years, implement_year, default=0.0
    )
    default_base, default_scen = _gas_capture_defaults(spec, years, implement_year)
    gas_base = gas_base.fillna(default_base)
    gas_scen = gas_scen.fillna(default_scen)
    # A capture efficiency is a fraction of generated methane, so it cannot
    # exceed 1. Nothing downstream bounds it -- the field is typed as a bare
    # float map and the engine multiplies straight through -- so an out-of-range
    # value silently produced negative emissions rather than an error.
    return gas_base.clip(0.0, 1.0), gas_scen.clip(0.0, 1.0)


def _override_series(
    variant,
    years: pd.Index,
    implement_year: int,
    *,
    field: str,
    upper: Optional[float],
    optional_years: bool = False,
) -> Optional[tuple[pd.Series, pd.Series]]:
    """A site's stated rate as (baseline, scenario) series, or None if it states none.

    With ``optional_years`` a year may be null, carried as NaN: that year
    states nothing and the model's own rate applies.

    Refused rather than clipped when out of range: unlike a capture efficiency,
    an override is a number someone typed in place of the model's, and quietly
    modelling a different one would show them a result for a site they did not
    describe.
    """
    if variant is None:
        return None
    try:
        if optional_years:
            base, scen = common.optional_variant_series(variant, years, implement_year)
        else:
            base, scen = common.variant_series(variant, years, implement_year, default=None)
    except CustomError as error:
        raise CustomError("invalid_parameters", f"{field}: {error.message}") from error
    for label, series in (("baseline", base), ("scenario", scen)):
        # NaN is a year that states nothing where years may (`optional_years`),
        # and a malformed number everywhere else.
        bad = np.isinf(series) | (series < 0)
        if not optional_years:
            bad |= series.isna()
        if upper is not None:
            bad |= series > upper
        if bool(bad.any()):
            year = int(series.index[bad.to_numpy()][0])
            bound = f"between 0 and {upper:g}" if upper is not None else "0 or more"
            raise CustomError(
                "invalid_parameters",
                f"{label}: {field} for {year} is {series.loc[year]:g}; it must be {bound}.",
            )
    return base, scen


def _site_decomposition_rates(
    spec: CityLandfillSpec,
    city_ks: tuple,
    years: pd.Index,
    implement_year: int,
) -> tuple:
    """A site's (baseline, scenario) k: its own where it states one, else the city's."""
    override = _override_series(
        spec.k_override, years, implement_year, field="k_override", upper=None
    )
    if override is None:
        return city_ks
    k_base, k_scen = override
    return (
        common.uniform_decomposition_rates(k_base),
        common.uniform_decomposition_rates(k_scen),
    )


def run_advanced_dst_city_defaults(
    request: "AdvancedDSTCityRequest",
) -> Dict[str, List[Dict[str, pd.Series]]]:
    """The rates the model uses at each site that does not state its own.

    Returns ``{"baseline": [...], "scenario": [...]}``, one entry per landfill
    in request order, with the request's ``other_landfills`` last when it has
    them, each ``{"k", "oxidation", "gas_capture_efficiency"}`` indexed by year:

    * ``k`` -- the decomposition rate (1/yr) derived from the city's
      temperature, precipitation and composition after food waste prevention.
      Every biodegradable component decays at this one rate, and every site in
      the city shares it, because they all receive the city's waste.
    * ``oxidation`` -- the factor looked up from the site's type and whether it
      captures gas, raised to the ``biocover`` floor in the years the request
      has one. That is exactly the number a year of ``oxidation_override``
      replaces. A null year of ``gas_capture_efficiency`` counts as capturing
      at the type's default, which is above 0 except at an open dump.
    * ``gas_capture_efficiency`` -- the capture efficiency (0-1) of the site's
      type that year: what a null year of its ``gas_capture_efficiency``
      becomes. Reported whether or not the site collects gas.

    None of them reflects an override or a stated efficiency in the request:
    this is what those stand in for, so a caller can show it beside the box a
    user types into. Nothing here runs the decay model, so it is cheap enough
    to call on every edit.
    """
    request = _with_other_landfills(request)
    implement_year = int(request.implement_year)
    years = _model_years(request)

    # The same composition `run_advanced_dst_city` derives k from: as authored,
    # then with the prevented food taken out.
    baseline_fractions = common.fractions_to_df(request.waste_fractions["baseline"], years)
    scenario_fractions = common.fractions_to_df(
        request.waste_fractions["scenario"] or request.waste_fractions["baseline"], years
    )
    baseline_total, scenario_total = common.variant_series(
        request.waste_mass, years, implement_year, default=None
    )
    prevented_baseline, prevented_scenario = common.variant_series(
        request.food_waste_prevention, years, implement_year, default=0.0
    )
    baseline_fractions, _ = _prevent_food_waste(
        baseline_fractions, baseline_total, prevented_baseline
    )
    scenario_fractions, _ = _prevent_food_waste(
        scenario_fractions, scenario_total, prevented_scenario
    )
    ks_baseline, ks_scenario = _city_decomposition_rates(
        request, years, baseline_fractions, scenario_fractions
    )

    result: Dict[str, List[Dict[str, pd.Series]]] = {"baseline": [], "scenario": []}
    for spec in request.landfills:
        base_type, scen_type = _site_types(spec)
        gas_base, gas_scen = _gas_capture(spec, years, implement_year)
        default_gas_base, default_gas_scen = _gas_capture_defaults(
            spec, years, implement_year
        )
        bio_base, bio_scen = common.variant_series(
            spec.biocover, years, implement_year, default=0.0
        )
        result["baseline"].append({
            "k": ks_baseline.food,
            "oxidation": common.oxidation_series(
                base_type, base_type, gas_base, bio_base, implement_year, years
            ),
            "gas_capture_efficiency": default_gas_base,
        })
        result["scenario"].append({
            "k": ks_scenario.food,
            "oxidation": common.oxidation_series(
                base_type, scen_type, gas_scen, bio_scen, implement_year, years
            ),
            "gas_capture_efficiency": default_gas_scen,
        })
    return result


# --------------------------------------------------------------------------- #
# Where the methane goes, and how much of it is already buried
# --------------------------------------------------------------------------- #
def _m3_to_tons_ch4(series: pd.Series) -> pd.Series:
    """m^3 of methane to tons, through the arithmetic the city's own total uses."""
    return series.map(City.convert_methane_m3_to_ton_co2e) / 28


def _flare_series(landfill, index: pd.Index) -> pd.Series:
    """The landfill's flare destruction efficiency per year, as the kernel reads it."""
    flaring = landfill.flaring
    if flaring is None:
        return pd.Series(1.0, index=index)
    if np.isscalar(flaring):
        return pd.Series(float(flaring), index=index)
    return pd.Series(flaring, dtype=float).reindex(index)


def _methane_budget(parameters: CityParameters) -> pd.DataFrame:
    """Where the city's methane goes each year, in tons, for one variant.

    At the city's landfills, ``generated = captured + oxidised + emitted``:

    - ``generated``: what the buried waste produces (``Landfill.ch4``), before
      anything is done about it.
    - ``captured``: collected and destroyed, i.e. generation x capture efficiency
      x flare efficiency. The gas a flare fails to destroy is not here; the
      kernel releases it with everything else (``model_v2.estimate_emissions2``).
    - ``oxidised``: broken down in the cover on the way out.
    - ``emitted``: what reaches the air, exactly the landfill part of the city's
      total (``parameters.landfill_emissions``).

    ``diversion`` is composting and digestion, and ``emitted + diversion`` is
    the city's ``total``, the series the endpoint's headline is. Only the city's
    own waste: a site's waste from outside the baseline (``site_inflow``) is not
    in ``parameters.landfills``, as it is not in the total.

    ``oxidised`` is taken as the remainder, so the identity holds to the last
    float; the kernel does not keep it separately.
    """
    total = parameters.total_emissions["total"]
    index = total.index
    generated = pd.Series(0.0, index=index)
    captured = pd.Series(0.0, index=index)
    for landfill in parameters.landfills:
        if landfill is None or landfill.emissions is None:
            continue
        generated = generated + _m3_to_tons_ch4(landfill.ch4.sum(axis=1)).reindex(
            index, fill_value=0.0
        )
        destroyed = landfill.captured.sum(axis=1) * _flare_series(
            landfill, landfill.captured.index
        )
        captured = captured + _m3_to_tons_ch4(destroyed).reindex(index, fill_value=0.0)
    emitted = parameters.landfill_emissions["total"].reindex(index, fill_value=0.0)
    diversion = parameters.diversion_emissions["total"].reindex(index, fill_value=0.0)
    return pd.DataFrame(
        {
            "generated": generated,
            "captured": captured,
            "oxidised": generated - captured - emitted,
            "emitted": emitted,
            "diversion": diversion,
            "total": total,
        },
        index=index,
    )


def _committed_emissions(parameters: CityParameters, before_year: int) -> pd.DataFrame:
    """The city's landfill methane from waste buried before ``before_year``, in tons.

    What its landfills would go on emitting if none of its waste arrived from
    ``before_year`` on. Each landfill is rerun on a copy with that year's and
    every later year's deposits removed; capture, flaring and oxidation stay as
    the run has them, since those act on the gas whenever it is made. Exact,
    not an estimate: first-order decay is linear in what is buried, so the
    methane of earlier waste does not depend on what is buried after it. Before
    ``before_year`` this equals the landfills' own emissions. Composting and
    digestion are left out: they emit in the year the waste is treated, so none
    of theirs is ever already buried.

    The copies leave the run's own landfills, and everything read from them,
    untouched.
    """
    total = pd.Series(0.0, index=parameters.total_emissions.index)
    for landfill in parameters.landfills:
        if landfill is None or landfill.emissions is None:
            continue
        buried = copy.deepcopy(landfill)
        mass = landfill.waste_mass_df.copy()
        mass.loc[mass.index >= before_year, :] = 0.0
        buried.waste_mass_df = mass
        buried.estimate_emissions(skip_ox=True)
        total = total + _m3_to_tons_ch4(buried.emissions["total"]).reindex(
            total.index, fill_value=0.0
        )
    return pd.DataFrame({"total": total})


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_advanced_dst_city(
    request: AdvancedDSTCityRequest,
    *,
    with_mass_flow: bool = False,
    with_site_emissions: bool = False,
    with_methane_budget: bool = False,
    committed_before: Optional[int] = None,
) -> dict:
    """Run the city-level advanced DST.

    Returns ``{"baseline": total_emissions_df, "scenario": total_emissions_df}``
    (city totals summed across landfills + diversion), each indexed by year with a
    ``total`` column.

    With ``with_mass_flow=True`` the result carries an extra ``"mass_flow"`` key
    describing where the tonnage went, per variant: ``generated`` (per component,
    before prevention), ``prevented``, ``diverted`` (per pathway per component,
    everything the pathway was sent), ``landfilled`` (the residual per component), and
    ``sites`` (per landfill, in request order). A diversion pathway the request
    never uses is omitted from ``diverted`` rather than sent as zeros — the
    payload is several times the emissions one, so this is worth the asymmetry;
    a missing pathway reads as zero. Those are the very frames the
    emissions are computed from, not a second derivation — callers that need to
    *show* the mass flow should read it from here rather than reimplementing the
    diversion split, which is per waste type and does not survive being
    approximated as a scalar on the total.

    With ``with_site_emissions=True`` the result carries a ``"site_emissions"``
    key: per variant, one entry per landfill in request order, each
    ``{"city", "other", "total"}`` in tons of methane per year. ``city`` is the
    methane this city's waste produced at that site and is what the city's own
    total is built from; ``other`` is what waste from outside the baseline
    produced there (see ``CityLandfillSpec.accepted_waste_mass``), and is zero
    for a site that only takes the city's waste. They are two independent passes
    through the decay kernel, summed -- never a ratio applied to one of them,
    which is wrong by tens of percent wherever the city's share of a site moves
    over time. Asking for them changes no emissions.

    With ``with_methane_budget=True`` the result carries a ``"methane_budget"``
    key: per variant, a frame by year of ``generated``, ``captured``,
    ``oxidised`` and ``emitted`` at the city's landfills (the first is the sum of
    the other three), ``diversion`` (composting and digestion) and ``total``
    (``emitted + diversion``, the same series as the variant's own). Tons of
    methane; see ``_methane_budget``.

    With ``committed_before=Y`` the result carries a ``"committed"`` key:
    ``{"before": Y, "baseline": frame, "scenario": frame}``, each frame a
    ``total`` by year of the methane the city's landfills emit from waste buried
    before ``Y`` -- what they would still emit if no more of its waste arrived
    from ``Y`` on. ``Y`` must lie within the model's years, or one past the last.
    See ``_committed_emissions``. Neither option changes any emissions.

    A request with ``other_landfills`` has them modelled as one more landfill,
    after the listed ones: they are the last entry of ``site_emissions`` and of
    the mass flow's ``sites``. See ``_with_other_landfills`` for the shares they
    take.
    """
    request = _with_other_landfills(request)
    implement_year = int(request.implement_year)

    # --- Years: validate every landfill's open/close window ---
    years = _model_years(request)
    if committed_before is not None and not (
        years[0] <= int(committed_before) <= years[-1] + 1
    ):
        raise CustomError(
            "invalid_committed_before",
            f"committed_before must be between {years[0]} and {years[-1] + 1}; "
            f"got {committed_before}.",
        )

    # --- City composition + total generated waste, per variant ---
    baseline_fractions = common.fractions_to_df(request.waste_fractions["baseline"], years)
    scenario_fractions = common.fractions_to_df(
        request.waste_fractions["scenario"] or request.waste_fractions["baseline"], years
    )
    baseline_total, scenario_total = common.variant_series(request.waste_mass, years, implement_year, default=None)

    # Per-component generated mass as authored, before any food is prevented —
    # the top band of the mass flow, and what `prevented` is measured against.
    generated_baseline = baseline_fractions.mul(baseline_total, axis=0)
    generated_scenario = scenario_fractions.mul(scenario_total, axis=0)
    generated_scenario.loc[: implement_year - 1, :] = generated_baseline.loc[
        : implement_year - 1, :
    ]

    # --- Food waste prevention: shrink the generated stream before anything
    # else reads it. Everything downstream (per-component masses, diversion,
    # k-values, the compost emission factor) is then derived from the prevented
    # composition and total, which is what keeps the two in step. ---
    prevented_baseline, prevented_scenario = common.variant_series(
        request.food_waste_prevention, years, implement_year, default=0.0
    )
    baseline_fractions, baseline_total = _prevent_food_waste(
        baseline_fractions, baseline_total, prevented_baseline
    )
    scenario_fractions, scenario_total = _prevent_food_waste(
        scenario_fractions, scenario_total, prevented_scenario
    )

    wgen_baseline = baseline_fractions.mul(baseline_total, axis=0)
    wgen_scenario = scenario_fractions.mul(scenario_total, axis=0)
    wgen_scenario.loc[: implement_year - 1, :] = wgen_baseline.loc[: implement_year - 1, :]

    # --- Diversion (per variant), with scenario tracking baseline pre-implement ---
    city = City(request.city_name)
    div_variant = request.diversion_fractions
    baseline_div = (common.variant_get(div_variant, "baseline") or {}) if div_variant is not None else {}
    scenario_div_raw = div_variant["scenario"] if div_variant is not None else None
    scenario_div = scenario_div_raw if scenario_div_raw is not None else baseline_div

    divs_baseline = _diverted_masses(
        baseline_fractions, baseline_total, baseline_div, city, years
    )
    divs_scenario_own = _diverted_masses(
        scenario_fractions, scenario_total, scenario_div, city, years
    )
    divs_scenario = _splice_divs(divs_baseline, divs_scenario_own, implement_year)

    # Guard against diverting more of a material than the city generates: a
    # negative residual. It is exactly the boundary
    # ``run_advanced_dst_city_limits`` reports, so a caller respecting those
    # bounds cannot trip this.
    #
    # The residual is kept rather than discarded: it is the landfilled mass per
    # waste type, which is what the mass flow reports and what the per-landfill
    # split is taken from.
    #
    # The message reaches the advanced DST's user verbatim, so it names the
    # waste type as the composition legend does and leaves the variant out. The
    # ADST sends the same per-year series on both halves, so the baseline always
    # trips first, even when the diversion that overreached is an intervention;
    # "baseline" would point the user at the wrong thing. The year says when.
    net_masses: Dict[str, pd.DataFrame] = {}
    for label, wgen, divs in (
        ("baseline", wgen_baseline, divs_baseline),
        ("scenario", wgen_scenario, divs_scenario),
    ):
        residual = wgen.sub(divs.sum(), fill_value=0.0)
        if (residual < -OVER_DIVERSION_TOLERANCE_TONS).to_numpy().any():
            shortfall = residual.min()
            component = str(shortfall.idxmin())
            year = int(residual[component].idxmin())
            raise CustomError(
                "over_diversion",
                f"More {common.WASTE_COMPONENT_NAMES[component]} would be sent to "
                f"diversion in {year} than the city generates.",
            )
        net_masses[label] = residual

    # The mix the city is left to bury, as per-year component shares. Any waste
    # arriving at a site from outside the baseline is given this same mix, so
    # that both streams at a site decay at one `k` and therefore superpose --
    # see `site_inflow` for why that is load-bearing rather than tidy.
    # The fallback is the composition as *shares* rather than `wgen_*`, the
    # composition as masses: a year the city generates nothing has an all-zero
    # mass frame and so no fallback at all, which left a site's outside waste
    # with a zero mix and deposited none of it.
    residual_mix = {
        "baseline": site_inflow.residual_composition(net_masses["baseline"], baseline_fractions),
        "scenario": site_inflow.residual_composition(net_masses["scenario"], scenario_fractions),
    }

    # --- City-wide decomposition rates + compost emission factors ---
    # A site that states its own k decays at that instead; see the landfill loop.
    ref_year = _reference_year(implement_year, years)
    ks_baseline, ks_scenario = _city_decomposition_rates(
        request, years, baseline_fractions, scenario_fractions
    )
    mef_baseline = _mef_compost(baseline_fractions, ref_year)
    mef_scenario = _mef_compost(scenario_fractions, ref_year)

    # --- Parameters ---
    city_instance_attrs = common.city_instance_attrs(city, request.country)
    baseline_parameters = _make_city_parameters(
        request, baseline_fractions, ks_baseline, mef_baseline, divs_baseline, city_instance_attrs, implement_year, 0
    )
    scenario_parameters = _make_city_parameters(
        request, scenario_fractions, ks_scenario, mef_scenario, divs_scenario, city_instance_attrs, implement_year, 1
    )
    baseline_params_dict = baseline_parameters.update_cityparams_dict()
    scenario_params_dict = scenario_parameters.update_cityparams_dict()

    # --- Per-landfill split of landfilled waste (top-level time series) ---
    n_landfills = len(request.landfills)
    baseline_timeline = common.variant_get(request.landfill_split_timeline, "baseline")
    scenario_timeline = request.landfill_split_timeline["scenario"] or baseline_timeline
    baseline_split = _split_timeline_to_df(baseline_timeline, years, n_landfills)
    scenario_split = _split_timeline_to_df(scenario_timeline, years, n_landfills)

    # Each variant is gated on its own open/close years -- a `siteClosure`
    # intervention can move them -- then validated as submitted and renormalized
    # over whatever is actually accepting waste.
    baseline_accepting = _accepting_mask(
        _accepting_windows(request, "baseline"), baseline_split.columns, years
    )
    scenario_accepting = _accepting_mask(
        _accepting_windows(request, "scenario"), scenario_split.columns, years
    )
    # Scenario tracks baseline before changes take effect, and the splice comes
    # first so that validation and renormalization both act on the shares the
    # model will actually use. Ordered the other way round, a caller was
    # rejected for a pre-implement scenario row this line then discarded --
    # while the pre-implement renormalization it had just done was overwritten
    # regardless, so only the error was observable.
    scenario_split.loc[: implement_year - 1, :] = baseline_split.loc[: implement_year - 1, :]
    scenario_accepting.loc[: implement_year - 1, :] = baseline_accepting.loc[
        : implement_year - 1, :
    ]
    _validate_shares(baseline_split, baseline_accepting, "baseline")
    _validate_shares(scenario_split, scenario_accepting, "scenario")
    baseline_split = _renormalize_over_open(baseline_split, baseline_accepting)
    scenario_split = _renormalize_over_open(scenario_split, scenario_accepting)

    # --- Build a landfill (baseline + scenario) per spec ---
    baseline_landfills: List = []
    scenario_landfills: List = []
    baseline_masses: List[pd.DataFrame] = []
    scenario_masses: List[pd.DataFrame] = []
    baseline_ox: List[pd.Series] = []
    scenario_ox: List[pd.Series] = []
    # One entry per site, `None` where the site takes only the city's waste.
    # Deliberately kept out of `*_parameters.landfills`: that list is what
    # `City.sum_landfill_emissions` sums, and a city does not answer for a
    # neighbour's waste. Keeping these out of it is the whole reason the city's
    # total is bit-identical rather than merely close.
    baseline_surplus: List[Optional[object]] = []
    scenario_surplus: List[Optional[object]] = []
    baseline_surplus_masses: List[Optional[pd.DataFrame]] = []
    scenario_surplus_masses: List[Optional[pd.DataFrame]] = []

    for index, spec in enumerate(request.landfills):
        base_type, scen_type = _site_types(spec)

        b_open, b_close = (int(x) for x in spec.landfill_open_close["baseline"])
        s_dates = spec.landfill_open_close["scenario"] or spec.landfill_open_close["baseline"]
        s_open, s_close = (int(x) for x in s_dates)

        gas_base, gas_scen = _gas_capture(spec, years, implement_year)
        flare_base, flare_scen = common.variant_series(spec.flaring, years, implement_year, default=common.DEFAULT_FLARE_EFFICIENCY)
        bio_base, bio_scen = common.variant_series(spec.biocover, years, implement_year, default=0.0)
        share_base = baseline_split[index]
        share_scen = scenario_split[index]

        base_combusts = bool(common.variant_get(spec.combusts, "baseline") or False)
        scen_combusts = bool(common.variant_get(spec.combusts, "scenario") or False)

        base_depth = common.variant_get(spec.depth, "baseline")
        scen_depth = common.variant_get(spec.depth, "scenario")
        mcf_base = common.mcf_series(
            base_type, base_type, implement_year, years, base_depth, base_depth
        )
        mcf_scen = common.mcf_series(
            base_type, scen_type, implement_year, years, base_depth, scen_depth
        )
        ox_override = _override_series(
            spec.oxidation_override, years, implement_year,
            field="oxidation_override", upper=1.0, optional_years=True,
        )
        ox_override_base, ox_override_scen = ox_override or (None, None)
        ox_base = common.oxidation_series(
            base_type, base_type, gas_base, bio_base, implement_year, years,
            override=ox_override_base,
        )
        ox_scen = common.oxidation_series(
            base_type, scen_type, gas_scen, bio_scen, implement_year, years,
            override=ox_override_scen,
        )
        baseline_ox.append(ox_base)
        scenario_ox.append(ox_scen)
        ks_site_base, ks_site_scen = _site_decomposition_rates(
            spec, (ks_baseline, ks_scenario), years, implement_year
        )

        # Net-of-diversion city waste, scaled to this landfill's per-year share.
        mass_base = LandfillWasteMassDF.create_advanced(wgen_baseline, divs_baseline, share_base.copy()).df
        mass_scen = LandfillWasteMassDF.create_advanced(wgen_scenario, divs_scenario, share_scen.copy()).df

        # What the city sends here, at the gate -- read before the combustion
        # and window lines below, because a stated gate total is measured at the
        # weighbridge and a site that burns its intake still accepted all of it.
        surplus_base_masses = surplus_scen_masses = None
        if spec.accepted_waste_mass is not None:
            accepted_base, accepted_scen = common.variant_series(
                spec.accepted_waste_mass, years, implement_year, default=0.0
            )
            surplus_base_masses = site_inflow.surplus_masses(
                site_inflow.surplus_intake(accepted_base, mass_base.sum(axis=1)),
                residual_mix["baseline"],
            )
            surplus_scen_masses = site_inflow.surplus_masses(
                site_inflow.surplus_intake(accepted_scen, mass_scen.sum(axis=1)),
                residual_mix["scenario"],
            )

        # A combusting facility burns its intake and deposits only the reject.
        # Scaled per variant here, ahead of both the window and the pre-implement
        # splice below, so a site that starts combusting at implement_year keeps
        # depositing everything until then -- and one that stops burning starts to.
        mass_base = mass_base * _deposited_share(base_combusts, city)
        mass_scen = mass_scen * _deposited_share(scen_combusts, city)
        # Each variant is windowed on its own open/close years first, and only
        # then does the scenario take baseline's pre-implement rows. The other
        # order spliced baseline's mass in and let the *scenario* window zero it
        # again: a site stated as opening later in the scenario than in the
        # baseline buried nothing in the years between, though those years are
        # before `implement_year` and are supposed to match baseline exactly.
        mass_base = common.apply_window(mass_base, b_open, b_close)
        mass_scen = common.apply_window(mass_scen, s_open, s_close)
        mass_scen.loc[: implement_year - 1, :] = mass_base.loc[: implement_year - 1, :]
        baseline_masses.append(mass_base)
        scenario_masses.append(mass_scen)

        # The surplus is the same waste in the same hole, so it takes the same
        # three transformations in the same order. Anything else and the two
        # streams disagree at a boundary -- a closure year, an implement year,
        # the year a site starts burning -- which is exactly where a reader
        # would notice and could not explain it.
        if surplus_base_masses is not None:
            surplus_base_masses = surplus_base_masses * _deposited_share(base_combusts, city)
            surplus_scen_masses = surplus_scen_masses * _deposited_share(scen_combusts, city)
            surplus_base_masses = common.apply_window(surplus_base_masses, b_open, b_close)
            surplus_scen_masses = common.apply_window(surplus_scen_masses, s_open, s_close)
            surplus_scen_masses.loc[: implement_year - 1, :] = surplus_base_masses.loc[
                : implement_year - 1, :
            ]
            # A stated total at or below the city's allocation in every year
            # leaves nothing to model, and running the kernel on an all-zero
            # frame would only produce an all-zero frame.
            if not (
                surplus_base_masses.to_numpy().any() or surplus_scen_masses.to_numpy().any()
            ):
                surplus_base_masses = surplus_scen_masses = None
        baseline_surplus_masses.append(surplus_base_masses)
        scenario_surplus_masses.append(surplus_scen_masses)

        # Lifted into a dict so the surplus stream below is built from the very
        # same arguments rather than a copy of them that has to be kept in step:
        # two streams at one site must agree on every parameter but their mass,
        # or they decay differently and stop adding up to the site.
        base_landfill_kwargs = dict(
            open_year=b_open, close_year=b_close, site_type_idx=base_type,
            mcf=mcf_base, gas_capture_efficiency=gas_base, flaring=flare_base,
            oxidation_factor=ox_base, ks=ks_site_base, city_params_dict=baseline_params_dict,
            city_instance_attrs=city_instance_attrs, implement_year=implement_year,
            scenario=0, landfill_index=index,
        )
        baseline_landfills.append(common.build_landfill(**base_landfill_kwargs))
        # The model evaluates from `open_date` onward (`model_v2.estimate_emissions2`
        # builds its year range from it), so the scenario has to start wherever
        # its mass frame can first be nonzero -- and the splice above puts
        # baseline's pre-implement mass into it. A scenario stated as opening
        # later than baseline therefore still buries waste from baseline's open
        # year, and starting the model at `s_open` truncated those years out of
        # the scenario's emissions frame entirely, leaving the two halves a
        # different shape for a caller trying to subtract them.
        scenario_open_for_model = min(b_open, s_open)
        scen_landfill_kwargs = dict(
            open_year=scenario_open_for_model, close_year=s_close, site_type_idx=scen_type,
            mcf=mcf_scen, gas_capture_efficiency=gas_scen, flaring=flare_scen,
            oxidation_factor=ox_scen, ks=ks_site_scen, city_params_dict=scenario_params_dict,
            city_instance_attrs=city_instance_attrs, implement_year=implement_year,
            scenario=1, landfill_index=index,
        )
        scenario_landfills.append(common.build_landfill(**scen_landfill_kwargs))

        # The surplus twin. Same open year above all else: `model_v2` builds its
        # year range from `open_date`, so a twin opened anywhere else returns a
        # differently indexed frame and the site total silently gains NaNs where
        # the two failed to line up.
        baseline_surplus.append(
            common.build_landfill(**base_landfill_kwargs)
            if surplus_base_masses is not None
            else None
        )
        scenario_surplus.append(
            common.build_landfill(**scen_landfill_kwargs)
            if surplus_scen_masses is not None
            else None
        )

    # --- Wire up and run the engine ---
    baseline_parameters.landfills = baseline_landfills
    scenario_parameters.landfills = scenario_landfills
    baseline_parameters.repopulate_attr_dicts()
    scenario_parameters.repopulate_attr_dicts()

    for landfill, mass, ox in zip(baseline_landfills, baseline_masses, baseline_ox):
        landfill.waste_mass_df = mass
        landfill.oxidation_factor = ox
        landfill.estimate_emissions(skip_ox=True)
    for landfill, mass, ox in zip(scenario_landfills, scenario_masses, scenario_ox):
        landfill.waste_mass_df = mass
        landfill.oxidation_factor = ox
        landfill.estimate_emissions(skip_ox=True)

    # The surplus streams, run the same way -- but only after
    # `repopulate_attr_dicts` above has settled the city's parameter dict, which
    # these never receive because they are not in `parameters.landfills`.
    for streams, landfills, masses, oxidations in (
        (baseline_surplus, baseline_landfills, baseline_surplus_masses, baseline_ox),
        (scenario_surplus, scenario_landfills, scenario_surplus_masses, scenario_ox),
    ):
        for surplus, city_stream, mass, ox in zip(streams, landfills, masses, oxidations):
            if surplus is None:
                continue
            site_inflow.adopt_city_params(city_stream, surplus)
            surplus.waste_mass_df = mass
            surplus.oxidation_factor = ox
            surplus.estimate_emissions(skip_ox=True)

    city.baseline_parameters = baseline_parameters
    city.scenario_parameters[0] = scenario_parameters

    # Diversion (compost/anaerobic) emissions, then aggregate with landfills.
    city.estimate_diversion_emissions(scenario=0)
    city.estimate_diversion_emissions(scenario=1)

    # mef_compost is a per-variant scalar applied to every year, so a scenario
    # whose composition differs from baseline would emit different compost
    # emissions even before implement_year (the diverted masses are spliced, but
    # the scalar factor is not). Re-impose the baseline == scenario-before-
    # implement_year convention on organic emissions, matching every other quantity.
    scenario_parameters.organic_emissions.loc[: implement_year - 1, :] = (
        baseline_parameters.organic_emissions.loc[: implement_year - 1, :]
    )

    city.sum_landfill_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=1)

    result: dict = {
        "baseline": baseline_parameters.total_emissions,
        "scenario": scenario_parameters.total_emissions,
    }
    if with_site_emissions:
        result["site_emissions"] = {
            "baseline": [
                site_inflow.emission_frames(city_stream, surplus)
                for city_stream, surplus in zip(baseline_landfills, baseline_surplus)
            ],
            "scenario": [
                site_inflow.emission_frames(city_stream, surplus)
                for city_stream, surplus in zip(scenario_landfills, scenario_surplus)
            ],
        }
    if with_methane_budget:
        result["methane_budget"] = {
            "baseline": _methane_budget(baseline_parameters),
            "scenario": _methane_budget(scenario_parameters),
        }
    if committed_before is not None:
        before = int(committed_before)
        result["committed"] = {
            "before": before,
            "baseline": _committed_emissions(baseline_parameters, before),
            "scenario": _committed_emissions(scenario_parameters, before),
        }
    if with_mass_flow:
        result["mass_flow"] = {
            "baseline": _mass_flow(
                generated_baseline, wgen_baseline, divs_baseline,
                net_masses["baseline"], baseline_masses,
            ),
            "scenario": _mass_flow(
                generated_scenario, wgen_scenario, divs_scenario,
                net_masses["scenario"], scenario_masses,
            ),
        }
    return result
