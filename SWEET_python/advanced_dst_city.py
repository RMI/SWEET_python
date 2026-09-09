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
  (so the frontend only sends overall pathway fractions, not per-component splits).
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

Diversion math, reject rates, the per-landfill split, and emissions aggregation
mirror the tested ``City`` machinery (``_calculate_diverted_masses``,
``LandfillWasteMassDF.create_advanced``, ``estimate_diversion_emissions``,
``sum_landfill_emissions``); the leaf primitives are reused directly.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import SWEET_python.defaults_2019 as defaults_2019
from SWEET_python.city_params import City, CityParameters, CustomError
from SWEET_python.class_defs import DivsDF, LandfillType, LandfillWasteMassDF, Variant
from SWEET_python import dst_common as common
from SWEET_python.dst_common import YearlyFloat, YearlyFractions

__all__ = ["AdvancedDSTCityRequest", "CityLandfillSpec", "run_advanced_dst_city"]

#: The diversion pathways `DivsDF` carries, in the order the mass flow reports.
DIVERSION_PATHWAYS: tuple[str, ...] = ("compost", "anaerobic", "combustion", "recycling")
SHARE_SUM_TOLERANCE = 0.02


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
    gas_capture_efficiency: Variant[YearlyFloat] = Field(
        ..., description="Fraction of gas captured per year; 0 means no capture."
    )
    flaring: Optional[Variant[YearlyFloat]] = Field(
        None, description="Flare destruction efficiency per year (defaults to 0.98)."
    )
    biocover: Optional[Variant[YearlyFloat]] = Field(
        None, description="Biocover oxidation floor per year (a fraction; defaults to 0)."
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
        ..., min_length=1, description="One entry per landfill in the city."
    )
    landfill_split_timeline: Variant[Dict[int, List[float]]] = Field(
        ...,
        description=(
            "Fraction of the city's landfilled (net-of-diversion) waste sent to "
            "each landfill per year: {year: [frac_landfill_0, frac_landfill_1, ...]}. "
            "The list order matches the `landfills` list, and each year's fractions "
            "should sum to ~1."
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
    on organics and recycling only on recyclables, each with its own reject
    rate, so the split is per waste type and approximating it as one scalar on
    the total quietly composts metal and glass.

    ``sites`` is per landfill in request order, already windowed by each one's
    open/close years. It sums to ``landfilled`` in every year that has at least
    one open landfill, because the split is renormalized over whichever ones are
    accepting waste (see ``_renormalize_over_open``). The one case where the two
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


def _diverted_masses(
    fractions_df: pd.DataFrame,
    total_generated: pd.Series,
    div_fracs: Dict[str, Dict[int, float]],
    city: City,
    years: pd.Index,
) -> tuple[DivsDF, DivsDF]:
    """Per-pathway, per-component diverted mass (tons/yr): ``(net, gross)``.

    For each pathway: mass into the pathway = pathway_fraction * total_generated;
    that mass is split across the pathway's eligible components using the city
    composition (normalized within those components); then reject/yield rates are
    applied. Matches City._calculate_diverted_masses, minus the legacy shims.

    Both bases are returned because they answer different questions. ``net`` is
    what actually leaves the waste stream — rejects stay in it and are landfilled
    as their own material — so it is what the landfilled residual and the mass
    flow are built from. ``gross`` is what was *demanded* of each component, and
    that is the basis over-diversion has to be measured on: a pathway fed more of
    a material than the city generates is impossible whether or not enough of it
    is later rejected back.
    """
    div_dfs: Dict[str, pd.DataFrame] = {}
    gross_dfs: Dict[str, pd.DataFrame] = {}
    for pathway in DIVERSION_PATHWAYS:
        components = sorted(city.div_components[pathway])  # deterministic column order
        sub = fractions_df[components]
        denom = sub.sum(axis=1)

        pathway_fraction = common.yearly_to_series(div_fracs.get(pathway), years, default=0.0)

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

        split = sub.div(denom, axis=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        mass_into_pathway = pathway_fraction * total_generated
        gross = split.mul(mass_into_pathway, axis=0)  # years x components
        gross_dfs[pathway] = gross

        if pathway == "compost":
            ncnt = city.non_compostable_not_targeted
            ncnt_total = sum(split[c] * float(ncnt.get(c, 0.0)) for c in components)
            net = gross.mul(1.0 - ncnt_total, axis=0)
            keep = pd.Series({c: 1.0 - float(city.unprocessable.get(c, 0.0)) for c in components})
            net = net.mul(keep, axis=1)
        elif pathway == "anaerobic":
            net = gross  # no loss
        elif pathway == "combustion":
            net = gross * (1.0 - float(city.combustion_reject_rate))
        else:  # recycling — reject_rates are yield multipliers (fraction kept)
            yields = pd.Series({c: float(city.recycling_reject_rates.get(c, 1.0)) for c in components})
            net = gross.mul(yields, axis=1)

        div_dfs[pathway] = net

    def collect(frames: Dict[str, pd.DataFrame]) -> DivsDF:
        return DivsDF(
            compost=frames["compost"],
            anaerobic=frames["anaerobic"],
            combustion=frames["combustion"],
            recycling=frames["recycling"],
        )

    return collect(div_dfs), collect(gross_dfs)


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
    unburnable reject -- the same ``combustion_reject_rate`` the combustion
    diversion pathway applies, so an incinerator modeled as a facility and one
    modeled as upstream combustion leave the same residue behind.
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
    somewhere_open_rows = accepting.any(axis=1)
    checked = split.loc[somewhere_open_rows]
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
# more organics than a city generates. ``_diverted_masses`` splits a pathway's
# mass across its eligible components *proportionally to the composition*, which
# makes the constraint closed-form rather than a search.
#
# With ``f_c`` the normalized fraction of component ``c`` and
# ``D_p = sum(f_c for c in components(p))``, pathway ``p`` consumes
# ``divfrac_p / D_p`` of every component it touches. So per component:
#
#     use(c) = sum(divfrac_p / D_p  for p containing c)  <=  1
#
# The binding component is whichever has the largest ``use``. This is stated per
# component rather than as an organic/recyclable pool pair because the pathways
# do not partition into two: compost and anaerobic digestion share a component
# set, recycling overlaps it on wood and paper, and combustion draws on all ten.
# Summing per component needs no special case for any of that.
#
# This bounds GROSS diversion, before reject rates, which is the same basis
# ``run_advanced_dst_city``'s ``over_diversion`` guard measures on. The two are
# the same boundary on purpose: a caller respecting these limits cannot produce a
# request the model rejects on mass balance.
SHARE_LIMIT_TOLERANCE = 1e-9


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
    div_fracs: Dict[str, pd.Series],
    city: City,
    skip: Optional[str] = None,
) -> pd.DataFrame:
    """Fraction of each component's own mass that the pathways claim, per year.

    ``skip`` leaves one pathway out, which is what makes "how much more can this
    pathway take" answerable: the rest of the demand is what it has to fit into.
    """
    denominators = _component_denominators(fractions, city)
    use = pd.DataFrame(0.0, index=fractions.index, columns=fractions.columns)
    for pathway in DIVERSION_PATHWAYS:
        if pathway == skip:
            continue
        denominator = denominators[pathway]
        # A pathway with none of its components present draws on nothing; its
        # share of a zero-size pool is undefined rather than infinite.
        claim = div_fracs[pathway].divide(denominator).replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)
        for component in sorted(city.div_components[pathway]):
            use[component] = use[component] + claim
    return use


def _max_pathway_fraction(
    fractions: pd.DataFrame,
    div_fracs: Dict[str, pd.Series],
    city: City,
    pathway: str,
) -> pd.Series:
    """Largest fraction of generated waste ``pathway`` may take, per year.

    Every other pathway is held at its own value for that year. The pathway is
    bounded by each component it draws on: what is left of that component after
    the others have taken their share, scaled back up by ``D_p``.
    """
    components = sorted(city.div_components[pathway])
    others = _component_use(fractions, div_fracs, city, skip=pathway)
    headroom = (1.0 - others[components]).min(axis=1).clip(lower=0.0)
    denominator = _component_denominators(fractions, city)[pathway]
    return (headroom * denominator).clip(lower=0.0, upper=1.0)


def _max_food_waste_prevention(
    fractions: pd.DataFrame,
    total: pd.Series,
    div_fracs: Dict[str, pd.Series],
    city: City,
    tolerance: float = 1e-4,
) -> pd.Series:
    """Largest food-waste-prevention fraction each year's diversion demand permits.

    Prevention shrinks the organic pool — it removes food — so a compost fraction
    that fitted before it takes effect can stop fitting after. Bisected rather
    than solved: removing food raises the organic pathways' claim while lowering
    recycling's, and which component binds can change with it, so the closed form
    is per component but the envelope over components is not monotone in general.

    Per year, and deliberately so. Folding dated interventions into a per-year
    series is the caller's concern — a cap for one intervention is the minimum of
    these over the years it is in force — and the model has no notion of one.
    """
    def fits(shares: pd.Series) -> pd.Series:
        """Whether each year's diversion demand survives that year's prevention.

        Takes a share *per year* rather than one scalar, because every step
        below is row-wise — ``_prevent_food_waste`` renormalizes within a year,
        the denominators are per-year sums, and ``starved`` is a per-year mask.
        So one call answers every year at its own probe, and the bisection needs
        one per halving instead of one per year per halving.
        """
        shrunk_fractions, _ = _prevent_food_waste(fractions, total, shares)
        use = _component_use(shrunk_fractions, div_fracs, city)
        within = use.max(axis=1) <= 1.0 + SHARE_LIMIT_TOLERANCE

        # A pathway asked for a share of a pool that prevention has emptied is
        # not satisfiable, but the per-component arithmetic reads it as zero
        # rather than as too much: with D_p at zero, `_diverted_masses` splits
        # the pathway's mass by 0/0 and diverts nothing at all. So the ratio is
        # continuous right up to an empty pool and then drops to nothing, and a
        # probe landing exactly on the empty case would call it feasible.
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
    return low.where(~unconstrained, 1.0)


def _limits_for_variant(
    fractions: pd.DataFrame,
    total: pd.Series,
    prevented: pd.Series,
    div_fracs: Dict[str, pd.Series],
    city: City,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """One variant's bounds, from that variant's own inputs."""
    prevented_fractions, _ = _prevent_food_waste(fractions, total, prevented)
    use = _component_use(prevented_fractions, div_fracs, city)

    # A pathway asked for a share of a pool with nothing in it is infeasible,
    # and `component_use` cannot say so: with `D_p` at zero the claim is 0/0,
    # which `_component_use` reads as zero rather than as too much, so the row
    # looks comfortably under 1. `run_advanced_dst_city` raises
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
    ``_diverted_masses`` splits on, and the basis is gross diverted mass, the
    same one the ``over_diversion`` guard measures. A caller respecting these
    bounds cannot produce a request the model rejects on mass balance.

    All of it reflects the prevention already in the request; a caller asking
    "how much further could I go" gets an answer consistent with where it is.
    """
    implement_year = int(request.implement_year)
    open_close_pairs = []
    for spec in request.landfills:
        b_open, b_close = (int(x) for x in spec.landfill_open_close["baseline"])
        s_dates = spec.landfill_open_close["scenario"] or spec.landfill_open_close["baseline"]
        s_open, s_close = (int(x) for x in s_dates)
        open_close_pairs.append((b_open, b_close))
        open_close_pairs.append((s_open, s_close))
    model_start = common.validate_years(open_close_pairs, implement_year)
    years = pd.Index(range(model_start, common.MODEL_YEAR_MAX + 1), name="year")

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
# Public entry point
# --------------------------------------------------------------------------- #
def run_advanced_dst_city(
    request: AdvancedDSTCityRequest, *, with_mass_flow: bool = False
) -> dict:
    """Run the city-level advanced DST.

    Returns ``{"baseline": total_emissions_df, "scenario": total_emissions_df}``
    (city totals summed across landfills + diversion), each indexed by year with a
    ``total`` column.

    With ``with_mass_flow=True`` the result carries an extra ``"mass_flow"`` key
    describing where the tonnage went, per variant: ``generated`` (per component,
    before prevention), ``prevented``, ``diverted`` (per pathway per component,
    net of reject rates), ``landfilled`` (the residual per component), and
    ``sites`` (per landfill, in request order). A diversion pathway the request
    never uses is omitted from ``diverted`` rather than sent as zeros — the
    payload is several times the emissions one, so this is worth the asymmetry;
    a missing pathway reads as zero. Those are the very frames the
    emissions are computed from, not a second derivation — callers that need to
    *show* the mass flow should read it from here rather than reimplementing the
    diversion split, which is per waste type and does not survive being
    approximated as a scalar on the total.
    """
    implement_year = int(request.implement_year)

    # --- Years: validate every landfill's open/close window ---
    open_close_pairs = []
    for spec in request.landfills:
        b_open, b_close = (int(x) for x in spec.landfill_open_close["baseline"])
        s_dates = spec.landfill_open_close["scenario"] or spec.landfill_open_close["baseline"]
        s_open, s_close = (int(x) for x in s_dates)
        open_close_pairs.append((b_open, b_close))
        open_close_pairs.append((s_open, s_close))
    model_start = common.validate_years(open_close_pairs, implement_year)
    years = pd.Index(range(model_start, common.MODEL_YEAR_MAX + 1), name="year")

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

    divs_baseline, gross_baseline = _diverted_masses(
        baseline_fractions, baseline_total, baseline_div, city, years
    )
    divs_scenario_own, gross_scenario_own = _diverted_masses(
        scenario_fractions, scenario_total, scenario_div, city, years
    )
    divs_scenario = _splice_divs(divs_baseline, divs_scenario_own, implement_year)
    gross_scenario = _splice_divs(gross_baseline, gross_scenario_own, implement_year)

    # Guard against diverting more of a material than the city generates.
    #
    # Measured on GROSS diverted mass, before reject rates. Rejects stay in the
    # waste stream and are landfilled as their own material, so a net-basis check
    # passes a pathway that was fed material which does not exist as long as
    # enough of it is rejected back -- the city composts phantom wood and then
    # landfills the rejects of it. Gross is also exactly the boundary
    # ``run_advanced_dst_city_limits`` reports, so a caller respecting those
    # bounds cannot trip this.
    #
    # The net residual is kept rather than discarded: it is the landfilled mass
    # per waste type, which is what the mass flow reports and what the
    # per-landfill split is taken from.
    net_masses: Dict[str, pd.DataFrame] = {}
    for label, wgen, divs, gross in (
        ("baseline", wgen_baseline, divs_baseline, gross_baseline),
        ("scenario", wgen_scenario, divs_scenario, gross_scenario),
    ):
        demanded = wgen.sub(gross.sum(), fill_value=0.0)
        if (demanded < -1e-6).to_numpy().any():
            shortfall = demanded.min()
            component = str(shortfall.idxmin())
            year = int(demanded[component].idxmin())
            raise CustomError(
                "over_diversion",
                f"{label}: the diversion pathways are sent more {component} in "
                f"{year} than the city generates.",
            )
        net_masses[label] = wgen.sub(divs.sum(), fill_value=0.0)

    # --- City-wide decomposition rates + compost emission factors ---
    ref_year = min(max(implement_year, int(years.min())), int(years.max()))
    ks_baseline, ks_scenario = common.decomposition_rates(
        request.temperature, request.precipitation, implement_year, years,
        common.representative_vector(baseline_fractions, ref_year),
        common.representative_vector(scenario_fractions, ref_year),
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

    for index, spec in enumerate(request.landfills):
        base_type = int(spec.landfill_type["baseline"])
        scen_type = int(spec.landfill_type["scenario"]) if spec.landfill_type["scenario"] is not None else base_type

        b_open, b_close = (int(x) for x in spec.landfill_open_close["baseline"])
        s_dates = spec.landfill_open_close["scenario"] or spec.landfill_open_close["baseline"]
        s_open, s_close = (int(x) for x in s_dates)

        gas_base, gas_scen = common.variant_series(spec.gas_capture_efficiency, years, implement_year, default=0.0)
        # A capture efficiency is a fraction of generated methane, so it cannot
        # exceed 1. Nothing downstream bounds it -- the field is typed as a bare
        # float map and the engine multiplies straight through -- so an out-of-range
        # value silently produced negative emissions rather than an error.
        gas_base = gas_base.clip(0.0, 1.0)
        gas_scen = gas_scen.clip(0.0, 1.0)
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
        ox_base = common.oxidation_series(base_type, base_type, gas_base, bio_base, implement_year, years)
        ox_scen = common.oxidation_series(base_type, scen_type, gas_scen, bio_scen, implement_year, years)
        baseline_ox.append(ox_base)
        scenario_ox.append(ox_scen)

        # Net-of-diversion city waste, scaled to this landfill's per-year share.
        mass_base = LandfillWasteMassDF.create_advanced(wgen_baseline, divs_baseline, share_base.copy()).df
        mass_scen = LandfillWasteMassDF.create_advanced(wgen_scenario, divs_scenario, share_scen.copy()).df
        # A combusting facility burns its intake and deposits only the reject.
        # Scaled per variant here, before the pre-implement splice below, so a
        # site that starts combusting at implement_year keeps depositing
        # everything until then -- and one that stops burning starts to.
        mass_base = mass_base * _deposited_share(base_combusts, city)
        mass_scen = mass_scen * _deposited_share(scen_combusts, city)
        mass_scen.loc[: implement_year - 1, :] = mass_base.loc[: implement_year - 1, :]
        mass_base = common.apply_window(mass_base, b_open, b_close)
        mass_scen = common.apply_window(mass_scen, s_open, s_close)
        baseline_masses.append(mass_base)
        scenario_masses.append(mass_scen)

        baseline_landfills.append(common.build_landfill(
            open_year=b_open, close_year=b_close, site_type_idx=base_type,
            mcf=mcf_base, gas_capture_efficiency=gas_base, flaring=flare_base,
            oxidation_factor=ox_base, ks=ks_baseline, city_params_dict=baseline_params_dict,
            city_instance_attrs=city_instance_attrs, implement_year=implement_year,
            scenario=0, landfill_index=index,
        ))
        scenario_landfills.append(common.build_landfill(
            open_year=s_open, close_year=s_close, site_type_idx=scen_type,
            mcf=mcf_scen, gas_capture_efficiency=gas_scen, flaring=flare_scen,
            oxidation_factor=ox_scen, ks=ks_scenario, city_params_dict=scenario_params_dict,
            city_instance_attrs=city_instance_attrs, implement_year=implement_year,
            scenario=1, landfill_index=index,
        ))

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
