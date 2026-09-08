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
    open/close years. Those totals do NOT always sum to ``landfilled``: a year
    in which some share is allocated to a closed landfill loses that mass, since
    the split timeline is honoured as submitted rather than renormalized over
    whichever landfills happen to be open. The gap is reported rather than
    hidden — ``landfilled`` is what the city disposed of, ``sites`` is what
    arrived somewhere.
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
) -> DivsDF:
    """Build a DivsDF of per-pathway, per-component diverted mass (tons/yr).

    For each pathway: mass into the pathway = pathway_fraction * total_generated;
    that mass is split across the pathway's eligible components using the city
    composition (normalized within those components); then reject/yield rates are
    applied. Matches City._calculate_diverted_masses, minus the legacy shims.
    """
    div_dfs: Dict[str, pd.DataFrame] = {}
    for pathway in DIVERSION_PATHWAYS:
        components = sorted(city.div_components[pathway])  # deterministic column order
        sub = fractions_df[components]
        denom = sub.sum(axis=1)
        split = sub.div(denom, axis=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        pathway_fraction = common.yearly_to_series(div_fracs.get(pathway), years, default=0.0)
        mass_into_pathway = pathway_fraction * total_generated
        gross = split.mul(mass_into_pathway, axis=0)  # years x components

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
    unburnable reject -- the same ``combustion_reject_rate`` the combustion
    diversion pathway applies, so an incinerator modeled as a facility and one
    modeled as upstream combustion leave the same residue behind.
    """
    return float(city.combustion_reject_rate) if combusts else 1.0


def _validate_shares(shares: List[pd.Series], years: pd.Index, label: str) -> None:
    """Each year's landfill shares should sum to ~1 (all net waste is landfilled somewhere)."""
    total = sum(shares)
    bad = total[(total < 1.0 - SHARE_SUM_TOLERANCE) | (total > 1.0 + SHARE_SUM_TOLERANCE)]
    if not bad.empty:
        first_year = int(bad.index[0])
        raise CustomError(
            "invalid_parameters",
            f"{label} landfill waste_share fractions must sum to ~1 per year "
            f"(year {first_year} sums to {float(bad.iloc[0]):.3f}).",
        )


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

    divs_baseline = _diverted_masses(baseline_fractions, baseline_total, baseline_div, city, years)
    divs_scenario = _splice_divs(
        divs_baseline,
        _diverted_masses(scenario_fractions, scenario_total, scenario_div, city, years),
        implement_year,
    )

    # Guard against diverting more than is generated (negative landfilled mass).
    # The residual is kept rather than discarded: it is the landfilled mass per
    # waste type, which is what the mass flow reports and what the per-landfill
    # split is taken from.
    net_masses: Dict[str, pd.DataFrame] = {}
    for label, wgen, divs in (("baseline", wgen_baseline, divs_baseline), ("scenario", wgen_scenario, divs_scenario)):
        net = wgen.sub(divs.sum(), fill_value=0.0)
        if (net < -1e-6).to_numpy().any():
            raise CustomError(
                "over_diversion",
                f"{label}: diversion exceeds generated waste (negative landfilled mass).",
            )
        net_masses[label] = net

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
    # Scenario tracks baseline before changes take effect.
    scenario_split.loc[: implement_year - 1, :] = baseline_split.loc[: implement_year - 1, :]

    # --- Build a landfill (baseline + scenario) per spec ---
    baseline_landfills: List = []
    scenario_landfills: List = []
    baseline_masses: List[pd.DataFrame] = []
    scenario_masses: List[pd.DataFrame] = []
    baseline_ox: List[pd.Series] = []
    scenario_ox: List[pd.Series] = []
    baseline_shares: List[pd.Series] = []
    scenario_shares: List[pd.Series] = []

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
        baseline_shares.append(share_base)
        scenario_shares.append(share_scen)

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

        baseline_landfills.append(common.build_landfill(
            open_year=b_open, close_year=b_close, site_type_idx=base_type,
            mcf=mcf_base, gas_capture_efficiency=gas_base, flaring=flare_base,
            oxidation_factor=ox_base, ks=ks_baseline, city_params_dict=baseline_params_dict,
            city_instance_attrs=city_instance_attrs, implement_year=implement_year,
            scenario=0, landfill_index=index,
        ))
        # The model evaluates from `open_date` onward (`model_v2.estimate_emissions2`
        # builds its year range from it), so the scenario has to start wherever
        # its mass frame can first be nonzero -- and the splice above puts
        # baseline's pre-implement mass into it. A scenario stated as opening
        # later than baseline therefore still buries waste from baseline's open
        # year, and starting the model at `s_open` truncated those years out of
        # the scenario's emissions frame entirely, leaving the two halves a
        # different shape for a caller trying to subtract them.
        scenario_open_for_model = min(b_open, s_open)
        scenario_landfills.append(common.build_landfill(
            open_year=scenario_open_for_model, close_year=s_close, site_type_idx=scen_type,
            mcf=mcf_scen, gas_capture_efficiency=gas_scen, flaring=flare_scen,
            oxidation_factor=ox_scen, ks=ks_scenario, city_params_dict=scenario_params_dict,
            city_instance_attrs=city_instance_attrs, implement_year=implement_year,
            scenario=1, landfill_index=index,
        ))

    _validate_shares(baseline_shares, years, "baseline")
    _validate_shares(scenario_shares, years, "scenario")

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
