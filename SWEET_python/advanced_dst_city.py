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
* each landfill carries its own type / open-close / gas capture / flaring /
  biocover. The split of the city's *landfilled* (net-of-diversion) waste across
  landfills is a top-level time series, ``landfill_split_timeline``:
  ``{year: [frac per landfill]}`` ordered to match the ``landfills`` list, with
  each year's fractions summing to ~1.

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

DIVERSION_PATHWAYS = ["compost", "anaerobic", "combustion", "recycling"]
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
    temperature: float = Field(10.0, description="Average annual temperature, deg C.")
    country: Optional[str] = Field(None, description="ISO3 country code (identity).")
    rmi_id: Optional[int] = Field(None, description="City/site identifier (identity).")


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
def run_advanced_dst_city(request: AdvancedDSTCityRequest) -> dict[str, pd.DataFrame]:
    """Run the city-level advanced DST.

    Returns ``{"baseline": total_emissions_df, "scenario": total_emissions_df}``
    (city totals summed across landfills + diversion), each indexed by year with a
    ``total`` column.
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
    for label, wgen, divs in (("baseline", wgen_baseline, divs_baseline), ("scenario", wgen_scenario, divs_scenario)):
        net = wgen.sub(divs.sum(), fill_value=0.0)
        if (net < -1e-6).to_numpy().any():
            raise CustomError(
                "over_diversion",
                f"{label}: diversion exceeds generated waste (negative landfilled mass).",
            )

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

    return {
        "baseline": baseline_parameters.total_emissions,
        "scenario": scenario_parameters.total_emissions,
    }
