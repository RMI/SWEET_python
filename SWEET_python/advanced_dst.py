"""
Advanced DST (adst) — single-site methane modeling.

This is the modeling backend for the ``/v1/site_emissions/adst`` endpoint. It is
the "advanced" successor to ``City.sdst_v1_5``, but deliberately much simpler:
the caller supplies waste mass and composition directly as time series, so this
module does *no* waste generation, population growth, or TRACE reconciliation.
It just turns the supplied numbers into a baseline and a scenario emissions
time series.

Design notes
------------
* Inputs arrive as :class:`SWEET_python.class_defs.Variant` objects carrying a
  ``baseline`` and an optional ``scenario`` value. When ``scenario`` is omitted
  it falls back to ``baseline``.
* The scenario variant equals the baseline variant for every year before
  ``implement_year`` and switches to the scenario inputs from ``implement_year``
  onward — the same convention the rest of the SWEET model uses.
* ``run_advanced_dst`` is a pure function: it returns the result and never
  mutates shared state. It builds a throwaway :class:`City`/:class:`Landfill`
  internally purely to reuse the tested emission engine and summation logic.
* Eventually this module is meant to grow to handle whole-city, multi-site, and
  diversion modeling together; for now it handles one site at a time.
"""

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import SWEET_python.defaults_2019 as defaults_2019
from SWEET_python.city_params import City, CityParameters, CustomError
from SWEET_python.class_defs import DecompositionRates, Fraction, LandfillType, Variant
from SWEET_python.landfill import Landfill
from SWEET_python.singapore_k import compute_singapore_k

__all__ = ["AdvancedDSTRequest", "run_advanced_dst"]

# The model is only ever evaluated over this window. Open years are validated
# against MODEL_YEAR_MIN to avoid accidental giant simulations.
MODEL_YEAR_MIN = 1950
MODEL_YEAR_MAX = 2050

# Order matters: waste_fractions vectors are sent in this column order.
WASTE_COMPONENTS: List[str] = [
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
]
# The SWEET engine only generates methane from the biodegradable components
# (these are the components that have a k value and an L_0).
DEGRADABLE_COMPONENTS: List[str] = [
    "food",
    "green",
    "wood",
    "paper_cardboard",
    "textiles",
]

# Methane correction factor by landfill type (index == LandfillType value).
# 0 landfill, 1 controlled dump, 2 open dump.
MCF_BY_TYPE: List[float] = [1.0, 0.6, 0.4]
SITE_TYPE_NAMES: List[str] = ["landfill", "controlled_dumpsite", "dumpsite"]

# Oxidation factor lookup, mirroring City.sdst_v1_5 / Landfill.estimate_emissions.
OX_NOCAP: Dict[str, float] = {"landfill": 0.1, "controlled_dumpsite": 0.05, "dumpsite": 0.0}
OX_CAP: Dict[str, float] = {"landfill": 0.22, "controlled_dumpsite": 0.1, "dumpsite": 0.0}
# A dump that is upgraded ("ameliorated") to an engineered landfill with gas
# capture gets a higher oxidation factor than a landfill that always existed.
OX_REMEDIATED_TO_LANDFILL = 0.18

DEFAULT_FLARE_EFFICIENCY = 0.98


# A yearly time series arrives over the wire as {year: value}. Pydantic coerces
# JSON string keys ("2025") to ints.
YearlyFloat = Dict[int, float]
YearlyFractions = Dict[int, List[float]]


class AdvancedDSTRequest(BaseModel):
    """Request body for the advanced single-site DST endpoint.

    Every per-scenario input is a :class:`Variant`. The frontend sends the raw
    UI numbers (a total-mass series and a fractions series); all math, including
    splitting the total into components, happens here in Python.
    """

    precipitation: float = Field(..., ge=0, description="Average annual precipitation, mm/year.")
    implement_year: int = Field(..., description="Year scenario changes take effect.")
    waste_mass: Variant[YearlyFloat] = Field(
        ..., description="Total landfilled waste mass per year (tons), {year: tons}."
    )
    waste_fractions: Variant[YearlyFractions] = Field(
        ...,
        description=(
            "Waste composition per year, {year: [10 fractions]} in the order "
            "food, green, wood, paper_cardboard, textiles, plastic, metal, "
            "glass, rubber, other."
        ),
    )
    landfill_type: Variant[LandfillType] = Field(
        ..., description="Site type: 0 landfill, 1 controlled dump, 2 open dump."
    )
    landfill_open_close: Variant[Tuple[int, int]] = Field(
        ..., description="(open_year, close_year) of the site."
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
    temperature: float = Field(10.0, description="Average annual temperature, deg C.")
    country: Optional[str] = Field(
        None, description="ISO3 country code. Stored for identity; does not affect the math."
    )
    rmi_id: Optional[int] = Field(None, description="Site identifier. Stored for identity.")


# --------------------------------------------------------------------------- #
# Input -> time series helpers
# --------------------------------------------------------------------------- #
def _variant_get(variant: Optional[Variant], label: str):
    """Return ``variant[label]`` with a graceful fallback to baseline."""
    if variant is None:
        return None
    value = variant[label]
    if value is None and label == "scenario":
        return variant["baseline"]
    return value


def _yearly_to_series(
    mapping: Optional[Dict[int, float]],
    years: pd.Index,
    default: Optional[float] = None,
) -> pd.Series:
    """Turn a ``{year: value}`` map into a series aligned to ``years``.

    Missing interior years are forward/back filled. When ``mapping`` is empty a
    ``default`` is required, otherwise we raise rather than silently model zeros.
    """
    if not mapping:
        if default is None:
            raise CustomError("invalid_parameters", "Missing required yearly time series.")
        return pd.Series(float(default), index=years)
    series = pd.Series({int(k): float(v) for k, v in mapping.items()})
    series = series.sort_index().reindex(years).ffill().bfill()
    if default is not None:
        series = series.fillna(float(default))
    return series


def _fractions_to_df(mapping: Optional[Dict[int, List[float]]], years: pd.Index) -> pd.DataFrame:
    """Turn a ``{year: [10 fractions]}`` map into a years x components frame."""
    if not mapping:
        raise CustomError("invalid_parameters", "Missing waste_fractions time series.")
    rows: Dict[int, List[float]] = {}
    for year, vector in mapping.items():
        if len(vector) != len(WASTE_COMPONENTS):
            raise CustomError(
                "invalid_parameters",
                f"waste_fractions for year {year} must have {len(WASTE_COMPONENTS)} values.",
            )
        rows[int(year)] = [float(x) for x in vector]
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=WASTE_COMPONENTS)
    return frame.sort_index().reindex(years).ffill().bfill()


def _variant_series(
    variant: Optional[Variant],
    years: pd.Index,
    implement_year: int,
    default: Optional[float],
) -> Tuple[pd.Series, pd.Series]:
    """Build (baseline_series, scenario_series) for a yearly Variant input.

    The scenario series equals baseline before ``implement_year`` and switches to
    the scenario inputs from ``implement_year`` onward.
    """
    baseline_map = _variant_get(variant, "baseline") if variant is not None else None
    baseline = _yearly_to_series(baseline_map, years, default=default)

    scenario_map = variant["scenario"] if variant is not None else None
    if scenario_map is None:
        return baseline, baseline.copy()

    scenario_input = _yearly_to_series(scenario_map, years, default=default)
    scenario = baseline.copy()
    scenario.loc[implement_year:] = scenario_input.loc[implement_year:]
    return baseline, scenario


def _apply_window(mass_df: pd.DataFrame, open_year: int, close_year: int) -> pd.DataFrame:
    """Zero out waste deposited before the site opens or after it closes."""
    windowed = mass_df.copy()
    windowed.loc[: int(open_year) - 1, :] = 0.0
    windowed.loc[int(close_year):, :] = 0.0
    return windowed


# --------------------------------------------------------------------------- #
# Model parameter construction
# --------------------------------------------------------------------------- #
def _representative_vector(fractions_df: pd.DataFrame, ref_year: int) -> SimpleNamespace:
    """A single composition vector used for the k-value lookup.

    The Wang et al. (2024) k method maps a composition to a single rate via a
    coarse 8x8x8 lookup, and the existing ``advanced_dst`` path takes one vector
    per scenario. We sample the composition at ``ref_year`` (the implement year),
    matching the granularity of the rest of the scenario switch.
    """
    row = fractions_df.loc[ref_year]
    return SimpleNamespace(**{component: float(row[component]) for component in WASTE_COMPONENTS})


def _reindex_decomp(ks: DecompositionRates, years: pd.Index) -> DecompositionRates:
    """compute_singapore_k returns 1990..2050 series; align them to model years."""

    def fix(series: pd.Series) -> pd.Series:
        return series.reindex(years).ffill().bfill()

    return DecompositionRates(
        food=fix(ks.food),
        green=fix(ks.green),
        wood=fix(ks.wood),
        paper_cardboard=fix(ks.paper_cardboard),
        textiles=fix(ks.textiles),
    )


def _decomposition_rates(
    temperature: float,
    precipitation: float,
    implement_year: int,
    years: pd.Index,
    baseline_vector: SimpleNamespace,
    scenario_vector: SimpleNamespace,
) -> Tuple[DecompositionRates, DecompositionRates]:
    """Per-variant decomposition rates (Wang et al. 2024).

    The baseline landfill keeps baseline composition for all years, so its k is
    constant. The scenario landfill uses baseline k before ``implement_year`` and
    scenario k after, which ``compute_singapore_k(advanced_dst=True)`` produces.
    """
    baseline_only = {"baseline": baseline_vector, "scenario": baseline_vector}
    scenario_mix = {"baseline": baseline_vector, "scenario": scenario_vector}

    ks_baseline, _ = compute_singapore_k(
        baseline_only, temperature, precipitation,
        advanced_dst=True, implement_year=implement_year,
    )
    ks_scenario, _ = compute_singapore_k(
        scenario_mix, temperature, precipitation,
        advanced_dst=True, implement_year=implement_year,
    )
    return _reindex_decomp(ks_baseline, years), _reindex_decomp(ks_scenario, years)


def _oxidation_series(
    baseline_type: int,
    scenario_type: int,
    gas_capture: pd.Series,
    biocover: pd.Series,
    implement_year: int,
    years: pd.Index,
) -> pd.Series:
    """Per-year oxidation factor.

    Before ``implement_year`` the site is ``baseline_type``; after, it is
    ``scenario_type``. For each year, gas capture (efficiency > 0) selects the
    capped vs uncapped oxidation factor. A dump upgraded to an engineered
    landfill with capture gets the higher remediated factor. The biocover series
    acts as a floor.
    """
    baseline_name = SITE_TYPE_NAMES[baseline_type]
    scenario_name = SITE_TYPE_NAMES[scenario_type]
    ameliorated = scenario_type < baseline_type

    values = []
    for year in years:
        has_capture = float(gas_capture.loc[year]) > 0
        if year < implement_year:
            value = OX_CAP[baseline_name] if has_capture else OX_NOCAP[baseline_name]
        elif has_capture:
            if ameliorated and scenario_type == LandfillType.landfill.value:
                value = OX_REMEDIATED_TO_LANDFILL
            else:
                value = OX_CAP[scenario_name]
        else:
            value = OX_NOCAP[scenario_name]
        values.append(max(value, float(biocover.loc[year])))
    return pd.Series(values, index=years, dtype=float)


def _city_instance_attrs(city: City, country: Optional[str]) -> Dict:
    """The slice of City config the engine reads (notably ``components``)."""
    return {
        "city_name": city.city_name,
        "country": country,
        "components": city.components,
        "div_components": city.div_components,
        "waste_types": city.waste_types,
        "unprocessable": city.unprocessable,
        "non_compostable_not_targeted": city.non_compostable_not_targeted,
        "combustion_reject_rate": city.combustion_reject_rate,
        "recycling_reject_rates": city.recycling_reject_rates,
    }


def _make_parameters(
    request: AdvancedDSTRequest,
    fractions_df: pd.DataFrame,
    ks: DecompositionRates,
    city_instance_attrs: Dict,
    implement_year: int,
    scenario: int,
) -> CityParameters:
    """A minimal CityParameters carrying just what the engine needs.

    Growth rates and year_of_data_pop are read by the engine but unused for a
    directly-supplied mass series, so they are set to inert values.
    """
    return CityParameters(
        precip=request.precipitation,
        precip_zone=defaults_2019.get_precipitation_zone(request.precipitation),
        temperature=request.temperature,
        growth_rate_historic=1.0,
        growth_rate_future=1.0,
        year_of_data_pop={"baseline": implement_year, "scenario": implement_year},
        mef_compost=0.0,
        scenario=scenario,
        implement_year=implement_year,
        city_instance_attrs=city_instance_attrs,
        ks=ks,
        waste_fractions=fractions_df,
        rmi_id=request.rmi_id,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validate(open_close_pairs: List[Tuple[int, int]], implement_year: int) -> int:
    """Validate site dates and the implement year. Returns the model start year."""
    open_years = []
    for open_year, close_year in open_close_pairs:
        open_year, close_year = int(open_year), int(close_year)
        if not (MODEL_YEAR_MIN <= open_year <= MODEL_YEAR_MAX):
            raise CustomError(
                "invalid_year",
                f"Landfill open year must be between {MODEL_YEAR_MIN} and {MODEL_YEAR_MAX} (got {open_year}).",
            )
        if not (MODEL_YEAR_MIN <= close_year <= MODEL_YEAR_MAX):
            raise CustomError(
                "invalid_year",
                f"Landfill close year must be between {MODEL_YEAR_MIN} and {MODEL_YEAR_MAX} (got {close_year}).",
            )
        if close_year < open_year:
            raise CustomError("invalid_year", "Landfill close year must be on or after open year.")
        open_years.append(open_year)

    model_start = min(open_years)
    if not (model_start <= int(implement_year) <= MODEL_YEAR_MAX):
        raise CustomError(
            "invalid_year",
            f"implement_year must be between {model_start} and {MODEL_YEAR_MAX} (got {implement_year}).",
        )
    return model_start


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_advanced_dst(request: AdvancedDSTRequest) -> Dict[str, pd.DataFrame]:
    """Run the advanced single-site DST.

    Returns ``{"baseline": total_emissions_df, "scenario": total_emissions_df}``
    where each frame is indexed by year with a ``total`` column (tons CO2e).
    """
    implement_year = int(request.implement_year)

    baseline_open, baseline_close = (int(x) for x in request.landfill_open_close["baseline"])
    scenario_dates = request.landfill_open_close["scenario"] or request.landfill_open_close["baseline"]
    scenario_open, scenario_close = (int(x) for x in scenario_dates)

    model_start = _validate(
        [(baseline_open, baseline_close), (scenario_open, scenario_close)],
        implement_year,
    )
    years = pd.Index(range(model_start, MODEL_YEAR_MAX + 1), name="year")

    # --- Waste mass by component (fractions x total), per variant ---
    baseline_fractions = _fractions_to_df(request.waste_fractions["baseline"], years)
    scenario_fractions = _fractions_to_df(
        request.waste_fractions["scenario"] or request.waste_fractions["baseline"], years
    )
    baseline_total, scenario_total = _variant_series(request.waste_mass, years, implement_year, default=None)

    baseline_mass = baseline_fractions.mul(baseline_total, axis=0)
    scenario_mass = scenario_fractions.mul(scenario_total, axis=0)
    # Scenario tracks baseline before changes take effect.
    scenario_mass.loc[: implement_year - 1, :] = baseline_mass.loc[: implement_year - 1, :]
    baseline_mass = _apply_window(baseline_mass, baseline_open, baseline_close)
    scenario_mass = _apply_window(scenario_mass, scenario_open, scenario_close)

    # --- Decomposition rates ---
    ref_year = min(max(implement_year, int(years.min())), int(years.max()))
    ks_baseline, ks_scenario = _decomposition_rates(
        request.temperature,
        request.precipitation,
        implement_year,
        years,
        _representative_vector(baseline_fractions, ref_year),
        _representative_vector(scenario_fractions, ref_year),
    )

    # --- MCF / gas capture / flaring / oxidation series ---
    baseline_type = int(request.landfill_type["baseline"])
    scenario_type = int(request.landfill_type["scenario"]) if request.landfill_type["scenario"] is not None else baseline_type

    mcf_baseline = pd.Series(MCF_BY_TYPE[baseline_type], index=years, dtype=float)
    mcf_scenario = mcf_baseline.copy()
    mcf_scenario.loc[implement_year:] = MCF_BY_TYPE[scenario_type]

    gas_baseline, gas_scenario = _variant_series(request.gas_capture_efficiency, years, implement_year, default=0.0)
    flare_baseline, flare_scenario = _variant_series(request.flaring, years, implement_year, default=DEFAULT_FLARE_EFFICIENCY)
    biocover_baseline, biocover_scenario = _variant_series(request.biocover, years, implement_year, default=0.0)

    ox_baseline = _oxidation_series(baseline_type, baseline_type, gas_baseline, biocover_baseline, implement_year, years)
    ox_scenario = _oxidation_series(baseline_type, scenario_type, gas_scenario, biocover_scenario, implement_year, years)

    # --- Assemble city / landfills and run the engine ---
    city = City("advanced_dst_site")
    city_instance_attrs = _city_instance_attrs(city, request.country)

    baseline_parameters = _make_parameters(request, baseline_fractions, ks_baseline, city_instance_attrs, implement_year, 0)
    scenario_parameters = _make_parameters(request, scenario_fractions, ks_scenario, city_instance_attrs, implement_year, 1)

    baseline_landfill = Landfill(
        open_date=baseline_open,
        close_date=baseline_close,
        site_type=SITE_TYPE_NAMES[baseline_type],
        mcf=mcf_baseline,
        city_params_dict=baseline_parameters.update_cityparams_dict(),
        city_instance_attrs=city_instance_attrs,
        landfill_index=0,
        gas_capture=bool((gas_baseline > 0).any()),
        gas_capture_efficiency=gas_baseline,
        flaring=flare_baseline,
        oxidation_factor=ox_baseline,
        ks=ks_baseline,
        advanced=True,
        implementation_year=implement_year,
        scenario=0,
    )
    scenario_landfill = Landfill(
        open_date=scenario_open,
        close_date=scenario_close,
        site_type=SITE_TYPE_NAMES[scenario_type],
        mcf=mcf_scenario,
        city_params_dict=scenario_parameters.update_cityparams_dict(),
        city_instance_attrs=city_instance_attrs,
        landfill_index=0,
        gas_capture=bool((gas_scenario > 0).any()),
        gas_capture_efficiency=gas_scenario,
        flaring=flare_scenario,
        oxidation_factor=ox_scenario,
        ks=ks_scenario,
        advanced=True,
        implementation_year=implement_year,
        scenario=1,
    )

    baseline_parameters.landfills = [baseline_landfill]
    scenario_parameters.landfills = [scenario_landfill]
    baseline_parameters.repopulate_attr_dicts()
    scenario_parameters.repopulate_attr_dicts()

    baseline_landfill.waste_mass_df = baseline_mass
    baseline_landfill.oxidation_factor = ox_baseline
    scenario_landfill.waste_mass_df = scenario_mass
    scenario_landfill.oxidation_factor = ox_scenario

    baseline_landfill.estimate_emissions(skip_ox=True)
    scenario_landfill.estimate_emissions(skip_ox=True)

    # No diversion in the advanced single-site DST: the supplied mass is the
    # final landfilled mass, so there are no organic (compost/anaerobic) emissions.
    baseline_parameters.organic_emissions = pd.DataFrame(0.0, index=years, columns=DEGRADABLE_COMPONENTS)
    scenario_parameters.organic_emissions = pd.DataFrame(0.0, index=years, columns=DEGRADABLE_COMPONENTS)

    city.baseline_parameters = baseline_parameters
    city.scenario_parameters[0] = scenario_parameters
    city.sum_landfill_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=1)

    return {
        "baseline": baseline_parameters.total_emissions,
        "scenario": scenario_parameters.total_emissions,
    }
