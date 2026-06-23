"""
Advanced DST (adst) — single-site methane modeling.

This is the modeling backend for the ``/v1/site_emissions/adst`` endpoint. It is
the "advanced" successor to ``City.sdst_v1_5``, but deliberately much simpler:
the caller supplies waste mass and composition directly as time series, so this
module does *no* waste generation, population growth, diversion accounting, or
TRACE reconciliation. It just turns the supplied numbers into a baseline and a
scenario emissions time series for a single landfill.

For the multi-landfill, diversion-aware city version, see ``advanced_dst_city``.
Shared plumbing lives in ``dst_common``.

Design notes
------------
* Inputs arrive as :class:`SWEET_python.class_defs.Variant` objects carrying a
  ``baseline`` and an optional ``scenario`` value (which falls back to baseline).
* The scenario variant equals the baseline variant for every year before
  ``implement_year`` and switches to the scenario inputs from ``implement_year``
  onward — the same convention the rest of the SWEET model uses.
* ``run_advanced_dst`` is a pure function: it returns the result and never
  mutates shared state. It builds a throwaway :class:`City`/:class:`Landfill`
  internally purely to reuse the tested emission engine.
* Here ``waste_mass`` is the final landfilled mass (no diversion). The city-level
  endpoint instead treats ``waste_mass`` as total generated waste and subtracts
  diversions.
"""

from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

import SWEET_python.defaults_2019 as defaults_2019
from SWEET_python.city_params import City, CityParameters
from SWEET_python.class_defs import LandfillType, Variant
from SWEET_python import dst_common as common
from SWEET_python.dst_common import YearlyFloat, YearlyFractions

__all__ = ["AdvancedDSTRequest", "run_advanced_dst"]


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
    landfill_open_close: Variant[tuple[int, int]] = Field(
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


def _make_parameters(
    request: AdvancedDSTRequest,
    fractions_df: pd.DataFrame,
    ks,
    city_instance_attrs: dict,
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


def run_advanced_dst(request: AdvancedDSTRequest) -> dict[str, pd.DataFrame]:
    """Run the advanced single-site DST.

    Returns ``{"baseline": total_emissions_df, "scenario": total_emissions_df}``
    where each frame is indexed by year with a ``total`` column.
    """
    implement_year = int(request.implement_year)

    baseline_open, baseline_close = (int(x) for x in request.landfill_open_close["baseline"])
    scenario_dates = request.landfill_open_close["scenario"] or request.landfill_open_close["baseline"]
    scenario_open, scenario_close = (int(x) for x in scenario_dates)

    model_start = common.validate_years(
        [(baseline_open, baseline_close), (scenario_open, scenario_close)],
        implement_year,
    )
    years = pd.Index(range(model_start, common.MODEL_YEAR_MAX + 1), name="year")

    # --- Waste mass by component (fractions x total), per variant ---
    baseline_fractions = common.fractions_to_df(request.waste_fractions["baseline"], years)
    scenario_fractions = common.fractions_to_df(
        request.waste_fractions["scenario"] or request.waste_fractions["baseline"], years
    )
    baseline_total, scenario_total = common.variant_series(request.waste_mass, years, implement_year, default=None)

    baseline_mass = baseline_fractions.mul(baseline_total, axis=0)
    scenario_mass = scenario_fractions.mul(scenario_total, axis=0)
    scenario_mass.loc[: implement_year - 1, :] = baseline_mass.loc[: implement_year - 1, :]
    baseline_mass = common.apply_window(baseline_mass, baseline_open, baseline_close)
    scenario_mass = common.apply_window(scenario_mass, scenario_open, scenario_close)

    # --- Decomposition rates ---
    ref_year = min(max(implement_year, int(years.min())), int(years.max()))
    ks_baseline, ks_scenario = common.decomposition_rates(
        request.temperature,
        request.precipitation,
        implement_year,
        years,
        common.representative_vector(baseline_fractions, ref_year),
        common.representative_vector(scenario_fractions, ref_year),
    )

    # --- MCF / gas capture / flaring / oxidation series ---
    baseline_type = int(request.landfill_type["baseline"])
    scenario_type = int(request.landfill_type["scenario"]) if request.landfill_type["scenario"] is not None else baseline_type

    mcf_baseline = common.mcf_series(baseline_type, baseline_type, implement_year, years)
    mcf_scenario = common.mcf_series(baseline_type, scenario_type, implement_year, years)

    gas_baseline, gas_scenario = common.variant_series(request.gas_capture_efficiency, years, implement_year, default=0.0)
    flare_baseline, flare_scenario = common.variant_series(request.flaring, years, implement_year, default=common.DEFAULT_FLARE_EFFICIENCY)
    biocover_baseline, biocover_scenario = common.variant_series(request.biocover, years, implement_year, default=0.0)

    ox_baseline = common.oxidation_series(baseline_type, baseline_type, gas_baseline, biocover_baseline, implement_year, years)
    ox_scenario = common.oxidation_series(baseline_type, scenario_type, gas_scenario, biocover_scenario, implement_year, years)

    # --- Assemble city / landfills and run the engine ---
    city = City("advanced_dst_site")
    city_instance_attrs = common.city_instance_attrs(city, request.country)

    baseline_parameters = _make_parameters(request, baseline_fractions, ks_baseline, city_instance_attrs, implement_year, 0)
    scenario_parameters = _make_parameters(request, scenario_fractions, ks_scenario, city_instance_attrs, implement_year, 1)

    baseline_landfill = common.build_landfill(
        open_year=baseline_open, close_year=baseline_close, site_type_idx=baseline_type,
        mcf=mcf_baseline, gas_capture_efficiency=gas_baseline, flaring=flare_baseline,
        oxidation_factor=ox_baseline, ks=ks_baseline,
        city_params_dict=baseline_parameters.update_cityparams_dict(),
        city_instance_attrs=city_instance_attrs, implement_year=implement_year, scenario=0,
    )
    scenario_landfill = common.build_landfill(
        open_year=scenario_open, close_year=scenario_close, site_type_idx=scenario_type,
        mcf=mcf_scenario, gas_capture_efficiency=gas_scenario, flaring=flare_scenario,
        oxidation_factor=ox_scenario, ks=ks_scenario,
        city_params_dict=scenario_parameters.update_cityparams_dict(),
        city_instance_attrs=city_instance_attrs, implement_year=implement_year, scenario=1,
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

    # No diversion in the single-site DST: the supplied mass is the final
    # landfilled mass, so there are no organic (compost/anaerobic) emissions.
    baseline_parameters.organic_emissions = pd.DataFrame(0.0, index=years, columns=common.DEGRADABLE_COMPONENTS)
    scenario_parameters.organic_emissions = pd.DataFrame(0.0, index=years, columns=common.DEGRADABLE_COMPONENTS)

    city.baseline_parameters = baseline_parameters
    city.scenario_parameters[0] = scenario_parameters
    city.sum_landfill_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=1)

    return {
        "baseline": baseline_parameters.total_emissions,
        "scenario": scenario_parameters.total_emissions,
    }
