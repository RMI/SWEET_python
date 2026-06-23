"""
Shared helpers for the advanced DST endpoints (single-site `advanced_dst` and
city-level `advanced_dst_city`).

Everything here is input-shape-agnostic plumbing: turning {year: value} maps into
year-aligned pandas objects, applying the baseline/scenario implement-year splice,
deriving per-year MCF/oxidation, computing decomposition rates, and constructing a
Landfill from already-built parameter series. The endpoint modules own their own
request models and orchestration; this module owns the math primitives they share.
"""

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pandas as pd

import SWEET_python.defaults_2019 as defaults_2019
from SWEET_python.city_params import City, CustomError
from SWEET_python.class_defs import DecompositionRates
from SWEET_python.landfill import Landfill
from SWEET_python.singapore_k import compute_singapore_k

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
# The SWEET engine only generates landfill methane from the biodegradable
# components (these are the ones with a k value and an L_0).
DEGRADABLE_COMPONENTS: List[str] = [
    "food",
    "green",
    "wood",
    "paper_cardboard",
    "textiles",
]

# Methane correction factor by landfill type (index == LandfillType value):
# 0 landfill, 1 controlled dump, 2 open dump.
MCF_BY_TYPE: List[float] = [1.0, 0.6, 0.4]
SITE_TYPE_NAMES: List[str] = ["landfill", "controlled_dumpsite", "dumpsite"]

# Oxidation factor lookup, mirroring City.sdst_v1_5 / Landfill.estimate_emissions.
OX_NOCAP: Dict[str, float] = {"landfill": 0.1, "controlled_dumpsite": 0.05, "dumpsite": 0.0}
OX_CAP: Dict[str, float] = {"landfill": 0.22, "controlled_dumpsite": 0.1, "dumpsite": 0.0}
# A dump upgraded ("ameliorated") to an engineered landfill with gas capture gets
# a higher oxidation factor than a landfill that always existed.
OX_REMEDIATED_TO_LANDFILL = 0.18

DEFAULT_FLARE_EFFICIENCY = 0.98
LANDFILL_LANDFILL_TYPE = 0  # LandfillType.landfill

# A yearly time series arrives over the wire as {year: value}. Pydantic coerces
# JSON string keys ("2025") to ints.
YearlyFloat = Dict[int, float]
YearlyFractions = Dict[int, List[float]]


# --------------------------------------------------------------------------- #
# Input -> time series
# --------------------------------------------------------------------------- #
def variant_get(variant, label: str):
    """Return ``variant[label]`` with a graceful fallback to baseline."""
    if variant is None:
        return None
    value = variant[label]
    if value is None and label == "scenario":
        return variant["baseline"]
    return value


def yearly_to_series(
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


def fractions_to_df(mapping: Optional[Dict[int, List[float]]], years: pd.Index) -> pd.DataFrame:
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


def variant_series(
    variant,
    years: pd.Index,
    implement_year: int,
    default: Optional[float],
) -> Tuple[pd.Series, pd.Series]:
    """Build (baseline_series, scenario_series) for a yearly Variant input.

    The scenario series equals baseline before ``implement_year`` and switches to
    the scenario inputs from ``implement_year`` onward.
    """
    baseline_map = variant_get(variant, "baseline") if variant is not None else None
    baseline = yearly_to_series(baseline_map, years, default=default)

    scenario_map = variant["scenario"] if variant is not None else None
    if scenario_map is None:
        return baseline, baseline.copy()

    scenario_input = yearly_to_series(scenario_map, years, default=default)
    scenario = baseline.copy()
    scenario.loc[implement_year:] = scenario_input.loc[implement_year:]
    return baseline, scenario


def apply_window(mass_df: pd.DataFrame, open_year: int, close_year: int) -> pd.DataFrame:
    """Zero out waste deposited before the site opens or after it closes."""
    windowed = mass_df.copy()
    windowed.loc[: int(open_year) - 1, :] = 0.0
    windowed.loc[int(close_year):, :] = 0.0
    return windowed


# --------------------------------------------------------------------------- #
# Decomposition rates (Wang et al. 2024)
# --------------------------------------------------------------------------- #
def representative_vector(fractions_df: pd.DataFrame, ref_year: int) -> SimpleNamespace:
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


def decomposition_rates(
    temperature: float,
    precipitation: float,
    implement_year: int,
    years: pd.Index,
    baseline_vector: SimpleNamespace,
    scenario_vector: SimpleNamespace,
) -> Tuple[DecompositionRates, DecompositionRates]:
    """Per-variant decomposition rates.

    The baseline keeps baseline composition for all years, so its k is constant.
    The scenario uses baseline k before ``implement_year`` and scenario k after,
    which ``compute_singapore_k(advanced_dst=True)`` produces.
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


# --------------------------------------------------------------------------- #
# Per-landfill parameter series and construction
# --------------------------------------------------------------------------- #
def oxidation_series(
    baseline_type: int,
    scenario_type: int,
    gas_capture: pd.Series,
    biocover: pd.Series,
    implement_year: int,
    years: pd.Index,
) -> pd.Series:
    """Per-year oxidation factor for one landfill.

    Before ``implement_year`` the site is ``baseline_type``; after, it is
    ``scenario_type``. For each year, gas capture (efficiency > 0) selects the
    capped vs uncapped oxidation factor. A dump upgraded to an engineered landfill
    with capture gets the higher remediated factor. ``biocover`` acts as a floor.
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
            if ameliorated and scenario_type == LANDFILL_LANDFILL_TYPE:
                value = OX_REMEDIATED_TO_LANDFILL
            else:
                value = OX_CAP[scenario_name]
        else:
            value = OX_NOCAP[scenario_name]
        values.append(max(value, float(biocover.loc[year])))
    return pd.Series(values, index=years, dtype=float)


def mcf_series(baseline_type: int, scenario_type: int, implement_year: int, years: pd.Index) -> pd.Series:
    """MCF series for one landfill: baseline type before implement_year, scenario after."""
    series = pd.Series(MCF_BY_TYPE[baseline_type], index=years, dtype=float)
    series.loc[implement_year:] = MCF_BY_TYPE[scenario_type]
    return series


def build_landfill(
    *,
    open_year: int,
    close_year: int,
    site_type_idx: int,
    mcf: pd.Series,
    gas_capture_efficiency: pd.Series,
    flaring: pd.Series,
    oxidation_factor: pd.Series,
    ks: DecompositionRates,
    city_params_dict: dict,
    city_instance_attrs: dict,
    implement_year: int,
    scenario: int,
    landfill_index: int = 0,
) -> Landfill:
    """Construct one advanced Landfill from already-built per-year series.

    The caller assigns ``waste_mass_df`` afterward (advanced landfills don't derive
    it in __init__) and then calls ``estimate_emissions(skip_ox=True)``.
    """
    return Landfill(
        open_date=open_year,
        close_date=close_year,
        site_type=SITE_TYPE_NAMES[site_type_idx],
        mcf=mcf,
        city_params_dict=city_params_dict,
        city_instance_attrs=city_instance_attrs,
        landfill_index=landfill_index,
        gas_capture=bool((gas_capture_efficiency > 0).any()),
        gas_capture_efficiency=gas_capture_efficiency,
        flaring=flaring,
        oxidation_factor=oxidation_factor,
        ks=ks,
        advanced=True,
        implementation_year=implement_year,
        scenario=scenario,
    )


def city_instance_attrs(city: City, country: Optional[str]) -> Dict:
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


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_years(open_close_pairs: List[Tuple[int, int]], implement_year: int) -> int:
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
