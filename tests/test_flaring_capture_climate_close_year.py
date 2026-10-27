"""Four places where SWEET's paths disagreed, now settled one way (Hugh, 2026-10-23).

1. Flaring: a landfill with no flaring series is flared at 0.98, as on every other
   path. The annual engine used to destroy all of its captured gas, which is every
   landfill on the city paths (city model, city DST, Custom Location).
2. Gas capture left blank takes the default, and a site remediated in a scenario
   takes the remediated value: 30% for a dumpsite made a controlled dumpsite, 50% for
   a controlled dumpsite made a sanitary landfill. The site tool used to raise on a
   blank efficiency.
3. An unknown climate is 15 C and 1,000 mm. The advanced DST defaulted to 10 C, the
   site-tool prefill to 20 C, missing rainfall took the wettest moisture class, and a
   missing temperature gave a NaN decay rate.
4. An unknown close year means the site never closes (2051), so it takes 2050's
   waste. Climate TRACE's site path closed it in 2050, and the site tool refused 2051.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python import gas_capture
from SWEET_python.advanced_dst import AdvancedDSTRequest
from SWEET_python.advanced_dst_city import AdvancedDSTCityRequest
from SWEET_python.city_params import City, CustomError
from SWEET_python.class_defs import SplitFractions, Variant
from SWEET_python.constants import (
    DEFAULT_FLARE_EFFICIENCY,
    MODEL_END_YEAR,
    NEVER_CLOSES,
    UNKNOWN_CLIMATE_PRECIPITATION_MM,
    UNKNOWN_CLIMATE_TEMPERATURE_C,
)

COMPONENTS = ["food", "green", "wood", "paper_cardboard", "textiles",
              "plastic", "metal", "glass", "rubber", "other"]
FLAT = [0.1] * 10


# --------------------------------------------------------------------------- #
# 1. Flaring
# --------------------------------------------------------------------------- #

def _custom_location_all_captured(flaring=None):
    """A Custom Location sending all its disposal to a landfill with capture."""
    city = City("flaring")
    city.dst_baseline_blank("Mexico", 1_000_000, 700.0, 18.0)
    baseline = city.baseline_parameters
    baseline.split_fractions = SplitFractions(
        landfill_w_capture=1.0, landfill_wo_capture=0.0, dumpsite=0.0
    )
    city._calculate_divs()
    baseline.repopulate_attr_dicts()
    landfill = baseline.landfills[0]
    if flaring is not None:
        landfill.flaring = pd.Series(flaring, index=range(1950, MODEL_END_YEAR + 1))
    landfill.estimate_emissions()
    return landfill


def test_a_city_landfill_with_no_flaring_flares_at_the_default():
    unset = _custom_location_all_captured()
    assert unset.flaring is None  # the city paths set none
    stated = _custom_location_all_captured(flaring=DEFAULT_FLARE_EFFICIENCY)
    pd.testing.assert_frame_equal(unset.emissions, stated.emissions)


def test_the_capture_bucket_emits_four_percent_more_than_with_perfect_flaring():
    landfill = _custom_location_all_captured()
    unset = landfill.emissions["total"]
    perfect = _custom_location_all_captured(flaring=1.0).emissions["total"]
    # The 2% of captured gas the flare misses leaves through the flare, not the
    # cover, so it is not oxidised: 1 + 0.6 x 0.02 / (0.4 x (1 - 0.22)) = 1.038,
    # in every year the site captures.
    ox = landfill.oxidation_factor
    expected = 1 + 0.6 * (1 - DEFAULT_FLARE_EFFICIENCY) / ((1 - 0.6) * (1 - ox))
    ratio = unset.loc[2000:] / perfect.loc[2000:]
    assert np.allclose(ratio, expected, rtol=1e-12)


# --------------------------------------------------------------------------- #
# 2. Gas capture: the remediation rule
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "baseline_type, scenario_type, expected",
    [
        (0, 0, 0.6),   # sanitary landfill
        (1, 1, 0.45),  # controlled dumpsite
        (2, 2, 0.0),   # dumpsite
        (2, 1, 0.3),   # dumpsite remediated to a controlled dumpsite
        (1, 0, 0.5),   # controlled dumpsite remediated to a sanitary landfill
        (2, 0, 0.5),   # dumpsite taken straight to a landfill: by the new type
        (0, 2, 0.0),   # a downgrade is not a remediation
    ],
)
def test_default_capture_follows_the_remediation_rule(baseline_type, scenario_type, expected):
    assert gas_capture.default_capture_efficiency(baseline_type, scenario_type) == expected


def _sdst(types, gas, open_close=(1990, 2040), mass=50_000.0):
    """The site tool on a blank site, as WasteMAP's sdst_v1_5 endpoint drives it."""
    years = pd.Index(range(open_close[0], MODEL_END_YEAR + 1))
    fractions = pd.DataFrame(np.tile(FLAT, (len(years), 1)), index=years, columns=COMPONENTS)
    waste_year = Variant[int](baseline=2024, scenario=2024)
    city = City("sdst")
    city.cityparams_obj_for_blank_site(
        country="MEX", population=None, precipitation=800.0, temperature=18.0,
        waste_fractions=Variant(baseline=FLAT, scenario=FLAT),
        waste_mass_year=waste_year, growth_rate_override=0.0,
    )
    city.sdst_v1_5(
        precipitation=800.0,
        new_waste_fractions={"baseline": fractions, "scenario": fractions},
        new_landfill_types=Variant(baseline=[types[0]], scenario=[types[1]]),
        new_gas_efficiency=Variant(baseline=[gas[0]], scenario=[gas[1]]),
        new_landfill_open_close_dates=Variant(baseline=[open_close], scenario=[open_close]),
        scenario=1,
        landfill_split_timeline=Variant(
            baseline={y: [1.0] for y in years}, scenario={y: [1.0] for y in years}
        ),
        waste_burning={"baseline": 0.0, "scenario": 0.0},
        fancy_ox=None,
        new_waste_mass=Variant(baseline=mass, scenario=mass),
        waste_mass_year=waste_year,
        depths=Variant(baseline=[None], scenario=[None]),
        ks_overrides=Variant(baseline=0.05, scenario=0.05),
        implement_year=2026,
        growth_rate_override=0.0,
        country_growth_defaults=[1.0, 1.0],
    )
    return city.baseline_parameters.landfills[0], city.scenario_parameters[0].landfills[0]


@pytest.mark.parametrize(
    "types, baseline_capture, scenario_capture, scenario_ox",
    [
        ((1, 1), 0.45, 0.45, 0.1),   # a controlled dumpsite, unchanged
        ((2, 1), 0.0, 0.3, 0.1),     # a dumpsite remediated to a controlled dumpsite
        ((1, 0), 0.45, 0.5, 0.18),   # a controlled dumpsite remediated to a landfill
        ((2, 2), 0.0, 0.0, 0.0),     # a dumpsite captures nothing, so no capture oxidation
    ],
)
def test_the_site_tool_fills_a_blank_efficiency(types, baseline_capture, scenario_capture, scenario_ox):
    """It raised TypeError on a blank (None > 0.0) before."""
    baseline, scenario = _sdst(types, (None, None))
    assert (baseline.gas_capture_efficiency == baseline_capture).all()
    assert (scenario.gas_capture_efficiency.loc[2026:] == scenario_capture).all()
    assert (scenario.gas_capture_efficiency.loc[:2025] == baseline_capture).all()
    assert (scenario.oxidation_factor.loc[2026:] == scenario_ox).all()
    assert scenario.gas_capture is (scenario_capture != 0)


def test_a_stated_efficiency_still_wins():
    baseline, scenario = _sdst((2, 1), (0.0, 0.7))
    assert (baseline.gas_capture_efficiency == 0.0).all()
    assert (scenario.gas_capture_efficiency.loc[2026:] == 0.7).all()


# --------------------------------------------------------------------------- #
# 3. Unknown climate
# --------------------------------------------------------------------------- #

def test_the_unknown_climate_is_one_pair():
    assert (UNKNOWN_CLIMATE_TEMPERATURE_C, UNKNOWN_CLIMATE_PRECIPITATION_MM) == (15.0, 1000.0)


@pytest.mark.parametrize("request_model", [AdvancedDSTRequest, AdvancedDSTCityRequest])
def test_the_advanced_dst_defaults_to_the_unknown_climate_temperature(request_model):
    assert request_model.model_fields["temperature"].default == UNKNOWN_CLIMATE_TEMPERATURE_C


def _k(temperature, precip):
    from SWEET_python.singapore_k import compute_singapore_k
    city = City("climate")
    city.dst_baseline_blank("Mexico", 1_000_000, 700.0, 18.0)
    ks, _ = compute_singapore_k(city.baseline_parameters.waste_fractions, temperature, precip)
    return ks.food


@pytest.mark.parametrize("temperature", [np.nan, None])
def test_a_missing_temperature_decays_at_15_c_not_nan(temperature):
    # It gave a NaN rate, and so NaN emissions.
    k = _k(temperature, 1000.0)
    assert k.notna().all()
    pd.testing.assert_series_equal(k, _k(UNKNOWN_CLIMATE_TEMPERATURE_C, 1000.0))


@pytest.mark.parametrize("precip", [np.nan, None])
def test_missing_rainfall_decays_at_1000_mm_not_the_wettest_class(precip):
    # It took the wettest moisture class, as if over 2,000 mm.
    pd.testing.assert_series_equal(_k(25.0, precip), _k(25.0, UNKNOWN_CLIMATE_PRECIPITATION_MM))
    assert (_k(25.0, precip) < _k(25.0, 2500.0)).all()


# --------------------------------------------------------------------------- #
# 4. Unknown close year
# --------------------------------------------------------------------------- #

def _trace_site(close_year):
    row = pd.Series({
        "asset_identifier": 999801, "iso3_country": "USA", "type": "Sanitary Landfill",
        "income_id": "HIC", "annual_incoming_waste": 100_000.0, "incoming_waste_year": 2023.0,
        "site_open_year": 1990.0, "site_close_year": close_year, "waste_depth": 10.0,
        "landfill_gas_collection": False, "gas_collection_efficiency": np.nan,
        "latitude": 39.0, "longitude": -95.0, "precipitation": 800.0, "temperature": 15.0,
    })
    pop_data = pd.DataFrame(
        {"growth_rate_historic": [1.02], "growth_rate_future": [1.01]}, index=["USA"]
    )
    city = City("999801")
    city.site_only_estimate_trace(row, pop_data=pop_data)
    return city.baseline_parameters.landfills[0]


def test_a_climate_trace_site_with_no_close_year_takes_2050s_waste():
    landfill = _trace_site(np.nan)
    assert landfill.close_date == NEVER_CLOSES
    assert landfill.fraction_of_waste_vector.at[MODEL_END_YEAR] == 1.0


def test_a_climate_trace_site_takes_waste_in_its_close_year_and_not_after():
    landfill = _trace_site(2040.0)
    assert landfill.fraction_of_waste_vector.at[2040] == 1.0
    assert landfill.fraction_of_waste_vector.at[2041] == 0.0
    assert landfill.fraction_of_waste_vector.at[1990] == 1.0  # its open year


def test_a_climate_trace_site_closing_in_2050_takes_2050s_waste():
    landfill = _trace_site(2050.0)
    assert landfill.fraction_of_waste_vector.at[MODEL_END_YEAR] == 1.0


def test_the_site_tool_accepts_never_closing():
    """It refused 2051 ("Landfill years must be between 1950 and 2050") before."""
    open_site, _ = _sdst((0, 0), (0.0, 0.0), open_close=(1990, NEVER_CLOSES))
    closing, _ = _sdst((0, 0), (0.0, 0.0), open_close=(1990, 2040))
    assert open_site.waste_mass_df.loc[MODEL_END_YEAR].sum() > 0
    assert closing.waste_mass_df.loc[2040].sum() > 0  # its close year takes waste
    assert closing.waste_mass_df.loc[2041].sum() == 0


def test_the_site_tool_still_refuses_a_close_year_past_never_closing():
    with pytest.raises(CustomError):
        _sdst((0, 0), (0.0, 0.0), open_close=(1990, NEVER_CLOSES + 1))
