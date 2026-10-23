"""A city that reports no diversion takes its country's default diversion shares.

`City.load_andre_params` has two branches that fill a city's composting and
incineration shares from `defaults_2019` (the country's row, else its region's): one
for a city that reports no treatment data at all, and one for a city that reports no
diversion and landfill shares adding up to less than 100%. Both test whether every
diversion share is blank, but the compost share was turned from blank into 0 before
that test, so it never passed: no city ever took the defaults. In the June 2026 cities
table no city was flagged "Defaults Used: Diversion", and 99 cities with no treatment
data diverted nothing, among them Vienna, Eindhoven and Turku.

A blank share now stays blank until the default checks have seen it.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python import defaults_2019
from SWEET_python.city_params import City

TREATMENTS = [
    "waste_treatment_compost_percent",
    "waste_treatment_anaerobic_digestion_percent",
    "waste_treatment_incineration_percent",
    "waste_treatment_advanced_thermal_treatment_percent",
    "waste_treatment_recycling_percent",
    "waste_treatment_sanitary_landfill_landfill_gas_system_percent",
    "waste_treatment_controlled_landfill_percent",
    "waste_treatment_landfill_unspecified_percent",
    "waste_treatment_open_dump_percent",
]
COMPOSITION = [
    "composition_food_organic_waste_percent",
    "composition_yard_garden_green_waste_percent",
    "composition_wood_percent",
    "composition_paper_cardboard_percent",
    "composition_textiles_percent",
    "composition_plastic_percent",
    "composition_metal_percent",
    "composition_glass_percent",
    "composition_rubber_leather_percent",
    "composition_other_percent",
]


def _vienna(**reported):
    """A cities-query row for Vienna with no treatment data, plus ``reported``."""
    row = pd.Series(
        {
            "population_data_source": "World Bank",
            "country": "Austria",
            "iso": "AUT",
            "population_year": 2020.0,
            "population_count": 1_900_000.0,
            "historic_growth_rate": 1.005,
            "future_growth_rate": 1.004,
            "latitude": 48.2,
            "longitude": 16.4,
            "mean_yearly_precip_2000_2021": 650.0,
            "mean_yearly_temp_2000_2021": 11.0,
            "msw_generated_metric_tons_per_year": 1_000_000.0,
            "msw_generated_year": 2020.0,
            "msw_collected_metric_tons_per_year": np.nan,
            "msw_collected_year": np.nan,
            **{column: np.nan for column in TREATMENTS + COMPOSITION},
            **reported,
        }
    )
    return pd.concat([row, pd.Series([2020.0, 2020.0], index=["data_collection_year"] * 2)])


def _loaded(row):
    city = City("Vienna")
    city.load_andre_params(row)
    return city


def _shares(city):
    divs = city.baseline_parameters.div_fractions
    return divs.compost.iloc[0], divs.combustion.iloc[0]


def _emissions(city):
    for landfill in city.baseline_parameters.landfills:
        landfill.estimate_emissions()
    city.estimate_diversion_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=0, simple=True)
    return city.baseline_parameters.total_emissions["total"]


AUSTRIA = (
    defaults_2019.fraction_composted_country["AUT"],
    defaults_2019.fraction_incinerated_country["AUT"],
)


def test_a_city_with_no_treatment_data_takes_its_countrys_shares():
    city = _loaded(_vienna())

    assert city.diversion_defaults is True
    assert city.landfill_split_defaults is True
    assert _shares(city) == pytest.approx(AUSTRIA, rel=1e-12)
    assert AUSTRIA[0] > 0 and AUSTRIA[1] > 0  # the test means something only if so


def test_a_city_with_partial_landfill_shares_and_no_diversion_takes_them_too():
    city = _loaded(_vienna(waste_treatment_open_dump_percent=40.0))

    assert city.diversion_defaults is True
    assert _shares(city) == pytest.approx(AUSTRIA, rel=1e-12)


def test_landfill_shares_adding_up_to_everything_mean_no_diversion():
    city = _loaded(_vienna(waste_treatment_landfill_unspecified_percent=100.0))

    assert city.diversion_defaults is False
    assert _shares(city) == (0.0, 0.0)


def test_a_reported_share_is_used_and_the_others_are_zero():
    city = _loaded(_vienna(waste_treatment_compost_percent=10.0))

    assert city.diversion_defaults is False
    assert _shares(city) == pytest.approx((0.10, 0.0), rel=1e-12)


def test_reported_zeros_are_not_blanks():
    """A city that reports 0% for each diversion diverts nothing; a blank one takes defaults."""
    zeros = {
        column: 0.0
        for column in TREATMENTS
        if column.endswith(("compost_percent", "digestion_percent", "incineration_percent",
                            "thermal_treatment_percent", "recycling_percent"))
    }
    reported_zero = _loaded(_vienna(**zeros))
    blank = _loaded(_vienna())

    assert reported_zero.diversion_defaults is False
    assert _shares(reported_zero) == (0.0, 0.0)
    # Diverting Austria's default shares lowers 2025's methane.
    assert _emissions(blank).loc[2025] < _emissions(reported_zero).loc[2025]
