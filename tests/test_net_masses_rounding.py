"""A city that incinerates all of a waste type keeps its landfills.

Subtracting the diverted tonnes from the generated tonnes leaves a few nanotonnes below
zero when all of a type is diverted: Osaka's paper at -6e-11 t, Yokohama's food at
-9e-5 t. `mass_checker_math` accepts that (it allows MASS_CHECK_TOLERANCE of the city's
waste), but `_calculate_net_masses` kept the negative, so `load_andre_params` printed
"Invalid new value" and returned with no landfills. WasteMAP's cities build then
stopped on Osaka (build 20261110T181252Z).

The fixture is the two cities' rows from WasteMAP's cities table (dev database,
2026-11-10), as `load_andre_params` receives them.
"""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SWEET_python.city_params import MASS_CHECK_TOLERANCE, City, _zero_accepted_deficits

FIXTURE = Path(__file__).parent / "fixtures" / "city_rows_osaka_yokohama.json"
CITIES = json.loads(FIXTURE.read_text())


def _load(name):
    saved = CITIES[name]
    # JSON has no NaN: a missing value comes back as None, which SWEET reads as NaN.
    data = [np.nan if v is None else v for v in saved["row"]["data"]]
    row = pd.Series(data, index=saved["row"]["index"], name=saved["row"]["name"])
    city = City(name)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        city.load_andre_params(row, income_id=saved["income_id"])
    return city, printed.getvalue()


@pytest.mark.parametrize("name", sorted(CITIES))
def test_a_city_that_incinerates_a_whole_waste_type_keeps_its_landfills(name):
    city, printed = _load(name)
    baseline = city.baseline_parameters
    assert "Invalid new value" not in printed
    assert not baseline.input_problems
    assert baseline.landfills, "load_andre_params returned before building the landfills"
    assert (pd.DataFrame(baseline.net_masses) >= 0).all().all()


@pytest.mark.parametrize("name", sorted(CITIES))
def test_the_city_then_models_its_emissions(name):
    """What WasteMAP's cities build does next with each city."""
    city, _ = _load(name)
    for landfill in city.baseline_parameters.landfills:
        landfill.estimate_emissions()
    city.estimate_diversion_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=0, simple=True)


def test_rounding_below_zero_becomes_zero():
    net = pd.Series({"food": 400.0, "paper_cardboard": -6e-11, "metal": -3.6e-12})
    out = _zero_accepted_deficits(net, total=1_000_000.0)
    assert out.tolist() == [400.0, 0.0, 0.0]


def test_a_real_deficit_stays_negative_so_it_is_still_refused():
    net = pd.Series({"food": -50.0, "paper_cardboard": 10.0})
    total = 1_000_000.0
    assert 50.0 > MASS_CHECK_TOLERANCE * total
    out = _zero_accepted_deficits(net, total)
    assert out["food"] == -50.0


def test_each_year_is_judged_against_that_years_waste():
    net = pd.DataFrame({"food": [-5.0, -5.0], "wood": [1.0, -1e-9]}, index=[2020, 2021])
    total = pd.Series([1_000_000.0, 100_000.0], index=[2020, 2021])
    out = _zero_accepted_deficits(net, total)
    # 5 t is within 1e-5 of 1 Mt (10 t) but not of 100 kt (1 t).
    assert out.loc[2020, "food"] == 0.0
    assert out.loc[2021, "food"] == -5.0
    assert out.loc[2021, "wood"] == 0.0
