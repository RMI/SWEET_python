"""The city DST: no change is the baseline, and more diversion is less methane.

An end-to-end check of WasteMAP's cities on 2026-11-10 found two bugs.

- `load_csv_new` read incineration's metal, glass and other shares as 0, though
  incineration takes every type and the cities table publishes all three. So every
  scenario with incineration fell short of its share and took the re-split, which
  moved recycling off paper: London's no-change scenario was 19% above its baseline
  in 2040, Bristol's 7%, and 5 more points of incineration raised Seattle's 14%.
- The re-split steered each diversion to the types fewest diversions take, and burned
  a uniform share of the rest. 5 points of recycling raised Navi Mumbai's 2040
  methane 24%: composting took all the food first and paper was landfilled.

The rows are the four cities as the review's cities table published them (WasteMAP
#969 with SWEET #126, built 2026-11-10), cut to the columns `load_csv_new` reads.
"""

import contextlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from SWEET_python import defaults_2019
from SWEET_python.city_params import City, CustomError, DiversionFractions

ROWS = json.loads(
    (Path(__file__).parent / "fixtures" / "city_rows_dst_review_20261110.json").read_text()
)
IMPLEMENT_YEAR = 2026
SLIDERS = ("compost", "anaerobic", "combustion", "recycling")


def _load(name, row=None, no_rejects=False):
    """What WasteMAP's `load_city` does with a cities-table row."""
    city = City(name)
    if no_rejects:
        city.unprocessable = {w: 0.0 for w in city.unprocessable}
        city.non_compostable_not_targeted = {w: 0.0 for w in city.non_compostable_not_targeted}
        city.recycling_reject_rates = {w: 1.0 for w in city.recycling_reject_rates}
    with contextlib.redirect_stdout(io.StringIO()):
        city.load_csv_new(pd.DataFrame([row or ROWS[name]]), dst=True)
        city.baseline_parameters.sites_method = False
        city._calculate_divs()
        for landfill in city.baseline_parameters.landfills:
            landfill.estimate_emissions()
        city.estimate_diversion_emissions(scenario=0)
        city.sum_landfill_emissions(scenario=0)
    return city


def _shares(city):
    return city.baseline_parameters.div_fractions.iloc[0].to_dict()


def _scenario(city, **changes):
    """The scenario's methane by year, as `implement_dst_simple_v1_5` runs it."""
    shares = _shares(city)
    shares.update(changes)
    with contextlib.redirect_stdout(io.StringIO()):
        city.implement_dst_changes_simple_v1_5(
            DiversionFractions(**shares), 0, 0, 0.0, 0.0, IMPLEMENT_YEAR, 1, 0.0
        )
    return city.scenario_parameters[0].total_emissions["total"]


def _baseline(city):
    return city.baseline_parameters.total_emissions["total"]


def test_incineration_reads_its_published_metal_glass_and_other():
    city = _load("London")
    mix = city.baseline_parameters.div_component_fractions.combustion.iloc[0]
    assert mix["metal"] == pytest.approx(0.04)
    assert mix["glass"] == pytest.approx(0.12)
    assert mix["other"] == pytest.approx(0.08)
    assert mix.sum() == pytest.approx(1.0, abs=1e-4)


def test_a_table_from_before_those_columns_burns_none_of_them():
    """Tables built before WasteMAP #716 (August 2026) have no such columns."""
    row = {
        k: v for k, v in ROWS["London"].items()
        if not any(f"Incinerated {label} " in k for label in ("Metal", "Glass", "Other"))
    }
    mix = _load("London", row).baseline_parameters.div_component_fractions.combustion.iloc[0]
    assert (mix[["metal", "glass", "other"]] == 0).all()


@pytest.mark.parametrize("name", ["London", "Bristol", "Seattle", "Navi Mumbai"])
def test_no_change_is_the_baseline(name):
    """London and Bristol incinerate and recycle; before the fix they were 19% and 7%
    above their baselines in 2040."""
    city = _load(name)
    scenario = _scenario(city)
    assert not city.scenario_parameters[0].adjusted_diversion_constituents
    pd.testing.assert_series_equal(scenario, _baseline(city), check_names=False, rtol=1e-12)


@pytest.mark.parametrize("name", ["London", "Bristol", "Seattle", "Navi Mumbai"])
@pytest.mark.parametrize("slider", SLIDERS)
def test_five_more_points_of_any_diversion_lower_methane(name, slider):
    """Every year after implementation, for the four cities the review flagged."""
    city = _load(name)
    shares = _shares(city)
    scenario = _scenario(city, **{slider: shares[slider] + 0.05})
    after = slice(IMPLEMENT_YEAR + 1, None)
    assert (scenario.loc[after] < _baseline(city).loc[after]).all()


def test_the_reviews_cases_now_lower_methane():
    seattle = _load("Seattle")
    up = _scenario(seattle, combustion=0.05)
    assert up.loc[2040] / _baseline(seattle).loc[2040] - 1 < -0.05  # was +14%
    navi = _load("Navi Mumbai")
    up = _scenario(navi, recycling=0.05)
    assert up.loc[2040] / _baseline(navi).loc[2040] - 1 < -0.15  # was +24%


@pytest.mark.parametrize("name", ["London", "Bristol", "Seattle", "Navi Mumbai"])
@pytest.mark.parametrize("slider", SLIDERS)
def test_raising_a_slider_never_landfills_more(name, slider, monkeypatch):
    """One point at a time, from the city's own share until SWEET refuses.

    Rejects and composting's and digestion's own emissions are switched off: they are
    what can make one diversion worse for methane than another, as when composting
    takes food the incinerator would have burned and landfills its rejects. Without
    them, methane is the landfill's, and it must never rise.
    """
    monkeypatch.setattr(defaults_2019, "MEF_ANAEROBIC", 0.0)
    row = dict(ROWS[name], **{"MEF: Compost": 0.0})
    city = _load(name, row, no_rejects=True)
    # Rounding noise, where a late year's methane is old waste decaying to almost none.
    noise = 1e-12 * float(_baseline(city).max())
    start = _shares(city)[slider]
    previous = _scenario(city)
    steps = 0
    for k in range(1, 101):
        try:
            current = _scenario(city, **{slider: round(start + k / 100, 6)})
        except CustomError:
            break
        assert (current <= previous * (1 + 1e-9) + noise).all(), f"{slider} {start + k / 100:.2f}"
        previous = current
        steps += 1
    assert steps >= 5
