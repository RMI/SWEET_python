"""The city DST: no change is the baseline, and more diversion is less methane.

An end-to-end check of WasteMAP's cities on 2026-11-10 found two bugs.

- `load_csv_new` read incineration's metal, glass and other shares as 0, though
  incineration takes every type and the cities table publishes all three. So every
  scenario with incineration fell short of its share and took the solver, which
  moved recycling off paper: London's no-change scenario was 19% above its baseline
  in 2040, Bristol's 7%, and 5 more points of incineration raised Seattle's 14%.
- The solver steered each diversion to the types fewest diversions take. 5 points of
  recycling raised Navi Mumbai's 2040 methane 24%: composting took all the food
  first and paper was landfilled.

Now composting, digestion and recycling keep their own mixes, and incineration burns
what they leave, in the baseline and every scenario alike. The DST works that mix
out itself rather than reading the table's incineration columns.

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
from SWEET_python.city_params import (
    City,
    CustomError,
    DiversionFractions,
    DivComponentFractions,
    WasteFractions,
)

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


def test_incineration_burns_what_the_others_leave():
    """London recycles 30% of its waste and none of it is food, so 34 of the 70 points
    left are food: its incinerator burns 49% food, not the 34% of the whole mix."""
    mix = _load("London").baseline_parameters.div_component_fractions.combustion.iloc[0]
    assert mix.sum() == pytest.approx(1.0)
    assert mix["food"] == pytest.approx(0.34 / 0.70, rel=1e-4)
    assert mix["metal"] == pytest.approx((0.04 - 0.30 * 0.04 / 0.66) / 0.70, rel=1e-3)


@pytest.mark.parametrize(
    "variant",
    ["without metal, glass and other (before WasteMAP #716)", "all food"],
)
def test_the_tables_incineration_mix_is_not_read(variant):
    """The DST works incineration out from the other diversions, so a table's own
    incineration columns, however built, change nothing."""
    columns = [k for k in ROWS["London"] if "Incinerated" in k]
    if variant.startswith("without"):
        row = {k: v for k, v in ROWS["London"].items()
               if not any(f"Incinerated {label} " in k for label in ("Metal", "Glass", "Other"))}
    else:
        row = dict(ROWS["London"], **{k: (100.0 if "Incinerated Food " in k else 0.0) for k in columns})
    city = _load("London", row)
    pd.testing.assert_series_equal(_baseline(city), _baseline(_load("London")), rtol=1e-12)
    pd.testing.assert_series_equal(_scenario(city), _baseline(city), check_names=False, rtol=1e-12)


def test_a_city_that_diverts_all_its_waste_fits():
    """Kitakyushu incinerates 74.95% of its waste and recycles the other 25.05%. What
    recycling leaves can come out a hair under 74.95%; that still fits."""
    city = City("Kitakyushu-like")
    city._divs_from_component_fractions = lambda *a, **k: None  # no masses needed here
    waste = {w: 0.0 for w in city.waste_types}
    waste.update(food=0.4915, paper_cardboard=0.2643, plastic=0.0871, metal=0.008,
                 glass=0.006, other=0.1431)
    recyclable = [w for w in city.waste_types if w in city.div_components["recycling"]]
    pool = sum(waste[w] for w in recyclable)
    mixes = {d: WasteFractions(**{w: 0.0 for w in city.waste_types})
             for d in ("compost", "anaerobic", "combustion")}
    mixes["recycling"] = WasteFractions(**{w: (waste[w] / pool if w in recyclable else 0.0)
                                           for w in city.waste_types})
    adjusted, _, _, fractions = city.mass_checker_math(
        div_fractions=DiversionFractions(compost=0.0, anaerobic=0.0,
                                         combustion=0.7494817885697365,
                                         recycling=0.2505182114302635),
        div_component_fractions=DivComponentFractions(**mixes),
        waste_fractions=WasteFractions(**waste),
        scenario=0,
    )
    assert not adjusted
    assert sum(fractions.combustion.model_dump().values()) == pytest.approx(1.0)


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
