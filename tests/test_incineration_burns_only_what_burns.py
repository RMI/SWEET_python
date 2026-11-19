"""An incineration share counts only waste that burns.

Metal and glass don't burn. Recycling takes its share of them first (and is read as
counting any metal recovered from incinerator ash); what it leaves is landfilled,
whether it was sorted out before the furnace or pulled from the ash. So incineration
burns the same share of every *burnable* type composting, digestion and recycling
leave, and can take at most all of it:

- a baseline whose data asks for more burns all the burnable waste left, and that is
  recorded as the city's incineration;
- a city DST scenario that asks for more is refused, naming the most it can take;
- an advanced DST site that burns its intake keeps the metal and glass it receives.

Osaka and Yokohama are the cities table's rows (dev database, 2026-11-10).
"""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SWEET_python import dst_allocation
from SWEET_python.advanced_dst_city import AdvancedDSTCityRequest, run_advanced_dst_city
from SWEET_python.city_params import (
    City,
    CustomError,
    DiversionFractions,
    DivComponentFractions,
    WasteFractions,
)
from SWEET_python.constants import DIVERSION_ELIGIBILITY, UNBURNABLE_TYPES, WASTE_TYPES

FIXTURES = Path(__file__).parent / "fixtures"
OSAKA_YOKOHAMA = json.loads((FIXTURES / "city_rows_osaka_yokohama.json").read_text())
DST_ROWS = json.loads((FIXTURES / "city_rows_dst_review_20261110.json").read_text())

# London in the cities table: incineration 47% and recycling 30%.
LONDON = {w: 0.0 for w in WASTE_TYPES}
LONDON.update(food=0.34, paper_cardboard=0.36, plastic=0.06, metal=0.04, glass=0.12, other=0.08)
#: What London has left to burn once recycling takes 30 of the 66 recyclable points.
LONDON_BURNABLE_LEFT = 0.70 - (0.04 + 0.12) * (1 - 0.30 / 0.66)


def test_incineration_takes_every_type_but_metal_and_glass():
    assert set(UNBURNABLE_TYPES) == {"metal", "glass"}
    assert set(DIVERSION_ELIGIBILITY["combustion"]) == set(WASTE_TYPES) - {"metal", "glass"}
    assert "other" in DIVERSION_ELIGIBILITY["combustion"]


def _check(city, combustion, scenario):
    """`mass_checker_math` on London, recycling 30% in proportion to its recyclables."""
    recyclable = [w for w in WASTE_TYPES if w in city.div_components["recycling"]]
    pool = sum(LONDON[w] for w in recyclable)
    mixes = {d: WasteFractions() for d in ("compost", "anaerobic", "combustion")}
    mixes["recycling"] = WasteFractions(**{w: LONDON[w] / pool for w in recyclable})
    shares = DiversionFractions(compost=0.0, anaerobic=0.0, combustion=combustion, recycling=0.30)
    _, _, _, fractions = city.mass_checker_math(
        div_fractions=shares,
        div_component_fractions=DivComponentFractions(**mixes),
        waste_fractions=WasteFractions(**LONDON),
        scenario=scenario,
    )
    return shares, fractions


@pytest.fixture
def city():
    city = City("London-like")
    city._divs_from_component_fractions = lambda *a, **k: None  # no masses needed here
    return city


def test_a_baseline_asking_for_more_burns_all_the_burnable_waste_left(city):
    shares, fractions = _check(city, combustion=0.70, scenario=0)
    assert shares.combustion == pytest.approx(LONDON_BURNABLE_LEFT)
    burned = fractions.combustion.model_dump()
    assert burned["metal"] == burned["glass"] == 0.0
    assert sum(burned.values()) == pytest.approx(1.0)


def test_a_baseline_that_fits_is_left_alone(city):
    shares, _ = _check(city, combustion=0.47, scenario=0)
    assert shares.combustion == 0.47


def test_a_scenario_asking_for_more_is_refused(city):
    with pytest.raises(CustomError, match=r"at most 61\.3%.*metal and glass do not burn"):
        _check(city, combustion=0.62, scenario=1)
    _check(city, combustion=0.61, scenario=1)


def _load_andre(name):
    saved = OSAKA_YOKOHAMA[name]
    data = [np.nan if v is None else v for v in saved["row"]["data"]]
    row = pd.Series(data, index=saved["row"]["index"], name=saved["row"]["name"])
    city = City(name)
    with contextlib.redirect_stdout(io.StringIO()):
        city.load_andre_params(row, income_id=saved["income_id"])
    return city, row


@pytest.mark.parametrize("name", sorted(OSAKA_YOKOHAMA))
def test_the_cities_table_records_what_was_burned(name):
    """Osaka reports 90.3% incinerated and 9.7% recycled; 5% of its waste is metal and
    glass, and recycling takes only some of it. The rest can't burn, so Osaka burns
    85.9%, and that is the share the table publishes."""
    city, row = _load_andre(name)
    baseline = city.baseline_parameters
    reported = float(row["waste_treatment_incineration_percent"]) / 100
    share = float(baseline.div_fractions["combustion"].iloc[0])
    assert share < reported
    # Every year carries it, and the tonnes burned are that share of the waste.
    assert (baseline.div_fractions["combustion"] == share).all()
    burned = pd.DataFrame(baseline.divs.combustion)
    generated = baseline.waste_generated_df
    for year in (2000, 2020, 2040):
        assert burned.loc[year].sum() / generated.loc[year].sum() == pytest.approx(share, rel=1e-9)
    assert "metal" not in burned.columns and "glass" not in burned.columns
    # Metal and glass recycling leaves reach the landfill.
    net = pd.DataFrame(baseline.net_masses).loc[2020]
    assert net["metal"] + net["glass"] > 0


def _load_published(row):
    city = City("London")
    with contextlib.redirect_stdout(io.StringIO()):
        city.load_csv_new(pd.DataFrame([row]), dst=True)
    return city


def test_a_published_share_above_the_burnable_waste_left_is_cut_to_it():
    """A table built before metal and glass stopped burning can publish 70% for London;
    the DST burns the 61.3% left to burn, as the cities build now would."""
    row = dict(DST_ROWS["London"], **{"Diversions: Incineration (%)": 70.0})
    city = _load_published(row)
    share = city.baseline_parameters.div_fractions["combustion"].iloc[0]
    assert share == pytest.approx(LONDON_BURNABLE_LEFT, rel=1e-3)
    assert share <= dst_allocation.max_incineration(
        LONDON,
        {"recycling": {w: 0.30 * LONDON[w] / 0.66 for w in ("paper_cardboard", "plastic", "metal", "glass", "other")}},
    ) + 1e-9


def test_a_published_share_that_fits_is_read_as_it_is():
    city = _load_published(DST_ROWS["London"])
    assert city.baseline_parameters.div_fractions["combustion"].iloc[0] == pytest.approx(0.47)


# --------------------------------------------------------------------------- #
# The advanced DST
# --------------------------------------------------------------------------- #
YEARS = range(2000, 2051)
FRACTIONS = [0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.02, 0.03, 0.0, 0.05]  # metal 2%, glass 3%
TOTAL_WASTE = 20_000.0


def _site(combusts):
    return dict(
        landfill_type={"baseline": 0, "scenario": 0},
        landfill_open_close={"baseline": [2000, 2050], "scenario": [2000, 2050]},
        gas_capture_efficiency={"baseline": {y: 0.0 for y in YEARS}, "scenario": {y: 0.0 for y in YEARS}},
        combusts={"baseline": combusts, "scenario": combusts},
    )


def _adst(diversion=None):
    request = AdvancedDSTCityRequest(
        city_name="Burnville",
        precipitation=1200.0,
        temperature=20.0,
        implement_year=2025,
        waste_mass={"baseline": {y: TOTAL_WASTE for y in YEARS}, "scenario": {y: TOTAL_WASTE for y in YEARS}},
        waste_fractions={"baseline": {y: FRACTIONS for y in YEARS}, "scenario": {y: FRACTIONS for y in YEARS}},
        landfills=[_site(True), _site(False)],
        landfill_split_timeline={"baseline": {y: [0.5, 0.5] for y in YEARS}, "scenario": {y: [0.5, 0.5] for y in YEARS}},
        diversion_fractions=diversion,
        country="USA",
    )
    return run_advanced_dst_city(request, with_mass_flow=True)


def test_a_burning_site_keeps_the_metal_and_glass_it_receives():
    """Half the city's waste goes to the incinerator: it keeps the 5% that is metal and
    glass, 500 t a year, and burns the rest."""
    flow = _adst()["mass_flow"]["baseline"]
    incinerator, landfill = flow["sites"][0], flow["sites"][1]
    for year in (2000, 2030, 2050):
        assert incinerator[year] == pytest.approx(0.5 * TOTAL_WASTE * 0.05, rel=1e-9)
        assert landfill[year] == pytest.approx(0.5 * TOTAL_WASTE, rel=1e-9)


def test_the_metal_and_glass_it_keeps_make_no_methane():
    result = _adst()
    only_landfill = AdvancedDSTCityRequest(
        city_name="Burnville",
        precipitation=1200.0,
        temperature=20.0,
        implement_year=2025,
        waste_mass={"baseline": {y: TOTAL_WASTE for y in YEARS}, "scenario": {y: TOTAL_WASTE for y in YEARS}},
        waste_fractions={"baseline": {y: FRACTIONS for y in YEARS}, "scenario": {y: FRACTIONS for y in YEARS}},
        landfills=[_site(False)],
        landfill_split_timeline={"baseline": {y: [1.0] for y in YEARS}, "scenario": {y: [1.0] for y in YEARS}},
        country="USA",
    )
    alone = run_advanced_dst_city(only_landfill)["baseline"]["total"]
    for year in (2010, 2030, 2050):
        assert result["baseline"]["total"][year] == pytest.approx(0.5 * alone[year], rel=1e-9)


def test_the_combustion_diversion_burns_no_metal_or_glass():
    combustion = {"combustion": {y: 0.4 for y in YEARS}}
    flow = _adst({"baseline": combustion, "scenario": combustion})["mass_flow"]["baseline"]
    burnt = flow["diverted"]["combustion"]
    assert (burnt["metal"] == 0).all() and (burnt["glass"] == 0).all()
    assert burnt.sum(axis=1)[2030] == pytest.approx(0.4 * TOTAL_WASTE, rel=1e-12)
