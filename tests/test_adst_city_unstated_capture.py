"""A site that collects gas at an efficiency nobody stated captures at its type's default.

In ``CityLandfillSpec.gas_capture_efficiency`` a year may be null: the site has
a collection system and the caller does not know how well it works. That year
captures at the type's default -- 60% for a landfill, 45% for a controlled
dump, 0 for an open dump, ``city_params.GAS_EFF_OPTIONS`` -- and
``run_advanced_dst_city_defaults`` reports the figure, so a caller can show it
greyed in the box a user types into.

Before, a caller with no efficiency had to send 0, which modelled a site that
collects gas and captures none of it. The advanced DST sent 0 for every such
site, including the Mexico City sample's Rincon Verde.
"""

import pandas as pd
import pytest

from SWEET_python.advanced_dst_city import (
    AdvancedDSTCityRequest,
    _gas_capture,
    run_advanced_dst_city,
    run_advanced_dst_city_defaults,
)
from SWEET_python.city_params import GAS_EFF_OPTIONS
from SWEET_python.dst_common import GAS_CAPTURE_DEFAULT, OX_CAP, OX_NOCAP, variant_series

YEARS = range(2000, 2051)
FRACTIONS = [0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.02, 0.03, 0.0, 0.05]  # sums to 1.0
IMPLEMENT_YEAR = 2025
TOTAL_WASTE = 20_000.0

LANDFILL, CONTROLLED_DUMP, OPEN_DUMP = 0, 1, 2
TYPE_DEFAULTS = {LANDFILL: 0.6, CONTROLLED_DUMP: 0.45, OPEN_DUMP: 0.0}


def _constant(value):
    return {y: value for y in YEARS}


def _landfill(*, site_type=LANDFILL, scenario_type=None, gas=None, **fields):
    """A site that collects gas at an unstated efficiency unless `gas` says otherwise."""
    gas = _constant(None) if gas is None else gas
    spec = dict(
        landfill_type={
            "baseline": site_type,
            "scenario": site_type if scenario_type is None else scenario_type,
        },
        landfill_open_close={"baseline": [2000, 2050], "scenario": [2000, 2050]},
        gas_capture_efficiency={"baseline": gas, "scenario": gas},
    )
    spec.update(fields)
    return spec


def _request(landfills) -> AdvancedDSTCityRequest:
    """A city with no diversion, splitting its waste evenly across `landfills`."""
    share = 1.0 / len(landfills)
    return AdvancedDSTCityRequest(
        city_name="Captureville",
        precipitation=1200.0,
        temperature=20.0,
        implement_year=IMPLEMENT_YEAR,
        waste_mass={"baseline": _constant(TOTAL_WASTE), "scenario": _constant(TOTAL_WASTE)},
        waste_fractions={"baseline": _constant(FRACTIONS), "scenario": _constant(FRACTIONS)},
        landfills=landfills,
        landfill_split_timeline={
            "baseline": _constant([share] * len(landfills)),
            "scenario": _constant([share] * len(landfills)),
        },
        country="USA",
    )


def _as_stated(defaults, index):
    """One site's reported default capture, shaped as a stated efficiency."""
    return {
        variant: {
            int(y): float(v)
            for y, v in defaults[variant][index]["gas_capture_efficiency"].items()
        }
        for variant in ("baseline", "scenario")
    }


def _assert_same_run(actual_request, expected_request):
    actual = run_advanced_dst_city(actual_request, with_site_emissions=True)
    expected = run_advanced_dst_city(expected_request, with_site_emissions=True)
    for half in ("baseline", "scenario"):
        pd.testing.assert_series_equal(actual[half]["total"], expected[half]["total"])
        for site, (got, want) in enumerate(
            zip(actual["site_emissions"][half], expected["site_emissions"][half])
        ):
            pd.testing.assert_series_equal(
                got.total["total"], want.total["total"], obj=f"{half} site {site}"
            )


def test_the_defaults_are_city_params_site_type_table():
    """One table, not a copy: the advanced DST reads city_params' own."""
    assert GAS_CAPTURE_DEFAULT == {
        "landfill": GAS_EFF_OPTIONS["Sanitary Landfill"],
        "controlled_dumpsite": GAS_EFF_OPTIONS["Controlled Dumpsite"],
        "dumpsite": GAS_EFF_OPTIONS["Dumpsite"],
    }
    assert GAS_CAPTURE_DEFAULT == {"landfill": 0.6, "controlled_dumpsite": 0.45, "dumpsite": 0.0}


@pytest.mark.parametrize("site_type", [LANDFILL, CONTROLLED_DUMP, OPEN_DUMP])
def test_an_unstated_efficiency_runs_as_the_type_default_stated(site_type):
    default = TYPE_DEFAULTS[site_type]
    _assert_same_run(
        _request([_landfill(site_type=site_type)]),
        _request([_landfill(site_type=site_type, gas=_constant(default))]),
    )


def test_an_unstated_efficiency_is_not_zero():
    """The fault this fixes: a landfill that collects gas captures some of it."""
    unstated = run_advanced_dst_city(_request([_landfill()]))
    none = run_advanced_dst_city(_request([_landfill(gas=_constant(0.0))]))
    for half in ("baseline", "scenario"):
        assert (unstated[half]["total"] < none[half]["total"]).loc[2001:].all()


@pytest.mark.parametrize("site_type", [LANDFILL, CONTROLLED_DUMP, OPEN_DUMP])
def test_defaults_report_the_type_capture_whether_or_not_the_site_collects(site_type):
    for gas in (None, _constant(0.0), _constant(0.3)):
        defaults = run_advanced_dst_city_defaults(
            _request([_landfill(site_type=site_type, gas=gas)])
        )
        for variant in ("baseline", "scenario"):
            reported = defaults[variant][0]["gas_capture_efficiency"]
            assert list(reported.index) == list(YEARS)
            assert (reported == TYPE_DEFAULTS[site_type]).all()


def test_the_default_follows_a_type_change_at_implement_year():
    """An open dump upgraded to a landfill captures nothing, then the landfill's 60%."""
    request = _request([_landfill(site_type=OPEN_DUMP, scenario_type=LANDFILL)])
    defaults = run_advanced_dst_city_defaults(request)
    baseline = defaults["baseline"][0]["gas_capture_efficiency"]
    scenario = defaults["scenario"][0]["gas_capture_efficiency"]
    assert (baseline == 0.0).all()
    assert (scenario.loc[: IMPLEMENT_YEAR - 1] == 0.0).all()
    assert (scenario.loc[IMPLEMENT_YEAR:] == 0.6).all()

    gas_base, gas_scen = _gas_capture(request.landfills[0], pd.Index(YEARS), IMPLEMENT_YEAR)
    pd.testing.assert_series_equal(gas_base, baseline, check_names=False)
    pd.testing.assert_series_equal(gas_scen, scenario, check_names=False)


def test_defaults_fed_back_as_stated_efficiencies_change_nothing():
    """The round trip: the figure a user sees greyed is the figure the model runs."""
    landfills = [
        _landfill(site_type=LANDFILL),
        _landfill(site_type=CONTROLLED_DUMP),
        _landfill(site_type=OPEN_DUMP),
        _landfill(site_type=OPEN_DUMP, scenario_type=LANDFILL),
        _landfill(site_type=CONTROLLED_DUMP, biocover={"baseline": _constant(0.5)}),
    ]
    unstated = _request(landfills)
    defaults = run_advanced_dst_city_defaults(unstated)
    stated = _request(
        [
            {**spec, "gas_capture_efficiency": _as_stated(defaults, index)}
            for index, spec in enumerate(landfills)
        ]
    )
    _assert_same_run(unstated, stated)

    # And the oxidation the defaults report is the same either way.
    restated = run_advanced_dst_city_defaults(stated)
    for variant in ("baseline", "scenario"):
        for index in range(len(landfills)):
            pd.testing.assert_series_equal(
                restated[variant][index]["oxidation"], defaults[variant][index]["oxidation"]
            )


def test_an_unstated_efficiency_counts_as_capture_for_oxidation():
    """A landfill capturing at its default oxidises at the capped rate, 22% not 10%."""
    for gas, expected in (
        (None, OX_CAP["landfill"]),
        (_constant(0.0), OX_NOCAP["landfill"]),
    ):
        defaults = run_advanced_dst_city_defaults(_request([_landfill(gas=gas)]))
        assert (defaults["baseline"][0]["oxidation"] == expected).all()


def test_a_series_can_switch_collection_on_without_an_efficiency():
    """No system until 2030, then one nobody measured: 0, then the landfill's 60%."""
    spec = _landfill(gas={2000: 0.0, 2030: None})
    gas_base, gas_scen = _gas_capture(
        _request([spec]).landfills[0], pd.Index(YEARS), IMPLEMENT_YEAR
    )
    for series in (gas_base, gas_scen):
        assert (series.loc[:2029] == 0.0).all()
        assert (series.loc[2030:] == 0.6).all()


def test_a_stated_efficiency_wins_over_the_default():
    spec = _landfill(gas={2000: 0.3, 2030: None, 2040: 0.8})
    gas_base, _ = _gas_capture(_request([spec]).landfills[0], pd.Index(YEARS), IMPLEMENT_YEAR)
    assert (gas_base.loc[:2029] == 0.3).all()
    assert (gas_base.loc[2030:2039] == 0.6).all()
    assert (gas_base.loc[2040:] == 0.8).all()


def test_stated_numbers_read_exactly_as_before():
    """A request with no null year takes the path it took before nulls existed."""
    years = pd.Index(YEARS)
    gas = {2000: 0.2, 2010: 0.35, 2035: 1.4}
    spec = _request([_landfill(site_type=CONTROLLED_DUMP, gas=gas)]).landfills[0]
    before = variant_series(
        {"baseline": gas, "scenario": gas}, years, IMPLEMENT_YEAR, default=0.0
    )
    after = _gas_capture(spec, years, IMPLEMENT_YEAR)
    for got, want in zip(after, before):
        pd.testing.assert_series_equal(got, want.clip(0.0, 1.0), check_names=False)


def test_an_empty_series_still_captures_nothing():
    spec = _landfill(gas={})
    gas_base, gas_scen = _gas_capture(
        _request([spec]).landfills[0], pd.Index(YEARS), IMPLEMENT_YEAR
    )
    assert (gas_base == 0.0).all() and (gas_scen == 0.0).all()


def test_other_landfills_take_an_unstated_efficiency_too():
    """`other_landfills` is a landfill spec like any other, null years included."""
    body = _request([_landfill(gas=_constant(0.0))]).model_dump()
    body["landfill_split_timeline"] = {
        "baseline": _constant([0.5]),
        "scenario": _constant([0.5]),
    }
    other = _landfill(site_type=CONTROLLED_DUMP)
    del other["landfill_open_close"]
    stated = {**other, "gas_capture_efficiency": {"baseline": _constant(0.45), "scenario": None}}

    unstated_request = AdvancedDSTCityRequest(**{**body, "other_landfills": other})
    stated_request = AdvancedDSTCityRequest(**{**body, "other_landfills": stated})
    _assert_same_run(unstated_request, stated_request)
    defaults = run_advanced_dst_city_defaults(unstated_request)
    assert (defaults["baseline"][-1]["gas_capture_efficiency"] == 0.45).all()
