"""Why other landfills hold a city's waste, and the methane that waste makes.

Other landfills take a listed landfill's share in the years it is shut, and
whatever the listed shares leave. So they can emit plenty in a year they
receive little: their waste may be from decades before the listed sites opened.
``run_advanced_dst_city(..., with_other_landfills_reasons=True)`` splits their
share of each year's waste, and their methane, by why the waste is there.

The tests that matter most: the shares are the ones the run used, the methane
by reason adds up to other landfills' own, and asking changes nothing.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python.advanced_dst_city import (
    OTHER_LANDFILLS_REASONS,
    AdvancedDSTCityRequest,
    run_advanced_dst_city,
)
from SWEET_python.constants import MODEL_START_YEAR
from SWEET_python.dst_common import MODEL_YEAR_MAX, NEVER_CLOSES

YEARS = range(MODEL_START_YEAR, MODEL_YEAR_MAX + 1)
FRACTIONS = [0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.02, 0.03, 0.0, 0.05]  # sums to 1.0
IMPLEMENT_YEAR = 2025
LANDFILL, OPEN_DUMP = 0, 2
ALWAYS_OPEN = (MODEL_START_YEAR, NEVER_CLOSES)


def _constant(value):
    return {y: value for y in YEARS}


def _site(dates=ALWAYS_OPEN, scenario_dates=None):
    return dict(
        landfill_type={"baseline": LANDFILL, "scenario": LANDFILL},
        landfill_open_close={
            "baseline": list(dates),
            "scenario": list(scenario_dates or dates),
        },
        gas_capture_efficiency={"baseline": _constant(0.3), "scenario": _constant(0.3)},
    )


OTHER = dict(
    landfill_type={"baseline": OPEN_DUMP, "scenario": OPEN_DUMP},
    gas_capture_efficiency={"baseline": _constant(0.0), "scenario": _constant(0.0)},
)


def _request(landfills, shares, *, other=True) -> AdvancedDSTCityRequest:
    body = dict(
        city_name="Elsewhereville",
        precipitation=1200.0,
        temperature=20.0,
        implement_year=IMPLEMENT_YEAR,
        waste_mass={"baseline": _constant(20_000.0), "scenario": _constant(20_000.0)},
        waste_fractions={"baseline": _constant(FRACTIONS), "scenario": _constant(FRACTIONS)},
        landfills=landfills,
        landfill_split_timeline={
            "baseline": _constant(list(shares)),
            "scenario": _constant(list(shares)),
        },
        country="USA",
    )
    if other:
        body["other_landfills"] = OTHER
    return AdvancedDSTCityRequest(**body)


def _reasons(request):
    return run_advanced_dst_city(
        request, with_site_emissions=True, with_other_landfills_reasons=True
    )


# A site that opens in 2000 and closes in 2030 at 70%, one always open at 20%,
# and 10% nobody claims.
OPENS_AND_CLOSES = ([_site((2000, 2030)), _site()], [0.7, 0.2])


def test_shares_say_why_other_landfills_hold_each_year():
    share = _reasons(_request(*OPENS_AND_CLOSES))["other_landfills"]["baseline"]["share"]

    expected = {
        1990: dict(named=0.2, before_opening=0.7, after_closing=0.0, unassigned=0.1),
        2010: dict(named=0.9, before_opening=0.0, after_closing=0.0, unassigned=0.1),
        # Its close year is still an intake year.
        2030: dict(named=0.9, before_opening=0.0, after_closing=0.0, unassigned=0.1),
        2031: dict(named=0.2, before_opening=0.0, after_closing=0.7, unassigned=0.1),
    }
    for year, row in expected.items():
        assert share.loc[year].to_dict() == pytest.approx(row), year
    np.testing.assert_allclose(share.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_shares_are_the_ones_the_run_used():
    result = run_advanced_dst_city(
        _request(*OPENS_AND_CLOSES), with_mass_flow=True, with_other_landfills_reasons=True
    )
    for variant in ("baseline", "scenario"):
        flow = result["mass_flow"][variant]
        landfilled = flow["landfilled"].sum(axis=1)
        other_share = flow["sites"][-1] / landfilled
        reasons = result["other_landfills"][variant]["share"][list(OTHER_LANDFILLS_REASONS)]
        np.testing.assert_allclose(
            reasons.sum(axis=1).reindex(other_share.index), other_share, rtol=1e-12
        )


def test_methane_by_reason_adds_up_to_other_landfills_own():
    result = _reasons(_request(*OPENS_AND_CLOSES))
    for variant in ("baseline", "scenario"):
        own = result["site_emissions"][variant][-1].city["total"]
        by_reason = result["other_landfills"][variant]["emissions"]
        assert list(by_reason.columns) == list(OTHER_LANDFILLS_REASONS)
        np.testing.assert_allclose(
            by_reason.sum(axis=1).reindex(own.index), own, rtol=1e-9, atol=1e-9
        )


def test_methane_follows_the_waste_that_made_it():
    emissions = _reasons(_request(*OPENS_AND_CLOSES))["other_landfills"]["baseline"][
        "emissions"
    ]
    # Waste sent before the site opened keeps emitting long after 2000, and
    # waste sent after it closed makes none before 2031.
    assert emissions.loc[2026, "before_opening"] > 0
    assert emissions.loc[:2031, "after_closing"].eq(0).all()
    assert emissions.loc[2040, "after_closing"] > 0


def test_a_scenario_closure_is_after_closing():
    request = _request(
        [_site(scenario_dates=(MODEL_START_YEAR, 2035)), _site()], [0.7, 0.2]
    )
    result = _reasons(request)["other_landfills"]
    assert result["baseline"]["share"].loc[2040, "after_closing"] == 0.0
    assert result["scenario"]["share"].loc[2040, "after_closing"] == pytest.approx(0.7)
    assert result["scenario"]["share"].loc[2035, "after_closing"] == 0.0


def test_shares_over_one_scale_the_shut_sites_to_what_is_left():
    # 0.7 + 0.5 = 1.2. Before 2000 the open site claims 0.5, leaving 0.5, which
    # the unopened site's 0.7 more than covers. From 2000 the two are scaled
    # down to fit and other landfills hold nothing.
    share = _reasons(_request([_site((2000, NEVER_CLOSES)), _site()], [0.7, 0.5]))[
        "other_landfills"
    ]["baseline"]["share"]
    assert share.loc[1990].to_dict() == pytest.approx(
        dict(named=0.5, before_opening=0.5, after_closing=0.0, unassigned=0.0)
    )
    assert share.loc[2010].to_dict() == pytest.approx(
        dict(named=1.0, before_opening=0.0, after_closing=0.0, unassigned=0.0)
    )


def test_no_listed_landfills_leaves_it_all_unassigned():
    result = _reasons(_request([], []))["other_landfills"]["baseline"]
    assert result["share"]["unassigned"].eq(1.0).all()
    assert result["emissions"][["before_opening", "after_closing"]].eq(0).all().all()


def test_a_request_without_other_landfills_has_none():
    result = _reasons(_request([_site()], [1.0], other=False))
    assert result["other_landfills"] is None


def test_asking_changes_no_emissions():
    request = _request(*OPENS_AND_CLOSES)
    plain = run_advanced_dst_city(request, with_site_emissions=True)
    asked = _reasons(request)
    for variant in ("baseline", "scenario"):
        pd.testing.assert_frame_equal(plain[variant], asked[variant], check_exact=True)
        for mine, theirs in zip(
            plain["site_emissions"][variant], asked["site_emissions"][variant], strict=True
        ):
            pd.testing.assert_frame_equal(mine.total, theirs.total, check_exact=True)
