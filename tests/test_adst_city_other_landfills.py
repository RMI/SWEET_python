"""Other landfills take whatever share of a city's landfilled waste the listed ones leave.

``AdvancedDSTCityRequest.other_landfills`` lets a caller state the sites it
knows about and leave the rest of the city's disposal to one more landfill,
without balancing the shares to 1 first. Each year other landfills take 1 minus
the shares of the listed landfills open that year, so a shut site's share goes
to them rather than being spread over the sites still open.

The test that matters most is the equivalence: a request with other landfills
is the same run, bit for bit, as one that lists that landfill itself at the
leftover share. That is what makes other landfills a convenience for the caller
rather than a second model to keep in step with the first.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from SWEET_python.advanced_dst_city import (
    AdvancedDSTCityRequest,
    run_advanced_dst_city,
    run_advanced_dst_city_defaults,
    run_advanced_dst_city_limits,
)
from SWEET_python.city_params import CustomError
from SWEET_python.constants import MODEL_START_YEAR
from SWEET_python.dst_common import MODEL_YEAR_MAX, NEVER_CLOSES

YEARS = range(MODEL_START_YEAR, MODEL_YEAR_MAX + 1)
FRACTIONS = [0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.02, 0.03, 0.0, 0.05]  # sums to 1.0
IMPLEMENT_YEAR = 2025
TOTAL_WASTE = 20_000.0

LANDFILL, CONTROLLED_DUMP, OPEN_DUMP = 0, 1, 2
ALWAYS_OPEN = (MODEL_START_YEAR, NEVER_CLOSES)


def _constant(value):
    return {y: value for y in YEARS}


def _site(*, site_type=LANDFILL, gas=0.0, dates=ALWAYS_OPEN, scenario_dates=None):
    return dict(
        landfill_type={"baseline": site_type, "scenario": site_type},
        landfill_open_close={
            "baseline": list(dates),
            "scenario": list(scenario_dates or dates),
        },
        gas_capture_efficiency={"baseline": _constant(gas), "scenario": _constant(gas)},
    )


def _other(*, site_type=OPEN_DUMP, gas=0.0, **fields):
    """Other landfills: a site with no open/close years of its own."""
    return dict(
        landfill_type={"baseline": site_type, "scenario": site_type},
        gas_capture_efficiency={"baseline": _constant(gas), "scenario": _constant(gas)},
        **fields,
    )


def _body(landfills, shares=None, *, other=None, scenario_shares=None, **extra):
    """A city composting and recycling 10% each, with `landfills` at `shares`."""
    body = dict(
        city_name="Elsewhereville",
        precipitation=1200.0,
        temperature=20.0,
        implement_year=IMPLEMENT_YEAR,
        waste_mass={"baseline": _constant(TOTAL_WASTE), "scenario": _constant(TOTAL_WASTE)},
        waste_fractions={"baseline": _constant(FRACTIONS), "scenario": _constant(FRACTIONS)},
        diversion_fractions={
            "baseline": {"compost": _constant(0.1), "recycling": _constant(0.1)},
            "scenario": {"compost": _constant(0.1), "recycling": _constant(0.1)},
        },
        landfills=landfills,
        country="USA",
    )
    if shares is not None:
        body["landfill_split_timeline"] = {
            "baseline": _constant(list(shares)),
            "scenario": scenario_shares
            if scenario_shares is not None
            else _constant(list(shares)),
        }
    if other is not None:
        body["other_landfills"] = other
    body.update(extra)
    return body


def _request(*args, **kwargs) -> AdvancedDSTCityRequest:
    return AdvancedDSTCityRequest(**_body(*args, **kwargs))


def _run(request):
    return run_advanced_dst_city(request, with_mass_flow=True, with_site_emissions=True)


def _site_shares(result, variant):
    """Each landfill's share of what the city landfilled, per year."""
    flow = result["mass_flow"][variant]
    landfilled = flow["landfilled"].sum(axis=1)
    return pd.concat(flow["sites"], axis=1).div(landfilled, axis=0)


def _assert_same_run(left, right):
    for variant in ("baseline", "scenario"):
        pd.testing.assert_frame_equal(left[variant], right[variant], check_exact=True)
        for mine, theirs in zip(
            left["site_emissions"][variant], right["site_emissions"][variant], strict=True
        ):
            pd.testing.assert_frame_equal(mine.total, theirs.total, check_exact=True)
        for mine, theirs in zip(
            left["mass_flow"][variant]["sites"],
            right["mass_flow"][variant]["sites"],
            strict=True,
        ):
            pd.testing.assert_series_equal(mine, theirs, check_exact=True)


# --------------------------------------------------------------------------- #
# Other landfills take the rest
# --------------------------------------------------------------------------- #
def test_other_landfills_take_what_the_listed_shares_leave():
    result = _run(_request([_site(), _site()], [0.25, 0.375], other=_other()))

    for variant in ("baseline", "scenario"):
        shares = _site_shares(result, variant)
        assert list(shares.columns) == [0, 1, 2]
        assert shares[0].to_numpy() == pytest.approx(0.25)
        assert shares[1].to_numpy() == pytest.approx(0.375)
        assert shares[2].to_numpy() == pytest.approx(0.375)


def test_a_run_with_other_landfills_is_the_run_that_lists_them():
    """The equivalence that makes other landfills safe to rely on."""
    with_other = _run(
        _request(
            [_site(gas=0.6), _site(site_type=OPEN_DUMP)],
            [0.25, 0.375],
            other=_other(site_type=LANDFILL),
        )
    )
    listed = _run(
        _request(
            [_site(gas=0.6), _site(site_type=OPEN_DUMP), _site(site_type=LANDFILL)],
            [0.25, 0.375, 0.375],
        )
    )
    _assert_same_run(with_other, listed)


def test_other_landfills_alone_take_all_of_the_city_s_waste():
    alone = _run(_request([], other=_other()))
    listed = _run(_request([_site(site_type=OPEN_DUMP)], [1.0]))

    _assert_same_run(alone, listed)
    assert _site_shares(alone, "baseline")[0].to_numpy() == pytest.approx(1.0)


def test_a_shut_site_s_share_goes_to_other_landfills():
    """Not spread over the sites still open, as it is without other landfills."""
    closes = 2030
    result = _run(
        _request(
            [_site(dates=(MODEL_START_YEAR, closes)), _site()],
            [0.25, 0.375],
            other=_other(),
        )
    )

    shares = _site_shares(result, "baseline")
    assert shares.loc[: closes - 1, 0].to_numpy() == pytest.approx(0.25)
    assert shares.loc[closes:, 0].to_numpy() == pytest.approx(0.0)
    assert shares[1].to_numpy() == pytest.approx(0.375)
    assert shares.loc[: closes - 1, 2].to_numpy() == pytest.approx(0.375)
    assert shares.loc[closes:, 2].to_numpy() == pytest.approx(0.625)


def test_a_site_not_yet_open_leaves_its_share_to_other_landfills():
    opens = 2010
    shares = _site_shares(
        _run(_request([_site(dates=(opens, NEVER_CLOSES))], [0.4], other=_other())),
        "baseline",
    )
    assert shares.loc[: opens - 1, 1].to_numpy() == pytest.approx(1.0)
    assert shares.loc[opens:, 1].to_numpy() == pytest.approx(0.6)


def test_a_scenario_closure_sends_the_share_to_other_landfills_in_the_scenario():
    closes = 2035
    result = _run(
        _request(
            [_site(scenario_dates=(MODEL_START_YEAR, closes)), _site()],
            [0.25, 0.375],
            other=_other(),
        )
    )

    baseline = _site_shares(result, "baseline")
    scenario = _site_shares(result, "scenario")
    assert baseline[2].to_numpy() == pytest.approx(0.375)
    assert scenario.loc[: closes - 1, 2].to_numpy() == pytest.approx(0.375)
    assert scenario.loc[closes:, 2].to_numpy() == pytest.approx(0.625)
    assert scenario[1].to_numpy() == pytest.approx(0.375)


def test_shares_over_one_are_scaled_down_and_other_landfills_take_nothing():
    shares = _site_shares(
        _run(_request([_site(), _site()], [0.8, 0.6], other=_other())), "baseline"
    )
    assert shares[0].to_numpy() == pytest.approx(0.8 / 1.4)
    assert shares[1].to_numpy() == pytest.approx(0.6 / 1.4)
    assert shares[2].to_numpy() == pytest.approx(0.0)


def test_a_tolerated_negative_share_sends_no_negative_waste_anywhere():
    """-0.01 passes the range check as a rounding error. Used as given, it buried
    negative waste at the listed site and gave other landfills 101% of the rest."""
    result = _run(_request([_site()], [-0.01], other=_other()))

    for variant in ("baseline", "scenario"):
        shares = _site_shares(result, variant)
        assert shares[0].to_numpy() == pytest.approx(0.0)
        assert shares[1].to_numpy() == pytest.approx(1.0)
        assert (pd.concat(result["mass_flow"][variant]["sites"], axis=1) >= 0).all().all()


def test_other_landfills_open_with_a_listed_site_that_opens_before_the_model():
    early = MODEL_START_YEAR - 10
    shares = _site_shares(
        _run(_request([_site(dates=(early, NEVER_CLOSES))], [0.4], other=_other())),
        "baseline",
    )
    assert int(shares.index.min()) == early
    assert shares[1].to_numpy() == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# The other entry points see the same landfills
# --------------------------------------------------------------------------- #
def test_defaults_report_other_landfills_last():
    with_other = run_advanced_dst_city_defaults(
        _request([_site(gas=0.6)], [0.25], other=_other(site_type=OPEN_DUMP))
    )
    listed = run_advanced_dst_city_defaults(
        _request([_site(gas=0.6), _site(site_type=OPEN_DUMP)], [0.25, 0.75])
    )
    for variant in ("baseline", "scenario"):
        assert len(with_other[variant]) == 2
        for mine, theirs in zip(with_other[variant], listed[variant], strict=True):
            pd.testing.assert_series_equal(mine["k"], theirs["k"], check_exact=True)
            pd.testing.assert_series_equal(
                mine["oxidation"], theirs["oxidation"], check_exact=True
            )


def test_an_empty_scenario_split_is_the_baseline_s_with_other_landfills():
    """`scenario: {}` means the baseline's shares, as it does without other landfills."""
    empty = _request([_site()], [0.25], other=_other(), scenario_shares={})
    stated = _request([_site()], [0.25], other=_other())

    _assert_same_run(_run(empty), _run(stated))
    for entry_point in (run_advanced_dst_city_limits, run_advanced_dst_city_defaults):
        entry_point(empty)  # refused before, as "landfill_split_timeline is required"


def test_limits_take_a_request_with_only_other_landfills():
    with_other = run_advanced_dst_city_limits(_request([], other=_other()))
    listed = run_advanced_dst_city_limits(_request([_site()], [1.0]))
    for variant in ("baseline", "scenario"):
        pd.testing.assert_frame_equal(
            with_other[variant]["max_diversion"], listed[variant]["max_diversion"]
        )


# --------------------------------------------------------------------------- #
# Requests that are refused, and the one that still is without other landfills
# --------------------------------------------------------------------------- #
def test_without_other_landfills_the_shares_must_still_sum_to_one():
    with pytest.raises(CustomError, match="sum to ~1"):
        run_advanced_dst_city(_request([_site(), _site()], [0.25, 0.375]))


@pytest.mark.parametrize("shares", [[1.5, 0.0], [-0.5, 0.25]])
def test_a_listed_share_must_still_be_a_fraction(shares):
    with pytest.raises(CustomError, match="between 0 and 1"):
        run_advanced_dst_city(_request([_site(), _site()], shares, other=_other()))


def test_a_pre_implement_scenario_row_the_model_discards_is_not_refused():
    scenario = {y: ([1.5, 0.0] if y < IMPLEMENT_YEAR else [0.25, 0.375]) for y in YEARS}
    result = _run(
        _request(
            [_site(), _site()], [0.25, 0.375], other=_other(), scenario_shares=scenario
        )
    )
    assert _site_shares(result, "scenario")[2].to_numpy() == pytest.approx(0.375)


@pytest.mark.parametrize(
    "field, value",
    [
        ("landfill_open_close", {"baseline": [2000, 2050], "scenario": [2000, 2050]}),
        ("accepted_waste_mass", {"baseline": {2020: 1.0}, "scenario": {2020: 1.0}}),
    ],
)
def test_other_landfills_take_no_dates_and_no_gate_total(field, value):
    with pytest.raises(ValidationError, match=f"other_landfills take no {field}"):
        _request([], other=_other(**{field: value}))


def test_a_request_needs_listed_landfills_or_other_landfills():
    with pytest.raises(ValidationError, match="unless other_landfills is given"):
        _request([])


def test_listed_landfills_still_need_their_shares():
    with pytest.raises(ValidationError, match="landfill_split_timeline is required"):
        _request([_site()], other=_other())
