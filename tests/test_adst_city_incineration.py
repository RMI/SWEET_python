"""A combusting facility in the city-level advanced DST burns what it receives.

``CityLandfillSpec.combusts`` marks a facility as an incinerator (with or
without energy recovery): the share of the city's disposed waste routed to it
still arrives, but only the unburnable reject -- ``City.combustion_reject_rate``,
10% -- is deposited and left to decay. The burnt 90% is destroyed, producing no
methane in this model, and is *not* re-routed to another site.

Before this existed the endpoint had no notion of burning at all, so a caller
describing an incinerator could only submit it as an ordinary landfill: the
facility decayed 100% of its intake and reported methane an incinerator cannot
physically produce. ``City.sdst_v1_5`` has always modeled a waste-to-energy site
correctly (combustion at 100% of intake for every open year, 10% reject
landfilled), so this brings the city-level path to parity with it.

The assertions lean on the engine being exactly linear in deposited mass --
oxidation, gas capture, flaring and MCF are all multiplicative fractions, and k
does not depend on mass -- so scaling a site's deposits by 0.1 scales its
emissions by 0.1 to floating point.
"""

import pytest

from SWEET_python.advanced_dst_city import (
    AdvancedDSTCityRequest,
    run_advanced_dst_city,
)
from SWEET_python.city_params import City

YEARS = range(2000, 2051)
FRACTIONS = [0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.02, 0.03, 0.0, 0.05]  # sums to 1.0
IMPLEMENT_YEAR = 2025
TOTAL_WASTE = 20_000.0

REJECT_RATE = City("reject_rate_probe").combustion_reject_rate


def _landfill(*, combusts=None, gas=0.0, site_type=2):
    spec = dict(
        landfill_type={"baseline": site_type, "scenario": site_type},
        landfill_open_close={"baseline": [2000, 2050], "scenario": [2000, 2050]},
        gas_capture_efficiency={
            "baseline": {y: gas for y in YEARS},
            "scenario": {y: gas for y in YEARS},
        },
    )
    if combusts is not None:
        spec["combusts"] = combusts
    return spec


def _request(landfills, shares) -> AdvancedDSTCityRequest:
    """A city with no diversion, so every tonne generated is disposed."""
    return AdvancedDSTCityRequest(
        city_name="Burnville",
        precipitation=1200.0,
        temperature=20.0,
        implement_year=IMPLEMENT_YEAR,
        waste_mass={
            "baseline": {y: TOTAL_WASTE for y in YEARS},
            "scenario": {y: TOTAL_WASTE for y in YEARS},
        },
        waste_fractions={
            "baseline": {y: FRACTIONS for y in YEARS},
            "scenario": {y: FRACTIONS for y in YEARS},
        },
        landfills=landfills,
        landfill_split_timeline={
            "baseline": {y: list(shares) for y in YEARS},
            "scenario": {y: list(shares) for y in YEARS},
        },
        country="USA",
    )


def _totals(result, half):
    return result[half]["total"]


def test_reject_rate_is_the_documented_ten_percent():
    """Guards the constant the rest of this module's arithmetic assumes."""
    assert REJECT_RATE == pytest.approx(0.1)


def test_a_combusting_facility_emits_only_the_rejects_share():
    depositing = _totals(run_advanced_dst_city(_request([_landfill()], [1.0])), "baseline")
    burning = _totals(
        run_advanced_dst_city(
            _request([_landfill(combusts={"baseline": True, "scenario": True})], [1.0])
        ),
        "baseline",
    )

    # Every year, not just the total: the reject is a flat share of intake.
    assert (depositing > 0).any()
    for year in YEARS:
        assert burning[year] == pytest.approx(
            depositing[year] * REJECT_RATE, rel=1e-9, abs=1e-12
        )


def test_omitting_combusts_is_unchanged_behaviour():
    """Back-compat: a caller that never heard of the field gets what it always got."""
    absent = _totals(run_advanced_dst_city(_request([_landfill()], [1.0])), "baseline")
    explicit_false = _totals(
        run_advanced_dst_city(
            _request([_landfill(combusts={"baseline": False, "scenario": False})], [1.0])
        ),
        "baseline",
    )

    for year in YEARS:
        assert absent[year] == pytest.approx(explicit_false[year], rel=1e-12)


def test_the_burnt_share_is_destroyed_not_moved_to_the_other_site():
    """A city half-served by an incinerator emits 55% of an all-landfill one.

    If the burnt 90% were re-routed to the surviving landfill instead of
    destroyed, this would come out at 100%.
    """
    all_landfill = _totals(
        run_advanced_dst_city(_request([_landfill(), _landfill()], [0.5, 0.5])),
        "baseline",
    )
    one_burns = _totals(
        run_advanced_dst_city(
            _request(
                [_landfill(), _landfill(combusts={"baseline": True, "scenario": True})],
                [0.5, 0.5],
            )
        ),
        "baseline",
    )

    expected_ratio = 0.5 + 0.5 * REJECT_RATE
    for year in YEARS:
        if all_landfill[year] == 0:
            continue
        assert one_burns[year] == pytest.approx(
            all_landfill[year] * expected_ratio, rel=1e-9
        )


def test_converting_a_landfill_to_an_incinerator_starts_at_implement_year():
    result = run_advanced_dst_city(
        _request([_landfill(combusts={"baseline": False, "scenario": True})], [1.0])
    )
    baseline, scenario = _totals(result, "baseline"), _totals(result, "scenario")

    # Nothing has changed before the conversion, so neither has any emission.
    # The engine emits a year's deposit from the *following* year onward, so
    # implement_year itself is still identical — the conversion changes what is
    # buried in 2025, and that shows up in 2026. That lag is the engine's
    # convention, not this field's: a siteTypeChange behaves the same way.
    for year in range(2000, IMPLEMENT_YEAR + 1):
        assert scenario[year] == pytest.approx(baseline[year], rel=1e-12)

    # From then on only the reject is deposited — but the tail of everything
    # buried beforehand keeps decaying, so this is a divergence that widens,
    # not a step straight down to 10%.
    for year in range(IMPLEMENT_YEAR + 1, 2051):
        assert scenario[year] < baseline[year]
    assert scenario[2050] / baseline[2050] < 0.2


def test_the_residue_pile_keeps_the_facilitys_own_gas_capture():
    """The reject is deposited *here*, so this site's own kit still applies."""
    without_capture = _totals(
        run_advanced_dst_city(
            _request(
                [_landfill(combusts={"baseline": True, "scenario": True}, gas=0.0)],
                [1.0],
            )
        ),
        "baseline",
    )
    with_capture = _totals(
        run_advanced_dst_city(
            _request(
                [_landfill(combusts={"baseline": True, "scenario": True}, gas=0.6)],
                [1.0],
            )
        ),
        "baseline",
    )

    assert (without_capture > 0).any()
    assert (with_capture.loc[2010:] < without_capture.loc[2010:]).all()
