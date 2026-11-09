"""A combusting facility in the city-level advanced DST burns what it receives.

``CityLandfillSpec.combusts`` marks a facility as an incinerator (with or
without energy recovery): the share of the city's disposed waste routed to it
still arrives, and only the reject -- ``City.combustion_reject_rate`` -- is
deposited and left to decay. The rest is destroyed, producing no methane in
this model, and is *not* re-routed to another site.

The reject is 0: everything sent to an incinerator stops producing methane.
It was 10% of the burnable waste, landfilled as the material it was, until
2026-10. What an incinerator leaves is ash with almost no organic carbon (see
the comment on ``City.combustion_reject_rate``).

Before ``combusts`` existed the endpoint had no notion of burning at all, so a
caller describing an incinerator could only submit it as an ordinary landfill:
the facility decayed 100% of its intake and reported methane an incinerator
cannot physically produce.

The tests with a reject set (``with_a_reject``) keep the machinery that deposits
one honest, in case the rate is ever raised again. They lean on the engine being
exactly linear in deposited mass -- oxidation, gas capture, flaring and MCF are
all multiplicative fractions, and k does not depend on mass -- so scaling a
site's deposits by the rate scales its emissions by the rate to floating point.
"""

import pytest

from SWEET_python import advanced_dst_city
from SWEET_python.advanced_dst_city import (
    AdvancedDSTCityRequest,
    run_advanced_dst_city,
)
from SWEET_python.city_params import City

YEARS = range(2000, 2051)
FRACTIONS = [0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.02, 0.03, 0.0, 0.05]  # sums to 1.0
IMPLEMENT_YEAR = 2025
TOTAL_WASTE = 20_000.0

#: A reject rate to exercise the deposit path with: the old 10%.
SET_REJECT = 0.1

BURNS = {"baseline": True, "scenario": True}


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


def _request(landfills, shares, diversion=None) -> AdvancedDSTCityRequest:
    """A city with no diversion unless it says, so every tonne generated is disposed."""
    return AdvancedDSTCityRequest(
        diversion_fractions=diversion,
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


@pytest.fixture
def with_a_reject(monkeypatch):
    """Every ``City`` the endpoint builds carries ``SET_REJECT``."""
    original = City.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.combustion_reject_rate = SET_REJECT

    monkeypatch.setattr(City, "__init__", init)
    assert advanced_dst_city.City is City
    return SET_REJECT


# --------------------------------------------------------------------------- #
# The default: nothing sent to an incinerator decays
# --------------------------------------------------------------------------- #
def test_the_reject_rate_is_zero():
    assert City("reject_rate_probe").combustion_reject_rate == 0.0


def test_a_combusting_facility_emits_nothing():
    depositing = _totals(run_advanced_dst_city(_request([_landfill()], [1.0])), "baseline")
    burning = _totals(
        run_advanced_dst_city(_request([_landfill(combusts=BURNS)], [1.0])), "baseline"
    )

    assert (depositing > 0).any()
    for year in YEARS:
        assert burning[year] == 0.0


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
    """A city half-served by an incinerator emits half of an all-landfill one.

    If the burnt half were re-routed to the surviving landfill instead of
    destroyed, this would come out at 100%.
    """
    all_landfill = _totals(
        run_advanced_dst_city(_request([_landfill(), _landfill()], [0.5, 0.5])),
        "baseline",
    )
    one_burns = _totals(
        run_advanced_dst_city(
            _request([_landfill(), _landfill(combusts=BURNS)], [0.5, 0.5])
        ),
        "baseline",
    )

    for year in YEARS:
        if all_landfill[year] == 0:
            continue
        assert one_burns[year] == pytest.approx(all_landfill[year] * 0.5, rel=1e-9)


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

    # From then on nothing is deposited, but everything buried beforehand keeps
    # decaying, so the scenario falls away from the baseline rather than to 0.
    for year in range(IMPLEMENT_YEAR + 1, 2051):
        assert 0 < scenario[year] < baseline[year]
    assert scenario[2050] / baseline[2050] < 0.2


def test_the_combustion_pathway_diverts_all_it_takes():
    """Upstream combustion, the city DST's way, keeps nothing back either."""
    combustion = {"combustion": {y: 0.4 for y in YEARS}}
    request = _request(
        [_landfill()], [1.0], diversion={"baseline": combustion, "scenario": combustion}
    )
    flow = run_advanced_dst_city(request, with_mass_flow=True)["mass_flow"]["baseline"]

    generated = flow["generated"].sum(axis=1)
    burnt = flow["diverted"]["combustion"].sum(axis=1)
    for year in YEARS:
        assert burnt[year] == pytest.approx(0.4 * generated[year], rel=1e-12)


# --------------------------------------------------------------------------- #
# With a reject set: what is kept back is deposited at the facility
# --------------------------------------------------------------------------- #
def test_a_reject_is_deposited_as_its_share_of_intake(with_a_reject):
    depositing = _totals(run_advanced_dst_city(_request([_landfill()], [1.0])), "baseline")
    burning = _totals(
        run_advanced_dst_city(_request([_landfill(combusts=BURNS)], [1.0])), "baseline"
    )

    # Every year, not just the total: the reject is a flat share of intake.
    assert (depositing > 0).any()
    for year in YEARS:
        assert burning[year] == pytest.approx(
            depositing[year] * with_a_reject, rel=1e-9, abs=1e-12
        )


def test_a_reject_is_not_moved_to_the_other_site(with_a_reject):
    all_landfill = _totals(
        run_advanced_dst_city(_request([_landfill(), _landfill()], [0.5, 0.5])),
        "baseline",
    )
    one_burns = _totals(
        run_advanced_dst_city(
            _request([_landfill(), _landfill(combusts=BURNS)], [0.5, 0.5])
        ),
        "baseline",
    )

    expected_ratio = 0.5 + 0.5 * with_a_reject
    for year in YEARS:
        if all_landfill[year] == 0:
            continue
        assert one_burns[year] == pytest.approx(
            all_landfill[year] * expected_ratio, rel=1e-9
        )


def test_a_reject_pile_keeps_its_own_gas_capture(with_a_reject):
    """The reject is deposited *here*, so this site's own kit still applies."""
    without_capture = _totals(
        run_advanced_dst_city(_request([_landfill(combusts=BURNS, gas=0.0)], [1.0])),
        "baseline",
    )
    with_capture = _totals(
        run_advanced_dst_city(_request([_landfill(combusts=BURNS, gas=0.6)], [1.0])),
        "baseline",
    )

    assert (without_capture > 0).any()
    assert (with_capture.loc[2010:] < without_capture.loc[2010:]).all()
