"""``run_advanced_dst_city(request, with_mass_flow=True)`` reports the model's own frames.

The mass flow is not a second model or an alternative path: every frame it
returns is one the emissions were already computed from. Before this existed
they were locals, consumed and dropped -- the landfilled residual was built only
for the negative-mass sign check and then discarded. The flag keeps them and
returns them under a ``mass_flow`` key. So the first thing worth pinning is that
asking for it changes nothing: same emissions, to the bit.

That is the whole point of the feature. A caller that wants to *show* where the
tonnage went would otherwise re-derive the diversion split, and that split is
per waste type with a per-component reject rate -- compost and anaerobic
digestion draw only on organics, recycling only on recyclables -- so
approximating it as one scalar on the total quietly composts metal and glass.

The bands are ``generated`` (per component, before prevention), ``prevented``,
``diverted`` (per pathway per component, net of rejects), ``landfilled`` (the
residual) and ``sites`` (per landfill, in request order, windowed by open/close).
They balance per component per year, with one deliberate exception that
``test_the_only_gap_left_is_a_year_with_nowhere_to_put_the_waste`` pins.
"""

import pandas as pd
import pytest

from SWEET_python import dst_common as common
from SWEET_python.advanced_dst_city import (
    DIVERSION_PATHWAYS,
    AdvancedDSTCityRequest,
    run_advanced_dst_city,
)
from SWEET_python.city_params import CustomError

YEARS = range(2000, 2051)
# Food-heavy so food-waste prevention has something to bite on. Sums to 1.0.
FRACTIONS = [0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.02, 0.03, 0.0, 0.05]
IMPLEMENT_YEAR = 2025
TOTAL_WASTE = 20_000.0
AFTER = 2040          # comfortably past both the implement year and the early close
BEFORE = 2010         # comfortably before the implement year


def _landfill(*, open_close=(2000, 2050)):
    return dict(
        landfill_type={"baseline": 2, "scenario": 2},
        landfill_open_close={"baseline": list(open_close), "scenario": list(open_close)},
        gas_capture_efficiency={
            "baseline": {y: 0.0 for y in YEARS},
            "scenario": {y: 0.0 for y in YEARS},
        },
    )


def _request(landfills, shares, *, diversion=None, food_prevention=None):
    """A city whose scenario may divert and/or prevent; baseline never does."""
    extra = {}
    if diversion is not None:
        extra["diversion_fractions"] = {"baseline": {}, "scenario": diversion}
    if food_prevention is not None:
        extra["food_waste_prevention"] = {
            "baseline": {y: 0.0 for y in YEARS},
            "scenario": {y: food_prevention for y in YEARS},
        }
    return AdvancedDSTCityRequest(
        city_name="Flowville",
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
        **extra,
    )


def _busy_request(shares=(1.0,), landfills=None):
    """Diversion and prevention both on, so every band carries mass."""
    return _request(
        landfills if landfills is not None else [_landfill()],
        shares,
        diversion={"compost": {y: 0.2 for y in YEARS}},
        food_prevention=0.5,
    )


# --------------------------------------------------------------------------- #
# It is a report, not a second model
# --------------------------------------------------------------------------- #

def test_the_default_return_carries_no_mass_flow():
    result = run_advanced_dst_city(_busy_request())
    assert sorted(result) == ["baseline", "scenario"]


def test_asking_for_the_mass_flow_does_not_change_the_emissions():
    """The flag must be pure reporting. If this fails, it has become a fork."""
    plain = run_advanced_dst_city(_busy_request())
    with_flow = run_advanced_dst_city(_busy_request(), with_mass_flow=True)

    assert sorted(with_flow) == ["baseline", "mass_flow", "scenario"]
    for variant in ("baseline", "scenario"):
        pd.testing.assert_frame_equal(plain[variant], with_flow[variant])


def test_both_variants_are_reported():
    mass_flow = run_advanced_dst_city(_busy_request(), with_mass_flow=True)["mass_flow"]
    assert sorted(mass_flow) == ["baseline", "scenario"]
    for variant in mass_flow.values():
        assert sorted(variant) == ["diverted", "generated", "landfilled", "prevented", "sites"]


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #

def test_every_frame_carries_all_ten_components_in_one_order():
    """Each pathway frame natively holds only what that pathway can draw on --
    compost has four columns, not ten -- and frame subtraction reorders them
    alphabetically. Callers are promised one stable shape instead."""
    flow = run_advanced_dst_city(_busy_request(), with_mass_flow=True)["mass_flow"]["scenario"]

    frames = [flow["generated"], flow["prevented"], flow["landfilled"]]
    frames += list(flow["diverted"].values())
    assert frames, "fixture produced no frames to check"
    for frame in frames:
        assert list(frame.columns) == list(common.WASTE_COMPONENTS)
        assert not frame.isna().to_numpy().any(), "absent components must be zero, not NaN"


def test_sites_is_one_series_per_landfill_in_request_order():
    landfills = [_landfill(), _landfill()]
    flow = run_advanced_dst_city(
        _busy_request(shares=(0.75, 0.25), landfills=landfills), with_mass_flow=True
    )["mass_flow"]["scenario"]

    assert len(flow["sites"]) == len(landfills)
    first, second = (float(s.loc[AFTER]) for s in flow["sites"])
    assert second > 0
    # 0.75 / 0.25 -- position in the list is the identity, so a reorder shows up here.
    assert first == pytest.approx(3.0 * second)


# --------------------------------------------------------------------------- #
# The bands account for the tonnage
# --------------------------------------------------------------------------- #

def test_the_bands_balance_per_component_and_year():
    """generated - prevented - diverted == landfilled, everywhere."""
    flow = run_advanced_dst_city(_busy_request(), with_mass_flow=True)["mass_flow"]["scenario"]

    diverted_total = sum(
        flow["diverted"].values(),
        start=pd.DataFrame(0.0, index=flow["generated"].index, columns=flow["generated"].columns),
    )
    residual = flow["generated"] - flow["prevented"] - diverted_total - flow["landfilled"]
    assert float(residual.abs().to_numpy().max()) == pytest.approx(0.0, abs=1e-6)


def test_generated_is_the_stream_before_prevention_and_prevented_is_the_difference():
    """`generated` is what the caller authored; prevention is reported separately
    rather than folded into it, so a reader can see both."""
    flow = run_advanced_dst_city(_busy_request(), with_mass_flow=True)["mass_flow"]["scenario"]

    generated = flow["generated"].loc[AFTER]
    prevented = flow["prevented"].loc[AFTER]

    # Authored total, not the shrunken one.
    assert float(generated.sum()) == pytest.approx(TOTAL_WASTE)
    # Half the food, and nothing else moves: prevention is food-only by definition.
    assert float(prevented["food"]) == pytest.approx(0.5 * float(generated["food"]))
    assert float(prevented.drop("food").abs().sum()) == pytest.approx(0.0, abs=1e-9)


def test_the_scenario_bands_match_the_baseline_before_the_implement_year():
    """Nothing the scenario changes may leak backwards in time."""
    flow = run_advanced_dst_city(_busy_request(), with_mass_flow=True)["mass_flow"]

    pd.testing.assert_series_equal(
        flow["scenario"]["generated"].loc[BEFORE],
        flow["baseline"]["generated"].loc[BEFORE],
    )
    assert float(flow["scenario"]["prevented"].loc[BEFORE].abs().sum()) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Diversion detail, which is the reason the feature exists
# --------------------------------------------------------------------------- #

def test_a_pathway_the_request_never_uses_is_omitted_rather_than_sent_as_zeros():
    """Most cities run two of the four, so this roughly halves the payload. The
    documented contract is that a missing pathway reads as zero."""
    flow = run_advanced_dst_city(_busy_request(), with_mass_flow=True)["mass_flow"]["scenario"]

    assert sorted(flow["diverted"]) == ["compost"]
    assert float(flow["diverted"]["compost"].to_numpy().sum()) > 0
    unused = set(DIVERSION_PATHWAYS) - set(flow["diverted"])
    assert unused == {"anaerobic", "combustion", "recycling"}


def test_diverted_is_net_of_the_reject_rates():
    """The requested share is what is *collected*; each component rejects part of
    it. A caller multiplying the share by the total would overstate diversion --
    which is the approximation this payload exists to make unnecessary."""
    flow = run_advanced_dst_city(_busy_request(), with_mass_flow=True)["mass_flow"]["scenario"]

    landfilled_plus_diverted = float(
        flow["landfilled"].loc[AFTER].sum() + flow["diverted"]["compost"].loc[AFTER].sum()
    )
    net_generated = float(
        flow["generated"].loc[AFTER].sum() - flow["prevented"].loc[AFTER].sum()
    )
    collected = 0.2 * net_generated
    actually_composted = float(flow["diverted"]["compost"].loc[AFTER].sum())

    assert 0 < actually_composted < collected, "rejects should shrink the collected share"
    # Nothing vanished: the rejects went to landfill.
    assert landfilled_plus_diverted == pytest.approx(net_generated)


def test_compost_only_draws_on_the_components_it_can_take():
    """The per-waste-type split is the substance of `diverted`: metal and glass
    are not compostable and must stay at zero."""
    flow = run_advanced_dst_city(_busy_request(), with_mass_flow=True)["mass_flow"]["scenario"]
    compost = flow["diverted"]["compost"].loc[AFTER]

    for inert in ("metal", "glass", "plastic"):
        assert float(compost[inert]) == pytest.approx(0.0, abs=1e-9), f"{inert} was composted"
    assert float(compost["food"]) > 0


# --------------------------------------------------------------------------- #
# The one place the bands deliberately do not tie out
# --------------------------------------------------------------------------- #

def test_a_site_receives_nothing_in_a_year_it_is_closed():
    landfills = [_landfill(), _landfill(open_close=(2000, 2030))]
    flow = run_advanced_dst_city(
        _busy_request(shares=(0.5, 0.5), landfills=landfills), with_mass_flow=True
    )["mass_flow"]["scenario"]

    closed = flow["sites"][1]
    assert float(closed.loc[2020]) > 0, "it should have received waste while open"
    assert float(closed.loc[AFTER]) == pytest.approx(0.0), "closed sites take nothing"


def test_a_share_pointed_at_a_closed_site_is_redistributed_not_lost():
    """A share aimed at a shut landfill is not a share of anything.

    It used to be: the split was honoured as submitted, so a year still routing
    half the waste to a closed site scaled that half onto it and the window then
    zeroed it — the mass left the model, and `sites` came to half of
    `landfilled`. The split is meant to account for *all* the post-diversion
    waste, so it is now gated to the open landfills and renormalized over them,
    and the open site receives the whole of it.
    """
    landfills = [_landfill(), _landfill(open_close=(2000, 2030))]
    flow = run_advanced_dst_city(
        _busy_request(shares=(0.5, 0.5), landfills=landfills), with_mass_flow=True
    )["mass_flow"]["scenario"]

    landfilled = float(flow["landfilled"].loc[AFTER].sum())
    arrived = [float(s.loc[AFTER]) for s in flow["sites"]]

    assert landfilled > 0
    assert sum(arrived) == pytest.approx(landfilled)
    # The whole stream goes to the one site still open, not half of it.
    assert arrived[0] == pytest.approx(landfilled)
    assert arrived[1] == pytest.approx(0.0)


def test_the_only_gap_left_is_a_year_with_nowhere_to_put_the_waste():
    """`sites` falls short of `landfilled` in exactly one case, on purpose.

    Renormalizing over the open landfills closes the gap whenever at least one
    is accepting waste. A year in which *none* is open has no denominator to
    renormalize over: that city's post-diversion waste has nowhere to go, which
    is a real thing for a single-site city past its site's closure. The shares
    go to zero rather than raising, and the gap is reported rather than hidden.
    """
    flow = run_advanced_dst_city(
        _busy_request(shares=(1.0,), landfills=[_landfill(open_close=(2000, 2030))]),
        with_mass_flow=True,
    )["mass_flow"]["scenario"]

    landfilled = float(flow["landfilled"].loc[AFTER].sum())
    arrived = sum(float(s.loc[AFTER]) for s in flow["sites"])

    assert landfilled > 0, "the city still generates and disposes of waste"
    assert arrived == pytest.approx(0.0), "but no landfill is open to receive it"


def test_sites_sum_to_landfilled_when_every_site_is_open():
    """The complement of the test above: with no closed site there is no gap."""
    flow = run_advanced_dst_city(
        _busy_request(shares=(0.5, 0.5), landfills=[_landfill(), _landfill()]),
        with_mass_flow=True,
    )["mass_flow"]["scenario"]

    landfilled = float(flow["landfilled"].loc[AFTER].sum())
    arrived = sum(float(s.loc[AFTER]) for s in flow["sites"])
    assert arrived == pytest.approx(landfilled)


# --------------------------------------------------------------------------- #
# The submitted split has to be a split
# --------------------------------------------------------------------------- #
def test_a_negative_share_is_refused_rather_than_buried():
    """Summing to one does not make a row a split.

    ``[-0.5, 1.5]`` sums to exactly 1.0 and passed every check the sum could
    make, then scaled a negative mass onto the first landfill and buried it
    there — a site accumulating negative stock, decaying into negative methane.
    Nothing downstream catches it: ``create_advanced`` multiplies straight
    through, and ``over_diversion`` measures the city's residual before the
    split is applied.
    """
    with pytest.raises(CustomError) as excinfo:
        run_advanced_dst_city(
            _busy_request(shares=(-0.5, 1.5), landfills=[_landfill(), _landfill()])
        )
    assert excinfo.value.code == "invalid_parameters"
    assert "between 0 and 1" in excinfo.value.message
    # The message names the year and which landfill was out of range.
    assert "landfill 0" in excinfo.value.message


def test_a_share_above_one_is_refused_too():
    with pytest.raises(CustomError) as excinfo:
        run_advanced_dst_city(
            _busy_request(shares=(1.5, -0.5), landfills=[_landfill(), _landfill()])
        )
    assert excinfo.value.code == "invalid_parameters"
    assert "between 0 and 1" in excinfo.value.message


def test_a_well_formed_split_still_runs():
    """The guard is a floor, not a tightening: ordinary splits are untouched."""
    flow = run_advanced_dst_city(
        _busy_request(shares=(0.3, 0.7), landfills=[_landfill(), _landfill()]),
        with_mass_flow=True,
    )["mass_flow"]["scenario"]
    landfilled = float(flow["landfilled"].loc[AFTER].sum())
    arrived = [float(s.loc[AFTER]) for s in flow["sites"]]
    assert arrived[0] == pytest.approx(0.3 * landfilled)
    assert arrived[1] == pytest.approx(0.7 * landfilled)


def test_a_pre_implement_scenario_row_the_splice_discards_is_not_validated():
    """The scenario's own shares before `implement_year` are never used.

    Scenario tracks baseline until changes take effect, so the whole
    pre-implement half of `scenario_split` is overwritten with baseline's. It
    was validated first, though, so a caller who left those rows malformed —
    within the contract, since the model ignores them — was rejected for a
    number that would have been thrown away. Renormalization had the same
    ordering problem, but its result was overwritten by the splice, so only the
    error was ever observable.
    """
    landfills = [_landfill(), _landfill()]
    baseline = {y: [0.5, 0.5] for y in YEARS}
    # Malformed only before the implement year, well-formed from it onward.
    scenario = {
        y: ([0.25, 0.25] if y < IMPLEMENT_YEAR else [0.5, 0.5]) for y in YEARS
    }
    request = AdvancedDSTCityRequest(
        city_name="Flowville",
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
        landfill_split_timeline={"baseline": baseline, "scenario": scenario},
        country="USA",
    )

    result = run_advanced_dst_city(request, with_mass_flow=True)

    # And the pre-implement scenario really does track baseline.
    base = result["mass_flow"]["baseline"]["sites"]
    scen = result["mass_flow"]["scenario"]["sites"]
    for index in range(len(landfills)):
        assert float(scen[index].loc[BEFORE]) == pytest.approx(
            float(base[index].loc[BEFORE])
        )


def test_a_post_implement_scenario_row_is_still_validated():
    """The loosening stops at `implement_year` — those shares are used."""
    landfills = [_landfill(), _landfill()]
    scenario = {
        y: ([0.5, 0.5] if y < IMPLEMENT_YEAR else [0.25, 0.25]) for y in YEARS
    }
    request = AdvancedDSTCityRequest(
        city_name="Flowville",
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
            "baseline": {y: [0.5, 0.5] for y in YEARS},
            "scenario": scenario,
        },
        country="USA",
    )
    with pytest.raises(CustomError) as excinfo:
        run_advanced_dst_city(request)
    assert excinfo.value.code == "invalid_parameters"
    assert "sum to ~1" in excinfo.value.message
    assert str(IMPLEMENT_YEAR) in excinfo.value.message
