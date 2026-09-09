"""A named site can receive waste the city did not send.

Until now a city's disposal sites shared out exactly the city's landfilled
residual: ``landfill_split_timeline`` is *fractions that must sum to 1.0*
(``_validate_shares``), so a site receiving mass from anywhere else was
structurally unsayable. Real sites take waste from neighbouring municipalities,
from private haulers, from a regional catchment the baseline never describes --
and the site's own operator knows its gate total, not the city's share of it.

So a spec may now carry ``accepted_waste_mass``: everything crossing that site's
weighbridge in a year, **from every source**, city waste included. It is a gate
total, not the outside share -- the outside share is what it exceeds the city's
own allocation by, which is the model's subtraction to do rather than the user's.
A figure at or below that allocation therefore means "no outside waste", not a
smaller city stream. Two consequences, and this module exists to pin both:

**The city's own emissions must not move.** A city is responsible for the waste
it generates, wherever that waste goes. Other-source waste is somebody else's
inventory. The guarantee is structural rather than arithmetic: the other-source
landfills are never added to ``parameters.landfills``, which is the list
``City.sum_landfill_emissions`` iterates -- so the city total is bit-identical,
not merely close.

**A site's total is the sum of its streams, exactly.** SWEET's landfill
emissions are exactly linear in deposited mass -- deposited mass enters the
first-order-decay kernel once, multiplicatively (``model_v2`` ``ch4_produce =
ks_values * L_0[waste] * waste_masses * exp_term * mcf_values``), and every step
after it is a mass-independent factor. Measured here at ~1e-16 relative. That is
what makes "the city's share of this site's emissions" a physical quantity
rather than an allocation convention someone had to invent.

Linearity needs the two streams to agree on every parameter but mass, which the
implementation gets by building the surplus twin from the city stream's own
kwargs. Composition is not one of those parameters: ``k`` is computed once from
the city's generated mix and handed to every landfill, so a stream's own mix
moves only its per-component masses. The surplus carries the city's
post-diversion residual mix because that is the defensible reading of a gate
observation, not because superposition would otherwise fail.
"""

import numpy as np
import pytest

from SWEET_python.advanced_dst_city import (
    AdvancedDSTCityRequest,
    run_advanced_dst_city,
)

YEARS = list(range(1990, 2051))
# Food-heavy, so diversion has something to bite on and the residual
# composition is visibly different from the generated one.
FRACTIONS = [0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.02, 0.03, 0.0, 0.05]
IMPLEMENT_YEAR = 2025
GENERATED = 100_000.0


def _spec(
    *,
    open_close=(1990, 2051),
    accepted=None,
    accepted_scenario=None,
    combusts=None,
    combusts_scenario=None,
):
    spec = dict(
        landfill_type={"baseline": 2, "scenario": 2},
        landfill_open_close={"baseline": list(open_close), "scenario": list(open_close)},
        gas_capture_efficiency={
            "baseline": {y: 0.0 for y in YEARS},
            "scenario": {y: 0.0 for y in YEARS},
        },
    )
    if accepted is not None:
        spec["accepted_waste_mass"] = {
            "baseline": dict(accepted),
            "scenario": dict(accepted_scenario if accepted_scenario is not None else accepted),
        }
    if combusts is not None or combusts_scenario is not None:
        spec["combusts"] = {
            "baseline": bool(combusts),
            "scenario": bool(combusts if combusts_scenario is None else combusts_scenario),
        }
    return spec


def _request(specs, shares, *, diversion=None, generated=None):
    extra = {}
    if diversion is not None:
        extra["diversion_fractions"] = {
            "baseline": {"compost": {y: diversion for y in YEARS}},
            "scenario": {"compost": {y: diversion for y in YEARS}},
        }
    return AdvancedDSTCityRequest(
        city_name="Overflow",
        precipitation=1200.0,
        temperature=20.0,
        implement_year=IMPLEMENT_YEAR,
        waste_mass={
            "baseline": dict(generated) if generated else {y: GENERATED for y in YEARS},
            "scenario": dict(generated) if generated else {y: GENERATED for y in YEARS},
        },
        waste_fractions={
            "baseline": {y: FRACTIONS for y in YEARS},
            "scenario": {y: FRACTIONS for y in YEARS},
        },
        landfills=specs,
        landfill_split_timeline={
            "baseline": {y: list(shares) for y in YEARS},
            "scenario": {y: list(shares) for y in YEARS},
        },
        country="USA",
        **extra,
    )


def _total(frame):
    return np.asarray(frame["total"], dtype=float)


# `accepted_waste_mass` is the whole gate total, not the neighbour's share of
# it, so every fixture here has to clear whatever the city sends. With 25%
# composted the city buries ~78,154 t/yr of its 100,000 t; a site taking the
# city's entire residual and 100,000 t is therefore comfortably over.
GATE_WITH_NEIGHBOUR = {y: 100_000.0 for y in YEARS}

# A gate total that GROWS away from the city's flat allocation, so the city's
# share of the site falls year on year. Any attribution built on a same-year
# tonnage ratio gets this case badly wrong; see the last test.
GATE_GROWING = {y: 100_000.0 + 3_000.0 * (1.05 ** (y - 1990)) for y in YEARS}


# --------------------------------------------------------------------------- #
# The city's emissions are the city's waste, wherever it goes
# --------------------------------------------------------------------------- #

def test_other_source_waste_does_not_change_the_city_total():
    """The whole feature, in one assertion. Bit-identical, not merely close."""
    without = run_advanced_dst_city(_request([_spec()], (1.0,), diversion=0.25))
    with_other = run_advanced_dst_city(
        _request([_spec(accepted=GATE_WITH_NEIGHBOUR)], (1.0,), diversion=0.25)
    )

    for variant in ("baseline", "scenario"):
        assert without[variant].equals(with_other[variant]), (
            f"{variant}: other-source waste moved the city's own emissions"
        )


def test_an_absent_field_and_a_zero_series_are_the_same_request():
    """No silent second code path for the city that never uses this."""
    absent = run_advanced_dst_city(_request([_spec()], (1.0,), diversion=0.25))
    zeros = run_advanced_dst_city(
        _request([_spec(accepted={y: 0.0 for y in YEARS})], (1.0,), diversion=0.25)
    )

    for variant in ("baseline", "scenario"):
        assert absent[variant].equals(zeros[variant])


def test_site_shares_still_have_to_sum_to_one():
    """Other-source mass is additive, not an escape hatch from the split.

    The city's landfilled waste still goes entirely to the city's own sites --
    this feature opens the *site* side, not the city side.
    """
    from SWEET_python.city_params import CustomError

    with pytest.raises(CustomError):
        run_advanced_dst_city(
            _request([_spec(accepted=GATE_WITH_NEIGHBOUR), _spec()], (0.5, 0.2))
        )


# --------------------------------------------------------------------------- #
# A site's total is the sum of its streams
# --------------------------------------------------------------------------- #

def test_a_sites_emissions_split_into_the_city_and_the_rest():
    result = run_advanced_dst_city(
        _request([_spec(accepted=GATE_WITH_NEIGHBOUR)], (1.0,), diversion=0.25),
        with_site_emissions=True,
    )
    site = result["site_emissions"]["baseline"][0]

    city, other, total = _total(site.city), _total(site.other), _total(site.total)
    residual = np.abs(total - (city + other))
    scale = np.where(total == 0, 1.0, total)

    assert np.max(residual / scale) < 1e-12, "a site's streams do not add to its total"
    assert other.sum() > 0, "the neighbour's waste produced no emissions"


def test_the_city_contribution_is_what_the_city_alone_would_have_produced():
    """Superposition, stated as the thing a reader actually wants to trust.

    The city's slice of a shared site emits exactly what it would emit at a site
    nobody else used. If this fails, the attribution is a convention rather than
    a measurement.
    """
    alone = run_advanced_dst_city(
        _request([_spec()], (1.0,), diversion=0.25), with_site_emissions=True
    )
    shared = run_advanced_dst_city(
        _request([_spec(accepted=GATE_WITH_NEIGHBOUR)], (1.0,), diversion=0.25),
        with_site_emissions=True,
    )

    a = _total(alone["site_emissions"]["baseline"][0].city)
    b = _total(shared["site_emissions"]["baseline"][0].city)
    scale = np.where(a == 0, 1.0, a)

    assert np.max(np.abs(a - b) / scale) < 1e-12


def test_a_site_with_no_other_source_reports_a_total_equal_to_its_city_share():
    result = run_advanced_dst_city(
        _request([_spec()], (1.0,), diversion=0.25), with_site_emissions=True
    )
    site = result["site_emissions"]["baseline"][0]

    assert np.allclose(_total(site.other), 0.0)
    assert np.allclose(_total(site.total), _total(site.city), rtol=0, atol=1e-12)


def test_the_city_shares_of_every_site_add_up_to_the_city_total():
    """The two outputs reconcile: city emissions are the per-site city shares
    plus whatever the diversion pathways themselves emit."""
    request = _request(
        [_spec(accepted=GATE_WITH_NEIGHBOUR), _spec(accepted=GATE_GROWING)],
        (0.6, 0.4),
        diversion=0.25,
    )
    result = run_advanced_dst_city(request, with_site_emissions=True)

    per_site = sum(
        _total(site.city) for site in result["site_emissions"]["baseline"]
    )
    city_total = _total(result["baseline"])

    # The gap is the compost/anaerobic emissions, which are the city's but are
    # not at any landfill. It must be non-negative and smooth, not noise.
    diversion_share = city_total - per_site
    assert np.all(diversion_share > -1e-9)
    assert diversion_share.sum() > 0


# --------------------------------------------------------------------------- #
# The bug this feature is most likely to be "simplified" into
# --------------------------------------------------------------------------- #

def test_the_city_share_is_not_this_years_tonnage_ratio():
    """Do NOT reimplement attribution as (city tons / site tons) x site total.

    This year's emissions come from decades of deposit cohorts, each with its own
    city/other split. Where the split moves over time -- a growing neighbour, a
    city that started diverting -- the same-year ratio is wrong by tens of
    percent, and wrong in a direction that looks plausible. The kernel has to run
    on the city's own deposit series, which is what the implementation does by
    giving each stream its own Landfill object.

    This test fails if someone replaces that with the ratio.
    """
    result = run_advanced_dst_city(
        _request([_spec(accepted=GATE_GROWING)], (1.0,), diversion=0.0),
        with_site_emissions=True,
    )
    site = result["site_emissions"]["baseline"][0]
    city, total = _total(site.city), _total(site.total)

    settled = total > 0.01 * total.max()
    emission_share = (city / np.where(total == 0, 1.0, total))[settled]

    # No diversion in this run, so the city's allocation is its whole stream.
    city_tons = np.array([GENERATED for _ in YEARS])
    gate_tons = np.array([GATE_GROWING[y] for y in YEARS])
    mass_share = (city_tons / gate_tons)[settled]

    # The two shares must visibly disagree: emissions lag mass, so a shrinking
    # city holds a larger share of the emissions than of this year's intake.
    assert np.max(np.abs(emission_share - mass_share)) > 0.02, (
        "mass share and emission share agree -- either the test city is too "
        "static to be a regression guard, or attribution has been reduced to a ratio"
    )
    assert np.all(emission_share[-10:] > mass_share[-10:])


# --------------------------------------------------------------------------- #
# Assumptions the rest of the engine is entitled to keep making
# --------------------------------------------------------------------------- #

def test_no_surplus_emits_exactly_zero():
    """`==`, not `approx`. E(0) is exactly 0.0, so any residual is a bug."""
    result = run_advanced_dst_city(
        _request([_spec(accepted={y: 0.0 for y in YEARS})], (1.0,)),
        with_site_emissions=True,
    )

    assert (_total(result["site_emissions"]["baseline"][0].other) == 0.0).all()


def test_the_mass_flows_sites_band_is_still_only_the_city():
    """`sum(sites) == landfilled` is asserted in eleven places elsewhere.

    The band reports where the *city's* waste went, and other-source waste is
    not the city's. Folding it in here would be the easy mistake, and it would
    break the one identity the mass flow exists to keep.
    """
    plain = run_advanced_dst_city(_request([_spec()], (1.0,)), with_mass_flow=True)
    with_other = run_advanced_dst_city(
        _request([_spec(accepted=GATE_WITH_NEIGHBOUR)], (1.0,)), with_mass_flow=True
    )

    for variant in ("baseline", "scenario"):
        a = plain["mass_flow"][variant]
        b = with_other["mass_flow"][variant]
        assert a["sites"][0].equals(b["sites"][0])
        assert a["landfilled"].equals(b["landfilled"])


def test_a_combusting_site_burns_everything_it_accepts():
    """The surplus meets the same furnace the city's waste does.

    Compared at the gate but deposited after the burn, so a site stated at
    150,000 t buries the reject of 150,000 t -- not 150,000 t of residue.
    """
    burning = run_advanced_dst_city(
        _request(
            [_spec(accepted=GATE_WITH_NEIGHBOUR, combusts=True)],
            (1.0,),
            diversion=0.25,
        ),
        with_site_emissions=True,
    )
    depositing = run_advanced_dst_city(
        _request([_spec(accepted=GATE_WITH_NEIGHBOUR)], (1.0,), diversion=0.25),
        with_site_emissions=True,
    )

    burnt = _total(burning["site_emissions"]["baseline"][0].other)
    kept = _total(depositing["site_emissions"]["baseline"][0].other)

    assert burnt.sum() > 0, "an incinerator's residue still decays"
    # The unburnable reject is 10%, the same rate the combustion pathway uses.
    assert burnt.sum() == pytest.approx(kept.sum() * 0.1, rel=1e-9)


def test_the_limits_endpoint_ignores_the_new_field():
    """Bounds are about what the city generates, which a gate total says
    nothing about."""
    from SWEET_python.advanced_dst_city import run_advanced_dst_city_limits

    plain = run_advanced_dst_city_limits(_request([_spec()], (1.0,)))
    with_other = run_advanced_dst_city_limits(
        _request([_spec(accepted=GATE_WITH_NEIGHBOUR)], (1.0,))
    )

    assert plain.keys() == with_other.keys()
    for variant in plain:
        assert str(plain[variant]) == str(with_other[variant])


# --------------------------------------------------------------------------- #
# The mix the surplus is given, where the city has none to lend it
# --------------------------------------------------------------------------- #

def test_outside_waste_still_has_a_composition_when_the_city_generates_none():
    """A regional site whose city has not started collecting yet.

    The surplus is split by the city's post-diversion residual mix, and in a
    year the city buries nothing that mix is 0/0. The fallback has to be the
    composition as *shares*: the composition as masses is all-zero in exactly
    the years it would be needed, which left every share at zero and deposited
    none of the site's inflow. Thirty years of a 200,000 t/yr site vanished, and
    nothing said so.
    """
    late = {y: (0.0 if y < 2030 else GENERATED) for y in YEARS}
    result = run_advanced_dst_city(
        _request(
            [_spec(accepted={y: 200_000.0 for y in YEARS})],
            (1.0,),
            generated=late,
        ),
        with_site_emissions=True,
    )
    site = result["site_emissions"]["baseline"][0]
    other = _total(site.other)
    before = np.array([other[i] for i, y in enumerate(YEARS) if y < 2030])

    assert before.sum() > 0, "the site's own inflow produced no methane at all"
    # It is the whole of the site in those years, the city having sent nothing.
    city = _total(site.city)
    city_before = np.array([city[i] for i, y in enumerate(YEARS) if y < 2030])
    assert np.allclose(city_before, 0.0)
    assert np.allclose(
        before,
        np.array([_total(site.total)[i] for i, y in enumerate(YEARS) if y < 2030]),
    )


# --------------------------------------------------------------------------- #
# The scenario half is spliced to baseline before the implement year
# --------------------------------------------------------------------------- #

def test_a_changed_gate_total_takes_effect_only_from_the_implement_year():
    """The surplus follows the same baseline-until-implement rule as everything else.

    The splice that enforces this for a changed *intake* is upstream, in
    ``dst_common.variant_series`` -- it builds the scenario series equal to
    baseline before ``implement_year`` already. This pins the behaviour end to
    end rather than the line that implements it; the line is pinned by
    ``test_a_site_that_starts_burning_...`` below, which exercises a difference
    ``variant_series`` cannot see.
    """
    doubled = {y: 300_000.0 for y in YEARS}
    result = run_advanced_dst_city(
        _request(
            [_spec(accepted=GATE_WITH_NEIGHBOUR, accepted_scenario=doubled)],
            (1.0,),
        ),
        with_site_emissions=True,
    )
    baseline_other = _total(result["site_emissions"]["baseline"][0].other)
    scenario_other = _total(result["site_emissions"]["scenario"][0].other)

    before = [i for i, y in enumerate(YEARS) if y < IMPLEMENT_YEAR]
    after = [i for i, y in enumerate(YEARS) if y >= IMPLEMENT_YEAR]

    # Identical deposits before the implement year, so identical emissions.
    assert np.allclose(
        baseline_other[before], scenario_other[before], rtol=0, atol=1e-9
    ), "the scenario rewrote history"
    # And strictly more afterwards, since the scenario takes 3x the waste. The
    # gap opens gradually: the extra tonnage has to decay before it shows up.
    assert scenario_other[after[-1]] > baseline_other[after[-1]] * 1.5


def test_a_site_that_starts_burning_still_buries_the_surplus_until_then():
    """The frame-level splice, on a difference the series-level one cannot see.

    ``variant_series`` splices the intake, so a variant-differing gate total is
    already baseline-before-implement by the time this module sees it. Two
    things are not: the open/close window and ``combusts``, both applied to the
    mass frame afterwards. A site that starts incinerating at the implement year
    must still deposit the surplus in full before then -- and without the splice
    it deposits only the 10% reject for the whole run, quietly erasing 90% of
    decades of history that the scenario is not supposed to be able to rewrite.
    """
    result = run_advanced_dst_city(
        _request(
            # 200 kt against an undiverted 100 kt city, so half the gate is
            # surplus and there is something for the burning to bite on.
            [_spec(accepted={y: 200_000.0 for y in YEARS}, combusts=False, combusts_scenario=True)],
            (1.0,),
        ),
        with_site_emissions=True,
    )
    baseline_other = _total(result["site_emissions"]["baseline"][0].other)
    scenario_other = _total(result["site_emissions"]["scenario"][0].other)

    before = [i for i, y in enumerate(YEARS) if y < IMPLEMENT_YEAR]
    after = [i for i, y in enumerate(YEARS) if y >= IMPLEMENT_YEAR]

    assert baseline_other[before].sum() > 0, "nothing was buried to compare"
    assert np.allclose(
        baseline_other[before], scenario_other[before], rtol=0, atol=1e-9
    ), "burning from the implement year reached back and unburied earlier waste"
    # And from the implement year the scenario buries only the reject, so its
    # emissions fall away from baseline's.
    assert scenario_other[after[-1]] < baseline_other[after[-1]]
