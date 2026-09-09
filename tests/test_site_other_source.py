"""A named site can receive waste the city did not send.

Until now a city's disposal sites shared out exactly the city's landfilled
residual: ``landfill_split_timeline`` is *fractions that must sum to 1.0*
(``_validate_shares``), so a site receiving mass from anywhere else was
structurally unsayable. Real sites take waste from neighbouring municipalities,
from private haulers, from a regional catchment the baseline never describes --
and the site's own operator knows its gate total, not the city's share of it.

So a spec may now carry ``other_source_mass``: absolute tons per year arriving
at that site from outside this baseline. Two consequences, and this module
exists to pin both:

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

The linearity has one precondition and it is load-bearing: every stream at a
site must share one ``k``. ``k`` is a step function of composition, so two
streams at one landfill with different mixes do not superpose. Other-source
waste therefore carries the city's own post-diversion residual composition --
which is also the physically right answer, since a gate observation is
downstream of whatever diversion happened upstream of it.
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


def _spec(*, open_close=(1990, 2051), other=None, combusts=None):
    spec = dict(
        landfill_type={"baseline": 2, "scenario": 2},
        landfill_open_close={"baseline": list(open_close), "scenario": list(open_close)},
        gas_capture_efficiency={
            "baseline": {y: 0.0 for y in YEARS},
            "scenario": {y: 0.0 for y in YEARS},
        },
    )
    if other is not None:
        spec["other_source_mass"] = {"baseline": dict(other), "scenario": dict(other)}
    if combusts is not None:
        spec["combusts"] = {"baseline": combusts, "scenario": combusts}
    return spec


def _request(specs, shares, *, diversion=None):
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
            "baseline": {y: GENERATED for y in YEARS},
            "scenario": {y: GENERATED for y in YEARS},
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


# A neighbour whose tonnage RISES while the city's stays flat, so the city's
# share of the site falls over time. Any attribution built on a same-year ratio
# gets this case badly wrong; see the last test.
RISING_NEIGHBOUR = {y: 3_000.0 * (1.05 ** (y - 1990)) for y in YEARS}
FLAT_NEIGHBOUR = {y: 25_000.0 for y in YEARS}


# --------------------------------------------------------------------------- #
# The city's emissions are the city's waste, wherever it goes
# --------------------------------------------------------------------------- #

def test_other_source_waste_does_not_change_the_city_total():
    """The whole feature, in one assertion. Bit-identical, not merely close."""
    without = run_advanced_dst_city(_request([_spec()], (1.0,), diversion=0.25))
    with_other = run_advanced_dst_city(
        _request([_spec(other=FLAT_NEIGHBOUR)], (1.0,), diversion=0.25)
    )

    for variant in ("baseline", "scenario"):
        assert without[variant].equals(with_other[variant]), (
            f"{variant}: other-source waste moved the city's own emissions"
        )


def test_an_absent_field_and_a_zero_series_are_the_same_request():
    """No silent second code path for the city that never uses this."""
    absent = run_advanced_dst_city(_request([_spec()], (1.0,), diversion=0.25))
    zeros = run_advanced_dst_city(
        _request([_spec(other={y: 0.0 for y in YEARS})], (1.0,), diversion=0.25)
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
            _request([_spec(other=FLAT_NEIGHBOUR), _spec()], (0.5, 0.2))
        )


# --------------------------------------------------------------------------- #
# A site's total is the sum of its streams
# --------------------------------------------------------------------------- #

def test_a_sites_emissions_split_into_the_city_and_the_rest():
    result = run_advanced_dst_city(
        _request([_spec(other=FLAT_NEIGHBOUR)], (1.0,), diversion=0.25),
        with_site_emissions=True,
    )
    site = result["site_emissions"]["baseline"][0]

    city, other, total = _total(site["city"]), _total(site["other"]), _total(site["total"])
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
        _request([_spec(other=FLAT_NEIGHBOUR)], (1.0,), diversion=0.25),
        with_site_emissions=True,
    )

    a = _total(alone["site_emissions"]["baseline"][0]["city"])
    b = _total(shared["site_emissions"]["baseline"][0]["city"])
    scale = np.where(a == 0, 1.0, a)

    assert np.max(np.abs(a - b) / scale) < 1e-12


def test_a_site_with_no_other_source_reports_a_total_equal_to_its_city_share():
    result = run_advanced_dst_city(
        _request([_spec()], (1.0,), diversion=0.25), with_site_emissions=True
    )
    site = result["site_emissions"]["baseline"][0]

    assert np.allclose(_total(site["other"]), 0.0)
    assert np.allclose(_total(site["total"]), _total(site["city"]), rtol=0, atol=1e-12)


def test_the_city_shares_of_every_site_add_up_to_the_city_total():
    """The two outputs reconcile: city emissions are the per-site city shares
    plus whatever the diversion pathways themselves emit."""
    request = _request(
        [_spec(other=FLAT_NEIGHBOUR), _spec(other=RISING_NEIGHBOUR)],
        (0.6, 0.4),
        diversion=0.25,
    )
    result = run_advanced_dst_city(request, with_site_emissions=True)

    per_site = sum(
        _total(site["city"]) for site in result["site_emissions"]["baseline"]
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
        _request([_spec(other=RISING_NEIGHBOUR)], (1.0,), diversion=0.0),
        with_site_emissions=True,
    )
    site = result["site_emissions"]["baseline"][0]
    city, total = _total(site["city"]), _total(site["total"])

    settled = total > 0.01 * total.max()
    emission_share = (city / np.where(total == 0, 1.0, total))[settled]

    city_tons = np.array([GENERATED for _ in YEARS])
    mass_share = (city_tons / (city_tons + np.array([RISING_NEIGHBOUR[y] for y in YEARS])))[settled]

    # The two shares must visibly disagree: emissions lag mass, so a shrinking
    # city holds a larger share of the emissions than of this year's intake.
    assert np.max(np.abs(emission_share - mass_share)) > 0.02, (
        "mass share and emission share agree -- either the test city is too "
        "static to be a regression guard, or attribution has been reduced to a ratio"
    )
    assert np.all(emission_share[-10:] > mass_share[-10:])
