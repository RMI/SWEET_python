"""A site in the city-level advanced DST can state its own k and oxidation rate.

``CityLandfillSpec.k_override`` and ``oxidation_override`` are the site DST's
``ks_overrides`` / ``oxidation_override``, per landfill: the model derives both
rates itself, and a site that states one is modelled at that instead.
``run_advanced_dst_city_defaults`` reports what the model would use at each
site that states nothing, so a caller can show it beside the box a user types
into.

The tests that matter most are the round trips: feeding the reported defaults
back in as overrides must reproduce the un-overridden run bit for bit. That is
what makes the number a user sees the number the model runs, and it keeps the
two derivations -- one in the defaults, one in the run -- from drifting apart.
"""

import pandas as pd
import pytest

from SWEET_python.advanced_dst_city import (
    AdvancedDSTCityRequest,
    run_advanced_dst_city,
    run_advanced_dst_city_defaults,
)
from SWEET_python.city_params import CustomError
from SWEET_python.dst_common import OX_CAP, OX_NOCAP

YEARS = range(2000, 2051)
FRACTIONS = [0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.02, 0.03, 0.0, 0.05]  # sums to 1.0
IMPLEMENT_YEAR = 2025
TOTAL_WASTE = 20_000.0

LANDFILL, CONTROLLED_DUMP, OPEN_DUMP = 0, 1, 2


def _constant(value):
    return {y: value for y in YEARS}


def _landfill(*, site_type=OPEN_DUMP, scenario_type=None, gas=0.0, biocover=None, **overrides):
    spec = dict(
        landfill_type={
            "baseline": site_type,
            "scenario": site_type if scenario_type is None else scenario_type,
        },
        landfill_open_close={"baseline": [2000, 2050], "scenario": [2000, 2050]},
        gas_capture_efficiency={"baseline": _constant(gas), "scenario": _constant(gas)},
    )
    if biocover is not None:
        spec["biocover"] = {"baseline": _constant(biocover), "scenario": _constant(biocover)}
    spec.update(overrides)
    return spec


def _request(landfills, *, prevention=None) -> AdvancedDSTCityRequest:
    """A city with no diversion, splitting its waste evenly across `landfills`."""
    share = 1.0 / len(landfills)
    body = dict(
        city_name="Overrideville",
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
    if prevention is not None:
        body["food_waste_prevention"] = {"baseline": _constant(0.0), "scenario": _constant(prevention)}
    return AdvancedDSTCityRequest(**body)


def _as_variant(defaults, index, field):
    """One site's reported default, shaped as the override field that replaces it."""
    return {
        variant: {int(y): float(v) for y, v in defaults[variant][index][field].items()}
        for variant in ("baseline", "scenario")
    }


def _site_totals(result, half, index):
    return result["site_emissions"][half][index].total["total"]


# --------------------------------------------------------------------------- #
# The defaults are what the model runs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prevention", [None, 0.9])
def test_defaults_fed_back_as_overrides_change_nothing(prevention):
    """Every site, every variant, bit for bit -- including a type change at
    implement_year, gas capture, and food waste prevention moving k."""
    landfills = [
        _landfill(site_type=LANDFILL, gas=0.6),
        _landfill(site_type=OPEN_DUMP, scenario_type=LANDFILL),
        _landfill(site_type=CONTROLLED_DUMP),
    ]
    plain = _request(landfills, prevention=prevention)
    defaults = run_advanced_dst_city_defaults(plain)

    overridden = _request(
        [
            {
                **spec,
                "k_override": _as_variant(defaults, index, "k"),
                "oxidation_override": _as_variant(defaults, index, "oxidation"),
            }
            for index, spec in enumerate(landfills)
        ],
        prevention=prevention,
    )

    expected = run_advanced_dst_city(plain, with_site_emissions=True)
    actual = run_advanced_dst_city(overridden, with_site_emissions=True)
    for half in ("baseline", "scenario"):
        pd.testing.assert_series_equal(actual[half]["total"], expected[half]["total"])
        for index in range(len(landfills)):
            pd.testing.assert_series_equal(
                _site_totals(actual, half, index), _site_totals(expected, half, index)
            )


def test_defaults_report_the_type_and_capture_oxidation_before_biocover():
    defaults = run_advanced_dst_city_defaults(
        _request(
            [
                _landfill(site_type=LANDFILL),
                _landfill(site_type=LANDFILL, gas=0.5),
                _landfill(site_type=CONTROLLED_DUMP, biocover=0.5),
                _landfill(site_type=CONTROLLED_DUMP, scenario_type=OPEN_DUMP),
            ]
        )
    )
    first = lambda variant, index: defaults[variant][index]["oxidation"].iloc[0]
    last = lambda variant, index: defaults[variant][index]["oxidation"].iloc[-1]

    assert first("baseline", 0) == OX_NOCAP["landfill"]
    assert first("baseline", 1) == OX_CAP["landfill"]
    # The cover is its own input, so it does not show up in the default it sits on.
    assert first("baseline", 2) == OX_NOCAP["controlled_dumpsite"]
    # The scenario's type takes over from implement_year, and only then.
    assert first("scenario", 3) == OX_NOCAP["controlled_dumpsite"]
    assert last("scenario", 3) == OX_NOCAP["dumpsite"]
    assert last("baseline", 3) == OX_NOCAP["controlled_dumpsite"]


def test_defaults_give_every_site_the_city_k_and_ignore_overrides():
    plain = [_landfill(site_type=LANDFILL), _landfill(site_type=OPEN_DUMP)]
    stated = [
        _landfill(site_type=LANDFILL, k_override={"baseline": _constant(0.9)}),
        _landfill(site_type=OPEN_DUMP, oxidation_override={"baseline": _constant(0.7)}),
    ]
    from_plain = run_advanced_dst_city_defaults(_request(plain))
    from_stated = run_advanced_dst_city_defaults(_request(stated))

    for variant in ("baseline", "scenario"):
        pd.testing.assert_series_equal(
            from_plain[variant][0]["k"], from_plain[variant][1]["k"]
        )
        for index in range(2):
            for field in ("k", "oxidation"):
                pd.testing.assert_series_equal(
                    from_stated[variant][index][field], from_plain[variant][index][field]
                )
    assert 0.0 < from_plain["baseline"][0]["k"].iloc[0] < 1.0


def test_prevention_moves_the_default_k():
    """k is derived from what the city is left with, so taking food out of the
    scenario changes the scenario's k from implement_year -- and not before.

    The Singapore k is a banded lookup on composition, so it takes most of the
    food to cross a band here; 40% leaves k exactly where it was."""
    defaults = run_advanced_dst_city_defaults(
        _request([_landfill()], prevention=0.9)
    )
    baseline_k = defaults["baseline"][0]["k"]
    scenario_k = defaults["scenario"][0]["k"]
    assert scenario_k.loc[IMPLEMENT_YEAR - 1] == baseline_k.loc[IMPLEMENT_YEAR - 1]
    assert scenario_k.loc[IMPLEMENT_YEAR] != baseline_k.loc[IMPLEMENT_YEAR]


# --------------------------------------------------------------------------- #
# An override changes its own site, and only that
# --------------------------------------------------------------------------- #
def test_k_override_moves_only_its_own_site():
    plain = _request([_landfill(), _landfill()])
    defaults = run_advanced_dst_city_defaults(plain)
    derived_k = float(defaults["baseline"][0]["k"].iloc[0])
    faster = _request(
        [_landfill(k_override={"baseline": _constant(derived_k * 3)}), _landfill()]
    )

    before = run_advanced_dst_city(plain, with_site_emissions=True)
    after = run_advanced_dst_city(faster, with_site_emissions=True)
    for half in ("baseline", "scenario"):
        pd.testing.assert_series_equal(
            _site_totals(after, half, 1), _site_totals(before, half, 1)
        )
        # Faster decay releases the same carbon sooner: more methane in the
        # site's first years than at the slower derived rate.
        early = slice(2001, 2005)
        assert (
            _site_totals(after, half, 0).loc[early] > _site_totals(before, half, 0).loc[early]
        ).all()


def test_oxidation_override_scales_an_uncapped_site_exactly():
    """An open dump captures nothing and oxidises nothing, so stating 30% leaves
    exactly 70% of its methane -- the engine is linear in (1 - oxidation)."""
    before = run_advanced_dst_city(_request([_landfill(), _landfill()]), with_site_emissions=True)
    after = run_advanced_dst_city(
        _request([_landfill(oxidation_override={"baseline": _constant(0.3)}), _landfill()]),
        with_site_emissions=True,
    )
    for half in ("baseline", "scenario"):
        pd.testing.assert_series_equal(
            _site_totals(after, half, 0), _site_totals(before, half, 0) * 0.7
        )
        pd.testing.assert_series_equal(
            _site_totals(after, half, 1), _site_totals(before, half, 1)
        )


def test_scenario_half_of_an_override_takes_effect_at_implement_year():
    result = run_advanced_dst_city(
        _request(
            [
                _landfill(
                    oxidation_override={"baseline": _constant(0.0), "scenario": _constant(0.5)}
                )
            ]
        )
    )
    baseline, scenario = result["baseline"]["total"], result["scenario"]["total"]
    pd.testing.assert_series_equal(
        scenario.loc[: IMPLEMENT_YEAR - 1], baseline.loc[: IMPLEMENT_YEAR - 1]
    )
    pd.testing.assert_series_equal(
        scenario.loc[IMPLEMENT_YEAR:], baseline.loc[IMPLEMENT_YEAR:] * 0.5
    )


def test_biocover_still_floors_a_stated_oxidation_rate():
    """A stated rate replaces the type lookup, not the cover: a site that adds a
    50% biocover oxidises at 50% even where it states 20%."""
    with_cover = run_advanced_dst_city(
        _request([_landfill(biocover=0.5, oxidation_override={"baseline": _constant(0.2)})])
    )
    cover_alone = run_advanced_dst_city(_request([_landfill(biocover=0.5)]))
    pd.testing.assert_series_equal(with_cover["baseline"]["total"], cover_alone["baseline"]["total"])

    above_cover = run_advanced_dst_city(
        _request([_landfill(biocover=0.5, oxidation_override={"baseline": _constant(0.8)})])
    )
    pd.testing.assert_series_equal(
        above_cover["baseline"]["total"], cover_alone["baseline"]["total"] * (0.2 / 0.5)
    )


# --------------------------------------------------------------------------- #
# Out-of-range values are refused, not clipped
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field, value, message",
    [
        ("k_override", -0.1, "k_override for 2010 is -0.1; it must be 0 or more"),
        ("oxidation_override", 1.2, "oxidation_override for 2010 is 1.2; it must be between 0 and 1"),
        ("oxidation_override", -0.5, "oxidation_override for 2010 is -0.5; it must be between 0 and 1"),
    ],
)
def test_out_of_range_override_is_refused(field, value, message):
    series = _constant(0.1)
    series[2010] = value
    with pytest.raises(CustomError) as raised:
        run_advanced_dst_city(_request([_landfill(**{field: {"baseline": series}})]))
    assert raised.value.code == "invalid_parameters"
    assert message in raised.value.message


def test_an_empty_override_names_its_field():
    with pytest.raises(CustomError) as raised:
        run_advanced_dst_city(_request([_landfill(k_override={"baseline": {}})]))
    assert raised.value.message.startswith("k_override:")
