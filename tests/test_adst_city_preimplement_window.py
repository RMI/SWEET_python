"""The scenario matches baseline before ``implement_year``, whatever its window.

The advanced city DST runs two variants and splices them: the scenario's own
inputs take effect at ``implement_year``, and every year before that is meant to
be baseline's, exactly. That convention is what makes the two halves comparable
— a caller diffs them to read the effect of its interventions, so any
pre-implement divergence is attributed to a change the user never made.

A landfill's open/close window is per variant, because a ``siteClosure``
intervention moves it. Applying the *scenario's* window to rows that had just
been spliced from baseline broke the convention in the one direction the window
can act on them: a site stated as opening later in the scenario buried nothing
in the years between the two opening years, though those years precede
``implement_year``.
"""

import pytest

from SWEET_python import dst_common as common
from SWEET_python.advanced_dst_city import (
    AdvancedDSTCityRequest,
    run_advanced_dst_city,
)

YEARS = range(1980, 2051)
IMPLEMENT_YEAR = 2025
TOTAL_TONS = 100_000.0
FRACTIONS = [0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0]


def _landfill(baseline_open, scenario_open):
    return dict(
        landfill_type={"baseline": 2, "scenario": 2},
        landfill_open_close={
            "baseline": [baseline_open, common.MODEL_YEAR_MAX],
            "scenario": [scenario_open, common.MODEL_YEAR_MAX],
        },
        gas_capture_efficiency={
            "baseline": {y: 0.0 for y in YEARS},
            "scenario": {y: 0.0 for y in YEARS},
        },
    )


def _request(baseline_open=1980, scenario_open=1980):
    return AdvancedDSTCityRequest(
        city_name="Windowville",
        country="USA",
        precipitation=1200.0,
        temperature=20.0,
        implement_year=IMPLEMENT_YEAR,
        waste_mass={
            "baseline": {y: TOTAL_TONS for y in YEARS},
            "scenario": {y: TOTAL_TONS for y in YEARS},
        },
        waste_fractions={
            "baseline": {y: FRACTIONS for y in YEARS},
            "scenario": {y: FRACTIONS for y in YEARS},
        },
        landfills=[_landfill(baseline_open, scenario_open)],
        landfill_split_timeline={
            "baseline": {y: [1.0] for y in YEARS},
            "scenario": {y: [1.0] for y in YEARS},
        },
    )


def test_a_later_scenario_opening_does_not_empty_the_years_before_it():
    """Baseline opens 1980, the scenario says 2010, and both precede 2025.

    Those thirty years are pre-implement, so the scenario has to emit exactly
    what baseline emits in them. Two things used to break that: the splice put
    baseline's mass in and the scenario's window took it straight back out, and
    the model's year range started at the scenario's own open year so those
    years were not evaluated at all.
    """
    result = run_advanced_dst_city(_request(baseline_open=1980, scenario_open=2010))
    baseline, scenario = result["baseline"], result["scenario"]

    for year in (2010, 2015, IMPLEMENT_YEAR - 1):
        assert float(scenario.loc[year, "total"]) == pytest.approx(
            float(baseline.loc[year, "total"])
        ), f"scenario diverges from baseline in {year}, before the implement year"
    # And the years the scenario used to drop are modeled, with the stock the
    # thirty years of baseline deposition put in place.
    assert 1985 in scenario.index
    assert float(baseline.loc[2010, "total"]) > 0


def test_the_two_variants_cover_the_same_years():
    """A caller diffs the halves, so they have to be the same shape.

    The truncated scenario window shortened its emissions frame too: 1980-2050
    against 2010-2050, which does not subtract.
    """
    result = run_advanced_dst_city(_request(baseline_open=1980, scenario_open=2010))

    assert list(result["baseline"].index) == list(result["scenario"].index)


def test_emissions_match_before_the_implement_year():
    result = run_advanced_dst_city(_request(baseline_open=1980, scenario_open=2010))
    baseline, scenario = result["baseline"], result["scenario"]

    pre = baseline.index[baseline.index < IMPLEMENT_YEAR]
    assert len(pre) > 0
    assert baseline.loc[pre].equals(scenario.loc[pre])


def test_the_scenario_window_still_applies_from_the_implement_year():
    """The fix is about pre-implement rows only — a real closure still bites."""
    closing = _request()
    spec = closing.landfills[0]
    spec.landfill_open_close["scenario"] = [1980, 2030]

    result = run_advanced_dst_city(closing)
    baseline, scenario = result["baseline"], result["scenario"]

    # Identical while both are open and taking waste.
    assert float(scenario.loc[2029, "total"]) == pytest.approx(
        float(baseline.loc[2029, "total"])
    )
    # Then the scenario stops accepting, so its stock decays away while
    # baseline's keeps growing.
    assert float(scenario.loc[2040, "total"]) < float(baseline.loc[2040, "total"])


def test_an_unchanged_window_is_unaffected():
    """No output change when the two windows agree, which is the common case."""
    result = run_advanced_dst_city(_request(baseline_open=1980, scenario_open=1980))

    assert result["baseline"].equals(result["scenario"])
