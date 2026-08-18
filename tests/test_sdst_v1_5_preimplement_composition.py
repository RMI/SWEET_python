"""Regression test: sdst scenario == baseline before the implementation year.

``City.sdst_v1_5`` models the baseline and the scenario as two independent
landfills. Before the scenario's implementation year nothing has been changed, so
the scenario landfill must receive exactly the same waste as the baseline — the
same total mass *and* the same composition — and therefore emit exactly the same
methane in every year before ``implement_year``.

For real (TRACE-reconciled) sites the engine already enforced this by splicing the
scenario waste back to the baseline pre-implement. The blank/custom-site branch
(``baseline_data=None`` — a "Custom Location" in the /sdst UI) omitted that splice:
it applied the *scenario* waste composition to every year, so a scenario that
changed the waste composition wrongly back-dated the new composition onto deposits
from before the implementation year. This test drives that exact path (a custom
site with a different scenario composition) and asserts the pre-implement
scenario emissions equal the baseline. It fails on the pre-fix engine.
"""

import pandas as pd
import pytest

from SWEET_python.city_params import City
from SWEET_python.class_defs import Variant

COMPONENT_ORDER = [
    "food",
    "green",
    "wood",
    "paper_cardboard",
    "textiles",
    "plastic",
    "metal",
    "glass",
    "rubber",
    "other",
]

OPEN_YEAR = 2000
CLOSE_YEAR = 2050
IMPLEMENT_YEAR = 2025
MODEL_YEAR_MAX = 2050

# Baseline and scenario compositions differ; both sum to 1.0.
BASELINE_FRACTIONS = [0.5, 0.1, 0.05, 0.1, 0.05, 0.1, 0.02, 0.03, 0.0, 0.05]
SCENARIO_FRACTIONS = [0.3, 0.1, 0.05, 0.2, 0.05, 0.15, 0.05, 0.05, 0.0, 0.05]


def _run_sdst(baseline_fractions, scenario_fractions):
    """Drive City.sdst_v1_5 for a blank/custom single site (baseline_data=None)."""
    years = pd.Index(range(OPEN_YEAR, MODEL_YEAR_MAX + 1))

    def _expand(values):
        return pd.DataFrame(
            [list(values)] * len(years), index=years, columns=COMPONENT_ORDER, dtype=float
        )

    waste_mass_year = Variant[int](baseline=2025, scenario=2025)
    city = City("preimplement_composition_test")
    city.cityparams_obj_for_blank_site(
        country="BRA",
        population=None,
        precipitation=500.0,
        temperature=10.0,
        waste_fractions=Variant(baseline=list(baseline_fractions), scenario=list(scenario_fractions)),
        waste_mass_year=waste_mass_year,
        growth_rate_override=0.0,
    )

    city.sdst_v1_5(
        precipitation=500.0,
        new_waste_fractions={"baseline": _expand(baseline_fractions), "scenario": _expand(scenario_fractions)},
        new_landfill_types=Variant(baseline=[2], scenario=[0]),
        new_gas_efficiency=Variant(baseline=[0.0], scenario=[0.6]),
        new_landfill_open_close_dates=Variant(
            baseline=[(OPEN_YEAR, CLOSE_YEAR)], scenario=[(OPEN_YEAR, CLOSE_YEAR)]
        ),
        scenario=1,
        landfill_split_timeline=Variant(
            baseline={year: [1.0] for year in years}, scenario={year: [1.0] for year in years}
        ),
        new_landfill_latlons=None,
        new_landfill_areas=None,
        new_covertypes=None,
        new_coverthicknesses=None,
        waste_burning=Variant(baseline=0.0, scenario=0.0),
        new_landfill_flaring=Variant(baseline=[0.98], scenario=[0.98]),
        fancy_ox=None,
        new_waste_mass=Variant(baseline=10000.0, scenario=10000.0),
        waste_mass_year=waste_mass_year,
        depths=Variant(baseline=[3.0], scenario=[3.0]),
        ks_overrides=Variant(baseline=0.2, scenario=0.2),
        biocover={"baseline": 0.0, "scenario": 0.0},
        oxidation_override=None,
        baseline_data=None,
        implement_year=IMPLEMENT_YEAR,
        growth_rate_override=0.0,
        country_growth_defaults=[1.0, 1.0],
    )
    return (
        city.baseline_parameters.total_emissions["total"],
        city.scenario_parameters[0].total_emissions["total"],
    )


def test_scenario_matches_baseline_before_implement_year():
    baseline, scenario = _run_sdst(BASELINE_FRACTIONS, SCENARIO_FRACTIONS)

    for year in range(OPEN_YEAR, IMPLEMENT_YEAR):
        assert scenario.loc[year] == pytest.approx(baseline.loc[year], abs=1e-9), (
            f"scenario emissions in {year} (< implement year {IMPLEMENT_YEAR}) must "
            f"equal baseline: got scenario={scenario.loc[year]}, baseline={baseline.loc[year]}"
        )


def test_scenario_diverges_from_baseline_after_implement_year():
    # Sanity check that the scenario really is a different scenario (dump -> landfill
    # with gas capture), so the pre-implement equality above is a real constraint,
    # not a degenerate baseline==scenario run.
    baseline, scenario = _run_sdst(BASELINE_FRACTIONS, SCENARIO_FRACTIONS)

    assert scenario.loc[2050] < baseline.loc[2050]


def test_pre_implement_equality_holds_when_only_composition_changes():
    # Even with no landfill-type/gas change at all, a pure composition change must
    # not affect pre-implement emissions.
    years = pd.Index(range(OPEN_YEAR, MODEL_YEAR_MAX + 1))

    def _expand(values):
        return pd.DataFrame(
            [list(values)] * len(years), index=years, columns=COMPONENT_ORDER, dtype=float
        )

    waste_mass_year = Variant[int](baseline=2025, scenario=2025)
    city = City("preimplement_composition_only")
    city.cityparams_obj_for_blank_site(
        country="BRA",
        population=None,
        precipitation=500.0,
        temperature=10.0,
        waste_fractions=Variant(baseline=list(BASELINE_FRACTIONS), scenario=list(SCENARIO_FRACTIONS)),
        waste_mass_year=waste_mass_year,
        growth_rate_override=0.0,
    )
    city.sdst_v1_5(
        precipitation=500.0,
        new_waste_fractions={"baseline": _expand(BASELINE_FRACTIONS), "scenario": _expand(SCENARIO_FRACTIONS)},
        new_landfill_types=Variant(baseline=[0], scenario=[0]),
        new_gas_efficiency=Variant(baseline=[0.0], scenario=[0.0]),
        new_landfill_open_close_dates=Variant(
            baseline=[(OPEN_YEAR, CLOSE_YEAR)], scenario=[(OPEN_YEAR, CLOSE_YEAR)]
        ),
        scenario=1,
        landfill_split_timeline=Variant(
            baseline={year: [1.0] for year in years}, scenario={year: [1.0] for year in years}
        ),
        new_landfill_latlons=None,
        new_landfill_areas=None,
        new_covertypes=None,
        new_coverthicknesses=None,
        waste_burning=Variant(baseline=0.0, scenario=0.0),
        new_landfill_flaring=None,
        fancy_ox=None,
        new_waste_mass=Variant(baseline=10000.0, scenario=10000.0),
        waste_mass_year=waste_mass_year,
        depths=Variant(baseline=[3.0], scenario=[3.0]),
        ks_overrides=Variant(baseline=0.2, scenario=0.2),
        biocover={"baseline": 0.0, "scenario": 0.0},
        oxidation_override=None,
        baseline_data=None,
        implement_year=IMPLEMENT_YEAR,
        growth_rate_override=0.0,
        country_growth_defaults=[1.0, 1.0],
    )
    baseline = city.baseline_parameters.total_emissions["total"]
    scenario = city.scenario_parameters[0].total_emissions["total"]
    for year in range(OPEN_YEAR, IMPLEMENT_YEAR):
        assert scenario.loc[year] == pytest.approx(baseline.loc[year], abs=1e-9)
