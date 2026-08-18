"""Regression tests for cover-oxidation being applied by EMISSION year.

WasteMAP issue #719 ("Bio cover in site DST not working properly"): raising the
oxidation factor from an implementation year on (e.g. installing biocover) must
reduce methane *emitted* from that year on, regardless of when the waste was
deposited. The annual model previously applied oxidation along the deposit-year
axis, so for a landfill that had already closed there was no post-implementation
deposited waste for the higher oxidation to act on and the change had no effect.
"""
import pandas as pd
import pytest

from SWEET_python.model_v2 import SWEET

COMPONENTS = ["food", "green", "wood", "paper_cardboard", "textiles"]


def _run(open_date, close_date, oxidation_factor):
    """Run estimate_emissions2 for a single landfill with a given oxidation series."""
    years = pd.Index(range(open_date, 2051))
    if not isinstance(oxidation_factor, pd.Series):
        oxidation_factor = pd.Series(oxidation_factor, index=years)
    # Waste is only deposited while the landfill is open.
    waste = pd.DataFrame(0.0, index=years, columns=COMPONENTS)
    waste.loc[open_date:close_date, :] = 1000.0
    landfill_attrs = {
        "open_date": open_date,
        "close_date": close_date,
        "ks": {c: pd.Series(0.2, index=years) for c in COMPONENTS},
        "waste_mass_df": waste,
        "mcf": pd.Series(1.0, index=years),
        "gas_capture_efficiency": pd.Series(0.0, index=years),
        "oxidation_factor": oxidation_factor,
        "flaring": pd.Series(0.98, index=years),
    }
    model = SWEET(
        city_instance_attrs={"components": COMPONENTS},
        city_params_dict={
            "year_of_data_pop": 2025,
            "growth_rate_historic": 1.0,
            "growth_rate_future": 1.0,
        },
        landfill_instance_attrs=landfill_attrs,
    )
    _, emissions, _, _ = model.estimate_emissions2()
    return emissions


def _step_oxidation(open_date, low, high, implement_year):
    years = pd.Index(range(open_date, 2051))
    series = pd.Series(low, index=years)
    series.loc[implement_year:] = high
    return series


def test_biocover_reduces_emissions_from_a_closed_landfill():
    """The #719 case: landfill closed in 2011, biocover raises oxidation to 0.9
    starting 2025. Emissions after 2025 must fall even though no waste is
    deposited after 2011."""
    open_date, close_date, implement = 1990, 2011, 2025
    base = _run(open_date, close_date, 0.1)
    biocover = _run(
        open_date,
        close_date,
        _step_oxidation(open_date, 0.1, 0.9, implement),
    )

    # Before the implementation year the two runs are identical.
    assert biocover.loc[2020, "total"] == pytest.approx(base.loc[2020, "total"])

    # From the implementation year on, emitted methane is oxidised at 0.9 instead
    # of 0.1: emission = ch4 * (1 - ox). No gas capture, so the ratio is exact.
    assert base.loc[2030, "total"] > 0
    assert biocover.loc[2030, "total"] == pytest.approx(
        base.loc[2030, "total"] * (1 - 0.9) / (1 - 0.1)
    )
    assert biocover.loc[2030, "total"] < base.loc[2030, "total"]


def test_oxidation_is_applied_by_emission_year_not_deposit_year():
    """A step change in oxidation at an implementation year affects every
    emission year >= that year, independent of deposit year."""
    ox = _step_oxidation(1990, 0.1, 0.5, 2025)
    emissions = _run(1990, 2050, ox)
    flat_low = _run(1990, 2050, 0.1)
    flat_high = _run(1990, 2050, 0.5)

    # Year before the step matches the low-oxidation run; year at/after the step
    # matches the high-oxidation run -- i.e. keyed on emission year.
    assert emissions.loc[2024, "total"] == pytest.approx(flat_low.loc[2024, "total"])
    assert emissions.loc[2025, "total"] == pytest.approx(flat_high.loc[2025, "total"])
    assert emissions.loc[2030, "total"] == pytest.approx(flat_high.loc[2030, "total"])


def test_constant_oxidation_is_unchanged():
    """Guard: when oxidation does not vary over years the result is unaffected by
    the emission-year fix (a constant series broadcasts identically either way)."""
    emissions = _run(1990, 2050, 0.1)
    assert emissions.loc[1991, "total"] == pytest.approx(107657.55599791845)
