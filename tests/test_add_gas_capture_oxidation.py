"""Regression tests for the oxidation the City DST's "add gas capture" lever sets
(``add_gas`` in ``City.implement_dst_changes_simple_v1_5``, generic branch).

Bug (fixed): after the implementation year, ``add_gas`` gives the landfill
without capture 60% gas capture and an oxidation factor of 0.22, and converts the
dumpsite to a controlled dumpsite with 30% capture and 0.1. It sets
``skip_ox = True`` so ``Landfill.estimate_emissions`` keeps those oxidation
factors. The emissions loop at the end of the method overwrote that with
``skip_ox = scenario_parameters.sites_method``, which is falsy on this branch, so
``estimate_emissions`` reset each oxidation factor to the default for a site
without gas capture: 0.1 for the landfill, 0 for the dumpsite. Gas capture still
applied; the oxidation never did. A city that landfills everything came out 60%
below the no-change scenario after the implementation year instead of 65.3%.

0.22 and 0.1 are the values ``Landfill.estimate_emissions`` itself gives a
landfill and a controlled dumpsite with gas collection (its ``ox_cap`` table, and
``defaults_2019.oxidation_factor["with_lfg"]``).

These tests build a city without gas capture from country defaults
(``dst_baseline_blank``, no database), give it a split, and run the real
``implement_dst_changes_simple_v1_5``.
"""

import copy

import pandas as pd
import pytest

from SWEET_python.city_params import City, DiversionFractions
from SWEET_python.class_defs import SplitFractions
from SWEET_python.landfill import Landfill

_COUNTRY, _POP, _PRECIP, _TEMP = "Mexico", 2_000_000, 600.0, 17.0
_IMPLEMENT_YEAR, _SCENARIO = 2026, 1
_BEFORE = slice(None, _IMPLEMENT_YEAR)
_AFTER = slice(_IMPLEMENT_YEAR + 1, None)

# Each site add_gas converts: (gas capture it adds, oxidation factor before, after).
_LANDFILL_WITHOUT_CAPTURE = (0.6, 0.1, 0.22)
_DUMPSITE = (0.3, 0.0, 0.1)


def _city_with_split(with_capture, without_capture, dumpsite) -> City:
    """A city whose baseline sends its disposal to the generic split given, as
    WasteMAP's ``load_city`` builds one from the cities table."""
    city = City("add_gas_oxidation_regression")
    city.dst_baseline_blank(_COUNTRY, _POP, _PRECIP, _TEMP)
    baseline = city.baseline_parameters
    baseline.split_fractions = SplitFractions(
        landfill_w_capture=with_capture,
        landfill_wo_capture=without_capture,
        dumpsite=dumpsite,
    )
    city._calculate_divs()
    baseline.repopulate_attr_dicts()
    for landfill in baseline.landfills:
        landfill.estimate_emissions()
    city.estimate_diversion_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=0)
    return city


@pytest.fixture(scope="module")
def city_without_capture() -> City:
    """Half its disposal to a landfill without gas capture, half to a dumpsite: a
    city the UI offers "add gas capture" to."""
    return _city_with_split(0.0, 0.5, 0.5)


def _implement(
    city: City,
    add_gas: int,
    new_gas_pct: float = 0.0,
    div_fractions: DiversionFractions = None,
    sites_method=None,
):
    """Run the city DST on a copy of ``city``, keeping its diversions unless
    others are given."""
    city = copy.deepcopy(city)
    if div_fractions is None:
        div = city.baseline_parameters.div_fractions.iloc[0]
        div_fractions = DiversionFractions(
            compost=float(div["compost"]),
            anaerobic=float(div["anaerobic"]),
            combustion=float(div["combustion"]),
            recycling=float(div["recycling"]),
        )
    if sites_method is not None:
        city.baseline_parameters.sites_method = sites_method
    city.implement_dst_changes_simple_v1_5(
        div_fractions,
        add_gas,
        0,
        new_gas_pct,
        0.0,
        _IMPLEMENT_YEAR,
        _SCENARIO,
    )
    return city.scenario_parameters[_SCENARIO - 1]


@pytest.mark.parametrize("new_gas_pct", [0.0, 0.2])
@pytest.mark.parametrize(
    "index, site",
    [(1, _LANDFILL_WITHOUT_CAPTURE), (2, _DUMPSITE)],
    ids=["landfill", "dumpsite"],
)
def test_add_gas_keeps_the_oxidation_it_sets(city_without_capture, index, site, new_gas_pct):
    """Before the fix the landfill came out at 0.1 and the dumpsite at 0 in every
    year. The UI also lets a city add a new landfill with gas capture alongside."""
    _, before, after = site
    oxidation = _implement(city_without_capture, 1, new_gas_pct).landfills[index].oxidation_factor

    assert isinstance(oxidation, pd.Series), f"reset to {oxidation!r}"
    assert (oxidation.loc[_BEFORE] == before).all()
    assert (oxidation.loc[_AFTER] == after).all()


@pytest.mark.parametrize(
    "index, site",
    [(1, _LANDFILL_WITHOUT_CAPTURE), (2, _DUMPSITE)],
    ids=["landfill", "dumpsite"],
)
def test_add_gas_emissions_include_the_oxidation(city_without_capture, index, site):
    """After the implementation year each converted site emits what it did, less
    the gas captured, with the new oxidation in place of the old: 0.4 x 0.78 / 0.9
    for the landfill and 0.7 x 0.9 for the dumpsite. Before the fix the factors
    were capture alone, 0.4 and 0.7. Up to the implementation year nothing moves."""
    capture, before, after = site
    no_change = _implement(city_without_capture, 0).landfills[index].emissions["total"]
    added = _implement(city_without_capture, 1).landfills[index].emissions["total"]

    pd.testing.assert_series_equal(added.loc[_BEFORE], no_change.loc[_BEFORE])
    factor = (1 - capture) * (1 - after) / (1 - before)
    pd.testing.assert_series_equal(added.loc[_AFTER], no_change.loc[_AFTER] * factor, rtol=1e-12)


def test_a_city_that_landfills_everything_falls_65_percent():
    """No diversion and all disposal to a landfill without capture: add_gas cuts
    the city's emissions after the implementation year by 65.3%, not 60%."""
    city = _city_with_split(0.0, 1.0, 0.0)
    no_diversion = DiversionFractions(compost=0.0, anaerobic=0.0, combustion=0.0, recycling=0.0)
    no_change = _implement(city, 0, div_fractions=no_diversion).total_emissions["total"]
    added = _implement(city, 1, div_fractions=no_diversion).total_emissions["total"]

    assert 1 - added.loc[2040] / no_change.loc[2040] == pytest.approx(0.6533, abs=1e-4)


def test_without_add_gas_each_site_keeps_its_default_oxidation(city_without_capture):
    """The no-change scenario is untouched: every site takes the default for its
    type and gas capture, and the landfills emit what the baseline does."""
    scenario = _implement(city_without_capture, 0)

    assert [lf.oxidation_factor for lf in scenario.landfills] == [0.22, 0.1, 0]
    for scenario_lf, baseline_lf in zip(
        scenario.landfills, city_without_capture.baseline_parameters.landfills
    ):
        pd.testing.assert_frame_equal(scenario_lf.emissions, baseline_lf.emissions, rtol=1e-12)


@pytest.mark.parametrize(
    "sites_method, add_gas, skip_ox",
    [
        (False, 0, False),
        (False, 1, True),
        # WasteMAP's Brazilian cities-table rows took this branch with these same
        # generic landfills before RMI/WasteMAP#851. Real sites are ``advanced``
        # and keep the oxidation they carry either way.
        (True, 0, True),
        (True, 1, True),
    ],
)
def test_each_branch_passes_its_own_skip_ox(
    city_without_capture, monkeypatch, sites_method, add_gas, skip_ox
):
    passed = []
    estimate_emissions = Landfill.estimate_emissions

    def recording(self, skip_ox=False, **kwargs):
        passed.append(skip_ox)
        return estimate_emissions(self, skip_ox=skip_ox, **kwargs)

    monkeypatch.setattr(Landfill, "estimate_emissions", recording)
    _implement(city_without_capture, add_gas, sites_method=sites_method)

    assert passed == [skip_ox] * 3
