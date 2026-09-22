"""The gas-capture boolean gates the site-type default -- in BOTH trace paths.

`site_only_estimate_trace` and `citysite_estimate_trace` used to carry copies of this
logic and the copies drifted: the citysite copy applied the site-type default with no
check on the flag, while still reading that same flag to pick oxidation. On the
10_05_26 submission that gave 1,994 Brazilian city-linked sites a 0.60/0.45 capture
efficiency they had no recorded system for -- and a Landfill object carrying
`gas_capture=False` next to `gas_capture_efficiency=0.6`.

Same history as the MCF table (see test_mcf.py), same fix: one definition, two callers.
These tests exercise the shared resolver directly, so they pin the rule without needing
a live City run.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python.city_params import (
    GAS_EFF_OPTIONS,
    OX_OPTIONS,
    _resolve_gas_capture,
)


YEARS = range(1970, 2051)
TYPES = ["Sanitary Landfill", "Controlled Dumpsite", "Dumpsite"]


def _canonical(gce=np.nan):
    return pd.Series({"gas_collection_efficiency": gce})


def _rows(pairs):
    """Multi-row site frame: [(reported_emissions_year, gas_collection_efficiency), ...]"""
    return pd.DataFrame(
        pairs, columns=["reported_emissions_year", "gas_collection_efficiency"]
    )


# --------------------------------------------------------------------------- #
# The rule itself
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("site_type", TYPES)
@pytest.mark.parametrize("flag", [False, None, np.nan, pd.NA, 0, "No"])
def test_no_recorded_system_means_zero_capture(site_type, flag):
    """A flag that is not a yes gets zero capture, never the site-type default.

    This is the assertion the citysite copy failed. `pd.NA` is in the list because
    the un-hardened copy raised "boolean value of NA is ambiguous" on it.
    """
    presence, ox, gce = _resolve_gas_capture(
        flag, site_type, _canonical(), _rows([]), YEARS
    )
    assert presence is False
    assert (gce == 0).all()
    assert ox == OX_OPTIONS["ox_nocap"][site_type]


@pytest.mark.parametrize("site_type", TYPES)
@pytest.mark.parametrize("flag", [True, "Yes", 1])
def test_recorded_system_gets_the_site_type_default(site_type, flag):
    presence, ox, gce = _resolve_gas_capture(
        flag, site_type, _canonical(), _rows([]), YEARS
    )
    assert presence is True
    assert (gce == GAS_EFF_OPTIONS[site_type]).all()
    assert ox == OX_OPTIONS["ox_cap"][site_type]


def test_a_flagged_dumpsite_still_captures_nothing():
    # Dumpsite's default IS zero, so "flag true, capture zero" is correct here and
    # must not be read as the gate failing.
    presence, _, gce = _resolve_gas_capture(True, "Dumpsite", _canonical(), _rows([]), YEARS)
    assert presence is True
    assert (gce == 0).all()


# --------------------------------------------------------------------------- #
# Measured values override the gate
# --------------------------------------------------------------------------- #

def test_measured_efficiency_overrides_a_false_flag():
    """GHGRP measures gas being collected at 87 sites LMOP flags as having no system.
    The measurement wins -- see build_gas_capture_rates.classify_assets in the TRACE
    repo, which documents the same precedence.
    """
    rows = _rows([(2020, 0.35), (2021, 0.45)])
    presence, _, gce = _resolve_gas_capture(False, "Sanitary Landfill", _canonical(), rows, YEARS)
    assert presence is False          # the flag is still reported honestly
    assert gce.loc[2020] == 0.35      # but the measurement drives the model
    assert gce.loc[2021] == 0.45
    assert gce.loc[1999] == pytest.approx(0.40)   # off-year baseline = site mean


def test_scalar_measured_value_on_a_single_row_site():
    presence, _, gce = _resolve_gas_capture(
        False, "Sanitary Landfill", _canonical(gce=0.72), None, YEARS
    )
    assert presence is False
    assert (gce == 0.72).all()


# --------------------------------------------------------------------------- #
# The NaN-poisoning trap
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("flag,expected", [(True, 0.6), (False, 0.0)])
def test_capture_without_a_reported_year_falls_back_to_the_gate(flag, expected):
    """A site can carry capture data and no reported emissions year at all -- the
    input query selects the two independently, and a null CH4_reported is exactly what
    routes a site to 'to be modeled'. Dropping every row once left .mean() as NaN and
    poisoned the whole series. It must fall back to the GATED default, not to NaN and
    not to the default unconditionally.
    """
    rows = _rows([(np.nan, 0.5)])
    _, _, gce = _resolve_gas_capture(flag, "Sanitary Landfill", _canonical(), rows, YEARS)
    assert gce.notna().all()
    assert (gce == expected).all()


def test_a_year_without_a_capture_value_is_dropped_not_averaged_as_nan():
    rows = _rows([(2020, 0.5), (2021, np.nan)])
    _, _, gce = _resolve_gas_capture(True, "Sanitary Landfill", _canonical(), rows, YEARS)
    assert gce.notna().all()
    assert gce.loc[2020] == 0.5
    assert gce.loc[2021] == 0.5   # baseline = mean of the one usable row


# --------------------------------------------------------------------------- #
# Both callers go through the resolver
# --------------------------------------------------------------------------- #

def test_neither_trace_path_keeps_its_own_copy_of_the_tables():
    """Guards the refactor: if someone re-inlines the tables into either `_trace`
    method, the drift can start again. The two surviving `gas_eff_options` locals
    belong to `sinar_city_and_site` and `site_only_estimate`, which are not the
    Climate TRACE pipeline path.
    """
    import inspect

    from SWEET_python.city_params import City

    for method in (City.site_only_estimate_trace, City.citysite_estimate_trace):
        src = inspect.getsource(method)
        assert "gas_eff_options" not in src, f"{method.__name__} re-inlined the table"
        assert "ox_options" not in src, f"{method.__name__} re-inlined the table"
        assert "_resolve_gas_capture" in src, f"{method.__name__} bypasses the resolver"


def test_oxidation_and_capture_never_disagree_about_presence():
    """The bug's signature: oxidation said no-capture while capture said 0.6. No input
    may produce that combination again.
    """
    for site_type in TYPES:
        for flag in [True, False, None, np.nan, pd.NA, "Yes", "No", 0, 1]:
            presence, ox, gce = _resolve_gas_capture(
                flag, site_type, _canonical(), _rows([]), YEARS
            )
            expected_ox = OX_OPTIONS["ox_cap" if presence else "ox_nocap"][site_type]
            assert ox == expected_ox, (site_type, flag)
            if not presence:
                assert (gce == 0).all(), (site_type, flag)
