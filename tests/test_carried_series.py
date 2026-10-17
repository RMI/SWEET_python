"""Gas collection efficiency and oxidation from rows that carry their own year.

The Climate TRACE pipeline's consolidated input gives a site one row per year and marks
each with `row_year`. For those rows a reported value belongs to its year: one value is
used in every year, and several are carried back and forward (the first back to the
start, each forward to the next report, the latest to the end). Rows without `row_year`
-- the legacy input, and every other caller -- keep the old rule, which
test_gas_capture_gate.py pins.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python.city_params import (
    GAS_EFF_OPTIONS,
    _build_oxidation_series,
    _carried_series,
    _resolve_gas_capture,
)

YEARS = range(1970, 2051)

# Polk County North Central Landfill (GHGRP 1006859), as GHGRP reports it.
POLK = {2010: 0.58, 2011: 0.58, 2012: 0.74, 2013: 0.77, 2014: 0.77, 2015: 0.78,
        2016: 0.78, 2017: 0.78, 2018: 0.67, 2019: 0.67, 2020: 0.67, 2021: 0.67,
        2022: 0.67, 2023: 0.76}


def _rows(year_values, column="gas_collection_efficiency", **extra):
    frame = pd.DataFrame({"row_year": list(year_values), column: list(year_values.values())})
    for name, value in extra.items():
        frame[name] = value
    return frame


def test_one_value_is_used_in_every_year():
    series = _carried_series([2015, 2018, 2023], [0.76, 0.76, 0.76], YEARS)
    assert (series == 0.76).all() and list(series.index) == list(YEARS)


def test_several_values_are_carried_back_and_forward():
    series = _carried_series(list(POLK), list(POLK.values()), YEARS)
    assert series.loc[1970] == 0.58      # the first report, carried back
    assert series.loc[2011] == 0.58
    assert series.loc[2012] == 0.74
    assert series.loc[2020] == 0.67
    assert series.loc[2023] == 0.76
    assert series.loc[2050] == 0.76      # the latest report, carried forward


def test_a_gap_carries_the_last_report_forward():
    series = _carried_series([2010, 2015], [0.5, 0.7], YEARS)
    assert series.loc[2014] == 0.5 and series.loc[2015] == 0.7


def test_nothing_reported_is_none():
    assert _carried_series([2010, np.nan], [np.nan, 0.5], YEARS) is None


def test_two_values_in_one_year_are_averaged():
    series = _carried_series([2010, 2010, 2012], [0.4, 0.6, 0.8], YEARS)
    assert series.loc[2011] == pytest.approx(0.5)


def test_a_modelled_site_keeps_its_measured_efficiency():
    """No CH4, so no reported_emissions_year. Keyed on that, the measured 0.82 was
    replaced by the sanitary default of 0.6."""
    rows = _rows({2021: 0.82, 2022: 0.82}, reported_emissions_year=np.nan)
    _, _, gce = _resolve_gas_capture(True, "Sanitary Landfill", pd.Series(dtype=object), rows, YEARS)
    assert (gce == 0.82).all()
    assert GAS_EFF_OPTIONS["Sanitary Landfill"] != 0.82


def test_gas_capture_follows_the_reports_year_by_year():
    rows = _rows(POLK, reported_emissions_year=list(POLK))
    _, _, gce = _resolve_gas_capture(False, "Sanitary Landfill", pd.Series(dtype=object), rows, YEARS)
    assert gce.loc[1990] == 0.58 and gce.loc[2019] == 0.67 and gce.loc[2040] == 0.76


def test_rows_without_a_year_of_their_own_keep_the_old_rule():
    rows = pd.DataFrame({"reported_emissions_year": [2020, 2021], "gas_collection_efficiency": [0.35, 0.45]})
    _, _, gce = _resolve_gas_capture(True, "Sanitary Landfill", pd.Series(dtype=object), rows, YEARS)
    assert gce.loc[1999] == pytest.approx(0.40)   # the site mean, as before


def test_oxidation_follows_the_same_rule():
    rows = _rows({2013: 0.10, 2020: 0.22}, column="oxidation")
    series = _build_oxidation_series(0.1, rows.iloc[0], rows, YEARS)
    assert series.loc[1980] == 0.10 and series.loc[2019] == 0.10 and series.loc[2045] == 0.22


def test_oxidation_without_input_keeps_the_default():
    rows = _rows({2013: np.nan}, column="oxidation")
    series = _build_oxidation_series(0.05, rows.iloc[0], rows, YEARS)
    assert (series == 0.05).all()


def test_nullable_columns_are_read():
    years = pd.array([2010, pd.NA, 2012], dtype="Int64")
    values = pd.array([0.5, 0.6, pd.NA], dtype="Float64")
    assert (_carried_series(years, values, YEARS) == 0.5).all()


def test_a_measured_efficiency_on_undated_rows_is_kept():
    """Rows carrying `row_year` whose years are all blank must not fall through to the
    emissions-year rule, which drops a modelled site's value for the default."""
    rows = pd.DataFrame({"row_year": [np.nan, np.nan], "gas_collection_efficiency": [0.82, 0.82],
                         "reported_emissions_year": [np.nan, np.nan]})
    _, _, gce = _resolve_gas_capture(True, "Sanitary Landfill", pd.Series(dtype=object), rows, YEARS)
    assert (gce == 0.82).all()


def test_undated_rows_without_a_value_get_the_gated_default():
    rows = pd.DataFrame({"row_year": [np.nan], "gas_collection_efficiency": [np.nan],
                         "reported_emissions_year": [np.nan]})
    _, _, gce = _resolve_gas_capture(True, "Sanitary Landfill", pd.Series(dtype=object), rows, YEARS)
    assert (gce == GAS_EFF_OPTIONS["Sanitary Landfill"]).all()


def test_undated_oxidation_is_one_value_in_every_year():
    """The legacy rule would overwrite 2020 with that row's 0.1; on these rows the
    emissions year does not date the value."""
    rows = pd.DataFrame({"row_year": [np.nan, np.nan], "oxidation": [0.1, 0.3],
                         "reported_emissions_year": [2020, np.nan]})
    series = _build_oxidation_series(0.05, rows.iloc[0], rows, YEARS)
    assert np.allclose(series.to_numpy(), 0.2)
