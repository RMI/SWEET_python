"""A site takes waste in its open year, its close year and every year between.

It used to stop a year early, so every closed site lost its close year's waste. The
boundary is drawn in several places; each is checked here.
"""

import inspect

import pandas as pd
import pytest

from SWEET_python import city_params, constants
from SWEET_python import dst_common
from SWEET_python.advanced_dst_city import _accepting_mask

YEARS = pd.Index(range(1970, 2051))


def test_the_flag_climate_trace_reads_says_so():
    assert constants.INTAKE_THROUGH_CLOSE_YEAR is True


def test_the_advanced_dst_window_keeps_the_open_and_close_years():
    mass = pd.DataFrame(1.0, index=YEARS, columns=["food"])
    windowed = dst_common.apply_window(mass, 1990, 2040)
    assert windowed.loc[1989, "food"] == 0.0
    assert windowed.loc[1990, "food"] == 1.0
    assert windowed.loc[2040, "food"] == 1.0
    assert windowed.loc[2041, "food"] == 0.0
    assert windowed["food"].sum() == 51


def test_the_city_tool_accepts_waste_through_the_close_year():
    accepting = _accepting_mask([(1990, 2040), (2000, 2000)], pd.Index(["a", "b"]), YEARS)
    assert accepting.loc[2040, "a"] and not accepting.loc[2041, "a"]
    assert accepting["b"].sum() == 1  # opens and closes the same year: that one year


def test_never_closing_still_takes_the_last_model_year():
    mass = pd.DataFrame(1.0, index=YEARS, columns=["food"])
    assert dst_common.apply_window(mass, 1990, constants.NEVER_CLOSES).loc[2050, "food"] == 1.0


def test_the_site_window_keeps_the_open_and_close_years():
    accepting = city_params._intake_years(YEARS, 1990, 2040)
    assert accepting.loc[1989] == 0.0
    assert accepting.loc[1990] == 1.0
    assert accepting.loc[2040] == 1.0
    assert accepting.loc[2041] == 0.0
    assert accepting.sum() == 51
    assert city_params._intake_years(YEARS, 2000, 2000).sum() == 1  # opens and closes that year
    assert city_params._intake_years(YEARS, 1990, constants.NEVER_CLOSES).loc[2050] == 1.0


@pytest.mark.parametrize(
    "method, calls",
    [
        # Not run here: it reads its weather from the live database. Nothing calls it today.
        ("site_only_estimate", 1),
        ("site_only_estimate_trace", 1),
        ("citysite_estimate_trace", 2),  # one city at the site, and several
    ],
)
def test_each_site_path_windows_its_intake_with_the_shared_rule(method, calls):
    source = inspect.getsource(getattr(city_params.City, method))
    assert source.count("_intake_years(") == calls
    assert "close_date-1" not in source.replace(" ", "")
