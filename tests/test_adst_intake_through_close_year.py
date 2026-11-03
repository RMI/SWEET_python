"""The advanced DST takes a site's waste in its open year, its close year and every year between.

SWEET main made this the rule in SWEET#112 (decision ``intake-through-close-year``). The
advanced DST draws the boundary in its own two places, both checked here.
"""

import pandas as pd

from SWEET_python import dst_common
from SWEET_python.advanced_dst_city import _accepting_mask

YEARS = pd.Index(range(1970, 2051))


def test_the_window_keeps_the_open_and_close_years():
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
    assert dst_common.apply_window(mass, 1990, dst_common.NEVER_CLOSES).loc[2050, "food"] == 1.0
