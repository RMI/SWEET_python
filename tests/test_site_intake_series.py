"""A site with several reported years: average them at their own years, project, overwrite.

The average used to be grown from the latest report, though it describes the middle of
the reports: an average of 2014-2019 grown from 2019 understated a growing site's whole
series by two or three years of growth.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python.city_params import _site_intake_series

YEARS = np.arange(1970, 2051)


def _growth(at_years):
    """2% a year, relative to 2000. Any fixed reference year gives the same series."""
    return 1.02 ** (np.asarray(at_years, dtype=float) - 2000)


def _series(report_years, tonnes, fallback=np.nan):
    out = _site_intake_series(report_years, tonnes, YEARS, _growth, fallback)
    return pd.Series(out, index=YEARS)


def test_reports_on_the_population_curve_give_that_curve_everywhere():
    years = [2014, 2015, 2016, 2017, 2018, 2019]
    tonnes = [1000 * 1.02 ** (y - 2014) for y in years]
    out = _series(years, tonnes)
    expected = 1000 * 1.02 ** (YEARS - 2014)
    np.testing.assert_allclose(out.to_numpy(), expected)
    # Anchored at 2019, the average (the 2016-17 level) was ~5% low in every other year.
    assert out[2030] == pytest.approx(1000 * 1.02 ** 16)


def test_a_year_reported_twice_counts_once_and_takes_the_mean():
    out = _series([2015, 2015, 2015, 2016], [100.0, 300.0, 200.0, 204.0])
    assert out[2015] == pytest.approx(200.0)
    assert out[2016] == pytest.approx(204.0)
    # Level from {2015: 200, 2016: 204} at their own years: 200/1.02^15 and 204/1.02^16.
    level = np.mean([200 / 1.02 ** 15, 204 / 1.02 ** 16])
    assert out[2020] == pytest.approx(level * 1.02 ** 20)


def test_reports_from_2010_on_set_the_level_when_there_are_any():
    out = _series([2005, 2012], [999.0, 100.0])
    assert out[2020] == pytest.approx(100 * 1.02 ** 8)
    assert out[2005] == 999.0  # still overwritten with its own value


def test_older_reports_set_the_level_when_there_are_no_recent_ones():
    out = _series([2003, 2005], [100.0, 1.02 ** 2 * 100])
    assert out[2010] == pytest.approx(100 * 1.02 ** 7)


def test_undated_values_are_averaged_and_anchored_at_2020():
    out = _series([np.nan, np.nan], [100.0, 300.0])
    assert out[2020] == pytest.approx(200.0)
    assert out[2021] == pytest.approx(204.0)


def test_with_no_value_at_all_the_canonical_intake_is_used():
    out = _series([2015], [np.nan], fallback=50.0)
    assert out[2020] == pytest.approx(50.0)
