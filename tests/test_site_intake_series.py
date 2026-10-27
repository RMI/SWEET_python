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


# --------------------------------------------------------------------------- #
# Clearly shrinking sites follow their straight line
# --------------------------------------------------------------------------- #

from SWEET_python.city_params import _regularized_incomplete_beta, _shrinking_line  # noqa: E402

SHRINKING = {2014: 1000.0, 2015: 900.0, 2016: 820.0, 2017: 700.0, 2018: 610.0, 2019: 500.0}


@pytest.mark.parametrize("t, df", [(12.706, 1), (4.303, 2), (2.776, 4), (2.228, 10)])
def test_the_p_value_helper_gives_the_t_tables_five_percent(t, df):
    assert _regularized_incomplete_beta(df / 2, 0.5, df / (df + t * t)) == pytest.approx(0.05, abs=1e-4)


def test_a_shrinking_site_follows_its_line_after_its_last_report_and_stops_at_zero():
    out = _series(list(SHRINKING), list(SHRINKING.values()))
    slope, intercept = _shrinking_line(pd.Series(SHRINKING))
    assert slope == pytest.approx(-99.71, abs=0.01)
    assert out[2020] == pytest.approx(intercept + slope * 2020)  # about 407 t, not ~760
    assert out[2030] == 0.0  # the line crosses zero in 2024
    for year, tonnes in SHRINKING.items():
        assert out[year] == tonnes  # reported years keep their own values


def test_before_its_first_report_a_shrinking_site_follows_population_back():
    out = _series(list(SHRINKING), list(SHRINKING.values()))
    slope, intercept = _shrinking_line(pd.Series(SHRINKING))
    at_2014 = intercept + slope * 2014
    assert out[2004] == pytest.approx(at_2014 / 1.02 ** 10)  # not ~2,000 t up the line


@pytest.mark.parametrize(
    "reports",
    [
        {2014: 1000.0, 2016: 500.0},  # two years always fit a line
        {2014: 1000.0, 2015: 950.0, 2016: 920.0, 2017: 900.0},  # down 10%: not clearly shrinking
        {2014: 1000.0, 2015: 300.0, 2016: 1200.0, 2017: 400.0, 2018: 900.0, 2019: 350.0},  # noise
        {2014: 500.0, 2015: 600.0, 2016: 700.0},  # growing
    ],
)
def test_other_sites_keep_the_population_projection(reports):
    assert _shrinking_line(pd.Series(reports)) is None
