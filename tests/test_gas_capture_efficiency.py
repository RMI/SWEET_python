"""Regression tests for reading the source gas-capture efficiency in the
city DST pipeline path (`City.load_andre_params`).

Change: `load_andre_params` previously hardcoded the landfill gas-capture
efficiency to 0.6 for every city, discarding the source
`gas_capture_efficiency_percent` value that the map-data pipeline already
selects from the database. It now reads that source value (a percentage, e.g.
50 -> 0.50) via `_normalize_gas_capture_efficiency`, falling back to the 0.6
model default when the source is missing / NaN / non-numeric. The value flows
through `_calculate_divs` into the with-capture landfill and out to
`cities_for_map_*.csv`'s "Methane Capture Efficiency (%)" column.

In the live DB this column is almost always null (one city currently reports a
real 50%). These unit tests pin the normalizer's behaviour and are DB-free.
"""

import pytest

from SWEET_python.city_params import _normalize_gas_capture_efficiency as normalize


@pytest.mark.parametrize(
    "raw, expected",
    [
        (50, 0.50),        # the real source value currently in the DB (percent form)
        (50.0, 0.50),
        ("50", 0.50),      # numeric string (some DB drivers return object dtype)
        (25, 0.25),        # another percentage
        (0.5, 0.50),       # already a fraction -> used as-is
        (0, 0.0),          # an explicit zero is a real reported value
        (150, 1.0),        # implausible high percent -> clamped to 1.0
    ],
)
def test_source_values_are_normalized(raw, expected):
    assert normalize(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, float("nan"), "", "n/a", "unknown"])
def test_missing_or_nonnumeric_falls_back_to_default(raw):
    # Missing / NaN / non-numeric -> the 0.6 model default (the common case).
    assert normalize(raw) == pytest.approx(0.6)


def test_custom_default_is_respected():
    assert normalize(None, default=0.45) == pytest.approx(0.45)
