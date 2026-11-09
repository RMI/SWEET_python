"""The ``City`` paths burn everything sent to combustion, too.

``City.combustion_reject_rate`` is 0 (decision ``combustion-reject``), and
``tests/test_adst_city_incineration.py`` pins it through the city-level advanced
DST. The ``City`` paths read the same field on their own, in
``_calculate_diverted_masses`` and its siblings, so a stale 10% in one of them
would pass those tests. This drives one end to end: a Custom Location
(``dst_baseline_blank``) in Japan, whose country default incinerates 76% of its
waste. The combustion it diverts must be that whole 76%.
"""

import pytest

from SWEET_python.city_params import City

YEAR = 2040


def _combustion_share():
    """Combusted mass over generated mass, and the share asked for, in ``YEAR``."""
    city = City("Custom Location")
    city.dst_baseline_blank("JPN", 2_000_000, 1500.0, 15.0)
    parameters = city.baseline_parameters
    generated = parameters.waste_generated_df
    generated = getattr(generated, "df", generated)
    burnt = parameters.divs.combustion.sum(axis=1).loc[YEAR]
    asked = parameters.div_fractions.loc[YEAR, "combustion"]
    return burnt / generated.sum(axis=1).loc[YEAR], asked


def test_a_custom_location_burns_all_it_sends_to_combustion():
    share, asked = _combustion_share()

    assert asked > 0.5, "the country default no longer incinerates; pick another"
    assert share == pytest.approx(asked, rel=1e-9)


def test_a_set_reject_still_comes_off_the_combustion(monkeypatch):
    original = City.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.combustion_reject_rate = 0.1

    monkeypatch.setattr(City, "__init__", init)
    share, asked = _combustion_share()

    assert share == pytest.approx(asked * 0.9, rel=1e-9)
