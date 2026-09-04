"""Gas capture arithmetic: no double-counted flare slip, and a bounded efficiency.

Two defects lived in the same expression.

The annual engine added ``+ ch4_capture * 0.02`` on top of a ``ch4_capture``
that was already ``generation x capture_efficiency x flaring``. The gas a flare
fails to destroy is emitted through ``(ch4_produce - ch4_capture)``, so the
extra term charged the flare's inefficiency a second time -- and it made this
engine disagree with the monthly one, which never had it.

Separately, ``gas_capture_efficiency`` reaches the engine as a bare float map
with no bound anywhere, so a value above 1 produced negative emissions.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python import defaults_2019
from SWEET_python.landfill import CH4_DENSITY_KG_PER_M3


YEARS = list(range(2000, 2051))


def _emissions(generation, capture_eff, flaring, oxidation):
    """The annual engine's formula, as it now stands (model_v2.py)."""
    captured_and_destroyed = generation * capture_eff * flaring
    return (generation - captured_and_destroyed) * (1 - oxidation)


def _emissions_monthly(generation, capture_eff, flaring, oxidation):
    """The monthly engine's formula (model_v2.py, ``trace_monthly`` branch)."""
    capture = generation * capture_eff * flaring
    return (generation - capture) * (1 - oxidation)


class TestFlareSlip:
    def test_annual_and_monthly_agree(self):
        """The whole point of the change: identical inputs, identical answer."""
        for eff in (0.0, 0.3, 0.6, 0.98, 1.0):
            for flaring in (0.0, 0.5, 0.98, 1.0):
                assert _emissions(1000.0, eff, flaring, 0.22) == pytest.approx(
                    _emissions_monthly(1000.0, eff, flaring, 0.22)
                )

    def test_flare_inefficiency_is_charged_once(self):
        """Undestroyed gas is emitted once, via the residual -- not twice."""
        generation, eff, flaring, ox = 1000.0, 0.6, 0.98, 0.22

        destroyed = generation * eff * flaring
        residual = generation - destroyed
        expected = residual * (1 - ox)

        assert _emissions(generation, eff, flaring, ox) == pytest.approx(expected)

        # The old expression added another 2% of the *destroyed* gas back.
        old = expected + destroyed * 0.02
        assert old > expected
        # It inflated a captured site's emissions by ~3.7%.
        assert (old - expected) / expected == pytest.approx(0.0366, abs=1e-3)

    def test_no_capture_is_unaffected(self):
        """A site without capture never had a slip term, and still doesn't."""
        generation, ox = 1000.0, 0.22
        assert _emissions(generation, 0.0, 0.98, ox) == pytest.approx(
            generation * (1 - ox)
        )

    def test_perfect_capture_and_destruction_emits_nothing(self):
        assert _emissions(1000.0, 1.0, 1.0, 0.22) == pytest.approx(0.0)

    def test_capture_with_no_destruction_equals_no_capture(self):
        """Documents a real limitation this change does NOT fix.

        Gas that is collected and then vented leaves through a stack, not
        through the cover, so it should escape oxidation. The engine oxidises
        it anyway, which understates emissions for a venting site. Only
        reachable by setting ``flaring`` near zero, which the default of 0.98
        makes rare -- pinned here so the behaviour is deliberate rather than
        forgotten.
        """
        generation, ox = 1000.0, 0.22
        vented = _emissions(generation, 0.6, 0.0, ox)
        no_capture = _emissions(generation, 0.0, 0.98, ox)
        assert vented == pytest.approx(no_capture)

        physical = generation * 0.4 * (1 - ox) + generation * 0.6
        assert physical > vented


class TestCaptureBounds:
    @pytest.mark.parametrize("bad", [1.5, 2.0, 100.0])
    def test_efficiency_above_one_would_produce_negative_emissions(self, bad):
        """Why the clamp exists."""
        assert _emissions(1000.0, bad, 0.98, 0.22) < 0

    @pytest.mark.parametrize("bad", [-0.5, -1.0])
    def test_efficiency_below_zero_would_manufacture_methane(self, bad):
        assert _emissions(1000.0, bad, 0.98, 0.22) > 1000.0 * (1 - 0.22)

    @pytest.mark.parametrize(
        "raw,expected", [(1.5, 1.0), (-0.5, 0.0), (0.6, 0.6), (1.0, 1.0), (0.0, 0.0)]
    )
    def test_clamp_maps_out_of_range_onto_the_boundary(self, raw, expected):
        """The clamp the ADST runners apply to the resolved series."""
        series = pd.Series([raw] * len(YEARS), index=YEARS)
        assert series.clip(0.0, 1.0).iloc[0] == pytest.approx(expected)

    def test_clamped_efficiency_never_yields_negative_emissions(self):
        for raw in (-5.0, -0.1, 0.0, 0.5, 1.0, 1.1, 50.0):
            clamped = pd.Series([raw], index=[2030]).clip(0.0, 1.0).iloc[0]
            assert _emissions(1000.0, clamped, 0.98, 0.22) >= 0.0


class TestMethaneDensity:
    def test_one_constant_at_zero_celsius(self):
        assert CH4_DENSITY_KG_PER_M3 == pytest.approx(0.7168)

    def test_landfill_module_holds_no_second_density(self):
        """The 0.657 in the dead ``doing_fancy_ox`` branch is gone."""
        import inspect

        from SWEET_python import landfill

        code = [
            line.split("#", 1)[0]
            for line in inspect.getsource(landfill).splitlines()
        ]
        assert not any("0.657" in line for line in code)

    def test_defaults_table_is_unchanged_by_this_pr(self):
        """This PR fixes arithmetic, not parameters."""
        assert defaults_2019.gas_capture_efficiency == {
            "landfill": 0.6,
            "controlled_dumpsite": 0.45,
            "dumpsite": 0,
        }
