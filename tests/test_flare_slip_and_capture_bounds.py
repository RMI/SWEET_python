"""Gas capture arithmetic: flare slip counted once and not oxidised, and a bounded efficiency.

Three defects lived in the same expression.

The annual engine added ``+ ch4_capture * 0.02`` on top of a ``ch4_capture``
that was already ``generation x capture_efficiency x flaring``, charging the
flare's inefficiency a second time -- and it made this engine disagree with the
monthly one, which never had it.

Both engines then oxidised the flare slip with the uncaptured gas. Slip leaves
through the flare and never passes the cover, so it is now emitted as is, as in
the SWEET Excel workbook ('CH4 Emit'!D10):

    emitted = generated x (1 - capture) x (1 - oxidation)
            + generated x capture x (1 - flaring)

Separately, ``gas_capture_efficiency`` reaches the engine as a bare float map
with no bound anywhere, so a value above 1 produced negative emissions.
"""

import numpy as np
import pandas as pd
import pytest

from SWEET_python import defaults_2019
from SWEET_python.landfill import CH4_DENSITY_KG_PER_M3
from SWEET_python.model_v2 import SWEET


YEARS = list(range(2000, 2051))
COMPONENTS = ["food", "green", "wood", "paper_cardboard", "textiles"]


def _emissions(generation, capture_eff, flaring, oxidation):
    """The formula both engines should follow."""
    uncaptured = generation * (1 - capture_eff)
    slip = generation * capture_eff * (1 - flaring)
    return uncaptured * (1 - oxidation) + slip


def _run_engines(capture_eff, flaring, oxidation):
    """Yearly (generated, emitted) from each engine for 1,000 t of food a year."""
    years = pd.Index(YEARS)
    landfill = {
        "open_date": YEARS[0],
        "close_date": None,
        "ks": {c: pd.Series(0.2, index=years) for c in COMPONENTS},
        "waste_mass_df": pd.DataFrame({c: 1000.0 if c == "food" else 0.0 for c in COMPONENTS}, index=years),
        "mcf": pd.Series(1.0, index=years),
        "gas_capture_efficiency": pd.Series(capture_eff, index=years),
        "oxidation_factor": pd.Series(oxidation, index=years),
        "flaring": pd.Series(flaring, index=years),
    }
    city = {"components": COMPONENTS}
    params = {"year_of_data_pop": None, "growth_rate_historic": None, "growth_rate_future": None}

    _, q, ch4, _ = SWEET(city, params, dict(landfill, ks=dict(landfill["ks"]))).estimate_emissions2()
    annual = (ch4.sum(axis=1), q["total"])
    _, q, ch4, _ = SWEET(city, params, dict(landfill, ks=dict(landfill["ks"]))).estimate_emissions_monthly()
    by_year = lambda frame: frame.groupby(frame.index.year).sum()
    monthly = (by_year(ch4).sum(axis=1), by_year(q["total"]))
    return {"annual": annual, "monthly": monthly}


CASES = [(0.0, 0.98, 0.10), (0.6, 0.98, 0.22), (0.3, 0.5, 0.10), (0.6, 0.0, 0.22), (1.0, 1.0, 0.22)]


class TestFlareSlip:
    @pytest.mark.parametrize("engine", ["annual", "monthly"])
    @pytest.mark.parametrize("capture_eff, flaring, oxidation", CASES)
    def test_each_engine_follows_the_formula(self, engine, capture_eff, flaring, oxidation):
        generated, emitted = _run_engines(capture_eff, flaring, oxidation)[engine]
        expected = _emissions(generated, capture_eff, flaring, oxidation)
        np.testing.assert_allclose(emitted, expected, rtol=1e-12, atol=1e-9)

    def test_flare_slip_is_not_oxidised(self):
        """At 60% capture, 98% flaring and 0.22 oxidation, the slip is 1.2% of
        generation and adds 0.8% to emissions over oxidising it with the rest."""
        generation, eff, flaring, ox = 1000.0, 0.6, 0.98, 0.22
        oxidised_slip = (generation - generation * eff * flaring) * (1 - ox)
        now = _emissions(generation, eff, flaring, ox)
        assert now - oxidised_slip == pytest.approx(generation * eff * (1 - flaring) * ox)
        assert now / oxidised_slip - 1 == pytest.approx(0.0082, abs=1e-4)

    def test_flare_inefficiency_is_charged_once(self):
        """The old annual expression added another 2% of the *destroyed* gas back."""
        generation, eff, flaring, ox = 1000.0, 0.6, 0.98, 0.22
        destroyed = generation * eff * flaring
        now = _emissions(generation, eff, flaring, ox)
        old = (generation - destroyed) * (1 - ox) + destroyed * 0.02
        assert old > now

    def test_no_capture_is_unaffected(self):
        assert _emissions(1000.0, 0.0, 0.98, 0.22) == pytest.approx(1000.0 * (1 - 0.22))

    def test_perfect_capture_and_destruction_emits_nothing(self):
        assert _emissions(1000.0, 1.0, 1.0, 0.22) == pytest.approx(0.0)

    def test_vented_gas_escapes_oxidation(self):
        """Gas collected and then not destroyed leaves through a stack, not the
        cover. It used to be oxidised anyway, so a venting site came out the same
        as one with no capture at all."""
        generation, ox = 1000.0, 0.22
        vented = _emissions(generation, 0.6, 0.0, ox)
        assert vented == pytest.approx(generation * 0.4 * (1 - ox) + generation * 0.6)
        assert vented > _emissions(generation, 0.0, 0.98, ox)


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
