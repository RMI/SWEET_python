"""Our decay engines against the SWEET Excel workbook (v4.0.6), on identical inputs.

``scripts/excel_engine_check.py`` filled the workbook with six landfills, let
Excel recalculate, and saved what Excel used and what it got under
``tests/fixtures/excel_sweet_4_0_6/``: waste by stream and year, k, L0, MCF,
collection efficiency, oxidation, flare destruction, and the methane generated
and emitted. These tests run both engines on exactly those inputs, so a failure
means an engine changed how it computes, not what it is given.

What the tests pin down:

- The annual engine (``estimate_emissions2``) reproduces Excel's generation
  exactly. It is the same formula: waste placed in year y emits from y + 1 at
  k x L0 x MCF x exp(-k(t - y - 0.5)).
- Its emissions match Excel exactly too. Until 2026-10 they did not: ours
  oxidised the gas a flare fails to destroy, which Excel lets out as is.
- The monthly engine, the one the TRACE pipeline runs, spreads each year's waste
  over its months and decays month by month. That is a different discretisation
  of the same model, so it is held to a tolerance, not to equality.

Excel itself is wrong in one cell: see ``KNOWN_EXCEL_DEFECTS`` in the script.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "excel_engine_check.py"
FIXTURES = sorted((Path(__file__).parent / "fixtures" / "excel_sweet_4_0_6").glob("*.csv"))

# Loaded by path: ``scripts/`` is not a package.
_spec = importlib.util.spec_from_file_location("excel_engine_check", SCRIPT)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

# The monthly engine's gap to Excel over a site's life, and in its worst year.
# Measured on these fixtures: the worst years are the first after a site closes
# and leap years, where whole months of deposits fall either side of the
# half-year emission lag.
MONTHLY_TOTAL_PCT = 0.5
MONTHLY_WORST_YEAR_PCT = 4.0


def _side(path: Path) -> pd.DataFrame:
    excel = pd.read_csv(path)
    excel.insert(0, "case", path.stem)
    side = check.compare(excel)
    # As in the script's summary: skip a site's first sliver of gas, where both
    # sides are near zero, and the year Excel gets wrong.
    return side[
        (side["excel_generated_m3"] > 0.01 * side["excel_generated_m3"].max())
        & (side["excel_defect"] == "")
    ]


def test_fixtures_exist():
    # By name, not count: a stale extra CSV must not stand in for a missing case.
    assert {p.stem for p in FIXTURES} == set(check.CASES), "re-run scripts/excel_engine_check.py --write-fixtures"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_annual_engine_generation_matches_excel(path):
    side = _side(path)
    np.testing.assert_allclose(side["annual_generated_m3"], side["excel_generated_m3"], rtol=1e-9)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_annual_engine_emissions_match_excel(path):
    side = _side(path)
    np.testing.assert_allclose(side["annual_emitted_m3"], side["excel_emitted_m3"], rtol=1e-9)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_monthly_engine_within_tolerance_of_excel(path):
    side = _side(path)
    for quantity in ("generated", "emitted"):
        ours, theirs = side[f"monthly_{quantity}_m3"], side[f"excel_{quantity}_m3"]
        total_pct = 100 * (ours.sum() / theirs.sum() - 1)
        assert abs(total_pct) < MONTHLY_TOTAL_PCT, f"{quantity}: lifetime gap {total_pct:.3f}%"
        worst = (100 * (ours / theirs - 1)).abs().max()
        assert worst < MONTHLY_WORST_YEAR_PCT, f"{quantity}: worst year {worst:.3f}%"
