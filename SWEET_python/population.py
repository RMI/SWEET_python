"""Country population by year, from the UN's World Population Prospects 2024.

SWEET grows a place's waste with its country's population, year by year: the factor
for a year is P(year) / P(anchor year) (see class_defs.growth_factors_for_years).
Climate TRACE's pipeline hands SWEET the same numbers through ``pop_data``; the
WasteMAP tools, which have no ``pop_data``, read them here.

``pops_yearly.csv`` is Climate TRACE's ``static_data/pops_yearly.csv`` (1970-2050)
unchanged, extended back to 1950 from the same source, because the site tool models
a landfill from its real opening year. Built with Climate TRACE's
``diagnostic_scripts/generate_pops_yearly.py`` from
``WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx``, "Total Population, as of 1
January (thousands)": estimates to 2023, medium variant after.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from SWEET_python.constants import MODEL_END_YEAR, MODEL_START_YEAR

_TABLE_PATH = Path(__file__).with_name("pops_yearly.csv")


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    table = pd.read_csv(_TABLE_PATH, index_col=0)
    table.columns = [int(c) for c in table.columns]
    return table


def country_population_series(iso3: Optional[str]) -> Optional[pd.Series]:
    """The country's population for every year in the table, or ``None`` if it has none."""
    table = _table()
    if not isinstance(iso3, str) or iso3 not in table.index:
        return None
    series = table.loc[iso3].astype(float)
    if series.isna().any() or not (series > 0).all():
        return None
    return series.copy()  # a copy, so no caller can edit the cached table


def average_growth_rates(
    series: pd.Series,
    anchor_year: int,
    start_year: int = MODEL_START_YEAR,
    end_year: int = MODEL_END_YEAR,
) -> tuple[float, float]:
    """The two constant yearly multipliers that match ``series`` at the ends of the window.

    Historic: ``(P(anchor) / P(start)) ** (1 / (anchor - start))``. Future:
    ``(P(end) / P(anchor)) ** (1 / (end - anchor))``. SWEET does not apply these when
    it has the series; they are the one-number growth rate a person reads, and what
    the constant-rate fallback compounds. An anchor at either end of the window gives
    that side the other side's rate.
    """
    anchor = min(max(int(anchor_year), start_year), end_year)
    p = series
    historic = (p[anchor] / p[start_year]) ** (1 / (anchor - start_year)) if anchor > start_year else None
    future = (p[end_year] / p[anchor]) ** (1 / (end_year - anchor)) if anchor < end_year else None
    historic = future if historic is None else historic
    future = historic if future is None else future
    return float(historic), float(future)
