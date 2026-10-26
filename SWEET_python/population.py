"""Country population by year, from the UN's World Population Prospects 2024.

SWEET grows a place's waste with its country's population, year by year: the factor
for a year is P(year) / P(anchor year) (see class_defs.growth_factors_for_years).
The WasteMAP tools, which have no ``pop_data``, read them here. Climate TRACE's
pipeline reads this same file (1970 onward, in its ``util/population.py``) and hands
SWEET those numbers through ``pop_data``, so there is one copy of the table, versioned
with this package.

``pops_yearly.csv`` runs from 1950 because the site tool models a landfill from its
real opening year. Built with Climate TRACE's
``diagnostic_scripts/generate_pops_yearly.py`` from
``WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx``, "Total Population, as of 1
January (thousands)": estimates to 2023, medium variant after.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
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


def parse_city_population(value) -> Optional[dict[int, float]]:
    """A city's population by year, from a mapping or its JSON text, or ``None``.

    Climate TRACE hands a site's nearby city in as JSON text on the site row, because a
    row is saved to parquet between stages and text survives that. Keys may be ints or
    the strings JSON makes of them. Years with no positive number are dropped: the UN
    leaves a city blank in the years it was under 50,000 people, and a blank means "not
    a city yet", not zero people.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"city population must be a mapping of year to people, not {type(value).__name__}")
    years = {}
    for year, people in value.items():
        people = pd.to_numeric(people, errors="coerce")
        if pd.notna(people) and people > 0:
            years[int(year)] = float(people)
    return years or None


def city_population_series(
    city: Optional[Mapping[int, float]], country: Optional[pd.Series]
) -> Optional[pd.Series]:
    """A city's population in every year of ``country``'s index, its gaps filled.

    - **In a year the city has a number**, that number.
    - **Between two of them**, constant growth from one to the next (log-linear), for a
      source that reports every five years.
    - **Before its first year or after its last**, the city's nearest number moved by
      the country's population: ``city[first] * country[year] / country[first]``. A city
      appears in the UN's list only once it passes 50,000 people, so half the cities a
      site is near start after 1975 and a quarter after 2000.

    The country's shape fills the ends because the other ways bend: a straight line back
    from a fast-growing city reaches zero within a few decades, and its growth rate
    compounded back shrinks it to nearly nothing. The country's shape joins smoothly,
    never goes below zero, and is what a site with no city gets anyway. A city grows
    faster than its country, so it slightly overstates a city's population before its
    first year.

    ``None`` when there is no city, no country series, or no city year inside the
    country's years; the caller then grows by the country alone.
    """
    # decision: population-growth
    if not city or country is None:
        return None
    observed = pd.Series(city, dtype=float).sort_index()
    observed = observed[observed.index.isin(country.index)]
    if observed.empty:
        return None

    series = pd.Series(np.nan, index=country.index, dtype=float)
    series[observed.index] = observed.values
    first, last = observed.index[0], observed.index[-1]
    inside = (series.index >= first) & (series.index <= last)
    # Only the missing years: a reported number survives exactly, not via exp(log(x)).
    gaps = inside & series.isna().to_numpy()
    series[gaps] = np.exp(np.log(series[inside]).interpolate(method="index"))[gaps[inside]]
    before = series.index < first
    series[before] = observed[first] * country[before] / country[first]
    after = series.index > last
    series[after] = observed[last] * country[after] / country[last]
    return series
