"""The site type to assume for a site whose type nobody recorded.

Every SWEET path that has to guess a site type calls :func:`default_site_type`
or :func:`landfill_is_default`, and so do Climate TRACE's pipeline and WasteMAP.
Before 2026-10-13 the guess was made in nine places by three different rules,
so a Kazakh site with no type was a sanitary landfill in Climate TRACE and an
open dump in WasteMAP's site tool. Do not reintroduce a local copy.

The rule is Climate TRACE's, moved here unchanged from
``landfill_table_ops.fill_landfill_types``. A site is assumed to be a sanitary
landfill when its country's disposal system is engineered, and an open dump
otherwise. That is a claim about the national system, which is knowable even
when the individual site is in no registry: there are no unmanaged waste piles
in Denmark or the United States. A country counts as engineered if any of these
holds:

- **It is high income** in the World Bank grouping Climate TRACE reads
  (:mod:`SWEET_python.country_income`).
- **Its SWEET region is in** :data:`LANDFILL_DEFAULT_REGIONS`. That is
  ``defaults_2019.landfill_default_regions`` plus Western Asia and Central Asia.
  Central Asia joins on the argument that admits Russia: the post-Soviet
  republics inherited one engineered "polygon" standard, so splitting
  Kazakhstan from Russia would be arbitrary.
- **It is named in** :data:`LANDFILL_DEFAULT_COUNTRIES`. Mexico, Brazil and
  Costa Rica are there because their own disposal statistics say engineered
  landfill while their region does not (What a Waste 3.0: Mexico open-dumps 3%
  of its waste, Brazil 13%). China and Russia already qualify by region; naming
  them keeps a change to SWEET's region map from re-typing the two largest
  inventories.

:data:`DUMPSITE_COUNTRIES` overrides all three. Mongolia (Central Asia in
SWEET's map) and Yemen (Western Asia) are swept in by a region whose average
does not describe them.

A country SWEET cannot place in a region, or no country at all, is an open dump.

This decides MCF and oxidation, so it is worth about 1.5x on a site's modelled
methane: a sanitary landfill is MCF 1.0 with oxidation 0.10, and an open dump of
unknown depth is MCF 0.6 with no oxidation.
"""

from typing import Optional

import pandas as pd

from SWEET_python import defaults_2019
from SWEET_python.country_income import INCOME_ID_BY_ISO3

SANITARY_LANDFILL = "Sanitary Landfill"
DUMPSITE = "Dumpsite"

LANDFILL_DEFAULT_REGIONS = frozenset(
    set(defaults_2019.landfill_default_regions) | {"Western Asia", "Central Asia"}
)
LANDFILL_DEFAULT_COUNTRIES = frozenset({"CHN", "RUS", "MEX", "BRA", "CRI"})
DUMPSITE_COUNTRIES = frozenset({"MNG", "YEM"})


def _blank(value) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value)) or value == ""


def income_for(iso3: Optional[str]) -> Optional[str]:
    """The World Bank income group SWEET holds for a country, or None."""
    if _blank(iso3):
        return None
    return INCOME_ID_BY_ISO3.get(str(iso3).upper())


def landfill_is_default(iso3: Optional[str], income_id: Optional[str] = None) -> bool:
    """True where a site of unknown type should be assumed a sanitary landfill.

    Args:
        iso3: The country's ISO 3166 alpha-3 code.
        income_id: The caller's own income group for the country, if it has
            one. A missing value (None, NaN or empty) falls back to
            :func:`income_for`, which reads the same World Bank table.

    Returns:
        False for an unknown or unplaceable country.
    """
    if _blank(iso3):
        return False
    iso3 = str(iso3).upper()
    if iso3 in DUMPSITE_COUNTRIES:
        return False
    if _blank(income_id):
        income_id = income_for(iso3)
    return (
        income_id == "HIC"
        or defaults_2019.region_lookup_iso3.get(iso3) in LANDFILL_DEFAULT_REGIONS
        or iso3 in LANDFILL_DEFAULT_COUNTRIES
    )


def default_site_type(iso3: Optional[str], income_id: Optional[str] = None) -> str:
    """``"Sanitary Landfill"`` or ``"Dumpsite"`` for a site whose type is unknown.

    See :func:`landfill_is_default` for the arguments.
    """
    return SANITARY_LANDFILL if landfill_is_default(iso3, income_id) else DUMPSITE
