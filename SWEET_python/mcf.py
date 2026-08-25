"""Methane correction factor (MCF) — the single source of truth for SWEET.

MCF is the fraction of the degradable carbon deposited at a site that decomposes
anaerobically. It multiplies modelled methane generation linearly, so a change
here moves every downstream emissions number by the same proportion.

Every MCF in SWEET resolves through :func:`mcf_for_site`. The table used to be
copied literally into ten call sites across ``city_params`` and ``dst_common``,
which is how the values drifted apart; do not reintroduce a literal.

Site types
----------
SWEET types a site as one of three (the index is the ``LandfillType`` value):

===  ====================  ====
idx  SWEET label           MCF
===  ====================  ====
0    landfill              1.0
1    controlled dumpsite   0.6
2    dumpsite (open dump)  0.6
===  ====================  ====

Index 0 is the IPCC "managed - anaerobic" category, 1.0.

Indices 1 and 2 both take the IPCC "uncategorised SWDS" default of 0.6. IPCC
2006 Vol. 5 Ch. 3 Table 3.1 splits unmanaged sites by waste depth - deeper than
``DEEP_SITE_DEPTH_M`` is "unmanaged deep" (0.8), shallower is "unmanaged
shallow" (0.4) - and prescribes 0.6 for a site whose depth is unknown. Waste
depth is unpopulated for every dump and controlled dump in the data we model,
so 0.6 is the category that actually applies, not a compromise between the
other two.

A controlled dumpsite carries that same 0.6 rather than a value of its own.
"Controlled dumpsite" is a SWEET/WasteMAP label, not an IPCC category, and it
does not map onto one: the IPCC rows a partly-managed site could plausibly sit
in span a wide range, and our inputs record nothing about how well any
individual site is run. "Uncategorised" is the honest reading of what we know.

Consequence worth knowing: MCF no longer distinguishes a controlled dump from
an open dump, so converting one to the other changes modelled generation only
through cover oxidation and gas capture, which SWEET models separately.

Depth
-----
Where a depth *is* supplied for a dump it selects the specific IPCC category
instead of the uncategorised default. A depth of ``None`` or ``NaN`` means
"unknown", not "shallow", and keeps the per-type value - that distinction is
the whole point of the uncategorised row. Depth never applies to an engineered
landfill (index 0), which is already fully anaerobic at 1.0.

Note that the deep threshold is applied as a strict ``>``, so a site recorded at
exactly 5.0 m reads as shallow. IPCC words the category as ">= 5 m". The
difference only matters for a site whose depth is recorded as exactly 5, and the
strict comparison is what SWEET has always used.
"""

from typing import Dict, List, Optional

import pandas as pd

# IPCC 2006 Vol. 5 Ch. 3 Table 3.1.
MCF_MANAGED_ANAEROBIC = 1.0
MCF_UNMANAGED_DEEP = 0.8
MCF_UNCATEGORISED = 0.6
MCF_UNMANAGED_SHALLOW = 0.4

DEEP_SITE_DEPTH_M = 5.0

LANDFILL_TYPE = 0
CONTROLLED_DUMPSITE_TYPE = 1
DUMPSITE_TYPE = 2
# Only the two dump types read a depth. An engineered landfill is 1.0 regardless.
DEPTH_SENSITIVE_TYPES = (CONTROLLED_DUMPSITE_TYPE, DUMPSITE_TYPE)

# Indexed by LandfillType.
MCF_BY_TYPE: List[float] = [
    MCF_MANAGED_ANAEROBIC,
    MCF_UNCATEGORISED,
    MCF_UNCATEGORISED,
]

SITE_TYPE_NAMES: List[str] = ["landfill", "controlled_dumpsite", "dumpsite"]

# The same table keyed by the site-type strings the site/city estimate paths
# carry, rather than by LandfillType index.
MCF_BY_SITE_TYPE_NAME: Dict[str, float] = {
    "Sanitary Landfill": MCF_MANAGED_ANAEROBIC,
    "Controlled Dumpsite": MCF_UNCATEGORISED,
    "Dumpsite": MCF_UNCATEGORISED,
}


def mcf_for_site(
    site_type_idx: int,
    depth: Optional[float] = None,
    *,
    trust_shallow_depth: bool = True,
) -> float:
    """MCF for one site, given its type and (optionally) its waste depth.

    Args:
        site_type_idx: ``LandfillType`` value - 0 landfill, 1 controlled dump,
            2 open dump.
        depth: Waste depth in metres, or ``None``/``NaN`` when it is unknown.
            Unknown keeps the per-type value; it is not read as shallow.
        trust_shallow_depth: Whether a supplied depth at or below
            ``DEEP_SITE_DEPTH_M`` may be read as IPCC "unmanaged shallow" (0.4).
            Pass ``False`` for a caller that cannot represent "unknown depth"
            and therefore sends a placeholder number when it has no answer -
            for such a caller a shallow reading would silently apply 0.4 to
            every site rather than only to the genuinely shallow ones. The deep
            bump still applies either way.

    Returns:
        The methane correction factor, between 0 and 1.
    """
    if site_type_idx not in DEPTH_SENSITIVE_TYPES:
        return MCF_BY_TYPE[site_type_idx]
    if depth is None or pd.isna(depth):
        return MCF_BY_TYPE[site_type_idx]
    if depth > DEEP_SITE_DEPTH_M:
        return MCF_UNMANAGED_DEEP
    if trust_shallow_depth:
        return MCF_UNMANAGED_SHALLOW
    return MCF_BY_TYPE[site_type_idx]
