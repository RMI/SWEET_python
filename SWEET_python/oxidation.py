"""Cover oxidation factor -- the single source of truth for SWEET's defaults.

The oxidation factor is the share of the methane leaving a site, after capture,
that bacteria in the cover oxidise before it reaches the air. Emissions are
multiplied by ``1 - oxidation``, so a change here moves every downstream number.

A site's default is set by its type and whether it captures gas: a site that
manages its gas is assumed to manage its cover too. The table used to be copied
literally into seven places across ``city_params``, ``landfill``, ``dst_common``
and ``defaults_2019``; do not reintroduce a literal. See MODEL_DECISIONS.md,
``oxidation``.

===  ====================  ===============  ============
idx  SWEET label           without capture  with capture
===  ====================  ===============  ============
0    landfill              0.10             0.22
1    controlled dumpsite   0.05             0.10
2    dumpsite (open dump)  0                0
===  ====================  ===============  ============

A dump upgraded to a landfill with capture in a site-tool or advanced DST
scenario gets ``OX_REMEDIATED_TO_LANDFILL`` rather than a landfill's 0.22.
"""

from typing import Dict, List

#: Keyed by the site-type names the DST paths and Landfill carry.
OX_NO_CAPTURE: Dict[str, float] = {
    "landfill": 0.1,
    "controlled_dumpsite": 0.05,
    "dumpsite": 0.0,
}
OX_WITH_CAPTURE: Dict[str, float] = {
    "landfill": 0.22,
    "controlled_dumpsite": 0.1,
    "dumpsite": 0.0,
}

#: A dump upgraded to an engineered landfill with gas capture.
OX_REMEDIATED_TO_LANDFILL = 0.18

#: The site-type strings the trace and city-site paths carry, to the keys above.
#: The same three types ``mcf.MCF_BY_SITE_TYPE_NAME`` keys.
SITE_TYPE_KEYS: Dict[str, str] = {
    "Sanitary Landfill": "landfill",
    "Controlled Dumpsite": "controlled_dumpsite",
    "Dumpsite": "dumpsite",
}

#: The two tables keyed by those strings, under the keys the city paths use.
OX_OPTIONS_BY_SITE_TYPE_NAME: Dict[str, Dict[str, float]] = {
    "ox_nocap": {name: OX_NO_CAPTURE[key] for name, key in SITE_TYPE_KEYS.items()},
    "ox_cap": {name: OX_WITH_CAPTURE[key] for name, key in SITE_TYPE_KEYS.items()},
}


def ox_options() -> Dict[str, Dict[str, float]]:
    """A fresh copy of both tables under the keys the DST methods use,
    ``{"ox_nocap": ..., "ox_cap": ...}``, for a method that keeps it on ``self``."""
    return {"ox_nocap": dict(OX_NO_CAPTURE), "ox_cap": dict(OX_WITH_CAPTURE)}
