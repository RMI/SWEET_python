"""Default gas-capture efficiency -- the single source of truth for SWEET.

The share of the methane a site generates that its collection system captures,
used when a site is known to have a system but reports no efficiency. It was
copied into six places across ``city_params`` and ``defaults_2019``; import it
instead. See MODEL_DECISIONS.md, ``gas-capture-efficiency``.

===  ====================  ====
idx  SWEET label           eff.
===  ====================  ====
0    landfill              0.6
1    controlled dumpsite   0.45
2    dumpsite (open dump)  0
===  ====================  ====
"""

from typing import Dict, List

#: Keyed by the site-type names the DST paths and Landfill carry.
GAS_CAPTURE_EFFICIENCY: Dict[str, float] = {
    "landfill": 0.6,
    "controlled_dumpsite": 0.45,
    "dumpsite": 0.0,
}

#: The same, keyed by the site-type strings the trace and city-site paths carry.
GAS_CAPTURE_EFFICIENCY_BY_SITE_TYPE_NAME: Dict[str, float] = {
    "Sanitary Landfill": GAS_CAPTURE_EFFICIENCY["landfill"],
    "Controlled Dumpsite": GAS_CAPTURE_EFFICIENCY["controlled_dumpsite"],
    "Dumpsite": GAS_CAPTURE_EFFICIENCY["dumpsite"],
}

#: The same, indexed by LandfillType (0 landfill, 1 controlled dump, 2 dumpsite).
GAS_CAPTURE_EFFICIENCY_BY_TYPE: List[float] = [
    GAS_CAPTURE_EFFICIENCY["landfill"],
    GAS_CAPTURE_EFFICIENCY["controlled_dumpsite"],
    GAS_CAPTURE_EFFICIENCY["dumpsite"],
]

#: The city DST's "add gas capture" turns a dumpsite into a controlled dumpsite
#: capturing this much. Not the 0.45 a controlled dump otherwise defaults to; see
#: MODEL_DECISIONS.md, gas-capture-efficiency.
CONVERTED_DUMP_CAPTURE_EFFICIENCY = 0.3

#: What the site tool would give a site upgraded in its scenario, by LandfillType,
#: when the user states no efficiency. Unreached today: the site tool refuses a
#: blank efficiency before it gets here.
UPGRADED_SITE_CAPTURE_EFFICIENCY_BY_TYPE: List[float] = [0.5, 0.3, 0.0]
