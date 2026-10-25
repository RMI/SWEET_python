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

# A site remediated in a scenario captures less than one built to its new type:
# the gas system is retrofitted into waste already in place. A controlled dumpsite
# assumes 45%; a dumpsite remediated to a controlled dumpsite, 30%; a controlled
# dumpsite remediated to a sanitary landfill, 50%. See MODEL_DECISIONS.md,
# gas-capture-efficiency.

#: A dumpsite remediated to a controlled dumpsite: the city DST's "add gas
#: capture" does this to its dumpsite share.
CONVERTED_DUMP_CAPTURE_EFFICIENCY = 0.3

#: A controlled dumpsite remediated to a sanitary landfill. A dumpsite taken
#: straight to a sanitary landfill gets this too: the rule goes by the new type.
UPGRADED_CONTROLLED_DUMP_CAPTURE_EFFICIENCY = 0.5

#: The same, indexed by the LandfillType a site is remediated to.
REMEDIATED_CAPTURE_EFFICIENCY_BY_TYPE: List[float] = [
    UPGRADED_CONTROLLED_DUMP_CAPTURE_EFFICIENCY,
    CONVERTED_DUMP_CAPTURE_EFFICIENCY,
    GAS_CAPTURE_EFFICIENCY["dumpsite"],
]


def default_capture_efficiency(baseline_type: int, scenario_type: int) -> float:
    """The capture a site gets when no efficiency is stated, by LandfillType.

    A scenario type better (lower) than the baseline's is a remediation, and takes
    the remediated value; otherwise the site gets its type's default. For a
    baseline, pass the same type twice.
    """
    if scenario_type < baseline_type:
        return REMEDIATED_CAPTURE_EFFICIENCY_BY_TYPE[scenario_type]
    return GAS_CAPTURE_EFFICIENCY_BY_TYPE[scenario_type]
