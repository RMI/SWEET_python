"""How much of a country's landfilled waste goes to landfills that capture their gas.

A city or Custom Location with no landfill shares of its own takes its country's or
region's disposal split (``defaults_2019.disposal_split_for``, ``City.load_andre_params``).
That split comes from IPCC 2019 Table 2.1, which says how much waste a country landfills
and dumps, but not how much of the landfilled waste goes to sites with a gas collection
system. Until this module, every defaulted landfill was modelled without capture: a city
in the US or the EU emitted as if no landfill there collected its gas.

``capture_share`` gives the share of a country's landfilled waste that goes to landfills
with an active gas collection system:

1. the country's own share, where research gives one (``CAPTURE_SHARE_COUNTRY``);
2. else ``HIGH_INCOME_SHARE`` (0.5) for a World Bank high-income country, the main test
   of SWEET's site-type rule (``site_type_defaults.landfill_is_default``): gas-collection
   rules track it (the EU Landfill Directive, US federal rules, Japan, Korea);
3. else none, as before.

``with_default_capture`` moves that share of a defaulted split's "landfill without
capture" to "landfill with capture"; the capture bucket then captures
``gas_capture.GAS_CAPTURE_EFFICIENCY["landfill"]`` (60%) of its methane, as a reported
"sanitary landfill with landfill gas system" share does. The dumpsite share is untouched.
A city that reports its own landfill shares keeps them (``city-treatment-buckets``).

**Provisional.** Only four countries have a researched share, and two of those are
derived rather than published. The 0.5 is a placeholder, not a finding: the Gulf states
and Russia are high income but capture little, and upper-middle-income countries with
real capture (Turkey, Mexico, Chile before 2023...) get none. Each country needs its own
researched share before this is relied on. See MODEL_DECISIONS.md,
``default-landfill-capture-share``.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Optional

__all__ = [
    "CAPTURE_SHARE_COUNTRY",
    "HIGH_INCOME_SHARE",
    "CaptureShare",
    "capture_share",
    "with_default_capture",
]


class CaptureShare(NamedTuple):
    #: Share of landfilled waste going to landfills with gas collection, 0-1.
    share: float
    #: The year the evidence describes.
    year: int
    #: What the share is a share of: "mass landfilled", "methane generated"...
    basis: str
    source: str
    url: str


#: The share a high-income country without a researched share of its own takes.
#: A placeholder, Hugh 2026-11-19: "a default of 0.5 where we don't have serious research".
HIGH_INCOME_SHARE = 0.5

# decision: default-landfill-capture-share
CAPTURE_SHARE_COUNTRY: dict[str, CaptureShare] = {
    # Derived, not published: GHGRP 2021 emissions at MSW landfills without a collection
    # system (21.3 Mt CO2e) against those with one (61.8), back-calculated to methane
    # generated (~90% at sites with a system), then scaled for the ~11% of generation at
    # landfills that don't report (assumed uncollected). Cross-checks: the Inventory
    # recovers 60% of generated methane (2021); 845 of 1,125 GHGRP landfills had a system
    # (2023).
    "USA": CaptureShare(
        0.85, 2021, "methane generated (derived)",
        "US EPA GHG Inventory 1990-2021, ch. 7, Table 7-3; EPA GHGRP waste sector profile 2011-2023, Tables 8-9",
        "https://www.epa.gov/system/files/documents/2024-12/waste_profile_ry23.pdf",
    ),
    # "66% of MSW disposed at landfills with LFG recovery systems" (ECCC Landfill Gas
    # Inventory, about 2023); 37% of methane generated is recovered.
    "CAN": CaptureShare(
        0.66, 2023, "mass landfilled",
        "ECCC, Landfill Methane Regulations Overview (June 2026), citing the ECCC Landfill Gas Inventory",
        "https://acuerdochilecanada.mma.gob.cl/wp-content/uploads/2026/06/ECCC-Presentation-Landfill-Methane-Regulations-Overview-June-2026.pdf",
    ),
    # 3,350 kt to municipal landfills with capture, 279 kt without (2019). Municipal
    # landfills only: non-municipal and farm fills are left out.
    "NZL": CaptureShare(
        0.92, 2019, "mass landfilled",
        "Climate Change Commission (2021), Supporting Evidence ch. 8 (waste), Table 8.1",
        "https://www.climatecommission.govt.nz/assets/Evidence-21/Evidence-CH-8-reducing-emissions-waste.pdf",
    ),
    # Derived: 45% of methane generated is captured nationally (5-year inventory mean), and
    # a site with capture takes about 75% of its own gas, so 0.45 / 0.75. Secondary source.
    "AUS": CaptureShare(
        0.6, 2022, "methane generated (derived)",
        "Macintosh et al. (ANU, 2024), quoting DCCEEW (2024) ACCU landfill gas method reform options paper",
        "https://law.anu.edu.au/files/2024-06/Reform%20of%20ACCU%20Landfill%20Method%20-%20a%20critique%20070623%20%28final%29.pdf",
    ),
}


def capture_share(
    iso3: Optional[str],
    income_id: Optional[str] = None,
    shares: Optional[Mapping[str, CaptureShare]] = None,
) -> float:
    """The share of *iso3*'s landfilled waste that goes to landfills with gas capture.

    *income_id* is the country's World Bank income group ('HIC', 'UMC', ...), from the
    caller, as for ``site_type_defaults.landfill_is_default``; without one, only a
    researched share applies.
    """
    shares = CAPTURE_SHARE_COUNTRY if shares is None else shares
    row = shares.get(str(iso3).upper()) if iso3 else None
    if row is not None:
        return float(row.share)
    if isinstance(income_id, str) and income_id == "HIC":
        return HIGH_INCOME_SHARE
    return 0.0


def with_default_capture(
    split: Mapping[str, float],
    iso3: Optional[str],
    income_id: Optional[str] = None,
    shares: Optional[Mapping[str, CaptureShare]] = None,
) -> dict:
    """*split* (a defaulted disposal split) with the country's capture share of its
    "landfill without capture" moved to "landfill with capture". The total is unchanged."""
    out = dict(split)
    share = capture_share(iso3, income_id, shares)
    if share <= 0:
        return out
    moved = out.get("landfill_wo_capture", 0.0) * share
    out["landfill_w_capture"] = out.get("landfill_w_capture", 0.0) + moved
    out["landfill_wo_capture"] = out.get("landfill_wo_capture", 0.0) - moved
    return out
