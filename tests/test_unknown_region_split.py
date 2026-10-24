"""A place in a region IPCC gives no landfill/dump split takes the site-type rule.

IPCC 2019 Refinement Table 2.1 (Updated) has no open-dump or landfill fraction for Central
Asia or Rest of Oceania. SWEET used to fill both with 100% dumped, so every city there with
no split of its own was modelled as an open dump, although `site_type_defaults` (the rule
for any site of unknown type, which Climate TRACE uses too) calls Central Asia landfill.
The rows are gone; such a place now falls through to the rule.

In the June 2026 cities table this moves eight cities from dumpsite to landfill (Dushanbe,
Tashkent, Bishkek, …), +50% methane in 2025. Fiji, Samoa and the other Pacific islands stay
dumps, because the rule says so.
"""

import pytest

from SWEET_python import defaults_2019, site_type_defaults


@pytest.mark.parametrize("region", ["Central Asia", "Rest of Oceania"])
def test_ipcc_gives_these_regions_no_split(region):
    assert region not in defaults_2019.fraction_open_dumped
    assert region not in defaults_2019.fraction_landfilled


@pytest.mark.parametrize(
    "iso3, region",
    [("UZB", "Central Asia"), ("TJK", "Central Asia"), ("KGZ", "Central Asia"),
     ("PLW", "Rest of Oceania"), ("FJI", "Rest of Oceania"), ("VUT", "Rest of Oceania")],
)
def test_a_custom_location_there_follows_the_site_type_rule(iso3, region):
    if iso3 in defaults_2019.fraction_open_dumped_country:
        pytest.skip(f"{iso3} has a country row of its own")
    split = defaults_2019.disposal_split_for(iso3, region)
    landfill = site_type_defaults.landfill_is_default(iso3)

    assert split["landfill_w_capture"] == 0.0
    assert split["landfill_wo_capture"] == (1.0 if landfill else 0.0)
    assert split["dumpsite"] == (0.0 if landfill else 1.0)


def test_central_asia_is_landfill_by_the_rule():
    """The point of the change: the rule and the old filled row disagreed here."""
    assert site_type_defaults.landfill_is_default("UZB")
    assert defaults_2019.disposal_split_for("UZB", "Central Asia")["landfill_wo_capture"] == 1.0
