"""The methane correction factor table and its depth rule.

MCF multiplies methane generation linearly, so these values move every emissions
number SWEET produces. They had no test coverage before, which is how the copies
scattered through city_params drifted apart.
"""

import numpy as np
import pandas as pd
import pytest

import SWEET_python.mcf as mcf
from SWEET_python import dst_common


LANDFILL, CONTROLLED_DUMP, OPEN_DUMP = 0, 1, 2


def test_table_matches_ipcc_categories():
    assert mcf.MCF_BY_TYPE == [1.0, 0.6, 0.6]
    assert mcf.MCF_BY_SITE_TYPE_NAME == {
        "Sanitary Landfill": 1.0,
        "Controlled Dumpsite": 0.6,
        "Dumpsite": 0.6,
    }


def test_both_dump_types_share_the_uncategorised_value():
    # "Controlled dumpsite" is not an IPCC category and our inputs say nothing
    # about how well a site is run, so it takes the same uncategorised value as
    # an open dump. Converting one to the other is deliberately an MCF no-op.
    assert mcf.mcf_for_site(CONTROLLED_DUMP) == mcf.mcf_for_site(OPEN_DUMP)


@pytest.mark.parametrize("site_type", [CONTROLLED_DUMP, OPEN_DUMP])
@pytest.mark.parametrize("unknown", [None, np.nan, pd.NA])
def test_unknown_depth_is_uncategorised_not_shallow(site_type, unknown):
    # The whole point of the uncategorised row: absent depth must not be read as
    # a claim that the site is shallow.
    assert mcf.mcf_for_site(site_type, unknown) == mcf.MCF_UNCATEGORISED


@pytest.mark.parametrize("site_type", [CONTROLLED_DUMP, OPEN_DUMP])
def test_supplied_depth_selects_the_specific_ipcc_category(site_type):
    assert mcf.mcf_for_site(site_type, 12.0) == mcf.MCF_UNMANAGED_DEEP
    assert mcf.mcf_for_site(site_type, 2.0) == mcf.MCF_UNMANAGED_SHALLOW
    # The threshold is strict, so exactly 5 m reads as shallow.
    assert mcf.mcf_for_site(site_type, 5.0) == mcf.MCF_UNMANAGED_SHALLOW
    assert mcf.mcf_for_site(site_type, 5.01) == mcf.MCF_UNMANAGED_DEEP


@pytest.mark.parametrize("depth", [None, np.nan, 0.5, 40.0])
def test_engineered_landfill_ignores_depth(depth):
    assert mcf.mcf_for_site(LANDFILL, depth) == mcf.MCF_MANAGED_ANAEROBIC


@pytest.mark.parametrize("depth", [0.0, 3.0, 5.0])
def test_untrusted_shallow_depth_falls_back_to_uncategorised(depth):
    # The DST request always carries a number for depth, so a shallow reading
    # there would apply 0.4 to every default scenario rather than to the
    # genuinely shallow sites.
    assert (
        mcf.mcf_for_site(OPEN_DUMP, depth, trust_shallow_depth=False)
        == mcf.MCF_UNCATEGORISED
    )


def test_untrusted_depth_still_takes_the_deep_bump():
    assert (
        mcf.mcf_for_site(OPEN_DUMP, 12.0, trust_shallow_depth=False)
        == mcf.MCF_UNMANAGED_DEEP
    )


def test_numpy_integer_site_type_resolves():
    assert mcf.mcf_for_site(np.int64(OPEN_DUMP)) == mcf.MCF_UNCATEGORISED


def test_dst_common_reexports_stay_in_sync():
    # advanced_dst and the WasteMAP backend import these names from dst_common.
    assert dst_common.MCF_BY_TYPE is mcf.MCF_BY_TYPE
    assert dst_common.DEEP_SITE_DEPTH_M == mcf.DEEP_SITE_DEPTH_M
    assert dst_common.DEEP_DUMP_MCF == mcf.MCF_UNMANAGED_DEEP
    assert dst_common.DEEP_MCF_DUMP_TYPES == mcf.DEPTH_SENSITIVE_TYPES


def test_mcf_series_splices_at_implement_year():
    years = pd.Index(range(2020, 2026))
    series = dst_common.mcf_series(OPEN_DUMP, LANDFILL, 2023, years)
    assert series.loc[2022] == mcf.MCF_UNCATEGORISED
    assert series.loc[2023] == mcf.MCF_MANAGED_ANAEROBIC
    assert series.loc[2025] == mcf.MCF_MANAGED_ANAEROBIC


def test_mcf_series_reads_a_supplied_shallow_depth():
    # adst's depth argument defaults to None, so a number there is a real answer
    # and the shallow category applies.
    years = pd.Index(range(2020, 2026))
    series = dst_common.mcf_series(
        OPEN_DUMP, OPEN_DUMP, 2023, years, baseline_depth=2.0, scenario_depth=2.0
    )
    assert (series == mcf.MCF_UNMANAGED_SHALLOW).all()
