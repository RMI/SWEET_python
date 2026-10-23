"""Each modelling number has one named home, and the old names still reach it.

The oxidation table was written out in seven places, the compost factor six times
and GWP 28 as eight literals. A change to one copy could silently miss the others,
and Climate TRACE and WasteMAP import some of these tables by their old names. These
tests pin the homes, the aliases, and that no copy has crept back in.
"""

import ast
import re
from pathlib import Path

import pytest

import SWEET_python.defaults_2019 as defaults_2019
import SWEET_python.dst_allocation as dst_allocation
import SWEET_python.dst_common as dst_common
import SWEET_python.gas_capture as gas_capture
import SWEET_python.landfill as landfill
import SWEET_python.oxidation as oxidation
from SWEET_python import constants
from SWEET_python.city_params import GAS_EFF_OPTIONS, OX_OPTIONS, City

PACKAGE = Path(constants.__file__).parent


# --------------------------------------------------------------------------- #
# The old names other repositories import are the same objects
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "old",
    [
        # Climate TRACE
        "region_lookup_iso3",
        "waste_fraction_defaults",
        "landfill_default_regions",
        "gas_capture_efficiency",
        "oxidation_factor",
        # WasteMAP
        "waste_fractions_country",
        "country_to_iso3",
    ],
)
def test_lower_case_names_alias_the_upper_case_tables(old):
    assert getattr(defaults_2019, old) is getattr(defaults_2019, old.upper())


def test_waste_composition_for_is_still_importable():
    # Climate TRACE imports the function by this name; functions keep theirs.
    from SWEET_python.defaults_2019 import waste_composition_for  # noqa: F401


def test_re_exported_names_are_the_homes():
    assert dst_common.OX_NOCAP is oxidation.OX_NO_CAPTURE
    assert dst_common.OX_CAP is oxidation.OX_WITH_CAPTURE
    assert dst_common.OX_REMEDIATED_TO_LANDFILL == oxidation.OX_REMEDIATED_TO_LANDFILL
    assert dst_common.DEFAULT_FLARE_EFFICIENCY == constants.DEFAULT_FLARE_EFFICIENCY
    assert dst_common.MODEL_YEAR_MIN == constants.MODEL_YEAR_MIN
    assert dst_common.DEGRADABLE_COMPONENTS == list(constants.DEGRADABLE_TYPES)
    assert dst_common.WASTE_COMPONENTS == list(constants.WASTE_TYPES)
    assert landfill.CH4_DENSITY_KG_PER_M3 == constants.CH4_DENSITY_KG_PER_M3
    assert OX_OPTIONS is oxidation.OX_OPTIONS_BY_SITE_TYPE_NAME
    assert GAS_EFF_OPTIONS is gas_capture.GAS_CAPTURE_EFFICIENCY_BY_SITE_TYPE_NAME
    assert defaults_2019.GAS_CAPTURE_EFFICIENCY is gas_capture.GAS_CAPTURE_EFFICIENCY
    assert defaults_2019.CH4_TO_CO2E == constants.GWP100_CH4


def test_the_site_type_name_tables_match_the_keyed_ones():
    for name, key in oxidation.SITE_TYPE_KEYS.items():
        assert OX_OPTIONS["ox_nocap"][name] == oxidation.OX_NO_CAPTURE[key]
        assert OX_OPTIONS["ox_cap"][name] == oxidation.OX_WITH_CAPTURE[key]
        assert GAS_EFF_OPTIONS[name] == gas_capture.GAS_CAPTURE_EFFICIENCY[key]
    assert gas_capture.GAS_CAPTURE_EFFICIENCY_BY_TYPE == [
        gas_capture.GAS_CAPTURE_EFFICIENCY[k]
        for k in ("landfill", "controlled_dumpsite", "dumpsite")
    ]


def test_defaults_2019_oxidation_table_is_built_from_the_home():
    ox = defaults_2019.OXIDATION_FACTOR
    for key in ("landfill", "controlled_dumpsite", "dumpsite"):
        assert ox["with_lfg"][key] == oxidation.OX_WITH_CAPTURE[key]
        assert ox["without_lfg"][key] == oxidation.OX_NO_CAPTURE[key]


# --------------------------------------------------------------------------- #
# A City gets its own copies
# --------------------------------------------------------------------------- #

def test_a_city_carries_copies_of_the_diversion_tables():
    a, b = City("a"), City("b")
    assert a.div_components == constants.DIVERSION_ELIGIBILITY
    assert a.unprocessable == constants.UNPROCESSABLE_SHARE
    assert a.non_compostable_not_targeted == constants.NON_COMPOSTABLE_NOT_TARGETED
    assert a.recycling_reject_rates == constants.RECYCLING_YIELD
    assert a.combustion_reject_rate == constants.COMBUSTION_REJECT_RATE
    # Changing one city's table must not change another's, or the constant.
    a.recycling_reject_rates["wood"] = 0.0
    a.div_components["compost"] = constants.WasteTypeSet({"food"})
    assert b.recycling_reject_rates["wood"] == constants.RECYCLING_YIELD["wood"] == 0.8
    assert b.div_components["compost"] == constants.DIVERSION_ELIGIBILITY["compost"]


def test_dst_allocation_eligibility_is_the_home_as_plain_sets():
    assert dst_allocation.DEFAULT_ELIGIBILITY == {
        k: set(v) for k, v in constants.DIVERSION_ELIGIBILITY.items()
    }


# --------------------------------------------------------------------------- #
# The compost factor gives what its six copies gave, to the bit
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("food,green", [(0.5, 0.1), (0.3, 0.3), (1.0, 0.0), (0.0, 1.0), (0.123, 0.456)])
def test_compost_factor_is_the_formula_it_replaced(food, green):
    written_out = (
        0.0055 * food / (food + green) + 0.0139 * green / (food + green)
    ) * 1.1023 * 0.7
    assert defaults_2019.compost_emission_factor(food, green) == written_out


# --------------------------------------------------------------------------- #
# No copy creeps back in
# --------------------------------------------------------------------------- #

# Numbers that now live in exactly one place, and the file that place is in.
_HOMES = {
    "0.0055": "defaults_2019.py",
    "0.0139": "defaults_2019.py",
    "1.1023": "constants.py",
    "0.7168": "constants.py",
    "0.22": "oxidation.py",
}


def _code_numbers(path: Path):
    """Every numeric literal in a file's code, not its comments or docstrings."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if not isinstance(node.value, bool):
                yield node.value, node.lineno


def test_no_module_writes_out_a_number_that_has_a_home():
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        for value, line in _code_numbers(path):
            for literal, home in _HOMES.items():
                # defaults_2019's country tables hold shares, some of which are 0.22.
                if path.name in (home, "defaults_2019.py") and literal == "0.22":
                    continue
                if path.name != home and value == float(literal):
                    offenders.append(f"{path.relative_to(PACKAGE)}:{line} writes {literal}")
    assert not offenders, "use the named constant:\n" + "\n".join(offenders)


def test_no_module_divides_or_multiplies_by_a_literal_28():
    pattern = re.compile(r"[*/]\s*28\b")
    offenders = [
        f"{path.relative_to(PACKAGE)}:{i}"
        for path in PACKAGE.rglob("*.py")
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line.split("#", 1)[0])
    ]
    assert not offenders, "use constants.GWP100_CH4:\n" + "\n".join(offenders)
