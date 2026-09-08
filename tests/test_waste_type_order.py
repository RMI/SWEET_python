"""Model output frames carry their waste-type columns in one fixed order.

The eligibility collections on ``City`` (``components``, ``div_components``)
were plain ``set``s, and a ``set`` of ``str`` iterates in hash order, which
Python randomizes per process (PEP 456). They are also what the engines iterate
to build DataFrame columns, so the same code on the same inputs produced a
different column order in every process::

    PYTHONHASHSEED=0  ['food', 'paper_cardboard', 'textiles', 'green', 'wood']
    PYTHONHASHSEED=1  ['paper_cardboard', 'food', 'green', 'wood', 'textiles']

Values were unaffected -- except for the derived ``total`` column, which is a
sum ACROSS those columns, and floating-point addition is not associative: 307
of 297,806 dumped numbers moved by exactly 1 ULP depending on the seed.

``WasteTypeSet`` keeps set semantics and pins iteration to ``WASTE_TYPES``.
These tests pin both halves: the type's own contract, and the column order of
real model output.
"""

import copy
import json
import os
import subprocess
import sys
import textwrap

import pytest

from SWEET_python.class_defs import WasteFractions, WasteMasses
from SWEET_python.constants import WASTE_TYPES, WasteTypeSet
from SWEET_python.city_params import City, DiversionFractions


# The five degradable types the FOD engine models, in canonical order.
DEGRADABLE = ["food", "green", "wood", "paper_cardboard", "textiles"]


# --------------------------------------------------------------------------- #
# The canonical order itself
# --------------------------------------------------------------------------- #
def test_canonical_order_matches_the_pydantic_models():
    """WASTE_TYPES is the field order of WasteFractions / WasteMasses.

    Frames built from a ``model_dump()`` (every ``divs_df``) take their order
    from those models; frames built from a component collection take it from
    ``WASTE_TYPES``. If the two drift apart the package is back to two column
    orders, so pin them together.
    """
    assert WASTE_TYPES == tuple(WasteFractions.model_fields)
    assert WASTE_TYPES == tuple(WasteMasses.model_fields)


def test_canonical_order_is_the_expected_ten_types():
    assert WASTE_TYPES == (
        "food",
        "green",
        "wood",
        "paper_cardboard",
        "textiles",
        "plastic",
        "metal",
        "glass",
        "rubber",
        "other",
    )


# --------------------------------------------------------------------------- #
# WasteTypeSet
# --------------------------------------------------------------------------- #
class TestWasteTypeSet:
    def test_iterates_in_canonical_order_whatever_the_construction_order(self):
        forwards = WasteTypeSet(DEGRADABLE)
        backwards = WasteTypeSet(reversed(DEGRADABLE))
        assert list(forwards) == DEGRADABLE
        assert list(backwards) == DEGRADABLE

    def test_every_iteration_protocol_sees_the_same_order(self):
        s = WasteTypeSet(DEGRADABLE)
        assert list(s) == DEGRADABLE
        assert tuple(s) == tuple(DEGRADABLE)
        assert [w for w in s] == DEGRADABLE
        # The dict-comprehension form the model uses to build frames.
        assert list({w: 0.0 for w in s}) == DEGRADABLE

    def test_the_order_is_not_alphabetical(self):
        """Guards against 'fixing' this with sorted(), which is a different order."""
        assert list(WasteTypeSet(DEGRADABLE)) != sorted(DEGRADABLE)

    def test_still_behaves_as_a_set(self):
        s = WasteTypeSet(DEGRADABLE)
        assert "food" in s
        assert "plastic" not in s
        assert len(s) == 5
        assert s == set(DEGRADABLE)
        assert s == frozenset(DEGRADABLE)

    @pytest.mark.parametrize(
        "op",
        [
            lambda a, b: a & b,
            lambda a, b: a | b,
            lambda a, b: a - b,
            lambda a, b: a ^ b,
            lambda a, b: a.intersection(b),
            lambda a, b: a.union(b),
            lambda a, b: a.difference(b),
            lambda a, b: a.symmetric_difference(b),
        ],
    )
    def test_set_operations_stay_ordered(self, op):
        """frozenset's operators return the base class; ours must not."""
        result = op(WasteTypeSet(DEGRADABLE), WasteTypeSet(["food", "green", "metal"]))
        assert isinstance(result, WasteTypeSet)
        assert list(result) == [w for w in WASTE_TYPES if w in result]

    def test_mixing_with_a_plain_set_keeps_the_order_on_the_left(self):
        """Python resolves the LEFT operand's __and__ first, and set's wins.

        So ``ordered & plain`` is ordered and ``plain & ordered`` is not --
        a language rule, not something the type can override. Documented on
        WasteTypeSet; asserted here so the asymmetry is not a surprise.
        """
        s = WasteTypeSet(DEGRADABLE)
        assert isinstance(s & {"green", "food"}, WasteTypeSet)
        assert not isinstance({"green", "food"} & s, WasteTypeSet)

    def test_unknown_members_sort_after_the_known_ones(self):
        s = WasteTypeSet(["zebra", "other", "food", "aardvark"])
        assert list(s) == ["food", "other", "aardvark", "zebra"]

    def test_survives_deepcopy(self):
        """The DST deep-copies baseline parameters to build a scenario."""
        s = copy.deepcopy(WasteTypeSet(DEGRADABLE))
        assert isinstance(s, WasteTypeSet)
        assert list(s) == DEGRADABLE


# --------------------------------------------------------------------------- #
# The City collections
# --------------------------------------------------------------------------- #
class TestCityCollections:
    def test_components_and_eligibility_are_ordered_sets(self):
        """A plain set here is the bug; catch it at the source, not downstream."""
        city = City("x")
        assert isinstance(city.components, WasteTypeSet)
        for div, eligible in city.div_components.items():
            assert isinstance(eligible, WasteTypeSet), div

    def test_components_iterate_in_canonical_order(self):
        city = City("x")
        assert list(city.components) == DEGRADABLE
        assert list(city.div_components["compost"]) == [
            "food",
            "green",
            "wood",
            "paper_cardboard",
        ]
        assert list(city.div_components["recycling"]) == [
            "wood",
            "paper_cardboard",
            "textiles",
            "plastic",
            "metal",
            "glass",
            "rubber",
            "other",
        ]

    def test_waste_types_list_is_the_canonical_order(self):
        assert City("x").waste_types == list(WASTE_TYPES)


# --------------------------------------------------------------------------- #
# Real model output
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dst_run():
    """One city-DST run: blank-city baseline plus a four-stream scenario."""
    city = City("x")
    city.dst_baseline_blank("Algeria", 2_594_000, 716.81, 18.38)
    city.implement_dst_changes_simple_v1_5(
        DiversionFractions(compost=0.20, anaerobic=0.10, combustion=0.10, recycling=0.20),
        0,
        0,
        0.0,
        0.0,
        2026,
        1,
        0.10,
    )
    return city


class TestModelOutputColumnOrder:
    def test_landfill_emissions_frames_are_pinned(self, dst_run):
        """The frame in the bug report: a landfill's per-waste-type CH4."""
        landfill = dst_run.baseline_parameters.landfills[0]
        assert list(landfill.ch4.columns) == DEGRADABLE
        assert list(landfill.captured.columns) == DEGRADABLE
        assert list(landfill.waste_mass_after_degredation.columns) == DEGRADABLE
        # `emissions` carries the derived cross-column sum as a final column.
        assert list(landfill.emissions.columns) == DEGRADABLE + ["total"]

    def test_divs_frames_are_pinned(self, dst_run):
        """Diverted mass per stream -- each stream's eligible types, in order.

        The baseline frames are built from ``div_components``, so they carry
        only the eligible types: this is the path the hash seed used to move.
        """
        divs = dst_run.baseline_parameters.divs_df
        assert list(divs.compost.columns) == ["food", "green", "wood", "paper_cardboard"]
        assert list(divs.anaerobic.columns) == ["food", "green", "wood", "paper_cardboard"]
        assert list(divs.combustion.columns) == list(WASTE_TYPES)
        assert list(divs.recycling.columns) == [
            "wood",
            "paper_cardboard",
            "textiles",
            "plastic",
            "metal",
            "glass",
            "rubber",
            "other",
        ]

    def test_both_divs_construction_paths_agree_on_the_order(self, dst_run):
        """The scenario frames come from a pydantic ``model_dump()`` instead.

        That path was always deterministic -- but it ordered columns by the
        model's field order while the eligibility path ordered them by hash.
        Now both are WASTE_TYPES, which is the point of pinning the canonical
        order to the pydantic models.
        """
        divs = dst_run.scenario_parameters[0].divs_df
        for div in ("compost", "anaerobic", "combustion", "recycling"):
            assert list(getattr(divs, div).columns) == list(WASTE_TYPES), div

    def test_component_fraction_frames_are_pinned(self, dst_run):
        fracs = dst_run.baseline_parameters.div_component_fractions
        for div in ("compost", "anaerobic", "combustion", "recycling"):
            cols = list(getattr(fracs, div).columns)
            assert cols == [w for w in WASTE_TYPES if w in set(cols)], div

    def test_every_waste_type_frame_follows_the_canonical_order(self, dst_run):
        """The general invariant, over every frame the two runs expose."""
        checked = 0
        for params in (dst_run.baseline_parameters, dst_run.scenario_parameters[0]):
            for landfill in params.landfills:
                for attr in ("ch4", "captured", "emissions", "waste_mass_after_degredation"):
                    frame = getattr(landfill, attr, None)
                    if frame is None:
                        continue
                    cols = [c for c in frame.columns if c != "total"]
                    assert cols == [w for w in WASTE_TYPES if w in set(cols)], (attr, cols)
                    checked += 1
        assert checked > 0

    def test_waste_mass_df_is_alphabetical_by_index_union(self, dst_run):
        """A KNOWN second order, deliberately left alone -- not the bug.

        ``waste_mass_df`` is ``waste_generated_df - DivsDF.sum()``, and
        ``sum()`` builds its columns with ``Index.union``, which sorts. That is
        deterministic, so it was never part of the hash-seed problem, and those
        lines are being edited by an open PR. Pinned here so the remaining
        inconsistency is visible rather than folklore; it affects nothing
        downstream, because every consumer slices this frame by name.
        """
        frame = dst_run.baseline_parameters.landfills[0].waste_mass_df
        assert list(frame.columns) == sorted(WASTE_TYPES)


# --------------------------------------------------------------------------- #
# The property itself: same inputs, different process, same columns
# --------------------------------------------------------------------------- #
_CHILD = textwrap.dedent(
    """
    import json
    from SWEET_python.city_params import City, DiversionFractions

    city = City("x")
    city.dst_baseline_blank("Algeria", 2_594_000, 716.81, 18.38)
    city.implement_dst_changes_simple_v1_5(
        DiversionFractions(compost=0.20, anaerobic=0.10, combustion=0.10, recycling=0.20),
        0, 0, 0.0, 0.0, 2026, 1, 0.10)

    baseline = city.baseline_parameters
    scenario = city.scenario_parameters[0]
    print(json.dumps({
        "components": list(city.components),
        "ch4": list(baseline.landfills[0].ch4.columns),
        "emissions": list(baseline.landfills[0].emissions.columns),
        "divs_compost": list(scenario.divs_df.compost.columns),
        "divs_recycling": list(scenario.divs_df.recycling.columns),
        # The derived total is a sum ACROSS the columns, so its last bit
        # followed the column order. repr() round-trips a float exactly.
        "total_1971": repr(float(baseline.landfills[0].emissions["total"].iloc[1])),
    }))
    """
)


def _columns_under_hash_seed(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def test_column_order_does_not_depend_on_the_hash_seed():
    """The regression itself. Fails on the pre-fix code, whatever the seeds.

    Two subprocesses because PYTHONHASHSEED is read once at interpreter start:
    it cannot be varied in-process.
    """
    assert _columns_under_hash_seed(0) == _columns_under_hash_seed(1)
