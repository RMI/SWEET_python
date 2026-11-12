"""A composition surveyed after diversion gets the diverted waste added back.

Seattle's composition comes from a survey of its residential garbage at the transfer
stations, which says it "measured waste disposal, not generation". Read as generated
waste, its recycled materials come out twice: Seattle recycles 58.0% of its waste, 46.1%
of its surveyed composition is material SWEET can recycle, and SWEET stops with a
CustomError.

`load_andre_params(..., composition_after_diversion=True)` adds the diverted waste back
first (`after_diversion.composition_before_diversion`). SWEET's diversion from the result
has to leave exactly the surveyed composition, the tonnage must not move, and SWEET
still landfills its rejects on top, as it does for every city.

The fixture is six cities' rows from WasteMAP's cities table (dev database,
2026-11-10), as `load_andre_params` receives them. All six report diverting more than
their mix holds, which makes them good inputs for the switch. Whether a city's survey
was really taken after diversion is WasteMAP's call: it reads only Seattle's and
Bristol's that way, because adding the diversion back makes the other four's mixes less
like their countries' (WasteMAP's check_composition_after_diversion.py).
"""

import contextlib
import io
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SWEET_python.after_diversion import composition_before_diversion
from SWEET_python.city_params import CONTROLLED_DUMPSITE_LANDFILL, City, CustomError
from SWEET_python.constants import DIVERSION_ELIGIBILITY, WASTE_TYPES

FIXTURE = Path(__file__).parent / "fixtures" / "city_rows_surveyed_after_diversion.json"
CITIES = json.loads(FIXTURE.read_text())
PATHWAYS = ("compost", "anaerobic", "combustion", "recycling")


def _row(name):
    saved = CITIES[name]
    # JSON has no NaN: a missing value comes back as None, which SWEET reads as NaN.
    data = [np.nan if v is None else v for v in saved["row"]["data"]]
    return pd.Series(data, index=saved["row"]["index"], name=saved["row"]["name"])


def _load(name, row=None, **flags):
    city = City(name)
    with contextlib.redirect_stdout(io.StringIO()):
        city.load_andre_params(
            _row(name) if row is None else row,
            income_id=CITIES[name]["income_id"],
            **flags,
        )
    return city


def _surveyed(row):
    """The row's composition, as shares of 1, in SWEET's waste-type names."""
    columns = {
        "food": "composition_food_organic_waste_percent",
        "green": "composition_yard_garden_green_waste_percent",
        "wood": "composition_wood_percent",
        "paper_cardboard": "composition_paper_cardboard_percent",
        "textiles": "composition_textiles_percent",
        "plastic": "composition_plastic_percent",
        "metal": "composition_metal_percent",
        "glass": "composition_glass_percent",
        "rubber": "composition_rubber_leather_percent",
        "other": "composition_other_percent",
    }
    shares = {w: np.nan_to_num(float(row[c])) for w, c in columns.items()}
    total = sum(shares.values())
    return {w: s / total for w, s in shares.items()}


def _left_after_diversion(generated, diverted):
    """SWEET's own diversion: each pathway takes its share from its types, in proportion."""
    left = dict(generated)
    for path, share in diverted.items():
        types = [w for w in generated if w in DIVERSION_ELIGIBILITY[path]]
        pool = sum(generated[w] for w in types)
        for w in types:
            left[w] -= share * generated[w] / pool
    return left


@pytest.mark.parametrize("name", sorted(CITIES))
def test_as_reported_the_city_diverts_more_than_its_survey_holds(name):
    """Why WasteMAP skipped these six."""
    with pytest.raises(CustomError, match="can be at most"):
        _load(name)


@pytest.mark.parametrize("name", sorted(CITIES))
def test_with_the_diversion_added_back_the_city_models(name):
    city = _load(name, composition_after_diversion=True)
    baseline = city.baseline_parameters
    assert not baseline.input_problems
    # The composition holds every diverted tonne, so no pathway was cut or moved.
    assert not baseline.adjusted_diversion_constituents
    assert len(baseline.landfills) == CONTROLLED_DUMPSITE_LANDFILL + 1
    assert (pd.DataFrame(baseline.net_masses) >= 0).all().all()
    for landfill in baseline.landfills:
        landfill.estimate_emissions()
    city.estimate_diversion_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=0, simple=True)
    assert baseline.total_emissions.loc[2025].sum() > 0


@pytest.mark.parametrize("name", sorted(CITIES))
def test_diverting_from_the_composition_leaves_the_survey(name):
    """Checked on the objects SWEET built: its composition and its pathways' shares of it.
    This is what each pathway takes before its rejects, which the next test covers."""
    city = _load(name, composition_after_diversion=True)
    baseline = city.baseline_parameters
    year = baseline.waste_fractions.index[0]
    generated = baseline.waste_fractions.loc[year]
    diverted = baseline.div_fractions.loc[year]
    left = generated.copy()
    for path in PATHWAYS:
        shares = getattr(baseline.div_component_fractions, path).loc[year]
        left -= (diverted[path] * shares).reindex(left.index, fill_value=0.0)
    surveyed = pd.Series(_surveyed(_row(name)))
    expected = (1 - diverted.sum()) * surveyed.reindex(left.index)
    assert np.allclose(left.to_numpy(), expected.to_numpy(), atol=1e-9)
    assert generated.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("name", sorted(CITIES))
def test_sweet_also_landfills_the_rejects(name):
    """The landfills get the survey's waste plus each pathway's rejects (`diversion-rejects`):
    the recycling and compost residue a survey of garbage trucks does not see."""
    city = _load(name, composition_after_diversion=True)
    baseline = city.baseline_parameters
    year = baseline.waste_fractions.index[0]
    generated = float(baseline.waste_generated_df.loc[year].sum())
    taken = float(baseline.div_fractions.loc[year].sum())
    surveyed = pd.Series(_surveyed(_row(name)))
    landfilled = pd.DataFrame(baseline.net_masses).loc[year]
    survey_part = (1 - taken) * generated * surveyed.reindex(landfilled.index)
    rejects = landfilled - survey_part
    assert (rejects > -1e-6 * generated).all()
    assert rejects.sum() > 0
    if name == "Seattle":
        assert landfilled.sum() / generated == pytest.approx(0.515, abs=5e-4)


def test_a_flagged_survey_with_no_food_or_green_loads():
    """The compost emission factor needs food or green; with neither it is 0, as unflagged."""
    row = _row("Seattle").copy()
    for column in ("composition_food_organic_waste_percent", "composition_other_percent",
                   "composition_metal_percent"):
        row[column] = np.nan
    row["composition_paper_cardboard_percent"] = 50.0
    row["composition_plastic_percent"] = 30.0
    row["composition_glass_percent"] = 20.0
    row["waste_treatment_recycling_percent"] = 20.0
    row["waste_treatment_controlled_landfill_percent"] = 80.0
    city = _load("Seattle", row=row, composition_after_diversion=True)
    assert city.baseline_parameters.mef_compost == 0


def test_diversion_scaled_to_a_hair_under_100_percent_is_refused():
    """10 + 30 + 70 scales to 0.9999999999999999, not 1: nothing is left to survey."""
    row = _row("Seattle").copy()
    row["waste_treatment_compost_percent"] = 10.0
    row["waste_treatment_incineration_percent"] = 30.0
    row["waste_treatment_recycling_percent"] = 70.0
    with pytest.raises(ValueError, match="none was left to survey"):
        _load("Seattle", row=row, composition_after_diversion=True)


MIXED = {
    "food": 0.3, "green": 0.1, "wood": 0.05, "paper_cardboard": 0.2, "textiles": 0.05,
    "plastic": 0.15, "metal": 0.05, "glass": 0.05, "rubber": 0.02, "other": 0.03,
}


def _left_after_diverting(generated, shares):
    pools = {p: sum(generated[w] for w in DIVERSION_ELIGIBILITY[p] if w in generated) for p in shares}
    left = {
        w: generated[w] - sum(shares[p] * generated[w] / pools[p] for p in shares if w in DIVERSION_ELIGIBILITY[p])
        for w in generated
    }
    total = sum(left.values())
    return {w: share / total for w, share in left.items()}


def test_diversion_of_all_but_a_millionth_still_solves():
    """Overlapping pathways that take almost everything used to run out of passes."""
    shares = {"compost": 0.999994, "recycling": 0.000005}
    generated = composition_before_diversion(MIXED, shares)
    assert sum(generated.values()) == pytest.approx(1, abs=1e-12)
    left = _left_after_diverting(generated, shares)
    assert max(abs(left[w] - MIXED[w]) for w in MIXED) < 1e-6


def test_a_type_rounding_cannot_resolve_is_refused_by_name():
    """A sliver of wood that compost and digestion must have taken 91% of the waste from."""
    surveyed = {"wood": 0.0042, "textiles": 0.317, "glass": 0.278, "rubber": 0.139, "other": 0.262}
    shares = {"compost": 0.30574032247, "anaerobic": 0.60408096140, "recycling": 0.09017770613}
    with pytest.raises(RuntimeError, match="almost all diverted"):
        composition_before_diversion(surveyed, shares)


@pytest.mark.parametrize("name", ["Seattle", "Canberra"])
def test_the_tonnage_is_not_touched(name):
    """Both surveyed their tonnage in their population year, so the loader keeps it as is."""
    row = _row(name)
    city = _load(name, composition_after_diversion=True)
    assert city.baseline_parameters.waste_mass.iloc[0] == pytest.approx(
        float(row["msw_generated_metric_tons_per_year"])
    )


def test_seattle_gets_back_the_recyclables_it_recycled():
    generated = _load("Seattle", composition_after_diversion=True).baseline_parameters.waste_fractions.iloc[0]
    # 58.04% recycled, from the recyclable 46.1% of the 41.96% landfilled: paper goes
    # from 20.3% of the landfilled waste to 34.1% of the generated waste.
    assert generated["paper_cardboard"] == pytest.approx(0.341, abs=5e-4)
    assert generated["food"] == pytest.approx(0.2257, abs=5e-4)


def test_a_city_with_no_composition_of_its_own_is_refused():
    row = _row("Seattle").copy()
    for column in row.index:
        if column.startswith("composition_"):
            row[column] = np.nan
    with pytest.raises(ValueError, match="no usable composition"):
        _load("Seattle", row=row, composition_after_diversion=True)


def test_without_diversion_the_survey_is_what_the_city_generates():
    surveyed = {"food": 2.0, "paper_cardboard": 1.0, "glass": 1.0}
    assert composition_before_diversion(surveyed, {}) == pytest.approx(
        {"food": 0.5, "paper_cardboard": 0.25, "glass": 0.25}
    )


def test_incineration_alone_leaves_the_mix_as_surveyed():
    """An incinerator takes every type in proportion, so the mix it leaves is the mix it got."""
    surveyed = {"food": 0.5, "paper_cardboard": 0.3, "glass": 0.2}
    assert composition_before_diversion(surveyed, {"combustion": 0.9}) == pytest.approx(surveyed)


def test_any_mix_of_diversions_round_trips():
    rng = random.Random(20261110)
    for _ in range(200):
        surveyed = {w: rng.random() for w in WASTE_TYPES}
        total = sum(surveyed.values())
        surveyed = {w: s / total for w, s in surveyed.items()}
        cut = sorted(rng.random() * 0.97 for _ in range(len(PATHWAYS)))
        bounds = [0.0, *cut]
        diverted = {p: bounds[i + 1] - bounds[i] for i, p in enumerate(PATHWAYS)}
        generated = composition_before_diversion(surveyed, diverted)
        assert sum(generated.values()) == pytest.approx(1.0)
        left = _left_after_diversion(generated, diverted)
        kept = 1 - sum(diverted.values())
        assert left == pytest.approx({w: kept * s for w, s in surveyed.items()}, abs=1e-9)


def test_diverting_everything_leaves_nothing_to_survey():
    with pytest.raises(ValueError, match="none was left to survey"):
        composition_before_diversion({"food": 1.0}, {"compost": 0.6, "combustion": 0.4})


def test_a_pathway_whose_types_were_all_taken_cannot_be_split():
    """Nothing recyclable left in the landfill says nothing about which recyclables went."""
    with pytest.raises(ValueError, match="none of the types it takes"):
        composition_before_diversion({"food": 0.7, "green": 0.3}, {"recycling": 0.4})
