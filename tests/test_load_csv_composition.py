"""`City.load_csv_new` has to model a composition that adds up to 100%.

WasteMAP's city DST loads a city from its published cities table with
`load_csv_new`. Brazil's precomputed rows carry their source composition as it
is: São Paulo's ten `Waste Components` sum to 99%, Brasília's to 87%, and the
national default the other ~4,960 Brazilian municipalities get sums to 100.1%.
Every other loader here normalizes a composition; `load_csv_new` did not, so:

- `implement_dst_changes_simple_v1_5` rejected São Paulo and Brasília outright.
  Its `|sum - 1| > 1e-2` check fails Brasília by 13 points and São Paulo by
  float rounding (1 - 0.99 = 0.010000000000000009), so every scenario request
  for Brazil's two largest cities was a 400 "Invalid waste fractions".
- The other Brazilian cities ran, but their typed masses missed or exceeded
  the city's waste, and the no-change scenario drifted from the baseline.
"""

import pandas as pd
import pytest

from SWEET_python.city_params import City, DiversionFractions
from SWEET_python.constants import WASTE_TYPES

IMPLEMENT_YEAR = 2026

LABELS = {
    "food": "Food",
    "green": "Green",
    "wood": "Wood",
    "paper_cardboard": "Paper and Cardboard",
    "textiles": "Textiles",
    "plastic": "Plastic",
    "metal": "Metal",
    "glass": "Glass",
    "rubber": "Rubber/Leather",
    "other": "Other",
}

# The waste types each diversion takes, for the columns `load_csv_new` reads.
# `make_cities_table` derives a city's diversion components as its composition's
# shares within each set, which reproduces the published values.
DIVERTED = {
    ("Composted", "Composted"): ["food", "green", "wood", "paper_cardboard"],
    ("Anaerobically Digested", "Digested"): ["food", "green", "wood", "paper_cardboard"],
    ("Incinerated", "Incinerated"): [
        "food", "green", "wood", "paper_cardboard", "textiles", "plastic", "rubber",
    ],
    ("Recycled", "Recycled"): [
        "wood", "paper_cardboard", "textiles", "plastic", "metal", "glass", "rubber", "other",
    ],
}

# Compositions as the June 2026 cities table publishes them, in percent.
SAO_PAULO = {"food": 47.0, "paper_cardboard": 16.0, "plastic": 10.0, "metal": 2.0,
             "glass": 1.0, "rubber": 1.0, "other": 22.0}  # 99%
BRASILIA = {"food": 53.0, "paper_cardboard": 6.0, "plastic": 5.0, "metal": 1.0,
            "glass": 4.0, "other": 18.0}  # 87%
BRAZIL_DEFAULT = {"food": 53.5, "paper_cardboard": 17.6, "plastic": 17.5, "metal": 2.2,
                  "glass": 4.0, "other": 5.3}  # 100.1%, e.g. Salvador

# São Paulo's and Brasília's other inputs, as published.
CITIES = {
    "São Paulo": dict(
        composition=SAO_PAULO, population=10_000_000.0, waste=4_700_000.0,
        historic=3.26, future=0.70, precip=1417.67, temperature=19.39,
        compost=3.0, recycling=0.0, landfill=100.0, dumpsite=0.0,
    ),
    "Brasília": dict(
        composition=BRASILIA, population=4_800_000.0, waste=1_199_862.0,
        historic=4.47, future=0.60, precip=1416.11, temperature=21.64,
        compost=18.2, recycling=23.1, landfill=50.0, dumpsite=50.0,
    ),
}


def _cities_table_row(name, composition, *, population, waste, historic, future,
                      precip, temperature, compost, recycling, landfill, dumpsite):
    """One city-year of WasteMAP's cities table, cut to the columns `load_csv_new` reads."""
    percent = {waste_type: composition.get(waste_type, 0.0) for waste_type in WASTE_TYPES}
    row = {
        "City": name,
        "Country ISO3": "BRA",
        # Written as "True", as make_cities_table does for every Brazil row.
        "Uses Sites Method": True,
        "Population": population,
        "Year of Data Collection (Population)": 2022,
        "Population Growth Rate: Historic (%)": historic,
        "Population Growth Rate: Future (%)": future,
        "Average Annual Precipitation (mm/year)": precip,
        "Precipitation Zone": "Moderately Wet",
        "Temperature (C)": temperature,
        "Waste Generation Rate (tons/year)": waste,
        "Waste Generation Rate per Capita (kg/person/day)": waste / population * 1000 / 365,
        "Data Source (Waste Mass)": "SINIR 2014",
        "MEF: Compost": 0.004243855,
        "Methane Capture Efficiency (%)": 0.0,
        "Percent of Waste to Landfills with Gas Capture (%)": 0.0,
        "Percent of Waste to Landfills without Gas Capture (%)": landfill,
        "Percent of Waste to Dumpsites (%)": dumpsite,
        "Diversions: Compost (%)": compost,
        "Diversions: Anaerobic Digestion (%)": 0.0,
        "Diversions: Incineration (%)": 0.0,
        "Diversions: Recycling (%)": recycling,
    }
    row.update({f"Waste Components: {LABELS[w]} (%)": percent[w] for w in WASTE_TYPES})
    for (verb, total), taken in DIVERTED.items():
        pool = sum(percent[w] for w in taken)
        row.update(
            {
                f"Diversion Components: {verb} {LABELS[w]} (% of Total {total})": percent[w] / pool * 100
                for w in taken
            }
        )
    return pd.DataFrame([row])


def _load(name, composition=None):
    """What WasteMAP's `load_city` does with a cities-table row."""
    inputs = dict(CITIES.get(name, CITIES["São Paulo"]))
    if composition is not None:
        inputs["composition"] = composition
    city = City(name)
    city.load_csv_new(_cities_table_row(name, **inputs), dst=True)
    city._calculate_divs()
    for landfill in city.baseline_parameters.landfills:
        landfill.estimate_emissions()
    city.estimate_diversion_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=0)
    return city


CASES = pytest.mark.parametrize(
    "name, composition",
    [
        ("São Paulo", SAO_PAULO),
        ("Brasília", BRASILIA),
        # São Paulo's inputs with the default most Brazilian cities publish.
        ("a city on Brazil's default", BRAZIL_DEFAULT),
    ],
    ids=["sao-paulo-99pct", "brasilia-87pct", "brazil-default-100.1pct"],
)


@CASES
def test_the_composition_is_normalized(name, composition):
    city = _load(name, composition)
    fractions = city.baseline_parameters.waste_fractions.iloc[0]
    published = pd.Series(composition).reindex(fractions.index, fill_value=0.0)

    # Each type keeps its published share of the reported total.
    pd.testing.assert_series_equal(
        fractions, published / published.sum(), check_names=False, rtol=1e-12
    )
    # So the typed masses add up to the city's waste: none goes missing.
    masses = city.baseline_parameters.waste_masses.model_dump()
    inputs = CITIES.get(name, CITIES["São Paulo"])
    assert sum(masses.values()) == pytest.approx(inputs["waste"], rel=1e-12)


@CASES
def test_no_change_reproduces_the_baseline(name, composition):
    city = _load(name, composition)
    baseline = city.baseline_parameters.total_emissions["total"].copy()
    unchanged = city.baseline_parameters.div_fractions.iloc[0]

    # Raised "Invalid waste fractions" for São Paulo and Brasília; drifted
    # from the baseline for the default composition.
    city.implement_dst_changes_simple_v1_5(
        DiversionFractions(**unchanged.to_dict()), 0, 0, 0.0, 0.0, IMPLEMENT_YEAR, 1
    )
    scenario = city.scenario_parameters[0].total_emissions["total"]
    pd.testing.assert_series_equal(scenario, baseline, rtol=1e-9)


def test_a_composition_that_adds_up_loads_unchanged():
    # Every non-Brazil row of the cities table already sums to 100%, so
    # normalizing must leave its fractions exactly as published.
    adds_up = {"food": 50.0, "paper_cardboard": 25.0, "plastic": 12.5, "other": 12.5}
    city = _load("São Paulo", adds_up)
    fractions = city.baseline_parameters.waste_fractions.iloc[0]
    published = pd.Series(adds_up).reindex(fractions.index, fill_value=0.0) / 100
    assert (fractions == published).all()
