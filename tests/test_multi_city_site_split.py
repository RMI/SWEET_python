"""A site several cities share is split by each city's own waste, not its population.

Climate TRACE links Brazil's sites to the cities that send waste there (SINIR's
``city_to_facility``: the % of each city's waste sent to each site). Turning a % of a
city's waste into a share of a site's needs the city's size. Population stood in for
it, and Brazil's collected waste per person runs from about 0.2 to 1.5 kg/day between
cities, so a town sharing a site with a city took several times its own waste.

The numbers are SINIR's for site 17521 (Acre) in 2019: intake 76,851 t. Bujari
(10,266 people, 684 t) sends it 100% of its waste, Rio Branco (407,319 people,
133,475 t) 57.11%. By population Bujari took 3,248 t of the site, 4.7 times its whole
waste.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from SWEET_python import city_params
from SWEET_python.city_params import _city_shares_of_site

YEARS = range(1970, 2051)
BUJARI, RIO_BRANCO = 1001, 1002
SITE_INTAKE_2019 = 76_851.0


def _rows(with_waste=True, rio_branco_waste=133_475.0):
    rows = pd.DataFrame(
        {
            "city_id": [BUJARI, RIO_BRANCO, BUJARI, RIO_BRANCO],
            "ctf_year": [2018, 2018, 2019, 2019],
            "ctf_percent_of_waste_sent": [100.0, 60.0, 100.0, 57.11],
            "city_population": [10_266.0, 407_319.0, 10_266.0, 407_319.0],
            "city_population_year": [2022, 2022, 2022, 2022],
            "city_waste_tonnes": [684.0, rio_branco_waste, 684.0, rio_branco_waste],
            "city_waste_year": [2019, 2019, 2019, 2019],
        }
    )
    if not with_waste:
        rows = rows.drop(columns=["city_waste_tonnes", "city_waste_year"])
    return rows


def test_a_shared_site_is_split_by_the_tonnes_each_city_sends():
    shares = _city_shares_of_site(_rows(), YEARS)
    bujari, rio_branco = 684.0 * 1.00, 133_475.0 * 0.5711
    assert shares.at[2019, BUJARI] == pytest.approx(bujari / (bujari + rio_branco))
    # Bujari takes its own 684 t of the site's intake, not 3,248 t.
    assert shares.at[2019, BUJARI] * SITE_INTAKE_2019 == pytest.approx(683, abs=1)
    assert shares.at[2019, RIO_BRANCO] * SITE_INTAKE_2019 == pytest.approx(76_168, abs=1)


def test_each_year_uses_that_years_percent_and_the_shares_sum_to_one():
    shares = _city_shares_of_site(_rows(), YEARS)
    assert shares.at[2018, BUJARI] == pytest.approx(684.0 / (684.0 + 133_475.0 * 0.60))
    assert np.allclose(shares.sum(axis=1), 1.0)
    # A year with no report takes each city's average %.
    average = 684.0 / (684.0 + 133_475.0 * (0.60 + 0.5711) / 2)
    assert shares.at[2030, BUJARI] == pytest.approx(average)


def test_without_waste_figures_the_split_is_by_population_as_before():
    shares = _city_shares_of_site(_rows(with_waste=False), YEARS)
    bujari, rio_branco = 10_266.0 * 1.00, 407_319.0 * 0.5711
    assert shares.at[2019, BUJARI] == pytest.approx(bujari / (bujari + rio_branco))
    assert shares.at[2019, BUJARI] * SITE_INTAKE_2019 == pytest.approx(3_248, abs=1)


@pytest.mark.parametrize("missing", [np.nan, 0.0])
def test_one_city_without_a_figure_puts_the_whole_site_on_population(missing):
    """Tonnes and people are never mixed at one site."""
    shares = _city_shares_of_site(_rows(rio_branco_waste=missing), YEARS)
    expected = _city_shares_of_site(_rows(with_waste=False), YEARS)
    pd.testing.assert_frame_equal(shares, expected)


def test_the_climate_trace_path_splits_through_the_shared_function():
    source = inspect.getsource(city_params.City.citysite_estimate_trace)
    assert "_city_shares_of_site(" in source
    assert "city_population" not in source


def _site_row():
    return pd.Series(
        {
            "asset_identifier": 17521,
            "iso3_country": "BRA",
            "type": "Sanitary Landfill",
            "income_id": "UMC",
            "annual_incoming_waste": SITE_INTAKE_2019,
            "incoming_waste_year": 2019.0,
            "site_open_year": 2005.0,
            "site_close_year": np.nan,
            "waste_depth": 10.0,
            "landfill_gas_collection": False,
            "gas_collection_efficiency": np.nan,
            "latitude": -9.97,
            "longitude": -67.81,
            "precipitation": 2000.0,
            "temperature": 25.0,
        }
    )


def _run(rows):
    pop_data = pd.DataFrame(
        {"growth_rate_historic": [1.01], "growth_rate_future": [1.01]}, index=["BRA"]
    )
    city = city_params.City("17521")
    city.citysite_estimate_trace(_site_row(), citysite_rows=rows, pop_data=pop_data)
    return city.baseline_parameters


def _site_waste(baseline):
    return sum(lf.waste_mass_df.sum(axis=1) for lf in baseline.landfills)


def test_the_site_keeps_its_intake_and_only_the_split_moves():
    by_waste = _run(_rows())
    by_population = _run(_rows(with_waste=False))
    assert _site_waste(by_waste).at[2019] > 0
    pd.testing.assert_series_equal(_site_waste(by_waste), _site_waste(by_population))

    share = by_waste.fraction_of_waste_df.at[2019, BUJARI]
    assert share == pytest.approx(684.0 / (684.0 + 133_475.0 * 0.5711))
    bujari = next(lf for lf in by_waste.landfills if lf.city_id == BUJARI)
    assert bujari.waste_mass_df.sum(axis=1).at[2019] == pytest.approx(
        share * _site_waste(by_waste).at[2019]
    )
