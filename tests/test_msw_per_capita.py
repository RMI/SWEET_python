"""Per-capita waste generation defaults (`defaults_2019.msw_per_capita_country`).

The table is IPCC 2019 Refinement, Vol. 5 Ch. 2, Table 2A.1 (Updated), 2010
column, in tonnes per person per year, converted to kg per person per day. Every
row matched the IPCC's except one: Indonesia, which the IPCC lists at 0.19, had
no row. A Custom Location or a cities-table city in Indonesia with no waste
figure of its own was sized from the South-Eastern Asia default instead, 0.46,
2.4 times the country's own row.
"""

import numpy as np
import pytest

from SWEET_python import defaults_2019
from SWEET_python.city_params import City

_POP = 1_000_000


def _kg_per_day(tonnes_per_year):
    return tonnes_per_year / 365 * 1000


def test_indonesia_has_its_own_row():
    assert defaults_2019.msw_per_capita_country["IDN"] == pytest.approx(_kg_per_day(0.19))


def test_every_south_eastern_asian_country_the_ipcc_lists_has_a_row():
    # Table 2A.1's South-Eastern Asia block. Cambodia and Timor-Leste are not in
    # it, so they take the regional default.
    ipcc_rows = {
        "BRN": 0.32, "IDN": 0.19, "LAO": 0.26, "MYS": 0.55, "MMR": 0.16,
        "PHL": 0.18, "SGP": 1.28, "THA": 0.64, "VNM": 0.53,
    }
    for iso3, tonnes_per_year in ipcc_rows.items():
        assert defaults_2019.msw_per_capita_country[iso3] == pytest.approx(
            _kg_per_day(tonnes_per_year)
        ), iso3


def test_a_custom_location_in_indonesia_is_sized_from_indonesias_row():
    city = City("custom")
    city.dst_baseline_blank("Indonesia", _POP, 2000.0, 27.0)

    # Was 460,000: the South-Eastern Asia default, 0.46 t per person per year.
    assert np.allclose(city.baseline_parameters.waste_mass, 0.19 * _POP)


def test_a_country_without_a_row_still_takes_its_regions_default():
    city = City("custom")
    city.dst_baseline_blank("Cambodia", _POP, 2000.0, 27.0)

    assert np.allclose(city.baseline_parameters.waste_mass, 0.46 * _POP)
