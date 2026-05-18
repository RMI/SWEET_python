import pandas as pd
import pytest

from SWEET_python.model_v2 import SWEET


def _run_model(open_date: int):
    years = pd.Index(range(open_date, 2051))
    components = ["food", "green", "wood", "paper_cardboard", "textiles"]
    landfill_attrs = {
        "open_date": open_date,
        "close_date": 2050,
        "ks": {component: pd.Series(0.2, index=years) for component in components},
        "waste_mass_df": pd.DataFrame(1000.0, index=years, columns=components),
        "mcf": pd.Series(1.0, index=years),
        "gas_capture_efficiency": pd.Series(0.0, index=years),
        "oxidation_factor": pd.Series(0.1, index=years),
        "flaring": pd.Series(0.98, index=years),
    }

    model = SWEET(
        city_instance_attrs={"components": components},
        city_params_dict={
            "year_of_data_pop": 2025,
            "growth_rate_historic": 1.0,
            "growth_rate_future": 1.0,
        },
        landfill_instance_attrs=landfill_attrs,
    )
    _, emissions, _, _ = model.estimate_emissions2()
    return emissions


def test_annual_model_keeps_1990_start_behavior():
    emissions = _run_model(1990)

    assert emissions.index[0] == 1990
    assert emissions.loc[1990, "total"] == pytest.approx(0.0)
    assert emissions.loc[1991, "total"] == pytest.approx(107657.55599791845)


def test_annual_model_uses_pre_1990_waste_history():
    emissions = _run_model(1982)

    assert emissions.index[0] == 1982
    assert emissions.loc[1982, "total"] == pytest.approx(0.0)
    assert emissions.loc[1990, "total"] == pytest.approx(474001.3640763213)
    assert emissions.loc[1990, "total"] > _run_model(1990).loc[1990, "total"]
