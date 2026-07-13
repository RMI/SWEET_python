import os
import sys
from pathlib import Path

sys.path.append("/app/SWEET_python/SWEET_python")
sys.path.append("/app/SWEET_python")

from geopy.extra.rate_limiter import RateLimiter
import ssl
import certifi
from pydantic import BaseModel, validator
from typing import List, Dict, Union, Any, Set, Optional
import pandas as pd
import numpy as np
import pycountry  # TODO: confirm whether this import is still needed.
from SWEET_python.class_defs import *
import copy
from geopy.geocoders import Nominatim
import asyncpg
import socket
from fastapi import HTTPException
from SWEET_python.landfill import Landfill
from SWEET_python.singapore_k import compute_singapore_k
import SWEET_python.defaults_2019 as defaults_2019
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError
from datetime import datetime
import time


def _build_oxidation_series(default_value, canonical_row, time_series_rows, years_range):
    """Per-year oxidation factor for one modeled landfill, preferring per-site input
    oxidation over the type/gas-capture default.

    A site can have several input rows from independent sources. Oxidation carries no
    year of its own in the source data -- the value that varies row-to-row is the
    *emissions* year (``reported_emissions_year``), not an oxidation year -- so we use
    the site-wide mean of the available input oxidation values as a constant baseline
    across all model years, then overwrite individual years with that year's value
    (mean of any conflicting same-year rows) where an emissions year is present.

    When the site has no usable input oxidation, fall back to ``default_value`` (the
    type/gas-capture default) broadcast across all years -- i.e. the prior behaviour.
    Input values are used as-is (no clamping); e.g. a measured 0.35 passes through.

    ``time_series_rows`` is a DataFrame for multi-row sites and a Series for single-row
    sites; ``canonical_row`` is the single deduped row. Either may carry ``oxidation``.
    """
    years_index = pd.Index(years_range)
    series = pd.Series(float(default_value), index=years_index)

    # Assemble the site's input rows as a frame so single-row (Series) and multi-row
    # (DataFrame) sites are handled uniformly. Prefer the multi-row frame -- it holds
    # every source record -- and fall back to the single canonical row.
    if isinstance(time_series_rows, pd.DataFrame):
        frame = time_series_rows
    elif isinstance(canonical_row, pd.DataFrame):
        frame = canonical_row
    elif isinstance(canonical_row, pd.Series):
        frame = canonical_row.to_frame().T
    else:
        frame = None

    # This default fallback is intentional and load-bearing for callers whose input
    # frame carries no per-year `oxidation` column -- e.g. the older city DST /
    # make_cities_table path, which predates per-year oxidation and supplies a single
    # baseline value elsewhere. The Climate TRACE sites AND cities pipelines both SELECT
    # `oxidation` into their multi-row frames (landfill_table_ops), so measured oxidation
    # IS used there; this only defaults when the column is genuinely absent. (Edge case,
    # not produced by any current caller: oxidation present only on canonical_row while a
    # DataFrame time_series_rows lacks it would be skipped here.)
    if frame is None or 'oxidation' not in frame.columns:
        return series

    ox = pd.to_numeric(frame['oxidation'], errors='coerce')
    if not ox.notna().any():
        return series  # no input oxidation -> keep the type/gas-capture default

    # 1) Site-wide mean as the full-series baseline.
    series[:] = float(ox[ox.notna()].mean())

    # 2) Overwrite individual years where an emissions year ties a value to a year.
    if 'reported_emissions_year' in frame.columns:
        yr = pd.to_numeric(frame['reported_emissions_year'], errors='coerce')
        mask = ox.notna() & yr.notna()
        if mask.any():
            per_year = pd.Series(ox[mask].to_numpy(), index=yr[mask].astype(int).to_numpy())
            per_year = per_year.groupby(level=0).mean()
            lo, hi = int(years_index.min()), int(years_index.max())
            per_year = per_year[(per_year.index >= lo) & (per_year.index <= hi)]
            if not per_year.empty:
                series.loc[per_year.index] = per_year.to_numpy()

    return series


# The way this model is set up is based on the unit of a City, corresponding to the City class.
# Cities can have multiple sets of CityParameters, one for each scenario.
# Sets of CityParameters can have one or more landfills, dumpsites, waste to energy, etc.
# Even for modeling a single landfill, City and CityParameters classes need to be used.
class CityParameters(BaseModel):
    waste_fractions: Optional[Union[pd.DataFrame, pd.Series]] = None  # WasteFractions
    div_fractions: Optional[pd.DataFrame] = None  # DiversionFractions
    split_fractions: Optional[SplitFractions] = None
    div_component_fractions: Optional[Union[DivComponentFractionsDF, pd.DataFrame, Dict[str, pd.DataFrame]]] = None
    precip: Optional[float] = None
    growth_rate_historic: Optional[float] = None
    growth_rate_future: Optional[float] = None
    waste_per_capita: Optional[Union[pd.Series, float]] = None
    precip_zone: Optional[str] = None
    ks: Optional[DecompositionRates] = None
    gas_capture_efficiency: Optional[pd.Series] = None  # float
    mef_compost: float | None = None
    waste_mass: Optional[pd.Series] = None  # float
    landfills: Optional[List[Landfill]] = None
    non_zero_landfills: Optional[List[Landfill]] = None
    non_compostable_not_targeted_total: Optional[pd.Series] = None
    waste_masses: WasteMasses | pd.DataFrame | None = None
    divs: DivMasses | DivMassesAnnual | None = None
    year_of_data_pop: Optional[Union[Dict[str, Any], int]] = None
    year_of_data_msw: Optional[int] = None
    scenario: Optional[int] = 0
    implement_year: Optional[int] = None
    organic_emissions: Optional[pd.DataFrame] = None
    landfill_emissions: Optional[pd.DataFrame] = None
    diversion_emissions: Optional[pd.DataFrame] = None
    total_emissions: Optional[pd.DataFrame] = None
    adjusted_diversion_constituents: Optional[bool] = False
    input_problems: Optional[bool] = False
    divs_df: Optional[pd.DataFrame] = None
    waste_generated_df: WasteGeneratedDF | None = None
    city_instance_attrs: Optional[Dict[str, Any]] = None
    population: Optional[float] = None
    temp: Optional[float] = None
    net_masses: Optional[pd.DataFrame] = None
    temperature: Optional[float] = None
    waste_burning_emissions: Optional[pd.DataFrame] = None
    source_pop: Optional[str] = None
    source_msw: Optional[str] = None
    defaults_used: Optional[Dict[str, bool]] = None
    rmi_id: Optional[int] = None
    sites_method: Optional[bool] = None
    sites_info_dict: Optional[Dict[str, Any]] = None
    bf: Optional[float] = None
    fraction_of_waste_df: Optional[pd.DataFrame] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_dump_for_serialization(self):
        data = self.model_dump()

        def convert_sets_to_lists(data):
            if isinstance(data, dict):
                return {k: convert_sets_to_lists(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_sets_to_lists(v) for v in data]
            elif isinstance(data, set):
                return list(data)
            elif isinstance(data, pd.DataFrame):
                return data.to_dict(orient="records")
            else:
                return data

        return convert_sets_to_lists(data)

    def repopulate_attr_dicts(self):
        city_params_dict = self.model_dump()
        keys_to_remove = ["landfills", "non_zero_landfills"]
        for key in keys_to_remove:
            if key in city_params_dict:
                del city_params_dict[key]

        if self.landfills is not None:
            for landfill in self.landfills:
                landfill.city_params_dict = city_params_dict
                if hasattr(landfill, "model"):
                    landfill.model.city_params_dict = city_params_dict
                    landfill.model.landfill_instance_attrs = landfill.model_dump()

    def _singapore_k(
        self, advanced_baseline=False, advanced_dst=False, implement_year=None, for_trace_reported_projections=False
    ) -> None:
        """
        Calculate k values using the method from Wang et al (2024).
        Delegates to SWEET_python.singapore_k.compute_singapore_k.

        Args:
            advanced_baseline (bool): Flag to indicate if advanced baseline calculations are needed.
            advanced_dst (bool): Flag to indicate if advanced diversion scenario calculations are needed.
            implement_year (int): Year when the diversion scenario is implemented.
            for_trace_reported_projections (bool): Flag for trace reported projection calculations.

        Returns:
            None
        """
        self.temp = self.temperature
        if self.temperature is None or pd.isna(self.temperature):
            asset = (self.city_instance_attrs or {}).get("city_name", self.rmi_id)
            print(
                f"WARNING: _singapore_k: missing temperature for asset {asset}; "
                f"decomposition rate k will be NaN (this shouldn't happen)"
            )
        self.ks, self.bf = compute_singapore_k(
            self.waste_fractions,
            self.temperature,
            self.precip,
            advanced_baseline=advanced_baseline,
            advanced_dst=advanced_dst,
            implement_year=implement_year,
            for_trace_reported_projections=for_trace_reported_projections,
        )

    def update_cityparams_dict(self) -> None:
        """
        Updates the city parameters dictionary with new values.

        Args:
            city_params_dict (dict): The dictionary containing the new values.

        Returns:
            None
        """
        city_params_dict = self.model_dump()
        keys_to_remove = ["landfills", "non_zero_landfills"]
        for key in keys_to_remove:
            if key in city_params_dict:
                del city_params_dict[key]

        if self.landfills is not None:
            for landfill in self.landfills:
                landfill.city_params_dict = city_params_dict
                if hasattr(landfill, "model"):
                    landfill.model.city_params_dict = city_params_dict

        return city_params_dict


class CustomError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(self.message)


class City:
    def __init__(self, city_name: str):
        """
        Initializes a new City instance.

        Args:
            city_name (str): The name of the city.
        """
        self.city_name = city_name
        self.country = None
        self.iso3 = None
        self.baseline_parameters = None
        self.scenario_parameters = {}
        self.components = {"food", "green", "wood", "paper_cardboard", "textiles"}
        self.div_components = {
            "compost": {"food", "green", "wood", "paper_cardboard"},
            "anaerobic": {"food", "green", "wood", "paper_cardboard"},
            "combustion": {
                "food",
                "green",
                "wood",
                "paper_cardboard",
                "textiles",
                "plastic",
                "rubber",
            },
            "recycling": {
                "wood",
                "paper_cardboard",
                "textiles",
                "plastic",
                "rubber",
                "metal",
                "glass",
                "other",
            },
        }
        self.waste_types = [
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
        ]
        self.unprocessable = {
            "food": 0.0192,
            "green": 0.042522,
            "wood": 0.07896,
            "paper_cardboard": 0.12,
        }
        self.non_compostable_not_targeted = {
            "food": 0.1,
            "green": 0.05,
            "wood": 0.05,
            "paper_cardboard": 0.1,
        }
        self.combustion_reject_rate = 0.1
        self.recycling_reject_rates = {
            "wood": 0.8,
            "paper_cardboard": 0.775,
            "textiles": 0.99,
            "plastic": 0.875,
            "metal": 0.955,
            "glass": 0.88,
            "rubber": 0.78,
            "other": 0.87,
        }
        self.latitude = None
        self.longitude = None
        self.years_range = range(1990, 2051)

    def load_from_csv(self, db: pd.DataFrame, scenario: int = 0) -> None:
        """
        DEPRECATED
        Loads model parameters from the RMI WasteMAP GitHub repo data file.

        Args:
            db (pd.DataFrame): DataFrame containing model parameters for all cities.
            scenario (str): The scenario name, defaults to 'baseline'.

        Returns:
            None
        """
        city_data = db.loc[self.city_name]

        self.country = city_data["Country ISO3"].values[0]

        waste_fractions = WasteFractions(
            food=city_data["Waste Components: Food (%)"].values[0] / 100,
            green=city_data["Waste Components: Green (%)"].values[0] / 100,
            wood=city_data["Waste Components: Wood (%)"].values[0] / 100,
            paper_cardboard=city_data[
                "Waste Components: Paper and Cardboard (%)"
            ].values[0]
            / 100,
            textiles=city_data["Waste Components: Textiles (%)"].values[0] / 100,
            plastic=city_data["Waste Components: Plastic (%)"].values[0] / 100,
            metal=city_data["Waste Components: Metal (%)"].values[0] / 100,
            glass=city_data["Waste Components: Glass (%)"].values[0] / 100,
            rubber=city_data["Waste Components: Rubber/Leather (%)"].values[0] / 100,
            other=city_data["Waste Components: Other (%)"].values[0] / 100,
        )

        div_fractions = DiversionFractions(
            compost=city_data["Diversions: Compost (%)"].values[0] / 100,
            anaerobic=city_data["Diversions: Anaerobic Digestion (%)"].values[0] / 100,
            combustion=city_data["Diversions: Incineration (%)"].values[0] / 100,
            recycling=city_data["Diversions: Recycling (%)"].values[0] / 100,
        )

        split_fractions = SplitFractions(
            landfill_w_capture=city_data[
                "Percent of Waste to Landfills with Gas Capture (%)"
            ].values[0]
            / 100,
            landfill_wo_capture=city_data[
                "Percent of Waste to Landfills without Gas Capture (%)"
            ].values[0]
            / 100,
            dumpsite=city_data["Percent of Waste to Dumpsites (%)"].values[0] / 100,
        )

        div_component_fractions = DivComponentFractions(
            compost=WasteFractions(
                food=city_data[
                    "Diversion Components: Composted Food (% of Total Composted)"
                ].values[0]
                / 100,
                green=city_data[
                    "Diversion Components: Composted Green (% of Total Composted)"
                ].values[0]
                / 100,
                wood=city_data[
                    "Diversion Components: Composted Wood (% of Total Composted)"
                ].values[0]
                / 100,
                paper_cardboard=city_data[
                    "Diversion Components: Composted Paper and Cardboard (% of Total Composted)"
                ].values[0]
                / 100,
                textiles=0,
                plastic=0,
                metal=0,
                glass=0,
                rubber=0,
                other=0,
            ),
            anaerobic=WasteFractions(
                food=city_data[
                    "Diversion Components: Anaerobically Digested Food (% of Total Digested)"
                ].values[0]
                / 100,
                green=city_data[
                    "Diversion Components: Anaerobically Digested Green (% of Total Digested)"
                ].values[0]
                / 100,
                wood=city_data[
                    "Diversion Components: Anaerobically Digested Wood (% of Total Digested)"
                ].values[0]
                / 100,
                paper_cardboard=city_data[
                    "Diversion Components: Anaerobically Digested Paper and Cardboard (% of Total Digested)"
                ].values[0]
                / 100,
                textiles=0,
                plastic=0,
                metal=0,
                glass=0,
                rubber=0,
                other=0,
            ),
            combustion=WasteFractions(
                food=city_data[
                    "Diversion Components: Incinerated Food (% of Total Incinerated)"
                ].values[0]
                / 100,
                green=city_data[
                    "Diversion Components: Incinerated Green (% of Total Incinerated)"
                ].values[0]
                / 100,
                wood=city_data[
                    "Diversion Components: Incinerated Wood (% of Total Incinerated)"
                ].values[0]
                / 100,
                paper_cardboard=city_data[
                    "Diversion Components: Incinerated Paper and Cardboard (% of Total Incinerated)"
                ].values[0]
                / 100,
                textiles=city_data[
                    "Diversion Components: Incinerated Textiles (% of Total Incinerated)"
                ].values[0]
                / 100,
                plastic=city_data[
                    "Diversion Components: Incinerated Plastic (% of Total Incinerated)"
                ].values[0]
                / 100,
                metal=0,
                glass=0,
                rubber=city_data[
                    "Diversion Components: Incinerated Rubber/Leather (% of Total Incinerated)"
                ].values[0]
                / 100,
                other=0,
            ),
            recycling=WasteFractions(
                wood=city_data[
                    "Diversion Components: Recycled Wood (% of Total Recycled)"
                ].values[0]
                / 100,
                paper_cardboard=city_data[
                    "Diversion Components: Recycled Paper and Cardboard (% of Total Recycled)"
                ].values[0]
                / 100,
                plastic=city_data[
                    "Diversion Components: Recycled Plastic (% of Total Recycled)"
                ].values[0]
                / 100,
                rubber=city_data[
                    "Diversion Components: Recycled Rubber/Leather (% of Total Recycled)"
                ].values[0]
                / 100,
                textiles=city_data[
                    "Diversion Components: Recycled Textiles (% of Total Recycled)"
                ].values[0]
                / 100,
                glass=city_data[
                    "Diversion Components: Recycled Glass (% of Total Recycled)"
                ].values[0]
                / 100,
                metal=city_data[
                    "Diversion Components: Recycled Metal (% of Total Recycled)"
                ].values[0]
                / 100,
                other=city_data[
                    "Diversion Components: Recycled Other (% of Total Recycled)"
                ].values[0]
                / 100,
                food=0,
                green=0,
            ),
        )

        # ks = DecompositionRates(
        #     food=city_data['k: Food'].values[0],
        #     green=city_data['k: Green'].values[0],
        #     wood=city_data['k: Wood'].values[0],
        #     paper_cardboard=city_data['k: Paper and Cardboard'].values[0],
        #     textiles=city_data['k: Textiles'].values[0]
        # )

        non_compostable_not_targeted_total = sum(
            [
                self.non_compostable_not_targeted[x]
                * getattr(div_component_fractions.compost, x)
                for x in self.div_components["compost"]
            ]
        )
        if np.isnan(non_compostable_not_targeted_total):
            non_compostable_not_targeted_total = 0

        gas_capture_efficiency = (
            city_data["Methane Capture Efficiency (%)"].values[0] / 100
        )
        mef_compost = city_data["MEF: Compost"].values[0]
        waste_mass = city_data["Waste Generation Rate (tons/year)"].values[0]

        year_of_data_pop = city_data["Year of Data Collection"].values[0]

        city_instance_attrs = {
            "city_name": self.city_name,
            "country": self.country,
            "components": self.components,
            "div_components": self.div_components,
            "waste_types": self.waste_types,
            "unprocessable": self.unprocessable,
            "non_compostable_not_targeted": self.non_compostable_not_targeted,
            "combustion_reject_rate": self.combustion_reject_rate,
            "recycling_reject_rates": self.recycling_reject_rates,
        }

        city_parameters = CityParameters(
            waste_fractions=waste_fractions,
            div_fractions=div_fractions,
            split_fractions=split_fractions,
            div_component_fractions=div_component_fractions,
            precip=float(city_data["Average Annual Precipitation (mm/year)"].values[0]),
            growth_rate_historic=city_data[
                "Population Growth Rate: Historic (%)"
            ].values[0]
            / 100
            + 1,
            growth_rate_future=city_data["Population Growth Rate: Future (%)"].values[0]
            / 100
            + 1,
            waste_per_capita=city_data[
                "Waste Generation Rate per Capita (kg/person/day)"
            ].values[0],
            precip_zone=city_data["Precipitation Zone"].values[0],
            # ks=ks,
            gas_capture_efficiency=gas_capture_efficiency,
            mef_compost=mef_compost,
            waste_mass=waste_mass,
            non_compostable_not_targeted_total=non_compostable_not_targeted_total,
            year_of_data_pop=year_of_data_pop,
            scenario=scenario,
            city_instance_attrs=city_instance_attrs,
            population=city_data["Population"].values[0],
        )

        # Filter out the 'landfills' and 'non_zero_landfills' attributes from CityParameters
        # city_params = {k: v for k, v in city_parameters.__dict__.items() if k not in ['landfills', 'non_zero_landfills']}
        # city_params = copy.deepcopy(city_parameters.__dict__)

        self.baseline_parameters = city_parameters

    def load_csv_new(self, db: pd.DataFrame, scenario: int = 0, dst: bool = False) -> None:
        """
        Loads model parameters from the RMI WasteMAP GitHub repo data file.
        This replaces the deprecated load_from_csv method.

        Args:
            db (pd.DataFrame): DataFrame containing model parameters for all cities.
            scenario (str): The scenario name, defaults to 'baseline'.

        Returns:
            None
        """
        city_data = db #.loc[self.city_name]

        self.country = city_data["Country ISO3"].values[0]

        # Define the range of years
        years = range(1990, 2051)

        waste_fractions = WasteFractions(
            food=city_data["Waste Components: Food (%)"].values[0] / 100,
            green=city_data["Waste Components: Green (%)"].values[0] / 100,
            wood=city_data["Waste Components: Wood (%)"].values[0] / 100,
            paper_cardboard=city_data[
                "Waste Components: Paper and Cardboard (%)"
            ].values[0]
            / 100,
            textiles=city_data["Waste Components: Textiles (%)"].values[0] / 100,
            plastic=city_data["Waste Components: Plastic (%)"].values[0] / 100,
            metal=city_data["Waste Components: Metal (%)"].values[0] / 100,
            glass=city_data["Waste Components: Glass (%)"].values[0] / 100,
            rubber=city_data["Waste Components: Rubber/Leather (%)"].values[0] / 100,
            other=city_data["Waste Components: Other (%)"].values[0] / 100,
        )
        waste_fractions_dict = waste_fractions.model_dump()
        waste_fractions = pd.DataFrame(waste_fractions_dict, index=years)

        div_fractions = DiversionFractions(
            compost=city_data["Diversions: Compost (%)"].values[0] / 100,
            anaerobic=city_data["Diversions: Anaerobic Digestion (%)"].values[0] / 100,
            combustion=city_data["Diversions: Incineration (%)"].values[0] / 100,
            recycling=city_data["Diversions: Recycling (%)"].values[0] / 100,
        )
        div_fractions_dict = div_fractions.model_dump()
        div_fractions = pd.DataFrame(div_fractions_dict, index=years)
        div_fractions = div_fractions.fillna(0)

        split_fractions = SplitFractions(
            landfill_w_capture=city_data["Percent of Waste to Landfills with Gas Capture (%)"].values[0] / 100,
            landfill_wo_capture=city_data["Percent of Waste to Landfills without Gas Capture (%)"].values[0] / 100,
            dumpsite=city_data["Percent of Waste to Dumpsites (%)"].values[0] / 100,
        )

        div_component_fractions = DivComponentFractions(
            compost=WasteFractions(
                food=city_data["Diversion Components: Composted Food (% of Total Composted)"].values[0] / 100,
                green=city_data["Diversion Components: Composted Green (% of Total Composted)"].values[0] / 100,
                wood=city_data["Diversion Components: Composted Wood (% of Total Composted)"].values[0] / 100,
                paper_cardboard=city_data["Diversion Components: Composted Paper and Cardboard (% of Total Composted)"].values[0] / 100,
                textiles=0,
                plastic=0,
                metal=0,
                glass=0,
                rubber=0,
                other=0,
            ),
            anaerobic=WasteFractions(
                food=city_data["Diversion Components: Anaerobically Digested Food (% of Total Digested)"].values[0] / 100,
                green=city_data["Diversion Components: Anaerobically Digested Green (% of Total Digested)"].values[0] / 100,
                wood=city_data["Diversion Components: Anaerobically Digested Wood (% of Total Digested)"].values[0] / 100,
                paper_cardboard=city_data["Diversion Components: Anaerobically Digested Paper and Cardboard (% of Total Digested)"].values[0] / 100,
                textiles=0,
                plastic=0,
                metal=0,
                glass=0,
                rubber=0,
                other=0,
            ),
            combustion=WasteFractions(
                food=city_data["Diversion Components: Incinerated Food (% of Total Incinerated)"].values[0] / 100,
                green=city_data["Diversion Components: Incinerated Green (% of Total Incinerated)"].values[0] / 100,
                wood=city_data["Diversion Components: Incinerated Wood (% of Total Incinerated)"].values[0] / 100,
                paper_cardboard=city_data["Diversion Components: Incinerated Paper and Cardboard (% of Total Incinerated)"].values[0] / 100,
                textiles=city_data["Diversion Components: Incinerated Textiles (% of Total Incinerated)"].values[0] / 100,
                plastic=city_data["Diversion Components: Incinerated Plastic (% of Total Incinerated)"].values[0] / 100,
                metal=0,
                glass=0,
                rubber=city_data["Diversion Components: Incinerated Rubber/Leather (% of Total Incinerated)"].values[0] / 100,
                other=0,
            ),
            recycling=WasteFractions(
                wood=city_data["Diversion Components: Recycled Wood (% of Total Recycled)"].values[0] / 100,
                paper_cardboard=city_data["Diversion Components: Recycled Paper and Cardboard (% of Total Recycled)"].values[0] / 100,
                plastic=city_data["Diversion Components: Recycled Plastic (% of Total Recycled)"].values[0] / 100,
                rubber=city_data["Diversion Components: Recycled Rubber/Leather (% of Total Recycled)"].values[0] / 100,
                textiles=city_data["Diversion Components: Recycled Textiles (% of Total Recycled)"].values[0] / 100,
                glass=city_data["Diversion Components: Recycled Glass (% of Total Recycled)"].values[0] / 100,
                metal=city_data["Diversion Components: Recycled Metal (% of Total Recycled)"].values[0] / 100,
                other=city_data["Diversion Components: Recycled Other (% of Total Recycled)"].values[0] / 100,
                food=0,
                green=0,
            ),
        )

        compost_dict = div_component_fractions.compost.model_dump()
        compost = pd.DataFrame(compost_dict, index=years)
        anaerobic_dict = div_component_fractions.anaerobic.model_dump()
        anaerobic = pd.DataFrame(anaerobic_dict, index=years)
        combustion_dict = div_component_fractions.combustion.model_dump()
        combustion = pd.DataFrame(combustion_dict, index=years)
        recycling_dict = div_component_fractions.recycling.model_dump()
        recycling = pd.DataFrame(recycling_dict, index=years)
        div_component_fractions = DivComponentFractionsDF(
            compost=compost,
            anaerobic=anaerobic,
            combustion=combustion,
            recycling=recycling,
        )

        # ks = DecompositionRates(
        #     food=city_data['k: Food'].values[0],
        #     green=city_data['k: Green'].values[0],
        #     wood=city_data['k: Wood'].values[0],
        #     paper_cardboard=city_data['k: Paper and Cardboard'].values[0],
        #     textiles=city_data['k: Textiles'].values[0]
        # )

        non_compostable_not_targeted_total = sum(
            [
                self.non_compostable_not_targeted[x]
                * div_component_fractions.compost.loc[2000, x]
                for x in self.div_components["compost"]
            ]
        )
        non_compostable_not_targeted_total = pd.Series(
            non_compostable_not_targeted_total, index=years
        )
        if non_compostable_not_targeted_total.isna().all():
            non_compostable_not_targeted_total = pd.Series(0, index=years)

        gas_capture_efficiency = city_data["Methane Capture Efficiency (%)"].values[0] / 100
        gas_capture_efficiency = pd.Series(gas_capture_efficiency, index=years)

        mef_compost = city_data["MEF: Compost"].values[0]

        waste_mass = city_data["Waste Generation Rate (tons/year)"].values[0]
        waste_mass = pd.Series(waste_mass, index=years)

        year_of_data_pop = city_data["Year of Data Collection (Population)"].values[0]

        city_instance_attrs = {
            "city_name": self.city_name,
            "country": self.country,
            "components": self.components,
            "div_components": self.div_components,
            "waste_types": self.waste_types,
            "unprocessable": self.unprocessable,
            "non_compostable_not_targeted": self.non_compostable_not_targeted,
            "combustion_reject_rate": self.combustion_reject_rate,
            "recycling_reject_rates": self.recycling_reject_rates,
        }

        waste_masses = {
            x: waste_mass.at[2000] * waste_fractions.loc[2000, x]
            for x in self.waste_types
        }
        waste_masses = WasteMasses(**waste_masses)

        sites_method = city_data["Uses Sites Method"].values[0]
        if (sites_method == True) and (dst == False):
            associated_sites = city_data["Associated Sites"].values[0]
            lifespans = city_data["Lifespans"].values[0]
            site_types = city_data["Site Types"].values[0]
            percent_waste_to_sites = city_data["Percent Waste to Sites"].values[0]
            mcfs = city_data["MCFs"].values[0]
            ox_values = city_data["Ox Values"].values[0]
            gascap_effs = city_data["Gas Capture Efficiencies"].values[0]
            latlons = city_data["Latitude Longitude of Sites"].values[0]
            sites_info_dict = {
                "associated_sites": associated_sites,
                "lifespans": lifespans,
                "site_types": site_types,
                "percent_waste_to_sites": percent_waste_to_sites,
                "mcfs": mcfs,
                "ox_values": ox_values,
                "gascap_effs": gascap_effs,
                "latlons": latlons,
            }
        else:
            sites_info_dict = {}

        try:
            city_parameters = CityParameters(
                waste_fractions=waste_fractions,
                div_fractions=div_fractions,
                split_fractions=split_fractions,
                div_component_fractions=div_component_fractions,
                precip=float(
                    city_data["Average Annual Precipitation (mm/year)"].values[0]
                ),
                temperature=float(city_data["Temperature (C)"].values[0]),
                growth_rate_historic=city_data[
                    "Population Growth Rate: Historic (%)"
                ].values[0]
                / 100
                + 1,
                growth_rate_future=city_data[
                    "Population Growth Rate: Future (%)"
                ].values[0]
                / 100
                + 1,
                waste_per_capita=city_data[
                    "Waste Generation Rate per Capita (kg/person/day)"
                ].values[0],
                precip_zone=city_data["Precipitation Zone"].values[0],
                # ks=ks,
                gas_capture_efficiency=gas_capture_efficiency,
                mef_compost=mef_compost,
                waste_mass=waste_mass,
                waste_masses=waste_masses,
                non_compostable_not_targeted_total=non_compostable_not_targeted_total,
                year_of_data_pop=year_of_data_pop,
                scenario=scenario,
                city_instance_attrs=city_instance_attrs,
                population=city_data["Population"].values[0],
                source_msw=city_data["Data Source (Waste Mass)"].values[0],
                sites_method=sites_method,
                sites_info_dict=sites_info_dict,
            )
        except Exception as e:
            raise CustomError(
                "city_params_error", f"Error creating CityParameters instance: {e}"
            )

        # Filter out the 'landfills' and 'non_zero_landfills' attributes from CityParameters
        # city_params = {k: v for k, v in city_parameters.__dict__.items() if k not in ['landfills', 'non_zero_landfills']}
        # city_params = copy.deepcopy(city_parameters.__dict__)

        self.baseline_parameters = city_parameters

    def load_andre_params(self, row, backfill=False):
        """
        Loads model parameters from the internal RMI WasteMAP database. Defaults are used
        where data is missing, incomplete, or incompatible.

        Args:
            row (tuple): row[0] is the index of the row in the dataframe used for input,
            row[1] is the row itself.

        Returns:
            None
        """

        if backfill:
            years = range(2010, 2051)
            current_row = row[row['Year Emissions'] == 2025]
            data_source = current_row["Data Source (Population)"].iloc[0]
            country = current_row["Country"].iloc[0]
            iso3 = current_row["Country ISO3"].iloc[0]
            region = defaults_2019.region_lookup_iso3[iso3]
            year_of_data_pop = current_row["Year of Data Collection (Population)"].iloc[0]
            year_of_data_msw = current_row["Year of Data Collection (MSW)"].iloc[0]
            waste_mass = row["Waste Generation Rate (tons/year)"]
            waste_fractions = row.loc[:, [
                "Waste Components: Food (%)", 
                "Waste Components: Green (%)", 
                "Waste Components: Wood (%)", 
                "Waste Components: Paper and Cardboard (%)", 
                "Waste Components: Textiles (%)", 
                "Waste Components: Plastic (%)", 
                "Waste Components: Metal (%)", 
                "Waste Components: Glass (%)", 
                "Waste Components: Rubber/Leather (%)",
                "Waste Components: Other (%)",
            ]] / 100
            waste_fractions = waste_fractions.rename(columns={
                "Waste Components: Food (%)": "food", 
                "Waste Components: Green (%)": "green", 
                "Waste Components: Wood (%)": "wood", 
                "Waste Components: Paper and Cardboard (%)": "paper_cardboard", 
                "Waste Components: Textiles (%)": "textiles",
                "Waste Components: Plastic (%)": "plastic", 
                "Waste Components: Metal (%)": "metal", 
                "Waste Components: Glass (%)": "glass", 
                "Waste Components: Rubber/Leather (%)": "rubber", 
                "Waste Components: Other (%)": "other"
            })
            waste_fractions.index = row['Year Emissions'].values
            div_fractions = row.loc[:, [
                "Diversions: Compost (%)",
                "Diversions: Anaerobic Digestion (%)",
                "Diversions: Incineration (%)",
                "Diversions: Recycling (%)",
            ]]
            div_component_fractions = {
                "compost": waste_fractions.loc[:, list(self.div_components["compost"])].div(waste_fractions.loc[:, list(self.div_components["compost"])].sum(axis=1), axis=0),
                "anaerobic": waste_fractions.loc[:, list(self.div_components["anaerobic"])].div(waste_fractions.loc[:, list(self.div_components["anaerobic"])].sum(axis=1), axis=0),
                "combustion": waste_fractions.loc[:, list(self.div_components["combustion"])].div(waste_fractions.loc[:, list(self.div_components["combustion"])].sum(axis=1), axis=0),
                "recycling": waste_fractions.loc[:, list(self.div_components["recycling"])].div(waste_fractions.loc[:, list(self.div_components["recycling"])].sum(axis=1), axis=0),
            }
            precipitation_zone = defaults_2019.get_precipitation_zone(current_row["Average Annual Precipitation (mm/year)"].iloc[0])
            mef_compost = ((0.0055 * waste_fractions["food"].values[0] / (waste_fractions["food"].values[0] + waste_fractions["green"].values[0])+ 0.0139 * waste_fractions["green"].values[0] / (waste_fractions["food"].values[0] + waste_fractions["green"].values[0]))* 1.1023 * 0.7)
            baseline = CityParameters(
                waste_fractions=waste_fractions,
                div_fractions=div_fractions,
                div_component_fractions=div_component_fractions,
                precip=current_row["Average Annual Precipitation (mm/year)"].iloc[0],
                growth_rate_historic=current_row["Population Growth Rate: Historic (%)"].iloc[0] / 100 + 1,
                growth_rate_future=current_row["Population Growth Rate: Future (%)"].iloc[0] / 100 + 1,
                precip_zone=precipitation_zone,
                gas_capture_efficiency=None,
                mef_compost=mef_compost,
                waste_mass=waste_mass,
                year_of_data_pop=year_of_data_pop,
                year_of_data_msw=year_of_data_msw,
                scenario=0,
                temp=current_row["Temperature (C)"].iloc[0],
                temperature=current_row["Temperature (C)"].iloc[0],
            )
            baseline._singapore_k(implement_year=year_of_data_msw)
            self.baseline_parameters = baseline
        else:
            data_source = row["population_data_source"]
            country = row["country"]
            self.country = country
            iso3 = row["iso"]
            self.iso3 = iso3
            region = defaults_2019.region_lookup[country]
            self.region = region
            year_of_data_pop = row["population_year"]
            assert np.isnan(year_of_data_pop) == False, "Population year is missing"
            year_of_data_msw = row["msw_collected_year"]
            if np.isnan(year_of_data_msw):
                year_of_data_msw = row["msw_generated_year"]
            if np.isnan(year_of_data_msw):
                year_of_data_msw = row["data_collection_year"].iloc[0]
            year_of_data_msw = int(year_of_data_msw)

            # Define the range of years
            years = range(1990, 2051)

            # Hardcode missing population values
            population = float(row["population_count"])
            if self.city_name == "Pago Pago":
                population = 3656
                year_of_data_pop = 2010
            elif self.city_name == "Kano":
                population = 2828861
                year_of_data_pop = 2006
            elif self.city_name == "Ramallah":
                population = 38998
                year_of_data_pop = 2017
            elif self.city_name == "Soweto":
                population = 1271628
                year_of_data_pop = 2011
            elif self.city_name == "Kadoma City":
                population = 116300
                year_of_data_pop = 2022
            elif self.city_name == "Mbare":
                population = 450000
                year_of_data_pop = 2020
            elif self.city_name == "Masvingo City":
                population = 90286
                year_of_data_pop = 2022
            elif self.city_name == "Limbe":
                population = 84223
                year_of_data_pop = 2005
            elif self.city_name == "Labe":
                population = 200000
                year_of_data_pop = 2014

            growth_rate_historic = row["historic_growth_rate"]
            growth_rate_future = row["future_growth_rate"]

            self.latitude = float(row['latitude'])
            self.longitude = float(row['longitude'])

            self.waste_mass_defaults = False

            # Get waste total
            try:
                waste_mass_load = float(
                    row["msw_generated_metric_tons_per_year"]
                )  # unit is tons
                if np.isnan(waste_mass_load):
                    waste_mass_load = float(row["msw_collected_metric_tons_per_year"])
                waste_per_capita = (
                    waste_mass_load * 1000 / population / 365
                )  # unit is kg/person/day
            except:
                waste_mass_load = float(
                    row["msw_generated_metric_tons_per_year"].replace(",", "")
                )
                if np.isnan(waste_mass_load):
                    waste_mass_load = float(
                        row["msw_collected_metric_tons_per_year"].replace(",", "")
                    )
                waste_per_capita = waste_mass_load * 1000 / population / 365
            if waste_mass_load != waste_mass_load:
                # Use per capita default
                self.waste_mass_defaults = True
                if iso3 in defaults_2019.msw_per_capita_country:
                    waste_per_capita = defaults_2019.msw_per_capita_country[iso3]
                    year_of_data_msw = 2019
                else:
                    waste_per_capita = defaults_2019.msw_per_capita_defaults[region]
                    year_of_data_msw = 2019
                waste_mass_load = waste_per_capita * population / 1000 * 365

            # Subtract mass that is informally collected
            # self.informal_fraction = np.nan_to_num(row['percent_informal_sector_percent_collected_by_informal_sector_percent']) / 100
            # self.waste_mass = self.waste_mass_load * (1 - self.informal_fraction)
            waste_mass = waste_mass_load

            # Adjust waste mass to account for difference in reporting years between msw and population
            # if self.data_source == 'World Bank':
            if year_of_data_msw != year_of_data_pop:
                year_difference = year_of_data_pop - year_of_data_msw
                if year_of_data_msw < year_of_data_pop:
                    waste_mass *= growth_rate_historic**year_difference
                    waste_per_capita = waste_mass * 1000 / population / 365
                else:
                    waste_mass *= growth_rate_future**year_difference
                    waste_per_capita = waste_mass * 1000 / population / 365

            # Waste fractions
            waste_fractions = pd.Series(
                {
                    "food": row["composition_food_organic_waste_percent"] / 100,
                    "green": row["composition_yard_garden_green_waste_percent"] / 100,
                    "wood": row["composition_wood_percent"] / 100,
                    "paper_cardboard": row["composition_paper_cardboard_percent"] / 100,
                    "textiles": row["composition_textiles_percent"] / 100,
                    "plastic": row["composition_plastic_percent"] / 100,
                    "metal": row["composition_metal_percent"] / 100,
                    "glass": row["composition_glass_percent"] / 100,
                    "rubber": row["composition_rubber_leather_percent"] / 100,
                    "other": row["composition_other_percent"] / 100,
                }
            )

            # Add zeros where there are no values unless all values are nan, in which case use defaults
            self.waste_fractions_defaults = False
            if waste_fractions.isna().all():
                self.waste_fractions_defaults = True
                waste_fractions = defaults_2019.waste_composition_for(iso3, region)
            else:
                waste_fractions.fillna(0, inplace=True)
                # waste_fractions['textiles'] = 0

            if (waste_fractions.sum() < 0.98) or (waste_fractions.sum() > 1.02):
                self.waste_fractions_defaults = True
                # print('waste fractions do not sum to 1')
                waste_fractions = defaults_2019.waste_composition_for(iso3, region)

            waste_fractions_dict = waste_fractions.to_dict()

            # Normalize waste fractions to sum to 1
            s = sum([x for x in waste_fractions_dict.values()])
            waste_fractions = {x: waste_fractions[x] / s for x in waste_fractions.keys()}
            waste_fractions = pd.DataFrame(waste_fractions, index=years)

            try:
                # Calculate MEF for compost -- emissions from composted waste
                mef_compost = (
                    (
                        0.0055
                        * waste_fractions_dict["food"]
                        / (waste_fractions_dict["food"] + waste_fractions_dict["green"])
                        + 0.0139
                        * waste_fractions_dict["green"]
                        / (waste_fractions_dict["food"] + waste_fractions_dict["green"])
                    )
                    * 1.1023
                    * 0.7
                )  # / 28
            except:
                mef_compost = 0

            # Precipitation
            precip = float(row["mean_yearly_precip_2000_2021"])
            # precip_data = pd.read_excel('/Users/hugh/Downloads/Cities Waste Dataset_2010-2019_precip.xlsx')
            # self.precip = precip_data[precip_data['city_original'] == self.name]['total_precipitation(mm)_1970_2000'].values[0]
            precip_zone = defaults_2019.get_precipitation_zone(precip)
            temperature = row["mean_yearly_temp_2000_2021"]

            # depth
            depth = 3  # m

            # k values, which are decomposition rates
            # ks = defaults_2019.k_defaults[precip_zone]

            # Model components
            components = set(["food", "green", "wood", "paper_cardboard", "textiles"])

            # Compost params
            compost_components = set(["food", "green", "wood", "paper_cardboard"])
            compost_fraction = float(row["waste_treatment_compost_percent"]) / 100
            if np.isnan(compost_fraction):
                compost_fraction = 0.0

            # Anaerobic digestion params
            anaerobic_components = set(["food", "green", "wood", "paper_cardboard"])
            anaerobic_fraction = (
                float(row["waste_treatment_anaerobic_digestion_percent"]) / 100
            )

            # Combustion params
            combustion_components = set(
                [
                    "food",
                    "green",
                    "wood",
                    "paper_cardboard",
                    "textiles",
                    "plastic",
                    "rubber",
                ]
            )
            value1 = float(row["waste_treatment_incineration_percent"])
            value2 = float(row["waste_treatment_advanced_thermal_treatment_percent"])
            if np.isnan(value1) and np.isnan(value2):
                combustion_fraction = np.nan
            else:
                combustion_fraction = (np.nan_to_num(value1) + np.nan_to_num(value2)) / 100

            # Recycling params
            recycling_components = set(
                [
                    "wood",
                    "paper_cardboard",
                    "textiles",
                    "plastic",
                    "rubber",
                    "metal",
                    "glass",
                    "other",
                ]
            )
            recycling_fraction = float(row["waste_treatment_recycling_percent"]) / 100

            # How much waste is diverted to landfill with gas capture
            gas_capture_percent = (
                np.nan_to_num(
                    row["waste_treatment_sanitary_landfill_landfill_gas_system_percent"]
                )
                / 100
            )

            div_components = {}
            div_components["compost"] = compost_components
            div_components["anaerobic"] = anaerobic_components
            div_components["combustion"] = combustion_components
            div_components["recycling"] = recycling_components

            # Determine if we need to use defaults for landfills and diversion fractions
            landfill_inputs = [
                float(row["waste_treatment_sanitary_landfill_landfill_gas_system_percent"]),
                float(row["waste_treatment_controlled_landfill_percent"]),
                float(row["waste_treatment_landfill_unspecified_percent"]),
                float(row["waste_treatment_open_dump_percent"]),
            ]
            all_nan_fill = all(np.isnan(value) for value in landfill_inputs)
            total_fill = sum(0 if np.isnan(x) else x for x in landfill_inputs) / 100
            diversions = [
                compost_fraction,
                anaerobic_fraction,
                combustion_fraction,
                recycling_fraction,
            ]
            all_nan_div = all(np.isnan(value) for value in diversions)

            # First case to check: all diversions and landfills are 0. Use defaults.
            self.diversion_defaults = False
            self.landfill_split_defaults = False
            if all_nan_fill and all_nan_div:
                if iso3 in defaults_2019.fraction_composted_country:
                    compost_fraction = defaults_2019.fraction_composted_country[iso3]
                    self.diversion_defaults = True
                elif region in defaults_2019.fraction_composted:
                    compost_fraction = defaults_2019.fraction_composted[region]
                    self.diversion_defaults = True
                else:
                    compost_fraction = 0.0

                if iso3 in defaults_2019.fraction_incinerated_country:
                    combustion_fraction = defaults_2019.fraction_incinerated_country[iso3]
                    self.diversion_defaults = True
                elif region in defaults_2019.fraction_incinerated:
                    combustion_fraction = defaults_2019.fraction_incinerated[region]
                    self.diversion_defaults = True
                else:
                    combustion_fraction = 0.0

                if iso3 in ["CAN", "CHE", "DEU"]:
                    split_fractions = {
                        "landfill_w_capture": 0.0,
                        "landfill_wo_capture": 1.0,
                        "dumpsite": 0.0,
                    }
                else:
                    if iso3 in defaults_2019.fraction_open_dumped_country:
                        split_fractions = {
                            "landfill_w_capture": 0.0,
                            "landfill_wo_capture": defaults_2019.fraction_landfilled_country[
                                iso3
                            ],
                            "dumpsite": defaults_2019.fraction_open_dumped_country[iso3],
                        }
                        self.landfill_split_defaults = True
                    elif region in defaults_2019.fraction_open_dumped:
                        split_fractions = {
                            "landfill_w_capture": 0.0,
                            "landfill_wo_capture": defaults_2019.fraction_landfilled[
                                region
                            ],
                            "dumpsite": defaults_2019.fraction_open_dumped[region],
                        }
                        self.landfill_split_defaults = True
                    else:
                        if region in defaults_2019.landfill_default_regions:
                            split_fractions = {
                                "landfill_w_capture": 0,
                                "landfill_wo_capture": 1,
                                "dumpsite": 0.0,
                            }
                        else:
                            split_fractions = {
                                "landfill_w_capture": 0,
                                "landfill_wo_capture": 0,
                                "dumpsite": 1,
                            }

            # Second case to check: all diversions are nan, but landfills are not. Use defaults for diversions if landfills sum to less than 1
            # This assumes that entered data is incomplete. Also, normalize landfills to sum to 1.
            # Caveat: if landfills sum to 1, assume diversions are supposed to be 0.
            elif all_nan_div and total_fill > 0.99:
                split_fractions = {
                    "landfill_w_capture": np.nan_to_num(
                        row["waste_treatment_sanitary_landfill_landfill_gas_system_percent"]
                    )
                    / 100,
                    "landfill_wo_capture": (
                        np.nan_to_num(row["waste_treatment_controlled_landfill_percent"])
                        + np.nan_to_num(row["waste_treatment_landfill_unspecified_percent"])
                    )
                    / 100,
                    "dumpsite": np.nan_to_num(row["waste_treatment_open_dump_percent"])
                    / 100,
                }
            elif all_nan_div and total_fill < 0.99:
                if iso3 in defaults_2019.fraction_composted_country:
                    compost_fraction = defaults_2019.fraction_composted_country[iso3]
                    self.diversion_defaults = True
                elif region in defaults_2019.fraction_composted:
                    compost_fraction = defaults_2019.fraction_composted[region]
                    self.diversion_defaults = True
                else:
                    compost_fraction = 0.0

                if iso3 in defaults_2019.fraction_incinerated_country:
                    combustion_fraction = defaults_2019.fraction_incinerated_country[iso3]
                    self.diversion_defaults = True
                elif region in defaults_2019.fraction_incinerated:
                    combustion_fraction = defaults_2019.fraction_incinerated[region]
                    self.diversion_defaults = True
                else:
                    combustion_fraction = 0.0

                split_fractions = {
                    "landfill_w_capture": np.nan_to_num(
                        row["waste_treatment_sanitary_landfill_landfill_gas_system_percent"]
                    )
                    / 100,
                    "landfill_wo_capture": (
                        np.nan_to_num(row["waste_treatment_controlled_landfill_percent"])
                        + np.nan_to_num(row["waste_treatment_landfill_unspecified_percent"])
                    )
                    / 100,
                    "dumpsite": np.nan_to_num(row["waste_treatment_open_dump_percent"])
                    / 100,
                }

            # Third case to check: all landfills are nan, but diversions are not. Use defaults for landfills
            elif all_nan_fill:
                if iso3 in ["CAN", "CHE", "DEU"]:
                    split_fractions = {
                        "landfill_w_capture": 0.0,
                        "landfill_wo_capture": 1.0,
                        "dumpsite": 0.0,
                    }
                else:
                    if iso3 in defaults_2019.fraction_open_dumped_country:
                        split_fractions = {
                            "landfill_w_capture": 0.0,
                            "landfill_wo_capture": defaults_2019.fraction_landfilled_country[
                                iso3
                            ],
                            "dumpsite": defaults_2019.fraction_open_dumped_country[iso3],
                        }
                        self.landfill_split_defaults = True
                    elif region in defaults_2019.fraction_open_dumped:
                        split_fractions = {
                            "landfill_w_capture": 0.0,
                            "landfill_wo_capture": defaults_2019.fraction_landfilled[
                                region
                            ],
                            "dumpsite": defaults_2019.fraction_open_dumped[region],
                        }
                        self.landfill_split_defaults = True
                    else:
                        if region in defaults_2019.landfill_default_regions:
                            split_fractions = {
                                "landfill_w_capture": 0.0,
                                "landfill_wo_capture": 1.0,
                                "dumpsite": 0.0,
                            }
                        else:
                            split_fractions = {
                                "landfill_w_capture": 0.0,
                                "landfill_wo_capture": 0.0,
                                "dumpsite": 1.0,
                            }

            # Fourth case to check: imported non-nan values in both landfills and diversions. Use the values.
            else:
                split_fractions = {
                    "landfill_w_capture": np.nan_to_num(
                        row["waste_treatment_sanitary_landfill_landfill_gas_system_percent"]
                    )
                    / 100,
                    "landfill_wo_capture": (
                        np.nan_to_num(row["waste_treatment_controlled_landfill_percent"])
                        + np.nan_to_num(row["waste_treatment_landfill_unspecified_percent"])
                    )
                    / 100,
                    "dumpsite": np.nan_to_num(row["waste_treatment_open_dump_percent"])
                    / 100,
                }

            # Normalize landfills to 1
            split_total = sum([x for x in split_fractions.values()])
            if split_total == 0:
                if region in defaults_2019.landfill_default_regions:
                    split_fractions = {
                        "landfill_w_capture": 0.0,
                        "landfill_wo_capture": 1.0,
                        "dumpsite": 0.0,
                    }
                else:
                    split_fractions = {
                        "landfill_w_capture": 0.0,
                        "landfill_wo_capture": 0.0,
                        "dumpsite": 1.0,
                    }
            split_total = sum([x for x in split_fractions.values()])
            for site in split_fractions.keys():
                split_fractions[site] /= split_total

            # Replace diversion NaN values with 0
            (
                compost_fraction,
                anaerobic_fraction,
                combustion_fraction,
                recycling_fraction,
            ) = [
                np.nan_to_num(x)
                for x in [
                    compost_fraction,
                    anaerobic_fraction,
                    combustion_fraction,
                    recycling_fraction,
                ]
            ]

            # if self.iso3 == 'NGA':
            #     self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 0.0, 'dumpsite': 1.0}
            # Instantiate landfills
            # self.landfill_w_capture = Landfill(self, 1990, 2051, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_w_capture'], gas_capture=True)
            # self.landfill_wo_capture = Landfill(self, 1990, 2051, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_wo_capture'], gas_capture=False)
            # self.dumpsite = Landfill(self, 1990, 2051, 'dumpsite', 0.4, fraction_of_waste=self.split_fractions['dumpsite'], gas_capture=False)

            # landfills = [self.landfill_w_capture, self.landfill_wo_capture, self.dumpsite]
            # Only running model on landfills with non-zero waste reduces computation
            # non_zero_landfills = [x for x in self.landfills if x.fraction_of_waste > 0]

            divs = {}

            div_fractions_dict = {
                "compost": compost_fraction,
                "anaerobic": anaerobic_fraction,
                "combustion": combustion_fraction,
                "recycling": recycling_fraction,
            }

            # Normalize diversion fractions to sum to 1 if they exceed it
            s = sum(x for x in div_fractions_dict.values())
            if s > 1:
                for div in div_fractions_dict:
                    div_fractions_dict[div] /= s
            assert (
                sum(x for x in div_fractions_dict.values()) <= 1
            ), "Diversion fractions sum to more than 1"
            div_fractions = pd.DataFrame(div_fractions_dict, index=years)

            # # Use IPCC defaults if no data
            # if s == 0:
            #     self.div_fractions['compost'] = defaults.fraction_composted[self.region]
            #     self.div_fractions['combustion'] = defaults.fraction_incinerated[self.region]

            # UN Habitat has its own data import procedure
            # if data_source == 'UN Habitat':
            # pass
            # #self.changed_diversion, self.input_problems, self.div_component_fractions, self.divs = self.check_masses_un()

            # # Determine diversion waste type fractions
            # total_recovered_materials_with_rejects = float(row['total_recovered_materials_with_rejects'])
            # organic_waste_recovered = float(row['organic_waste_recovered'])
            # glass_recovered = float(row['glass_recovered'])
            # metal_recovered = float(row['metal_recovered'])
            # paper_or_cardboard = float(row['paper_or_cardboard'])
            # total_plastic_recovered = float(row['total_plastic_recovered'])
            # mixed_waste = float(row['mixed_waste'])
            # other_waste = float(row['other_waste'])
            # div_component_fractions, self.divs = self.determine_component_fractions_un()

            # # Calculate generated waste masses
            # waste_masses = {x: waste_fractions[x] * waste_mass for x in waste_fractions.keys()}
            # #self.changed_diversion, self.input_problems, self.div_component_fractions, self.divs = self.check_masses(self.div_fractions, self.divs)

            # # Adjust diversion waste type fractions (div_component_fractions) to make sure more waste is not diverted than generated
            # changed_diversion, input_problems, div_component_fractions, divs = self.check_masses_v2(self.div_fractions, self.div_component_fractions)
            # else:
            # Determine diversion waste type fractions
            def calculate_component_fractions(
                waste_fractions: WasteFractions, div_type: str
            ) -> WasteFractions:
                components = self.div_components[div_type]
                filtered_fractions = {
                    waste: waste_fractions[waste].at[2000] for waste in components
                }
                total = sum(filtered_fractions.values())
                normalized_fractions = {
                    waste: fraction / total
                    for waste, fraction in filtered_fractions.items()
                }
                return normalized_fractions

            div_component_fractions = DivComponentFractionsDF(
                compost=pd.DataFrame(
                    calculate_component_fractions(waste_fractions, "compost"), index=years
                ),
                anaerobic=pd.DataFrame(
                    calculate_component_fractions(waste_fractions, "anaerobic"), index=years
                ),
                combustion=pd.DataFrame(
                    calculate_component_fractions(waste_fractions, "combustion"),
                    index=years,
                ),
                recycling=pd.DataFrame(
                    calculate_component_fractions(waste_fractions, "recycling"), index=years
                ),
            )

            non_compostable_not_targeted_total = sum(
                [
                    self.non_compostable_not_targeted[x]
                    * div_component_fractions.compost.loc[2000, x]
                    for x in div_components["compost"]
                ]
            )
            non_compostable_not_targeted_total = pd.Series(
                non_compostable_not_targeted_total, index=years
            )
            if non_compostable_not_targeted_total.isna().all():
                non_compostable_not_targeted_total = pd.Series(0, index=years)

            gas_capture_efficiency = pd.Series(0.6, index=years)

            waste_mass = pd.Series(waste_mass, index=years)

            city_instance_attrs = {
                "city_name": self.city_name,
                "country": country,
                "components": components,
                "div_components": div_components,
                "waste_types": self.waste_types,
                "unprocessable": self.unprocessable,
                "non_compostable_not_targeted": self.non_compostable_not_targeted,
                "combustion_reject_rate": self.combustion_reject_rate,
                "recycling_reject_rates": self.recycling_reject_rates,
            }

            waste_masses = {
                x: waste_mass.at[2000] * waste_fractions.loc[2000, x]
                for x in self.waste_types
            }
            waste_masses = WasteMasses(**waste_masses)

            split_fractions_old = split_fractions
            split_fractions = SplitFractions(
                landfill_w_capture=split_fractions_old["landfill_w_capture"],
                landfill_wo_capture=split_fractions_old["landfill_wo_capture"],
                dumpsite=split_fractions_old["dumpsite"],
            )

            waste_masses_df = waste_fractions.multiply(waste_mass, axis=0)
            waste_generated_df = WasteGeneratedDF.create(
                waste_masses_df,
                1990,
                2050,
                year_of_data_pop,
                growth_rate_historic,
                growth_rate_future,
            )

            # Assign to CityParameters
            baseline = CityParameters(
                waste_fractions=waste_fractions,
                div_fractions=div_fractions,
                split_fractions=split_fractions,
                div_component_fractions=div_component_fractions,
                precip=precip,
                growth_rate_historic=growth_rate_historic,
                growth_rate_future=growth_rate_future,
                waste_per_capita=waste_per_capita,
                precip_zone=precip_zone,
                gas_capture_efficiency=gas_capture_efficiency,
                mef_compost=mef_compost,
                waste_mass=pd.Series(waste_mass, index=years),
                waste_masses=waste_masses,
                year_of_data_pop=year_of_data_pop,
                year_of_data_msw=year_of_data_msw,
                scenario=0,
                implement_year=None,
                divs_df=None,
                waste_generated_df=waste_generated_df,
                city_instance_attrs=city_instance_attrs,
                population=population,
                temp=temperature,
                temperature=temperature,
                waste_burning_emissions=None,
                non_compostable_not_targeted_total=non_compostable_not_targeted_total,
                source_pop=data_source,
            )
            self.baseline_parameters = baseline

            # Check masses consistency
            self._check_masses_v2(scenario=0)
            if baseline.input_problems:
                print("Input problems detected in baseline parameters.")
                return

            self._calculate_net_masses()
            if (baseline.net_masses < 0).any().any():
                print(f"Invalid new value")
                return

            self._calculate_divs()

    def finish_sites_prep(self):
        """
        Finalizes the preparation of site data for modeling.
        Returns:
            None
        """
        baseline = self.baseline_parameters
        baseline._singapore_k()

        self._check_masses_v2(scenario=0, advanced_baseline=True)

        if baseline.input_problems:
            raise ValueError("Invalid new values")

        self._calculate_net_masses(scenario=0, advanced_baseline=True)
        if (baseline.net_masses < 0).any().any():
            raise ValueError("Invalid new values")

        baseline.divs_df = DivsDF.create_advanced_baseline(
            baseline.divs,
            baseline.year_of_data_pop,
            baseline.growth_rate_historic,
            baseline.growth_rate_future,
        )

        landfills = baseline.sites_info_dict["associated_sites"]
        lifespans = baseline.sites_info_dict["lifespans"]
        site_types = baseline.sites_info_dict["site_types"]
        mcfs = baseline.sites_info_dict["mcfs"]
        gas_capture_efficiencies = baseline.sites_info_dict["gascap_effs"]
        gas_capture_presences = [True if x > 0 else False for x in gas_capture_efficiencies]
        latitudes_longitudes = baseline.sites_info_dict["latlons"]
        fractions_of_city_waste = baseline.sites_info_dict["percent_waste_to_sites"]
        oxidation_values = baseline.sites_info_dict["ox_values"]
        rmi_ids = landfills

        city_params_dict = baseline.update_cityparams_dict()

        baseline.landfills = []
        for i, landfill in enumerate(landfills):
            new_landfill = Landfill(
                open_date=lifespans[i][0],
                close_date=lifespans[i][1],
                site_type=site_types[i],
                mcf=pd.Series(mcfs[i], index=range(lifespans[i][0], 2051)),
                city_params_dict=city_params_dict,
                city_instance_attrs=baseline.city_instance_attrs,
                landfill_index=i,
                # fraction_of_waste=new_landfill_fracs[i],
                gas_capture=gas_capture_presences[i],
                scenario=0,
                new_baseline=True,
                gas_capture_efficiency=pd.Series(
                    gas_capture_efficiencies[i], index=range(lifespans[i][0], 2051)
                ),
                # flaring=pd.Series(flaring, index=year_range),
                # leachate_circulate=leachate_circulate[i],
                fraction_of_waste_vector=pd.Series(
                    fractions_of_city_waste[i], index=self.years_range
                ),
                advanced=True,
                latlon=latitudes_longitudes[i],
                ks=baseline.ks,
                oxidation_factor=pd.Series(
                    oxidation_values[i], index=range(lifespans[i][0], 2051)
                ),
                rmi_id=rmi_ids[i],
            )
            baseline.landfills.append(new_landfill)

        waste_masses_df = pd.DataFrame(
            baseline.waste_masses.model_dump(), index=self.years_range
        )
        baseline.waste_generated_df = WasteGeneratedDF.create(
            waste_masses_df,
            1990,
            2050,
            baseline.year_of_data_pop,
            baseline.growth_rate_historic,
            baseline.growth_rate_future,
        )

        baseline.repopulate_attr_dicts()
        for i, landfill in enumerate(baseline.landfills):
            # Might be able to do this more efficienctly...i'm looping over the pre implementation years twice sort of
            landfill.waste_mass_df = LandfillWasteMassDF.create_advanced(
                waste_generated_df=baseline.waste_generated_df.df,
                divs_df=baseline.divs_df,
                fraction_of_waste_series=landfill.fraction_of_waste_vector,
            ).df

        # scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in baseline.landfills:
            landfill.estimate_emissions(skip_ox=True)

        self.estimate_diversion_emissions(scenario=0)
        self.sum_landfill_emissions(scenario=0)

    def model_city_via_sites(self, row, linker, for_trace=False):
        """

        Handler function for modeling cities with site-by-site data.
        Args:
            row (tuple): row[0] is the index of the row in the dataframe used for input,
            row[1] is the row itself.
        Returns:
            None

        """

        if row[1]["waste_composition_data_source"] == "SINIR":
            self.sinar_city_and_site(row, linker, for_trace=True)

    def sinar_city_and_site(self, row, linker, for_trace=False):
        """
        Special loading function for Brazil data.

        Args:
            row (tuple): row[0] is the index of the row in the dataframe used for input,
            row[1] is the row itself.

        Returns:
            None
        """
        # Basic information
        # idx = row[0]
        row = row[1]
        self.years_range = range(1990, 2051)

        # Import basic information
        basics_dict = self.import_basics(row)
        data_source_pop = basics_dict["data_source_pop"]
        year_of_data_pop = basics_dict["year_of_data_pop"]
        year_of_data_msw = basics_dict["year_of_data_msw"]
        population = basics_dict["population"]
        growth_rate_historic = basics_dict["growth_rate_historic"]
        growth_rate_future = basics_dict["growth_rate_future"]
        waste_mass = basics_dict["waste_mass"]
        waste_per_capita = basics_dict["waste_per_capita"]
        waste_fractions = basics_dict["waste_fractions"]
        waste_mass_defaults = basics_dict["waste_mass_defaults"]
        waste_fractions_defaults = basics_dict["waste_fractions_defaults"]
        mef_compost = basics_dict["mef_compost"]
        precip = basics_dict["precip"]
        precip_zone = basics_dict["precip_zone"]
        temperature = basics_dict["temperature"]
        waste_masses = basics_dict["waste_masses"]
        waste_generated_df = basics_dict["waste_generated_df"]
        self.latitude = basics_dict["latitude"]
        self.longitude = basics_dict["longitude"]

        # Import div fractions
        div_dict = self.import_div_fractions(
            row,
            waste_fractions,
            waste_generated_df,
        )
        div_fractions = div_dict["div_fractions"]
        diversion_defaults = div_dict["diversion_defaults"]
        div_component_fractions = div_dict["div_component_fractions"]
        divs = div_dict["divs"]
        non_compostable_not_targeted_total = div_dict[
            "non_compostable_not_targeted_total"
        ]

        city_instance_attrs = {
            "city_name": self.city_name,
            "country": self.country,
            "components": self.components,
            "div_components": self.div_components,
            "waste_types": self.waste_types,
            "unprocessable": self.unprocessable,
            "non_compostable_not_targeted": self.non_compostable_not_targeted,
            "combustion_reject_rate": self.combustion_reject_rate,
            "recycling_reject_rates": self.recycling_reject_rates,
        }

        defaults_used = {
            "Waste Mass": waste_mass_defaults,
            "Waste Fractions": waste_fractions_defaults,
            "Diversion": diversion_defaults,
            "Landfill Fractions": False,
        }

        # Make a CityParameters instance
        baseline = CityParameters(
            waste_fractions=waste_fractions,
            div_fractions=div_fractions,
            div_component_fractions=div_component_fractions,
            precip=precip,
            growth_rate_historic=growth_rate_historic,
            growth_rate_future=growth_rate_future,
            waste_per_capita=waste_per_capita,
            precip_zone=precip_zone,
            mef_compost=mef_compost,
            waste_mass=pd.Series(waste_mass, index=self.years_range),
            waste_masses=waste_masses,
            year_of_data_pop=year_of_data_pop,
            year_of_data_msw=year_of_data_msw,
            scenario=0,
            implement_year=None,
            divs_df=None,
            city_instance_attrs=city_instance_attrs,
            population=population,
            temp=temperature,
            temperature=temperature,
            waste_burning_emissions=None,
            non_compostable_not_targeted_total=non_compostable_not_targeted_total,
            source_pop=data_source_pop,
            waste_generated_df=waste_generated_df,
            divs=divs,
            defaults_used=defaults_used,
        )
        self.baseline_parameters = baseline
        baseline._singapore_k(advanced_baseline=True)

        self._check_masses_v2(scenario=0, advanced_baseline=True)

        if baseline.input_problems:
            raise ValueError("Invalid new values")

        self._calculate_net_masses(scenario=0, advanced_baseline=True)
        if (baseline.net_masses < 0).any().any():
            raise ValueError("Invalid new values")

        baseline.divs_df = DivsDF.create_advanced_baseline(
            baseline.divs,
            baseline.year_of_data_pop,
            baseline.growth_rate_historic,
            baseline.growth_rate_future,
        )

        # Set up landfills
        get_site_type_idx = {
            "Sanitary Landfill": 0,
            "Controlled Dumpsite": 1,
            "Dumpsite": 2,
        }
        mcf_options = {
            "Sanitary Landfill": 1,
            "Controlled Dumpsite": 0.7,
            "Dumpsite": 0.4,
        }
        ox_options = {
            "ox_nocap": {
                "Sanitary Landfill": 0.1,
                "Controlled Dumpsite": 0.05,
                "Dumpsite": 0.0,
            },
            "ox_cap": {
                "Sanitary Landfill": 0.22,
                "Controlled Dumpsite": 0.1,
                "Dumpsite": 0.0,
            },
        }
        gas_eff_options = {
            "Sanitary Landfill": 0.6,
            "Controlled Dumpsite": 0.45,
            "Dumpsite": 0.0,
        }
        depth = 3
        landfills = linker["site_id"].unique().tolist()
        lifespans = {}
        site_types = {}
        mcfs = {}
        gas_capture_presences = {}
        oxidation_values = {}
        gas_capture_efficiencies = {}
        fractions_of_city_waste = {}
        latitudes_longitudes = {}
        rmi_ids = {}
        if (linker["ctf_year"].unique().shape[0] == 1) and (len(landfills) > 1):
            # We have multiple landfills and only one year, so we have to assume they exist in perpetuity
            for landfill in landfills:
                site_data = linker.loc[linker["site_id"] == landfill].sort_values(
                    by="ctf_year", ascending=False
                )
                site_data = site_data.loc[
                    site_data["ctf_percent_of_waste_sent"] > 0
                ].reset_index(drop=True)
                opens = linker.loc[
                    linker["site_id"] == landfill, "site_year_landfill_opened"
                ].unique()
                earliest_mention_in_data = int(site_data["ctf_year"].iat[-1])
                if len(opens) == 1 and np.isnan(opens[0]):
                    open = 1990
                else:
                    open = int(sorted(opens)[0])
                closes = linker.loc[
                    linker["site_id"] == landfill, "site_landfill_closure_year"
                ].unique()
                last_mention_in_data = int(site_data["ctf_year"].iat[0])
                if len(closes) == 1 and np.isnan(closes[0]):
                    close = 2050
                else:
                    close = int(sorted(closes, reverse=True)[0])
                    if last_mention_in_data > close:
                        close = last_mention_in_data
                lifespans[landfill] = (open, close)

                site_type = site_data.at[0, "site_type"]
                site_types[landfill] = site_type
                site_type_idx = get_site_type_idx[site_type]
                if (depth > 5) and (site_type_idx in (1, 2)):
                    mcfs[landfill] = 0.8
                else:
                    mcfs[landfill] = mcf_options[site_type]
                if "Yes" in site_data["fgc_lfg_collection_system_in_place"].unique():
                    gas_capture_presences[landfill] = True
                    oxidation_values[landfill] = ox_options["ox_cap"][site_type]
                else:
                    gas_capture_presences[landfill] = False
                    oxidation_values[landfill] = ox_options["ox_nocap"][site_type]
                gas_capture_efficiencies[landfill] = gas_eff_options[site_type]
                wastefrac = site_data[["ctf_year", "ctf_percent_of_waste_sent"]]
                fractions_of_city_waste[landfill] = (
                    wastefrac.set_index("ctf_year")[
                        "ctf_percent_of_waste_sent"
                    ].reindex(self.years_range, fill_value=0.0)
                ) / 100
                # This isn't good enough yet needs to be smarter.
                if open < earliest_mention_in_data:
                    fractions_of_city_waste[landfill].loc[open:close] = (
                        wastefrac["ctf_percent_of_waste_sent"].iat[-1] / 100
                    )
                if close > last_mention_in_data:
                    fractions_of_city_waste[landfill].loc[
                        last_mention_in_data:close
                    ] = (wastefrac["ctf_percent_of_waste_sent"].iat[0] / 100)
                latitudes_longitudes[landfill] = (
                    site_data.at[0, "site_latitude"],
                    site_data.at[0, "site_longitude"],
                )
                rmi_ids[landfill] = int(landfill)
        else:
            for landfill in landfills:
                earliest_landfill = False
                last_landfill = False
                site_data = linker.loc[linker["site_id"] == landfill].sort_values(
                    by="ctf_year", ascending=False
                )
                site_data = site_data.loc[
                    site_data["ctf_percent_of_waste_sent"] > 0
                ].reset_index(drop=True)
                if site_data.shape[0] == 0:
                    landfills = landfills[landfills != landfill]
                    linker = linker[linker["site_id"] != landfill]
                    continue
                if (
                    linker.sort_values(by="ctf_year", ascending=True)
                    .reset_index(drop=True)
                    .at[0, "site_id"]
                    == landfill
                ):
                    earliest_landfill = True
                if (
                    linker.sort_values(by="ctf_year", ascending=False)
                    .reset_index(drop=True)
                    .at[0, "site_id"]
                    == landfill
                ):
                    last_landfill = True
                opens = linker.loc[
                    linker["site_id"] == landfill, "site_year_landfill_opened"
                ].unique()
                earliest_mention_in_data = int(site_data["ctf_year"].iat[-1])
                if len(opens) == 1 and np.isnan(opens[0]):
                    if earliest_landfill:
                        open = 1990
                    else:
                        open = earliest_mention_in_data
                else:
                    open = int(sorted(opens)[0])
                closes = linker.loc[
                    linker["site_id"] == landfill, "site_landfill_closure_year"
                ].unique()
                last_mention_in_data = int(site_data["ctf_year"].iat[0])
                if len(closes) == 1 and np.isnan(closes[0]):
                    if last_landfill:
                        close = 2050
                    else:
                        close = last_mention_in_data
                else:
                    close = int(sorted(closes, reverse=True)[0])
                    if last_mention_in_data > close:
                        close = last_mention_in_data
                lifespans[landfill] = (open, close)

                site_type = site_data.at[0, "site_type"]
                site_types[landfill] = site_type
                site_type_idx = get_site_type_idx[site_type]
                if (depth > 5) and (site_type in (1, 2)):
                    mcfs[landfill] = 0.8
                else:
                    mcfs[landfill] = mcf_options[site_type]
                if "Yes" in site_data["fgc_lfg_collection_system_in_place"].unique():
                    gas_capture_presences[landfill] = True
                    oxidation_values[landfill] = ox_options["ox_cap"][site_type]
                    gas_capture_efficiencies[landfill] = gas_eff_options[site_type]
                else:
                    gas_capture_presences[landfill] = False
                    oxidation_values[landfill] = ox_options["ox_nocap"][site_type]
                    gas_capture_efficiencies[landfill] = 0

                wastefrac = site_data[["ctf_year", "ctf_percent_of_waste_sent"]]
                fractions_of_city_waste[landfill] = (
                    wastefrac.set_index("ctf_year")[
                        "ctf_percent_of_waste_sent"
                    ].reindex(self.years_range, fill_value=0.0)
                ) / 100
                # This isn't good enough yet needs to be smarter.
                if open < earliest_mention_in_data:
                    fractions_of_city_waste[landfill].loc[open:close] = (
                        wastefrac["ctf_percent_of_waste_sent"].iat[-1] / 100
                    )
                if close > last_mention_in_data:
                    fractions_of_city_waste[landfill].loc[
                        last_mention_in_data:close
                    ] = (wastefrac["ctf_percent_of_waste_sent"].iat[0] / 100)
                latitudes_longitudes[landfill] = (
                    site_data.at[0, "site_latitude"],
                    site_data.at[0, "site_longitude"],
                )
                rmi_ids[landfill] = int(landfill)

        # Make sure that the fractions of waste to landfills sum to 1 and there's no gaps between years starting after first landfill opens
        fractions_of_waste_to_landfills = pd.DataFrame(fractions_of_city_waste)

        def check_for_landfill_gaps(
            fractions_of_waste_to_landfills, pass_check_sum=False
        ):
            if not pass_check_sum:
                if fractions_of_waste_to_landfills.sum(axis=1).max() > (1 + 1e-5):
                    raise ValueError("Landfill fractions sum to more than 1")
            earliest_landfill_year = fractions_of_waste_to_landfills[
                fractions_of_waste_to_landfills.sum(axis=1) > 0
            ].index[0]
            last_landfill_year = fractions_of_waste_to_landfills[
                fractions_of_waste_to_landfills.sum(axis=1) > 0
            ].index[-1]
            if (
                fractions_of_waste_to_landfills.loc[
                    earliest_landfill_year:last_landfill_year
                ]
                .sum(axis=1)
                .max()
                < 1
            ):
                raise ValueError("Waste not allocated for all years")
            return earliest_landfill_year, last_landfill_year

        earliest_landfill_year, last_landfill_year = check_for_landfill_gaps(
            fractions_of_waste_to_landfills, pass_check_sum=True
        )

        # Normalize fractions of waste to landfills to sum to 1
        row_sums = fractions_of_waste_to_landfills.sum(axis=1)
        fractions_of_waste_to_landfills = fractions_of_waste_to_landfills.div(
            row_sums, axis=0
        ).fillna(0) #.infer_objects(copy=False)

        # Make a fake landfill if no data for old landfills
        if earliest_landfill_year > 1990:
            open = 1990
            close = earliest_landfill_year - 1
            lifespans["fake_landfill_early"] = (open, close)
            site_types["fake_landfill_early"] = (
                linker.sort_values(by="ctf_year", ascending=True)
                .reset_index(drop=True)
                .at[0, "site_type"]
            )
            mcfs["fake_landfill_early"] = mcf_options[site_types["fake_landfill_early"]]
            gas_capture_presence = (
                linker.sort_values(by="ctf_year", ascending=True)
                .reset_index(drop=True)
                .at[0, "fgc_lfg_collection_system_in_place"]
                == "Yes"
            )
            gas_capture_presences["fake_landfill_early"] = gas_capture_presence
            if gas_capture_presences["fake_landfill_early"]:
                oxidation_values["fake_landfill_early"] = ox_options["ox_cap"][
                    site_types["fake_landfill_early"]
                ]
            else:
                oxidation_values["fake_landfill_early"] = ox_options["ox_nocap"][
                    site_types["fake_landfill_early"]
                ]
            gas_capture_efficiencies["fake_landfill_early"] = gas_eff_options[
                site_types["fake_landfill_early"]
            ]
            fraction_of_city_waste = pd.Series(0, index=self.years_range)
            fraction_of_city_waste.loc[open:close] = 1.0
            fractions_of_city_waste["fake_landfill_early"] = fraction_of_city_waste
            latitudes_longitudes["fake_landfill_early"] = (self.lat, self.lon)
            rmi_ids["fake_landfill_early"] = 999999999
            landfills.append("fake_landfill_early")

        _, _ = check_for_landfill_gaps(fractions_of_waste_to_landfills)

        city_params_dict = baseline.update_cityparams_dict()

        baseline.landfills = []
        for i, landfill in enumerate(landfills):
            new_landfill = Landfill(
                open_date=lifespans[landfill][0],
                close_date=lifespans[landfill][1],
                site_type=site_types[landfill],
                mcf=pd.Series(
                    mcfs[landfill], index=range(lifespans[landfill][0], 2051)
                ),
                city_params_dict=city_params_dict,
                city_instance_attrs=baseline.city_instance_attrs,
                landfill_index=i,
                # fraction_of_waste=new_landfill_fracs[i],
                gas_capture=gas_capture_presences[landfill],
                scenario=0,
                new_baseline=True,
                gas_capture_efficiency=pd.Series(
                    gas_capture_efficiencies[landfill],
                    index=range(lifespans[landfill][0], 2051),
                ),
                # flaring=pd.Series(flaring, index=year_range),
                # leachate_circulate=leachate_circulate[i],
                fraction_of_waste_vector=fractions_of_city_waste[landfill],
                advanced=True,
                latlon=latitudes_longitudes[landfill],
                ks=baseline.ks,
                oxidation_factor=oxidation_values[landfill],
                rmi_id=rmi_ids[landfill],
            )
            baseline.landfills.append(new_landfill)

        baseline.repopulate_attr_dicts()
        for i, landfill in enumerate(baseline.landfills):
            # Might be able to do this more efficienctly...i'm looping over the pre implementation years twice sort of
            landfill.waste_mass_df = LandfillWasteMassDF.create_advanced(
                waste_generated_df=baseline.waste_generated_df.df,
                divs_df=baseline.divs_df,
                fraction_of_waste_series=landfill.fraction_of_waste_vector,
            ).df

        # scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?

        # if not hasattr(baseline.landfills[0], 'gas_capture_efficiency'):
        #     baseline.landfills[0].gas_capture_efficiency = 0.0

        if for_trace:
            for landfill in baseline.landfills:
                landfill.estimate_emissions(skip_ox=True, trace_monthly=True)

            self.estimate_diversion_emissions(scenario=0)
            self.sum_landfill_emissions(scenario=0, trace_monthly=True)
        else:
            for landfill in baseline.landfills:
                landfill.estimate_emissions(skip_ox=True)

            self.estimate_diversion_emissions(scenario=0)
            self.sum_landfill_emissions(scenario=0)

    def site_only_estimate(self, row=None, pop_data=None):
        """
        For generating estimates for sites only

        Args:
            id (int): The ID of the site.
            df (DataFrame): The DataFrame containing site data.

        Returns:
            None
        """
        # Basic information
        self.years_range = range(1990, 2051)

        # Import basic information
        basics_dict = self.import_basics_site(row, pop_data)
        data_source_pop = basics_dict["data_source_pop"]
        year_of_data_pop = basics_dict["year_of_data_pop"]
        year_of_data_msw = basics_dict["year_of_data_msw"]
        population = basics_dict["population"]
        growth_rate_historic = basics_dict["growth_rate_historic"]
        growth_rate_future = basics_dict["growth_rate_future"]
        waste_mass = basics_dict["waste_mass"]
        waste_per_capita = basics_dict["waste_per_capita"]
        waste_fractions = basics_dict["waste_fractions"]
        waste_mass_defaults = basics_dict["waste_mass_defaults"]
        waste_fractions_defaults = basics_dict["waste_fractions_defaults"]
        mef_compost = basics_dict["mef_compost"]
        precip = basics_dict["precip"]
        precip_zone = basics_dict["precip_zone"]
        temperature = basics_dict["temperature"]
        waste_masses = basics_dict["waste_masses"]
        waste_generated_df = basics_dict["waste_generated_df"]
        self.latitude = self.lat
        self.longitude = self.lon

        # Import div fractions
        div_dict = self.import_div_fractions_site(
            row,
            waste_fractions,
            waste_generated_df,
        )
        div_fractions = div_dict["div_fractions"]
        diversion_defaults = div_dict["diversion_defaults"]
        div_component_fractions = div_dict["div_component_fractions"]
        divs = div_dict["divs"]
        non_compostable_not_targeted_total = div_dict[
            "non_compostable_not_targeted_total"
        ]

        city_instance_attrs = {
            "city_name": self.city_name,
            "country": self.country,
            "components": self.components,
            "div_components": self.div_components,
            "waste_types": self.waste_types,
            "unprocessable": self.unprocessable,
            "non_compostable_not_targeted": self.non_compostable_not_targeted,
            "combustion_reject_rate": self.combustion_reject_rate,
            "recycling_reject_rates": self.recycling_reject_rates,
        }

        defaults_used = {
            "Waste Mass": waste_mass_defaults,
            "Waste Fractions": waste_fractions_defaults,
            "Diversion": diversion_defaults,
            "Landfill Fractions": False,
        }

        # Make a CityParameters instance
        baseline = CityParameters(
            waste_fractions=waste_fractions,
            div_fractions=div_fractions,
            div_component_fractions=div_component_fractions,
            precip=precip,
            growth_rate_historic=growth_rate_historic,
            growth_rate_future=growth_rate_future,
            waste_per_capita=waste_per_capita,
            precip_zone=precip_zone,
            mef_compost=mef_compost,
            waste_mass=pd.Series(waste_mass, index=self.years_range),
            waste_masses=waste_masses,
            year_of_data_pop=year_of_data_pop,
            year_of_data_msw=year_of_data_msw,
            scenario=0,
            implement_year=None,
            divs_df=None,
            city_instance_attrs=city_instance_attrs,
            population=population,
            temp=temperature,
            temperature=temperature,
            waste_burning_emissions=None,
            non_compostable_not_targeted_total=non_compostable_not_targeted_total,
            source_pop=data_source_pop,
            waste_generated_df=waste_generated_df,
            divs=divs,
            defaults_used=defaults_used,
        )
        self.baseline_parameters = baseline
        baseline._singapore_k(advanced_baseline=True)

        baseline.divs_df = DivsDF.create_advanced_baseline(
            baseline.divs,
            baseline.year_of_data_pop,
            baseline.growth_rate_historic,
            baseline.growth_rate_future,
        )

        # Set up landfills
        get_site_type_idx = {
            "Sanitary Landfill": 0,
            "Controlled Dumpsite": 1,
            "Dumpsite": 2,
        }
        mcf_options = {
            "Sanitary Landfill": 1,
            "Controlled Dumpsite": 0.7,
            "Dumpsite": 0.4,
        }
        ox_options = {
            "ox_nocap": {
                "Sanitary Landfill": 0.1,
                "Controlled Dumpsite": 0.05,
                "Dumpsite": 0.0,
            },
            "ox_cap": {
                "Sanitary Landfill": 0.22,
                "Controlled Dumpsite": 0.1,
                "Dumpsite": 0.0,
            },
        }
        gas_eff_options = {
            "Sanitary Landfill": 0.6,
            "Controlled Dumpsite": 0.45,
            "Dumpsite": 0.0,
        }
        depth = 3
        site_type = row["Site Type"].values[0]
        if site_type not in get_site_type_idx.keys():
            if self.region in [
                "Australia and New Zealand",
                "Eastern Asia",
                "North America",
                "Northern Europe",
                "Southern Europe",
                "Western Europe"
            ]:
                site_type = "Sanitary Landfill"
            else:
                site_type = "Dumpsite"
        site_type_idx = get_site_type_idx[site_type]
        city_params_dict = baseline.update_cityparams_dict()
        baseline.city_params_dict = city_params_dict
        baseline.landfills = []
        gas_capture_presence = row['has_gas_capture'].values[0]
        if gas_capture_presence == "Yes":
            gas_capture_presence = True
            oxidation_value = ox_options["ox_cap"][site_type]
        else:
            gas_capture_presence = False
            oxidation_value = ox_options["ox_nocap"][site_type]
        gas_capture_efficiency = gas_eff_options[site_type]
        if (depth > 5.0) and (site_type_idx in (1, 2)):
            mcf = 0.8
        else:
            mcf = mcf_options[site_type]
        open_date = row['Site Open Year'].fillna(1990).values[0]
        if open_date < 1990:
            open_date = 1990
        close_date = row['Site Close Year'].fillna(2051).values[0]
        fration_of_waste_vector = pd.Series(
                0.0, index=self.years_range
            )
        fration_of_waste_vector.loc[open_date:close_date] = 1.0
        new_landfill = Landfill(
            open_date=open_date,
            close_date=close_date,
            site_type=site_type,
            mcf=pd.Series(
                mcf, index=self.years_range
            ),
            city_params_dict=city_params_dict,
            city_instance_attrs=baseline.city_instance_attrs,
            landfill_index=0,
            gas_capture=gas_capture_presence,
            scenario=0,
            new_baseline=True,
            gas_capture_efficiency=pd.Series(
                gas_capture_efficiency,
                index=self.years_range,
            ),
            # flaring=pd.Series(flaring, index=year_range),
            # leachate_circulate=leachate_circulate[i],
            fraction_of_waste_vector=fration_of_waste_vector,
            advanced=True,
            latlon=(self.latitude, self.longitude),
            ks=baseline.ks,
            oxidation_factor=oxidation_value,
            rmi_id=row['RMI ID'].values[0],
        )
        baseline.landfills.append(new_landfill)

        baseline.repopulate_attr_dicts()
        for i, landfill in enumerate(baseline.landfills):
            # Might be able to do this more efficienctly...i'm looping over the pre implementation years twice sort of
            landfill.waste_mass_df = LandfillWasteMassDF.create_advanced(
                waste_generated_df=baseline.waste_generated_df.df,
                divs_df=baseline.divs_df,
                fraction_of_waste_series=landfill.fraction_of_waste_vector,
            ).df

        # scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in baseline.landfills:
            landfill.estimate_emissions(skip_ox=True)

        self.estimate_diversion_emissions(scenario=0)
        self.sum_landfill_emissions(scenario=0)

    def site_only_estimate_trace(self, canonical_row=None, time_series_rows=None, pop_data=None):
        """
        For generating estimates for sites only

        Args:
            id (int): The ID of the site.
            df (DataFrame): The DataFrame containing site data.

        Returns:
            None
        """
        # Basic information
        self.years_range = range(1990, 2051)

        # Import basic information
        basics_dict = self.import_basics_site(canonical_row, pop_data, usecase="trace", time_series_rows=time_series_rows)
        data_source_pop = basics_dict["data_source_pop"]
        year_of_data_pop = basics_dict["year_of_data_pop"]
        year_of_data_msw = basics_dict["year_of_data_msw"]
        if (year_of_data_msw is None) or (pd.isna(year_of_data_msw)):
            year_of_data_msw = 2025
        if (year_of_data_pop is None) or (pd.isna(year_of_data_pop)):
            year_of_data_pop = 2025
        population = basics_dict.get("population", 100)
        growth_rate_historic = basics_dict["growth_rate_historic"]
        growth_rate_future = basics_dict["growth_rate_future"]
        waste_mass = basics_dict["waste_mass"]
        waste_per_capita = basics_dict["waste_per_capita"]
        waste_fractions = basics_dict["waste_fractions"]
        waste_mass_defaults = basics_dict["waste_mass_defaults"]
        waste_fractions_defaults = basics_dict["waste_fractions_defaults"]
        mef_compost = basics_dict["mef_compost"]
        precip = basics_dict["precip"]
        precip_zone = basics_dict["precip_zone"]
        temperature = basics_dict["temperature"]
        waste_masses = basics_dict["waste_masses"]
        waste_generated_df = basics_dict["waste_generated_df"]
        self.latitude = self.lat
        self.longitude = self.lon
        #self.rmi_id = row['rmi_id']

        # Import div fractions
        div_dict = self.import_div_fractions_site(
            canonical_row,
            waste_fractions,
            waste_generated_df,
        )
        div_fractions = div_dict["div_fractions"]
        diversion_defaults = div_dict["diversion_defaults"]
        div_component_fractions = div_dict["div_component_fractions"]
        divs = div_dict["divs"]
        non_compostable_not_targeted_total = div_dict[
            "non_compostable_not_targeted_total"
        ]

        city_instance_attrs = {
            "city_name": self.city_name,
            "country": self.country,
            "components": self.components,
            "div_components": self.div_components,
            "waste_types": self.waste_types,
            "unprocessable": self.unprocessable,
            "non_compostable_not_targeted": self.non_compostable_not_targeted,
            "combustion_reject_rate": self.combustion_reject_rate,
            "recycling_reject_rates": self.recycling_reject_rates,
        }

        defaults_used = {
            "Waste Mass": waste_mass_defaults,
            "Waste Fractions": waste_fractions_defaults,
            "Diversion": diversion_defaults,
            "Landfill Fractions": False,
        }

        # Make a CityParameters instance
        baseline = CityParameters(
            waste_fractions=waste_fractions,
            div_fractions=div_fractions,
            div_component_fractions=div_component_fractions,
            precip=precip,
            growth_rate_historic=growth_rate_historic,
            growth_rate_future=growth_rate_future,
            waste_per_capita=waste_per_capita,
            precip_zone=precip_zone,
            mef_compost=mef_compost,
            waste_mass=pd.Series(waste_mass, index=self.years_range),
            waste_masses=waste_masses,
            year_of_data_pop=year_of_data_pop,
            year_of_data_msw=year_of_data_msw,
            scenario=0,
            implement_year=None,
            divs_df=None,
            city_instance_attrs=city_instance_attrs,
            population=population,
            temp=temperature,
            temperature=temperature,
            waste_burning_emissions=None,
            non_compostable_not_targeted_total=non_compostable_not_targeted_total,
            source_pop=data_source_pop,
            waste_generated_df=waste_generated_df,
            divs=divs,
            defaults_used=defaults_used,
        )
        self.baseline_parameters = baseline
        baseline._singapore_k(advanced_baseline=True)

        baseline.divs_df = DivsDF.create_advanced_baseline(
            baseline.divs,
            baseline.year_of_data_pop,
            baseline.growth_rate_historic,
            baseline.growth_rate_future,
        )

        # Set up landfills
        get_site_type_idx = {
            "Sanitary Landfill": 0,
            "Controlled Dumpsite": 1,
            "Dumpsite": 2,
        }
        mcf_options = {
            "Sanitary Landfill": 1,
            "Controlled Dumpsite": 0.7,
            "Dumpsite": 0.4,
        }
        ox_options = {
            "ox_nocap": {
                "Sanitary Landfill": 0.1,
                "Controlled Dumpsite": 0.05,
                "Dumpsite": 0.0,
            },
            "ox_cap": {
                "Sanitary Landfill": 0.22,
                "Controlled Dumpsite": 0.1,
                "Dumpsite": 0.0,
            },
        }
        gas_eff_options = {
            "Sanitary Landfill": 0.6,
            "Controlled Dumpsite": 0.45,
            "Dumpsite": 0.0,
        }
        # Get the most common non-NaN value, or 3 if all are NaN
        depth = canonical_row['waste_depth']
        site_type = canonical_row['type']
        if pd.isna(site_type) or site_type == '':
            if self.region in defaults_2019.landfill_default_regions:
                site_type = "Sanitary Landfill"
            else:
                site_type = "Dumpsite"
            if site_type not in get_site_type_idx.keys():
                if self.region in [
                    "Australia and New Zealand",
                    "Eastern Asia",
                    "North America",
                    "Northern Europe",
                    "Southern Europe",
                    "Western Europe"
                ]:
                    site_type = "Sanitary Landfill"
                else:
                    site_type = "Dumpsite"
        site_type_idx = get_site_type_idx[site_type]
        city_params_dict = baseline.update_cityparams_dict()
        baseline.landfills = []
        if 'landfill_gas_collection' in canonical_row.index:
            gas_capture_presence = canonical_row['landfill_gas_collection']
        else:
            gas_capture_presence = canonical_row['other7']

        # Handle pd.NA / missing: avoid "boolean value of NA is ambiguous" in comparisons
        if pd.isna(gas_capture_presence):
            gas_capture_presence = False
            oxidation_value = ox_options["ox_nocap"][site_type]
        elif (gas_capture_presence == "Yes") or (gas_capture_presence is True) or (gas_capture_presence == True):
            gas_capture_presence = True
            oxidation_value = ox_options["ox_cap"][site_type]
        else:
            try:
                if gas_capture_presence == gas_capture_presence:
                    if gas_capture_presence > 0:
                        gas_capture_presence = True
                        oxidation_value = ox_options["ox_cap"][site_type]
                    else:
                        gas_capture_presence = False
                        oxidation_value = ox_options["ox_nocap"][site_type]
                else:
                    gas_capture_presence = False
                    oxidation_value = ox_options["ox_nocap"][site_type]
            except:
                gas_capture_presence = False
                oxidation_value = ox_options["ox_nocap"][site_type]
        
        if isinstance(time_series_rows, pd.DataFrame):
            if time_series_rows['gas_collection_efficiency'].notna().any():
                gascap_df = time_series_rows[['reported_emissions_year', 'gas_collection_efficiency']]
                gascap_df = gascap_df.dropna(subset=['reported_emissions_year']).copy()
                gas_capture_efficiency_mean = gascap_df['gas_collection_efficiency'].mean()
                gas_capture_efficiency = pd.Series(gas_capture_efficiency_mean, index=self.years_range)
                gas_capture_efficiency.loc[gascap_df['reported_emissions_year'].values] = gascap_df['gas_collection_efficiency'].values
            else:
                if gas_capture_presence is True:
                    gas_capture_efficiency = gas_eff_options[site_type]
                else:
                    gas_capture_efficiency = 0
                gas_capture_efficiency = pd.Series(gas_capture_efficiency, index=self.years_range)
        else:
            gas_capture_efficiency = canonical_row['gas_collection_efficiency']
            if pd.isna(gas_capture_efficiency):
                if gas_capture_presence is True:
                    gas_capture_efficiency = gas_eff_options[site_type]
                else:
                    gas_capture_efficiency = 0
            gas_capture_efficiency = pd.Series(gas_capture_efficiency, index=self.years_range)
        
        if (depth > 5.0) and (site_type_idx in (1, 2)):
            mcf = 0.8
        else:
            mcf = mcf_options[site_type]
        open_date = canonical_row['site_open_year']
        if isinstance(open_date, str):
            if open_date[-2:] == '.0':
                open_date = int(open_date[:-2])
            elif open_date in ['NS', 'NSNSNSNS', 'NSNS', 'NO SABE', 'NO SUPO']:
                open_date = 1990
            else:
                open_date = int(open_date)
        else:
            if pd.isna(open_date) or open_date is None:
                open_date = 1990
            else:
                open_date = int(open_date)
        if open_date < 1990:
            open_date = 1990
        if open_date == 20007:
            open_date = 2007
        close_date = canonical_row['site_close_year']
        if isinstance(close_date, str):
            if close_date[-2:] == '.0':
                close_date = int(close_date[:-2])
            else:
                close_date = int(close_date)
        else:
            if pd.isna(close_date) or close_date is None:
                close_date = 2050
            else:
                close_date = int(close_date)
        fraction_of_waste_vector = pd.Series(
                0.0, index=self.years_range
            )
        fraction_of_waste_vector.loc[open_date:close_date-1] = 1.0
        id = int(canonical_row['asset_identifier'])
        oxidation_series = _build_oxidation_series(
            oxidation_value, canonical_row, time_series_rows, self.years_range
        )
        # Baseline flaring destruction efficiency, set explicitly to the canonical default
        # (was unset -> fell to model_v2's internal default). No per-site source exists;
        # mitigation raises it to 0.98/0.99 via _gccs_flaring (max/clip). Local import
        # avoids the dst_common <-> city_params import cycle.
        from SWEET_python.dst_common import DEFAULT_FLARE_EFFICIENCY
        flaring_series = pd.Series(DEFAULT_FLARE_EFFICIENCY, index=self.years_range)
        new_landfill = Landfill(
            open_date=open_date,
            close_date=close_date,
            site_type=site_type,
            mcf=pd.Series(
                mcf, index=self.years_range
            ),
            city_params_dict=city_params_dict,
            city_instance_attrs=baseline.city_instance_attrs,
            landfill_index=0,
            gas_capture=gas_capture_presence,
            scenario=0,
            new_baseline=True,
            gas_capture_efficiency=gas_capture_efficiency,
            flaring=flaring_series,
            # leachate_circulate=leachate_circulate[i],
            fraction_of_waste_vector=fraction_of_waste_vector,
            advanced=True,
            latlon=(self.latitude, self.longitude),
            ks=baseline.ks,
            oxidation_factor=oxidation_series,
            rmi_id=id,
        )
        baseline.landfills.append(new_landfill)

        baseline.repopulate_attr_dicts()
        for i, landfill in enumerate(baseline.landfills):
            # Might be able to do this more efficienctly...i'm looping over the pre implementation years twice sort of
            landfill.waste_mass_df = LandfillWasteMassDF.create_advanced(
                waste_generated_df=baseline.waste_generated_df.df,
                divs_df=baseline.divs_df,
                fraction_of_waste_series=landfill.fraction_of_waste_vector,
            ).df

        # scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in baseline.landfills:
            landfill.estimate_emissions(skip_ox=True, trace_monthly=True)

        self.estimate_diversion_emissions(scenario=0)
        self.sum_landfill_emissions(scenario=0, trace_monthly=True)

        for landfill in baseline.landfills:
            landfill.emissions = landfill.emissions.apply(self.convert_methane_m3_to_ton_co2e) / 28

    def citysite_estimate_trace(self, canonical_row=None, time_series_rows=None, citysite_rows=None, pop_data=None):
        """
        For generating estimates for sites only

        Args:
            id (int): The ID of the site.
            df (DataFrame): The DataFrame containing site data.

        Returns:
            None
        """
        # Basic information
        self.years_range = range(1990, 2051)

        # Import basic information
        basics_dict = self.import_basics_site(canonical_row, pop_data, usecase="trace", time_series_rows=time_series_rows)
        data_source_pop = basics_dict["data_source_pop"]
        year_of_data_pop = basics_dict["year_of_data_pop"]
        year_of_data_msw = basics_dict["year_of_data_msw"]
        population = basics_dict.get("population", 100)
        growth_rate_historic = basics_dict["growth_rate_historic"]
        growth_rate_future = basics_dict["growth_rate_future"]
        waste_mass = basics_dict["waste_mass"]
        waste_per_capita = basics_dict["waste_per_capita"]
        waste_fractions = basics_dict["waste_fractions"]
        waste_mass_defaults = basics_dict["waste_mass_defaults"]
        waste_fractions_defaults = basics_dict["waste_fractions_defaults"]
        mef_compost = basics_dict["mef_compost"]
        precip = basics_dict["precip"]
        precip_zone = basics_dict["precip_zone"]
        temperature = basics_dict["temperature"]
        waste_masses = basics_dict["waste_masses"]
        waste_generated_df = basics_dict["waste_generated_df"]
        self.latitude = self.lat
        self.longitude = self.lon
        #self.rmi_id = row['rmi_id']

        # Import div fractions
        div_dict = self.import_div_fractions_site(
            canonical_row,
            waste_fractions,
            waste_generated_df,
        )
        div_fractions = div_dict["div_fractions"]
        diversion_defaults = div_dict["diversion_defaults"]
        div_component_fractions = div_dict["div_component_fractions"]
        divs = div_dict["divs"]
        non_compostable_not_targeted_total = div_dict[
            "non_compostable_not_targeted_total"
        ]

        city_instance_attrs = {
            "city_name": self.city_name,
            "country": self.country,
            "components": self.components,
            "div_components": self.div_components,
            "waste_types": self.waste_types,
            "unprocessable": self.unprocessable,
            "non_compostable_not_targeted": self.non_compostable_not_targeted,
            "combustion_reject_rate": self.combustion_reject_rate,
            "recycling_reject_rates": self.recycling_reject_rates,
        }

        defaults_used = {
            "Waste Mass": waste_mass_defaults,
            "Waste Fractions": waste_fractions_defaults,
            "Diversion": diversion_defaults,
            "Landfill Fractions": False,
        }

        # Make a CityParameters instance
        baseline = CityParameters(
            waste_fractions=waste_fractions,
            div_fractions=div_fractions,
            div_component_fractions=div_component_fractions,
            precip=precip,
            growth_rate_historic=growth_rate_historic,
            growth_rate_future=growth_rate_future,
            waste_per_capita=waste_per_capita,
            precip_zone=precip_zone,
            mef_compost=mef_compost,
            waste_mass=pd.Series(waste_mass, index=self.years_range),
            waste_masses=waste_masses,
            year_of_data_pop=year_of_data_pop,
            year_of_data_msw=year_of_data_msw,
            scenario=0,
            implement_year=None,
            divs_df=None,
            city_instance_attrs=city_instance_attrs,
            population=population,
            temp=temperature,
            temperature=temperature,
            waste_burning_emissions=None,
            non_compostable_not_targeted_total=non_compostable_not_targeted_total,
            source_pop=data_source_pop,
            waste_generated_df=waste_generated_df,
            divs=divs,
            defaults_used=defaults_used,
        )
        self.baseline_parameters = baseline
        baseline._singapore_k(advanced_baseline=True)

        baseline.divs_df = DivsDF.create_advanced_baseline(
            baseline.divs,
            baseline.year_of_data_pop,
            baseline.growth_rate_historic,
            baseline.growth_rate_future,
        )

        # Set up landfills
        get_site_type_idx = {
            "Sanitary Landfill": 0,
            "Controlled Dumpsite": 1,
            "Dumpsite": 2,
        }
        mcf_options = {
            "Sanitary Landfill": 1,
            "Controlled Dumpsite": 0.7,
            "Dumpsite": 0.4,
        }
        ox_options = {
            "ox_nocap": {
                "Sanitary Landfill": 0.1,
                "Controlled Dumpsite": 0.05,
                "Dumpsite": 0.0,
            },
            "ox_cap": {
                "Sanitary Landfill": 0.22,
                "Controlled Dumpsite": 0.1,
                "Dumpsite": 0.0,
            },
        }
        gas_eff_options = {
            "Sanitary Landfill": 0.6,
            "Controlled Dumpsite": 0.45,
            "Dumpsite": 0.0,
        }
        # Get the most common non-NaN value, or 3 if all are NaN
        depth = canonical_row['waste_depth']
        site_type = canonical_row['type']
        if pd.isna(site_type) or site_type == '':
            if self.region in defaults_2019.landfill_default_regions:
                site_type = "Sanitary Landfill"
            else:
                site_type = "Dumpsite"
            if site_type not in get_site_type_idx.keys():
                if self.region in [
                    "Australia and New Zealand",
                    "Eastern Asia",
                    "North America",
                    "Northern Europe",
                    "Southern Europe",
                    "Western Europe"
                ]:
                    site_type = "Sanitary Landfill"
                else:
                    site_type = "Dumpsite"
        site_type_idx = get_site_type_idx[site_type]
        city_params_dict = baseline.update_cityparams_dict()
        baseline.landfills = []
        if 'landfill_gas_collection' in canonical_row.index:
            gas_capture_presence = canonical_row['landfill_gas_collection']
        else:
            gas_capture_presence = canonical_row['other7']

        if gas_capture_presence == "Yes" or gas_capture_presence == True:
            gas_capture_presence = True
            oxidation_value = ox_options["ox_cap"][site_type]
        else:
            gas_capture_presence = False
            oxidation_value = ox_options["ox_nocap"][site_type]
        
        if isinstance(time_series_rows, pd.DataFrame):
            if time_series_rows['gas_collection_efficiency'].notna().any():
                gascap_df = time_series_rows[['reported_emissions_year', 'gas_collection_efficiency']]
                gascap_df = gascap_df.dropna(subset=['reported_emissions_year']).copy()
                gas_capture_efficiency_mean = gascap_df['gas_collection_efficiency'].mean()
                gas_capture_efficiency = pd.Series(gas_capture_efficiency_mean, index=self.years_range)
                gas_capture_efficiency.loc[gascap_df['reported_emissions_year'].values] = gascap_df['gas_collection_efficiency'].values
            else:
                gas_capture_efficiency = canonical_row['gas_collection_efficiency']
                if pd.isna(gas_capture_efficiency):
                    gas_capture_efficiency = gas_eff_options[site_type]
                gas_capture_efficiency = pd.Series(gas_capture_efficiency, index=self.years_range)

        else:
            gas_capture_efficiency = canonical_row['gas_collection_efficiency']
            if pd.isna(gas_capture_efficiency):
                gas_capture_efficiency = gas_eff_options[site_type]
            gas_capture_efficiency = pd.Series(gas_capture_efficiency, index=self.years_range)
        
        if (depth > 5.0) and (site_type_idx in (1, 2)):
            mcf = 0.8
        else:
            mcf = mcf_options[site_type]
        open_date = canonical_row['site_open_year']
        if isinstance(open_date, str):
            if open_date[-2:] == '.0':
                open_date = int(open_date[:-2])
            elif open_date in ['NS', 'NSNSNSNS', 'NSNS', 'NO SABE', 'NO SUPO']:
                open_date = 1990
            else:
                open_date = int(open_date)
        else:
            if pd.isna(open_date) or open_date is None:
                open_date = 1990
            else:
                open_date = int(open_date)
        if open_date < 1990:
            open_date = 1990
        if open_date == 20007:
            open_date = 2007
        close_date = canonical_row['site_close_year']
        if isinstance(close_date, str):
            if close_date[-2:] == '.0':
                close_date = int(close_date[:-2])
            else:
                close_date = int(close_date)
        else:
            if pd.isna(close_date) or close_date is None:
                close_date = 2050
            else:
                close_date = int(close_date)

        id = int(canonical_row['asset_identifier'])
        oxidation_series = _build_oxidation_series(
            oxidation_value, canonical_row, time_series_rows, self.years_range
        )
        # Baseline flaring destruction efficiency, set explicitly to the canonical default
        # (see site_only_estimate_trace). Applies to both the single- and multi-city
        # landfill constructors below. Local import avoids the dst_common import cycle.
        from SWEET_python.dst_common import DEFAULT_FLARE_EFFICIENCY
        flaring_series = pd.Series(DEFAULT_FLARE_EFFICIENCY, index=self.years_range)
        baseline._singapore_k(advanced_baseline=True)
        if (citysite_rows is None) or (isinstance(citysite_rows, pd.Series)):
            fraction_of_waste_vector = pd.Series(
                    0.0, index=self.years_range
                )
            fraction_of_waste_vector.loc[open_date:close_date] = 1.0
            new_landfill = Landfill(
                open_date=open_date,
                close_date=close_date,
                site_type=site_type,
                mcf=pd.Series(
                    mcf, index=self.years_range
                ),
                city_params_dict=city_params_dict,
                city_instance_attrs=baseline.city_instance_attrs,
                landfill_index=0,
                gas_capture=gas_capture_presence,
                scenario=0,
                new_baseline=True,
                gas_capture_efficiency=gas_capture_efficiency,
                flaring=flaring_series,
                # leachate_circulate=leachate_circulate[i],
                fraction_of_waste_vector=fraction_of_waste_vector,
                advanced=True,
                latlon=(self.latitude, self.longitude),
                ks=baseline.ks,
                oxidation_factor=oxidation_series,
                rmi_id=id,
                city_id=citysite_rows['city_id']
            )
            baseline.landfills.append(new_landfill)
            baseline.fraction_of_waste_df = pd.DataFrame(1.0, index=self.years_range, columns=[citysite_rows['city_id']])
        else:
            # Calculate a fraction_of_waste_vector for each city's waste contribution to the site
            # Simulates different city sources to the same landfill by making virtual extra landfills
            cities_to_model = citysite_rows['city_id'].unique()
            populations = citysite_rows.groupby('city_id')['city_population'].mean()
            population_weights = populations / populations.sum()
            # total_weights = {}
            # for city_id in cities_to_model:
            #     specific_city_df = citysite_rows[citysite_rows['city_id'] == city_id]
            #     population_weight = population_weights[city_id]
            #     long_term_avg_pct = specific_city_df['ctf_percent_of_waste_sent'].mean()
            #     total_weight = long_term_avg_pct * population_weight
            #     total_weights[city_id] = total_weight
            # total_weights = pd.Series(total_weights) / 100
            # total_weights /= total_weights.sum()
            # # Go through and add real data on top of the average time series

            # fraction_of_waste_vectors = {}
            # for city_id in cities_to_model:
            #     fraction_of_waste_vectors[city_id] = pd.Series(total_weights[city_id], index=self.years_range)
            # fraction_of_waste_df = pd.DataFrame(fraction_of_waste_vectors)

            # for city_id in cities_to_model:
            #     specific_city_df = citysite_rows[citysite_rows['city_id'] == city_id]
            #     population_weight = population_weights[city_id]
            #     pct_sent_annual = specific_city_df.groupby('ctf_year')['ctf_percent_of_waste_sent'].mean() / 100
            #     pct_sent_mean = pct_sent_annual.mean()
            #     diffs = (pct_sent_annual - pct_sent_mean)
            #     fraction_of_waste_df.loc[pct_sent_annual.index, city_id] += diffs.values * population_weight
            #     fraction_of_waste_df.loc[pct_sent_annual.index, ~city_id] -= diffs.values * population_weight

            # 1) Population weights per city (pick how you choose the city population)
            # If you have city_population_year, use the latest per city:
            pop = (citysite_rows.sort_values('city_population_year')
                                .groupby('city_id')['city_population'].last())
            # Otherwise, your .mean() is OK:
            # pop = citysite_rows.groupby('city_id')['city_population'].mean()

            pop = pop.dropna()
            pop_w = pop / pop.sum()                         # weights sum to 1

            # 2) Average % sent per city (as fraction 0..1)
            avg_pct = (citysite_rows.groupby('city_id')['ctf_percent_of_waste_sent']
                    .mean().div(100.0))
            avg_pct = avg_pct.reindex(pop_w.index).fillna(0.0)

            # 3) Observed % by (year, city) pivot (as fraction 0..1)
            obs = (citysite_rows.groupby(['ctf_year','city_id'])['ctf_percent_of_waste_sent']
                .mean().div(100.0)).unstack('city_id')

            # Align to your target years and city set
            obs = obs.reindex(index=self.years_range, columns=pop_w.index)

            # 4) Fill missing observations with that city's average
            P = obs.fillna(avg_pct)                          # per-year, per-city percent-to-this-landfill

            # 5) Convert to shares of *landfill inflow* using population proxy and renormalize each year
            W = P.mul(pop_w, axis=1)                         # weight_i,t = pop_w_i * pct_i,t
            # row_sums = W.sum(axis=1)
            # # Identify years with no information (sum == 0 or all NaN)
            # zero_or_nan_rows = row_sums.isna() | (np.isclose(row_sums, 0.0))
            # if zero_or_nan_rows.any():
            #     # Fill those years with population weights so rows still sum to 1
            #     # Ensure pop_w aligns to columns
            #     pop_w_aligned = pop_w.reindex(W.columns).fillna(0.0)
            #     # Assign the same weights across the problematic rows
            #     W.loc[zero_or_nan_rows, :] = pop_w_aligned
            fraction_of_waste_df = W.div(W.sum(axis=1), axis=0).fillna(0.0)
            baseline.fraction_of_waste_df = fraction_of_waste_df

            for city_id in cities_to_model:
                new_landfill = Landfill(
                    open_date=open_date,
                    close_date=close_date,
                    site_type=site_type,
                    mcf=pd.Series(mcf, index=self.years_range),
                    city_params_dict=city_params_dict,
                    city_instance_attrs=baseline.city_instance_attrs,
                    landfill_index=0,
                    gas_capture=gas_capture_presence,
                    scenario=0,
                    new_baseline=True,
                    gas_capture_efficiency=gas_capture_efficiency,
                    flaring=flaring_series,
                    # leachate_circulate=leachate_circulate[i],
                    fraction_of_waste_vector=fraction_of_waste_df[city_id],
                    advanced=True,
                    latlon=(self.latitude, self.longitude),
                    ks=baseline.ks,
                    oxidation_factor=oxidation_series,
                    rmi_id=id,
                    city_id=city_id,
                )
                baseline.landfills.append(new_landfill)
        
            
        baseline.repopulate_attr_dicts()
        for i, landfill in enumerate(baseline.landfills):
            # Might be able to do this more efficienctly...i'm looping over the pre implementation years twice sort of
            landfill.waste_mass_df = LandfillWasteMassDF.create_advanced(
                waste_generated_df=baseline.waste_generated_df.df,
                divs_df=baseline.divs_df,
                fraction_of_waste_series=landfill.fraction_of_waste_vector,
            ).df

        # scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in baseline.landfills:
            landfill.estimate_emissions(skip_ox=True, trace_monthly=True)

        self.estimate_diversion_emissions(scenario=0)
        self.sum_landfill_emissions(scenario=0, trace_monthly=True)

        for landfill in baseline.landfills:
            landfill.emissions = landfill.emissions.apply(self.convert_methane_m3_to_ton_co2e) / 28

    def import_basics(self, row) -> None:
        """
        Import basic parameters for a city.

        Args:
            row (tuple): row[0] is the index of the row in the dataframe used for input,
            row[1] is the row itself.
        Returns:
            None
        """

        data_source = row["population_data_source"]
        self.country = row["country"]
        self.iso3 = row["iso"]
        self.region = defaults_2019.region_lookup[self.country]
        year_of_data_pop = row["population_year"]
        assert np.isnan(year_of_data_pop) == False, "Population year is missing"
        year_of_data_msw = row["msw_collected_year"]
        if np.isnan(year_of_data_msw):
            year_of_data_msw = row["msw_generated_year"]
        if np.isnan(year_of_data_msw):
            year_of_data_msw = row["data_collection_year"].iloc[0]
        year_of_data_msw = int(year_of_data_msw)
        population = float(row["population_count"])
        growth_rate_historic = row["historic_growth_rate"]
        growth_rate_future = row["future_growth_rate"]

        # lat lon
        self.lat = row["latitude"]
        self.lon = row["longitude"]

        # Temperature and precipitation
        temperature = row["mean_yearly_temp_2000_2021"]
        precipitation = float(row["mean_yearly_precip_2000_2021"])
        precip_zone = defaults_2019.get_precipitation_zone(precipitation)

        waste_mass_defaults = False
        # Get waste total
        try:
            # waste_mass_load = float(
            #     row["msw_generated_metric_tons_per_year"]
            # )  # unit is tons
            # if np.isnan(waste_mass_load):
            waste_mass_load = float(row["msw_collected_metric_tons_per_year"])
            waste_per_capita = (
                waste_mass_load * 1000 / population / 365
            )  # unit is kg/person/day
        except:
            # waste_mass_load = float(
            #     row["msw_generated_metric_tons_per_year"].replace(",", "")
            # )
            # if np.isnan(self.waste_mass_load):
            waste_mass_load = float(
                row["msw_collected_metric_tons_per_year"].replace(",", "")
            )
            waste_per_capita = waste_mass_load * 1000 / population / 365
        if waste_mass_load != waste_mass_load:
            # Use per capita default
            waste_mass_defaults = True
            if self.iso3 in defaults_2019.msw_per_capita_country:
                waste_per_capita = defaults_2019.msw_per_capita_country[self.iso3]
                year_of_data_msw = 2019
            else:
                waste_per_capita = defaults_2019.msw_per_capita_defaults[self.region]
                year_of_data_msw = 2019
            waste_mass_load = waste_per_capita * population / 1000 * 365

        # Subtract mass that is informally collected
        # self.informal_fraction = np.nan_to_num(row['percent_informal_sector_percent_collected_by_informal_sector_percent']) / 100
        # self.waste_mass = self.waste_mass_load * (1 - self.informal_fraction)
        waste_mass = waste_mass_load

        # Adjust waste mass to account for difference in reporting years between msw and population
        # if self.data_source == 'World Bank':
        if year_of_data_msw != year_of_data_pop:
            year_difference = year_of_data_pop - year_of_data_msw
            if year_of_data_msw < year_of_data_pop:
                waste_mass *= growth_rate_historic**year_difference
                waste_per_capita = waste_mass * 1000 / population / 365
            else:
                waste_mass *= growth_rate_future**year_difference
                waste_per_capita = waste_mass * 1000 / population / 365

        # Waste fractions
        waste_fractions = row[
            [
                "composition_food_organic_waste_percent",
                "composition_yard_garden_green_waste_percent",
                "composition_wood_percent",
                "composition_paper_cardboard_percent",
                "composition_plastic_percent",
                "composition_metal_percent",
                "composition_glass_percent",
                "composition_other_percent",
                "composition_rubber_leather_percent",
                "composition_textiles_percent",
            ]
        ]

        waste_fractions.rename(
            index={
                "composition_food_organic_waste_percent": "food",
                "composition_yard_garden_green_waste_percent": "green",
                "composition_wood_percent": "wood",
                "composition_paper_cardboard_percent": "paper_cardboard",
                "composition_plastic_percent": "plastic",
                "composition_metal_percent": "metal",
                "composition_glass_percent": "glass",
                "composition_other_percent": "other",
                "composition_rubber_leather_percent": "rubber",
                "composition_textiles_percent": "textiles",
            },
            inplace=True,
        )
        waste_fractions /= 100

        # Add zeros where there are no values unless all values are nan, in which case use defaults
        waste_fractions_defaults = False
        if waste_fractions.isna().all():
            waste_fractions_defaults = True
            waste_fractions = defaults_2019.waste_composition_for(self.iso3, self.region)
        else:
            waste_fractions.fillna(0, inplace=True)
            # waste_fractions['textiles'] = 0

        if (waste_fractions.sum() < 0.98) or (waste_fractions.sum() > 1.02):
            waste_fractions_defaults = True
            # print('waste fractions do not sum to 1')
            waste_fractions = defaults_2019.waste_composition_for(self.iso3, self.region)

        waste_fractions = waste_fractions.to_dict()

        # Normalize waste fractions to sum to 1
        s = sum([x for x in waste_fractions.values()])
        waste_fractions = {x: waste_fractions[x] / s for x in waste_fractions.keys()}
        self.years_range = range(1990, 2051)
        waste_fractions = pd.DataFrame(waste_fractions, index=self.years_range)
        waste_masses = waste_mass * waste_fractions

        try:
            # Calculate MEF for compost -- emissions from composted waste
            mef_compost = (
                (
                    0.0055
                    * waste_fractions["food"]
                    / (waste_fractions["food"] + waste_fractions["green"])
                    + 0.0139
                    * waste_fractions["green"]
                    / (waste_fractions["food"] + waste_fractions["green"])
                )
                * 1.1023
                * 0.7
            )  # / 28
            mef_compost = mef_compost.at[2000]
        except:
            mef_compost = 0

        # Model components
        self.components = set(["food", "green", "wood", "paper_cardboard", "textiles"])
        self.compost_components = set(
            ["food", "green", "wood", "paper_cardboard"]
        )  # Double check we don't want to include paper
        self.anaerobic_components = set(["food", "green", "wood", "paper_cardboard"])
        self.combustion_components = set(
            [
                "food",
                "green",
                "wood",
                "paper_cardboard",
                "textiles",
                "plastic",
                "rubber",
            ]
        )
        self.recycling_components = set(
            [
                "wood",
                "paper_cardboard",
                "textiles",
                "plastic",
                "rubber",
                "metal",
                "glass",
                "other",
            ]
        )

        self.div_components = {}
        self.div_components["compost"] = self.compost_components
        self.div_components["anaerobic"] = self.anaerobic_components
        self.div_components["combustion"] = self.combustion_components
        self.div_components["recycling"] = self.recycling_components

        # Calculate waste generated, which is like waste masses but adjusts for population growth
        waste_generated_df = WasteGeneratedDF.create(
            waste_masses,
            1990,
            2050,
            year_of_data_pop,
            growth_rate_historic,
            growth_rate_future,
        )

        return {
            "data_source_pop": data_source,
            "year_of_data_pop": year_of_data_pop,
            "year_of_data_msw": year_of_data_msw,
            "population": population,
            "growth_rate_historic": growth_rate_historic,
            "growth_rate_future": growth_rate_future,
            "waste_mass": waste_mass,
            "waste_per_capita": waste_per_capita,
            "waste_fractions": waste_fractions,
            "waste_mass_defaults": waste_mass_defaults,
            "waste_fractions_defaults": waste_fractions_defaults,
            "mef_compost": mef_compost,
            "precip": precipitation,
            "precip_zone": precip_zone,
            "temperature": temperature,
            "waste_masses": waste_masses,
            "waste_generated_df": waste_generated_df,
            "latitude": self.lat,
            "longitude": self.lon,
        }
    
    def import_basics_site(self, canonical_row, pop_data, usecase='wastemap', time_series_rows=None) -> None:
        """
        Import basic parameters for a city.

        Args:
            row (tuple): row[0] is the index of the row in the dataframe used for input,
            row[1] is the row itself.
        Returns:
            None
        """
        if usecase == "trace":
            #data_source_waste = row['area_source']
            self.iso3 = canonical_row["iso3_country"]
            iso3s = pd.read_csv(defaults_2019._find_iso3_csv())
            self.country = iso3s[iso3s['iso3'] == self.iso3]['name'].values[0]
            self.region = defaults_2019.region_lookup[self.country]
            population = 100
            
            current_year = datetime.now().year
            general_reference_year = current_year - 1

            # Check if row is a single row or multiple rows
            if (time_series_rows is None) or isinstance(time_series_rows, pd.Series):
                # Single row - use old method
                year_of_data_msw = canonical_row['incoming_waste_year']
                if np.isnan(year_of_data_msw):
                    close_date = canonical_row['site_close_year']
                    if not np.isnan(close_date) and close_date < general_reference_year:
                        year_of_data_msw = close_date - 1
                    else:
                        year_of_data_msw = general_reference_year
                year_of_data_pop = year_of_data_msw
                growth_rate_historic = pop_data.at[self.iso3, 'growth_rate_historic']
                growth_rate_future = pop_data.at[self.iso3, 'growth_rate_future']

                # Get waste total
                waste_mass_defaults = False
                waste_mass_load = float(canonical_row['annual_incoming_waste'])  # unit is tons
                if np.isnan(waste_mass_load):
                    waste_mass_defaults = True
                    if self.iso3 in defaults_2019.msw_per_capita_country:
                        waste_per_capita = defaults_2019.msw_per_capita_country[self.iso3]
                        year_of_data_msw = 2019
                    else:
                        waste_per_capita = defaults_2019.msw_per_capita_defaults[self.region]
                        year_of_data_msw = 2019
                    waste_mass_load = waste_per_capita * population / 1000 * 365
                waste_per_capita = waste_mass_load * 1000 / population / 365
                waste_mass = waste_mass_load

                # Adjust waste mass to account for difference in reporting years between msw and population
                if year_of_data_msw != year_of_data_pop:
                    year_difference = year_of_data_pop - year_of_data_msw
                    if year_of_data_msw < year_of_data_pop:
                        waste_mass *= growth_rate_historic**year_difference
                        waste_per_capita = waste_mass * 1000 / population / 365
                    else:
                        waste_mass *= growth_rate_future**year_difference
                        waste_per_capita = waste_mass * 1000 / population / 365
                    
            else:
                # Multiple rows - create time series
                waste_mass_defaults = False
                year_of_data_msw = time_series_rows['incoming_waste_year'].max()

                # Get rows with incoming_waste_year >= 2010
                recent_rows = time_series_rows[time_series_rows['incoming_waste_year'] >= 2010]

                if len(recent_rows) > 0:
                    # Calculate average annual incoming waste for recent years
                    avg_annual_waste = recent_rows['annual_incoming_waste'].mean()
                else:
                    # Fallback if no recent data
                    avg_annual_waste = time_series_rows['annual_incoming_waste'].mean()
                if np.isnan(avg_annual_waste):
                    avg_annual_waste = canonical_row['annual_incoming_waste']

                # Get growth rates
                growth_rate_historic = pop_data.at[self.iso3, 'growth_rate_historic']
                growth_rate_future = pop_data.at[self.iso3, 'growth_rate_future']

                # Create time series from 1990 to 2050
                years = np.arange(1990, 2051)
                waste_series = np.zeros(len(years))

                # Set 2020 as the baseline year
                baseline_year = 2020
                baseline_idx = baseline_year - 1990

                # Project backward and forwards
                for i, year in enumerate(years):
                    if year >= baseline_year:
                        # Future projection
                        years_since_baseline = year - baseline_year
                        waste_series[i] = avg_annual_waste * (growth_rate_future ** years_since_baseline)
                    else:
                        # Past projection - divide by historic growth rate (going backwards)
                        years_since_baseline = baseline_year - year
                        waste_series[i] = avg_annual_waste / (growth_rate_historic ** years_since_baseline)

                # Overwrite with actual data where available
                for _, data_row in time_series_rows.iterrows():
                    if not pd.isna(data_row['incoming_waste_year']):
                        year_idx = int(data_row['incoming_waste_year']) - 1990
                        if 0 <= year_idx < len(waste_series):
                            val = data_row['annual_incoming_waste']
                            if not np.isnan(val):
                                waste_series[year_idx] = val

                # Store the time series
                self.waste_time_series = pd.Series(waste_series, index=years)
                waste_mass = self.waste_time_series
                waste_per_capita = waste_mass * 1000 / population / 365
                
            # Location data (canonical_row is typically a Series for TRACE usecase)
            try:
                geometry = canonical_row['location']
                self.lon = float(geometry.x)
                self.lat = float(geometry.y)
            except Exception:
                try:
                    self.lon = float(canonical_row['longitude'])
                    self.lat = float(canonical_row['latitude'])
                except Exception:
                    # Back-compat: some callers pass a 1-row DataFrame
                    self.lon = float(canonical_row.iloc[0]['longitude'])
                    self.lat = float(canonical_row.iloc[0]['latitude'])

            year_of_data_pop = year_of_data_msw  # Assume population data is from the same year as waste data if not provided

            # Temperature and precipitation
            # Prefer caller-provided weather values (e.g., TRACE pipeline already joined
            # precipitation/temperature onto each site). Only fall back to DB lookup
            # for legacy callers that do not provide these fields.

            def _get_canonical_value(row_like, key: str):
                try:
                    if isinstance(row_like, dict):
                        return row_like.get(key)
                    if isinstance(row_like, pd.Series):
                        return row_like.get(key)
                    if isinstance(row_like, pd.DataFrame):
                        if key in row_like.columns and len(row_like) > 0:
                            return row_like.iloc[0][key]
                    # Fallback to __getitem__ for other row-likes
                    return row_like[key]
                except Exception:
                    return None

            _precip_provided = _get_canonical_value(canonical_row, 'precipitation')
            _temp_provided = _get_canonical_value(canonical_row, 'temperature')

            # If either key exists on the input row, treat it as authoritative and
            # skip the DB weather query entirely (even if values are NaN).
            #
            # NOTE: for pandas Series, `'x' in series` checks the index labels;
            # for DataFrame it checks columns; for dict it checks keys.
            _has_precip_key = False
            _has_temp_key = False
            try:
                _has_precip_key = 'precipitation' in canonical_row
                _has_temp_key = 'temperature' in canonical_row
            except Exception:
                _has_precip_key = False
                _has_temp_key = False

            if _has_precip_key or _has_temp_key:
                try:
                    precipitation = float(_precip_provided) if _precip_provided is not None and not pd.isna(_precip_provided) else np.nan
                except Exception:
                    precipitation = np.nan
                try:
                    temperature = float(_temp_provided) if _temp_provided is not None and not pd.isna(_temp_provided) else np.nan
                except Exception:
                    temperature = np.nan
                try:
                    precip_zone = defaults_2019.get_precipitation_zone(precipitation) if not np.isnan(precipitation) else None
                except Exception:
                    precip_zone = None
            else:
                # SQL query to get average precipitation and temperature using provided latitude and longitude
                QUERY_WEATHER = """
                WITH city_selection AS (
                    SELECT
                        'CustomCity' AS name,
                        :latitude AS latitude,
                        :longitude AS longitude
                ),
                global_weather_table AS (
                    SELECT
                        cs.name,
                        ROUND(AVG(value) FILTER (WHERE weather_type = 'precipitation')::numeric, 2) AS avg_total_precip,
                        ROUND(AVG(value) FILTER (WHERE weather_type = 'temperature')::numeric, 2) AS avg_temperature
                    FROM global_weather_data gwd
                    JOIN city_selection cs
                        ON gwd.weather_type IN ('precipitation', 'temperature')
                       AND gwd.bbox_geometry && ST_Buffer(
                            ST_SetSRID(ST_MakePoint(cs.longitude, cs.latitude), 4326),
                            0.5   -- degrees of buffer; tweak smaller/larger as needed
                       )
                       AND ST_Intersects(
                            gwd.bbox_geometry,
                            ST_Buffer(
                                ST_SetSRID(ST_MakePoint(cs.longitude, cs.latitude), 4326),
                                0.5
                            )
                       )
                    GROUP BY cs.name
                )
                SELECT * FROM global_weather_table;
                """
                from dotenv import load_dotenv
                # Try env vars first (Azure sets DB_HOST, DB_USER, etc.)
                DB_SERVER_IP = os.getenv("host") or os.getenv("DB_HOST")
                DB_PORT = int(os.getenv("port") or os.getenv("DB_PORT") or "5432")
                DB_USER = os.getenv("user") or os.getenv("DB_USER")
                DB_PASSWORD = os.getenv("password") or os.getenv("DB_PASSWORD")
                DB_NAME = os.getenv("dbname") or os.getenv("DB_NAME") or "postgres"
                # Fallback: local run, load from env file at known path
                if not all([DB_SERVER_IP, DB_USER, DB_PASSWORD]):
                    load_dotenv('/Users/hugh/Library/CloudStorage/OneDrive-RMI/Documents/RMI/SWEET/hugh.env')
                    DB_SERVER_IP = os.getenv("host") or os.getenv("DB_HOST")
                    DB_PORT = int(os.getenv("port") or os.getenv("DB_PORT") or "5432")
                    DB_USER = os.getenv("user") or os.getenv("DB_USER")
                    DB_PASSWORD = os.getenv("password") or os.getenv("DB_PASSWORD")
                    DB_NAME = os.getenv("dbname") or os.getenv("DB_NAME") or "postgres"
                if not all([DB_SERVER_IP, DB_USER, DB_PASSWORD]):
                    raise ValueError(
                        "Missing database credentials. Set DB_HOST, DB_USER, DB_PASSWORD (or host, user, password)."
                    )
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                DB_SSLMODE = ssl_context

                # Create the SQLAlchemy engine
                def _is_transient_db_error(err: Exception) -> bool:
                    msg = str(err).lower()
                    return (
                        "ssl syscall error" in msg
                        or "eof detected" in msg
                        or "server closed the connection" in msg
                        or "connection reset" in msg
                        or "in recovery mode" in msg
                        or "terminating connection" in msg
                    )

                max_attempts = int(os.getenv("DB_QUERY_MAX_ATTEMPTS", "5"))
                base_sleep_s = float(os.getenv("DB_QUERY_RETRY_BASE_SECONDS", "1.5"))
                statement_timeout_ms = int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "120000"))
                pool_recycle_s = int(os.getenv("DB_POOL_RECYCLE_SECONDS", "1800"))
                connect_timeout_s = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10"))

                weather_data = None
                last_exc: Exception | None = None
                for attempt in range(1, max_attempts + 1):
                    engine = create_engine(
                        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_SERVER_IP}:{DB_PORT}/{DB_NAME}",
                        connect_args={
                            "sslmode": "require",
                            "connect_timeout": connect_timeout_s,
                            "keepalives": 1,
                            "keepalives_idle": 30,
                            "keepalives_interval": 10,
                            "keepalives_count": 5,
                        },
                        pool_pre_ping=True,
                        pool_recycle=pool_recycle_s,
                    )
                    try:
                        with engine.connect() as connection:
                            try:
                                connection.execute(text("SET statement_timeout = :t"), {"t": statement_timeout_ms})
                            except Exception:
                                pass
                            result = connection.execute(
                                text(QUERY_WEATHER),
                                {"latitude": self.lat, "longitude": self.lon}
                            )
                            weather_data = result.mappings().fetchone()
                        last_exc = None
                        break
                    except (psycopg2.OperationalError, SQLAlchemyOperationalError) as e:
                        last_exc = e
                        if attempt >= max_attempts or not _is_transient_db_error(e):
                            break
                        sleep_s = base_sleep_s * (2 ** (attempt - 1))
                        print(f"Transient DB error during weather lookup (attempt {attempt}/{max_attempts}): {e}; retrying in {sleep_s:.1f}s")
                        time.sleep(sleep_s)
                    finally:
                        try:
                            engine.dispose()
                        except Exception:
                            pass

                if weather_data is None and last_exc is not None:
                    raise last_exc

                # Process the weather data
                if weather_data:
                    precipitation = float(weather_data["avg_total_precip"])
                    temperature = float(weather_data["avg_temperature"])
                    precip_zone = defaults_2019.get_precipitation_zone(precipitation)
                else:
                    precipitation = np.nan
                    temperature = np.nan
                    precip_zone = None

            # Waste fractions
            waste_fractions_defaults = True
            waste_fractions = defaults_2019.waste_composition_for(self.iso3, self.region)

            # Normalize waste fractions to sum to 1
            wf_norm = waste_fractions / waste_fractions.sum()
            waste_fractions = pd.DataFrame(
                np.repeat(wf_norm.values[np.newaxis, :], len(self.years_range), axis=0),
                index=self.years_range,
                columns=wf_norm.index
            )
            waste_fractions = waste_fractions.loc[1990:2050]
            waste_masses = waste_fractions.multiply(waste_mass, axis=0)

            try:
                # Calculate MEF for compost -- emissions from composted waste
                mef_compost = (
                    (
                        0.0055
                        * waste_fractions["food"]
                        / (waste_fractions["food"] + waste_fractions["green"])
                        + 0.0139
                        * waste_fractions["green"]
                        / (waste_fractions["food"] + waste_fractions["green"])
                    )
                    * 1.1023
                    * 0.7
                )  # / 28
                mef_compost = mef_compost.at[2000]
            except:
                mef_compost = 0

            # Model components
            self.components = set(["food", "green", "wood", "paper_cardboard", "textiles"])
            self.compost_components = set(
                ["food", "green", "wood", "paper_cardboard"]
            )  # Double check we don't want to include paper
            self.anaerobic_components = set(["food", "green", "wood", "paper_cardboard"])
            self.combustion_components = set(
                [
                    "food",
                    "green",
                    "wood",
                    "paper_cardboard",
                    "textiles",
                    "plastic",
                    "rubber",
                ]
            )
            self.recycling_components = set(
                [
                    "wood",
                    "paper_cardboard",
                    "textiles",
                    "plastic",
                    "rubber",
                    "metal",
                    "glass",
                    "other",
                ]
            )

            self.div_components = {}
            self.div_components["compost"] = self.compost_components
            self.div_components["anaerobic"] = self.anaerobic_components
            self.div_components["combustion"] = self.combustion_components
            self.div_components["recycling"] = self.recycling_components

            # Calculate waste generated, which is like waste masses but adjusts for population growth
            waste_generated_df = WasteGeneratedDF.create(
                waste_masses.loc[1990:2050, :],
                1990,
                2050,
                year_of_data_pop,
                growth_rate_historic,
                growth_rate_future,
            )

            return {
                "data_source_pop": 'UN',
                "year_of_data_pop": year_of_data_pop,
                "year_of_data_msw": year_of_data_msw,
                "population": population,
                "growth_rate_historic": growth_rate_historic,
                "growth_rate_future": growth_rate_future,
                "waste_mass": waste_mass,
                "waste_per_capita": waste_per_capita,
                "waste_fractions": waste_fractions,
                "waste_mass_defaults": waste_mass_defaults,
                "waste_fractions_defaults": waste_fractions_defaults,
                "mef_compost": mef_compost,
                "precip": precipitation,
                "precip_zone": precip_zone,
                "temperature": temperature,
                "waste_masses": waste_masses,
                "waste_generated_df": waste_generated_df,
            }
        else:
            data_source_waste = canonical_row['Data Source: Waste'].values[0]
            self.country = canonical_row["Country"].values[0]
            self.iso3 = canonical_row["Country ISO3"].values[0]
            self.region = defaults_2019.region_lookup[self.country]
            year_of_data_pop = 2025 #row["population_year"]
            assert np.isnan(year_of_data_pop) == False, "Population year is missing"
            try:
                year_of_data_msw = int(canonical_row["Waste in Place Year"].values[0])
            except:
                year_of_data_msw = 2025
            population = 100
            growth_rate_historic = pop_data.at[self.iso3, 'growth_rate_historic']
            growth_rate_future = pop_data.at[self.iso3, 'growth_rate_future']

            # lat lon
            self.lon = float(canonical_row['Longitude'].iloc[0])
            self.lat = float(canonical_row['Latitude'].iloc[0])

            # Temperature and precipitation
            # SQL query to get average precipitation and temperature using provided latitude and longitude
            QUERY_WEATHER = """
            WITH city_selection AS (
                SELECT
                    'CustomCity' AS name,
                    :latitude AS latitude,
                    :longitude AS longitude
            ),
            global_weather_table AS (
                SELECT
                    cs.name,
                    ROUND(AVG(value) FILTER (WHERE weather_type = 'precipitation')::numeric, 2) AS avg_total_precip,
                    ROUND(AVG(value) FILTER (WHERE weather_type = 'temperature')::numeric, 2) AS avg_temperature
                FROM global_weather_data gwd
                JOIN city_selection cs
                    ON gwd.weather_type IN ('precipitation', 'temperature')
                   AND gwd.bbox_geometry && ST_Buffer(
                        ST_SetSRID(ST_MakePoint(cs.longitude, cs.latitude), 4326),
                        0.5   -- degrees of buffer; tweak smaller/larger as needed
                   )
                   AND ST_Intersects(
                        gwd.bbox_geometry,
                        ST_Buffer(
                            ST_SetSRID(ST_MakePoint(cs.longitude, cs.latitude), 4326),
                            0.5
                        )
                   )
                GROUP BY cs.name
            )
            SELECT * FROM global_weather_table;
            """
            from dotenv import load_dotenv
            # Try env vars first (Azure sets DB_HOST, DB_USER, etc.)
            DB_SERVER_IP = os.getenv("host") or os.getenv("DB_HOST")
            DB_PORT = int(os.getenv("port") or os.getenv("DB_PORT") or "5432")
            DB_USER = os.getenv("user") or os.getenv("DB_USER")
            DB_PASSWORD = os.getenv("password") or os.getenv("DB_PASSWORD")
            DB_NAME = os.getenv("dbname") or os.getenv("DB_NAME") or "postgres"
            # Fallback: local run, load from env file at known path
            if not all([DB_SERVER_IP, DB_USER, DB_PASSWORD]):
                load_dotenv('/Users/hugh/Library/CloudStorage/OneDrive-RMI/Documents/RMI/SWEET/hugh.env')
                DB_SERVER_IP = os.getenv("host") or os.getenv("DB_HOST")
                DB_PORT = int(os.getenv("port") or os.getenv("DB_PORT") or "5432")
                DB_USER = os.getenv("user") or os.getenv("DB_USER")
                DB_PASSWORD = os.getenv("password") or os.getenv("DB_PASSWORD")
                DB_NAME = os.getenv("dbname") or os.getenv("DB_NAME") or "postgres"
            if not all([DB_SERVER_IP, DB_USER, DB_PASSWORD]):
                raise ValueError(
                    "Missing database credentials. Set DB_HOST, DB_USER, DB_PASSWORD (or host, user, password)."
                )
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            DB_SSLMODE = ssl_context

            # Create the SQLAlchemy engine
            max_attempts = int(os.getenv("DB_QUERY_MAX_ATTEMPTS", "5"))
            base_sleep_s = float(os.getenv("DB_QUERY_RETRY_BASE_SECONDS", "1.5"))
            statement_timeout_ms = int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "120000"))
            pool_recycle_s = int(os.getenv("DB_POOL_RECYCLE_SECONDS", "1800"))
            connect_timeout_s = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10"))

            weather_data = None
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                engine = create_engine(
                    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_SERVER_IP}:{DB_PORT}/{DB_NAME}",
                    connect_args={
                        "sslmode": "require",
                        "connect_timeout": connect_timeout_s,
                        "keepalives": 1,
                        "keepalives_idle": 30,
                        "keepalives_interval": 10,
                        "keepalives_count": 5,
                    },
                    pool_pre_ping=True,
                    pool_recycle=pool_recycle_s,
                )
                try:
                    with engine.connect() as connection:
                        try:
                            connection.execute(text("SET statement_timeout = :t"), {"t": statement_timeout_ms})
                        except Exception:
                            pass
                        result = connection.execute(
                            text(QUERY_WEATHER),
                            {"latitude": self.lat, "longitude": self.lon}
                        )
                        weather_data = result.mappings().fetchone()
                    last_exc = None
                    break
                except (psycopg2.OperationalError, SQLAlchemyOperationalError) as e:
                    last_exc = e
                    msg = str(e).lower()
                    transient = (
                        "ssl syscall error" in msg
                        or "eof detected" in msg
                        or "server closed the connection" in msg
                        or "connection reset" in msg
                        or "in recovery mode" in msg
                        or "terminating connection" in msg
                    )
                    if attempt >= max_attempts or not transient:
                        break
                    sleep_s = base_sleep_s * (2 ** (attempt - 1))
                    print(f"Transient DB error during weather lookup (attempt {attempt}/{max_attempts}): {e}; retrying in {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                finally:
                    try:
                        engine.dispose()
                    except Exception:
                        pass

            if weather_data is None and last_exc is not None:
                raise last_exc

            # Process the weather data
            if weather_data:
                precipitation = float(weather_data["avg_total_precip"])
                temperature = float(weather_data["avg_temperature"])
                precip_zone = defaults_2019.get_precipitation_zone(precipitation)
            else:
                precipitation = np.nan
                temperature = np.nan
                precip_zone = None

            # Get waste total
            waste_mass_defaults = False
            waste_mass_load = float(
                canonical_row['Waste Accepted (tons/year)'].values[0]
            )  # unit is tons
            if np.isnan(waste_mass_load):
                waste_mass_defaults = True
                if self.iso3 in defaults_2019.msw_per_capita_country:
                    waste_per_capita = defaults_2019.msw_per_capita_country[self.iso3]
                    year_of_data_msw = 2019
                else:
                    waste_per_capita = defaults_2019.msw_per_capita_defaults[self.region]
                    year_of_data_msw = 2019
                waste_mass_load = waste_per_capita * population / 1000 * 365
            waste_per_capita = waste_mass_load * 1000 / population / 365
            waste_mass = waste_mass_load

            # Adjust waste mass to account for difference in reporting years between msw and population
            if year_of_data_msw != year_of_data_pop:
                year_difference = year_of_data_pop - year_of_data_msw
                if year_of_data_msw < year_of_data_pop:
                    waste_mass *= growth_rate_historic**year_difference
                    waste_per_capita = waste_mass * 1000 / population / 365
                else:
                    waste_mass *= growth_rate_future**year_difference
                    waste_per_capita = waste_mass * 1000 / population / 365

            # Waste fractions
            waste_fractions_defaults = True
            waste_fractions = defaults_2019.waste_composition_for(self.iso3, self.region)

            # Normalize waste fractions to sum to 1
            wf_norm = waste_fractions / waste_fractions.sum()
            waste_fractions = pd.DataFrame(
                np.repeat(wf_norm.values[np.newaxis, :], len(self.years_range), axis=0),
                index=self.years_range,
                columns=wf_norm.index
            )
            waste_masses = waste_mass * waste_fractions

            try:
                # Calculate MEF for compost -- emissions from composted waste
                mef_compost = (
                    (
                        0.0055
                        * waste_fractions["food"]
                        / (waste_fractions["food"] + waste_fractions["green"])
                        + 0.0139
                        * waste_fractions["green"]
                        / (waste_fractions["food"] + waste_fractions["green"])
                    )
                    * 1.1023
                    * 0.7
                )  # / 28
                mef_compost = mef_compost.at[2000]
            except:
                mef_compost = 0

            # Model components
            self.components = set(["food", "green", "wood", "paper_cardboard", "textiles"])
            self.compost_components = set(
                ["food", "green", "wood", "paper_cardboard"]
            )  # Double check we don't want to include paper
            self.anaerobic_components = set(["food", "green", "wood", "paper_cardboard"])
            self.combustion_components = set(
                [
                    "food",
                    "green",
                    "wood",
                    "paper_cardboard",
                    "textiles",
                    "plastic",
                    "rubber",
                ]
            )
            self.recycling_components = set(
                [
                    "wood",
                    "paper_cardboard",
                    "textiles",
                    "plastic",
                    "rubber",
                    "metal",
                    "glass",
                    "other",
                ]
            )

            self.div_components = {}
            self.div_components["compost"] = self.compost_components
            self.div_components["anaerobic"] = self.anaerobic_components
            self.div_components["combustion"] = self.combustion_components
            self.div_components["recycling"] = self.recycling_components

            # Calculate waste generated, which is like waste masses but adjusts for population growth
            waste_generated_df = WasteGeneratedDF.create(
                waste_masses,
                1990,
                2050,
                year_of_data_pop,
                growth_rate_historic,
                growth_rate_future,
            )

            return {
                "data_source_pop": 'UN',
                "year_of_data_pop": year_of_data_pop,
                "year_of_data_msw": year_of_data_msw,
                "population": population,
                "growth_rate_historic": growth_rate_historic,
                "growth_rate_future": growth_rate_future,
                "waste_mass": waste_mass,
                "waste_per_capita": waste_per_capita,
                "waste_fractions": waste_fractions,
                "waste_mass_defaults": waste_mass_defaults,
                "waste_fractions_defaults": waste_fractions_defaults,
                "mef_compost": mef_compost,
                "precip": precipitation,
                "precip_zone": precip_zone,
                "temperature": temperature,
                "waste_masses": waste_masses,
                "waste_generated_df": waste_generated_df,
            }

    def import_div_fractions(self, row, waste_fractions, waste_generated_df) -> None:
        """
        Import diversion fractions for a city.

        Args:
            row (tuple): row[0] is the index of the row in the dataframe used for input,
            row[1] is the row itself.
        Returns:
            None
        """

        compost_fraction = 0 if pd.isna(row["waste_treatment_compost_percent"]) else float(row["waste_treatment_compost_percent"]) / 100
        anaerobic_fraction = 0 if pd.isna(row["waste_treatment_anaerobic_digestion_percent"]) else float(row["waste_treatment_anaerobic_digestion_percent"]) / 100
        value1 = float(row["waste_treatment_incineration_percent"])
        value2 = float(row["waste_treatment_advanced_thermal_treatment_percent"])
        if np.isnan(value1) and np.isnan(value2):
            combustion_fraction = 0
        else:
            combustion_fraction = (np.nan_to_num(value1) + np.nan_to_num(value2)) / 100
        recycling_fraction = 0 if pd.isna(row["waste_treatment_recycling_percent"]) else float(row["waste_treatment_recycling_percent"]) / 100

        # First case to check: all diversions and landfills are 0. Use defaults.
        diversion_defaults = False
        if np.isnan(compost_fraction):
            if self.iso3 in defaults_2019.fraction_composted_country:
                compost_fraction = defaults_2019.fraction_composted_country[self.iso3]
                diversion_defaults = True
            elif self.region in defaults_2019.fraction_composted:
                compost_fraction = defaults_2019.fraction_composted[self.region]
                diversion_defaults = True
            else:
                compost_fraction = 0.0

        if np.isnan(combustion_fraction):
            if self.iso3 in defaults_2019.fraction_incinerated_country:
                combustion_fraction = defaults_2019.fraction_incinerated_country[
                    self.iso3
                ]
                diversion_defaults = True
            elif self.region in defaults_2019.fraction_incinerated:
                combustion_fraction = defaults_2019.fraction_incinerated[self.region]
                diversion_defaults = True
            else:
                combustion_fraction = 0.0

        # Replace diversion NaN values with 0
        div_types = [
            compost_fraction,
            anaerobic_fraction,
            combustion_fraction,
            recycling_fraction,
        ]
        div_types = [0.0 if np.isnan(div) else div for div in div_types]
        (
            compost_fraction,
            anaerobic_fraction,
            combustion_fraction,
            recycling_fraction,
        ) = div_types
        div_fractions = pd.DataFrame(
            {
                "compost": compost_fraction,
                "anaerobic": anaerobic_fraction,
                "combustion": combustion_fraction,
                "recycling": recycling_fraction,
            },
            index=self.years_range,
        )

        def calculate_component_fractions(
            waste_fractions: pd.DataFrame, div_type: str
        ) -> pd.DataFrame:
            components = list(self.div_components[div_type])
            filtered_fractions = waste_fractions.loc[2000, components]
            total = filtered_fractions.sum()
            if total != 0:
                normalized_fractions = filtered_fractions / total
            else:
                normalized_fractions = pd.Series(0.0, index=filtered_fractions.index)
            return pd.DataFrame(
                [normalized_fractions] * len(self.years_range), index=self.years_range
            )

        div_component_fractions = DivComponentFractionsDF(
            compost=calculate_component_fractions(waste_fractions, "compost"),
            anaerobic=calculate_component_fractions(waste_fractions, "anaerobic"),
            combustion=calculate_component_fractions(waste_fractions, "combustion"),
            recycling=calculate_component_fractions(waste_fractions, "recycling"),
        )
        non_compostable_not_targeted_total = sum(
            [
                self.non_compostable_not_targeted[x]
                * div_component_fractions.compost.at[2000, x]
                for x in self.div_components["compost"]
            ]
        )
        non_compostable_not_targeted_total = pd.Series(
            non_compostable_not_targeted_total, index=self.years_range
        )
        if non_compostable_not_targeted_total.isna().all():
            non_compostable_not_targeted_total = pd.Series(0, index=self.years_range)

        diverted_masses = {}
        for div_type, df in div_component_fractions.model_dump().items():
            # Calculate the diverted masses for each waste type
            diverted_masses[div_type] = df.multiply(
                div_fractions.multiply(waste_generated_df.df.sum(axis=1), axis=0)[
                    div_type
                ],
                axis=0,
            )[list(self.div_components[div_type])]

        # Apply rejection rates to the diverted masses
        diverted_masses["compost"] = (
            diverted_masses["compost"]
            .multiply((1 - non_compostable_not_targeted_total), axis=0)
            .multiply((1 - pd.Series(self.unprocessable)), axis=1)
        )
        diverted_masses["combustion"] *= 1 - self.combustion_reject_rate
        for waste in diverted_masses["recycling"].columns:
            diverted_masses["recycling"][waste] *= self.recycling_reject_rates[waste]

        # Convert diverted masses to DivMassesAnnual
        divs = DivMassesAnnual(
            compost=diverted_masses["compost"],
            anaerobic=diverted_masses["anaerobic"],
            combustion=diverted_masses["combustion"],
            recycling=diverted_masses["recycling"],
        )

        return {
            "div_fractions": div_fractions,
            "diversion_defaults": diversion_defaults,
            "div_component_fractions": div_component_fractions,
            "non_compostable_not_targeted_total": non_compostable_not_targeted_total,
            "divs": divs,
        }
    
    def import_div_fractions_site(self, row, waste_fractions, waste_generated_df) -> None:
        """
        Import diversion fractions for a site.

        Args:
            row (tuple): row[0] is the index of the row in the dataframe used for input,
            row[1] is the row itself.
        Returns:
            None
        """

        compost_fraction = 0
        anaerobic_fraction = 0
        combustion_fraction = 0
        recycling_fraction = 0

        # Replace diversion NaN values with 0
        div_types = [
            compost_fraction,
            anaerobic_fraction,
            combustion_fraction,
            recycling_fraction,
        ]
        div_types = [0.0 if np.isnan(div) else div for div in div_types]
        (
            compost_fraction,
            anaerobic_fraction,
            combustion_fraction,
            recycling_fraction,
        ) = div_types
        div_fractions = pd.DataFrame(
            {
                "compost": compost_fraction,
                "anaerobic": anaerobic_fraction,
                "combustion": combustion_fraction,
                "recycling": recycling_fraction,
            },
            index=self.years_range,
        )

        def calculate_component_fractions(
            waste_fractions: pd.DataFrame, div_type: str
        ) -> pd.DataFrame:
            components = list(self.div_components[div_type])
            filtered_fractions = waste_fractions.loc[2000, components]
            total = filtered_fractions.sum()
            if total != 0:
                normalized_fractions = filtered_fractions / total
            else:
                normalized_fractions = pd.Series(0.0, index=filtered_fractions.index)
            return pd.DataFrame(
                [normalized_fractions] * len(self.years_range), index=self.years_range
            )

        div_component_fractions = DivComponentFractionsDF(
            compost=calculate_component_fractions(waste_fractions, "compost"),
            anaerobic=calculate_component_fractions(waste_fractions, "anaerobic"),
            combustion=calculate_component_fractions(waste_fractions, "combustion"),
            recycling=calculate_component_fractions(waste_fractions, "recycling"),
        )
        non_compostable_not_targeted_total = sum(
            [
                self.non_compostable_not_targeted[x]
                * div_component_fractions.compost.at[2000, x]
                for x in self.div_components["compost"]
            ]
        )
        non_compostable_not_targeted_total = pd.Series(
            non_compostable_not_targeted_total, index=self.years_range
        )
        if non_compostable_not_targeted_total.isna().all():
            non_compostable_not_targeted_total = pd.Series(0, index=self.years_range)

        diverted_masses = {}
        for div_type, df in div_component_fractions.model_dump().items():
            # Calculate the diverted masses for each waste type
            diverted_masses[div_type] = df.multiply(
                div_fractions.multiply(waste_generated_df.df.sum(axis=1), axis=0)[
                    div_type
                ],
                axis=0,
            )[list(self.div_components[div_type])]

        # Apply rejection rates to the diverted masses
        diverted_masses["compost"] = (
            diverted_masses["compost"]
            .multiply((1 - non_compostable_not_targeted_total), axis=0)
            .multiply((1 - pd.Series(self.unprocessable)), axis=1)
        )
        diverted_masses["combustion"] *= 1 - self.combustion_reject_rate
        for waste in diverted_masses["recycling"].columns:
            diverted_masses["recycling"][waste] *= self.recycling_reject_rates[waste]

        # Convert diverted masses to DivMassesAnnual
        divs = DivMassesAnnual(
            compost=diverted_masses["compost"],
            anaerobic=diverted_masses["anaerobic"],
            combustion=diverted_masses["combustion"],
            recycling=diverted_masses["recycling"],
        )

        return {
            "div_fractions": div_fractions,
            "diversion_defaults": False,
            "div_component_fractions": div_component_fractions,
            "non_compostable_not_targeted_total": non_compostable_not_targeted_total,
            "divs": divs,
        }

    def _calculate_divs(self, advanced_baseline=False, advanced_dst=False) -> None:
        """
        Calculate the diversion fractions and masses for a set of CityParameters.

        Args:
            advanced_baseline (bool): Whether to use advanced baseline parameters.
            advanced_dst (bool): Whether to use advanced diversion parameters.
        Returns:
            None
        """

        city_parameters = self.baseline_parameters
        city_parameters._singapore_k()

        # Create city-level dataframes
        start_year = 1990
        end_year = 2050
        years = range(start_year, end_year + 1)

        waste_masses_df = city_parameters.waste_fractions.multiply(
            city_parameters.waste_mass, axis=0
        )

        if isinstance(city_parameters.year_of_data_pop, dict):
            year_of_data_pop = city_parameters.year_of_data_pop["baseline"]
        else:
            year_of_data_pop = city_parameters.year_of_data_pop

        city_parameters.waste_generated_df = WasteGeneratedDF.create(
            waste_masses_df,
            start_year,
            end_year,
            year_of_data_pop,
            city_parameters.growth_rate_historic,
            city_parameters.growth_rate_future,
        ).df

        # if scenario == 0:
        #     self.baseline_parameters = city_parameters
        # else:
        #     self.scenario_parameters[scenario - 1] = city_parameters

        # Update other calculated attributes
        # self._calculate_waste_masses()
        self._calculate_diverted_masses()
        # city_parameters.divs_df = DivsDF.create(city_parameters.divs, start_year, end_year, city_parameters.year_of_data_pop, city_parameters.growth_rate_historic, city_parameters.growth_rate_future)
        city_parameters.divs_df = city_parameters.divs
        self._calculate_net_masses()

        city_params_dict = self.update_cityparams_dict(city_parameters)

        if not advanced_baseline and not advanced_dst:
            landfill_w_capture = Landfill(
                open_date=1990,
                close_date=2050,
                site_type="landfill",
                mcf=pd.Series(1, index=years),
                city_params_dict=city_params_dict,
                city_instance_attrs=city_parameters.city_instance_attrs,
                landfill_index=0,
                fraction_of_waste=city_parameters.split_fractions.landfill_w_capture,
                gas_capture=True,
                fraction_of_waste_vector=pd.Series(city_parameters.split_fractions.landfill_w_capture, index=years),
            )
            landfill_wo_capture = Landfill(
                open_date=1990,
                close_date=2050,
                site_type="landfill",
                mcf=pd.Series(1, index=years),
                city_params_dict=city_params_dict,
                city_instance_attrs=city_parameters.city_instance_attrs,
                landfill_index=1,
                fraction_of_waste=city_parameters.split_fractions.landfill_wo_capture,
                gas_capture=False,
                gas_capture_efficiency=0.0,
                fraction_of_waste_vector=pd.Series(city_parameters.split_fractions.landfill_wo_capture, index=years),
            )
            dumpsite = Landfill(
                open_date=1990,
                close_date=2050,
                site_type="dumpsite",
                mcf=pd.Series(0.4, index=years),
                city_params_dict=city_params_dict,
                city_instance_attrs=city_parameters.city_instance_attrs,
                landfill_index=2,
                fraction_of_waste=city_parameters.split_fractions.dumpsite,
                gas_capture=False,
                fraction_of_waste_vector=pd.Series(city_parameters.split_fractions.dumpsite, index=years),
            )

            landfills = [landfill_w_capture, landfill_wo_capture, dumpsite]
            non_zero_landfills = [
                x
                for x in [landfill_w_capture, landfill_wo_capture, dumpsite]
                if x.fraction_of_waste > 0
            ]

            city_parameters.landfills = landfills
            city_parameters.non_zero_landfills = non_zero_landfills

    # This should probably be a method of CityParameters
    def update_cityparams_dict(self, city_parameters: dict) -> None:
        """
        Updates the city parameters dictionary with new values.

        Args:
            city_params_dict (dict): The dictionary containing the new values.

        Returns:
            None
        """
        city_params_dict = city_parameters.model_dump()
        keys_to_remove = ["landfills", "non_zero_landfills"]
        for key in keys_to_remove:
            if key in city_params_dict:
                del city_params_dict[key]

        if city_parameters.landfills is not None:
            for landfill in city_parameters.landfills:
                landfill.city_params_dict = city_params_dict
                if hasattr(landfill, "model"):
                    landfill.model.city_params_dict = city_params_dict

        return city_params_dict

    def _calculate_waste_masses(self) -> None:
        waste_masses = {
            waste: frac * self.baseline_parameters.waste_mass
            for waste, frac in self.baseline_parameters.waste_fractions.model_dump().items()
        }
        self.baseline_parameters.waste_masses = WasteMasses(**waste_masses)

    def _calculate_diverted_masses(self, scenario: int = 0) -> None:
        """
        Calculate the diverted masses of different types of waste.

        Args:
            scenario (int): The scenario number to use (0 for baseline, or the number of the alternative scenario).
        """
        if scenario == 0:
            parameters = self.baseline_parameters
        else:
            parameters = self.scenario_parameters.get(scenario - 1)
            if parameters is None:
                raise ValueError(
                    f"Scenario '{scenario}' not found in scenario_parameters."
                )

        diverted_masses = {}

        # if isinstance(parameters.div_fractions.combustion, float):
        #     for div in parameters.div_component_fractions.model_dump().keys():
        #         diverted_masses[div] = {}
        #         fracs = getattr(parameters.div_component_fractions, div)
        #         s = sum(fracs.__dict__.values())
        #         # Make sure the component fractions add up to 1
        #         if s != 0 and np.abs(1 - s) > 0.01:
        #             print(s, 'problems', div)
        #         for waste in fracs.__fields__:
        #             diverted_masses[div][waste] = (
        #                 parameters.waste_mass *
        #                 getattr(parameters.div_fractions, div) *
        #                 getattr(fracs, waste)
        #             )
        # else:
        #     for div in ['compost', 'anaerobic', 'recycling']:
        #         diverted_masses[div] = {}
        #         fracs = getattr(parameters.div_component_fractions, div)
        #         s = sum(fracs.__dict__.values())
        #         # Make sure the component fractions add up to 1
        #         if s != 0 and np.abs(1 - s) > 0.01:
        #             print(s, 'problems', div)
        #         for waste in fracs.__fields__:
        #             diverted_masses[div][waste] = (
        #                 parameters.waste_mass *
        #                 getattr(parameters.div_fractions, div) *
        #                 getattr(fracs, waste)
        #             )

        #     diverted_masses['combustion'] = {}
        #     fracs = parameters.div_component_fractions.combustion
        #     s = sum(fracs.__dict__.values())
        #     # Make sure the component fractions add up to 1
        #     if s != 0 and np.abs(1 - s) > 0.01:
        #         print(s, 'problems', div)
        #     for waste in fracs.__fields__:
        #         diverted_masses['combustion'][waste] = {}
        #         for year in parameters.div_fractions.combustion.index:
        #             diverted_masses['combustion'][waste][year] = (
        #                     parameters.waste_mass *
        #                     parameters.div_fractions.combustion.at[year] *
        #                     getattr(fracs, waste)
        #                 )
        #     diverted_masses['combustion'] = pd.DataFrame(diverted_masses['combustion'])

        # Unsure if this is the right place for this...
        if isinstance(parameters.div_component_fractions.combustion, WasteFractions):
            div_component_fractions = parameters.div_component_fractions
            years = range(1990, 2051)
            compost_dict = div_component_fractions.compost.model_dump()
            compost = pd.DataFrame(compost_dict, index=years)[
                list(self.div_components["compost"])
            ]
            anaerobic_dict = div_component_fractions.anaerobic.model_dump()
            anaerobic = pd.DataFrame(anaerobic_dict, index=years)[
                list(self.div_components["anaerobic"])
            ]
            combustion_dict = div_component_fractions.combustion.model_dump()
            combustion = pd.DataFrame(combustion_dict, index=years)[
                list(self.div_components["combustion"])
            ]
            recycling_dict = div_component_fractions.recycling.model_dump()
            recycling = pd.DataFrame(recycling_dict, index=years)[
                list(self.div_components["recycling"])
            ]
            div_component_fractions = DivComponentFractionsDF(
                compost=compost,
                anaerobic=anaerobic,
                combustion=combustion,
                recycling=recycling,
            )
            parameters.div_component_fractions = div_component_fractions

        if isinstance(parameters.div_fractions, DiversionFractions):
            div_dict = parameters.div_fractions.model_dump()
            df = pd.DataFrame(
                [div_dict] * len(parameters.div_component_fractions.compost.index),
                index=parameters.div_component_fractions.compost.index,
                columns=div_dict.keys(),
            )
            parameters.div_fractions = df

        for div in parameters.div_component_fractions.model_dump().keys():
            # Get the component fractions for the current diversion type
            fracs = getattr(parameters.div_component_fractions, div)
            s = fracs.sum(axis=1).iat[0]

            # Ensure that the component fractions add up to 1 for each year
            if not (np.allclose(s, 1, atol=0.01) or np.all(s == 0)):
                print(f"Problems with {div}: Fractions do not sum to 1 across years.")

            # Calculate the diverted masses for each waste type
            if isinstance(parameters.waste_generated_df, pd.DataFrame):
                try:
                    diverted_masses[div] = fracs.multiply(
                        parameters.div_fractions.multiply(
                            parameters.waste_generated_df.sum(axis=1), axis=0
                        )[div],
                        axis=0,
                    )[list(self.div_components[div])]
                except:
                    diverted_masses[div] = fracs.multiply(
                        getattr(parameters.div_fractions, div)
                        * parameters.waste_generated_df.sum(axis=1),
                        axis=0,
                    )[list(self.div_components[div])]
            else:
                diverted_masses[div] = fracs.multiply(
                    parameters.div_fractions.multiply(
                        parameters.waste_generated_df.df.sum(axis=1), axis=0
                    )[div],
                    axis=0,
                )[list(self.div_components[div])]

        # # Reduce diverted masses by rejection rates
        # for waste in self.div_components['compost']:
        #     diverted_masses['compost'][waste] *= (
        #         1 - parameters.non_compostable_not_targeted_total
        #     ) * (1 - self.unprocessable[waste])
        # for waste in self.div_components['combustion']:
        #     diverted_masses['combustion'][waste] *= (1 - self.combustion_reject_rate)
        # for waste in self.div_components['recycling']:
        #     diverted_masses['recycling'][waste] *= self.recycling_reject_rates[waste]

        # Apply rejection rates to the diverted masses
        diverted_masses["compost"] = (
            diverted_masses["compost"]
            .multiply((1 - parameters.non_compostable_not_targeted_total), axis=0)
            .multiply((1 - pd.Series(self.unprocessable)), axis=1)
        )
        diverted_masses["combustion"] *= 1 - self.combustion_reject_rate
        for waste in diverted_masses["recycling"].columns:
            diverted_masses["recycling"][waste] *= self.recycling_reject_rates[waste]

        # if isinstance(parameters.div_fractions.combustion, float):
        #     divs = DivMasses(
        #         compost=WasteMasses(**diverted_masses['compost']),
        #         anaerobic=WasteMasses(**diverted_masses['anaerobic']),
        #         combustion=WasteMasses(**diverted_masses['combustion']),
        #         recycling=WasteMasses(**diverted_masses['recycling'])
        #     )
        # else:
        #     divs = DivMasses(
        #         compost=WasteMasses(**diverted_masses['compost']),
        #         anaerobic=WasteMasses(**diverted_masses['anaerobic']),
        #         combustion=diverted_masses['combustion'],
        #         recycling=WasteMasses(**diverted_masses['recycling'])
        #     )

        # Convert diverted masses to DivMassesAnnual
        divs = DivMassesAnnual(
            compost=diverted_masses["compost"],
            anaerobic=diverted_masses["anaerobic"],
            combustion=diverted_masses["combustion"],
            recycling=diverted_masses["recycling"],
        )

        # Save the results in the correct attribute
        parameters.divs = divs

    def dst_baseline_blank(
        self, country: str, population: int, precipitation: float, temperature: float
    ) -> None:
        """
        Initializes the baseline scenario with given parameters for a blank/custom city.

        Args:
            country (str): The country name.
            population (int): Population of the city.
            precipitation (float): Average annual precipitation in mm/year.

        Returns:
            None
        """

        # Initialize a new CityParameters instance with all required fields
        try:
            iso3 = pycountry.countries.search_fuzzy(country)[0].alpha_3
        except LookupError:
            raise ValueError(f"Country '{country}' not found.")

        region = defaults_2019.region_lookup_iso3.get(iso3)
        if region is None:
            raise ValueError(f"Region for ISO3 code '{iso3}' not found.")

        precip_zone = defaults_2019.get_precipitation_zone(precipitation)
        years = range(1990, 2051)

        # Calculate growth rates
        population_1950 = 751_000_000
        population_2020 = 4_300_000_000
        population_2035 = 5_300_000_000
        growth_rate_historic = (population_2020 / population_1950) ** (
            1 / (2020 - 1950)
        )
        growth_rate_future = (population_2035 / population_2020) ** (1 / (2035 - 2020))

        # Calculate waste per capita
        waste_per_capita = defaults_2019.msw_per_capita_country.get(
            iso3, defaults_2019.msw_per_capita_defaults.get(region, 0)
        )
        waste_mass = waste_per_capita * population / 1000 * 365  # in tons/year

        # Retrieve and normalize waste fractions
        waste_fractions_series = defaults_2019.waste_composition_for(iso3, region)

        waste_fractions_normalized = (
            waste_fractions_series / waste_fractions_series.sum()
        )
        waste_fractions = WasteFractions(**waste_fractions_normalized.to_dict())
        waste_fractions_df = pd.DataFrame(
            [waste_fractions_normalized] * len(years), index=years
        )

        year_of_data_pop = 2022

        # Calculate MEF for compost
        try:
            food_frac = waste_fractions_normalized["food"]
            green_frac = waste_fractions_normalized["green"]
            mef_compost = (
                (
                    0.0055 * food_frac / (food_frac + green_frac)
                    + 0.0139 * green_frac / (food_frac + green_frac)
                )
                * 1.1023
                * 0.7
            )
        except:
            mef_compost = 0.0

        # Get decomposition rates
        ks = defaults_2019.k_defaults.get(precip_zone, None)
        ks = DecompositionRates(
            food=pd.Series(ks.get("food", 0.0), index=years),
            green=pd.Series(ks.get("green", 0.0), index=years),
            wood=pd.Series(ks.get("wood", 0.0), index=years),
            paper_cardboard=pd.Series(ks.get("paper_cardboard", 0.0), index=years),
            textiles=pd.Series(ks.get("textiles", 0.0), index=years),
        )

        # Determine waste split fractions using .get() method
        dumpsite_frac = defaults_2019.fraction_open_dumped_country.get(
            iso3, defaults_2019.fraction_open_dumped.get(region, 0)
        )
        landfill_wo_capture_frac = defaults_2019.fraction_landfilled_country.get(
            iso3, defaults_2019.fraction_landfilled.get(region, 0)
        )
        landfill_w_capture_frac = 0.0  # Default as per original function

        try:
            split_fractions = SplitFractions(
                dumpsite=dumpsite_frac,
                landfill_wo_capture=landfill_wo_capture_frac,
                landfill_w_capture=landfill_w_capture_frac,
            )
        except KeyError:
            if self.region in defaults_2019.landfill_default_regions:
                split_fractions = SplitFractions(
                    landfill_w_capture=0.0, landfill_wo_capture=1.0, dumpsite=0.0
                )
            else:
                split_fractions = SplitFractions(
                    landfill_w_capture=0.0, landfill_wo_capture=0.0, dumpsite=1.0
                )

        # Normalize split fractions
        split_total = sum(split_fractions.model_dump().values())
        if split_total > 0:
            split_fractions = SplitFractions(
                **{
                    site: frac / split_total
                    for site, frac in split_fractions.model_dump().items()
                }
            )

        # Instantiate landfill objects
        years_range = range(1990, 2051)
        city_instance_attrs = {
            "city_name": self.city_name,
            "country": country,
            "components": self.components,
            "div_components": self.div_components,
            "waste_types": self.waste_types,
            "unprocessable": self.unprocessable,
            "non_compostable_not_targeted": self.non_compostable_not_targeted,
            "combustion_reject_rate": self.combustion_reject_rate,
            "recycling_reject_rates": self.recycling_reject_rates,
        }
        city_params_dict = {}  # Define appropriately or pass as needed

        # Diversion fractions
        compost_frac = defaults_2019.fraction_composted_country.get(
            iso3, defaults_2019.fraction_composted.get(region, 0.0)
        )
        combustion_frac = defaults_2019.fraction_incinerated_country.get(
            iso3, defaults_2019.fraction_incinerated.get(region, 0.0)
        )

        # Create DataFrame with the same values for all years
        div_fractions = pd.DataFrame(
            {
                "compost": compost_frac,
                "anaerobic": 0.0,
                "combustion": combustion_frac,
                "recycling": 0.0,
            },
            index=years,
        )

        def calculate_component_fractions(
            waste_fractions: WasteFractions, div_type: str
        ) -> WasteFractions:
            components = self.div_components[div_type]
            filtered_fractions = {
                waste: getattr(waste_fractions, waste) for waste in components
            }
            total = sum(filtered_fractions.values())
            normalized_fractions = {
                waste: fraction / total
                for waste, fraction in filtered_fractions.items()
            }
            return WasteFractions(
                **{
                    waste: normalized_fractions.get(waste, 0)
                    for waste in waste_fractions.model_dump().keys()
                }
            )

        div_component_fractions = DivComponentFractions(
            compost=calculate_component_fractions(waste_fractions, "compost"),
            anaerobic=calculate_component_fractions(waste_fractions, "anaerobic"),
            combustion=calculate_component_fractions(waste_fractions, "combustion"),
            recycling=calculate_component_fractions(waste_fractions, "recycling"),
        )

        # Calculate diversion component fractions
        compost_dict = div_component_fractions.compost.model_dump()
        compost = pd.DataFrame(compost_dict, index=years)
        anaerobic_dict = div_component_fractions.anaerobic.model_dump()
        anaerobic = pd.DataFrame(anaerobic_dict, index=years)
        combustion_dict = div_component_fractions.combustion.model_dump()
        combustion = pd.DataFrame(combustion_dict, index=years)
        recycling_dict = div_component_fractions.recycling.model_dump()
        recycling = pd.DataFrame(recycling_dict, index=years)
        div_component_fractions = DivComponentFractionsDF(
            compost=compost,
            anaerobic=anaerobic,
            combustion=combustion,
            recycling=recycling,
        )

        # Calculate non_compostable_not_targeted_total
        non_compostable_not_targeted_total = sum(
            [
                self.non_compostable_not_targeted[x]
                * div_component_fractions.model_dump().get("compost", {}).get(x, 0.0)
                for x in self.div_components["compost"]
            ]
        )
        non_compostable_not_targeted_total = pd.Series(
            non_compostable_not_targeted_total, index=years
        )
        if non_compostable_not_targeted_total.isna().all():
            non_compostable_not_targeted_total = pd.Series(0, index=years)

        # Create gas_capture_efficiency Series
        gas_capture_efficiency_value = 0.6
        gas_capture_efficiency_series = pd.Series(
            gas_capture_efficiency_value, index=years_range
        )

        waste_masses = WasteMasses(
            **(waste_fractions_normalized * waste_mass).to_dict()
        )
        waste_masses_df = waste_fractions_df * waste_mass
        waste_generated_df = WasteGeneratedDF.create(
            waste_masses_df,
            1990,
            2050,
            year_of_data_pop,
            growth_rate_historic,
            growth_rate_future,
        )

        # Assign to CityParameters
        baseline = CityParameters(
            waste_fractions=waste_fractions_df,
            div_fractions=div_fractions,
            split_fractions=split_fractions,
            div_component_fractions=div_component_fractions,
            precip=precipitation,
            temperature=temperature,
            growth_rate_historic=growth_rate_historic,
            growth_rate_future=growth_rate_future,
            waste_per_capita=waste_per_capita,
            precip_zone=precip_zone,
            gas_capture_efficiency=gas_capture_efficiency_series,
            mef_compost=mef_compost,
            waste_mass=pd.Series(waste_mass, index=years),
            waste_masses=waste_masses,
            year_of_data_pop=year_of_data_pop,
            scenario=0,
            implement_year=None,
            divs_df=None,
            waste_generated_df=waste_generated_df,
            city_instance_attrs=city_instance_attrs,
            population=population,
            waste_burning_emissions=None,
            non_compostable_not_targeted_total=non_compostable_not_targeted_total,
            ks=ks,
        )
        self.baseline_parameters = baseline

        # Check masses consistency
        self._check_masses_v2(scenario=0)
        if baseline.input_problems:
            print("Input problems detected in baseline parameters.")
            return

        self._calculate_net_masses()
        if (baseline.net_masses < 0).any().any():
            print(f"Invalid new value")
            return

        # Assign the baseline parameters to the city instance
        # baseline.repopulate_attr_dicts()
        self._calculate_divs()

        # landfill_w_capture = Landfill(
        #     open_date=1990,
        #     close_date=2051,
        #     site_type='landfill',
        #     mcf=pd.Series(1.0, index=years_range),
        #     city_params_dict=city_params_dict,
        #     city_instance_attrs=city_instance_attrs,
        #     landfill_index=0,
        #     fraction_of_waste=split_fractions.landfill_w_capture,
        #     gas_capture=True
        # )
        # landfill_wo_capture = Landfill(
        #     open_date=1990,
        #     close_date=2051,
        #     site_type='landfill',
        #     mcf=pd.Series(1.0, index=years_range),
        #     city_params_dict=city_params_dict,
        #     city_instance_attrs=city_instance_attrs,
        #     landfill_index=1,
        #     fraction_of_waste=split_fractions.landfill_wo_capture,
        #     gas_capture=False
        # )
        # dumpsite = Landfill(
        #     open_date=1990,
        #     close_date=2051,
        #     site_type='dumpsite',
        #     mcf=pd.Series(0.4, index=years_range),
        #     city_params_dict=city_params_dict,
        #     city_instance_attrs=city_instance_attrs,
        #     landfill_index=2,
        #     fraction_of_waste=split_fractions.dumpsite,
        #     gas_capture=False
        # )

        # landfills = [landfill_w_capture, landfill_wo_capture, dumpsite]
        # non_zero_landfills = [lf for lf in landfills if lf.fraction_of_waste > 0]

        # self.baseline_parameters.landfills = landfills
        # self.baseline_parameters.non_zero_landfills = non_zero_landfills
        self.baseline_parameters.repopulate_attr_dicts()

        # Estimate emissions for each landfill
        for landfill in baseline.landfills:
            landfill.estimate_emissions()

        # Calculate baseline emissions
        self.estimate_diversion_emissions(scenario=0)
        self.sum_landfill_emissions(scenario=0)

    # def _calculate_component_fractions(self, baseline: Optional['CityParameters'], div_type: Optional[str], waste_fractions: Optional[pd.Series] = None) -> 'DivComponentFractionsDF':
    #     """
    #     Helper function to calculate component fractions for diversions.

    #     Args:
    #         baseline (Optional[CityParameters]): The baseline city parameters. (Unused in this context)
    #         div_type (Optional[str]): The diversion type. (Unused in this context)
    #         waste_fractions (Optional[pd.Series]): The waste fractions series.

    #     Returns:
    #         DivComponentFractionsDF: Normalized waste fractions for all diversion types as DataFrame.
    #     """
    #     # Since div_type is None, calculate for all diversion types based on div_components
    #     if waste_fractions is None:
    #         raise ValueError("waste_fractions must be provided.")

    #     div_component_fractions = {}
    #     for div in self.div_components.keys():
    #         components = self.div_components[div]
    #         # Initialize fractions to 0
    #         component_fractions = {waste: 0.0 for waste in self.waste_types}
    #         # Assign fractions for relevant components
    #         total = 0.0
    #         for waste in components:
    #             component_fractions[waste] = waste_fractions[waste]
    #             total += waste_fractions[waste]
    #         # Normalize if total > 0
    #         if total > 0:
    #             for waste in component_fractions:
    #                 component_fractions[waste] /= total
    #         div_component_fractions[div] = component_fractions

    #     # Convert to DivComponentFractionsDF
    #     return DivComponentFractionsDF(
    #         compost=div_component_fractions['compost'],
    #         anaerobic=div_component_fractions['anaerobic'],
    #         combustion=div_component_fractions['combustion'],
    #         recycling=div_component_fractions['recycling'],
    #     )

    def cityparams_obj_for_blank_site(
        self,
        country: str,
        population: int,
        precipitation: float,
        temperature: float,
        waste_fractions: float,
        waste_mass_year: dict,
        growth_rate_override: float,
    ) -> None:
        """
        Initializes the baseline scenario with given parameters for a blank/custom city.

        Args:
            country (str): The country name.
            population (int): Population of the city.
            precipitation (float): Average annual precipitation in mm/year.

        Returns:
            None
        """

        # Initialize a new CityParameters instance with all required fields
        try:
            iso3 = pycountry.countries.search_fuzzy(country)[0].alpha_3
        except LookupError:
            raise ValueError(f"Country '{country}' not found.")

        region = defaults_2019.region_lookup_iso3.get(iso3)
        if region is None:
            raise ValueError(f"Region for ISO3 code '{iso3}' not found.")

        precip_zone = defaults_2019.get_precipitation_zone(precipitation)

        # Calculate growth rates
        # REPLACE WITH ANDRES TABLE
        # population_1950 = 751_000_000
        # population_2020 = 4_300_000_000
        # population_2035 = 5_300_000_000
        # growth_rate_historic = (population_2020 / population_1950) ** (1 / (2020 - 1950))
        # growth_rate_future = (population_2035 / population_2020) ** (1 / (2035 - 2020))

        growth_rate_historic = 1 + growth_rate_override
        growth_rate_future = 1 + growth_rate_override

        year_of_data_pop = {
            "baseline": waste_mass_year.baseline,
            "scenario": waste_mass_year.scenario,
        }

        if year_of_data_pop["scenario"] is None:
            year_of_data_pop["scenario"] = year_of_data_pop["baseline"]

        # Calculate MEF for compost
        try:
            # 0 is food, 1 is green
            food_frac = waste_fractions.baseline[0]
            green_frac = waste_fractions.baseline[1]
            mef_compost = (
                (
                    0.0055 * food_frac / (food_frac + green_frac)
                    + 0.0139 * green_frac / (food_frac + green_frac)
                )
                * 1.1023
                * 0.7
            )
        except:
            mef_compost = 0.0

        city_instance_attrs = {
            "city_name": self.city_name,
            "country": country,
            "components": self.components,
            "div_components": self.div_components,
            "waste_types": self.waste_types,
            "unprocessable": self.unprocessable,
            "non_compostable_not_targeted": self.non_compostable_not_targeted,
            "combustion_reject_rate": self.combustion_reject_rate,
            "recycling_reject_rates": self.recycling_reject_rates,
        }

        # Assign to CityParameters
        baseline = CityParameters(
            precip=precipitation,
            growth_rate_historic=growth_rate_historic,
            growth_rate_future=growth_rate_future,
            precip_zone=precip_zone,
            mef_compost=mef_compost,
            year_of_data_pop=year_of_data_pop,
            scenario=0,
            city_instance_attrs=city_instance_attrs,
            population=population,
            temperature=temperature,
        )
        self.baseline_parameters = baseline

    def _calc_compost_vol(self, compost_fraction: float, new: bool = False) -> tuple:
        """
        Calculate the mass of compost. Old, still in use a little, should go away.

        Args:
            compost_fraction (float): Fraction of waste that is compostable.
            new (bool): Flag to indicate if it's a new scenario.
        Returns:
            tuple: A tuple containing the compost mass and the compost waste fractions.
        """

        compost_total = compost_fraction * self.baseline_parameters.waste_mass
        fraction_compostable_types = sum(
            [
                self.baseline_parameters.waste_fractions.model_dump()[x]
                for x in self.div_components["compost"]
            ]
        )

        if compost_fraction != 0:
            compost_waste_fractions = {
                x: self.baseline_parameters.waste_fractions.model_dump()[x]
                / fraction_compostable_types
                for x in self.div_components["compost"]
            }
            non_compostable_not_targeted = {
                "food": 0.1,
                "green": 0.05,
                "wood": 0.05,
                "paper_cardboard": 0.1,
            }
            non_compostable_not_targeted_total = sum(
                [
                    non_compostable_not_targeted[x] * compost_waste_fractions[x]
                    for x in self.div_components["compost"]
                ]
            )

            compost = {}
            if (
                new
                and sum(
                    self.baseline_parameters.div_component_fractions.compost.model_dump().values()
                )
                != 0
            ):
                for waste in self.div_components["compost"]:
                    compost[waste] = (
                        compost_total
                        * (1 - non_compostable_not_targeted_total)
                        * self.baseline_parameters.div_component_fractions.compost.model_dump()[
                            waste
                        ]
                        * (1 - self.unprocessable[waste])
                    )
                compost_waste_fractions = (
                    self.baseline_parameters.div_component_fractions.compost
                )
            else:
                for waste in self.div_components["compost"]:
                    compost[waste] = (
                        compost_total
                        * (1 - non_compostable_not_targeted_total)
                        * compost_waste_fractions[waste]
                        * (1 - self.unprocessable[waste])
                    )
        else:
            compost = {x: 0 for x in self.div_components["compost"]}
            compost_waste_fractions = {x: 0 for x in self.div_components["compost"]}
            non_compostable_not_targeted = {
                "food": 0,
                "green": 0,
                "wood": 0,
                "paper_cardboard": 0,
            }
            non_compostable_not_targeted_total = 0

        self.compost_total = compost_total
        self.fraction_compostable_types = fraction_compostable_types
        # self.non_compostable_not_targeted = non_compostable_not_targeted

        return compost, compost_waste_fractions

    def _calc_anaerobic_vol(
        self, anaerobic_fraction: float, new: bool = False
    ) -> tuple:
        """
        Calculate the mass of anaerobically digested waste.Old, still in use a little, should go away.

        Args:
            anaerobic_fraction (float): Fraction of waste that is anaerobic.
            new (bool): Flag to indicate if it's a new scenario.
        Returns:
            tuple: A tuple containing the anaerobic mass and the anaerobic waste fractions.
        """
        anaerobic_total = anaerobic_fraction * self.baseline_parameters.waste_mass
        fraction_anaerobic_types = sum(
            [
                self.baseline_parameters.waste_fractions.model_dump()[x]
                for x in self.div_components["anaerobic"]
            ]
        )

        if anaerobic_fraction != 0:
            anaerobic_waste_fractions = {
                x: self.baseline_parameters.waste_fractions.model_dump()[x]
                / fraction_anaerobic_types
                for x in self.div_components["anaerobic"]
            }

            if (
                new
                and sum(
                    self.baseline_parameters.div_component_fractions.anaerobic.model_dump().values()
                )
                != 0
            ):
                anaerobic = {
                    x: anaerobic_total
                    * self.baseline_parameters.div_component_fractions.anaerobic.model_dump()[
                        x
                    ]
                    for x in self.div_components["anaerobic"]
                }
                anaerobic_waste_fractions = (
                    self.baseline_parameters.div_component_fractions.anaerobic
                )
            else:
                anaerobic = {
                    x: anaerobic_total * anaerobic_waste_fractions[x]
                    for x in self.div_components["anaerobic"]
                }
        else:
            anaerobic = {x: 0 for x in self.div_components["anaerobic"]}
            anaerobic_waste_fractions = {x: 0 for x in self.div_components["anaerobic"]}

        self.anaerobic_total = anaerobic_total
        return anaerobic, anaerobic_waste_fractions

    def _calc_combustion_vol(
        self, combustion_fraction: float, new: bool = False
    ) -> tuple:
        """
        Calculate the mass of combusted waste. Old, still in use a little, should go away.
        Args:
            combustion_fraction (float): Fraction of waste that is combusted.
            new (bool): Flag to indicate if it's a new scenario.
        Returns:
            tuple: A tuple containing the combustion mass and the combustion waste fractions.
        """
        combustion_total = combustion_fraction * self.baseline_parameters.waste_mass
        fraction_combustion_types = sum(
            [
                self.baseline_parameters.waste_fractions.model_dump()[x]
                for x in self.div_components["combustion"]
            ]
        )
        combustion_waste_fractions = {
            x: self.baseline_parameters.waste_fractions.model_dump()[x]
            / fraction_combustion_types
            for x in self.div_components["combustion"]
        }

        if (
            new
            and sum(
                self.baseline_parameters.div_component_fractions.combustion.model_dump().values()
            )
            != 0
        ):
            combustion = {
                x: combustion_total
                * self.baseline_parameters.div_component_fractions.combustion.model_dump()[
                    x
                ]
                * (1 - self.combustion_reject_rate)
                for x in self.div_components["combustion"]
            }
            combustion_waste_fractions = (
                self.baseline_parameters.div_component_fractions.combustion
            )
        else:
            combustion = {
                x: combustion_total
                * combustion_waste_fractions[x]
                * (1 - self.combustion_reject_rate)
                for x in self.div_components["combustion"]
            }

        return combustion, combustion_waste_fractions

    def _calc_recycling_vol(
        self, recycling_fraction: float, new: bool = False
    ) -> tuple:
        """
        Calculate the mass of recycled waste. Old, still in use a little, should go away.

        Args:
            recycling_fraction (float): Fraction of waste that is recycled.
            new (bool): Flag to indicate if it's a new scenario.
        Returns:
            tuple: A tuple containing the recycling mass and the recycling waste fractions.
        """

        recycling_total = recycling_fraction * self.baseline_parameters.waste_mass
        fraction_recyclable_types = sum(
            [
                self.baseline_parameters.waste_fractions.model_dump()[x]
                for x in self.div_components["recycling"]
            ]
        )
        recycling_reject_rates = self.recycling_reject_rates

        if recycling_fraction != 0:
            recycling_waste_fractions = {
                x: self.baseline_parameters.waste_fractions.model_dump()[x]
                / fraction_recyclable_types
                for x in self.div_components["recycling"]
            }

            if (
                new
                and sum(
                    self.baseline_parameters.div_component_fractions.recycling.model_dump().values()
                )
                != 0
            ):
                recycling = {
                    x: recycling_total
                    * self.baseline_parameters.div_component_fractions.recycling.model_dump()[
                        x
                    ]
                    * recycling_reject_rates[x]
                    for x in self.div_components["recycling"]
                }
                recycling_waste_fractions = (
                    self.baseline_parameters.div_component_fractions.recycling
                )
            else:
                recycling = {
                    x: recycling_total
                    * recycling_waste_fractions[x]
                    * recycling_reject_rates[x]
                    for x in self.div_components["recycling"]
                }
        else:
            recycling = {x: 0 for x in self.div_components["recycling"]}
            recycling_waste_fractions = {x: 0 for x in self.div_components["recycling"]}

        self.recycling_total = recycling_total
        return recycling, recycling_waste_fractions

    def estimate_diversion_emissions(self, scenario: int) -> None:
        """
        Estimates emissions from composted and anaerobically digested waste for a specific scenario.

        Args:
            scenario (int): The scenario number to use (0 for baseline, or the number of the alternative scenario).

        Returns:
            None: Updates the emissions attributes in the scenario parameters.
        """

        if scenario == 0:
            parameters = self.baseline_parameters
        else:
            parameters = self.scenario_parameters[scenario - 1]

        compost_emissions = parameters.divs_df.compost * parameters.mef_compost
        anaerobic_emissions = (
            parameters.divs_df.anaerobic
            * defaults_2019.mef_anaerobic
            * defaults_2019.ch4_to_co2e
        )

        parameters.organic_emissions = compost_emissions.add(
            anaerobic_emissions, fill_value=0
        )

    def sum_landfill_emissions(self, scenario: int, simple=False, trace_monthly=False) -> None:
        """
        Aggregates emissions produced by the landfills for a specific scenario.

        Args:
            scenario (int): The scenario number to use (0 for baseline, or the number of the alternative scenario).

        Returns:
            None: Updates the emissions attributes in the scenario parameters.
        """

        if scenario == 0:
            parameters = self.baseline_parameters
            organic_emissions = parameters.organic_emissions
            valid_landfills = [
                lf
                for lf in parameters.landfills
                if lf is not None and lf.emissions is not None
            ]
            years_union = valid_landfills[0].emissions.index
            for lf in valid_landfills[1:]:
                years_union = years_union.union(lf.emissions.index)
            landfill_emissions_list = [
                x.emissions.reindex(years_union, fill_value=0).map(
                    self.convert_methane_m3_to_ton_co2e
                )
                / 28
                for x in valid_landfills
            ]
        elif simple:
            parameters = self.scenario_parameters[scenario - 1]
            organic_emissions = parameters.organic_emissions
            # landfill_emissions = [x.emissions.map(self.convert_methane_m3_to_ton_co2e) for x in parameters.landfills]
            years_union = parameters.landfills[0].emissions.index
            # Union the index of each subsequent landfill with the years_union
            for x in parameters.landfills[1:]:
                years_union = years_union.union(x.emissions.index)
            landfill_emissions_list = [
                x.emissions.reindex(years_union, fill_value=0).map(
                    self.convert_methane_m3_to_ton_co2e
                )
                / 28
                for x in parameters.landfills
            ]
        else:
            parameters = self.scenario_parameters[scenario - 1]
            organic_emissions = parameters.organic_emissions
            # landfill_emissions = [x.emissions.map(self.convert_methane_m3_to_ton_co2e) for x in parameters.landfills]
            years_union = parameters.landfills[0].emissions.index
            # Union the index of each subsequent landfill with the years_union
            for x in parameters.landfills[1:]:
                years_union = years_union.union(x.emissions.index)
            landfill_emissions_list = [
                x.emissions.reindex(years_union, fill_value=0).map(
                    self.convert_methane_m3_to_ton_co2e
                )
                / 28
                for x in parameters.landfills
            ]

        # Concatenate all emissions dataframes
        # all_emissions = sum(landfill_emissions)

        # Reindex each landfill DataFrame to the full range of years, filling missing values with zeros
        # landfill_emissions = [
        #     x.emissions.reindex(years_union, fill_value=0).map(self.convert_methane_m3_to_ton_co2e)
        #     for x in parameters.landfills
        # ]

        # Sum the emissions dataframes
        summed_landfill_emissions = sum(landfill_emissions_list)

        # Group by the year index and sum the emissions for each year
        # summed_landfill_emissions = all_emissions.groupby(all_emissions.index).sum()

        # # Remove total
        summed_landfill_emissions.drop("total", axis=1, inplace=True)

        # summed_diversion_emissions = organic_emissions.loc[:, list(self.components)] / 28
        summed_diversion_emissions = (
            organic_emissions.reindex(
                columns=summed_landfill_emissions.columns, fill_value=0
            )
            / 28
        )

        # Repeat with addition of diverted waste emissions
        if trace_monthly:
            # First, convert annual index to datetime if it isn't already
            if not isinstance(summed_diversion_emissions.index, pd.DatetimeIndex):
                summed_diversion_emissions.index = pd.to_datetime(summed_diversion_emissions.index, format='%Y')

            # Resample to monthly frequency, forward-filling values, then divide by 12
            monthly_diversion_emissions = (
                summed_diversion_emissions
                .resample('MS')  # Month Start frequency
                .ffill()
                .reindex(summed_landfill_emissions.index)
                / 12  # Divide by 12 if you want monthly averages
            )
            summed_emissions = sum(
                [
                    summed_landfill_emissions.loc[:, list(self.components)],
                    monthly_diversion_emissions.loc[:, list(self.components)],
                ]
            )
        else:
            summed_emissions = sum(
                [
                    summed_landfill_emissions.loc[:, list(self.components)],
                    summed_diversion_emissions.loc[summed_landfill_emissions.index, :],
                ]
            )
        # summed_emissions = all_emissions.groupby(all_emissions.index).sum()
        # summed_emissions.drop('total', axis=1, inplace=True)
        # summed_emissions /= 28

        summed_landfill_emissions["total"] = summed_landfill_emissions.sum(axis=1)
        summed_diversion_emissions["total"] = summed_diversion_emissions.sum(axis=1)
        summed_emissions["total"] = summed_emissions.sum(axis=1)

        parameters.landfill_emissions = summed_landfill_emissions
        parameters.diversion_emissions = summed_diversion_emissions
        parameters.total_emissions = summed_emissions.astype(float).fillna(0)

    def _check_masses_v2(
        self,
        scenario: int,
        advanced_baseline: bool = False,
        advanced_dst: bool = False,
        implement_year: int = None,
    ) -> None:
        """
        Adjusts diversion waste type fractions/masses if more of a waste type is being diverted than generated.
        We have to make sure we're not trying to compost more food waste than we have, for example.

        Args:
            scenario (int): Scenario index.
            advanced_baseline (bool): Flag for advanced baseline scenario.
            advanced_dst (bool): Flag for advanced diversion scenario.
            implement_year (int): Year of implementation for the scenario.
        Returns:
            None: Updates the parameters in place.
        """
        if scenario == 0:
            parameters = self.baseline_parameters
        else:
            parameters = self.scenario_parameters[scenario - 1]

        if (not advanced_baseline) and (not advanced_dst):
            if isinstance(parameters.div_fractions, pd.DataFrame):
                diversion_fractions_instance = DiversionFractions(
                    compost=parameters.div_fractions.at[2000, "compost"],
                    anaerobic=parameters.div_fractions.at[2000, "anaerobic"],
                    combustion=parameters.div_fractions.at[2000, "combustion"],
                    recycling=parameters.div_fractions.at[2000, "recycling"],
                )
            else:
                diversion_fractions_instance = DiversionFractions(
                    compost=parameters.div_fractions.compost,
                    anaerobic=parameters.div_fractions.anaerobic,
                    combustion=parameters.div_fractions.combustion,
                    recycling=parameters.div_fractions.recycling,
                )
            div_component_fractions_instance = DivComponentFractions(
                compost=WasteFractions(
                    **parameters.div_component_fractions.compost.loc[2000, :]
                ),
                anaerobic=WasteFractions(
                    **parameters.div_component_fractions.anaerobic.loc[2000, :]
                ),
                combustion=WasteFractions(
                    **parameters.div_component_fractions.combustion.loc[2000, :]
                ),
                recycling=WasteFractions(
                    **parameters.div_component_fractions.recycling.loc[2000, :]
                ),
            )
            waste_fractions_instance = WasteFractions(
                **parameters.waste_fractions.loc[2000, :]
            )
            (
                parameters.adjusted_diversion_constituents,
                parameters.input_problems,
                parameters.divs,
                parameters.div_component_fractions,
            ) = self.mass_checker_math(
                div_fractions=diversion_fractions_instance,
                div_component_fractions=div_component_fractions_instance,
                waste_fractions=waste_fractions_instance,
                scenario=scenario,
            )

            return

        else:
            unique_divsets = parameters.div_fractions.drop_duplicates()

            # Initialize empty DataFrames to build the final instances
            div_masses_df = DivsDF(
                compost=pd.DataFrame(
                    index=parameters.div_fractions.index, columns=list(self.waste_types)
                ),
                anaerobic=pd.DataFrame(
                    index=parameters.div_fractions.index, columns=list(self.waste_types)
                ),
                combustion=pd.DataFrame(
                    index=parameters.div_fractions.index, columns=list(self.waste_types)
                ),
                recycling=pd.DataFrame(
                    index=parameters.div_fractions.index, columns=list(self.waste_types)
                ),
            )
            div_component_fractions_df = DivComponentFractionsDF(
                compost=pd.DataFrame(
                    index=parameters.div_fractions.index,
                    columns=parameters.div_component_fractions.compost.columns,
                ),
                anaerobic=pd.DataFrame(
                    index=parameters.div_fractions.index,
                    columns=parameters.div_component_fractions.anaerobic.columns,
                ),
                combustion=pd.DataFrame(
                    index=parameters.div_fractions.index,
                    columns=parameters.div_component_fractions.combustion.columns,
                ),
                recycling=pd.DataFrame(
                    index=parameters.div_fractions.index,
                    columns=parameters.div_component_fractions.recycling.columns,
                ),
            )

            for i, row in unique_divsets.iterrows():
                corresponding_year = row.name
                next_year = (
                    unique_divsets.index[i + 1]
                    if i + 1 < len(unique_divsets)
                    else parameters.div_fractions.index[-1] + 1
                )
                year_range = range(corresponding_year, next_year)

                diversion_fractions_instance = DiversionFractions(
                    compost=parameters.div_fractions.at[corresponding_year, "compost"],
                    anaerobic=parameters.div_fractions.at[
                        corresponding_year, "anaerobic"
                    ],
                    combustion=parameters.div_fractions.at[
                        corresponding_year, "combustion"
                    ],
                    recycling=parameters.div_fractions.at[
                        corresponding_year, "recycling"
                    ],
                )
                div_component_fractions_instance = DivComponentFractions(
                    compost=WasteFractions(
                        **parameters.div_component_fractions.compost.loc[
                            corresponding_year, :
                        ].to_dict()
                    ),
                    anaerobic=WasteFractions(
                        **parameters.div_component_fractions.anaerobic.loc[
                            corresponding_year, :
                        ].to_dict()
                    ),
                    combustion=WasteFractions(
                        **parameters.div_component_fractions.combustion.loc[
                            corresponding_year, :
                        ].to_dict()
                    ),
                    recycling=WasteFractions(
                        **parameters.div_component_fractions.recycling.loc[
                            corresponding_year, :
                        ].to_dict()
                    ),
                )
                waste_fractions_instance = WasteFractions(
                    **parameters.waste_fractions.loc[corresponding_year, :].to_dict()
                )
                (
                    parameters.adjusted_diversion_constituents,
                    parameters.input_problems,
                    divs,
                    div_component_fractions,
                ) = self.mass_checker_math(
                    div_fractions=diversion_fractions_instance,
                    div_component_fractions=div_component_fractions_instance,
                    waste_fractions=waste_fractions_instance,
                    scenario=scenario,
                    corresponding_year=corresponding_year,
                )

                # Populate the DataFrames for all years in the range
                # Check at some point if this is actually working right.
                # div_masses_df.loc[year_range, :] = pd.DataFrame([divs] * len(year_range), index=year_range)

                if isinstance(divs, dict):
                    if implement_year in year_range:
                        year_ranges = {}
                        year_ranges["baseline"] = range(year_range[0], implement_year)
                        year_ranges["scenario"] = range(
                            implement_year, year_range[-1] + 1
                        )
                        div_masses_df_split_up = {
                            "baseline": DivsDF(
                                compost=pd.DataFrame(
                                    index=year_ranges["baseline"],
                                    columns=list(self.waste_types),
                                ),
                                anaerobic=pd.DataFrame(
                                    index=year_ranges["baseline"],
                                    columns=list(self.waste_types),
                                ),
                                combustion=pd.DataFrame(
                                    index=year_ranges["baseline"],
                                    columns=list(self.waste_types),
                                ),
                                recycling=pd.DataFrame(
                                    index=year_ranges["baseline"],
                                    columns=list(self.waste_types),
                                ),
                            ),
                            "scenario": DivsDF(
                                compost=pd.DataFrame(
                                    index=year_ranges["scenario"],
                                    columns=list(self.waste_types),
                                ),
                                anaerobic=pd.DataFrame(
                                    index=year_ranges["scenario"],
                                    columns=list(self.waste_types),
                                ),
                                combustion=pd.DataFrame(
                                    index=year_ranges["scenario"],
                                    columns=list(self.waste_types),
                                ),
                                recycling=pd.DataFrame(
                                    index=year_ranges["scenario"],
                                    columns=list(self.waste_types),
                                ),
                            ),
                        }
                        for period in ["baseline", "scenario"]:
                            div_masses_df_split_up[period].compost.loc[
                                year_ranges[period], :
                            ] = pd.DataFrame(
                                [divs[period].compost.model_dump()]
                                * len(year_ranges[period]),
                                index=year_ranges[period],
                            )
                            div_masses_df_split_up[period].anaerobic.loc[
                                year_ranges[period], :
                            ] = pd.DataFrame(
                                [divs[period].anaerobic.model_dump()]
                                * len(year_ranges[period]),
                                index=year_ranges[period],
                            )
                            div_masses_df_split_up[period].combustion.loc[
                                year_ranges[period], :
                            ] = pd.DataFrame(
                                [divs[period].combustion.model_dump()]
                                * len(year_ranges[period]),
                                index=year_ranges[period],
                            )
                            div_masses_df_split_up[period].recycling.loc[
                                year_ranges[period], :
                            ] = pd.DataFrame(
                                [divs[period].recycling.model_dump()]
                                * len(year_ranges[period]),
                                index=year_ranges[period],
                            )

                        div_masses_df.compost.loc[year_range, :] = pd.concat(
                            [
                                div_masses_df_split_up["baseline"].compost,
                                div_masses_df_split_up["scenario"].compost,
                            ]
                        )
                        div_masses_df.anaerobic.loc[year_range, :] = pd.concat(
                            [
                                div_masses_df_split_up["baseline"].anaerobic,
                                div_masses_df_split_up["scenario"].anaerobic,
                            ]
                        )
                        div_masses_df.combustion.loc[year_range, :] = pd.concat(
                            [
                                div_masses_df_split_up["baseline"].combustion,
                                div_masses_df_split_up["scenario"].combustion,
                            ]
                        )
                        div_masses_df.recycling.loc[year_range, :] = pd.concat(
                            [
                                div_masses_df_split_up["baseline"].recycling,
                                div_masses_df_split_up["scenario"].recycling,
                            ]
                        )
                    elif implement_year < year_range[0]:
                        div_masses_df.compost.loc[year_range, :] = pd.DataFrame(
                            [divs["baseline"].compost] * len(year_range),
                            index=year_range,
                        )
                        div_masses_df.anaerobic.loc[year_range, :] = pd.DataFrame(
                            [divs["baseline"].anaerobic] * len(year_range),
                            index=year_range,
                        )
                        div_masses_df.combustion.loc[year_range, :] = pd.DataFrame(
                            [divs["baseline"].combustion] * len(year_range),
                            index=year_range,
                        )
                        div_masses_df.recycling.loc[year_range, :] = pd.DataFrame(
                            [divs["baseline"].recycling] * len(year_range),
                            index=year_range,
                        )
                    else:
                        div_masses_df.compost.loc[year_range, :] = pd.DataFrame(
                            [divs["scenario"].compost] * len(year_range),
                            index=year_range,
                        )
                        div_masses_df.anaerobic.loc[year_range, :] = pd.DataFrame(
                            [divs["scenario"].anaerobic] * len(year_range),
                            index=year_range,
                        )
                        div_masses_df.combustion.loc[year_range, :] = pd.DataFrame(
                            [divs["scenario"].combustion] * len(year_range),
                            index=year_range,
                        )
                        div_masses_df.recycling.loc[year_range, :] = pd.DataFrame(
                            [divs["scenario"].recycling] * len(year_range),
                            index=year_range,
                        )

                    div_component_fractions_df.compost.loc[year_range, :] = (
                        pd.DataFrame(
                            [div_component_fractions.compost.model_dump()]
                            * len(year_range),
                            index=year_range,
                        )
                    )
                    div_component_fractions_df.anaerobic.loc[year_range, :] = (
                        pd.DataFrame(
                            [div_component_fractions.anaerobic.model_dump()]
                            * len(year_range),
                            index=year_range,
                        )
                    )
                    div_component_fractions_df.combustion.loc[year_range, :] = (
                        pd.DataFrame(
                            [div_component_fractions.combustion.model_dump()]
                            * len(year_range),
                            index=year_range,
                        )
                    )
                    div_component_fractions_df.recycling.loc[year_range, :] = (
                        pd.DataFrame(
                            [div_component_fractions.recycling.model_dump()]
                            * len(year_range),
                            index=year_range,
                        )
                    )
                else:
                    try:
                        div_masses_df["compost"].loc[year_range, :] = pd.DataFrame(
                            [divs["compost"]] * len(year_range), index=year_range
                        )
                        div_masses_df["anaerobic"].loc[year_range, :] = pd.DataFrame(
                            [divs["anaerobic"]] * len(year_range), index=year_range
                        )
                        div_masses_df["combustion"].loc[year_range, :] = pd.DataFrame(
                            [divs["combustion"]] * len(year_range), index=year_range
                        )
                        div_masses_df["recycling"].loc[year_range, :] = pd.DataFrame(
                            [divs["recycling"]] * len(year_range), index=year_range
                        )

                        div_component_fractions_df["compost"].loc[year_range, :] = (
                            pd.DataFrame(
                                [div_component_fractions["compost"]] * len(year_range),
                                index=year_range,
                            )
                        )
                        div_component_fractions_df["anaerobic"].loc[year_range, :] = (
                            pd.DataFrame(
                                [div_component_fractions["anaerobic"]]
                                * len(year_range),
                                index=year_range,
                            )
                        )
                        div_component_fractions_df["combustion"].loc[year_range, :] = (
                            pd.DataFrame(
                                [div_component_fractions["combustion"]]
                                * len(year_range),
                                index=year_range,
                            )
                        )
                        div_component_fractions_df["recycling"].loc[year_range, :] = (
                            pd.DataFrame(
                                [div_component_fractions["recycling"]]
                                * len(year_range),
                                index=year_range,
                            )
                        )
                    except:
                        div_masses_df.compost.loc[year_range, :] = pd.DataFrame(
                            [divs.compost.model_dump()] * len(year_range),
                            index=year_range,
                        )
                        div_masses_df.anaerobic.loc[year_range, :] = pd.DataFrame(
                            [divs.anaerobic.model_dump()] * len(year_range),
                            index=year_range,
                        )
                        div_masses_df.combustion.loc[year_range, :] = pd.DataFrame(
                            [divs.combustion.model_dump()] * len(year_range),
                            index=year_range,
                        )
                        div_masses_df.recycling.loc[year_range, :] = pd.DataFrame(
                            [divs.recycling.model_dump()] * len(year_range),
                            index=year_range,
                        )

                        div_component_fractions_df.compost.loc[year_range, :] = (
                            pd.DataFrame(
                                [div_component_fractions.compost.model_dump()]
                                * len(year_range),
                                index=year_range,
                            )
                        )
                        div_component_fractions_df.anaerobic.loc[year_range, :] = (
                            pd.DataFrame(
                                [div_component_fractions.anaerobic.model_dump()]
                                * len(year_range),
                                index=year_range,
                            )
                        )
                        div_component_fractions_df.combustion.loc[year_range, :] = (
                            pd.DataFrame(
                                [div_component_fractions.combustion.model_dump()]
                                * len(year_range),
                                index=year_range,
                            )
                        )
                        div_component_fractions_df.recycling.loc[year_range, :] = (
                            pd.DataFrame(
                                [div_component_fractions.recycling.model_dump()]
                                * len(year_range),
                                index=year_range,
                            )
                        )

            try:
                # Create the final instances
                final_div_masses_annual = DivMassesAnnual(
                    compost=div_masses_df["compost"],
                    anaerobic=div_masses_df["anaerobic"],
                    combustion=div_masses_df["combustion"],
                    recycling=div_masses_df["recycling"],
                )

                final_div_component_fractions_df = DivComponentFractionsDF(
                    compost=div_component_fractions_df["compost"],
                    anaerobic=div_component_fractions_df["anaerobic"],
                    combustion=div_component_fractions_df["combustion"],
                    recycling=div_component_fractions_df["recycling"],
                )
            except:
                # Create the final instances
                final_div_masses_annual = DivMassesAnnual(
                    compost=div_masses_df.compost.loc[
                        :, list(self.div_components["compost"])
                    ],
                    anaerobic=div_masses_df.anaerobic.loc[
                        :, list(self.div_components["anaerobic"])
                    ],
                    combustion=div_masses_df.combustion.loc[
                        :, list(self.div_components["combustion"])
                    ],
                    recycling=div_masses_df.recycling.loc[
                        :, list(self.div_components["recycling"])
                    ],
                )

                final_div_component_fractions_df = DivComponentFractionsDF(
                    compost=div_component_fractions_df.compost.loc[
                        :, list(self.div_components["compost"])
                    ],
                    anaerobic=div_component_fractions_df.anaerobic.loc[
                        :, list(self.div_components["anaerobic"])
                    ],
                    combustion=div_component_fractions_df.combustion.loc[
                        :, list(self.div_components["combustion"])
                    ],
                    recycling=div_component_fractions_df.recycling.loc[
                        :, list(self.div_components["recycling"])
                    ],
                )

            parameters.divs = final_div_masses_annual
            parameters.div_component_fractions = final_div_component_fractions_df

            return

    def mass_checker_math(
        self,
        div_fractions: DiversionFractions,
        div_component_fractions: DivComponentFractions,
        waste_fractions: WasteFractions,
        scenario: int,
        corresponding_year: int = 2000,
    ) -> tuple:
        """
        This function has the actual math for check_masses_v2.

        Args:
            div_fractions (DiversionFractions): Diversion fractions.
            div_component_fractions (DivComponentFractions): Component fractions for each diversion type.
            waste_fractions (WasteFractions): Waste fractions.
            scenario (int): Scenario index.
            corresponding_year (int): Year of the scenario.
        Returns:
            tuple: A tuple containing adjusted diversion constituents, input problems, divs, and div component fractions.
        """
        components_multiplied_through = {}
        for div in div_component_fractions.model_dump().keys():
            components_multiplied_through[div] = {}
            for waste in getattr(div_component_fractions, div).model_dump().keys():
                components_multiplied_through[div][waste] = getattr(
                    div_fractions, div
                ) * getattr(getattr(div_component_fractions, div), waste)

        net = {}
        negative_catcher = False
        for waste in waste_fractions.model_dump().keys():
            s = sum(
                components_multiplied_through[div].get(waste, 0)
                for div in div_fractions.model_dump().keys()
            )
            net[waste] = getattr(waste_fractions, waste) - s
            if net[waste] < -1e-5:
                negative_catcher = True

        # Under-delivery: a requested treatment whose proportional component
        # fractions don't actually sum to its target (e.g. an empty/insufficient
        # eligible pool -> 0-ton "silent accept"). Such cases must NOT take the
        # happy path; route them through the solver, which reallocates or raises.
        under_delivery = any(
            np.abs(
                getattr(div_fractions, div)
                - sum(components_multiplied_through[div].values())
            )
            > 1e-5
            for div in div_fractions.model_dump().keys()
        )

        if (not negative_catcher) and (not under_delivery):
            # divs = self._divs_from_component_fractions(div_fractions, div_component_fractions, scenario=scenario)
            # parameters.divs = divs
            adjusted_diversion_constituents = False
            input_problems = False
            divs = self._divs_from_component_fractions(
                div_fractions,
                div_component_fractions,
                scenario=scenario,
                advanced=True,
                year=corresponding_year,
            )
            return (
                adjusted_diversion_constituents,
                input_problems,
                divs,
                div_component_fractions,
            )

        # --- Robust diversion allocation (replaces the legacy redistribution) ---
        # The old guard-ladder + redistribution loop that lived here falsely
        # rejected many *feasible* slider combinations, silently accepted some
        # impossible ones (diverting 0 tons), and could crash on a bare assert.
        # It is replaced by an exact min-cost max-flow allocation of the three
        # contended treatments (compost/anaerobic/recycling). Combustion stays a
        # uniform fraction of the leftover combustible mass (handled just below).
        # See SWEET_python/dst_allocation.py (+ dst_allocation_prototype.py).
        from SWEET_python import dst_allocation

        waste_dict = waste_fractions.model_dump()
        three_targets = {
            "compost": div_fractions.compost,
            "anaerobic": div_fractions.anaerobic,
            "recycling": div_fractions.recycling,
        }
        alloc_result = dst_allocation.solve_allocation(
            waste_dict,
            three_targets,
            eligibility=self.div_components,
            spare_combustibles=(div_fractions.combustion > 0),
        )

        if not alloc_result["feasible"]:
            # Build an actionable message naming a specific slider + its cap.
            def _cap(t, others):
                return dst_allocation.max_feasible_target(
                    t, waste_dict, others, eligibility=self.div_components
                )

            # Stage 1: a slider that on its own exceeds its eligible-waste pool
            # (clearest message -- a fixed, city-specific ceiling).
            for t in ("compost", "anaerobic", "recycling"):
                if three_targets[t] <= 0:
                    continue
                standalone = _cap(t, {k: 0.0 for k in three_targets if k != t})
                if three_targets[t] > standalone + 1e-6:
                    raise CustomError(
                        "INVALID_PARAMETERS",
                        f"{t.capitalize()} can be at most {standalone * 100:.1f}% "
                        f"of this city's waste (only that much is eligible for "
                        f"{t}), but {three_targets[t] * 100:.1f}% was requested.",
                    )
            # Stage 2: each fits alone but the combination over-draws a shared
            # waste type -- name a slider whose reduction restores feasibility.
            for t in ("compost", "anaerobic", "recycling"):
                if three_targets[t] <= 0:
                    continue
                cond = _cap(t, {k: v for k, v in three_targets.items() if k != t})
                if three_targets[t] > cond + 1e-6:
                    raise CustomError(
                        "INVALID_PARAMETERS",
                        f"{t.capitalize()} can be at most {cond * 100:.1f}% given "
                        f"the other diversion selections for this city, but "
                        f"{three_targets[t] * 100:.1f}% was requested. Reduce "
                        f"{t} or the other diversion sliders.",
                    )
            raise CustomError(
                "INVALID_PARAMETERS",
                "The requested diversion combination cannot be met by this "
                "city's waste composition. Reduce one or more diversion sliders.",
            )

        # Overwrite the three contended treatments with the flow allocation
        # (fraction-of-total-waste units, exactly what
        # components_multiplied_through holds).
        allocation = alloc_result["allocation"]
        for div in ("compost", "anaerobic", "recycling"):
            components_multiplied_through[div] = {w: 0.0 for w in self.waste_types}
            for w, val in allocation.get(div, {}).items():
                components_multiplied_through[div][w] = val

        # Combustion takes a uniform fraction of the leftover combustible mass.
        combustion_all = {}
        for waste in self.waste_types:
            consumed = sum(
                components_multiplied_through[div].get(waste, 0.0)
                for div in ("compost", "anaerobic", "recycling")
            )
            combustion_all[waste] = getattr(waste_fractions, waste) - consumed

        remainder = sum(
            combustion_all[w] for w in self.div_components["combustion"]
        )
        if div_fractions.combustion > remainder + 1e-9:
            raise CustomError(
                "INVALID_PARAMETERS",
                f"Incineration can be at most {remainder * 100:.1f}% of waste "
                f"after the requested compost, anaerobic digestion, and "
                f"recycling, but {div_fractions.combustion * 100:.1f}% was "
                f"requested. Reduce incineration or the other diversion sliders.",
            )
        combustion_fraction_of_remainder = (
            div_fractions.combustion / remainder if remainder > 1e-12 else 0.0
        )

        components_multiplied_through["combustion"] = {
            w: 0.0 for w in self.waste_types
        }
        for waste in self.div_components["combustion"]:
            components_multiplied_through["combustion"][waste] = (
                combustion_fraction_of_remainder * combustion_all[waste]
            )

        # Defensive invariants -> actionable CustomError instead of bare assert
        # (asserts are stripped under `python -O` and would surface as 500s).
        for d in div_fractions.model_dump().keys():
            if (
                np.abs(
                    getattr(div_fractions, d)
                    - sum(components_multiplied_through[d].values())
                )
                > 1e-3
            ):
                raise CustomError(
                    "INVALID_PARAMETERS",
                    f"Could not satisfy the requested {d} fraction with this "
                    f"city's waste composition. Adjust the diversion sliders.",
                )
            for w in components_multiplied_through[d]:
                if abs(components_multiplied_through[d][w]) < 1e-9:
                    components_multiplied_through[d][w] = 0
                if components_multiplied_through[d][w] < 0:
                    raise CustomError(
                        "INVALID_PARAMETERS",
                        f"Could not satisfy the requested diversion mix for "
                        f"'{w}'. Reduce composting, recycling, or anaerobic "
                        f"digestion.",
                    )

        adjusted_div_component_fractions = {
            div: {
                waste: (
                    components_multiplied_through[div][waste]
                    / getattr(div_fractions, div)
                    if getattr(div_fractions, div) != 0
                    else 0
                )
                for waste in components_multiplied_through[div]
            }
            for div in components_multiplied_through
        }

        adjusted_div_component_fractions = DivComponentFractions(
            **adjusted_div_component_fractions
        )

        divs = self._divs_from_component_fractions(
            div_fractions, adjusted_div_component_fractions, scenario=scenario
        )

        div_component_fractions = adjusted_div_component_fractions
        adjusted_diversion_constituents = True
        input_problems = False

        return (
            adjusted_diversion_constituents,
            input_problems,
            divs,
            div_component_fractions,
        )

        # else:

        #     # Here we check and adjust diversion components. We have four things to consider:
        #     # The waste mass/fractions before and after implement year (they only change once),
        #     # And the diversion fractions before and after implement year. Luckily, they change at the same time!
        #     # So, ideally, we would check the first combination during baseline, and then the second during dst.

        #     if advanced_baseline:
        #         div_fractions = parameters.div_fractions.loc[:, implement_year-1]
        #         div_component_fractions = parameters.div_component_fractions.loc[:, implement_year-1]
        #     else:
        #         div_fractions = parameters.div_fractions.loc[:, implement_year+1]
        #         div_component_fractions = parameters.div_component_fractions.loc[:, implement_year+1]

        #     unique_divsets = parameters.divs_df.drop_duplicates()

        #     components_multiplied_through = {}
        #     for div in div_component_fractions.model_dump().keys():
        #         components_multiplied_through[div] = {}
        #         for waste in getattr(div_component_fractions, div).model_dump().keys():
        #             components_multiplied_through[div][waste] = getattr(div_fractions, div) * getattr(getattr(div_component_fractions, div), waste)

        #     components_multiplied_through['combustion'] = pd.DataFrame(components_multiplied_through['combustion'])
        #     unique_divsets = components_multiplied_through['combustion'].drop_duplicates()

        #     div_component_fractions_adjusted = []
        #     divs = []

        #     for i in range(unique_divsets.shape[0]):
        #         divset = unique_divsets.iloc[i,:]
        #         components_multiplied_through_dummy = components_multiplied_through.copy()
        #         components_multiplied_through_dummy['combustion'] = {x: float(divset.at[x]) for x in divset.index}

        #         net = {}
        #         negative_catcher = False
        #         for waste in parameters.waste_fractions.model_dump().keys():
        #             s = sum(components_multiplied_through_dummy[div].get(waste, 0) for div in div_fractions.model_dump().keys())
        #             net[waste] = getattr(parameters.waste_fractions, waste) - s
        #             if net[waste] < -1e-3:
        #                 negative_catcher = True

        #         if not negative_catcher:
        #             #divs = self._divs_from_component_fractions(div_fractions, div_component_fractions, scenario=scenario)
        #             #parameters.divs = divs
        #             parameters.adjusted_diversion_constituents = False
        #             parameters.input_problems = False
        #             return

        #         if sum(getattr(div_fractions, div) for div in div_fractions.model_dump().keys()) > 1:
        #             raise CustomError("INVALID_PARAMETERS", f"Diversions sum to {sum(getattr(div_fractions, div) for div in div_fractions.model_dump().keys())}, but they must sum to 1 or less.")

        #         compostables = sum(getattr(parameters.waste_fractions, waste) for waste in ['food', 'green', 'wood', 'paper_cardboard'])
        #         if div_fractions.compost + div_fractions.anaerobic > compostables:
        #             raise CustomError("INVALID_PARAMETERS", f"Only food, green, wood, and paper/cardboard can be composted or anaerobically digested. Those waste types sum to {compostables}, but input values of compost and anaerobic digestion sum to {div_fractions.compost + div_fractions.anaerobic}.")

        #         for div in div_fractions.model_dump().keys():
        #             fraction = getattr(div_fractions, div)
        #             s = sum(getattr(parameters.waste_fractions, waste) for waste in self.div_components[div])
        #             if s < fraction:
        #                 components = self.div_components[div]
        #                 values = [getattr(parameters.waste_fractions, x) for x in components]
        #                 raise CustomError("INVALID_PARAMETERS", f"{div} too high. {div} applies to {components}, which are {values} of total waste--the sum of these is {sum(values)}, so only that much waste can be {div}, but input value was {fraction}.")

        #         non_combustables = sum(getattr(parameters.waste_fractions, waste) for waste in ['glass', 'metal', 'other'])
        #         if div_fractions.compost + div_fractions.anaerobic + div_fractions.combustion > (1 - non_combustables):
        #             s = div_fractions.compost + div_fractions.anaerobic + div_fractions.combustion
        #             raise CustomError("INVALID_PARAMETERS", f"Glass, metal, and other account for {non_combustables:.3f} of waste, and they can only be recycled. {div_fractions.compost} compost, {div_fractions.anaerobic} anaerobic, and {div_fractions.combustion} incineration were specified, summing to {s}, but only {1 - non_combustables} of waste can be diverted to these diversion types.")

        #         non_combustion = {}
        #         combustion_all = {}
        #         keys_of_interest = ['compost', 'anaerobic', 'recycling']
        #         for waste in parameters.waste_fractions.model_dump().keys():
        #             s = sum(components_multiplied_through_dummy[div].get(waste, 0) for div in keys_of_interest)
        #             non_combustion[waste] = s
        #             combustion_all[waste] = getattr(parameters.waste_fractions, waste) - s

        #         adjust_non_combustion = False
        #         for waste, frac in non_combustion.items():
        #             if frac > getattr(parameters.waste_fractions, waste):
        #                 adjust_non_combustion = True

        #         if adjust_non_combustion:
        #             div_component_fractions_adjusted = DivComponentFractions(**div_component_fractions.model_dump())

        #             dont_add_to = {waste for waste, frac in parameters.waste_fractions.model_dump().items() if frac == 0}
        #             problems = [set(waste for waste, frac in non_combustion.items() if frac > getattr(parameters.waste_fractions, waste))]
        #             dont_add_to.update(problems[0])

        #             while problems:
        #                 probs = problems.pop(0)
        #                 for waste in probs:
        #                     remove = {}
        #                     distribute = {}
        #                     overflow = {}
        #                     can_be_adjusted = []
        #                     div_total = sum(getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste) for div in keys_of_interest if waste in getattr(div_component_fractions_adjusted, div).model_dump().keys())
        #                     div_target = getattr(parameters.waste_fractions, waste)
        #                     diff = (div_total - div_target) / div_total

        #                     for div in keys_of_interest:
        #                         if getattr(div_fractions, div) == 0:
        #                             continue
        #                         distribute[div] = {}
        #                         component = getattr(getattr(div_component_fractions_adjusted, div), waste, 0)
        #                         to_be_removed = diff * component

        #                         to_distribute_to = [x for x in self.div_components[div] if x not in dont_add_to]
        #                         to_distribute_to_sum = sum(getattr(getattr(div_component_fractions_adjusted, div), x, 0) for x in to_distribute_to)
        #                         if to_distribute_to_sum == 0:
        #                             overflow[div] = 1
        #                             continue

        #                         for w in to_distribute_to:
        #                             add_amount = to_be_removed * (getattr(getattr(div_component_fractions_adjusted, div), w, 0) / to_distribute_to_sum)
        #                             if w not in distribute[div]:
        #                                 distribute[div][w] = [add_amount]
        #                             else:
        #                                 distribute[div][w].append(add_amount)

        #                         remove[div] = to_be_removed
        #                         can_be_adjusted.append(div)

        #                     for div in overflow:
        #                         component = getattr(getattr(div_component_fractions_adjusted, div), waste, 0)
        #                         to_be_removed = diff * component
        #                         to_distribute_to = [x for x in distribute.keys() if waste in self.div_components[x] and x not in overflow]
        #                         to_distribute_to_sum = sum(getattr(div_fractions, x) for x in to_distribute_to)
        #                         if to_distribute_to_sum == 0:
        #                             raise CustomError("INVALID_PARAMETERS", f"Combination of compost, anaerobic digestion, and recycling is too high")

        #                         for d in to_distribute_to:
        #                             to_be_removed_component = to_be_removed * (getattr(div_fractions, d) / to_distribute_to_sum) / getattr(div_fractions, d)
        #                             to_distribute_to_component = [x for x in getattr(div_component_fractions_adjusted, d).model_dump().keys() if x not in dont_add_to]
        #                             to_distribute_to_sum_component = sum(getattr(getattr(div_component_fractions_adjusted, d), x, 0) for x in to_distribute_to_component)
        #                             if to_distribute_to_sum_component == 0:
        #                                 raise CustomError("INVALID_PARAMETERS", f"Combination of compost, anaerobic digestion, and recycling is too high")

        #                             for w in to_distribute_to_component:
        #                                 add_amount = to_be_removed_component * getattr(getattr(div_component_fractions_adjusted, d), w, 0) / to_distribute_to_sum_component
        #                                 if w in distribute[d]:
        #                                     distribute[d][w].append(add_amount)

        #                             remove[d] += to_be_removed_component

        #                     for div in distribute:
        #                         for w in distribute[div]:
        #                             setattr(getattr(div_component_fractions_adjusted, div), w, getattr(getattr(div_component_fractions_adjusted, div), w) + sum(distribute[div][w]))

        #                     for div in remove:
        #                         setattr(getattr(div_component_fractions_adjusted, div), waste, getattr(getattr(div_component_fractions_adjusted, div), waste) - remove[div])

        #                 new_probs = {waste for waste in parameters.waste_fractions.model_dump().keys() if sum(getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste, 0) for div in keys_of_interest) > getattr(parameters.waste_fractions, waste) + 0.001}
        #                 if new_probs:
        #                     problems.append(new_probs)
        #                 dont_add_to.update(new_probs)

        #             components_multiplied_through_dummy = {
        #                 div: {waste: getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste) for waste in getattr(div_component_fractions_adjusted, div).model_dump().keys()}
        #                 for div in div_component_fractions_adjusted.model_dump().keys()
        #             }

        #         non_combustion = {}
        #         combustion_all = {}
        #         for waste in parameters.waste_fractions.model_dump().keys():
        #             s = sum(components_multiplied_through_dummy[div].get(waste, 0) for div in keys_of_interest)
        #             non_combustion[waste] = s
        #             combustion_all[waste] = getattr(parameters.waste_fractions, waste) - s

        #         adjust_non_combustion = False
        #         for waste, frac in non_combustion.items():
        #             if frac > (getattr(parameters.waste_fractions, waste) + 1e-5):
        #                 adjust_non_combustion = True
        #                 raise CustomError("INVALID_PARAMETERS", f"Combination of compost, anaerobic digestion, and recycling is too high")

        #         all_divs = sum(getattr(div_fractions, div) for div in div_fractions.model_dump().keys())

        #         assert np.abs(div_fractions.recycling - sum(components_multiplied_through_dummy['recycling'].values())) < 1e-3

        #         remainder = sum(fraction for waste_type, fraction in combustion_all.items() if waste_type in self.div_components['combustion'])
        #         combustion_fraction_of_remainder = div_fractions.combustion / remainder
        #         if combustion_fraction_of_remainder > (1 + 1e-5):
        #             non_combustables = [x for x in parameters.waste_fractions.model_dump().keys() if x not in self.div_components['combustion']]
        #             for waste in non_combustables:
        #                 if getattr(parameters.waste_fractions, waste) == 0:
        #                     continue
        #                 new_val = getattr(parameters.waste_fractions, waste) * all_divs
        #                 components_multiplied_through_dummy['recycling'][waste] = new_val

        #             available_div = sum(v for k, v in components_multiplied_through_dummy['recycling'].items() if k not in non_combustables)
        #             available_div_target = div_fractions.recycling - sum(v for k, v in components_multiplied_through_dummy['recycling'].items() if k in non_combustables)
        #             if available_div_target < 0:
        #                 too_much_frac = (sum(v for k, v in components_multiplied_through_dummy['recycling'].items() if k in non_combustables) - div_fractions.recycling) / sum(v for k, v in components_multiplied_through_dummy['recycling'].items() if k in non_combustables)
        #                 for key, value in components_multiplied_through_dummy['recycling'].items():
        #                     if key in non_combustables:
        #                         components_multiplied_through_dummy['recycling'][key] = value * (1 - too_much_frac)
        #                     else:
        #                         components_multiplied_through_dummy['recycling'][key] = 0
        #                 assert np.abs(div_fractions.recycling - sum(v for v in components_multiplied_through_dummy['recycling'].values())) < 1e-5

        #             else:
        #                 reduce_frac = (available_div - available_div_target) / available_div
        #                 for key, value in components_multiplied_through_dummy['recycling'].items():
        #                     if key not in non_combustables:
        #                         components_multiplied_through_dummy['recycling'][key] = value * (1 - reduce_frac)
        #                 assert np.abs(div_fractions.recycling - sum(v for v in components_multiplied_through_dummy['recycling'].values())) < 1e-5

        #             non_combustion = {}
        #             combustion_all = {}
        #             for waste in parameters.waste_fractions.model_dump().keys():
        #                 s = sum(components_multiplied_through_dummy[div].get(waste, 0) for div in keys_of_interest)
        #                 non_combustion[waste] = s
        #                 combustion_all[waste] = getattr(parameters.waste_fractions, waste) - s

        #             remainder = sum(fraction for waste_type, fraction in combustion_all.items() if waste_type in self.div_components['combustion'])
        #             combustion_fraction_of_remainder = div_fractions.combustion / remainder
        #             assert combustion_fraction_of_remainder < (1 + 1e-5)
        #             if combustion_fraction_of_remainder > 1:
        #                 combustion_fraction_of_remainder = 1

        #         for waste in self.div_components['combustion']:
        #             components_multiplied_through_dummy['combustion'][waste] = combustion_fraction_of_remainder * combustion_all[waste]

        #         for d in div_fractions.model_dump().keys():
        #             assert np.abs(getattr(div_fractions, d) - sum(components_multiplied_through_dummy[d].values())) < 1e-3
        #             for w in components_multiplied_through_dummy[d]:
        #                 if abs(components_multiplied_through_dummy[d][w]) < 1e-5:
        #                     components_multiplied_through_dummy[d][w] = 0
        #                 assert components_multiplied_through_dummy[d][w] >= 0

        #         adjusted_div_component_fractions = {
        #             div: {waste: components_multiplied_through_dummy[div][waste] / getattr(div_fractions, div) if getattr(div_fractions, div) != 0 else 0 for waste in components_multiplied_through_dummy[div]}
        #             for div in components_multiplied_through_dummy
        #         }

        #         adjusted_div_component_fractions = DivComponentFractions(**adjusted_div_component_fractions)

        #         divs_adj = self._divs_from_component_fractions(div_fractions, adjusted_div_component_fractions, scenario=scenario)
        #         divs.append(divs_adj)
        #         div_component_fractions_adjusted.append(adjusted_div_component_fractions)

        #     parameters.div_component_fractions = adjusted_div_component_fractions
        #     parameters.divs = divs
        #     parameters.adjusted_diversion_constituents = True
        #     parameters.input_problems = False

    def _divs_from_component_fractions(
        self,
        div_fractions: DiversionFractions,
        div_component_fractions: DivComponentFractions,
        scenario: int,
        advanced: bool = False,
        year: int = 2000,
    ) -> dict:
        """
        Calculates diverted masses from diversion fractions and component fractions,
        incorporating rejection rates.
        Currently only used as part of _check_masses_v2

        Args:
            div_fractions (DiversionFractions): Fractions of waste diverted to diversion types.
            div_component_fractions (DivComponentFractions): Waste type fractions of each diversion type.

        Returns:
            dict: Dictionary containing the resulting masses of waste components diverted to each diversion type.
        """
        if scenario == 0:
            parameters = self.baseline_parameters
        else:
            parameters = self.scenario_parameters[scenario - 1]

        non_compostable_not_targeted_total = sum(
            [
                self.non_compostable_not_targeted[x]
                * div_component_fractions.model_dump().get("compost", {}).get(x, 0.0)
                for x in self.div_components["compost"]
            ]
        )
        parameters.non_compostable_not_targeted_total = pd.Series(
            non_compostable_not_targeted_total, np.arange(1990, 2051)
        )
        if parameters.non_compostable_not_targeted_total.isna().all():
            parameters.non_compostable_not_targeted_total = pd.Series(
                0, index=np.arange(1990, 2051)
            )

        # Deal with waste mass that changes at implement_date first.
        waste_mass = parameters.waste_mass
        if isinstance(waste_mass, Variant) and (
            waste_mass["scenario"] != waste_mass["baseline"]
        ):
            compost_masses = {"baseline": {}, "scenario": {}}
            anaerobic_masses = {"baseline": {}, "scenario": {}}
            combustion_masses = {"baseline": {}, "scenario": {}}
            recycling_masses = {"baseline": {}, "scenario": {}}
            for period in ["baseline", "scenario"]:
                for waste in self.waste_types:
                    compost_masses[period][waste] = (
                        waste_mass[period]
                        * getattr(div_fractions, "compost", 0)
                        * getattr(
                            getattr(div_component_fractions, "compost", {}), waste, 0
                        )
                        * (1 - non_compostable_not_targeted_total)
                        * (1 - self.unprocessable.get(waste, 0))
                    )
                    anaerobic_masses[period][waste] = (
                        waste_mass[period]
                        * getattr(div_fractions, "anaerobic", 0)
                        * getattr(
                            getattr(div_component_fractions, "anaerobic", {}), waste, 0
                        )
                    )
                    combustion_masses[period][waste] = (
                        waste_mass[period]
                        * getattr(div_fractions, "combustion", 0)
                        * getattr(
                            getattr(div_component_fractions, "combustion", {}), waste, 0
                        )
                        * (1 - self.combustion_reject_rate)
                    )
                    recycling_masses[period][waste] = (
                        waste_mass[period]
                        * getattr(div_fractions, "recycling", 0)
                        * getattr(
                            getattr(div_component_fractions, "recycling", {}), waste, 0
                        )
                        * self.recycling_reject_rates.get(waste, 0)
                    )

            divs = {}
            divs["baseline"] = DivMasses(
                compost=WasteMasses(**compost_masses["baseline"]),
                anaerobic=WasteMasses(**anaerobic_masses["baseline"]),
                combustion=WasteMasses(**combustion_masses["baseline"]),
                recycling=WasteMasses(**recycling_masses["baseline"]),
            )
            divs["scenario"] = DivMasses(
                compost=WasteMasses(**compost_masses["scenario"]),
                anaerobic=WasteMasses(**anaerobic_masses["scenario"]),
                combustion=WasteMasses(**combustion_masses["scenario"]),
                recycling=WasteMasses(**recycling_masses["scenario"]),
            )

            return divs

        if isinstance(waste_mass, Variant):
            waste_mass = waste_mass["scenario"]
        if isinstance(waste_mass, pd.Series):
            waste_mass = waste_mass.iat[0]

        compost_masses = {}
        anaerobic_masses = {}
        combustion_masses = {}
        recycling_masses = {}

        # if advanced:
        #     waste_mass = waste_mass.at[year]
        #     try:
        #         non_compostable_not_targeted_total = non_compostable_not_targeted_total.at[year]
        #     except:
        #         pass

        for waste in self.waste_types:
            compost_masses[waste] = (
                waste_mass
                * getattr(div_fractions, "compost", 0)
                * getattr(getattr(div_component_fractions, "compost", {}), waste, 0)
                * (1 - non_compostable_not_targeted_total)
                * (1 - self.unprocessable.get(waste, 0))
            )
            anaerobic_masses[waste] = (
                waste_mass
                * getattr(div_fractions, "anaerobic", 0)
                * getattr(getattr(div_component_fractions, "anaerobic", {}), waste, 0)
            )
            combustion_masses[waste] = (
                waste_mass
                * getattr(div_fractions, "combustion", 0)
                * getattr(getattr(div_component_fractions, "combustion", {}), waste, 0)
                * (1 - self.combustion_reject_rate)
            )
            recycling_masses[waste] = (
                waste_mass
                * getattr(div_fractions, "recycling", 0)
                * getattr(getattr(div_component_fractions, "recycling", {}), waste, 0)
                * self.recycling_reject_rates.get(waste, 0)
            )

        divs = DivMasses(
            compost=WasteMasses(**compost_masses),
            anaerobic=WasteMasses(**anaerobic_masses),
            combustion=WasteMasses(**combustion_masses),
            recycling=WasteMasses(**recycling_masses),
        )

        return divs

    @staticmethod
    def calculate_reduction(
        value: float, limit: float, excess: float, total_reducible: float
    ) -> float:
        """
        Calculate the reduction of a diverted waste type based on a given limit.
        This method is used in calculating parameters from UN Habitat data.

        Args:
            value (float): Current value of the waste component.
            limit (float): Minimum allowable value of the waste component.
            excess (float): Diverted waste above limit for that type.
            total_reducible (float): Total reducible waste from all components.

        Returns:
            float: Amount by which the waste component should be reduced.
        """
        reducible = value - limit  # the amount we can reduce this component by
        reduction = min(
            reducible, excess * (reducible / total_reducible)
        )  # proportional reduction
        return reduction

    # def _create_divs_dataframe(self, baseline_divs, scenario_divs):
    #    """
    #    Create a DataFrame that merges baseline and scenario diversion data based on the implementation year.

    #    Args:
    #        baseline_divs (object): Baseline diversion data.
    #        scenario_divs (object): Scenario diversion data.
    #        implement_year (int): The year when the scenario diversions start being implemented.

    #    Returns:
    #        DataFrame: A DataFrame with years as the index and diversion data as the columns.
    #    """

    #    implement_year = self.scenario_parameters[0].implement_year

    #    baseline_data = {year: {waste: getattr(baseline_divs, waste) for waste in baseline_divs.model_dump()} for year in range(1990, implement_year)}
    #    scenario_data = {year: {waste: getattr(scenario_divs, waste) for waste in scenario_divs.model_dump()} for year in range(implement_year, 2051)}

    #    df = pd.concat([pd.DataFrame(baseline_data).T, pd.DataFrame(scenario_data).T])

    #    return df

    # def _create_waste_fractions_dataframe(self, advanced_dst: bool=False) -> pd.DataFrame:
    #    """
    #    Create a DataFrame that merges baseline and scenario waste fractions data based on the implementation year.

    #    Args:
    #        baseline_waste_fractions (object): Baseline waste fractions data.
    #        scenario_waste_fractions (object): Scenario waste fractions data.
    #        implement_year (int): The year when the scenario waste fractions start being implemented.

    #    Returns:
    #        DataFrame: A DataFrame with years as the index and waste fractions data as the columns.
    #    """
    #    # Come back to this, waste fractions should already be dataframe for advanced_dst
    #    if not advanced_dst:
    #        baseline_waste_fractions = self.baseline_parameters.waste_fractions
    #        scenario_waste_fractions = self.scenario_parameters[0].waste_fractions
    #    implement_year = self.scenario_parameters[0].implement_year

    #    baseline_data = {year: {waste: getattr(baseline_waste_fractions, waste) for waste in baseline_waste_fractions.model_dump()} for year in range(1990, implement_year)}
    #    scenario_data = {year: {waste: getattr(scenario_waste_fractions, waste) for waste in scenario_waste_fractions.model_dump()} for year in range(implement_year, 2051)}

    #    df = pd.concat([pd.DataFrame(baseline_data).T, pd.DataFrame(scenario_data).T])

    #    return df

    def _calculate_net_masses(
        self,
        scenario: int = 0,
        advanced_baseline: bool = False,
        advanced_dst: bool = False,
    ) -> None:
        """
        Calculate the net masses of different types of waste after diversion.

        Args:
            scenario (int): The scenario number to use (0 for baseline, or the number of the alternative scenario).
            advanced_baseline (bool): Flag to indicate if advanced baseline calculations are needed.
            advanced_dst (bool): Flag to indicate if advanced diversion scenario calculations are needed.

        Returns:
            None
        """
        if scenario == 0:
            parameters = self.baseline_parameters
        else:
            parameters = self.scenario_parameters.get(scenario - 1)
            if parameters is None:
                raise ValueError(
                    f"Scenario '{scenario}' not found in scenario_parameters."
                )

        divs = parameters.divs
        implement_year = parameters.implement_year

        # if advanced_dst:
        #     # Combine all divs DataFrames
        #     combined_divs = divs.compost.add(divs.anaerobic, fill_value=0)
        #     combined_divs = combined_divs.add(divs.combustion, fill_value=0)
        #     combined_divs = combined_divs.add(divs.recycling, fill_value=0)

        #     # Subtract the combined divs from waste_masses
        #     new_masses_df = parameters.waste_masses.sub(combined_divs, fill_value=0)

        #     # Assign the result to parameters.net_masses
        #     parameters.net_masses = new_masses_df

        #     return

        if advanced_baseline or advanced_dst:
            # Combine all divs DataFrames
            combined_divs = divs.compost.add(divs.anaerobic, fill_value=0)
            combined_divs = combined_divs.add(divs.combustion, fill_value=0)
            combined_divs = combined_divs.add(divs.recycling, fill_value=0)

            # Subtract the combined divs from waste_masses
            if not isinstance(parameters.waste_masses, pd.DataFrame):
                waste_masses = pd.DataFrame(
                    parameters.waste_masses.model_dump(), index=self.years_range
                )
                new_masses_df = waste_masses.sub(combined_divs, fill_value=0)
            else:
                new_masses_df = parameters.waste_masses.sub(combined_divs, fill_value=0)

            # Assign the result to parameters.net_masses
            parameters.net_masses = new_masses_df

            return

        # net_masses = {waste: parameters.waste_masses.model_dump()[waste] - (
        #                 getattr(divs.compost, waste) +
        #                 getattr(divs.anaerobic, waste) +
        #                 getattr(divs.combustion, waste) +
        #                 getattr(divs.recycling, waste)
        #             ) for waste in parameters.waste_fractions.model_dump()}

        # net = WasteMasses(**net_masses)
        if not parameters.waste_masses:
            waste_mass_dict = {}
            for col in self.waste_types:
                fraction = parameters.waste_fractions.at[1990, col]
                waste_mass_dict[col] = parameters.waste_mass.iloc[0] * fraction
            parameters.waste_masses = WasteMasses(**waste_mass_dict)

        try:
            combined_diversions = (
                pd.concat(
                    [divs.compost, divs.anaerobic, divs.combustion, divs.recycling],
                    axis=1,
                )
                .T.groupby(level=0)
                .sum()
                .T
            )
        except:
            diverted = {
                waste: divs.compost.model_dump().get(waste, 0)
                + divs.anaerobic.model_dump().get(waste, 0)
                + divs.combustion.model_dump().get(waste, 0)
                + divs.recycling.model_dump().get(waste, 0)
                for waste in self.waste_types
            }

        try:
            parameters.net_masses = parameters.waste_generated_df - combined_diversions
        except:
            net = {
                waste: parameters.waste_masses.model_dump()[waste]
                - diverted.get(waste, 0)
                for waste in self.waste_types
            }
            parameters.net_masses = pd.Series(net)

        # return net

    @staticmethod
    def convert_methane_m3_to_ton_co2e(volume_m3: float) -> float:
        """
        Convert methane volume in m^3 to equivalent tons of CO2e.

        Args:
            volume_m3 (float): Volume of methane in cubic meters.

        Returns:
            float: Equivalent CO2e in tons.
        """
        density_kg_per_m3 = 0.7168
        mass_kg = volume_m3 * density_kg_per_m3
        mass_ton = mass_kg / 1000
        mass_co2e = mass_ton * 28
        return mass_co2e

    @staticmethod
    def convert_co2e_to_methane_m3(mass_co2e: float) -> float:
        """
        Convert CO2e in tons to equivalent methane volume in m^3.

        Args:
            mass_co2e (float): CO2e in tons.

        Returns:
            float: Equivalent volume of methane in cubic meters.
        """
        density_kg_per_m3 = 0.7168
        mass_ton = mass_co2e / 28
        mass_kg = mass_ton * 1000
        volume_m3 = mass_kg / density_kg_per_m3
        return volume_m3

    def implement_dst_changes_simple(
        self,
        new_div_fractions: DiversionFractions,
        new_landfill_pct: float,
        new_gas_pct: float,
        implement_year: int,
        scenario: int,
        food_waste_prevention: float = 0,
    ) -> None:
        """
        API endpoint function for implementing CDST changes.

        Args:
            new_div_fractions (DiversionFractions): New diversion fractions.
            new_landfill_pct (float): New landfill percentage.
            new_gas_pct (float): New gas percentage.
            implement_year (int): Year of implementation.
            scenario (int): Scenario number.
            food_waste_prevention (float): Food waste prevention percentage.
        Returns:
            None
        """

        scenario_parameters = copy.deepcopy(self.baseline_parameters)
        self.scenario_parameters[scenario - 1] = scenario_parameters
        scenario_parameters.div_fractions = new_div_fractions

        food_fraction = scenario_parameters.waste_fractions.at[1990, "food"]
        food_waste_prevented = (
            food_waste_prevention * food_fraction * scenario_parameters.waste_mass
        )
        scenario_parameters.waste_mass -= food_waste_prevented
        scenario_parameters.waste_masses.food -= food_waste_prevented * food_fraction
        new_total_waste_fracs = 1 - food_waste_prevention * food_fraction
        if food_waste_prevention > 0:
            for frac in scenario_parameters.waste_fractions.columns:
                old_val = scenario_parameters.waste_fractions.at[1990, frac]
                new_val = old_val / new_total_waste_fracs
                scenario_parameters.waste_fractions.loc[:, frac] = new_val

        # Set new split fractions

        scenario_parameters.split_fractions.dumpsite = 1 - new_landfill_pct
        pct_landfill = 1 - scenario_parameters.split_fractions.dumpsite
        scenario_parameters.split_fractions.landfill_w_capture = (
            new_gas_pct * pct_landfill
        )
        scenario_parameters.split_fractions.landfill_wo_capture = (
            1 - new_gas_pct
        ) * pct_landfill
        scenario_parameters.landfills[0].fraction_of_waste = (
            scenario_parameters.split_fractions.landfill_w_capture
        )
        scenario_parameters.landfills[1].fraction_of_waste = (
            scenario_parameters.split_fractions.landfill_wo_capture
        )
        scenario_parameters.landfills[2].fraction_of_waste = (
            scenario_parameters.split_fractions.dumpsite
        )
        for lf in scenario_parameters.landfills:
            lf.scenario = 1
        scenario_parameters.non_zero_landfills = [
            lf for lf in scenario_parameters.landfills if lf.fraction_of_waste > 0
        ]
        scenario_parameters.implement_year = implement_year

        # Recalculate div_component_fractions
        waste_fractions = scenario_parameters.waste_fractions
        waste_fractions = WasteFractions(**waste_fractions.iloc[0].to_dict())

        # Check if any of the div component fractions are all zero. If so, recalculate so they sum to 1.
        try:
            for div in self.div_components.keys():
                div_component_fractions = getattr(
                    scenario_parameters.div_component_fractions, div
                )
                fractions_sum = div_component_fractions.iloc[0, :].sum()
                if fractions_sum == 0:
                    sum_relevant_fractions = scenario_parameters.waste_fractions.loc[
                        2000, list(self.div_components[div])
                    ].sum()
                    for waste in self.waste_types:
                        if waste in self.div_components[div]:
                            div_component_fractions.loc[:, waste] = (
                                scenario_parameters.waste_fractions.at[2000, waste]
                                / sum_relevant_fractions
                            )
                        else:
                            div_component_fractions.loc[:, waste] = (
                                scenario_parameters.waste_fractions.at[2000, waste]
                                / sum_relevant_fractions
                            )

                # Set the div_component_fractions in the scenario_parameters
                setattr(
                    scenario_parameters.div_component_fractions,
                    div,
                    div_component_fractions,
                )
        except:
            pass

        scenario_parameters.div_component_fractions = DivComponentFractions(
            compost=WasteFractions(
                **{
                    waste: scenario_parameters.div_component_fractions.compost.at[
                        2000, waste
                    ]
                    for waste in scenario_parameters.div_component_fractions.compost.columns
                }
            ),
            anaerobic=WasteFractions(
                **{
                    waste: scenario_parameters.div_component_fractions.anaerobic.at[
                        2000, waste
                    ]
                    for waste in scenario_parameters.div_component_fractions.anaerobic.columns
                }
            ),
            combustion=WasteFractions(
                **{
                    waste: scenario_parameters.div_component_fractions.combustion.at[
                        2000, waste
                    ]
                    for waste in scenario_parameters.div_component_fractions.combustion.columns
                }
            ),
            recycling=WasteFractions(
                **{
                    waste: scenario_parameters.div_component_fractions.recycling.at[
                        2000, waste
                    ]
                    for waste in scenario_parameters.div_component_fractions.recycling.columns
                }
            ),
        )
        scenario_parameters.non_compostable_not_targeted_total = sum(
            [
                self.non_compostable_not_targeted[x]
                * getattr(scenario_parameters.div_component_fractions.compost, x)
                for x in self.div_components["compost"]
            ]
        )
        if np.isnan(scenario_parameters.non_compostable_not_targeted_total):
            scenario_parameters.non_compostable_not_targeted_total = 0.0
        self._calculate_diverted_masses(
            scenario=scenario
        )  # This function could be moved to cityparameters class, and then it doesn't need scenario argument

        # scenario_parameters.repopulate_attr_dicts()
        self._check_masses_v2(scenario=scenario)

        if scenario_parameters.input_problems:
            raise CustomError("INVALID_PARAMETERS", "Invalid new value")

        self._calculate_net_masses(scenario=scenario)
        for w in scenario_parameters.net_masses.index:
            mass = scenario_parameters.net_masses.at[w]
            if mass < 0:
                raise CustomError(
                    "INVALID_PARAMETERS",
                    f"Negative mass for {w} in scenario {scenario}: {mass}",
                )

        scenario_parameters.divs_df = DivsDF(
            compost=scenario_parameters.divs_df.compost,
            anaerobic=scenario_parameters.divs_df.anaerobic,
            combustion=scenario_parameters.divs_df.combustion,
            recycling=scenario_parameters.divs_df.recycling,
        )

        # Convert divs to a DivMasses object
        compost_dict = self.baseline_parameters.divs.compost.iloc[0].to_dict()
        anaerobic_dict = self.baseline_parameters.divs.anaerobic.iloc[0].to_dict()
        combustion_dict = self.baseline_parameters.divs.combustion.iloc[0].to_dict()
        recycling_dict = self.baseline_parameters.divs.recycling.iloc[0].to_dict()

        def fill_missing_fields(d: dict) -> dict:
            return {field: d.get(field, 0.0) for field in self.waste_types}

        compost_dict_complete = fill_missing_fields(compost_dict)
        anaerobic_dict_complete = fill_missing_fields(anaerobic_dict)
        combustion_dict_complete = fill_missing_fields(combustion_dict)
        recycling_dict_complete = fill_missing_fields(recycling_dict)

        compost_wm = WasteMasses(**compost_dict_complete)
        anaerobic_wm = WasteMasses(**anaerobic_dict_complete)
        combustion_wm = WasteMasses(**combustion_dict_complete)
        recycling_wm = WasteMasses(**recycling_dict_complete)

        baseline_divs = DivMasses(
            compost=compost_wm,
            anaerobic=anaerobic_wm,
            combustion=combustion_wm,
            recycling=recycling_wm,
        )

        try:
            yr_pop = scenario_parameters.year_of_data_pop["baseline"]
        except:
            yr_pop = scenario_parameters.year_of_data_pop

        scenario_parameters.divs_df = DivsDF.create_simple(
            baseline_divs=baseline_divs,
            scenario_divs=scenario_parameters.divs,
            start_year=1990,
            end_year=2050,
            implement_year=implement_year,
            year_of_data_pop=yr_pop,
            growth_rate_historic=scenario_parameters.growth_rate_historic,
            growth_rate_future=scenario_parameters.growth_rate_future,
        )

        # combine these two loops maybe...though it still does six things, maybe doesn't matter
        scenario_parameters.repopulate_attr_dicts()
        for i, landfill in enumerate(scenario_parameters.landfills):
            # Might be able to do this more efficienctly...i'm looping over the pre implementation years twice sort of
            landfill.waste_mass_df = LandfillWasteMassDF.create(
                scenario_parameters.waste_generated_df,
                scenario_parameters.divs_df,
                landfill.fraction_of_waste,
                self.components,
            ).df
            landfill.waste_mass_df.loc[: (implement_year - 1), :] = (
                self.baseline_parameters.landfills[i].waste_mass_df.loc[
                    : (implement_year - 1), :
                ]
            )
            # print(landfill.waste_mass_df)

        # scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in scenario_parameters.landfills:
            landfill.estimate_emissions()
            # print(landfill.emissions)

        self.estimate_diversion_emissions(scenario=scenario)
        self.sum_landfill_emissions(scenario=scenario, simple=True)

    def implement_dst_changes_simple_v1_5(
        self,
        new_div_fractions: DiversionFractions,
        add_gas: bool,
        move_gas: bool,
        new_gas_pct: float,
        existing_gas_pct: float,
        implement_year: int,
        scenario: int,
        food_waste_prevention: float = 0,
    ) -> None:
        """
        API endpoint function for implementing CDST changes.

        Args:
            new_div_fractions (DiversionFractions): New diversion fractions.
            new_landfill_pct (float): New landfill percentage.
            new_gas_pct (float): New gas percentage.
            implement_year (int): Year of implementation.
            scenario (int): Scenario number.
            food_waste_prevention (float): Food waste prevention percentage.
        Returns:
            None
        """

        scenario_parameters = copy.deepcopy(self.baseline_parameters)
        self.scenario_parameters[scenario - 1] = scenario_parameters
        scenario_parameters.div_fractions = new_div_fractions
        waste_fractions_sum = scenario_parameters.waste_fractions.sum(axis=1).iat[0]
        food_fraction = scenario_parameters.waste_fractions["food"].iat[0]
        food_waste_prevented = (
            food_waste_prevention * food_fraction * scenario_parameters.waste_mass
        ).iat[0]
        scenario_parameters.waste_mass -= food_waste_prevented
        scenario_parameters.waste_masses.food -= food_waste_prevented
        # Food prevention removes food mass and shrinks the total; every other
        # type's mass is unchanged. Rescale ALL fractions by the same
        # total-reduction factor (the factor waste_mass was just reduced by,
        # above) so the invariant waste_fractions[w] * waste_mass == waste_masses[w]
        # stays true. The previous non-food-only rescale (old_nonfood/new_nonfood)
        # over-inflated non-food shares, so the allocator believed more
        # metal/glass/other existed than the unchanged masses actually hold ->
        # spurious "Negative mass for <type>".
        total_scale = 1 - food_waste_prevention * food_fraction  # reduced_total / original_total
        if food_waste_prevention > 0:
            for frac in scenario_parameters.waste_fractions.columns:
                if frac == "food":
                    scenario_parameters.waste_fractions.loc[:, "food"] = (
                        food_fraction * (1 - food_waste_prevention) / total_scale
                    )
                    continue
                old_val = scenario_parameters.waste_fractions[frac].iat[0]
                scenario_parameters.waste_fractions.loc[:, frac] = old_val / total_scale

        if np.abs(scenario_parameters.waste_fractions.sum(axis=1).iat[0] - 1) > 1e-2:
            raise CustomError(
                "INVALID_PARAMETERS",
                f"Invalid waste fractions: {scenario_parameters.waste_fractions}",
            )

        # Set new split fractions

        # scenario_parameters.split_fractions.dumpsite = 1 - new_landfill_pct
        # pct_landfill = 1 - scenario_parameters.split_fractions.dumpsite
        # scenario_parameters.split_fractions.landfill_w_capture = new_gas_pct * pct_landfill
        # scenario_parameters.split_fractions.landfill_wo_capture = (1 - new_gas_pct) * pct_landfill
        # scenario_parameters.landfills[0].fraction_of_waste = scenario_parameters.split_fractions.landfill_w_capture
        # scenario_parameters.landfills[1].fraction_of_waste = scenario_parameters.split_fractions.landfill_wo_capture
        # scenario_parameters.landfills[2].fraction_of_waste = scenario_parameters.split_fractions.dumpsite
        # for lf in scenario_parameters.landfills:
        #    lf.scenario = 1
        # scenario_parameters.non_zero_landfills = [lf for lf in scenario_parameters.landfills if lf.fraction_of_waste > 0]
        # scenario_parameters.implement_year = implement_year

        if not scenario_parameters.sites_method:
            skip_ox = False
            if add_gas:
                scenario_parameters.landfills[0].gas_capture_efficiency = pd.Series(
                    0.6, index=range(1990, 2051)
                )
                scenario_parameters.landfills[0].oxidation_factor = pd.Series(
                    0.22, index=range(1990, 2051)
                )
                scenario_parameters.landfills[0].mcf = pd.Series(
                    1, index=range(1990, 2051)
                )

                scenario_parameters.landfills[1].gas_capture_efficiency = pd.Series(
                    0.6, index=range(1990, 2051)
                )
                scenario_parameters.landfills[1].gas_capture_efficiency.loc[
                    :implement_year
                ] = 0.0
                scenario_parameters.landfills[1].oxidation_factor = pd.Series(
                    0.22, index=range(1990, 2051)
                )
                scenario_parameters.landfills[1].oxidation_factor.loc[
                    :implement_year
                ] = 0.1
                scenario_parameters.landfills[1].mcf = pd.Series(
                    1, index=range(1990, 2051)
                )
                # Here we convert the dumpsite to a controlled dumpsite w gas capture
                scenario_parameters.landfills[2].gas_capture_efficiency = pd.Series(
                    0.3, index=range(1990, 2051)
                )
                scenario_parameters.landfills[2].gas_capture_efficiency.loc[
                    :implement_year
                ] = 0.0
                scenario_parameters.landfills[2].oxidation_factor = pd.Series(
                    0.1, index=range(1990, 2051)
                )
                scenario_parameters.landfills[2].oxidation_factor.loc[
                    :implement_year
                ] = 0.0
                scenario_parameters.landfills[2].mcf = pd.Series(
                    0.7, index=range(1990, 2051)
                )
                scenario_parameters.landfills[2].mcf.loc[:implement_year] = 0.4
                skip_ox = True

            if move_gas:
                original_gas_pct = (
                    scenario_parameters.split_fractions.landfill_w_capture
                )
                scenario_parameters.split_fractions.landfill_w_capture = (
                    existing_gas_pct
                )
                if original_gas_pct > 0:
                    ratio = existing_gas_pct / original_gas_pct
                else:
                    ratio = 0
                scenario_parameters.split_fractions.landfill_wo_capture *= ratio
                scenario_parameters.split_fractions.dumpsite *= ratio

            if new_gas_pct > 0:
                scenario_parameters.landfills.append(
                    copy.deepcopy(scenario_parameters.landfills[0])
                )
                scenario_parameters.split_fractions.new_w_capture = new_gas_pct
                scenario_parameters.split_fractions.landfill_w_capture *= (
                    1 - new_gas_pct
                )
                scenario_parameters.split_fractions.landfill_wo_capture *= (
                    1 - new_gas_pct
                )
                scenario_parameters.split_fractions.dumpsite *= 1 - new_gas_pct
                total = sum(scenario_parameters.split_fractions.model_dump().values())
                if abs(total - 1.0) > 1e-3:
                    raise CustomError(
                        f"Invalid split fractions: {scenario_parameters.split_fractions}"
                    )
                scenario_parameters.landfills[0].fraction_of_waste = (
                    scenario_parameters.split_fractions.landfill_w_capture
                )
                scenario_parameters.landfills[1].fraction_of_waste = (
                    scenario_parameters.split_fractions.landfill_wo_capture
                )
                scenario_parameters.landfills[2].fraction_of_waste = (
                    scenario_parameters.split_fractions.dumpsite
                )
                scenario_parameters.landfills[3].fraction_of_waste = (
                    scenario_parameters.split_fractions.new_w_capture
                )

            for lf in scenario_parameters.landfills:
                lf.scenario = 1

        else:
            if add_gas:
                for lf in scenario_parameters.landfills:
                    if lf.site_type == "Sanitary Landfill":
                        lf.gas_capture_efficiency.loc[implement_year:] = 0.6
                        lf.oxidation_factor.loc[implement_year:] = 0.22
                        lf.mcf.loc[implement_year:] = 1

            if move_gas:
                # 1) grab the 2024 fractions and gas-flags as numpy arrays
                fracs24 = np.array(
                    [
                        lf.fraction_of_waste_vector.at[2024]
                        for lf in self.baseline_parameters.landfills
                    ]
                )
                is_gas = np.array(
                    [
                        lf.gas_capture_efficiency.at[2024] > 0
                        for lf in self.baseline_parameters.landfills
                    ]
                )

                # 2) compute the total baseline shares for gas vs no-gas
                sum_gas = fracs24[is_gas].sum()
                sum_nogas = fracs24[~is_gas].sum()

                if sum_gas == 0:
                    raise CustomError(
                        f"Invalid gas capture percentage: {existing_gas_pct}"
                    )

                if sum_nogas != 0:
                    # 3) build the two scale-factors
                    r_g = existing_gas_pct / sum_gas
                    r_ng = (1 - existing_gas_pct) / sum_nogas

                    # 4) apply them
                    ratios = np.where(is_gas, r_g, r_ng)
                    for lf, scale in zip(scenario_parameters.landfills, ratios):
                        lf.fraction_of_waste_vector.loc[implement_year:] *= scale

            if new_gas_pct > 0:
                current_gas_pct = 0
                gascap_lfs = []
                nogas_lfs = []
                for i, lf in enumerate(scenario_parameters.landfills):
                    if isinstance(lf.gas_capture_efficiency, pd.Series):
                        existing_gascap = lf.gas_capture_efficiency.at[2024]
                    else:
                        existing_gascap = lf.gas_capture_efficiency
                    if lf.gas_capture_efficiency > 0:
                        current_gas_pct += lf.fraction_of_waste_vector.at[2024]
                        gascap_lfs.append(i)
                    else:
                        nogas_lfs.append(i)

                new_nongas_pct = 1 - new_gas_pct
                for i in nogas_lfs:
                    scenario_parameters.landfills[i].fraction_of_waste_vector.loc[
                        implement_year:
                    ] *= new_nongas_pct

                fraction_of_waste_vector = pd.Series(0.0, index=self.years_range)
                fraction_of_waste_vector.loc[implement_year:] = new_gas_pct
                new_landfill = Landfill(
                    open_date=implement_year,
                    close_date=2050,
                    site_type="Sanitary Landfill",
                    mcf=pd.Series(1, index=range(implement_year, 2051)),
                    city_params_dict=scenario_parameters.landfills[i].city_params_dict,
                    city_instance_attrs=scenario_parameters.landfills[
                        i
                    ].city_instance_attrs,
                    landfill_index=i + 1,
                    # fraction_of_waste=new_landfill_fracs[i],
                    gas_capture=True,
                    scenario=0,
                    new_baseline=False,
                    gas_capture_efficiency=pd.Series(
                        0.6, index=range(implement_year, 2051)
                    ),
                    # flaring=pd.Series(flaring, index=year_range),
                    # leachate_circulate=leachate_circulate[i],
                    fraction_of_waste_vector=fraction_of_waste_vector,
                    advanced=True,
                    latlon=lf.latlon,
                    ks=scenario_parameters.ks,
                    oxidation_factor=pd.Series(0.22, index=range(implement_year, 2051)),
                    rmi_id=999999999,
                )
                scenario_parameters.landfills.append(new_landfill)

        # Recalculate div_component_fractions
        waste_fractions = scenario_parameters.waste_fractions
        waste_fractions = WasteFractions(**waste_fractions.iloc[0].to_dict())

        # Check if any of the div component fractions are all zero. If so, recalculate so they sum to 1.

        for div in self.div_components.keys():
            div_component_fractions = getattr(
                scenario_parameters.div_component_fractions, div
            )
            fractions_sum = div_component_fractions.iloc[0, :].sum()
            if fractions_sum == 0:
                sum_relevant_fractions = scenario_parameters.waste_fractions.loc[
                    2000, list(self.div_components[div])
                ].sum()
                for waste in self.waste_types:
                    if waste in self.div_components[div]:
                        div_component_fractions.loc[:, waste] = (
                            scenario_parameters.waste_fractions.at[2000, waste]
                            / sum_relevant_fractions
                        )
                    else:
                        div_component_fractions.loc[:, waste] = (
                            scenario_parameters.waste_fractions.at[2000, waste]
                            / sum_relevant_fractions
                        )

                # Set the div_component_fractions in the scenario_parameters
                setattr(
                    scenario_parameters.div_component_fractions,
                    div,
                    div_component_fractions,
                )

        scenario_parameters.div_component_fractions = DivComponentFractions(
            compost=WasteFractions(
                **{
                    waste: scenario_parameters.div_component_fractions.compost.at[
                        2000, waste
                    ]
                    for waste in scenario_parameters.div_component_fractions.compost.columns
                }
            ),
            anaerobic=WasteFractions(
                **{
                    waste: scenario_parameters.div_component_fractions.anaerobic.at[
                        2000, waste
                    ]
                    for waste in scenario_parameters.div_component_fractions.anaerobic.columns
                }
            ),
            combustion=WasteFractions(
                **{
                    waste: scenario_parameters.div_component_fractions.combustion.at[
                        2000, waste
                    ]
                    for waste in scenario_parameters.div_component_fractions.combustion.columns
                }
            ),
            recycling=WasteFractions(
                **{
                    waste: scenario_parameters.div_component_fractions.recycling.at[
                        2000, waste
                    ]
                    for waste in scenario_parameters.div_component_fractions.recycling.columns
                }
            ),
        )
        scenario_parameters.non_compostable_not_targeted_total = sum(
            [
                self.non_compostable_not_targeted[x]
                * getattr(scenario_parameters.div_component_fractions.compost, x)
                for x in self.div_components["compost"]
            ]
        )
        if isinstance(scenario_parameters.sites_method, pd.Series):
            if scenario_parameters.non_compostable_not_targeted_total.isna().sum() > 0:
                scenario_parameters.non_compostable_not_targeted_total.loc[:] = 0.0
        else:
            if np.isnan(scenario_parameters.non_compostable_not_targeted_total):
                scenario_parameters.non_compostable_not_targeted_total = 0.0

        self._calculate_diverted_masses(
            scenario=scenario
        )  # This function could be moved to cityparameters class, and then it doesn't need scenario argument

        # scenario_parameters.repopulate_attr_dicts()
        self._check_masses_v2(scenario=scenario)

        if scenario_parameters.input_problems:
            raise CustomError("INVALID_PARAMETERS", "Invalid new value")

        self._calculate_net_masses(scenario=scenario)
        for w in scenario_parameters.net_masses.index:
            mass = scenario_parameters.net_masses.at[w]
            if mass < 0:
                raise CustomError(
                    "INVALID_PARAMETERS",
                    f"Negative mass for {w} in scenario {scenario}: {mass}",
                )

        try:
            yr_pop = scenario_parameters.year_of_data_pop["baseline"]
        except:
            yr_pop = scenario_parameters.year_of_data_pop

        compost_dict = self.baseline_parameters.divs.compost.loc[yr_pop, :].to_dict()
        anaerobic_dict = self.baseline_parameters.divs.anaerobic.loc[
            yr_pop, :
        ].to_dict()
        combustion_dict = self.baseline_parameters.divs.combustion.loc[
            yr_pop, :
        ].to_dict()
        recycling_dict = self.baseline_parameters.divs.recycling.loc[
            yr_pop, :
        ].to_dict()

        def fill_missing_fields(d: dict) -> dict:
            return {field: d.get(field, 0.0) for field in self.waste_types}

        compost_dict_complete = fill_missing_fields(compost_dict)
        anaerobic_dict_complete = fill_missing_fields(anaerobic_dict)
        combustion_dict_complete = fill_missing_fields(combustion_dict)
        recycling_dict_complete = fill_missing_fields(recycling_dict)

        baseline_divs = DivMasses(
            compost=WasteMasses(**compost_dict_complete),
            anaerobic=WasteMasses(**anaerobic_dict_complete),
            combustion=WasteMasses(**combustion_dict_complete),
            recycling=WasteMasses(**recycling_dict_complete),
            waste_change_flag=False,
        )

        scenario_parameters.divs_df = DivsDF.create_simple(
            baseline_divs=baseline_divs,
            scenario_divs=scenario_parameters.divs,
            start_year=1990,
            end_year=2050,
            implement_year=implement_year,
            year_of_data_pop=yr_pop,
            growth_rate_historic=scenario_parameters.growth_rate_historic,
            growth_rate_future=scenario_parameters.growth_rate_future,
        )

        pos = self.years_range.index(implement_year)
        waste_masses_df = pd.concat(
            [
                pd.DataFrame(
                    self.baseline_parameters.waste_masses.model_dump(),
                    index=self.years_range[:pos],
                ),
                pd.DataFrame(
                    scenario_parameters.waste_masses.model_dump(),
                    index=self.years_range[pos:],
                ),
            ]
        )

        scenario_parameters.waste_generated_df = WasteGeneratedDF.create(
            waste_masses_df,
            1990,
            2050,
            scenario_parameters.year_of_data_pop,
            scenario_parameters.growth_rate_historic,
            scenario_parameters.growth_rate_future,
        )

        scenario_parameters.repopulate_attr_dicts()
        if scenario_parameters.sites_method:
            for i, landfill in enumerate(scenario_parameters.landfills):
                landfill.waste_mass_df = LandfillWasteMassDF.create_advanced(
                    waste_generated_df=scenario_parameters.waste_generated_df.df,
                    divs_df=scenario_parameters.divs_df,
                    fraction_of_waste_series=landfill.fraction_of_waste_vector,
                ).df
        else:
            for i, landfill in enumerate(scenario_parameters.landfills):
                landfill.waste_mass_df = LandfillWasteMassDF.create(
                    scenario_parameters.waste_generated_df.df,
                    scenario_parameters.divs_df,
                    landfill.fraction_of_waste,
                    self.components,
                ).df
                if i == 3:
                    landfill.waste_mass_df.loc[: (implement_year - 1), :] *= 0.0
                else:
                    landfill.waste_mass_df.loc[: (implement_year - 1), :] = (
                        self.baseline_parameters.landfills[i].waste_mass_df.loc[
                            : (implement_year - 1), :
                        ]
                    )

        # scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        skip_ox = scenario_parameters.sites_method
        for landfill in scenario_parameters.landfills:
            landfill.estimate_emissions(skip_ox=skip_ox)
            # print(landfill.emissions)

        self.estimate_diversion_emissions(scenario=scenario)
        self.sum_landfill_emissions(scenario=scenario, simple=True)

    def implement_dst_changes_advanced(
        self,
        population: float,
        precipitation: float,
        new_waste_mass: Dict,
        new_waste_fractions: Dict,
        new_div_fractions: Dict,
        new_landfill_types: Dict,
        new_landfill_open_close_dates: Dict,
        implement_year: float,
        scenario: int,
        # new_baseline: int,
        landfill_split_timeline: Dict,
        new_gas_efficiency: Dict,  # 0 means no gas capture, blank means figure out the efficiency for me
        new_landfill_fracs: Dict = None,
        new_landfill_flaring: Dict = None,
        new_landfill_cover: Dict = None,
        leachate_circulate: Dict = None,
        new_landfill_latlons: Dict = None,
        new_landfill_areas: Dict = None,
        new_covertypes: Dict = None,
        new_coverthicknesses: Dict = None,
        waste_burning: Dict = None,
        fancy_ox: Dict = {"baseline": False, "scenario": False},
        new_waste_mass_per_capita: bool = False,
        depths: Dict = None,
        k_values: Dict = None,
        waste_mass_year: int = None,
        ks_overrides: Dict = None,
        biocover: Dict = {"baseline": 0.0, "scenario": 0.0},
        oxidation_override: Dict = None,
    ) -> None:
        """
        API endpoint function for implementing advanced diversion scenario changes.
        Args:
            Lots
        Returns:
            None
        """

        scenario_parameters = copy.deepcopy(self.baseline_parameters)
        self.scenario_parameters[scenario - 1] = scenario_parameters
        scenario_parameters.div_fractions = new_div_fractions
        scenario_parameters.waste_fractions = new_waste_fractions
        scenario_parameters._singapore_k(
            implement_year=implement_year, advanced_dst=True
        )
        scenario_parameters.implement_year = implement_year

        pd.set_option("display.max_rows", None)

        if new_waste_mass:
            pass
        elif new_waste_mass_per_capita:
            new_waste_mass = {}
            new_waste_mass["baseline"] = new_waste_mass_per_capita * population
            new_waste_mass["scenario"] = new_waste_mass_per_capita * population
        else:
            new_waste_mass = {}
            new_waste_mass["baseline"] = scenario_parameters.waste_mass.iat[0]
            new_waste_mass["scenario"] = scenario_parameters.waste_mass.iat[0]
        scenario_parameters.waste_mass = new_waste_mass

        years = pd.Index(range(1990, 2051))
        waste_mass_series = pd.Series(index=years)
        waste_mass_series.loc[: waste_mass_year - 1] = new_waste_mass["baseline"]
        waste_mass_series.loc[waste_mass_year:] = new_waste_mass["scenario"]

        # Adjust for waste burning
        waste_burned = {}
        wb = None
        if waste_burning["baseline"] > 0:
            waste_burned["baseline"] = waste_burning["baseline"] * waste_mass_series
            waste_mass_series.loc[: waste_mass_year - 1] -= waste_burned[
                "baseline"
            ].loc[: waste_mass_year - 1]

            # Adjust the waste burning for growth rates to get real time series
            t = (
                waste_mass_series.index.values
                - scenario_parameters.year_of_data_pop["baseline"]
            )

            # Create growth rate array, using growth_rate_historic for years before year_of_data_pop and growth_rate_future after
            growth_rate = np.where(
                waste_mass_series.index.values
                < scenario_parameters.year_of_data_pop["baseline"],
                scenario_parameters.growth_rate_historic,
                scenario_parameters.growth_rate_future,
            )
            growth_factors = growth_rate**t

            # Apply growth factors
            waste_burned["baseline"] = waste_burned["baseline"].multiply(
                growth_factors, axis=0
            )
            wb = waste_burned["baseline"]

        if waste_burning["scenario"] > 0:
            waste_burned["scenario"] = waste_burning["scenario"] * waste_mass_series
            waste_mass_series.loc[waste_mass_year:] -= waste_burned["scenario"].loc[
                waste_mass_year:
            ]

            # Adjust the waste burning for growth rates to get real time series
            t = (
                waste_mass_series.index.values
                - scenario_parameters.year_of_data_pop["scenario"]
            )

            # Create growth rate array, using growth_rate_historic for years before year_of_data_pop and growth_rate_future after
            growth_rate = np.where(
                waste_mass_series.index.values
                < scenario_parameters.year_of_data_pop["scenario"],
                scenario_parameters.growth_rate_historic,
                scenario_parameters.growth_rate_future,
            )
            growth_factors = growth_rate**t

            # Apply growth factors
            waste_burned["scenario"] = waste_burned["scenario"].multiply(
                growth_factors, axis=0
            )
            if wb is not None:
                wb.loc[implement_year:] = waste_burned["scenario"].loc[implement_year:]
            else:
                wb = waste_burned["scenario"]
                wb.loc[: implement_year - 1] = 0

        if wb is None:
            wb = pd.Series(0, index=years)

        waste_burned = wb

        # New waste masses
        # waste_masses = {}
        # waste_masses['baseline'] = {waste: frac * new_waste_mass['baseline'] for waste, frac in new_waste_fractions['baseline'].model_dump().items()}
        # waste_masses['scenario'] = {waste: frac * new_waste_mass['scenario'] for waste, frac in new_waste_fractions['scenario'].model_dump().items()}
        # scenario_parameters.waste_masses['baseline'] = WasteMasses(**waste_masses['baseline'])
        # scenario_parameters.waste_masses['scenario'] = WasteMasses(**waste_masses['scenario'])

        # Create an empty DataFrame for waste masses by waste type
        waste_masses_df = pd.DataFrame(
            index=years, columns=new_waste_fractions["baseline"].model_dump().keys()
        )

        # Fill the DataFrame with the calculated waste masses
        for waste in waste_masses_df.columns:
            baseline_frac = new_waste_fractions["baseline"].model_dump()[waste]
            scenario_frac = new_waste_fractions["scenario"].model_dump()[waste]

            waste_masses_df.loc[: waste_mass_year - 1, waste] = (
                baseline_frac * waste_mass_series.loc[: waste_mass_year - 1]
            )
            waste_masses_df.loc[waste_mass_year:, waste] = (
                scenario_frac * waste_mass_series.loc[waste_mass_year:]
            )

        # Assign the DataFrame to scenario_parameters
        scenario_parameters.waste_masses = waste_masses_df

        # Update waste generated
        scenario_parameters.waste_generated_df = WasteGeneratedDF.create_advanced(
            waste_masses_df=waste_masses_df,
            start_year=1990,
            end_year=2050,
            year_of_data_pop=scenario_parameters.year_of_data_pop["baseline"],
            growth_rate_historic=scenario_parameters.growth_rate_historic,
            growth_rate_future=scenario_parameters.growth_rate_future,
            implement_year=waste_mass_year,
        ).df

        # Create a DataFrame for fraction_waste_timeline
        fraction_df = pd.DataFrame(landfill_split_timeline).transpose()
        fraction_df.columns = [f"Landfill_{i}" for i in range(fraction_df.shape[1])]
        fraction_df.index.name = "Year"

        # Set up new landfills
        city_params_dict = self.update_cityparams_dict(scenario_parameters)
        # mcfs = [1, 0.7, 0.4] # Should this include ameliorated?
        # mcf_ameliorated = [0.7, 0.4, 0.1]
        mcf_options = [1, 0.6, 0.4]
        gas_capture_efficiencies = {}
        gas_capture_efficiencies["ameliorated"] = [0.5, 0.3, 0]
        gas_capture_efficiencies["not_ameliorated"] = [0.6, 0.45, 0]
        self.ox_options = {
            "ox_nocap": {"landfill": 0.1, "controlled_dumpsite": 0.05, "dumpsite": 0.0},
            "ox_cap": {"landfill": 0.22, "controlled_dumpsite": 0.1, "dumpsite": 0.0},
        }
        landfill_types = ["landfill", "controlled_dumpsite", "dumpsite"]
        scenario_parameters.landfills = []
        for i, lf_type in enumerate(new_landfill_types["scenario"]):
            # Make the MCF, oxidation, and efficiency vectors
            years = pd.Index(range(1990, 2051))
            mcf = {}
            ox_value = {}
            gas_eff = {}

            # Get MCF
            old_lf_type = new_landfill_types["baseline"][i]
            mcf["baseline"] = mcf_options[old_lf_type]
            mcf["scenario"] = mcf_options[lf_type]

            if (depths["baseline"][i] > 5) and (old_lf_type in (1, 2)):
                mcf["baseline"] = 0.8

            if (depths["scenario"][i] > 5) and (lf_type in (1, 2)):
                mcf["scenario"] = 0.8

            # Handle baseline first
            if i >= len(new_gas_efficiency["baseline"]):
                ox_value["baseline"] = 0.0
                gas_eff["baseline"] = 0
            elif new_gas_efficiency["baseline"][i] == 0.0:
                ox_value["baseline"] = self.ox_options["ox_nocap"][
                    landfill_types[old_lf_type]
                ]
                gas_eff["baseline"] = 0
            # If there is gas capture, use the number or figure it out
            elif new_gas_efficiency["baseline"][i] > 0.0:
                ox_value["baseline"] = self.ox_options["ox_cap"][
                    landfill_types[old_lf_type]
                ]
                gas_eff["baseline"] = (
                    new_gas_efficiency["baseline"][i]
                    if new_gas_efficiency["baseline"][i] is not None
                    else gas_capture_efficiencies["not_ameliorated"][old_lf_type]
                )
            else:
                print("invalid gas efficiency value")

            # For scenario, handle no gas capture first
            if new_gas_efficiency["scenario"][i] == 0:
                ox_value["scenario"] = self.ox_options["ox_nocap"][
                    landfill_types[lf_type]
                ]
                gas_eff["scenario"] = 0
            # If there is gas capture, use the number or figure it out
            elif new_gas_efficiency["scenario"][i] > 0.0:
                if (
                    new_landfill_types["scenario"][i]
                    < new_landfill_types["baseline"][i]
                ):
                    ameliorated = True
                    if lf_type == 0:
                        ox_value["scenario"] = 0.18
                    else:
                        ox_value["scenario"] = self.ox_options["ox_cap"][
                            landfill_types[lf_type]
                        ]
                    gas_eff["scenario"] = (
                        new_gas_efficiency["baseline"][i]
                        if new_gas_efficiency["baseline"][i] is not None
                        else gas_capture_efficiencies["ameliorated"][lf_type]
                    )
                else:
                    ameliorated = False
                    ox_value["scenario"] = self.ox_options["ox_cap"][
                        landfill_types[lf_type]
                    ]
                    gas_eff["scenario"] = (
                        new_gas_efficiency["baseline"][i]
                        if new_gas_efficiency["baseline"][i] is not None
                        else gas_capture_efficiencies["not_ameliorated"][lf_type]
                    )
            else:
                print("invalid gas efficiency value")

            if i >= len(new_gas_efficiency["baseline"]):
                pass
            elif new_gas_efficiency["baseline"][i] is not None:
                gas_eff["baseline"] = new_gas_efficiency["baseline"][i]
            if new_gas_efficiency["scenario"][i] is not None:
                gas_eff["scenario"] = new_gas_efficiency["scenario"][i]

            # Create pandas Series for each: mcf, ox_value, and gas_eff
            mcf_series = pd.Series(index=years)
            ox_value_series = pd.Series(index=years)
            gas_eff_series = pd.Series(index=years)

            # Assign baseline values before implement_year and scenario values after
            mcf_series.loc[years < implement_year] = mcf["baseline"]
            mcf_series.loc[years >= implement_year] = mcf["scenario"]

            if biocover["baseline"] > 0:
                ox_value["baseline"] = biocover["baseline"]
            if biocover["scenario"] > 0:
                ox_value["scenario"] = biocover["scenario"]
            ox_value_series.loc[years < implement_year] = ox_value["baseline"]
            ox_value_series.loc[years >= implement_year] = ox_value["scenario"]

            gas_eff_series.loc[years < implement_year] = gas_eff["baseline"]
            gas_eff_series.loc[years >= implement_year] = gas_eff["scenario"]

            doing_fancy_ox = fancy_ox

            if ks_overrides is not None:
                ks_series = pd.Series(
                    [ks_overrides["baseline"]] * len(years), index=years
                )
                ks_series.loc[implement_year:] = ks_overrides["scenario"]
                landfill_ks = DecompositionRates(
                    food=ks_series,
                    green=ks_series,
                    wood=ks_series,
                    paper_cardboard=ks_series,
                    textiles=ks_series,
                )
            else:
                landfill_ks = scenario_parameters.ks

            if oxidation_override:
                if oxidation_override["baseline"]:
                    ox_value_series.loc[: implement_year - 1] = oxidation_override[
                        "baseline"
                    ]
                if oxidation_override["scenario"]:
                    ox_value_series.loc[implement_year:] = oxidation_override[
                        "scenario"
                    ]

            flaring = {}
            for s in ["baseline", "scenario"]:
                if new_landfill_flaring[s][i] is None:
                    flaring[s] = 1
                elif new_landfill_flaring[s][i] == 0:
                    flaring[s] = 1
                elif new_landfill_flaring[s][i] > 0:
                    flaring[s] = new_landfill_flaring[s][i]
            flaring_series = pd.Series(index=years)
            flaring_series.loc[years < implement_year] = flaring["baseline"]
            flaring_series.loc[years >= implement_year] = flaring["scenario"]

            new_landfill = Landfill(
                open_date=new_landfill_open_close_dates["scenario"][i][0],
                close_date=new_landfill_open_close_dates["scenario"][i][1],
                site_type=landfill_types[lf_type],
                mcf=mcf_series,
                city_params_dict=city_params_dict,
                city_instance_attrs=scenario_parameters.city_instance_attrs,
                landfill_index=i,
                # fraction_of_waste=new_landfill_fracs[i],
                gas_capture=False if new_gas_efficiency["scenario"][i] == 0.0 else True,
                scenario=scenario,
                new_baseline=False,
                gas_capture_efficiency=gas_eff_series,
                flaring=flaring_series,
                # leachate_circulate=leachate_circulate['scenario'][i],
                fraction_of_waste_vector=fraction_df[f"Landfill_{i}"],
                advanced=True,
                latlon=new_landfill_latlons["scenario"][i] if doing_fancy_ox else None,
                areas=new_landfill_areas["scenario"][i] if doing_fancy_ox else None,
                cover_types=new_covertypes["scenario"][i] if doing_fancy_ox else None,
                cover_thicknesses=(
                    new_coverthicknesses["scenario"][i] if doing_fancy_ox else None
                ),
                oxidation_factor=ox_value_series if not doing_fancy_ox else None,
                fancy_ox=fancy_ox,
                implementation_year=implement_year,
                ks=landfill_ks,
            )
            scenario_parameters.landfills.append(new_landfill)

        # Recalculate div_component_fractions
        # waste_fractions = scenario_parameters.waste_fractions

        # def calculate_component_fractions(waste_fractions: WasteFractions, div_type: str) -> WasteFractions:
        #     components = self.div_components[div_type]
        #     filtered_fractions = {waste: getattr(waste_fractions, waste) for waste in components}
        #     total = sum(filtered_fractions.values())
        #     normalized_fractions = {waste: fraction / total for waste, fraction in filtered_fractions.items()}
        #     return WasteFractions(**{waste: normalized_fractions.get(waste, 0) for waste in waste_fractions.model_dump().keys()})

        # scenario_parameters.div_component_fractions = {}
        # scenario_parameters.div_component_fractions['baseline'] = DivComponentFractions(
        #     compost=calculate_component_fractions(waste_fractions['baseline'], 'compost'),
        #     anaerobic=calculate_component_fractions(waste_fractions['baseline'], 'anaerobic'),
        #     combustion=calculate_component_fractions(waste_fractions['baseline'], 'combustion'),
        #     recycling=calculate_component_fractions(waste_fractions['baseline'], 'recycling'),
        # )
        # scenario_parameters.div_component_fractions['scenario'] = DivComponentFractions(
        #     compost=calculate_component_fractions(waste_fractions['scenario'], 'compost'),
        #     anaerobic=calculate_component_fractions(waste_fractions['scenario'], 'anaerobic'),
        #     combustion=calculate_component_fractions(waste_fractions['scenario'], 'combustion'),
        #     recycling=calculate_component_fractions(waste_fractions['scenario'], 'recycling'),
        # )

        def calculate_component_fractions(
            waste_fractions: dict, div_type: str, implement_year: int, years
        ) -> pd.DataFrame:
            # Extract the waste types from WasteFractions objects
            baseline_df = pd.DataFrame(
                waste_fractions["baseline"].model_dump(), index=[0]
            )
            scenario_df = pd.DataFrame(
                waste_fractions["scenario"].model_dump(), index=[0]
            )

            # Components are the subset of columns we're interested in
            components = list(self.div_components[div_type])

            # Filter only the relevant columns (components) for both baseline and scenario
            baseline_df = baseline_df[components]
            scenario_df = scenario_df[components]

            # Normalize each row (for baseline and scenario)
            baseline_normalized = baseline_df.div(
                baseline_df.sum(axis=1), axis=0
            ).fillna(0)
            scenario_normalized = scenario_df.div(
                scenario_df.sum(axis=1), axis=0
            ).fillna(0)

            # Create a mask for the years before and after the implement year
            mask = years < implement_year

            # Create a DataFrame with years as the index and assign baseline or scenario fractions
            result_df = pd.DataFrame(index=years, columns=components)

            # Assign baseline values for years before implement_year, scenario values after
            result_df.loc[mask, :] = np.tile(
                baseline_normalized.reindex(columns=result_df.columns).iloc[0].values,
                (mask.sum(), 1),
            )
            result_df.loc[~mask, :] = np.tile(
                scenario_normalized.reindex(columns=result_df.columns).iloc[0].values,
                ((~mask).sum(), 1),
            )

            return result_df

        # Call the function for each diversion type
        scenario_parameters.div_component_fractions = DivComponentFractionsDF(
            compost=calculate_component_fractions(
                scenario_parameters.waste_fractions, "compost", implement_year, years
            ),
            anaerobic=calculate_component_fractions(
                scenario_parameters.waste_fractions, "anaerobic", implement_year, years
            ),
            combustion=calculate_component_fractions(
                scenario_parameters.waste_fractions, "combustion", implement_year, years
            ),
            recycling=calculate_component_fractions(
                scenario_parameters.waste_fractions, "recycling", implement_year, years
            ),
        )
        scenario_parameters.non_compostable_not_targeted_total = sum(
            [
                self.non_compostable_not_targeted[x]
                * getattr(scenario_parameters.div_component_fractions.compost, x)
                for x in self.div_components["compost"]
            ]
        )
        scenario_parameters.non_compostable_not_targeted_total = pd.Series(
            scenario_parameters.non_compostable_not_targeted_total, index=years
        )
        if scenario_parameters.non_compostable_not_targeted_total.isna().all():
            scenario_parameters.non_compostable_not_targeted_total = pd.Series(
                0, index=years
            )
        self._calculate_diverted_masses(scenario=scenario)

        # Split the years for baseline and scenario
        baseline_years = years[years < implement_year]
        scenario_years = years[years >= implement_year]

        scenario_parameters.waste_fractions = pd.concat(
            [
                pd.DataFrame(
                    new_waste_fractions["baseline"].model_dump(), index=baseline_years
                ),
                pd.DataFrame(
                    new_waste_fractions["scenario"].model_dump(), index=scenario_years
                ),
            ]
        )

        # scenario_parameters.repopulate_attr_dicts()
        self._check_masses_v2(
            scenario=scenario,
            advanced_baseline=False,
            advanced_dst=True,
            implement_year=implement_year,
        )

        if scenario_parameters.input_problems:
            print(f"Invalid new value")
            return

        self._calculate_net_masses(scenario=scenario, advanced_dst=True)
        if (scenario_parameters.net_masses < 0).any().any():
            print(f"Invalid new value")
            return

        # This isn't set up yet for year of data pop scenario and implement year being different
        scenario_parameters.divs_df = DivsDF.implement_advanced(
            divs=scenario_parameters.divs,
            year_of_data_pop=scenario_parameters.year_of_data_pop["baseline"],
            growth_rate_historic=scenario_parameters.growth_rate_historic,
            growth_rate_future=scenario_parameters.growth_rate_future,
            implement_year=implement_year,
        )

        # combine these two loops maybe...though it still does six things, maybe doesn't matter
        scenario_parameters.repopulate_attr_dicts()
        for i, landfill in enumerate(scenario_parameters.landfills):
            # Might be able to do this more efficienctly...i'm looping over the pre implementation years twice sort of
            landfill.waste_mass_df = LandfillWasteMassDF.create_advanced(
                waste_generated_df=scenario_parameters.waste_generated_df,
                divs_df=scenario_parameters.divs_df,
                fraction_of_waste_series=landfill.fraction_of_waste_vector,
            ).df
            # print(landfill.waste_mass_df.loc[2028:2032, :])

            # landfill.waste_mass_df.loc[:(implement_year-1), :] = self.baseline_parameters.landfills[i].waste_mass_df.loc[:(implement_year-1), :]
            # landfill.waste_mass_df.to_csv('/Users/hugh/Library/CloudStorage/OneDrive-RMI/Documents/RMI/scratch_paper/new' + str(i) + '.csv')

        # scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in scenario_parameters.landfills:
            landfill.estimate_emissions(skip_ox=True)

        self.estimate_diversion_emissions(scenario=scenario)
        self.sum_landfill_emissions(scenario=scenario)

        # ADD WASTE BURNING EMISSIONS
        if (waste_burning["baseline"] > 0) or (waste_burning["scenario"] > 0):
            scenario_parameters.waste_burning_emissions = (
                waste_burned * 3.7 * 1000 / 1000 / 1000
            )  # g ch4 / kg waste to ton ch4 / ton waste
            scenario_parameters.waste_burning_emissions = (
                scenario_parameters.waste_burning_emissions.reindex(
                    scenario_parameters.total_emissions.index, fill_value=0
                )
            )
            scenario_parameters.total_emissions[
                "total"
            ] += scenario_parameters.waste_burning_emissions

        # if waste_burning['scenario'] > 0:
        #     scenario_parameters.waste_burning_emissions = waste_burned['scenario'] * 3.7 * 1000 / 1000 / 1000 # g ch4 / kg waste to ton ch4 / ton waste
        #     scenario_parameters.waste_burning_emissions = scenario_parameters.waste_burning_emissions.reindex(
        #         scenario_parameters.total_emissions.index, fill_value=0
        #     )
        #     scenario_parameters.total_emissions['total'] += scenario_parameters.waste_burning_emissions

    def sdst_v1_5(
        self,
        precipitation: float,
        new_waste_mass: Dict,
        new_waste_fractions: Dict,
        new_landfill_types: Dict,
        new_landfill_open_close_dates: Dict,
        implement_year: float,
        scenario: int,
        landfill_split_timeline: Dict,
        new_gas_efficiency: Dict,  # 0 means no gas capture, blank means figure out the efficiency for me
        new_landfill_fracs: Dict = None,
        new_landfill_flaring: Dict = None,
        new_landfill_cover: Dict = None,
        leachate_circulate: Dict = None,
        new_landfill_latlons: Dict = None,
        new_landfill_areas: Dict = None,
        new_covertypes: Dict = None,
        new_coverthicknesses: Dict = None,
        waste_burning: Dict = None,
        fancy_ox: Dict = {"baseline": False, "scenario": False},
        depths: Dict = None,
        k_values: Dict = None,
        waste_mass_year: int = None,
        ks_overrides: Dict = None,
        biocover: Dict = {"baseline": 0.0, "scenario": 0.0},
        oxidation_override: Dict = None,
        baseline_data: pd.DataFrame = None,
        growth_rate_override: float = None,
        country_growth_defaults: List[float] = None,
    ) -> None:
        """
        API endpoint function for implementing advanced diversion scenario changes.
        Args:
            Lots
        Returns:
            None
        """

        scenario_parameters = copy.deepcopy(self.baseline_parameters)
        baseline_parameters = copy.deepcopy(self.baseline_parameters)
        model_year_min = 1950
        model_year_max = 2050

        def _variant_value(value, label):
            try:
                return value[label]
            except Exception:
                return getattr(value, label, None)

        baseline_open_close_dates = _variant_value(
            new_landfill_open_close_dates, "baseline"
        )
        scenario_open_close_dates = _variant_value(
            new_landfill_open_close_dates, "scenario"
        ) or baseline_open_close_dates
        all_open_close_dates = list(baseline_open_close_dates or []) + list(
            scenario_open_close_dates or []
        )
        if not all_open_close_dates:
            raise CustomError("INVALID_PARAMETERS", "Landfill open/close dates are required.")

        open_years = [int(pair[0]) for pair in all_open_close_dates]
        close_years = [int(pair[1]) for pair in all_open_close_dates]
        if min(open_years) < model_year_min:
            raise CustomError(
                "INVALID_PARAMETERS",
                f"Landfill open year must be {model_year_min} or later.",
            )
        if min(close_years) < model_year_min or max(open_years + close_years) > model_year_max:
            raise CustomError(
                "INVALID_PARAMETERS",
                f"Landfill years must be between {model_year_min} and {model_year_max}.",
            )
        for open_year, close_year in zip(open_years, close_years):
            if close_year < open_year:
                raise CustomError(
                    "INVALID_PARAMETERS",
                    "Landfill close year must be after open year.",
                )
        if implement_year is None:
            implement_year = min(open_years)
        implement_year = int(implement_year)
        if implement_year < min(open_years) or implement_year > model_year_max:
            raise CustomError(
                "INVALID_PARAMETERS",
                f"Implementation year must be between {min(open_years)} and {model_year_max}.",
            )
        scenario_parameters.implement_year = implement_year

        model_start_year = min(open_years)
        years = pd.Index(range(model_start_year, model_year_max + 1))

        def _align_year_df(df: pd.DataFrame) -> pd.DataFrame:
            aligned = df.copy()
            aligned.index = pd.Index(aligned.index).astype(int)
            aligned = aligned.sort_index()
            return aligned.reindex(years).ffill().bfill()

        new_waste_fractions = {
            "baseline": _align_year_df(new_waste_fractions["baseline"]),
            "scenario": _align_year_df(new_waste_fractions["scenario"]),
        }

        waste_year_baseline = int(_variant_value(waste_mass_year, "baseline"))
        waste_year_scenario = _variant_value(waste_mass_year, "scenario")
        waste_year_scenario = (
            waste_year_baseline
            if waste_year_scenario is None
            else int(waste_year_scenario)
        )
        if (
            waste_year_baseline < model_year_min
            or waste_year_baseline > model_year_max
            or waste_year_scenario < model_year_min
            or waste_year_scenario > model_year_max
        ):
            raise CustomError(
                "INVALID_PARAMETERS",
                f"Waste mass year must be between {model_year_min} and {model_year_max}.",
            )

        def _generated_waste_masses() -> tuple[pd.DataFrame, pd.DataFrame]:
            baseline_mass = float(_variant_value(new_waste_mass, "baseline"))
            scenario_mass = _variant_value(new_waste_mass, "scenario")
            scenario_mass = baseline_mass if scenario_mass is None else float(scenario_mass)

            waste_mass_series_baseline = pd.Series(baseline_mass, index=years)
            waste_mass_series_scenario = waste_mass_series_baseline.copy()
            waste_mass_series_scenario.loc[implement_year:] = scenario_mass
            waste_masses_df_baseline_unadjusted = new_waste_fractions[
                "baseline"
            ].mul(waste_mass_series_baseline, axis=0)
            waste_masses_df_scenario_unadjusted = new_waste_fractions[
                "scenario"
            ].mul(waste_mass_series_scenario, axis=0)

            waste_masses_df_baseline = WasteGeneratedDF.create_advanced_2(
                waste_masses_df=waste_masses_df_baseline_unadjusted,
                start_year=model_start_year,
                end_year=model_year_max,
                year_of_data_pop_baseline=waste_year_baseline,
                year_of_data_pop_scenario=waste_year_scenario,
                growth_rate_historic=1 + growth_rate_override,
                growth_rate_future=1 + growth_rate_override,
                implement_year=None,
            ).df
            waste_masses_df_scenario = WasteGeneratedDF.create_advanced_2(
                waste_masses_df=waste_masses_df_scenario_unadjusted,
                start_year=model_start_year,
                end_year=model_year_max,
                year_of_data_pop_baseline=waste_year_baseline,
                year_of_data_pop_scenario=waste_year_scenario,
                growth_rate_historic=1 + growth_rate_override,
                growth_rate_future=1 + growth_rate_override,
                implement_year=implement_year,
            ).df
            return waste_masses_df_baseline, waste_masses_df_scenario

        generated_waste_masses_baseline, generated_waste_masses_scenario = (
            _generated_waste_masses()
        )

        def _fill_leading_trace_years(
            trace_waste_mass_df: pd.DataFrame,
            ratio: float,
            generated_waste_masses: pd.DataFrame,
        ) -> pd.DataFrame:
            scaled = trace_waste_mass_df.copy().reindex(years) * ratio
            valid_trace_rows = trace_waste_mass_df.dropna(how="all")
            if valid_trace_rows.empty:
                return generated_waste_masses.copy()

            first_trace_year = int(valid_trace_rows.index.min())
            leading_years = scaled.index < first_trace_year
            scaled.loc[leading_years] = scaled.loc[leading_years].combine_first(
                generated_waste_masses.loc[leading_years]
            )
            return scaled

        def _apply_open_close_window(
            waste_masses_df: pd.DataFrame, variant: str
        ) -> pd.DataFrame:
            date_pairs = _variant_value(new_landfill_open_close_dates, variant)
            if date_pairs is None and variant == "scenario":
                date_pairs = baseline_open_close_dates
            open_year, close_year = date_pairs[0]
            windowed = waste_masses_df.copy()
            windowed.loc[int(close_year) :] = 0
            windowed.loc[: int(open_year) - 1] = 0
            return windowed

        annoying_missing_incoming = False
        if baseline_data is not None:
            trace_waste_mass_df = baseline_data.get("incoming_waste_df")
            if trace_waste_mass_df is None:
                annoying_missing_incoming = True
                trace_waste_mass_df = pd.DataFrame(index=years, columns=["incoming_waste"])
                trace_waste_mass_df.iloc[:, :] = 10000.0
            else:
                trace_waste_mass_df = trace_waste_mass_df.copy()
                trace_waste_mass_df.index = pd.Index(trace_waste_mass_df.index).astype(int)
                trace_waste_mass_df = trace_waste_mass_df.sort_index()

            # Check if this is a no-FOD site. If it is, incoming_waste_df is total only, not broken out by component, so we have to use default waste fractions
            # to get a full df by waste type
            # Actually some no-FOD sites now have full breakdowns by waste type
            fod_site = bool(baseline_data.get("FOD", False))
            if not fod_site:
                if trace_waste_mass_df.shape[1] > 2:
                    pass
                else:
                    trace_waste_mass_df = new_waste_fractions['baseline'].mul(trace_waste_mass_df["incoming_waste"], axis=0)
                trace_values_around_baseline_year = trace_waste_mass_df.loc[waste_year_baseline-5:waste_year_baseline+5].mean().sum()
                trace_values_around_scenario_year = trace_waste_mass_df.loc[waste_year_scenario-5:waste_year_scenario+5].mean().sum()
                try:
                    if pd.isna(trace_values_around_baseline_year) or trace_values_around_baseline_year == 0:
                        baseline_ratio = 1
                    else:
                        baseline_ratio = new_waste_mass['baseline'] / trace_values_around_baseline_year
                    if pd.isna(trace_values_around_scenario_year) or trace_values_around_scenario_year == 0:
                        trace_waste_mass_df_unadjusted = new_waste_fractions['scenario'].mul(new_waste_mass['scenario'], axis=0)
                        trace_waste_mass_df_unadjusted.loc[:implement_year-1, :] = 0
                        trace_waste_mass_df = WasteGeneratedDF.create_advanced(
                            waste_masses_df=trace_waste_mass_df_unadjusted,
                            start_year=model_start_year,
                            end_year=model_year_max,
                            year_of_data_pop=waste_year_scenario,
                            growth_rate_historic=1+growth_rate_override,
                            growth_rate_future=1+growth_rate_override,
                            implement_year=implement_year,
                        ).df
                        scenario_ratio = 1
                    else:
                        scenario_ratio = new_waste_mass['scenario'] / trace_values_around_scenario_year
                except:
                    print('Error in waste mass ratio fudge')
                    baseline_ratio = 1
                    scenario_ratio = 1
                waste_masses_df_baseline = _fill_leading_trace_years(
                    trace_waste_mass_df,
                    baseline_ratio,
                    generated_waste_masses_baseline,
                )
                waste_masses_df_scenario = _fill_leading_trace_years(
                    trace_waste_mass_df,
                    scenario_ratio,
                    generated_waste_masses_scenario,
                )
                waste_masses_df_scenario.loc[:implement_year-1, :] = waste_masses_df_baseline.loc[:implement_year-1, :]
                waste_masses_df_baseline = _apply_open_close_window(
                    waste_masses_df_baseline, "baseline"
                )
                waste_masses_df_scenario = _apply_open_close_window(
                    waste_masses_df_scenario, "scenario"
                )
            else:
                trace_waste_mass_for_ratio = trace_waste_mass_df.reindex(years)
                trace_value_baseline_year = trace_waste_mass_for_ratio.loc[waste_year_baseline, :].sum()
                trace_value_scenario_year = trace_waste_mass_for_ratio.loc[waste_year_scenario, :].sum()
                try:
                    baseline_ratio = (
                        1
                        if pd.isna(trace_value_baseline_year) or trace_value_baseline_year == 0
                        else new_waste_mass['baseline'] / trace_value_baseline_year
                    )
                    scenario_ratio = (
                        1
                        if pd.isna(trace_value_scenario_year) or trace_value_scenario_year == 0
                        else new_waste_mass['scenario'] / trace_value_scenario_year
                    )
                except:
                    baseline_ratio = 1
                    scenario_ratio = 1
                waste_masses_df_baseline = _fill_leading_trace_years(
                    trace_waste_mass_df,
                    baseline_ratio,
                    generated_waste_masses_baseline,
                )
                waste_masses_df_scenario = _fill_leading_trace_years(
                    trace_waste_mass_df,
                    scenario_ratio,
                    generated_waste_masses_scenario,
                )
                waste_masses_df_scenario.loc[:implement_year-1, :] = waste_masses_df_baseline.loc[:implement_year-1, :]
                waste_masses_df_baseline = _apply_open_close_window(
                    waste_masses_df_baseline, "baseline"
                )
                waste_masses_df_scenario = _apply_open_close_window(
                    waste_masses_df_scenario, "scenario"
                )
        else:
            waste_masses_df_baseline = _apply_open_close_window(
                generated_waste_masses_baseline, "baseline"
            )
            waste_masses_df_scenario = _apply_open_close_window(
                generated_waste_masses_scenario, "scenario"
            )
        # Adjust for waste burning
        waste_burned = {}
        wb = None
        if waste_burning["baseline"] > 0:
            waste_burned["baseline"] = waste_burning["baseline"] * waste_masses_df_baseline
            waste_masses_df_baseline -= waste_burned["baseline"]
            wb_baseline = waste_burned["baseline"]

        if waste_burning["scenario"] > 0:
            waste_burned["scenario"] = pd.concat([
                waste_burning["baseline"] * waste_masses_df_scenario.loc[:implement_year-1, :],
                waste_burning["scenario"] * waste_masses_df_scenario.loc[implement_year:, :]
            ]) 
            waste_masses_df_scenario -= waste_burned["scenario"]

            wb_scenario = waste_burned["scenario"]
        
        # Make waste fractions dfs
        waste_fractions_df_baseline = new_waste_fractions["baseline"]
        waste_fractions_df_scenario = new_waste_fractions["scenario"]

        baseline_parameters.waste_fractions = waste_fractions_df_baseline
        scenario_parameters.waste_fractions = waste_fractions_df_scenario

        # Create a DataFrame for fraction_waste_timeline
        fraction_df = pd.DataFrame(landfill_split_timeline).transpose()
        fraction_df.columns = [f"Landfill_{i}" for i in range(fraction_df.shape[1])]
        fraction_df.index.name = "Year"

        # Set up new landfills
        city_params_dict = self.update_cityparams_dict(scenario_parameters)
        # mcfs = [1, 0.7, 0.4] # Should this include ameliorated?
        # mcf_ameliorated = [0.7, 0.4, 0.1]
        mcf_options = [1, 0.6, 0.4]
        gas_capture_efficiencies = {}
        gas_capture_efficiencies["ameliorated"] = [0.5, 0.3, 0]
        gas_capture_efficiencies["not_ameliorated"] = [0.6, 0.45, 0]
        self.ox_options = {
            "ox_nocap": {"landfill": 0.1, "controlled_dumpsite": 0.05, "dumpsite": 0.0},
            "ox_cap": {"landfill": 0.22, "controlled_dumpsite": 0.1, "dumpsite": 0.0},
        }
        landfill_types = ["landfill", "controlled_dumpsite", "dumpsite"]
        scenario_parameters.landfills = []

        # trace_mcf = baseline_data.get("mcf")
        # trace_gccs_efficiency = baseline_data.get("gas_capture_efficiency")
        # trace_oxidation_factor = baseline_data.get("oxidation_factor")

        new_lf_type = new_landfill_types["scenario"][0]
        # Make the MCF, oxidation, and efficiency vectors
        mcf = {}
        ox_value = {}
        gas_eff = {}

        # Get MCF
        old_lf_type = new_landfill_types["baseline"][0]
        mcf["baseline"] = mcf_options[old_lf_type]
        mcf["scenario"] = mcf_options[new_lf_type]

        if (depths["baseline"][0] > 5) and (old_lf_type in (1, 2)):
            mcf["baseline"] = 0.8

        if (depths["scenario"][0] > 5) and (new_lf_type in (1, 2)):
            mcf["scenario"] = 0.8

        # Handle baseline first
        if new_gas_efficiency["baseline"][0] == 0.0:
            ox_value["baseline"] = self.ox_options["ox_nocap"][
                landfill_types[old_lf_type]
            ]
            gas_eff["baseline"] = 0
        # If there is gas capture, use the number or figure it out
        elif new_gas_efficiency["baseline"][0] > 0.0:
            ox_value["baseline"] = self.ox_options["ox_cap"][
                landfill_types[old_lf_type]
            ]
            gas_eff["baseline"] = (
                new_gas_efficiency["baseline"][0]
                if new_gas_efficiency["baseline"][0] is not None
                else gas_capture_efficiencies["not_ameliorated"][old_lf_type]
            )
        else:
            print("invalid gas efficiency value")

        # For scenario, handle no gas capture first
        if new_gas_efficiency["scenario"][0] == 0:
            ox_value["scenario"] = self.ox_options["ox_nocap"][
                landfill_types[new_lf_type]
            ]
            gas_eff["scenario"] = 0
        # If there is gas capture, use the number or figure it out
        elif new_gas_efficiency["scenario"][0] > 0.0:
            if (
                new_landfill_types["scenario"][0]
                < new_landfill_types["baseline"][0]
            ):
                ameliorated = True
                if new_lf_type == 0:
                    ox_value["scenario"] = 0.18
                else:
                    ox_value["scenario"] = self.ox_options["ox_cap"][
                        landfill_types[new_lf_type]
                    ]
                gas_eff["scenario"] = (
                    new_gas_efficiency["baseline"][0]
                    if new_gas_efficiency["baseline"][0] is not None
                    else gas_capture_efficiencies["ameliorated"][new_lf_type]
                )
            else:
                ameliorated = False
                ox_value["scenario"] = self.ox_options["ox_cap"][
                    landfill_types[new_lf_type]
                ]
                gas_eff["scenario"] = (
                    new_gas_efficiency["baseline"][0]
                    if new_gas_efficiency["baseline"][0] is not None
                    else gas_capture_efficiencies["not_ameliorated"][new_lf_type]
                )
        else:
            print("invalid gas efficiency value")

        if new_gas_efficiency["baseline"][0] is not None:
            gas_eff["baseline"] = new_gas_efficiency["baseline"][0]
        if new_gas_efficiency["scenario"][0] is not None:
            gas_eff["scenario"] = new_gas_efficiency["scenario"][0]

        # Create pandas Series for each: mcf, ox_value, and gas_eff
        mcf_series_baseline = pd.Series(mcf["baseline"], index=years)
        mcf_series_scenario = mcf_series_baseline.copy()
        mcf_series_scenario.loc[implement_year:] = mcf["scenario"]
        # Oxidation factors can be fractional (e.g., biocover oxidation 0.5).
        # Build baseline first, apply baseline-only adjustments, then copy into
        # scenario so baseline/scenario match for all years < implement_year.
        ox_value_series_baseline = pd.Series(
            ox_value["baseline"], index=years, dtype=float
        )
        gas_eff_series_baseline = pd.Series(gas_eff["baseline"], index=years)
        gas_eff_series_scenario = gas_eff_series_baseline.copy()
        gas_eff_series_scenario.loc[implement_year:] = gas_eff["scenario"]

        doing_fancy_ox = fancy_ox

        if ks_overrides is not None:
            ks_series_baseline = pd.Series([ks_overrides["baseline"]] * len(years), index=years)
            ks_series_scenario = ks_series_baseline.copy()
            ks_series_scenario.loc[implement_year:] = ks_overrides["scenario"]
            landfill_ks_baseline = DecompositionRates(
                food=ks_series_baseline,
                green=ks_series_baseline,
                wood=ks_series_baseline,
                paper_cardboard=ks_series_baseline,
                textiles=ks_series_baseline,
            )
            landfill_ks_scenario = DecompositionRates(
                food=ks_series_scenario,
                green=ks_series_scenario,
                wood=ks_series_scenario,
                paper_cardboard=ks_series_scenario,
                textiles=ks_series_scenario,
            )
        else:
            landfill_ks_baseline = scenario_parameters.ks
            landfill_ks_scenario = landfill_ks_baseline.copy()

        if oxidation_override:
            if oxidation_override["baseline"]:
                ox_value_series_baseline.loc[:] = float(oxidation_override["baseline"])

        # Check if flaring is defined as a variable
        try:
            flaring_series_baseline = pd.Series(flaring["baseline"], index=years)
            flaring_series_scenario = flaring_series_baseline.copy()
            flaring_series_scenario.loc[implement_year:] = flaring["scenario"]
        except:
            flaring_series_baseline = pd.Series(0.98, index=years)
            flaring_series_scenario = flaring_series_baseline.copy()

        if biocover["baseline"] > 0:
            baseline_biocover = float(biocover["baseline"])
            ox_value_series_baseline.loc[
                ox_value_series_baseline < baseline_biocover
            ] = baseline_biocover

        # Scenario starts as baseline, then applies scenario changes from implement_year onward.
        ox_value_series_scenario = ox_value_series_baseline.copy()
        ox_value_series_scenario.loc[implement_year:] = float(ox_value["scenario"])

        if oxidation_override and oxidation_override["scenario"]:
            ox_value_series_scenario.loc[implement_year:] = float(
                oxidation_override["scenario"]
            )

        if biocover["scenario"] > 0:
            scenario_biocover = float(biocover["scenario"])
            mask_after_implement = ox_value_series_scenario.index >= implement_year
            ox_value_series_scenario.loc[
                mask_after_implement
                & (ox_value_series_scenario < scenario_biocover)
            ] = scenario_biocover

        new_landfill_baseline = Landfill(
            open_date=new_landfill_open_close_dates["baseline"][0][0],
            close_date=new_landfill_open_close_dates["baseline"][0][1],
            site_type=landfill_types[old_lf_type],
            mcf=mcf_series_baseline,
            city_params_dict=city_params_dict,
            city_instance_attrs=scenario_parameters.city_instance_attrs,
            landfill_index=0,
            # fraction_of_waste=new_landfill_fracs[i],
            gas_capture=False if new_gas_efficiency["baseline"][0] == 0.0 else True,
            scenario=scenario,
            new_baseline=False,
            gas_capture_efficiency=gas_eff_series_baseline,
            flaring=flaring_series_baseline,
            # leachate_circulate=leachate_circulate['scenario'][i],
            cover_thicknesses=(
                new_coverthicknesses["baseline"][0] if doing_fancy_ox else None
            ),
            oxidation_factor=ox_value_series_baseline if not doing_fancy_ox else None,
            fancy_ox=fancy_ox,
            implementation_year=implement_year,
            ks=landfill_ks_baseline,
            advanced=True,
        )
        baseline_parameters.landfills = [new_landfill_baseline]

        new_landfill_scenario = Landfill(
            open_date=new_landfill_open_close_dates["scenario"][0][0],
            close_date=new_landfill_open_close_dates["scenario"][0][1],
            site_type=landfill_types[new_lf_type],
            mcf=mcf_series_scenario,
            city_params_dict=city_params_dict,
            city_instance_attrs=scenario_parameters.city_instance_attrs,
            landfill_index=0,
            gas_capture=False if new_gas_efficiency["scenario"][0] == 0.0 else True,
            scenario=scenario,
            new_baseline=False,
            gas_capture_efficiency=gas_eff_series_scenario,
            flaring=flaring_series_scenario,
            # leachate_circulate=leachate_circulate['scenario'][i],
            cover_thicknesses=(
                new_coverthicknesses["scenario"][0] if doing_fancy_ox else None
            ),
            oxidation_factor=ox_value_series_scenario if not doing_fancy_ox else None,
            fancy_ox=fancy_ox,
            implementation_year=implement_year,
            ks=landfill_ks_scenario,
            advanced=True,
        )
        scenario_parameters.landfills = [new_landfill_scenario]

        baseline_parameters.repopulate_attr_dicts()
        scenario_parameters.repopulate_attr_dicts()
        for landfill in baseline_parameters.landfills:
            landfill.waste_mass_df = waste_masses_df_baseline
            landfill.oxidation_factor = ox_value_series_baseline
        for landfill in scenario_parameters.landfills:
            landfill.waste_mass_df = waste_masses_df_scenario
            landfill.oxidation_factor = ox_value_series_scenario

        for landfill in baseline_parameters.landfills:
            landfill.estimate_emissions(skip_ox=True)
        for landfill in scenario_parameters.landfills:
            landfill.estimate_emissions(skip_ox=True)

        components = ["food", "green", "wood", "paper_cardboard", "textiles"]
        baseline_parameters.organic_emissions = pd.DataFrame(0, index=years, columns=components)
        scenario_parameters.organic_emissions = pd.DataFrame(0, index=years, columns=components)
        self.baseline_parameters = baseline_parameters
        self.scenario_parameters[0] = scenario_parameters
        self.sum_landfill_emissions(scenario=0)
        self.sum_landfill_emissions(scenario=1)

        # # Adjust estimates to match previously-generated baseline in 2025 if no-FOD site
        # if baseline_data:
        #     fod_site = bool(baseline_data.get("FOD", False))
        #     if not fod_site:
        #         current_year = datetime.now().year
        #         old_emissions_baseline = baseline_data.get("emissions_df")
        #         old_emissions_baseline = old_emissions_baseline.resample("YS").sum()
        #         old_emissions_baseline.index = old_emissions_baseline.index.year
        #         #old_current_year_value = old_emissions_baseline.loc[current_year]
        #         #new_current_year_value = baseline_parameters.total_emissions.loc[current_year, 'total']
        #         #noFOD_scale_factor = new_current_year_value / old_current_year_value
        #         window = slice(current_year - 5, current_year + 5)
        #         num = old_emissions_baseline.loc[window, "total"]  # Series
        #         den = baseline_parameters.total_emissions.loc[window, "total"]  # Series
        #         valid = den != 0
        #         if valid.any():
        #             ratio = num[valid] / den[valid]
        #             noFOD_scale_factor = ratio.mean()
        #         else:
        #             noFOD_scale_factor = 1
        #         baseline_parameters.total_emissions = baseline_parameters.total_emissions.mul(noFOD_scale_factor, axis=0)
        #         scenario_parameters.total_emissions = scenario_parameters.total_emissions.mul(noFOD_scale_factor, axis=0)

        # ADD WASTE BURNING EMISSIONS
        if (waste_burning["baseline"] > 0) or (waste_burning["scenario"] > 0):
            baseline_parameters.waste_burning_emissions = (
                waste_burned * 3.7 * 1000 / 1000 / 1000
            )  # g ch4 / kg waste to ton ch4 / ton waste
            scenario_parameters.waste_burning_emissions = (
                waste_burned * 3.7 * 1000 / 1000 / 1000
            )  # g ch4 / kg waste to ton ch4 / ton waste

            baseline_parameters.waste_burning_emissions = (
                baseline_parameters.waste_burning_emissions.reindex(
                baseline_parameters.total_emissions.index, fill_value=0
            ))
            scenario_parameters.waste_burning_emissions = (
                scenario_parameters.waste_burning_emissions.reindex(
                scenario_parameters.total_emissions.index, fill_value=0
            ))
            baseline_parameters.total_emissions["total"] += baseline_parameters.waste_burning_emissions
            scenario_parameters.total_emissions["total"] += scenario_parameters.waste_burning_emissions

    def advanced_baseline(
        self,
        population: float,
        precipitation: float,
        new_waste_mass: None,
        new_waste_fractions: WasteFractions,
        new_div_fractions: DiversionFractions,
        new_landfill_types: List,
        new_landfill_open_close_dates: List,
        scenario: int,
        new_baseline: int,
        landfill_split_timeline: Dict,
        new_gas_efficiency: List,  # 0 means no gas capture, blank means figure out the efficiency for me
        new_landfill_fracs: List = None,
        new_landfill_flaring: List = None,
        new_landfill_cover: List = None,
        leachate_circulate: List = None,
        new_landfill_latlons: List = None,
        new_landfill_areas: List = None,
        new_covertypes: List = None,
        new_coverthicknesses: List = None,
        waste_burning: float = 0.0,
        fancy_ox: bool = False,
        new_waste_mass_per_capita: float = None,
        depth: float = None,
        ks_overrides: float = None,
        biocover: float = 0,
        oxidation_override: float = None,
    ) -> None:
        """
        API endpoint function for implementing SDST changes.
        Args:
            Lots
        Returns:
            None
        """

        scenario_parameters = copy.deepcopy(self.baseline_parameters)
        self.scenario_parameters[scenario - 1] = scenario_parameters
        scenario_parameters.div_fractions = new_div_fractions

        years = pd.Index(range(1990, 2051))
        waste_fractions_dict = new_waste_fractions.model_dump()
        new_waste_fractions = pd.DataFrame(waste_fractions_dict, index=years)
        scenario_parameters.waste_fractions = new_waste_fractions

        scenario_parameters._singapore_k(advanced_baseline=True)

        if new_waste_mass:
            pass
        elif new_waste_mass_per_capita:
            new_waste_mass = new_waste_mass_per_capita * population
        else:
            new_waste_mass = scenario_parameters.waste_mass
        scenario_parameters.waste_mass = new_waste_mass

        # Adjust for waste burning
        # waste_burned = {}
        if waste_burning > 0:
            waste_burned = pd.DataFrame(waste_burning * new_waste_mass, index=years)
            new_waste_mass -= waste_burning * new_waste_mass

            # Adjust the waste burning for growth rates to get real time series
            t = years - scenario_parameters.year_of_data_pop

            # Create growth rate array, using growth_rate_historic for years before year_of_data_pop and growth_rate_future after
            growth_rate = np.where(
                years < scenario_parameters.year_of_data_pop,
                scenario_parameters.growth_rate_historic,
                scenario_parameters.growth_rate_future,
            )
            growth_factors = growth_rate**t

            # Apply growth factors
            waste_burned = waste_burned.multiply(growth_factors, axis=0)

        # New waste masses
        waste_masses = {}
        waste_masses = pd.DataFrame(
            {
                col: new_waste_fractions.at[2000, col] * new_waste_mass
                for col in new_waste_fractions.columns
            },
            index=new_waste_fractions.index,
        )
        scenario_parameters.waste_masses = waste_masses  # WasteMasses(**waste_masses)

        # Update waste generated
        scenario_parameters.waste_generated_df = WasteGeneratedDF.create(
            waste_masses,
            1990,
            2050,
            scenario_parameters.year_of_data_pop["baseline"],
            scenario_parameters.growth_rate_historic,
            scenario_parameters.growth_rate_future,
        ).df

        # Create a DataFrame for fraction_waste_timeline
        fraction_df = pd.DataFrame(landfill_split_timeline).transpose()
        fraction_df.columns = [f"Landfill_{i}" for i in range(fraction_df.shape[1])]
        fraction_df.index.name = "Year"

        # Set up new landfills
        city_params_dict = self.update_cityparams_dict(scenario_parameters)
        # mcfs = [1, 0.7, 0.4] # Should this include ameliorated?
        # mcf_ameliorated = [0.7, 0.4, 0.1]
        mcf_options = [1, 0.6, 0.4]
        # mcfs['ameliorated'] = {}
        # mcf_options['not_ameliorated'] = {}
        # mcfs['ameliorated']['gas_capture'] = [0.18, 0, 0]
        # mcfs['ameliorated']['no_gas_capture'] = [0.1, 0, 0]
        # mcf_options['not_ameliorated']['gas_capture'] = [0.22, 0.1, 0]
        # mcf_options['not_ameliorated']['no_gas_capture'] = [0.1, 0.05, 0]
        gas_capture_efficiencies = {}
        gas_capture_efficiencies["ameliorated"] = [0.5, 0.3, 0]
        gas_capture_efficiencies["not_ameliorated"] = [0.6, 0.45, 0]
        self.ox_options = {
            "ox_nocap": {"landfill": 0.1, "controlled_dumpsite": 0.05, "dumpsite": 0.0},
            "ox_cap": {"landfill": 0.22, "controlled_dumpsite": 0.1, "dumpsite": 0.0},
        }
        landfill_types = ["landfill", "controlled_dumpsite", "dumpsite"]
        scenario_parameters.landfills = []
        for i, lf_type in enumerate(new_landfill_types):
            # Make the MCF, oxidation, and efficiency vectors
            years = pd.Index(range(1990, 2051))
            mcf = mcf_options[lf_type]
            if (depth > 5) and (lf_type in (1, 2)):
                mcf = 0.8
            # Handle no gas capture first
            if new_gas_efficiency[i] == 0:
                # mcf = mcf_options['not_ameliorated']['no_gas_capture'][lf_type]
                ox_value = self.ox_options["ox_nocap"][landfill_types[lf_type]]
                gas_eff = 0
            # If there is gas capture, use the number or figure it out
            else:
                # mcf = mcf_options['not_ameliorated']['gas_capture'][lf_type]
                ox_value = self.ox_options["ox_cap"][landfill_types[lf_type]]
                gas_eff = (
                    new_gas_efficiency[i]
                    if new_gas_efficiency[i] is not None
                    else gas_capture_efficiencies["not_ameliorated"][lf_type]
                )

            if new_gas_efficiency[i] is not None:
                gas_eff = new_gas_efficiency[i]

            oxs = [ox_value for year in years]
            mcfs = [mcf for year in years]
            gas_effs = [gas_eff for year in years]

            if biocover > 0:
                oxs = [biocover for year in years]

            # if fancy_ox:
            #     oxs = None

            if ks_overrides is not None:
                landfill_ks = DecompositionRates(
                    food=pd.Series([ks_overrides] * len(years), index=years),
                    green=pd.Series([ks_overrides] * len(years), index=years),
                    wood=pd.Series([ks_overrides] * len(years), index=years),
                    paper_cardboard=pd.Series([ks_overrides] * len(years), index=years),
                    textiles=pd.Series([ks_overrides] * len(years), index=years),
                )
            else:
                landfill_ks = scenario_parameters.ks

            if oxidation_override:
                oxs = [oxidation_override for year in years]

            if new_landfill_flaring[i] is None:
                flaring = [1 for year in years]
            elif new_landfill_flaring[i] == 0:
                flaring = [1 for year in years]
            elif new_landfill_flaring[i] > 0:
                flaring = [new_landfill_flaring[i] for year in years]

            new_landfill = Landfill(
                open_date=new_landfill_open_close_dates[i][0],
                close_date=new_landfill_open_close_dates[i][1],
                site_type=landfill_types[lf_type],
                mcf=pd.Series(mcfs, index=years),
                city_params_dict=city_params_dict,
                city_instance_attrs=scenario_parameters.city_instance_attrs,
                landfill_index=i,
                # fraction_of_waste=new_landfill_fracs[i],
                gas_capture=False if new_gas_efficiency[i] == 0 else True,
                scenario=scenario,
                new_baseline=True,
                gas_capture_efficiency=pd.Series(gas_effs, index=years),
                flaring=pd.Series(flaring, index=years),
                # leachate_circulate=leachate_circulate[i],
                fraction_of_waste_vector=fraction_df[f"Landfill_{i}"],
                advanced=True,
                latlon=new_landfill_latlons[i] if fancy_ox else None,
                areas=new_landfill_areas[i] if fancy_ox else None,
                cover_types=new_covertypes[i] if fancy_ox else None,
                cover_thicknesses=new_coverthicknesses[i] if fancy_ox else None,
                oxidation_factor=pd.Series(oxs, index=years) if not fancy_ox else None,
                fancy_ox=fancy_ox,
                ks=landfill_ks,
            )
            scenario_parameters.landfills.append(new_landfill)

        # Recalculate div_component_fractions
        waste_fractions = scenario_parameters.waste_fractions

        def calculate_component_fractions(
            waste_fractions: WasteFractions, div_type: str
        ) -> WasteFractions:
            components = list(self.div_components[div_type])
            filtered_fractions = waste_fractions.loc[2000, components]
            total = filtered_fractions.sum()
            if total != 0:
                normalized_fractions = filtered_fractions / total
            else:
                normalized_fractions = pd.Series(0.0, index=filtered_fractions.index)
            return WasteFractions(**normalized_fractions.to_dict())

        scenario_parameters.div_component_fractions = DivComponentFractions(
            compost=calculate_component_fractions(waste_fractions, "compost"),
            anaerobic=calculate_component_fractions(waste_fractions, "anaerobic"),
            combustion=calculate_component_fractions(waste_fractions, "combustion"),
            recycling=calculate_component_fractions(waste_fractions, "recycling"),
        )
        scenario_parameters.non_compostable_not_targeted_total = sum(
            [
                self.non_compostable_not_targeted[x]
                * getattr(scenario_parameters.div_component_fractions.compost, x)
                for x in self.div_components["compost"]
            ]
        )
        scenario_parameters.non_compostable_not_targeted_total = pd.Series(
            scenario_parameters.non_compostable_not_targeted_total, index=years
        )
        if scenario_parameters.non_compostable_not_targeted_total.isna().all():
            scenario_parameters.non_compostable_not_targeted_total = pd.Series(
                0, index=years
            )
        self._calculate_diverted_masses(
            scenario=scenario
        )  # This function could be moved to cityparameters class, and then it doesn't need scenario argument

        # scenario_parameters.repopulate_attr_dicts()
        self._check_masses_v2(scenario=scenario, advanced_baseline=True)

        if scenario_parameters.input_problems:
            raise ValueError("Invalid new values")

        self._calculate_net_masses(scenario=scenario, advanced_baseline=True)
        if (scenario_parameters.net_masses < 0).any().any():
            raise ValueError("Invalid new values")
            return

        scenario_parameters.divs_df = DivsDF.create_advanced_baseline(
            scenario_parameters.divs,
            scenario_parameters.year_of_data_pop["baseline"],
            scenario_parameters.growth_rate_historic,
            scenario_parameters.growth_rate_future,
        )

        # combine these two loops maybe...though it still does six things, maybe doesn't matter
        scenario_parameters.repopulate_attr_dicts()
        for i, landfill in enumerate(scenario_parameters.landfills):
            # Might be able to do this more efficienctly...i'm looping over the pre implementation years twice sort of
            landfill.waste_mass_df = LandfillWasteMassDF.create_advanced(
                waste_generated_df=scenario_parameters.waste_generated_df,
                divs_df=scenario_parameters.divs_df,
                fraction_of_waste_series=landfill.fraction_of_waste_vector,
            ).df

        # scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in scenario_parameters.landfills:
            landfill.estimate_emissions(skip_ox=True)

        self.estimate_diversion_emissions(scenario=scenario)
        self.sum_landfill_emissions(scenario=scenario)

        # ADD WASTE BURNING EMISSIONS
        if waste_burning > 0:
            scenario_parameters.waste_burning_emissions = (
                waste_burned * 3.7 * 1000 / 1000 / 1000
            )  # g ch4 / kg waste to ton ch4 / ton waste
            scenario_parameters.total_emissions[
                "total"
            ] += scenario_parameters.waste_burning_emissions

    async def sdst_prepopulate(
        self,
        DB_SERVER_IP: str,
        DB_PORT: int,
        DB_USER: str,
        DB_PASSWORD: str,
        DB_NAME: str,
        DB_SSLMODE: str,
        sites_list: pd.DataFrame,
        latlon: Optional[str] = None,
        site_id: Optional[int] = None,
    ) -> None:
        """
        API endpoing function to prepopulate the SDST,
        based on locatio or site RMI ID.

        Args:
            DB_SERVER_IP (str): Database server IP address.
            DB_PORT (int): Database server port.
            DB_USER (str): Database username.
            DB_PASSWORD (str): Database password.
            DB_NAME (str): Database name.
            DB_SSLMODE (str): SSL mode for database connection.
            sites_list (pd.DataFrame): DataFrame containing site information.
            latlon (Optional[str]): Latitude and longitude as a string. Defaults to None.
            site_id (Optional[int]): Site ID. Defaults to None.
        Returns:
            dict: Dictionary containing data for prepopulation.
        """
        parameters = CityParameters()

        # Then in sdst_prepopulate:
        geolocator = create_geolocator()
        if site_id:
            latlon = sites_list.loc[
                sites_list["RMI ID"] == site_id, ["Latitude", "Longitude"]
            ].values[0]
            country = sites_list.loc[sites_list["RMI ID"] == site_id, "Country"].values[
                0
            ]
            iso3 = sites_list.loc[
                sites_list["RMI ID"] == site_id, "Country ISO3"
            ].values[0]
            site_type = sites_list.loc[
                sites_list["RMI ID"] == site_id, "Site Type"
            ].values[0]
            region = defaults_2019.region_lookup_iso3.get(iso3)
            if region is None:
                raise ValueError(f"Region for ISO3 code '{iso3}' not found.")
        else:
            location = geolocator.reverse((latlon[0], latlon[1]), language="en")
            country = location.raw["address"].get("country")
            try:
                iso3 = pycountry.countries.search_fuzzy(country)[0].alpha_3
            except LookupError:
                raise ValueError(f"Country '{country}' not found.")
            region = defaults_2019.region_lookup_iso3.get(iso3)
            if region is None:
                raise ValueError(f"Region for ISO3 code '{iso3}' not found.")
            if region in defaults_2019.landfill_default_regions:
                site_type = "Landfill"
            else:
                site_type = "Dumpsite"

        if site_type in ["Landfill", "Sanitary Landfill"]:
            site_type = 0
            depth = 100
        elif site_type == "Controlled Dumpsite":
            site_type = 1
            depth = 100
        else:
            site_type = 2
            depth = 3

        # SQL query to get average precipitation and temperature using provided latitude and longitude
        QUERY_WEATHER = """
        WITH city_selection AS (
            SELECT
                'CustomCity' AS name,
                $1::numeric AS latitude,
                $2::numeric AS longitude
        ),
        global_weather_table AS (
            SELECT
                cs.name,
                ROUND(AVG(value) FILTER (WHERE weather_type = 'precipitation')::numeric, 2) AS avg_total_precip,
                ROUND(AVG(value) FILTER (WHERE weather_type = 'temperature')::numeric, 2) AS avg_temperature
            FROM global_weather_data, city_selection cs
            WHERE ST_Covers(
                    bbox_geometry,
                    ST_SetSRID(ST_MakePoint(cs.longitude, cs.latitude), 4326)
                )
            GROUP BY cs.name
        )
        SELECT * FROM global_weather_table;
        """

        # Best-effort reachability check (non-fatal): if DB is unreachable,
        # fall back to default weather values rather than crashing.
        try:
            socket.create_connection((DB_SERVER_IP, DB_PORT), timeout=5)
        except socket.error as e:
            print(
                f"Weather lookup failed (latlon={latlon}, db={DB_SERVER_IP}:{DB_PORT}/{DB_NAME}): cannot reach database: {e}"
            )
            rows = []
            conn = None

        conn = None
        rows = []
        try:
            # Connect asynchronously to the PostgreSQL database using asyncpg
            conn = await asyncpg.connect(
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                host=DB_SERVER_IP,
                port=DB_PORT,
                ssl=DB_SSLMODE,
            )

            # Execute the query with the latitude and longitude from latlon
            rows = await conn.fetch(QUERY_WEATHER, latlon[0], latlon[1])
        except Exception as exc:
            # Surface enough context to debug weather lookups but keep the service running
            print(
                f"Weather lookup failed (latlon={latlon}, db={DB_SERVER_IP}:{DB_PORT}/{DB_NAME}): {exc}"
            )
            rows = []
        finally:
            if conn:
                try:
                    await conn.close()
                except Exception as close_exc:
                    print(f"Failed to close weather DB connection: {close_exc}")

        # Convert the asyncpg Record objects into a list of dictionaries
        try:
            weather_data = [dict(row) for row in rows][0]
        except IndexError:
            weather_data = {
                "avg_total_precip": 1000,
                "avg_temperature": 20,
            }
            # raise HTTPException(
            #     status_code=500,
            #     detail="No weather data found for the given latitude and longitude.",
            # )

        # Waste fractions
        waste_fractions = defaults_2019.waste_composition_for(iso3, region)

        # Normalize the waste fractions so that they sum to 1.
        waste_fractions = waste_fractions / waste_fractions.sum()
        years = pd.Index(range(1990, 2051))
        waste_fractions_df = pd.DataFrame(
            np.tile(waste_fractions.values, (len(years), 1)),
            index=years,
            columns=waste_fractions.index,
        )
        parameters.precip = weather_data["avg_total_precip"]
        parameters.temperature = weather_data["avg_temperature"]

        parameters.waste_fractions = waste_fractions_df
        parameters._singapore_k(advanced_baseline=True)

        wf_out = waste_fractions_df.iloc[0].to_dict()

        growth_rate = defaults_2019.growth_rate_country[iso3] / 100

        if (not isinstance(latlon, list)) and (not isinstance(latlon, tuple)):
            latlon = latlon.tolist()

        if site_id is not None:
            waste_mass = sites_list.loc[
                    sites_list["RMI ID"] == site_id, "Waste Accepted (tons/year)"
                ].fillna(10_000).values[0]
            waste_mass_year = sites_list.loc[
                    sites_list["RMI ID"] == site_id, "Waste in Place Year"
                ].fillna(2024).values[0]
            site_open_year = sites_list.loc[
                    sites_list["RMI ID"] == site_id, "Site Open Year"
                ].fillna(2010).values[0]
            site_close_year = sites_list.loc[
                    sites_list["RMI ID"] == site_id, "Site Close Year"
                ].values[0]
        else:
            waste_mass = 10_000  # Default value if not provided
            waste_mass_year = 2025
            site_open_year = 1990
            site_close_year = 2050
        
        return {
            "iso3": iso3,
            "temperature": parameters.temperature,
            "precipitation": parameters.precip,
            "waste_fractions": wf_out,
            "degredation_constant_k": float(parameters.ks.food.iat[0]),
            "growth_rate": growth_rate,
            "latlon": latlon,
            "site_type": site_type,
            "depth": depth,
            "waste_mass": float(waste_mass),
            "waste_mass_year": int(waste_mass_year),
            'site_open_year': int(site_open_year) if pd.notna(site_open_year) else int(2010),
            'site_close_year': int(site_close_year) if pd.notna(site_close_year) else 2050
        }

def create_geolocator(
    user_agent: str = "WasteMAP/1.0 (hugh.runyan@rmi.org)",
) -> Nominatim:
    """
    Returns a Nominatim geolocator with:
    • Proper CA bundle via certifi
    • 1s rate limit on both geocode() and reverse()
    • A valid user-agent including a contact email
    """
    # build an SSL context that trusts certifi's root CAs
    ctx = ssl.create_default_context(cafile=certifi.where())

    geo = Nominatim(
        user_agent=user_agent,
        ssl_context=ctx,
        timeout=10,
    )

    # ensure we don't hammer the free API
    geo.geocode = RateLimiter(geo.geocode, min_delay_seconds=1)
    geo.reverse = RateLimiter(geo.reverse, min_delay_seconds=1)
    return geo
