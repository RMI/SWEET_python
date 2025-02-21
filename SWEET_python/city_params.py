#%%

import os
import sys

sys.path.append('/app/SWEET_python/SWEET_python')
sys.path.append('/app/SWEET_python')

from pydantic import BaseModel, validator
from typing import List, Dict, Union, Any, Set, Optional
import pandas as pd
import numpy as np
import pycountry # What am i using this for...seems dumb
from SWEET_python.class_defs import *
import inspect
import copy
from geopy.geocoders import Nominatim
import asyncpg
import socket
from fastapi import HTTPException
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
try:
    from landfill import Landfill
    import defaults_2019
except:
    from SWEET_python.landfill import Landfill
    import SWEET_python.defaults_2019 as defaults_2019
    
class CityParameters(BaseModel):
    waste_fractions: Optional[pd.DataFrame] = None #WasteFractions
    div_fractions: Optional[pd.DataFrame] = None #DiversionFractions
    split_fractions: Optional[SplitFractions] = None
    div_component_fractions: Optional[DivComponentFractionsDF] = None
    precip: Optional[float] = None
    growth_rate_historic: Optional[float] = None
    growth_rate_future: Optional[float] = None
    waste_per_capita: Optional[float] = None
    precip_zone: Optional[str] = None
    ks: Optional[DecompositionRates] = None
    gas_capture_efficiency: Optional[pd.Series] = None #float
    mef_compost: Optional[float] = None
    waste_mass: Optional[pd.Series] = None #float
    landfills: Optional[List[Landfill]] = None
    non_zero_landfills: Optional[List[Landfill]] = None
    non_compostable_not_targeted_total: Optional[pd.Series] = None
    waste_masses: Optional[WasteMasses] = None
    divs: Optional[DivMasses] = None
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
    waste_generated_df: Optional[WasteGeneratedDF] = None
    city_instance_attrs: Optional[Dict[str, Any]] = None
    population: Optional[float] = None
    temp: Optional[float] = None 
    net_masses: Optional[pd.DataFrame] = None
    temperature: Optional[float] = None
    waste_burning_emissions: Optional[pd.DataFrame] = None
    source_pop: Optional[str] = None
    source_msw: Optional[str] = None

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
                return data.to_dict(orient='records')
            else:
                return data

        return convert_sets_to_lists(data)

    def repopulate_attr_dicts(self):
        city_params_dict = self.model_dump()
        keys_to_remove = ['landfills', 'non_zero_landfills']
        for key in keys_to_remove:
            if key in city_params_dict:
                del city_params_dict[key]

        if self.landfills is not None:
            for landfill in self.landfills:
                landfill.city_params_dict = city_params_dict
                if hasattr(landfill, 'model'):
                    landfill.model.city_params_dict = city_params_dict
                    landfill.model.landfill_instance_attrs=landfill.model_dump()

    def _singapore_k(self, advanced_baseline=False, advanced_dst=False, implement_year=None) -> None:
        """
        Calculates and sets k values for the city based on the Singapore method.
        """
        # Start with kc, which accounts for waste composition
        # nb = self.waste_fractions['metal'] + self.waste_fractions['glass'] + self.waste_fractions['plastic'] + self.waste_fractions['other'] + self.waste_fractions['rubber']
        # bs = self.waste_fractions['wood'] + self.waste_fractions['paper_cardboard'] + self.waste_fractions['textiles']
        # bf = self.waste_fractions['food'] + self.waste_fractions['green']

        # Lookup array order is bs, bf, nb. Multiply by 8
        lookup_array = np.zeros((8, 8, 8))

        lookup_array[0, 0, 7] = 0.3 # lower left corner
        lookup_array[0, 0, 6] = 0.3 # this is all the bottom row
        lookup_array[1, 0, 6] = 0.3
        lookup_array[1, 0, 5] = 0.3
        lookup_array[2, 0, 6] = 0.3
        lookup_array[2, 0, 5] = 0.3
        lookup_array[2, 0, 4] = 0.3
        lookup_array[3, 0, 4] = 0.3
        lookup_array[3, 0, 3] = 0.3
        lookup_array[4, 0, 4] = 0.4
        lookup_array[4, 0, 3] = 0.5
        lookup_array[4, 0, 2] = 0.5
        lookup_array[5, 0, 2] = 0.5
        lookup_array[5, 0, 1] = 0.5
        lookup_array[6, 0, 2] = 0.3
        lookup_array[6, 0, 1] = 0.1
        lookup_array[6, 0, 0] = 0.1
        lookup_array[7, 0, 0] = 0.1 # lower right corner

        lookup_array[0, 1, 6] = 0.3 # second row from bottom
        lookup_array[0, 1, 5] = 0.3
        lookup_array[1, 1, 5] = 0.3
        lookup_array[1, 1, 4] = 0.3
        lookup_array[2, 1, 4] = 0.3
        lookup_array[2, 1, 3] = 0.3
        lookup_array[3, 1, 3] = 0.3
        lookup_array[3, 1, 2] = 0.3
        lookup_array[4, 1, 2] = 0.5
        lookup_array[4, 1, 1] = 0.1
        lookup_array[5, 1, 1] = 0.1
        lookup_array[5, 1, 0] = 0.1
        lookup_array[6, 1, 0] = 0.1

        lookup_array[0, 2, 6] = 0.3
        lookup_array[0, 2, 5] = 0.3
        lookup_array[0, 2, 4] = 0.3
        lookup_array[1, 2, 4] = 0.3
        lookup_array[1, 2, 3] = 0.3
        lookup_array[2, 2, 4] = 0.5
        lookup_array[2, 2, 3] = 0.7
        lookup_array[2, 2, 2] = 0.7
        lookup_array[3, 2, 2] = 0.7
        lookup_array[3, 2, 1] = 0.7
        lookup_array[4, 2, 2] = 0.5
        lookup_array[4, 2, 1] = 0.1
        lookup_array[4, 2, 0] = 0.1
        lookup_array[5, 2, 0] = 0.1
        lookup_array[6, 2, 0] = 0.1

        lookup_array[0, 3, 4] = 0.3
        lookup_array[0, 3, 3] = 0.3
        lookup_array[1, 3, 3] = 0.3
        lookup_array[1, 3, 2] = 0.3
        lookup_array[2, 3, 2] = 0.7
        lookup_array[2, 3, 1] = 0.7
        lookup_array[3, 3, 1] = 0.7
        lookup_array[3, 3, 0] = 0.7
        lookup_array[4, 3, 0] = 0.1

        lookup_array[0, 4, 4] = 0.3
        lookup_array[0, 4, 3] = 0.3
        lookup_array[0, 4, 2] = 0.3
        lookup_array[1, 4, 2] = 0.3
        lookup_array[1, 4, 1] = 0.5
        lookup_array[2, 4, 2] = 0.5
        lookup_array[2, 4, 1] = 0.5
        lookup_array[2, 4, 0] = 0.5
        lookup_array[3, 4, 0] = 0.5
        lookup_array[4, 4, 0] = 0.5

        lookup_array[0, 5, 2] = 0.7
        lookup_array[0, 5, 1] = 0.7
        lookup_array[1, 5, 1] = 0.7
        lookup_array[1, 5, 0] = 0.7
        lookup_array[2, 5, 0] = 0.5

        lookup_array[0, 6, 2] = 0.6
        lookup_array[0, 6, 1] = 0.5
        lookup_array[0, 6, 0] = 0.5
        lookup_array[1, 6, 0] = 0.5
        lookup_array[2, 6, 0] = 0.6

        lookup_array[0, 7, 0] = 0.5

        if advanced_dst:
            nb = {}
            bs = {}
            bf = {}

            nb['baseline'] = self.waste_fractions['baseline'].metal + self.waste_fractions['baseline'].glass + self.waste_fractions['baseline'].plastic + self.waste_fractions['baseline'].other + self.waste_fractions['baseline'].rubber
            bs['baseline']  = self.waste_fractions['baseline'].wood + self.waste_fractions['baseline'].paper_cardboard + self.waste_fractions['baseline'].textiles
            bf['baseline']  = self.waste_fractions['baseline'].food + self.waste_fractions['baseline'].green

            nb['scenario'] = self.waste_fractions['scenario'].metal + self.waste_fractions['scenario'].glass + self.waste_fractions['scenario'].plastic + self.waste_fractions['scenario'].other + self.waste_fractions['scenario'].rubber
            bs['scenario']  = self.waste_fractions['scenario'].wood + self.waste_fractions['scenario'].paper_cardboard + self.waste_fractions['scenario'].textiles
            bf['scenario']  = self.waste_fractions['scenario'].food + self.waste_fractions['scenario'].green

            bs_idx = {}
            bf_idx = {}
            nb_idx = {}

            bs_idx['baseline'] = int(bs['baseline'] * 8)
            bf_idx['baseline'] = int(bf['baseline'] * 8)
            nb_idx['baseline'] = int(nb['baseline'] * 8)

            bs_idx['scenario'] = int(bs['scenario'] * 8)
            bf_idx['scenario'] = int(bf['scenario'] * 8)
            nb_idx['scenario'] = int(nb['scenario'] * 8)

            if nb_idx['baseline'] == 8:
                nb_idx['baseline'] = 7
            if bs_idx['baseline'] == 8:
                bs_idx['baseline'] = 7
            if bf_idx['baseline'] == 8:
                bf_idx['baseline'] = 7

            if nb_idx['scenario'] == 8:
                nb_idx['scenario'] = 7
            if bs_idx['scenario'] == 8:
                bs_idx['scenario'] = 7
            if bf_idx['scenario'] == 8:
                bf_idx['scenario'] = 7

            kc = {}
            kc['baseline'] = lookup_array[bs_idx['baseline'], bf_idx['baseline'], nb_idx['baseline']]
            if kc['baseline'] == 0.0:
                print('Invalid value for k')

            kc['scenario'] = lookup_array[bs_idx['scenario'], bf_idx['scenario'], nb_idx['scenario']]
            if kc['scenario'] == 0.0:
                print('Invalid value for k')

        elif advanced_baseline:
            nb = {}
            bs = {}
            bf = {}

            nb = self.waste_fractions.at[2000, 'metal'] + self.waste_fractions.at[2000, 'glass'] + self.waste_fractions.at[2000, 'plastic'] + self.waste_fractions.at[2000, 'other'] + self.waste_fractions.at[2000, 'metal']
            bs  = self.waste_fractions.at[2000, 'wood'] + self.waste_fractions.at[2000, 'paper_cardboard'] + self.waste_fractions.at[2000, 'textiles']
            bf  = self.waste_fractions.at[2000, 'food'] + self.waste_fractions.at[2000, 'green']

            bs_idx = {}
            bf_idx = {}
            nb_idx = {}

            bs_idx = int(bs * 8)
            bf_idx = int(bf * 8)
            nb_idx = int(nb * 8)

            if nb_idx == 8:
                nb_idx = 7
            if bs_idx == 8:
                bs_idx = 7
            if bf_idx == 8:
                bf_idx = 7

            kc = {}
            kc = lookup_array[bs_idx, bf_idx, nb_idx]
            if kc == 0:
                print('Invalid value for k')

        else:
            nb = self.waste_fractions.at[2000, 'metal'] + self.waste_fractions.at[2000, 'glass'] + self.waste_fractions.at[2000, 'plastic'] + self.waste_fractions.at[2000, 'other'] + self.waste_fractions.at[2000, 'rubber']
            bs = self.waste_fractions.at[2000, 'wood']+ self.waste_fractions.at[2000, 'paper_cardboard'] + self.waste_fractions.at[2000, 'textiles']
            bf = self.waste_fractions.at[2000, 'food'] + self.waste_fractions.at[2000, 'green']

            bs_idx = int(bs * 8)
            bf_idx = int(bf * 8)
            nb_idx = int(nb * 8)

            if nb_idx == 8:
                nb_idx = 7
            if bs_idx == 8:
                bs_idx = 7
            if bf_idx == 8:
                bf_idx = 7

            kc = lookup_array[bs_idx, bf_idx, nb_idx]
            if kc == 0:
                print('Invalid value for k')

        # ft, accounts for temperature
        tmin = 0
        tmax = 55
        topt = 35
        self.temp = self.temperature
        t = self.temp + 10 # landfill is warmer than ambient

        num = (t - tmax) * (t - tmin) ** 2
        denom = (topt - tmin) * \
            ((topt - tmin) * \
            (t - topt) - \
            (topt - tmax) * \
            (topt + tmin - 2 * t))
        
        if denom != 0:
            tf = num / denom
        else:
            print('Invalid value for temperature factor')

        # fm, accounts for moisture
        # read more on this to make sure it handles dumpsites correctly. 

        if self.precip < 500:
            fm = 0.1
        elif self.precip >= 500 and self.precip < 1000:
            fm = 0.3
        elif self.precip >= 1000 and self.precip < 1500:
            fm = 0.5
        elif self.precip >= 1500 and self.precip < 2000:
            fm = 0.8
        else:
            fm = 1

        tf = float(tf)

        def create_series(kc, tf, fm, implement_year=None, advanced_baseline=False, advanced_dst=False):
            years = pd.Series(index=range(1960, 2074))
            if advanced_dst:
                # Create baseline series for years up to the implement year
                baseline_series = kc['baseline'] * tf * fm
                years.loc[:implement_year-1] = baseline_series
                
                # Create scenario series for years after the implement year
                scenario_series = kc['scenario'] * tf * fm
                years.loc[implement_year:] = scenario_series
            elif advanced_baseline:
                # Create baseline series for years up to the implement year
                baseline_series = kc * tf * fm
                years.loc[:] = baseline_series
            else:
                # Create baseline series for all years
                baseline_series = kc * tf * fm
                years.loc[:] = baseline_series
            
            return years

        if advanced_dst or advanced_baseline:
            vals = create_series(kc, tf, fm, implement_year, advanced_baseline=advanced_baseline, advanced_dst=advanced_dst)
            self.ks = DecompositionRates(
                food=vals,
                green=vals,
                wood=vals,
                paper_cardboard=vals,
                textiles=vals
            )
        else:
            vals = create_series(kc, tf, fm)
            self.ks = DecompositionRates(
                food=vals,
                green=vals,
                wood=vals,
                paper_cardboard=vals,
                textiles=vals
            )

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
            name (str): The name of the city.
        """
        self.city_name = city_name
        self.country = None
        self.iso3 = None
        self.baseline_parameters = None
        self.scenario_parameters = {}
        self.components = {'food', 'green', 'wood', 'paper_cardboard', 'textiles'}
        self.div_components = {
            'compost': {'food', 'green', 'wood', 'paper_cardboard'},
            'anaerobic': {'food', 'green', 'wood', 'paper_cardboard'},
            'combustion': {'food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'},
            'recycling': {'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'}
        }
        self.waste_types = ['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'metal', 'glass', 'rubber', 'other']
        self.unprocessable = {'food': 0.0192, 'green': 0.042522, 'wood': 0.07896, 'paper_cardboard': 0.12}
        self.non_compostable_not_targeted = {'food': 0.1, 'green': 0.05, 'wood': 0.05, 'paper_cardboard': 0.1}
        self.combustion_reject_rate = 0.1
        self.recycling_reject_rates = {
            'wood': 0.8,
            'paper_cardboard': 0.775,
            'textiles': 0.99,
            'plastic': 0.875,
            'metal': 0.955,
            'glass': 0.88,
            'rubber': 0.78,
            'other': 0.87
        }
        self.latitude = None
        self.longitude = None

    def load_from_csv(self, db: pd.DataFrame, scenario: int = 0) -> None:
        """
        Loads model parameters from the RMI WasteMAP GitHub repo data file.

        Args:
            db (pd.DataFrame): DataFrame containing model parameters for all cities.
            scenario (str): The scenario name, defaults to 'baseline'.

        Returns:
            None
        """
        city_data = db.loc[self.city_name]

        self.country = city_data['Country ISO3'].values[0]

        waste_fractions = WasteFractions(
            food=city_data['Waste Components: Food (%)'].values[0] / 100,
            green=city_data['Waste Components: Green (%)'].values[0] / 100,
            wood=city_data['Waste Components: Wood (%)'].values[0] / 100,
            paper_cardboard=city_data['Waste Components: Paper and Cardboard (%)'].values[0] / 100,
            textiles=city_data['Waste Components: Textiles (%)'].values[0] / 100,
            plastic=city_data['Waste Components: Plastic (%)'].values[0] / 100,
            metal=city_data['Waste Components: Metal (%)'].values[0] / 100,
            glass=city_data['Waste Components: Glass (%)'].values[0] / 100,
            rubber=city_data['Waste Components: Rubber/Leather (%)'].values[0] / 100,
            other=city_data['Waste Components: Other (%)'].values[0] / 100
        )

        div_fractions = DiversionFractions(
            compost=city_data['Diversons: Compost (%)'].values[0] / 100,
            anaerobic=city_data['Diversons: Anaerobic Digestion (%)'].values[0] / 100,
            combustion=city_data['Diversons: Incineration (%)'].values[0] / 100,
            recycling=city_data['Diversons: Recycling (%)'].values[0] / 100
        )

        split_fractions = SplitFractions(
            landfill_w_capture=city_data['Percent of Waste to Landfills with Gas Capture (%)'].values[0] / 100,
            landfill_wo_capture=city_data['Percent of Waste to Landfills without Gas Capture (%)'].values[0] / 100,
            dumpsite=city_data['Percent of Waste to Dumpsites (%)'].values[0] / 100
        )

        div_component_fractions = DivComponentFractions(
            compost=WasteFractions(
                food=city_data['Diversion Components: Composted Food (% of Total Composted)'].values[0] / 100,
                green=city_data['Diversion Components: Composted Green (% of Total Composted)'].values[0] / 100,
                wood=city_data['Diversion Components: Composted Wood (% of Total Composted)'].values[0] / 100,
                paper_cardboard=city_data['Diversion Components: Composted Paper and Cardboard (% of Total Composted)'].values[0] / 100,
                textiles=0, plastic=0, metal=0, glass=0, rubber=0, other=0
            ),
            anaerobic=WasteFractions(
                food=city_data['Diversion Components: Anaerobically Digested Food (% of Total Digested)'].values[0] / 100,
                green=city_data['Diversion Components: Anaerobically Digested Green (% of Total Digested)'].values[0] / 100,
                wood=city_data['Diversion Components: Anaerobically Digested Wood (% of Total Digested)'].values[0] / 100,
                paper_cardboard=city_data['Diversion Components: Anaerobically Digested Paper and Cardboard (% of Total Digested)'].values[0] / 100,
                textiles=0, plastic=0, metal=0, glass=0, rubber=0, other=0
            ),
            combustion=WasteFractions(
                food=city_data['Diversion Components: Incinerated Food (% of Total Incinerated)'].values[0] / 100,
                green=city_data['Diversion Components: Incinerated Green (% of Total Incinerated)'].values[0] / 100,
                wood=city_data['Diversion Components: Incinerated Wood (% of Total Incinerated)'].values[0] / 100,
                paper_cardboard=city_data['Diversion Components: Incinerated Paper and Cardboard (% of Total Incinerated)'].values[0] / 100,
                textiles=city_data['Diversion Components: Incinerated Textiles (% of Total Incinerated)'].values[0] / 100,
                plastic=city_data['Diversion Components: Incinerated Plastic (% of Total Incinerated)'].values[0] / 100,
                metal=0, glass=0, rubber=city_data['Diversion Components: Incinerated Rubber/Leather (% of Total Incinerated)'].values[0] / 100, other=0
            ),
            recycling=WasteFractions(
                wood=city_data['Diversion Components: Recycled Wood (% of Total Recycled)'].values[0] / 100,
                paper_cardboard=city_data['Diversion Components: Recycled Paper and Cardboard (% of Total Recycled)'].values[0] / 100,
                plastic=city_data['Diversion Components: Recycled Plastic (% of Total Recycled)'].values[0] / 100,
                rubber=city_data['Diversion Components: Recycled Rubber/Leather (% of Total Recycled)'].values[0] / 100,
                textiles=city_data['Diversion Components: Recycled Textiles (% of Total Recycled)'].values[0] / 100,
                glass=city_data['Diversion Components: Recycled Glass (% of Total Recycled)'].values[0] / 100,
                metal=city_data['Diversion Components: Recycled Metal (% of Total Recycled)'].values[0] / 100,
                other=city_data['Diversion Components: Recycled Other (% of Total Recycled)'].values[0] / 100,
                food=0, green=0
            )
        )

        # ks = DecompositionRates(
        #     food=city_data['k: Food'].values[0],
        #     green=city_data['k: Green'].values[0],
        #     wood=city_data['k: Wood'].values[0],
        #     paper_cardboard=city_data['k: Paper and Cardboard'].values[0],
        #     textiles=city_data['k: Textiles'].values[0]
        # )

        non_compostable_not_targeted_total = sum([
            self.non_compostable_not_targeted[x] * getattr(div_component_fractions.compost, x) for x in self.div_components['compost']
        ])
        if np.isnan(non_compostable_not_targeted_total):
            non_compostable_not_targeted_total = 0

        gas_capture_efficiency = city_data['Methane Capture Efficiency (%)'].values[0] / 100
        mef_compost = city_data['MEF: Compost'].values[0]
        waste_mass = city_data['Waste Generation Rate (tons/year)'].values[0]

        year_of_data_pop = city_data['Year of Data Collection'].values[0]

        city_instance_attrs = {
            'city_name': self.city_name,
            'country': self.country,
            'components': self.components,
            'div_components': self.div_components,
            'waste_types': self.waste_types,
            'unprocessable': self.unprocessable,
            'non_compostable_not_targeted': self.non_compostable_not_targeted,
            'combustion_reject_rate': self.combustion_reject_rate,
            'recycling_reject_rates': self.recycling_reject_rates
        }

        city_parameters = CityParameters(
            waste_fractions=waste_fractions,
            div_fractions=div_fractions,
            split_fractions=split_fractions,
            div_component_fractions=div_component_fractions,
            precip=float(city_data['Average Annual Precipitation (mm/year)'].values[0]),
            growth_rate_historic=city_data['Population Growth Rate: Historic (%)'].values[0] / 100 + 1,
            growth_rate_future=city_data['Population Growth Rate: Future (%)'].values[0] / 100 + 1,
            waste_per_capita=city_data['Waste Generation Rate per Capita (kg/person/day)'].values[0],
            precip_zone=city_data['Precipitation Zone'].values[0],
            #ks=ks,
            gas_capture_efficiency=gas_capture_efficiency,
            mef_compost=mef_compost,
            waste_mass=waste_mass,
            non_compostable_not_targeted_total=non_compostable_not_targeted_total,
            year_of_data_pop=year_of_data_pop,
            scenario=scenario,
            city_instance_attrs=city_instance_attrs,
            population=city_data['Population'].values[0]
        )

        # Filter out the 'landfills' and 'non_zero_landfills' attributes from CityParameters
        #city_params = {k: v for k, v in city_parameters.__dict__.items() if k not in ['landfills', 'non_zero_landfills']}
        #city_params = copy.deepcopy(city_parameters.__dict__)

        self.baseline_parameters = city_parameters

    def load_csv_new(self, db: pd.DataFrame, scenario: int = 0) -> None:
        """
        Loads model parameters from the RMI WasteMAP GitHub repo data file.

        Args:
            db (pd.DataFrame): DataFrame containing model parameters for all cities.
            scenario (str): The scenario name, defaults to 'baseline'.

        Returns:
            None
        """
        city_data = db.loc[self.city_name]

        self.country = city_data['Country ISO3'].values[0]

        # Define the range of years
        years = range(1960, 2074)

        waste_fractions = WasteFractions(
            food=city_data['Waste Components: Food (%)'].values[0] / 100,
            green=city_data['Waste Components: Green (%)'].values[0] / 100,
            wood=city_data['Waste Components: Wood (%)'].values[0] / 100,
            paper_cardboard=city_data['Waste Components: Paper and Cardboard (%)'].values[0] / 100,
            textiles=city_data['Waste Components: Textiles (%)'].values[0] / 100,
            plastic=city_data['Waste Components: Plastic (%)'].values[0] / 100,
            metal=city_data['Waste Components: Metal (%)'].values[0] / 100,
            glass=city_data['Waste Components: Glass (%)'].values[0] / 100,
            rubber=city_data['Waste Components: Rubber/Leather (%)'].values[0] / 100,
            other=city_data['Waste Components: Other (%)'].values[0] / 100
        )
        waste_fractions_dict = waste_fractions.model_dump()
        waste_fractions = pd.DataFrame(waste_fractions_dict, index=years)

        div_fractions = DiversionFractions(
            compost=city_data['Diversons: Compost (%)'].values[0] / 100,
            anaerobic=city_data['Diversons: Anaerobic Digestion (%)'].values[0] / 100,
            combustion=city_data['Diversons: Incineration (%)'].values[0] / 100,
            recycling=city_data['Diversons: Recycling (%)'].values[0] / 100
        )
        div_fractions_dict = div_fractions.model_dump()
        div_fractions = pd.DataFrame(div_fractions_dict, index=years)

        split_fractions = SplitFractions(
            landfill_w_capture=city_data['Percent of Waste to Landfills with Gas Capture (%)'].values[0] / 100,
            landfill_wo_capture=city_data['Percent of Waste to Landfills without Gas Capture (%)'].values[0] / 100,
            dumpsite=city_data['Percent of Waste to Dumpsites (%)'].values[0] / 100
        )

        div_component_fractions = DivComponentFractions(
            compost=WasteFractions(
                food=city_data['Diversion Components: Composted Food (% of Total Composted)'].values[0] / 100,
                green=city_data['Diversion Components: Composted Green (% of Total Composted)'].values[0] / 100,
                wood=city_data['Diversion Components: Composted Wood (% of Total Composted)'].values[0] / 100,
                paper_cardboard=city_data['Diversion Components: Composted Paper and Cardboard (% of Total Composted)'].values[0] / 100,
                textiles=0, plastic=0, metal=0, glass=0, rubber=0, other=0
            ),
            anaerobic=WasteFractions(
                food=city_data['Diversion Components: Anaerobically Digested Food (% of Total Digested)'].values[0] / 100,
                green=city_data['Diversion Components: Anaerobically Digested Green (% of Total Digested)'].values[0] / 100,
                wood=city_data['Diversion Components: Anaerobically Digested Wood (% of Total Digested)'].values[0] / 100,
                paper_cardboard=city_data['Diversion Components: Anaerobically Digested Paper and Cardboard (% of Total Digested)'].values[0] / 100,
                textiles=0, plastic=0, metal=0, glass=0, rubber=0, other=0
            ),
            combustion=WasteFractions(
                food=city_data['Diversion Components: Incinerated Food (% of Total Incinerated)'].values[0] / 100,
                green=city_data['Diversion Components: Incinerated Green (% of Total Incinerated)'].values[0] / 100,
                wood=city_data['Diversion Components: Incinerated Wood (% of Total Incinerated)'].values[0] / 100,
                paper_cardboard=city_data['Diversion Components: Incinerated Paper and Cardboard (% of Total Incinerated)'].values[0] / 100,
                textiles=city_data['Diversion Components: Incinerated Textiles (% of Total Incinerated)'].values[0] / 100,
                plastic=city_data['Diversion Components: Incinerated Plastic (% of Total Incinerated)'].values[0] / 100,
                metal=0, glass=0, rubber=city_data['Diversion Components: Incinerated Rubber/Leather (% of Total Incinerated)'].values[0] / 100, other=0
            ),
            recycling=WasteFractions(
                wood=city_data['Diversion Components: Recycled Wood (% of Total Recycled)'].values[0] / 100,
                paper_cardboard=city_data['Diversion Components: Recycled Paper and Cardboard (% of Total Recycled)'].values[0] / 100,
                plastic=city_data['Diversion Components: Recycled Plastic (% of Total Recycled)'].values[0] / 100,
                rubber=city_data['Diversion Components: Recycled Rubber/Leather (% of Total Recycled)'].values[0] / 100,
                textiles=city_data['Diversion Components: Recycled Textiles (% of Total Recycled)'].values[0] / 100,
                glass=city_data['Diversion Components: Recycled Glass (% of Total Recycled)'].values[0] / 100,
                metal=city_data['Diversion Components: Recycled Metal (% of Total Recycled)'].values[0] / 100,
                other=city_data['Diversion Components: Recycled Other (% of Total Recycled)'].values[0] / 100,
                food=0, green=0
            )
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

        non_compostable_not_targeted_total = sum([
            self.non_compostable_not_targeted[x] * div_component_fractions.compost.loc[2000, x] for x in self.div_components['compost']
        ])
        non_compostable_not_targeted_total = pd.Series(non_compostable_not_targeted_total, index=years)
        if non_compostable_not_targeted_total.isna().all():
            non_compostable_not_targeted_total = pd.Series(0, index=years)

        gas_capture_efficiency = city_data['Methane Capture Efficiency (%)'].values[0] / 100
        gas_capture_efficiency = pd.Series(gas_capture_efficiency, index=years)

        mef_compost = city_data['MEF: Compost'].values[0]

        waste_mass = city_data['Waste Generation Rate (tons/year)'].values[0]
        waste_mass = pd.Series(waste_mass, index=years)

        year_of_data_pop = city_data['Year of Data Collection (Population)'].values[0]

        city_instance_attrs = {
            'city_name': self.city_name,
            'country': self.country,
            'components': self.components,
            'div_components': self.div_components,
            'waste_types': self.waste_types,
            'unprocessable': self.unprocessable,
            'non_compostable_not_targeted': self.non_compostable_not_targeted,
            'combustion_reject_rate': self.combustion_reject_rate,
            'recycling_reject_rates': self.recycling_reject_rates
        }

        waste_masses = {x: waste_mass.at[2000] * waste_fractions.loc[2000, x] for x in self.waste_types}
        waste_masses = WasteMasses(**waste_masses)

        try:
            city_parameters = CityParameters(
                waste_fractions=waste_fractions,
                div_fractions=div_fractions,
                split_fractions=split_fractions,
                div_component_fractions=div_component_fractions,
                precip=float(city_data['Average Annual Precipitation (mm/year)'].values[0]),
                temperature=float(city_data['Temperature (C)'].values[0]),
                growth_rate_historic=city_data['Population Growth Rate: Historic (%)'].values[0] / 100 + 1,
                growth_rate_future=city_data['Population Growth Rate: Future (%)'].values[0] / 100 + 1,
                waste_per_capita=city_data['Waste Generation Rate per Capita (kg/person/day)'].values[0],
                precip_zone=city_data['Precipitation Zone'].values[0],
                #ks=ks,
                gas_capture_efficiency=gas_capture_efficiency,
                mef_compost=mef_compost,
                waste_mass=waste_mass,
                waste_masses=waste_masses,
                non_compostable_not_targeted_total=non_compostable_not_targeted_total,
                year_of_data_pop=year_of_data_pop,
                scenario=scenario,
                city_instance_attrs=city_instance_attrs,
                population=city_data['Population'].values[0],
                source_msw=city_data['Data Source (Waste Mass)'].values[0]
            )
        except Exception as e:
            raise CustomError('city_params_error', f"Error creating CityParameters instance: {e}")

        # Filter out the 'landfills' and 'non_zero_landfills' attributes from CityParameters
        #city_params = {k: v for k, v in city_parameters.__dict__.items() if k not in ['landfills', 'non_zero_landfills']}
        #city_params = copy.deepcopy(city_parameters.__dict__)

        self.baseline_parameters = city_parameters

    def load_andre_params(self, row):
        """
        Loads model parameters from the internal RMI WasteMAP database. Defaults are used
        where data is missing, incomplete, or incompatible.

        Args:
            row (tuple): row[0] is the index of the row in the dataframe used for input, 
            row[1] is the row itself.

        Returns:
            None
        """
        # Basic information
        #idx = row[0]
        row = row[1]
        data_source = row['population_data_source']
        country = row['country']
        self.country = country
        iso3 = row['iso']
        self.iso3 = iso3
        region = defaults_2019.region_lookup[country]
        self.region = region
        year_of_data_pop = row['population_year']
        assert np.isnan(year_of_data_pop) == False, 'Population year is missing'
        year_of_data_msw = row['msw_collected_year']
        if np.isnan(year_of_data_msw):
            year_of_data_msw = row['msw_generated_year']
        if np.isnan(year_of_data_msw):
            year_of_data_msw = row['data_collection_year'].iloc[0]
        year_of_data_msw = int(year_of_data_msw)

        # Define the range of years
        years = range(1960, 2074)
        
        # Hardcode missing population values
        population = float(row['population_count'])
        if self.city_name == 'Pago Pago':
            population = 3656
            year_of_data_pop = 2010
        elif self.city_name == 'Kano':
            population = 2828861
            year_of_data_pop = 2006
        elif self.city_name == 'Ramallah':
            population = 38998
            year_of_data_pop = 2017
        elif self.city_name == 'Soweto':
            population = 1271628
            year_of_data_pop = 2011
        elif self.city_name == 'Kadoma City':
            population = 116300
            year_of_data_pop = 2022
        elif self.city_name == 'Mbare':
            population = 450000
            year_of_data_pop = 2020
        elif self.city_name == 'Masvingo City':
            population = 90286
            year_of_data_pop = 2022
        elif self.city_name == 'Limbe':
            population = 84223
            year_of_data_pop = 2005
        elif self.city_name == 'Labe':
            population = 200000
            year_of_data_pop = 2014


        growth_rate_historic = row['historic_growth_rate']
        growth_rate_future = row['future_growth_rate']

        # lat lon
        self.latitude = row['latitude']
        self.longitude = row['longitude']

        self.waste_mass_defaults = False

        # Get waste total
        try:
            waste_mass_load = float(row['msw_collected_metric_tons_per_year']) # unit is tons
            if np.isnan(waste_mass_load):
                waste_mass_load = float(row['msw_generated_metric_tons_per_year'])
            waste_per_capita = waste_mass_load * 1000 / population / 365 #unit is kg/person/day
        except:
            waste_mass_load = float(row['msw_collected_metric_tons_per_year'].replace(',', ''))
            if np.isnan(waste_mass_load):
                waste_mass_load = float(row['msw_generated_metric_tons_per_year'].replace(',', ''))
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
        #self.informal_fraction = np.nan_to_num(row['percent_informal_sector_percent_collected_by_informal_sector_percent']) / 100
        #self.waste_mass = self.waste_mass_load * (1 - self.informal_fraction)
        waste_mass = waste_mass_load
        
        # Adjust waste mass to account for difference in reporting years between msw and population
        #if self.data_source == 'World Bank':
        if year_of_data_msw != year_of_data_pop:
            year_difference = year_of_data_pop - year_of_data_msw
            if year_of_data_msw < year_of_data_pop:
                waste_mass *= (growth_rate_historic ** year_difference)
                waste_per_capita = waste_mass * 1000 / population / 365
            else:
                waste_mass *= (growth_rate_future ** year_difference)
                waste_per_capita = waste_mass * 1000 / population / 365
        
        # Waste fractions
        waste_fractions = pd.Series({
            'food': row['composition_food_organic_waste_percent'] / 100,
            'green': row['composition_yard_garden_green_waste_percent'] / 100,
            'wood': row['composition_wood_percent'] / 100,
            'paper_cardboard': row['composition_paper_cardboard_percent'] / 100,
            'textiles': row['composition_textiles_percent'] / 100,
            'plastic': row['composition_plastic_percent'] / 100,
            'metal': row['composition_metal_percent'] / 100,
            'glass': row['composition_glass_percent'] / 100,
            'rubber': row['composition_rubber_leather_percent'] / 100,
            'other': row['composition_other_percent'] / 100
        })

        # Add zeros where there are no values unless all values are nan, in which case use defaults
        self.waste_fractions_defaults = False
        if waste_fractions.isna().all():
            self.waste_fractions_defaults = True
            if iso3 in defaults_2019.waste_fractions_country:
                waste_fractions = defaults_2019.waste_fractions_country.loc[iso3, :]
            else:
                if region == 'Rest of Oceania':
                    print(self.city_name)
                waste_fractions = defaults_2019.waste_fraction_defaults.loc[region, :]
        else:
            waste_fractions.fillna(0, inplace=True)
            #waste_fractions['textiles'] = 0
        
        if (waste_fractions.sum() < .98) or (waste_fractions.sum() > 1.02):
            self.waste_fractions_defaults = True
            #print('waste fractions do not sum to 1')
            if iso3 in defaults_2019.waste_fractions_country:
                waste_fractions = defaults_2019.waste_fractions_country.loc[iso3, :]
            else:
                if region == 'Rest of Oceania':
                    print(self.city_name)
                waste_fractions = defaults_2019.waste_fraction_defaults.loc[region, :]
    
        waste_fractions_dict = waste_fractions.to_dict()
        
        # Normalize waste fractions to sum to 1
        s = sum([x for x in waste_fractions_dict.values()])
        waste_fractions = {x: waste_fractions[x] / s for x in waste_fractions.keys()}
        waste_fractions = pd.DataFrame(waste_fractions, index=years)

        try:
            # Calculate MEF for compost -- emissions from composted waste
            mef_compost = (0.0055 * waste_fractions_dict['food']/(waste_fractions_dict['food'] + waste_fractions_dict['green']) + \
                           0.0139 * waste_fractions_dict['green']/(waste_fractions_dict['food'] + waste_fractions_dict['green'])) * 1.1023 * 0.7 # / 28
                           # Unit is Mg CO2e/Mg of organic waste, wtf, so convert to CH4. Mistake in sweet here
        except:
            mef_compost = 0
        
        # Precipitation
        precip = float(row['mean_yearly_precip_2000_2021'])
        #precip_data = pd.read_excel('/Users/hugh/Downloads/Cities Waste Dataset_2010-2019_precip.xlsx')
        #self.precip = precip_data[precip_data['city_original'] == self.name]['total_precipitation(mm)_1970_2000'].values[0]
        precip_zone = defaults_2019.get_precipitation_zone(precip)
    
        # depth
        depth = 1 # m
    
        # k values, which are decomposition rates
        # ks = defaults_2019.k_defaults[precip_zone]
        
        # Model components
        components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles'])
        
        # Compost params
        compost_components = set(['food', 'green', 'wood', 'paper_cardboard'])
        compost_fraction = float(row['waste_treatment_compost_percent']) / 100
        
        # Anaerobic digestion params
        anaerobic_components = set(['food', 'green', 'wood', 'paper_cardboard'])
        anaerobic_fraction = float(row['waste_treatment_anaerobic_digestion_percent']) / 100   
        
        # Combustion params
        combustion_components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'])
        value1 = float(row['waste_treatment_incineration_percent'])
        value2 = float(row['waste_treatment_advanced_thermal_treatment_percent'])
        if np.isnan(value1) and np.isnan(value2):
            combustion_fraction = np.nan
        else:
            combustion_fraction = (np.nan_to_num(value1) + np.nan_to_num(value2)) / 100
        
        # Recycling params
        recycling_components = set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'])
        recycling_fraction = float(row['waste_treatment_recycling_percent']) / 100
        
        # How much waste is diverted to landfill with gas capture
        gas_capture_percent = np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent']) / 100
        
        div_components = {}
        div_components['compost'] = compost_components
        div_components['anaerobic'] = anaerobic_components
        div_components['combustion'] = combustion_components
        div_components['recycling'] = recycling_components

        # Determine if we need to use defaults for landfills and diversion fractions
        landfill_inputs = [
            float(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent']),
            float(row['waste_treatment_controlled_landfill_percent']),
            float(row['waste_treatment_landfill_unspecified_percent']),
            float(row['waste_treatment_open_dump_percent'])
        ]
        all_nan_fill = all(np.isnan(value) for value in landfill_inputs)
        total_fill = sum(0 if np.isnan(x) else x for x in landfill_inputs) / 100
        diversions = [compost_fraction, anaerobic_fraction, combustion_fraction, recycling_fraction]
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

            if iso3 in ['CAN', 'CHE', 'DEU']:
                split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 1.0, 'dumpsite': 0.0}
            else:
                if iso3 in defaults_2019.fraction_open_dumped_country:
                    split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled_country[iso3],
                        'dumpsite': defaults_2019.fraction_open_dumped_country[iso3]
                    }
                    self.landfill_split_defaults = True
                elif region in defaults_2019.fraction_open_dumped:
                    split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled[region],
                        'dumpsite': defaults_2019.fraction_open_dumped[region]
                    }
                    self.landfill_split_defaults = True
                else:
                    if region in defaults_2019.landfill_default_regions:
                        split_fractions = {'landfill_w_capture': 0, 'landfill_wo_capture': 1, 'dumpsite': 0}
                    else:
                        split_fractions = {'landfill_w_capture': 0, 'landfill_wo_capture': 0, 'dumpsite': 1}

        # Second case to check: all diversions are nan, but landfills are not. Use defaults for diversions if landfills sum to less than 1
        # This assumes that entered data is incomplete. Also, normalize landfills to sum to 1.
        # Caveat: if landfills sum to 1, assume diversions are supposed to be 0. 
        elif all_nan_div and total_fill > .99:
            split_fractions = {
                'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
                'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
                                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
                'dumpsite': np.nan_to_num(row['waste_treatment_open_dump_percent'])/100
            }
        elif all_nan_div and total_fill < .99:
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
                'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
                'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
                                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
                'dumpsite': np.nan_to_num(row['waste_treatment_open_dump_percent'])/100
            }

        # Third case to check: all landfills are nan, but diversions are not. Use defaults for landfills
        elif all_nan_fill:
            if iso3 in ['CAN', 'CHE', 'DEU']:
                split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 1.0, 'dumpsite': 0.0}
            else:
                if iso3 in defaults_2019.fraction_open_dumped_country:
                    split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled_country[iso3],
                        'dumpsite': defaults_2019.fraction_open_dumped_country[iso3]
                    }
                    self.landfill_split_defaults = True
                elif region in defaults_2019.fraction_open_dumped:
                    split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled[region],
                        'dumpsite': defaults_2019.fraction_open_dumped[region]
                    }
                    self.landfill_split_defaults = True
                else:
                    if region in defaults_2019.landfill_default_regions:
                        split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 1.0, 'dumpsite': 0.0}
                    else:
                        split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 0.0, 'dumpsite': 1.0}
        
        # Fourth case to check: imported non-nan values in both landfills and diversions. Use the values. 
        else:
            split_fractions = {
                'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
                'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
                                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
                'dumpsite': np.nan_to_num(row['waste_treatment_open_dump_percent'])/100
            }
        
        # Normalize landfills to 1
        split_total = sum([x for x in split_fractions.values()])
        if split_total == 0:
            if region in defaults_2019.landfill_default_regions:
                split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 1.0, 'dumpsite': 0.0}
            else:
                split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 0.0, 'dumpsite': 1.0}
        split_total = sum([x for x in split_fractions.values()])
        for site in split_fractions.keys():
            split_fractions[site] /= split_total
        
        # Replace diversion NaN values with 0
        compost_fraction, anaerobic_fraction, combustion_fraction, recycling_fraction = [np.nan_to_num(x) for x in [compost_fraction, anaerobic_fraction, combustion_fraction, recycling_fraction]]

        # if self.iso3 == 'NGA':
        #     self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 0.0, 'dumpsite': 1.0}
        # Instantiate landfills
        # self.landfill_w_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_w_capture'], gas_capture=True)
        # self.landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_wo_capture'], gas_capture=False)
        # self.dumpsite = Landfill(self, 1960, 2073, 'dumpsite', 0.4, fraction_of_waste=self.split_fractions['dumpsite'], gas_capture=False)
        
        # landfills = [self.landfill_w_capture, self.landfill_wo_capture, self.dumpsite]
        # Only running model on landfills with non-zero waste reduces computation
        # non_zero_landfills = [x for x in self.landfills if x.fraction_of_waste > 0]
        
        divs = {}

        div_fractions_dict = {
            'compost': compost_fraction,
            'anaerobic': anaerobic_fraction,
            'combustion': combustion_fraction,
            'recycling': recycling_fraction
        }

        # Normalize diversion fractions to sum to 1 if they exceed it
        s = sum(x for x in div_fractions_dict.values())
        if  s > 1:
            for div in div_fractions_dict:
                div_fractions_dict[div] /= s
        assert sum(x for x in div_fractions_dict.values()) <= 1, 'Diversion fractions sum to more than 1'
        div_fractions = pd.DataFrame(div_fractions_dict, index=years)

        # # Use IPCC defaults if no data
        # if s == 0:
        #     self.div_fractions['compost'] = defaults.fraction_composted[self.region]
        #     self.div_fractions['combustion'] = defaults.fraction_incinerated[self.region]
        
        # UN Habitat has its own data import procedure
        if data_source == 'UN Habitat':
            return
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
        else:
            # Determine diversion waste type fractions
            def calculate_component_fractions(waste_fractions: WasteFractions, div_type: str) -> WasteFractions:
                components = self.div_components[div_type]
                filtered_fractions = {waste: waste_fractions[waste].at[2000] for waste in components}
                total = sum(filtered_fractions.values())
                normalized_fractions = {waste: fraction / total for waste, fraction in filtered_fractions.items()}
                return normalized_fractions

            div_component_fractions = DivComponentFractionsDF(
                compost=pd.DataFrame(calculate_component_fractions(waste_fractions, 'compost'), index=years),
                anaerobic=pd.DataFrame(calculate_component_fractions(waste_fractions, 'anaerobic'), index=years),
                combustion=pd.DataFrame(calculate_component_fractions(waste_fractions, 'combustion'), index=years),
                recycling=pd.DataFrame(calculate_component_fractions(waste_fractions, 'recycling'), index=years),
            )

        non_compostable_not_targeted_total = sum([
            self.non_compostable_not_targeted[x] * div_component_fractions.compost.loc[2000, x] for x in div_components['compost']
        ])
        non_compostable_not_targeted_total = pd.Series(non_compostable_not_targeted_total, index=years)
        if non_compostable_not_targeted_total.isna().all():
            non_compostable_not_targeted_total = pd.Series(0, index=years)

        gas_capture_efficiency = pd.Series(0.6, index=years)

        waste_mass = pd.Series(waste_mass, index=years)

        city_instance_attrs = {
            'city_name': self.city_name,
            'country': country,
            'components': components,
            'div_components': div_components,
            'waste_types': self.waste_types,
            'unprocessable': self.unprocessable,
            'non_compostable_not_targeted': self.non_compostable_not_targeted,
            'combustion_reject_rate': self.combustion_reject_rate,
            'recycling_reject_rates': self.recycling_reject_rates
        }

        waste_masses = {x: waste_mass.at[2000] * waste_fractions.loc[2000, x] for x in self.waste_types}
        waste_masses = WasteMasses(**waste_masses)

        split_fractions_old = split_fractions
        split_fractions = SplitFractions(
            landfill_w_capture=split_fractions_old['landfill_w_capture'],
            landfill_wo_capture=split_fractions_old['landfill_wo_capture'],
            dumpsite=split_fractions_old['dumpsite']
        )

        waste_masses_df = waste_fractions.multiply(waste_mass, axis=0)
        waste_generated_df = WasteGeneratedDF.create(
            waste_masses_df,
            1960, 
            2073, 
            year_of_data_pop, 
            growth_rate_historic, 
            growth_rate_future
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
            temp=None,
            temperature=None,
            waste_burning_emissions=None,
            non_compostable_not_targeted_total=non_compostable_not_targeted_total,
            source_pop=data_source,
        )
        self.baseline_parameters = baseline

        # Check masses consistency
        self._check_masses_v2(scenario=0)
        if baseline.input_problems:
            print('Input problems detected in baseline parameters.')
            return
        
        self._calculate_net_masses()
        if (baseline.net_masses < 0).any().any():
            print(f'Invalid new value')
            return
        
        # Assign the baseline parameters to the city instance
        try:
            self._calculate_divs()
        except:
            print('remove this after debug')

    def sinar_city_and_facility(self, row):
        """
        Loads model parameters from the internal RMI WasteMAP database. Defaults are used
        where data is missing, incomplete, or incompatible.

        Args:
            row (tuple): row[0] is the index of the row in the dataframe used for input, 
            row[1] is the row itself.

        Returns:
            None
        """
        # Basic information
        #idx = row[0]
        row = row[1]
        self.data_source = row['population_data_source']
        self.country = row['country']
        self.iso3 = row['iso']
        self.region = defaults_2019.region_lookup[self.country]
        self.year_of_data_pop = row['population_year']
        assert np.isnan(self.year_of_data_pop) == False, 'Population year is missing'
        self.year_of_data_msw = row['msw_collected_year']
        if np.isnan(self.year_of_data_msw):
            self.year_of_data_msw = row['msw_generated_year']
        if np.isnan(self.year_of_data_msw):
            self.year_of_data_msw = row['data_collection_year'].iloc[0]
        self.year_of_data_msw = int(self.year_of_data_msw)
        
        # Hardcode missing population values
        self.population = float(row['population_count'])
        if self.name == 'Pago Pago':
            self.population = 3656
            self.year_of_data_pop = 2010
        elif self.name == 'Kano':
            self.population = 2828861
            self.year_of_data_pop = 2006
        elif self.name == 'Ramallah':
            self.population = 38998
            self.year_of_data_pop = 2017
        elif self.name == 'Soweto':
            self.population = 1271628
            self.year_of_data_pop = 2011
        elif self.name == 'Kadoma City':
            self.population = 116300
            self.year_of_data_pop = 2022
        elif self.name == 'Mbare':
            self.population = 450000
            self.year_of_data_pop = 2020
        elif self.name == 'Masvingo City':
            self.population = 90286
            self.year_of_data_pop = 2022
        elif self.name == 'Limbe':
            self.population = 84223
            self.year_of_data_pop = 2005
        elif self.name == 'Labe':
            self.population = 200000
            self.year_of_data_pop = 2014

        self.growth_rate_historic = row['historic_growth_rate']
        self.growth_rate_future = row['future_growth_rate']

        # lat lon
        self.lat = row['latitude']
        self.lon = row['longitude']

        self.waste_mass_defaults = False

        # Get waste total
        try:
            self.waste_mass_load = float(row['msw_collected_metric_tons_per_year']) # unit is tons
            if np.isnan(self.waste_mass_load):
                self.waste_mass_load = float(row['msw_generated_metric_tons_per_year'])
            self.waste_per_capita = self.waste_mass_load * 1000 / self.population / 365 #unit is kg/person/day
        except:
            self.waste_mass_load = float(row['msw_collected_metric_tons_per_year'].replace(',', ''))
            if np.isnan(self.waste_mass_load):
                self.waste_mass_load = float(row['msw_generated_metric_tons_per_year'].replace(',', ''))
            self.waste_per_capita = self.waste_mass_load * 1000 / self.population / 365
        if self.waste_mass_load != self.waste_mass_load:
            # Use per capita default
            self.waste_mass_defaults = True
            if self.iso3 in defaults_2019.msw_per_capita_country:
                self.waste_per_capita = defaults_2019.msw_per_capita_country[self.iso3]
                self.year_of_data_msw = 2019
            else:
                self.waste_per_capita = defaults_2019.msw_per_capita_defaults[self.region]
                self.year_of_data_msw = 2019
            self.waste_mass_load = self.waste_per_capita * self.population / 1000 * 365
        
        # Subtract mass that is informally collected
        #self.informal_fraction = np.nan_to_num(row['percent_informal_sector_percent_collected_by_informal_sector_percent']) / 100
        #self.waste_mass = self.waste_mass_load * (1 - self.informal_fraction)
        self.waste_mass = self.waste_mass_load
        
        # Adjust waste mass to account for difference in reporting years between msw and population
        #if self.data_source == 'World Bank':
        if self.year_of_data_msw != self.year_of_data_pop:
            year_difference = self.year_of_data_pop - self.year_of_data_msw
            if self.year_of_data_msw < self.year_of_data_pop:
                self.waste_mass *= (self.growth_rate_historic ** year_difference)
                self.waste_per_capita = self.waste_mass * 1000 / self.population / 365
            else:
                self.waste_mass *= (self.growth_rate_future ** year_difference)
                self.waste_per_capita = self.waste_mass * 1000 / self.population / 365

        # # Collection coverage_stats
        # # Don't use these for now, as it seems like WB already adjusted total msw to account for these. 
        # coverage_by_area = float(row['waste_collection_coverage_total_percent_of_geographic_area_percent_of_geographic_area']) / 100
        # coverage_by_households = float(row['waste_collection_coverage_total_percent_of_households_percent_of_households']) / 100
        # coverage_by_pop = float(row['waste_collection_coverage_total_percent_of_population_percent_of_population']) / 100
        # coverage_by_waste = float(row['waste_collection_coverage_total_percent_of_waste_percent_of_waste']) / 100
        
        # if coverage_by_waste == coverage_by_waste:
        #     self.mass *= 
        
        # Waste fractions
        waste_fractions = row[['composition_food_organic_waste_percent', 
                             'composition_yard_garden_green_waste_percent', 
                             'composition_wood_percent',
                             'composition_paper_cardboard_percent',
                             'composition_plastic_percent',
                             'composition_metal_percent',
                             'composition_glass_percent',
                             'composition_other_percent',
                             'composition_rubber_leather_percent',
                             'composition_textiles_percent'
                             ]]
    
        waste_fractions.rename(index={'composition_food_organic_waste_percent': 'food',
                                        'composition_yard_garden_green_waste_percent': 'green',
                                        'composition_wood_percent': 'wood',
                                        'composition_paper_cardboard_percent': 'paper_cardboard',
                                        'composition_plastic_percent': 'plastic',
                                        'composition_metal_percent': 'metal',
                                        'composition_glass_percent': 'glass',
                                        'composition_other_percent': 'other',
                                        'composition_rubber_leather_percent': 'rubber',
                                        'composition_textiles_percent': 'textiles'
                                        }, inplace=True)
        waste_fractions /= 100
        
        # if self.region == 'Rest of Oceania':
        #     print(self.name)
        #     print(waste_fractions)

        # Add zeros where there are no values unless all values are nan, in which case use defaults
        self.waste_fractions_defaults = False
        if waste_fractions.isna().all():
            self.waste_fractions_defaults = True
            if self.iso3 in defaults_2019.waste_fractions_country:
                waste_fractions = defaults_2019.waste_fractions_country.loc[self.iso3, :]
            else:
                if self.region == 'Rest of Oceania':
                    print(self.name)
                waste_fractions = defaults_2019.waste_fraction_defaults.loc[self.region, :]
        else:
            waste_fractions.fillna(0, inplace=True)
            #waste_fractions['textiles'] = 0
        
        if (waste_fractions.sum() < .98) or (waste_fractions.sum() > 1.02):
            self.waste_fractions_defaults = True
            #print('waste fractions do not sum to 1')
            if self.iso3 in defaults_2019.waste_fractions_country:
                waste_fractions = defaults_2019.waste_fractions_country.loc[self.iso3, :]
            else:
                if self.region == 'Rest of Oceania':
                    print(self.name)
                waste_fractions = defaults_2019.waste_fraction_defaults.loc[self.region, :]
    
        self.waste_fractions = waste_fractions.to_dict()
        
        # Normalize waste fractions to sum to 1
        s = sum([x for x in self.waste_fractions.values()])
        self.waste_fractions = {x: self.waste_fractions[x] / s for x in self.waste_fractions.keys()}

        try:
            # Calculate MEF for compost -- emissions from composted waste
            self.mef_compost = (0.0055 * waste_fractions['food']/(waste_fractions['food'] + waste_fractions['green']) + \
                           0.0139 * waste_fractions['green']/(waste_fractions['food'] + waste_fractions['green'])) * 1.1023 * 0.7 # / 28
                           # Unit is Mg CO2e/Mg of organic waste, wtf, so convert to CH4. Mistake in sweet here
        except:
            self.mef_compost = 0
        
        # Precipitation
        self.precip = float(row['mean_yearly_precip_2000_2021'])
        #precip_data = pd.read_excel('/Users/hugh/Downloads/Cities Waste Dataset_2010-2019_precip.xlsx')
        #self.precip = precip_data[precip_data['city_original'] == self.name]['total_precipitation(mm)_1970_2000'].values[0]
        self.precip_zone = defaults_2019.get_precipitation_zone(self.precip)
    
        # depth
        #depth = 10
    
        # k values, which are decomposition rates
        self.ks = defaults_2019.k_defaults[self.precip_zone]
        
        # Model components
        self.components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles'])
        
        # Compost params
        self.compost_components = set(['food', 'green', 'wood', 'paper_cardboard']) # Double check we don't want to include paper
        self.compost_fraction = float(row['waste_treatment_compost_percent']) / 100
        
        # Anaerobic digestion params
        self.anaerobic_components = set(['food', 'green', 'wood', 'paper_cardboard'])
        self.anaerobic_fraction = float(row['waste_treatment_anaerobic_digestion_percent']) / 100   
        
        # Combustion params
        self.combustion_components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'])
        value1 = float(row['waste_treatment_incineration_percent'])
        value2 = float(row['waste_treatment_advanced_thermal_treatment_percent'])
        if np.isnan(value1) and np.isnan(value2):
            self.combustion_fraction = np.nan
        else:
            self.combustion_fraction = (np.nan_to_num(value1) + np.nan_to_num(value2)) / 100
        
        # Recycling params
        self.recycling_components = set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'])
        self.recycling_fraction = float(row['waste_treatment_recycling_percent']) / 100
        
        # How much waste is diverted to landfill with gas capture
        self.gas_capture_percent = np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent']) / 100
        
        self.div_components = {}
        self.div_components['compost'] = self.compost_components
        self.div_components['anaerobic'] = self.anaerobic_components
        self.div_components['combustion'] = self.combustion_components
        self.div_components['recycling'] = self.recycling_components

        # Determine if we need to use defaults for landfills and diversion fractions
        landfill_inputs = [
            float(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent']),
            float(row['waste_treatment_controlled_landfill_percent']),
            float(row['waste_treatment_landfill_unspecified_percent']),
            float(row['waste_treatment_open_dump_percent'])
        ]
        all_nan_fill = all(np.isnan(value) for value in landfill_inputs)
        total_fill = sum(0 if np.isnan(x) else x for x in landfill_inputs) / 100
        diversions = [self.compost_fraction, self.anaerobic_fraction, self.combustion_fraction, self.recycling_fraction]
        all_nan_div = all(np.isnan(value) for value in diversions)

        # First case to check: all diversions and landfills are 0. Use defaults.
        self.diversion_defaults = False
        self.landfill_split_defaults = False
        if all_nan_fill and all_nan_div:
            if self.iso3 in defaults_2019.fraction_composted_country:
                self.compost_fraction = defaults_2019.fraction_composted_country[self.iso3]
                self.diversion_defaults = True
            elif self.region in defaults_2019.fraction_composted:
                self.compost_fraction = defaults_2019.fraction_composted[self.region]
                self.diversion_defaults = True
            else:
                self.compost_fraction = 0.0

            if self.iso3 in defaults_2019.fraction_incinerated_country:
                self.combustion_fraction = defaults_2019.fraction_incinerated_country[self.iso3]
                self.diversion_defaults = True
            elif self.region in defaults_2019.fraction_incinerated:
                self.combustion_fraction = defaults_2019.fraction_incinerated[self.region]
                self.diversion_defaults = True
            else:
                self.combustion_fraction = 0.0

            if self.iso3 in ['CAN', 'CHE', 'DEU']:
                self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 1.0, 'dumpsite': 0.0}
            else:
                if self.iso3 in defaults_2019.fraction_open_dumped_country:
                    self.split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled_country[self.iso3],
                        'dumpsite': defaults_2019.fraction_open_dumped_country[self.iso3]
                    }
                    self.landfill_split_defaults = True
                elif self.region in defaults_2019.fraction_open_dumped:
                    self.split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled[self.region],
                        'dumpsite': defaults_2019.fraction_open_dumped[self.region]
                    }
                    self.landfill_split_defaults = True
                else:
                    if self.region in defaults_2019.landfill_default_regions:
                        self.split_fractions = {'landfill_w_capture': 0, 'landfill_wo_capture': 1, 'dumpsite': 0}
                    else:
                        self.split_fractions = {'landfill_w_capture': 0, 'landfill_wo_capture': 0, 'dumpsite': 1}

        # Second case to check: all diversions are nan, but landfills are not. Use defaults for diversions if landfills sum to less than 1
        # This assumes that entered data is incomplete. Also, normalize landfills to sum to 1.
        # Caveat: if landfills sum to 1, assume diversions are supposed to be 0. 
        elif all_nan_div and total_fill > .99:
            self.split_fractions = {
                'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
                'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
                                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
                'dumpsite': np.nan_to_num(row['waste_treatment_open_dump_percent'])/100
            }
        elif all_nan_div and total_fill < .99:
            if self.iso3 in defaults_2019.fraction_composted_country:
                self.compost_fraction = defaults_2019.fraction_composted_country[self.iso3]
                self.diversion_defaults = True
            elif self.region in defaults_2019.fraction_composted:
                self.compost_fraction = defaults_2019.fraction_composted[self.region]
                self.diversion_defaults = True
            else:
                self.compost_fraction = 0.0

            if self.iso3 in defaults_2019.fraction_incinerated_country:
                self.combustion_fraction = defaults_2019.fraction_incinerated_country[self.iso3]
                self.diversion_defaults = True
            elif self.region in defaults_2019.fraction_incinerated:
                self.combustion_fraction = defaults_2019.fraction_incinerated[self.region]
                self.diversion_defaults = True
            else:
                self.combustion_fraction = 0.0

            self.split_fractions = {
                'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
                'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
                                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
                'dumpsite': np.nan_to_num(row['waste_treatment_open_dump_percent'])/100
            }

        # Third case to check: all landfills are nan, but diversions are not. Use defaults for landfills
        elif all_nan_fill:
            if self.iso3 in ['CAN', 'CHE', 'DEU']:
                self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 1.0, 'dumpsite': 0.0}
            else:
                if self.iso3 in defaults_2019.fraction_open_dumped_country:
                    self.split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled_country[self.iso3],
                        'dumpsite': defaults_2019.fraction_open_dumped_country[self.iso3]
                    }
                    self.landfill_split_defaults = True
                elif self.region in defaults_2019.fraction_open_dumped:
                    self.split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled[self.region],
                        'dumpsite': defaults_2019.fraction_open_dumped[self.region]
                    }
                    self.landfill_split_defaults = True
                else:
                    if self.region in defaults_2019.landfill_default_regions:
                        self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 1.0, 'dumpsite': 0.0}
                    else:
                        self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 0.0, 'dumpsite': 1.0}
        
        # Fourth case to check: imported non-nan values in both landfills and diversions. Use the values. 
        else:
            self.split_fractions = {
                'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
                'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
                                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
                'dumpsite': np.nan_to_num(row['waste_treatment_open_dump_percent'])/100
            }
        
        # Normalize landfills to 1
        split_total = sum([x for x in self.split_fractions.values()])
        if split_total == 0:
            if self.region in defaults_2019.landfill_default_regions:
                self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 1.0, 'dumpsite': 0.0}
            else:
                self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 0.0, 'dumpsite': 1.0}
        split_total = sum([x for x in self.split_fractions.values()])
        for site in self.split_fractions.keys():
            self.split_fractions[site] /= split_total
        
        # Replace diversion NaN values with 0
        attrs = ['compost_fraction', 'anaerobic_fraction', 'combustion_fraction', 'recycling_fraction']
        for attr in attrs:
            if np.isnan(getattr(self, attr)):
                setattr(self, attr, 0.0)

        # if self.iso3 == 'NGA':
        #     self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 0.0, 'dumpsite': 1.0}
        # Instantiate landfills
        self.landfill_w_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_w_capture'], gas_capture=True)
        self.landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_wo_capture'], gas_capture=False)
        self.dumpsite = Landfill(self, 1960, 2073, 'dumpsite', 0.4, fraction_of_waste=self.split_fractions['dumpsite'], gas_capture=False)
        
        self.landfills = [self.landfill_w_capture, self.landfill_wo_capture, self.dumpsite]
        # Only running model on landfills with non-zero waste reduces computation
        self.non_zero_landfills = [x for x in self.landfills if x.fraction_of_waste > 0]
        
        self.divs = {}
        self.div_fractions = {}
        self.div_fractions['compost'] = self.compost_fraction
        self.div_fractions['anaerobic'] = self.anaerobic_fraction
        self.div_fractions['combustion'] = self.combustion_fraction
        self.div_fractions['recycling'] = self.recycling_fraction

        # Normalize diversion fractions to sum to 1 if they exceed it
        s = sum(x for x in self.div_fractions.values())
        if  s > 1:
            for div in self.div_fractions:
                self.div_fractions[div] /= s
        assert sum(x for x in self.div_fractions.values()) <= 1, 'Diversion fractions sum to more than 1'
        # # Use IPCC defaults if no data
        # if s == 0:
        #     self.div_fractions['compost'] = defaults.fraction_composted[self.region]
        #     self.div_fractions['combustion'] = defaults.fraction_incinerated[self.region]
        
        # UN Habitat has its own data import procedure
        if self.data_source == 'UN Habitat':
            #self.changed_diversion, self.input_problems, self.div_component_fractions, self.divs = self.check_masses_un()
            
            # Determine diversion waste type fractions
            self.total_recovered_materials_with_rejects = float(row['total_recovered_materials_with_rejects'])
            self.organic_waste_recovered = float(row['organic_waste_recovered'])
            self.glass_recovered = float(row['glass_recovered'])
            self.metal_recovered = float(row['metal_recovered'])
            self.paper_or_cardboard = float(row['paper_or_cardboard'])
            self.total_plastic_recovered = float(row['total_plastic_recovered'])
            self.mixed_waste = float(row['mixed_waste'])
            self.other_waste = float(row['other_waste'])
            self.div_component_fractions, self.divs = self.determine_component_fractions_un()

            # Calculate generated waste masses
            self.waste_masses = {x: self.waste_fractions[x] * self.waste_mass for x in self.waste_fractions.keys()}
            #self.changed_diversion, self.input_problems, self.div_component_fractions, self.divs = self.check_masses(self.div_fractions, self.divs)

            # Adjust diversion waste type fractions (div_component_fractions) to make sure more waste is not diverted than generated
            self.changed_diversion, self.input_problems, self.div_component_fractions, self.divs = self.check_masses_v2(self.div_fractions, self.div_component_fractions)
        else:
            # Determine diversion waste type fractions
            self.div_component_fractions = {}
            self.divs['compost'], self.div_component_fractions['compost'] = self.calc_compost_vol(self.div_fractions['compost'])
            self.divs['anaerobic'], self.div_component_fractions['anaerobic'] = self.calc_anaerobic_vol(self.div_fractions['anaerobic'])
            self.divs['combustion'], self.div_component_fractions['combustion'] = self.calc_combustion_vol(self.div_fractions['combustion'])
            self.divs['recycling'], self.div_component_fractions['recycling'] = self.calc_recycling_vol(self.div_fractions['recycling'])

            # Fill 0s for waste types not included in diversion types
            for c in self.waste_fractions.keys():
                if c not in self.divs['compost'].keys():
                    self.divs['compost'][c] = 0
                if c not in self.divs['anaerobic'].keys():
                    self.divs['anaerobic'][c] = 0
                if c not in self.divs['combustion'].keys():
                    self.divs['combustion'][c] = 0
                if c not in self.divs['recycling'].keys():
                    self.divs['recycling'][c] = 0
            
            # Save waste diverions calculated with default assumptions, and then update them if any components are net negative.
            self.divs_before_check = copy.deepcopy(self.divs)
            self.div_component_fractions_before = copy.deepcopy(self.div_component_fractions)
            self.waste_masses = {x: self.waste_fractions[x] * self.waste_mass for x in self.waste_fractions.keys()}
            
            self.net_masses_before_check = {}
            for waste in self.waste_masses.keys():
                net_mass = self.waste_masses[waste] - (self.divs['compost'][waste] + self.divs['anaerobic'][waste] + self.divs['combustion'][waste] + self.divs['recycling'][waste])
                self.net_masses_before_check[waste] = net_mass
            
            # if self.name in city_manual_baselines.manual_cities:
            #     city_manual_baselines.get_manual_baseline(self)
            #     self.changed_diversion = True
            #     self.input_problems = False
            #     # if self.name == 'Kitakyushu':
            #     #     self.divs['combustion']['paper_cardboard'] -= 410

            #     # Inefficiency factors
            #     self.non_compostable_not_targeted_total = sum([
            #         self.non_compostable_not_targeted[x] * \
            #         self.div_component_fractions['compost'][x] for x in self.compost_components
            #     ])
                
            #     # Reduce them by non-compostable and unprocessable and etc rates
            #     for waste in self.compost_components:
            #         self.divs['compost'][waste] = (
            #             self.divs['compost'][waste]  * 
            #             (1 - self.non_compostable_not_targeted_total) *
            #             (1 - self.unprocessable[waste])
            #         )
            #     for waste in self.combustion_components:
            #         self.divs['combustion'][waste] = (
            #             self.divs['combustion'][waste]  * 
            #             (1 - self.combustion_reject_rate)
            #         )
            #     for waste in self.recycling_components:
            #         self.divs['recycling'][waste] = (
            #             self.divs['recycling'][waste]  * 
            #             self.recycling_reject_rates[waste]
            #         )

            #else:
            # Adjust diversion waste type fractions to make sure more waste is not diverted than generated
            self.changed_diversion, self.input_problems, self.div_component_fractions, self.divs = self.check_masses_v2(self.div_fractions, self.div_component_fractions)
                #self.changed_diversion, self.input_problems, self.div_component_fractions, self.divs = self.check_masses(self.div_fractions, self.divs)

        # If adjusting diversion waste types failed to prevent net negative masses (more diverted than generated),
        # terminate operation, data is invalid for model.
        if self.input_problems:
            print('input problems')
            return

        self.net_masses_after_check = {}
        for waste in self.waste_masses.keys():
            net_mass = self.waste_masses[waste] - (self.divs['compost'][waste] + self.divs['anaerobic'][waste] + self.divs['combustion'][waste] + self.divs['recycling'][waste])
            self.net_masses_after_check[waste] = net_mass

        for waste in self.net_masses_after_check.values():
            if waste < -1:
                print(waste)
            if waste <= -1:
                print('blah')
            assert waste >= -1, 'Waste diversion is net negative ' + self.name

        # Baseline refers to loaded parameters, new refers to alternative scenario parameters determined by a user.
        self.new_divs = copy.deepcopy(self.divs)
        self.new_div_fractions = copy.deepcopy(self.div_fractions)
        self.new_div_component_fractions = copy.deepcopy(self.div_component_fractions)

        self.baseline_divs = copy.deepcopy(self.divs)
        self.baseline_div_fractions = copy.deepcopy(self.div_fractions)
        self.baseline_div_component_fractions = copy.deepcopy(self.div_component_fractions)

        self.split_fractions_baseline = copy.deepcopy(self.split_fractions)
        self.landfills_baseline = copy.deepcopy(self.landfills)

        self.split_fractions_new = copy.deepcopy(self.split_fractions)
        self.landfills_new = copy.deepcopy(self.landfills)

    def _calculate_divs(self, advanced_baseline=False, advanced_dst=False) -> None:
        
        city_parameters = self.baseline_parameters
        city_parameters._singapore_k()

        # Create city-level dataframes
        start_year = 1960
        end_year = 2073
        years = range(start_year, end_year + 1)
        
        waste_masses_df = city_parameters.waste_fractions.multiply(city_parameters.waste_mass, axis=0)

        if isinstance(city_parameters.year_of_data_pop, dict):
            year_of_data_pop = city_parameters.year_of_data_pop['baseline']
        else:
            year_of_data_pop = city_parameters.year_of_data_pop

        city_parameters.waste_generated_df = WasteGeneratedDF.create(waste_masses_df, start_year, end_year, year_of_data_pop, city_parameters.growth_rate_historic, city_parameters.growth_rate_future).df
        
        # if scenario == 0:
        #     self.baseline_parameters = city_parameters
        # else:
        #     self.scenario_parameters[scenario - 1] = city_parameters

        # Update other calculated attributes
        #self._calculate_waste_masses()
        self._calculate_diverted_masses()
        #city_parameters.divs_df = DivsDF.create(city_parameters.divs, start_year, end_year, city_parameters.year_of_data_pop, city_parameters.growth_rate_historic, city_parameters.growth_rate_future)
        city_parameters.divs_df = city_parameters.divs
        self._calculate_net_masses()

        city_params_dict = self.update_cityparams_dict(city_parameters)

        if not advanced_baseline and not advanced_dst:
            landfill_w_capture = Landfill(
                open_date=1960, 
                close_date=2073, 
                site_type='landfill', 
                mcf=pd.Series(1, index=years),
                city_params_dict=city_params_dict, 
                city_instance_attrs=city_parameters.city_instance_attrs, 
                landfill_index=0, 
                fraction_of_waste=city_parameters.split_fractions.landfill_w_capture, 
                gas_capture=True
            )
            landfill_wo_capture = Landfill(
                open_date=1960, 
                close_date=2073, 
                site_type='landfill', 
                mcf=pd.Series(1, index=years), 
                city_params_dict=city_params_dict, 
                city_instance_attrs=city_parameters.city_instance_attrs, 
                landfill_index=1, 
                fraction_of_waste=city_parameters.split_fractions.landfill_wo_capture, 
                gas_capture=False,
                gas_capture_efficiency=0.0
            )
            dumpsite = Landfill(
                open_date=1960, 
                close_date=2073, 
                site_type='dumpsite', 
                mcf=pd.Series(0.4, index=years), 
                city_params_dict=city_params_dict, 
                city_instance_attrs=city_parameters.city_instance_attrs, 
                landfill_index=2, 
                fraction_of_waste=city_parameters.split_fractions.dumpsite, 
                gas_capture=False
            )

            landfills = [landfill_w_capture, landfill_wo_capture, dumpsite]
            non_zero_landfills = [x for x in [landfill_w_capture, landfill_wo_capture, dumpsite] if x.fraction_of_waste > 0]

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
        keys_to_remove = ['landfills', 'non_zero_landfills']
        for key in keys_to_remove:
            if key in city_params_dict:
                del city_params_dict[key]

        if city_parameters.landfills is not None:
            for landfill in city_parameters.landfills:
                landfill.city_params_dict = city_params_dict
                if hasattr(landfill, 'model'):
                    landfill.model.city_params_dict = city_params_dict

        return city_params_dict

    def _calculate_waste_masses(self) -> None:
        waste_masses = {waste: frac * self.baseline_parameters.waste_mass for waste, frac in self.baseline_parameters.waste_fractions.model_dump().items()}
        self.baseline_parameters.waste_masses = WasteMasses(**waste_masses)

    def _calculate_diverted_masses(self, scenario: int=0) -> None:
        """
        Calculate the diverted masses of different types of waste.

        Args:
            scenario (int): The scenario number to use (0 for baseline, or the number of the alternative scenario).
        """
        if scenario == 0:
            parameters = self.baseline_parameters
        else:
            parameters = self.scenario_parameters.get(scenario-1)
            if parameters is None:
                raise ValueError(f"Scenario '{scenario}' not found in scenario_parameters.")

        diverted_masses = {}

        # if isinstance(parameters.div_fractions.combustion, float):
        #     for div in parameters.div_component_fractions.model_fields:
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
            years = range(1960, 2074)
            compost_dict = div_component_fractions.compost.model_dump()
            compost = pd.DataFrame(compost_dict, index=years)[list(self.div_components['compost'])]
            anaerobic_dict = div_component_fractions.anaerobic.model_dump()
            anaerobic = pd.DataFrame(anaerobic_dict, index=years)[list(self.div_components['anaerobic'])]
            combustion_dict = div_component_fractions.combustion.model_dump()
            combustion = pd.DataFrame(combustion_dict, index=years)[list(self.div_components['combustion'])]
            recycling_dict = div_component_fractions.recycling.model_dump()
            recycling = pd.DataFrame(recycling_dict, index=years)[list(self.div_components['recycling'])]
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
                columns=div_dict.keys()
            )
            parameters.div_fractions = df

        for div in parameters.div_component_fractions.model_fields:
            # Get the component fractions for the current diversion type
            fracs = getattr(parameters.div_component_fractions, div)
            s = fracs.sum(axis=1).iat[0]
        
            # Ensure that the component fractions add up to 1 for each year
            if not (np.allclose(s, 1, atol=0.01) or np.all(s == 0)):
                print(f"Problems with {div}: Fractions do not sum to 1 across years.")

            # Calculate the diverted masses for each waste type
            try:
                diverted_masses[div] = fracs.multiply(parameters.div_fractions.multiply(parameters.waste_generated_df.sum(axis=1), axis=0)[div], axis=0)[list(self.div_components[div])]
            except:
                diverted_masses[div] = fracs.multiply(getattr(parameters.div_fractions, div) * parameters.waste_generated_df.sum(axis=1), axis=0)[list(self.div_components[div])]


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
        diverted_masses['compost'] = diverted_masses['compost'].multiply(
            (1 - parameters.non_compostable_not_targeted_total), axis=0
        ).multiply(
            (1 - pd.Series(self.unprocessable)), axis=1
        )
        diverted_masses['combustion'] *= (1 - self.combustion_reject_rate)
        for waste in diverted_masses['recycling'].columns:
            diverted_masses['recycling'][waste] *= self.recycling_reject_rates[waste]

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
            compost=diverted_masses['compost'],
            anaerobic=diverted_masses['anaerobic'],
            combustion=diverted_masses['combustion'],
            recycling=diverted_masses['recycling']
        )

        # Save the results in the correct attribute
        parameters.divs = divs

    def dst_baseline_blank(
        self, 
        country: str, 
        population: int, 
        precipitation: float,
        temperature: float
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
        years = range(1960, 2074)
        
        # Calculate growth rates
        population_1950 = 751_000_000
        population_2020 = 4_300_000_000
        population_2035 = 5_300_000_000
        growth_rate_historic = (population_2020 / population_1950) ** (1 / (2020 - 1950))
        growth_rate_future = (population_2035 / population_2020) ** (1 / (2035 - 2020))

        # Calculate waste per capita
        waste_per_capita = defaults_2019.msw_per_capita_country.get(
            iso3, 
            defaults_2019.msw_per_capita_defaults.get(region, 0)
        )
        waste_mass = waste_per_capita * population / 1000 * 365  # in tons/year

        # Retrieve and normalize waste fractions
        if iso3 in defaults_2019.waste_fractions_country:
            waste_fractions_series = defaults_2019.waste_fractions_country.loc[iso3, :]
        else:
            waste_fractions_series = defaults_2019.waste_fraction_defaults.loc[region, :]

        waste_fractions_normalized = waste_fractions_series / waste_fractions_series.sum()
        waste_fractions = WasteFractions(**waste_fractions_normalized.to_dict())
        waste_fractions_df = pd.DataFrame(
            [waste_fractions_normalized] * len(years),
            index=years
        )

        year_of_data_pop = 2022

        # Calculate MEF for compost
        try:
            food_frac = waste_fractions_normalized['food']
            green_frac = waste_fractions_normalized['green']
            mef_compost = (
                (0.0055 * food_frac / (food_frac + green_frac) + 
                 0.0139 * green_frac / (food_frac + green_frac)) * 
                1.1023 * 0.7
            )
        except:
            mef_compost = 0.0

        # Get decomposition rates
        ks = defaults_2019.k_defaults.get(precip_zone, None)
        ks = DecompositionRates(
            food=pd.Series(ks.get('food', 0.0), index=years),
            green=pd.Series(ks.get('green', 0.0), index=years),
            wood=pd.Series(ks.get('wood', 0.0), index=years),
            paper_cardboard=pd.Series(ks.get('paper_cardboard', 0.0), index=years),
            textiles=pd.Series(ks.get('textiles', 0.0), index=years)
        )

        # Determine waste split fractions using .get() method
        dumpsite_frac = defaults_2019.fraction_open_dumped_country.get(
            iso3, 
            defaults_2019.fraction_open_dumped.get(region, 0)
        )
        landfill_wo_capture_frac = defaults_2019.fraction_landfilled_country.get(
            iso3, 
            defaults_2019.fraction_landfilled.get(region, 0)
        )
        landfill_w_capture_frac = 0.0  # Default as per original function

        try:
            split_fractions = SplitFractions(
                dumpsite=dumpsite_frac,
                landfill_wo_capture=landfill_wo_capture_frac,
                landfill_w_capture=landfill_w_capture_frac
            )
        except KeyError:
            if self.region in defaults_2019.landfill_default_regions:
                split_fractions = SplitFractions(
                    landfill_w_capture=0.0, 
                    landfill_wo_capture=1.0, 
                    dumpsite=0.0
                )
            else:
                split_fractions = SplitFractions(
                    landfill_w_capture=0.0, 
                    landfill_wo_capture=0.0, 
                    dumpsite=1.0
                )
        
        # Normalize split fractions
        split_total = sum(split_fractions.model_dump().values())
        if split_total > 0:
            split_fractions = SplitFractions(**{
                site: frac / split_total for site, frac in split_fractions.model_dump().items()
            })
        
        # Instantiate landfill objects
        years_range = range(1960, 2074)
        city_instance_attrs = {
            'city_name': self.city_name,
            'country': country,
            'components': self.components,
            'div_components': self.div_components,
            'waste_types': self.waste_types,
            'unprocessable': self.unprocessable,
            'non_compostable_not_targeted': self.non_compostable_not_targeted,
            'combustion_reject_rate': self.combustion_reject_rate,
            'recycling_reject_rates': self.recycling_reject_rates
        }
        city_params_dict = {}  # Define appropriately or pass as needed

        # Diversion fractions
        compost_frac = defaults_2019.fraction_composted_country.get(
            iso3, 
            defaults_2019.fraction_composted.get(region, 0.0)
        )
        combustion_frac = defaults_2019.fraction_incinerated_country.get(
            iso3, 
            defaults_2019.fraction_incinerated.get(region, 0.0)
        )

        # Create DataFrame with the same values for all years
        div_fractions = pd.DataFrame({
            'compost': compost_frac,
            'anaerobic': 0.0,
            'combustion': combustion_frac,
            'recycling': 0.0
        }, index=years)

        def calculate_component_fractions(waste_fractions: WasteFractions, div_type: str) -> WasteFractions:
            components = self.div_components[div_type]
            filtered_fractions = {waste: getattr(waste_fractions, waste) for waste in components}
            total = sum(filtered_fractions.values())
            normalized_fractions = {waste: fraction / total for waste, fraction in filtered_fractions.items()}
            return WasteFractions(**{waste: normalized_fractions.get(waste, 0) for waste in waste_fractions.model_fields})

        div_component_fractions = DivComponentFractions(
            compost=calculate_component_fractions(waste_fractions, 'compost'),
            anaerobic=calculate_component_fractions(waste_fractions, 'anaerobic'),
            combustion=calculate_component_fractions(waste_fractions, 'combustion'),
            recycling=calculate_component_fractions(waste_fractions, 'recycling'),
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
        non_compostable_not_targeted_total = sum([
            self.non_compostable_not_targeted[x] * div_component_fractions.model_dump().get('compost', {}).get(x, 0.0)
            for x in self.div_components['compost']
        ])
        non_compostable_not_targeted_total = pd.Series(non_compostable_not_targeted_total, index=years)
        if non_compostable_not_targeted_total.isna().all():
            non_compostable_not_targeted_total = pd.Series(0, index=years)

        # Create gas_capture_efficiency Series
        gas_capture_efficiency_value = 0.6
        gas_capture_efficiency_series = pd.Series(gas_capture_efficiency_value, index=years_range)

        waste_masses = WasteMasses(**(waste_fractions_normalized * waste_mass).to_dict())
        waste_masses_df = waste_fractions_df * waste_mass
        waste_generated_df = WasteGeneratedDF.create(
            waste_masses_df,
            1960, 
            2073, 
            year_of_data_pop, 
            growth_rate_historic, 
            growth_rate_future
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
            ks=ks
        )
        self.baseline_parameters = baseline

        # Check masses consistency
        self._check_masses_v2(scenario=0)
        if baseline.input_problems:
            print('Input problems detected in baseline parameters.')
            return
        
        self._calculate_net_masses()
        if (baseline.net_masses < 0).any().any():
            print(f'Invalid new value')
            return
        
        # Assign the baseline parameters to the city instance
        #baseline.repopulate_attr_dicts()
        self._calculate_divs()

        # landfill_w_capture = Landfill(
        #     open_date=1960,
        #     close_date=2073,
        #     site_type='landfill',
        #     mcf=pd.Series(1.0, index=years_range),
        #     city_params_dict=city_params_dict,
        #     city_instance_attrs=city_instance_attrs,
        #     landfill_index=0,
        #     fraction_of_waste=split_fractions.landfill_w_capture,
        #     gas_capture=True
        # )
        # landfill_wo_capture = Landfill(
        #     open_date=1960,
        #     close_date=2073,
        #     site_type='landfill',
        #     mcf=pd.Series(1.0, index=years_range),
        #     city_params_dict=city_params_dict,
        #     city_instance_attrs=city_instance_attrs,
        #     landfill_index=1,
        #     fraction_of_waste=split_fractions.landfill_wo_capture,
        #     gas_capture=False
        # )
        # dumpsite = Landfill(
        #     open_date=1960,
        #     close_date=2073,
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
        for landfill in baseline.non_zero_landfills:
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
        growth_rate_override: float
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
            "scenario": waste_mass_year.scenario
        }

        if year_of_data_pop['scenario'] is None:
            year_of_data_pop['scenario'] = year_of_data_pop['baseline']

        # Calculate MEF for compost
        try:
            # 0 is food, 1 is green
            food_frac = waste_fractions.baseline[0]
            green_frac = waste_fractions.baseline[1]
            mef_compost = (
                (0.0055 * food_frac / (food_frac + green_frac) + 
                 0.0139 * green_frac / (food_frac + green_frac)) * 
                1.1023 * 0.7
            )
        except:
            mef_compost = 0.0

        city_instance_attrs = {
            'city_name': self.city_name,
            'country': country,
            'components': self.components,
            'div_components': self.div_components,
            'waste_types': self.waste_types,
            'unprocessable': self.unprocessable,
            'non_compostable_not_targeted': self.non_compostable_not_targeted,
            'combustion_reject_rate': self.combustion_reject_rate,
            'recycling_reject_rates': self.recycling_reject_rates
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
        compost_total = compost_fraction * self.baseline_parameters.waste_mass
        fraction_compostable_types = sum([self.baseline_parameters.waste_fractions.model_dump()[x] for x in self.div_components['compost']])
        
        if compost_fraction != 0:
            compost_waste_fractions = {x: self.baseline_parameters.waste_fractions.model_dump()[x] / fraction_compostable_types for x in self.div_components['compost']}
            non_compostable_not_targeted = {'food': .1, 'green': .05, 'wood': .05, 'paper_cardboard': .1}
            non_compostable_not_targeted_total = sum([non_compostable_not_targeted[x] * compost_waste_fractions[x] for x in self.div_components['compost']])

            compost = {}
            if new and sum(self.baseline_parameters.div_component_fractions.compost.model_dump().values()) != 0:
                for waste in self.div_components['compost']:
                    compost[waste] = (
                        compost_total * 
                        (1 - non_compostable_not_targeted_total) *
                        self.baseline_parameters.div_component_fractions.compost.model_dump()[waste] *
                        (1 - self.unprocessable[waste])
                    )
                compost_waste_fractions = self.baseline_parameters.div_component_fractions.compost
            else:
                for waste in self.div_components['compost']:
                    compost[waste] = (
                        compost_total * 
                        (1 - non_compostable_not_targeted_total) *
                        compost_waste_fractions[waste] *
                        (1 - self.unprocessable[waste])
                    )
        else:
            compost = {x: 0 for x in self.div_components['compost']}
            compost_waste_fractions = {x: 0 for x in self.div_components['compost']}
            non_compostable_not_targeted = {'food': 0, 'green': 0, 'wood': 0, 'paper_cardboard': 0}
            non_compostable_not_targeted_total = 0
            
        self.compost_total = compost_total
        self.fraction_compostable_types = fraction_compostable_types
        #self.non_compostable_not_targeted = non_compostable_not_targeted

        return compost, compost_waste_fractions

    def _calc_anaerobic_vol(self, anaerobic_fraction: float, new: bool = False) -> tuple:
        anaerobic_total = anaerobic_fraction * self.baseline_parameters.waste_mass
        fraction_anaerobic_types = sum([self.baseline_parameters.waste_fractions.model_dump()[x] for x in self.div_components['anaerobic']])
        
        if anaerobic_fraction != 0:
            anaerobic_waste_fractions = {x: self.baseline_parameters.waste_fractions.model_dump()[x] / fraction_anaerobic_types for x in self.div_components['anaerobic']}
            
            if new and sum(self.baseline_parameters.div_component_fractions.anaerobic.model_dump().values()) != 0:
                anaerobic = {x: anaerobic_total * self.baseline_parameters.div_component_fractions.anaerobic.model_dump()[x] for x in self.div_components['anaerobic']}
                anaerobic_waste_fractions = self.baseline_parameters.div_component_fractions.anaerobic
            else:
                anaerobic = {x: anaerobic_total * anaerobic_waste_fractions[x] for x in self.div_components['anaerobic']}
        else:
            anaerobic = {x: 0 for x in self.div_components['anaerobic']}
            anaerobic_waste_fractions = {x: 0 for x in self.div_components['anaerobic']}
        
        self.anaerobic_total = anaerobic_total
        return anaerobic, anaerobic_waste_fractions

    def _calc_combustion_vol(self, combustion_fraction: float, new: bool = False) -> tuple:
        combustion_total = combustion_fraction * self.baseline_parameters.waste_mass
        fraction_combustion_types = sum([self.baseline_parameters.waste_fractions.model_dump()[x] for x in self.div_components['combustion']])
        combustion_waste_fractions = {x: self.baseline_parameters.waste_fractions.model_dump()[x] / fraction_combustion_types for x in self.div_components['combustion']}
        
        if new and sum(self.baseline_parameters.div_component_fractions.combustion.model_dump().values()) != 0:
            combustion = {x: combustion_total * self.baseline_parameters.div_component_fractions.combustion.model_dump()[x] * (1 - self.combustion_reject_rate) for x in self.div_components['combustion']}
            combustion_waste_fractions = self.baseline_parameters.div_component_fractions.combustion
        else:
            combustion = {x: combustion_total * combustion_waste_fractions[x] * (1 - self.combustion_reject_rate) for x in self.div_components['combustion']}

        return combustion, combustion_waste_fractions

    def _calc_recycling_vol(self, recycling_fraction: float, new: bool = False) -> tuple:
        recycling_total = recycling_fraction * self.baseline_parameters.waste_mass
        fraction_recyclable_types = sum([self.baseline_parameters.waste_fractions.model_dump()[x] for x in self.div_components['recycling']])
        recycling_reject_rates = self.recycling_reject_rates
        
        if recycling_fraction != 0:
            recycling_waste_fractions = {x: self.baseline_parameters.waste_fractions.model_dump()[x] / fraction_recyclable_types for x in self.div_components['recycling']}
            
            if new and sum(self.baseline_parameters.div_component_fractions.recycling.model_dump().values()) != 0:
                recycling = {x: recycling_total * self.baseline_parameters.div_component_fractions.recycling.model_dump()[x] * recycling_reject_rates[x] for x in self.div_components['recycling']}
                recycling_waste_fractions = self.baseline_parameters.div_component_fractions.recycling
            else:
                recycling = {x: recycling_total * recycling_waste_fractions[x] * recycling_reject_rates[x] for x in self.div_components['recycling']}
        else:
            recycling = {x: 0 for x in self.div_components['recycling']}
            recycling_waste_fractions = {x: 0 for x in self.div_components['recycling']}
        
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
        anaerobic_emissions = parameters.divs_df.anaerobic * defaults_2019.mef_anaerobic * defaults_2019.ch4_to_co2e

        parameters.organic_emissions = compost_emissions.add(anaerobic_emissions, fill_value=0)

    def sum_landfill_emissions(self, scenario: int, simple=False) -> None:
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
            #landfill_emissions = [x.emissions.map(self.convert_methane_m3_to_ton_co2e) for x in parameters.non_zero_landfills]
            years_union = parameters.non_zero_landfills[0].emissions.index
            # Union the index of each subsequent landfill with the years_union
            for x in parameters.non_zero_landfills[1:]:
                years_union = years_union.union(x.emissions.index)
            landfill_emissions_list = [
                x.emissions.reindex(years_union, fill_value=0).map(self.convert_methane_m3_to_ton_co2e) / 28
                for x in parameters.non_zero_landfills
            ]
        elif simple:
            parameters = self.scenario_parameters[scenario - 1]
            organic_emissions = parameters.organic_emissions
            #landfill_emissions = [x.emissions.map(self.convert_methane_m3_to_ton_co2e) for x in parameters.landfills]
            years_union = parameters.landfills[0].emissions.index
            # Union the index of each subsequent landfill with the years_union
            for x in parameters.landfills[1:]:
                years_union = years_union.union(x.emissions.index)
            landfill_emissions_list = [
                x.emissions.reindex(years_union, fill_value=0).map(self.convert_methane_m3_to_ton_co2e) / 28
                for x in parameters.landfills
            ]
        else:
            parameters = self.scenario_parameters[scenario - 1]
            organic_emissions = parameters.organic_emissions
            #landfill_emissions = [x.emissions.map(self.convert_methane_m3_to_ton_co2e) for x in parameters.landfills]
            years_union = parameters.landfills[0].emissions.index
            # Union the index of each subsequent landfill with the years_union
            for x in parameters.landfills[1:]:
                years_union = years_union.union(x.emissions.index)
            landfill_emissions_list = [
                x.emissions.reindex(years_union, fill_value=0).map(self.convert_methane_m3_to_ton_co2e) / 28
                for x in parameters.landfills
            ]

        # Concatenate all emissions dataframes
        #all_emissions = sum(landfill_emissions)

        # Reindex each landfill DataFrame to the full range of years, filling missing values with zeros
        # landfill_emissions = [
        #     x.emissions.reindex(years_union, fill_value=0).map(self.convert_methane_m3_to_ton_co2e) 
        #     for x in parameters.landfills
        # ]

        # Sum the emissions dataframes
        summed_landfill_emissions = sum(landfill_emissions_list)

        # Group by the year index and sum the emissions for each year
        #summed_landfill_emissions = all_emissions.groupby(all_emissions.index).sum()

        # # Remove total
        summed_landfill_emissions.drop('total', axis=1, inplace=True)

        #summed_diversion_emissions = organic_emissions.loc[:, list(self.components)] / 28
        summed_diversion_emissions = organic_emissions.reindex(columns=summed_landfill_emissions.columns, fill_value=0) / 28

        # Repeat with addition of diverted waste emissions
        summed_emissions = sum([summed_landfill_emissions.loc[:, list(self.components)], summed_diversion_emissions.loc[summed_landfill_emissions.index, :]])
        #summed_emissions = all_emissions.groupby(all_emissions.index).sum()
        #summed_emissions.drop('total', axis=1, inplace=True)
        #summed_emissions /= 28

        summed_landfill_emissions['total'] = summed_landfill_emissions.sum(axis=1)
        summed_diversion_emissions['total'] = summed_diversion_emissions.sum(axis=1)
        summed_emissions['total'] = summed_emissions.sum(axis=1)

        parameters.landfill_emissions = summed_landfill_emissions
        parameters.diversion_emissions = summed_diversion_emissions
        parameters.total_emissions = summed_emissions

    def _check_masses_v2(self, scenario: int, advanced_baseline: bool=False, advanced_dst: bool=False, implement_year: int=None) -> None:
        """
        Adjusts diversion waste type fractions if more of a waste type is being diverted than generated.

        Args:
            scenario (int): Scenario index.
        """
        if scenario == 0:
            parameters = self.baseline_parameters
        else:
            parameters = self.scenario_parameters[scenario-1]

        if (not advanced_baseline) and (not advanced_dst):
            if isinstance(parameters.div_fractions, pd.DataFrame):
                diversion_fractions_instance = DiversionFractions(
                    compost=parameters.div_fractions.at[2000, 'compost'],
                    anaerobic=parameters.div_fractions.at[2000, 'anaerobic'],
                    combustion=parameters.div_fractions.at[2000, 'combustion'],
                    recycling=parameters.div_fractions.at[2000, 'recycling']
                )
            else:
                diversion_fractions_instance = DiversionFractions(
                    compost=parameters.div_fractions.compost,
                    anaerobic=parameters.div_fractions.anaerobic,
                    combustion=parameters.div_fractions.combustion,
                    recycling=parameters.div_fractions.recycling
                )
            div_component_fractions_instance = DivComponentFractions(
                compost=WasteFractions(**parameters.div_component_fractions.compost.loc[2000, :]),
                anaerobic=WasteFractions(**parameters.div_component_fractions.anaerobic.loc[2000, :]),
                combustion=WasteFractions(**parameters.div_component_fractions.combustion.loc[2000, :]),
                recycling=WasteFractions(**parameters.div_component_fractions.recycling.loc[2000, :])
            )
            waste_fractions_instance = WasteFractions(**parameters.waste_fractions.loc[2000, :])
            parameters.adjusted_diversion_constituents, parameters.input_problems, parameters.divs, parameters.div_component_fractions = self.mass_checker_math(
                div_fractions=diversion_fractions_instance,
                div_component_fractions=div_component_fractions_instance,
                waste_fractions=waste_fractions_instance,
                scenario=scenario
            )

            return

        else:
            unique_divsets = parameters.div_fractions.drop_duplicates()

            # Initialize empty DataFrames to build the final instances
            div_masses_df = DivsDF(
                compost = pd.DataFrame(index=parameters.div_fractions.index, columns=list(self.waste_types)),
                anaerobic = pd.DataFrame(index=parameters.div_fractions.index, columns=list(self.waste_types)),
                combustion = pd.DataFrame(index=parameters.div_fractions.index, columns=list(self.waste_types)),
                recycling = pd.DataFrame(index=parameters.div_fractions.index, columns=list(self.waste_types))
            )
            div_component_fractions_df = DivComponentFractionsDF(
                compost = pd.DataFrame(index=parameters.div_fractions.index, columns=parameters.div_component_fractions.compost.columns),
                anaerobic = pd.DataFrame(index=parameters.div_fractions.index, columns=parameters.div_component_fractions.anaerobic.columns),
                combustion = pd.DataFrame(index=parameters.div_fractions.index, columns=parameters.div_component_fractions.combustion.columns),
                recycling = pd.DataFrame(index=parameters.div_fractions.index, columns=parameters.div_component_fractions.recycling.columns)
            )

            for i, row in unique_divsets.iterrows():
                corresponding_year = row.name
                next_year = unique_divsets.index[i + 1] if i + 1 < len(unique_divsets) else parameters.div_fractions.index[-1] + 1
                year_range = range(corresponding_year, next_year)

                diversion_fractions_instance = DiversionFractions(
                    compost=parameters.div_fractions.at[corresponding_year, 'compost'],
                    anaerobic=parameters.div_fractions.at[corresponding_year, 'anaerobic'],
                    combustion=parameters.div_fractions.at[corresponding_year, 'combustion'],
                    recycling=parameters.div_fractions.at[corresponding_year, 'recycling']
                )
                div_component_fractions_instance = DivComponentFractions(
                    compost=WasteFractions(**parameters.div_component_fractions.compost.loc[corresponding_year, :].to_dict()),
                    anaerobic=WasteFractions(**parameters.div_component_fractions.anaerobic.loc[corresponding_year, :].to_dict()),
                    combustion=WasteFractions(**parameters.div_component_fractions.combustion.loc[corresponding_year, :].to_dict()),
                    recycling=WasteFractions(**parameters.div_component_fractions.recycling.loc[corresponding_year, :].to_dict())
                )
                waste_fractions_instance = WasteFractions(**parameters.waste_fractions.loc[corresponding_year, :].to_dict())
                parameters.adjusted_diversion_constituents, parameters.input_problems, divs, div_component_fractions = self.mass_checker_math(
                    div_fractions=diversion_fractions_instance,
                    div_component_fractions=div_component_fractions_instance,
                    waste_fractions=waste_fractions_instance,
                    scenario=scenario,
                    corresponding_year=corresponding_year,
                )
                
                # Populate the DataFrames for all years in the range
                # Check at some point if this is actually working right.
                #div_masses_df.loc[year_range, :] = pd.DataFrame([divs] * len(year_range), index=year_range)

                if isinstance(divs, dict):
                    if implement_year in year_range:
                        year_ranges = {}
                        year_ranges['baseline'] = range(year_range[0], implement_year)
                        year_ranges['scenario'] = range(implement_year, year_range[-1] + 1)
                        div_masses_df_split_up = {
                            'baseline': DivsDF(
                                compost = pd.DataFrame(index=year_ranges['baseline'], columns=list(self.waste_types)),
                                anaerobic = pd.DataFrame(index=year_ranges['baseline'], columns=list(self.waste_types)),
                                combustion = pd.DataFrame(index=year_ranges['baseline'], columns=list(self.waste_types)),
                                recycling = pd.DataFrame(index=year_ranges['baseline'], columns=list(self.waste_types))
                            ),
                            'scenario': DivsDF(
                                compost = pd.DataFrame(index=year_ranges['scenario'], columns=list(self.waste_types)),
                                anaerobic = pd.DataFrame(index=year_ranges['scenario'], columns=list(self.waste_types)),
                                combustion = pd.DataFrame(index=year_ranges['scenario'], columns=list(self.waste_types)),
                                recycling = pd.DataFrame(index=year_ranges['scenario'], columns=list(self.waste_types))
                            )
                        }
                        for period in ['baseline', 'scenario']:
                            div_masses_df_split_up[period].compost.loc[year_ranges[period], :] = pd.DataFrame([divs[period].compost.model_dump()] * len(year_ranges[period]), index=year_ranges[period])
                            div_masses_df_split_up[period].anaerobic.loc[year_ranges[period], :] = pd.DataFrame([divs[period].anaerobic.model_dump()] * len(year_ranges[period]), index=year_ranges[period])
                            div_masses_df_split_up[period].combustion.loc[year_ranges[period], :] = pd.DataFrame([divs[period].combustion.model_dump()] * len(year_ranges[period]), index=year_ranges[period])
                            div_masses_df_split_up[period].recycling.loc[year_ranges[period], :] = pd.DataFrame([divs[period].recycling.model_dump()] * len(year_ranges[period]), index=year_ranges[period])

                        div_masses_df.compost.loc[year_range, :] = pd.concat([div_masses_df_split_up['baseline'].compost, div_masses_df_split_up['scenario'].compost])
                        div_masses_df.anaerobic.loc[year_range, :] = pd.concat([div_masses_df_split_up['baseline'].anaerobic, div_masses_df_split_up['scenario'].anaerobic])
                        div_masses_df.combustion.loc[year_range, :] = pd.concat([div_masses_df_split_up['baseline'].combustion, div_masses_df_split_up['scenario'].combustion])
                        div_masses_df.recycling.loc[year_range, :] = pd.concat([div_masses_df_split_up['baseline'].recycling, div_masses_df_split_up['scenario'].recycling])
                    elif implement_year < year_range[0]:
                        div_masses_df.compost.loc[year_range, :] = pd.DataFrame([divs['baseline'].compost] * len(year_range), index=year_range)
                        div_masses_df.anaerobic.loc[year_range, :] = pd.DataFrame([divs['baseline'].anaerobic] * len(year_range), index=year_range)
                        div_masses_df.combustion.loc[year_range, :] = pd.DataFrame([divs['baseline'].combustion] * len(year_range), index=year_range)
                        div_masses_df.recycling.loc[year_range, :] = pd.DataFrame([divs['baseline'].recycling] * len(year_range), index=year_range)
                    else:
                        div_masses_df.compost.loc[year_range, :] = pd.DataFrame([divs['scenario'].compost] * len(year_range), index=year_range)
                        div_masses_df.anaerobic.loc[year_range, :] = pd.DataFrame([divs['scenario'].anaerobic] * len(year_range), index=year_range)
                        div_masses_df.combustion.loc[year_range, :] = pd.DataFrame([divs['scenario'].combustion] * len(year_range), index=year_range)
                        div_masses_df.recycling.loc[year_range, :] = pd.DataFrame([divs['scenario'].recycling] * len(year_range), index=year_range)            

                    div_component_fractions_df.compost.loc[year_range, :] = pd.DataFrame([div_component_fractions.compost.model_dump()] * len(year_range), index=year_range)
                    div_component_fractions_df.anaerobic.loc[year_range, :] = pd.DataFrame([div_component_fractions.anaerobic.model_dump()] * len(year_range), index=year_range)
                    div_component_fractions_df.combustion.loc[year_range, :] = pd.DataFrame([div_component_fractions.combustion.model_dump()] * len(year_range), index=year_range)
                    div_component_fractions_df.recycling.loc[year_range, :] = pd.DataFrame([div_component_fractions.recycling.model_dump()] * len(year_range), index=year_range)
                else:
                    try:
                        div_masses_df['compost'].loc[year_range, :] = pd.DataFrame([divs['compost']] * len(year_range), index=year_range)
                        div_masses_df['anaerobic'].loc[year_range, :] = pd.DataFrame([divs['anaerobic']] * len(year_range), index=year_range)
                        div_masses_df['combustion'].loc[year_range, :] = pd.DataFrame([divs['combustion']] * len(year_range), index=year_range)
                        div_masses_df['recycling'].loc[year_range, :] = pd.DataFrame([divs['recycling']] * len(year_range), index=year_range)

                        div_component_fractions_df['compost'].loc[year_range, :] = pd.DataFrame([div_component_fractions['compost']] * len(year_range), index=year_range)
                        div_component_fractions_df['anaerobic'].loc[year_range, :] = pd.DataFrame([div_component_fractions['anaerobic']] * len(year_range), index=year_range)
                        div_component_fractions_df['combustion'].loc[year_range, :] = pd.DataFrame([div_component_fractions['combustion']] * len(year_range), index=year_range)
                        div_component_fractions_df['recycling'].loc[year_range, :] = pd.DataFrame([div_component_fractions['recycling']]* len(year_range), index=year_range)
                    except:
                        div_masses_df.compost.loc[year_range, :] = pd.DataFrame([divs.compost.model_dump()] * len(year_range), index=year_range)
                        div_masses_df.anaerobic.loc[year_range, :] = pd.DataFrame([divs.anaerobic.model_dump()] * len(year_range), index=year_range)
                        div_masses_df.combustion.loc[year_range, :] = pd.DataFrame([divs.combustion.model_dump()] * len(year_range), index=year_range)
                        div_masses_df.recycling.loc[year_range, :] = pd.DataFrame([divs.recycling.model_dump()] * len(year_range), index=year_range)
                    
                        div_component_fractions_df.compost.loc[year_range, :] = pd.DataFrame([div_component_fractions.compost.model_dump()] * len(year_range), index=year_range)
                        div_component_fractions_df.anaerobic.loc[year_range, :] = pd.DataFrame([div_component_fractions.anaerobic.model_dump()] * len(year_range), index=year_range)
                        div_component_fractions_df.combustion.loc[year_range, :] = pd.DataFrame([div_component_fractions.combustion.model_dump()] * len(year_range), index=year_range)
                        div_component_fractions_df.recycling.loc[year_range, :] = pd.DataFrame([div_component_fractions.recycling.model_dump()] * len(year_range), index=year_range)
            
            # Make sure at some point this doesn't do crazy shit when implement_year is early or something
            # if isinstance(parameters.waste_mass, dict):
            #     ratio = parameters.waste_mass['scenario'] / parameters.waste_mass['baseline']
            #     div_masses_df.compost.loc[parameters.implement_year:, :] *= ratio
            #     div_masses_df.anaerobic.loc[parameters.implement_year:, :] *= ratio
            #     div_masses_df.recycling.loc[parameters.implement_year:, :] *= ratio
            #     div_masses_df.combustion.loc[parameters.implement_year:, :] *= ratio
            
            try:
                # Create the final instances
                final_div_masses_annual = DivMassesAnnual(
                    compost=div_masses_df['compost'],
                    anaerobic=div_masses_df['anaerobic'],
                    combustion=div_masses_df['combustion'],
                    recycling=div_masses_df['recycling']
                )

                final_div_component_fractions_df = DivComponentFractionsDF(
                    compost=div_component_fractions_df['compost'],
                    anaerobic=div_component_fractions_df['anaerobic'],
                    combustion=div_component_fractions_df['combustion'],
                    recycling=div_component_fractions_df['recycling']
                )
            except:
                # Create the final instances
                final_div_masses_annual = DivMassesAnnual(
                    compost=div_masses_df.compost.loc[:, list(self.div_components['compost'])],
                    anaerobic=div_masses_df.anaerobic.loc[:, list(self.div_components['anaerobic'])],
                    combustion=div_masses_df.combustion.loc[:, list(self.div_components['combustion'])],
                    recycling=div_masses_df.recycling.loc[:, list(self.div_components['recycling'])]
                )

                final_div_component_fractions_df = DivComponentFractionsDF(
                    compost=div_component_fractions_df.compost.loc[:, list(self.div_components['compost'])],
                    anaerobic=div_component_fractions_df.anaerobic.loc[:, list(self.div_components['anaerobic'])],
                    combustion=div_component_fractions_df.combustion.loc[:, list(self.div_components['combustion'])],
                    recycling=div_component_fractions_df.recycling.loc[:, list(self.div_components['recycling'])]
                )
            
            parameters.divs = final_div_masses_annual
            parameters.div_component_fractions = final_div_component_fractions_df
            
            return
        
    def mass_checker_math(self, div_fractions: DiversionFractions, div_component_fractions: DivComponentFractions, waste_fractions: WasteFractions, scenario: int, corresponding_year: int=2000) -> tuple:
        components_multiplied_through = {}
        for div in div_component_fractions.model_fields:
            components_multiplied_through[div] = {}
            for waste in getattr(div_component_fractions, div).model_fields:
                components_multiplied_through[div][waste] = getattr(div_fractions, div) * getattr(getattr(div_component_fractions, div), waste)

        net = {}
        negative_catcher = False
        for waste in waste_fractions.model_fields:
            s = sum(components_multiplied_through[div].get(waste, 0) for div in div_fractions.model_fields)
            net[waste] = getattr(waste_fractions, waste) - s
            if net[waste] < -1e-3:
                negative_catcher = True

        if not negative_catcher:
            #divs = self._divs_from_component_fractions(div_fractions, div_component_fractions, scenario=scenario)
            #parameters.divs = divs
            adjusted_diversion_constituents = False
            input_problems = False
            divs = self._divs_from_component_fractions(div_fractions, div_component_fractions, scenario=scenario, advanced=True, year=corresponding_year)
            return adjusted_diversion_constituents, input_problems, divs, div_component_fractions

        if sum(getattr(div_fractions, div) for div in div_fractions.model_fields) > 1:
            raise CustomError("INVALID_PARAMETERS", f"Diversions sum to {sum(getattr(div_fractions, div) for div in div_fractions.model_fields)}, but they must sum to 1 or less.")
        
        compostables = sum(getattr(waste_fractions, waste) for waste in ['food', 'green', 'wood', 'paper_cardboard'])
        if div_fractions.compost + div_fractions.anaerobic > compostables:
            raise CustomError("INVALID_PARAMETERS", f"Only food, green, wood, and paper/cardboard can be composted or anaerobically digested. Those waste types sum to {compostables}, but input values of compost and anaerobic digestion sum to {div_fractions.compost + div_fractions.anaerobic}.")

        for div in div_fractions.model_fields:
            fraction = getattr(div_fractions, div)
            s = sum(getattr(waste_fractions, waste) for waste in self.div_components[div])
            if s < fraction:
                components = self.div_components[div]
                values = [getattr(waste_fractions, x) for x in components]
                raise CustomError("INVALID_PARAMETERS", f"{div} too high. {div} applies to {components}, which are {values} of total waste--the sum of these is {sum(values)}, so only that much waste can be {div}, but input value was {fraction}.")

        non_combustables = sum(getattr(waste_fractions, waste) for waste in ['glass', 'metal', 'other'])
        if div_fractions.compost + div_fractions.anaerobic + div_fractions.combustion > (1 - non_combustables):
            s = div_fractions.compost + div_fractions.anaerobic + div_fractions.combustion
            raise CustomError("INVALID_PARAMETERS", f"Glass, metal, and other account for {non_combustables:.3f} of waste, and they can only be recycled. {div_fractions.compost} compost, {div_fractions.anaerobic} anaerobic, and {div_fractions.combustion} incineration were specified, summing to {s}, but only {1 - non_combustables} of waste can be diverted to these diversion types.")

        non_combustion = {}
        combustion_all = {}
        keys_of_interest = ['compost', 'anaerobic', 'recycling']
        for waste in waste_fractions.model_fields:
            s = sum(components_multiplied_through[div].get(waste, 0) for div in keys_of_interest)
            non_combustion[waste] = s
            combustion_all[waste] = getattr(waste_fractions, waste) - s

        adjust_non_combustion = False
        for waste, frac in non_combustion.items():
            if frac > getattr(waste_fractions, waste):
                adjust_non_combustion = True

        if adjust_non_combustion:
            div_component_fractions_adjusted = DivComponentFractions(**div_component_fractions.model_dump())

            dont_add_to = {waste for waste, frac in waste_fractions.model_dump().items() if frac == 0}
            problems = [set(waste for waste, frac in non_combustion.items() if frac > getattr(waste_fractions, waste))]
            dont_add_to.update(problems[0])

            while problems:
                probs = problems.pop(0)
                for waste in probs:
                    remove = {}
                    distribute = {}
                    overflow = {}
                    can_be_adjusted = []
                    div_total = sum(getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste) for div in keys_of_interest if waste in getattr(div_component_fractions_adjusted, div).model_fields)
                    div_target = getattr(waste_fractions, waste)
                    diff = (div_total - div_target) / div_total

                    for div in keys_of_interest:
                        if getattr(div_fractions, div) == 0:
                            continue
                        distribute[div] = {}
                        component = getattr(getattr(div_component_fractions_adjusted, div), waste, 0)
                        to_be_removed = diff * component

                        to_distribute_to = [x for x in self.div_components[div] if x not in dont_add_to]
                        to_distribute_to_sum = sum(getattr(getattr(div_component_fractions_adjusted, div), x, 0) for x in to_distribute_to)
                        if to_distribute_to_sum == 0:
                            overflow[div] = 1
                            continue

                        for w in to_distribute_to:
                            add_amount = to_be_removed * (getattr(getattr(div_component_fractions_adjusted, div), w, 0) / to_distribute_to_sum)
                            if w not in distribute[div]:
                                distribute[div][w] = [add_amount]
                            else:
                                distribute[div][w].append(add_amount)

                        remove[div] = to_be_removed
                        can_be_adjusted.append(div)

                    for div in overflow:
                        component = getattr(getattr(div_component_fractions_adjusted, div), waste, 0)
                        to_be_removed = diff * component
                        to_distribute_to = [x for x in distribute.keys() if waste in self.div_components[x] and x not in overflow]
                        to_distribute_to_sum = sum(getattr(div_fractions, x) for x in to_distribute_to)
                        if to_distribute_to_sum == 0:
                            raise CustomError("INVALID_PARAMETERS", f"Combination of compost, anaerobic digestion, and recycling is too high")

                        for d in to_distribute_to:
                            to_be_removed_component = to_be_removed * (getattr(div_fractions, d) / to_distribute_to_sum) / getattr(div_fractions, d)
                            to_distribute_to_component = [x for x in getattr(div_component_fractions_adjusted, d).model_fields if x not in dont_add_to]
                            to_distribute_to_sum_component = sum(getattr(getattr(div_component_fractions_adjusted, d), x, 0) for x in to_distribute_to_component)
                            if to_distribute_to_sum_component == 0:
                                raise CustomError("INVALID_PARAMETERS", f"Combination of compost, anaerobic digestion, and recycling is too high")

                            for w in to_distribute_to_component:
                                add_amount = to_be_removed_component * getattr(getattr(div_component_fractions_adjusted, d), w, 0) / to_distribute_to_sum_component
                                if w in distribute[d]:
                                    distribute[d][w].append(add_amount)

                            remove[d] += to_be_removed_component

                    for div in distribute:
                        for w in distribute[div]:
                            setattr(getattr(div_component_fractions_adjusted, div), w, getattr(getattr(div_component_fractions_adjusted, div), w) + sum(distribute[div][w]))

                    for div in remove:
                        setattr(getattr(div_component_fractions_adjusted, div), waste, getattr(getattr(div_component_fractions_adjusted, div), waste) - remove[div])

                new_probs = {waste for waste in waste_fractions.model_fields if sum(getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste, 0) for div in keys_of_interest) > getattr(waste_fractions, waste) + 0.001}
                if new_probs:
                    problems.append(new_probs)
                dont_add_to.update(new_probs)

            components_multiplied_through = {
                div: {waste: getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste) for waste in getattr(div_component_fractions_adjusted, div).model_fields}
                for div in div_component_fractions_adjusted.model_fields
            }

        non_combustion = {}
        combustion_all = {}
        for waste in waste_fractions.model_fields:
            s = sum(components_multiplied_through[div].get(waste, 0) for div in keys_of_interest)
            non_combustion[waste] = s
            combustion_all[waste] = getattr(waste_fractions, waste) - s

        adjust_non_combustion = False
        for waste, frac in non_combustion.items():
            if frac > (getattr(waste_fractions, waste) + 1e-5):
                adjust_non_combustion = True
                raise CustomError("INVALID_PARAMETERS", f"Combination of compost, anaerobic digestion, and recycling is too high")

        all_divs = sum(getattr(div_fractions, div) for div in div_fractions.model_fields)

        assert np.abs(div_fractions.recycling - sum(components_multiplied_through['recycling'].values())) < 1e-3

        remainder = sum(fraction for waste_type, fraction in combustion_all.items() if waste_type in self.div_components['combustion'])
        combustion_fraction_of_remainder = div_fractions.combustion / remainder
        if combustion_fraction_of_remainder > (1 + 1e-5):
            non_combustables = [x for x in waste_fractions.model_fields if x not in self.div_components['combustion']]
            for waste in non_combustables:
                if getattr(waste_fractions, waste) == 0:
                    continue
                new_val = getattr(waste_fractions, waste) * all_divs
                components_multiplied_through['recycling'][waste] = new_val
            
            available_div = sum(v for k, v in components_multiplied_through['recycling'].items() if k not in non_combustables)
            available_div_target = div_fractions.recycling - sum(v for k, v in components_multiplied_through['recycling'].items() if k in non_combustables)
            if available_div_target < 0:
                too_much_frac = (sum(v for k, v in components_multiplied_through['recycling'].items() if k in non_combustables) - div_fractions.recycling) / sum(v for k, v in components_multiplied_through['recycling'].items() if k in non_combustables)
                for key, value in components_multiplied_through['recycling'].items():
                    if key in non_combustables:
                        components_multiplied_through['recycling'][key] = value * (1 - too_much_frac)
                    else:
                        components_multiplied_through['recycling'][key] = 0
                assert np.abs(div_fractions.recycling - sum(v for v in components_multiplied_through['recycling'].values())) < 1e-5

            else:
                reduce_frac = (available_div - available_div_target) / available_div
                for key, value in components_multiplied_through['recycling'].items():
                    if key not in non_combustables:
                        components_multiplied_through['recycling'][key] = value * (1 - reduce_frac)
                assert np.abs(div_fractions.recycling - sum(v for v in components_multiplied_through['recycling'].values())) < 1e-5

            non_combustion = {}
            combustion_all = {}
            for waste in waste_fractions.model_fields:
                s = sum(components_multiplied_through[div].get(waste, 0) for div in keys_of_interest)
                non_combustion[waste] = s
                combustion_all[waste] = getattr(waste_fractions, waste) - s

            remainder = sum(fraction for waste_type, fraction in combustion_all.items() if waste_type in self.div_components['combustion'])
            combustion_fraction_of_remainder = div_fractions.combustion / remainder
            assert combustion_fraction_of_remainder < (1 + 1e-5)
            if combustion_fraction_of_remainder > 1:
                combustion_fraction_of_remainder = 1

        for waste in self.div_components['combustion']:
            components_multiplied_through['combustion'][waste] = combustion_fraction_of_remainder * combustion_all[waste]

        for d in div_fractions.model_fields:
            assert np.abs(getattr(div_fractions, d) - sum(components_multiplied_through[d].values())) < 1e-3
            for w in components_multiplied_through[d]:
                if abs(components_multiplied_through[d][w]) < 1e-5:
                    components_multiplied_through[d][w] = 0
                assert components_multiplied_through[d][w] >= 0

        adjusted_div_component_fractions = {
            div: {waste: components_multiplied_through[div][waste] / getattr(div_fractions, div) if getattr(div_fractions, div) != 0 else 0 for waste in components_multiplied_through[div]}
            for div in components_multiplied_through
        }

        adjusted_div_component_fractions = DivComponentFractions(**adjusted_div_component_fractions)

        divs = self._divs_from_component_fractions(div_fractions, adjusted_div_component_fractions, scenario=scenario)

        div_component_fractions = adjusted_div_component_fractions
        adjusted_diversion_constituents = True
        input_problems = False

        return adjusted_diversion_constituents, input_problems, divs, div_component_fractions
        
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
        #     for div in div_component_fractions.model_fields:
        #         components_multiplied_through[div] = {}
        #         for waste in getattr(div_component_fractions, div).model_fields:
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
        #         for waste in parameters.waste_fractions.model_fields:
        #             s = sum(components_multiplied_through_dummy[div].get(waste, 0) for div in div_fractions.model_fields)
        #             net[waste] = getattr(parameters.waste_fractions, waste) - s
        #             if net[waste] < -1e-3:
        #                 negative_catcher = True

        #         if not negative_catcher:
        #             #divs = self._divs_from_component_fractions(div_fractions, div_component_fractions, scenario=scenario)
        #             #parameters.divs = divs
        #             parameters.adjusted_diversion_constituents = False
        #             parameters.input_problems = False
        #             return

        #         if sum(getattr(div_fractions, div) for div in div_fractions.model_fields) > 1:
        #             raise CustomError("INVALID_PARAMETERS", f"Diversions sum to {sum(getattr(div_fractions, div) for div in div_fractions.model_fields)}, but they must sum to 1 or less.")
                
        #         compostables = sum(getattr(parameters.waste_fractions, waste) for waste in ['food', 'green', 'wood', 'paper_cardboard'])
        #         if div_fractions.compost + div_fractions.anaerobic > compostables:
        #             raise CustomError("INVALID_PARAMETERS", f"Only food, green, wood, and paper/cardboard can be composted or anaerobically digested. Those waste types sum to {compostables}, but input values of compost and anaerobic digestion sum to {div_fractions.compost + div_fractions.anaerobic}.")

        #         for div in div_fractions.model_fields:
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
        #         for waste in parameters.waste_fractions.model_fields:
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
        #                     div_total = sum(getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste) for div in keys_of_interest if waste in getattr(div_component_fractions_adjusted, div).model_fields)
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
        #                             to_distribute_to_component = [x for x in getattr(div_component_fractions_adjusted, d).model_fields if x not in dont_add_to]
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

        #                 new_probs = {waste for waste in parameters.waste_fractions.model_fields if sum(getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste, 0) for div in keys_of_interest) > getattr(parameters.waste_fractions, waste) + 0.001}
        #                 if new_probs:
        #                     problems.append(new_probs)
        #                 dont_add_to.update(new_probs)

        #             components_multiplied_through_dummy = {
        #                 div: {waste: getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste) for waste in getattr(div_component_fractions_adjusted, div).model_fields}
        #                 for div in div_component_fractions_adjusted.model_fields
        #             }

        #         non_combustion = {}
        #         combustion_all = {}
        #         for waste in parameters.waste_fractions.model_fields:
        #             s = sum(components_multiplied_through_dummy[div].get(waste, 0) for div in keys_of_interest)
        #             non_combustion[waste] = s
        #             combustion_all[waste] = getattr(parameters.waste_fractions, waste) - s

        #         adjust_non_combustion = False
        #         for waste, frac in non_combustion.items():
        #             if frac > (getattr(parameters.waste_fractions, waste) + 1e-5):
        #                 adjust_non_combustion = True
        #                 raise CustomError("INVALID_PARAMETERS", f"Combination of compost, anaerobic digestion, and recycling is too high")

        #         all_divs = sum(getattr(div_fractions, div) for div in div_fractions.model_fields)

        #         assert np.abs(div_fractions.recycling - sum(components_multiplied_through_dummy['recycling'].values())) < 1e-3

        #         remainder = sum(fraction for waste_type, fraction in combustion_all.items() if waste_type in self.div_components['combustion'])
        #         combustion_fraction_of_remainder = div_fractions.combustion / remainder
        #         if combustion_fraction_of_remainder > (1 + 1e-5):
        #             non_combustables = [x for x in parameters.waste_fractions.model_fields if x not in self.div_components['combustion']]
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
        #             for waste in parameters.waste_fractions.model_fields:
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

        #         for d in div_fractions.model_fields:
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

    def _divs_from_component_fractions(self, div_fractions: DiversionFractions, div_component_fractions: DivComponentFractions, scenario: int, advanced: bool=False, year: int=2000) -> dict:
        """
        Calculates diverted masses from diversion fractions and component fractions,
        incorporating rejection rates.

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

        non_compostable_not_targeted_total = sum([
            self.non_compostable_not_targeted[x] * div_component_fractions.model_dump().get('compost', {}).get(x, 0.0)
            for x in self.div_components['compost']
        ])
        parameters.non_compostable_not_targeted_total = pd.Series(non_compostable_not_targeted_total, np.arange(1960, 2074))
        if parameters.non_compostable_not_targeted_total.isna().all():
            parameters.non_compostable_not_targeted_total = pd.Series(0, index=np.arange(1960, 2074))

        # Deal with waste mass that changes at implement_date first. 
        waste_mass = parameters.waste_mass
        if isinstance(waste_mass, Variant) and (waste_mass['scenario'] != waste_mass['baseline']):
            compost_masses = {'baseline': {}, 'scenario': {}}
            anaerobic_masses = {'baseline': {}, 'scenario': {}}
            combustion_masses = {'baseline': {}, 'scenario': {}}
            recycling_masses = {'baseline': {}, 'scenario': {}}
            for period in ['baseline', 'scenario']:
                for waste in self.waste_types:
                    compost_masses[period][waste] = (
                        waste_mass[period] * 
                        getattr(div_fractions, 'compost', 0) * 
                        getattr(getattr(div_component_fractions, 'compost', {}), waste, 0) * 
                        (1 - non_compostable_not_targeted_total) * 
                        (1 - self.unprocessable.get(waste, 0))
                    )
                    anaerobic_masses[period][waste] = (
                        waste_mass[period] * 
                        getattr(div_fractions, 'anaerobic', 0) * 
                        getattr(getattr(div_component_fractions, 'anaerobic', {}), waste, 0)
                    )
                    combustion_masses[period][waste] = (
                        waste_mass[period] * 
                        getattr(div_fractions, 'combustion', 0) * 
                        getattr(getattr(div_component_fractions, 'combustion', {}), waste, 0) * 
                        (1 - self.combustion_reject_rate)
                    )
                    recycling_masses[period][waste] = (
                        waste_mass[period] * 
                        getattr(div_fractions, 'recycling', 0) * 
                        getattr(getattr(div_component_fractions, 'recycling', {}), waste, 0) * 
                        self.recycling_reject_rates.get(waste, 0)
                    )

            divs = {}
            divs['baseline'] = DivMasses(
                compost=WasteMasses(**compost_masses['baseline']),
                anaerobic=WasteMasses(**anaerobic_masses['baseline']),
                combustion=WasteMasses(**combustion_masses['baseline']),
                recycling=WasteMasses(**recycling_masses['baseline'])
            )
            divs['scenario'] = DivMasses(
                compost=WasteMasses(**compost_masses['scenario']),
                anaerobic=WasteMasses(**anaerobic_masses['scenario']),
                combustion=WasteMasses(**combustion_masses['scenario']),
                recycling=WasteMasses(**recycling_masses['scenario'])
            )

            return divs

        if isinstance(waste_mass, Variant):
            waste_mass = waste_mass['scenario']
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
                waste_mass * 
                getattr(div_fractions, 'compost', 0) * 
                getattr(getattr(div_component_fractions, 'compost', {}), waste, 0) * 
                (1 - non_compostable_not_targeted_total) * 
                (1 - self.unprocessable.get(waste, 0))
            )
            anaerobic_masses[waste] = (
                waste_mass * 
                getattr(div_fractions, 'anaerobic', 0) * 
                getattr(getattr(div_component_fractions, 'anaerobic', {}), waste, 0)
            )
            combustion_masses[waste] = (
                waste_mass * 
                getattr(div_fractions, 'combustion', 0) * 
                getattr(getattr(div_component_fractions, 'combustion', {}), waste, 0) * 
                (1 - self.combustion_reject_rate)
            )
            recycling_masses[waste] = (
                waste_mass * 
                getattr(div_fractions, 'recycling', 0) * 
                getattr(getattr(div_component_fractions, 'recycling', {}), waste, 0) * 
                self.recycling_reject_rates.get(waste, 0)
            )

        divs = DivMasses(
            compost=WasteMasses(**compost_masses),
            anaerobic=WasteMasses(**anaerobic_masses),
            combustion=WasteMasses(**combustion_masses),
            recycling=WasteMasses(**recycling_masses)
        )

        return divs

    @staticmethod
    def calculate_reduction(value: float, limit: float, excess: float, total_reducible: float) -> float:
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
        reduction = min(reducible, excess * (reducible / total_reducible))  # proportional reduction
        return reduction
    
    def _create_divs_dataframe(self, baseline_divs, scenario_divs):
        """
        Create a DataFrame that merges baseline and scenario diversion data based on the implementation year.

        Args:
            baseline_divs (object): Baseline diversion data.
            scenario_divs (object): Scenario diversion data.
            implement_year (int): The year when the scenario diversions start being implemented.

        Returns:
            DataFrame: A DataFrame with years as the index and diversion data as the columns.
        """

        implement_year = self.scenario_parameters[0].implement_year

        baseline_data = {year: {waste: getattr(baseline_divs, waste) for waste in baseline_divs.model_dump()} for year in range(1960, implement_year)}
        scenario_data = {year: {waste: getattr(scenario_divs, waste) for waste in scenario_divs.model_dump()} for year in range(implement_year, 2074)}

        df = pd.concat([pd.DataFrame(baseline_data).T, pd.DataFrame(scenario_data).T])

        return df

    def _create_waste_fractions_dataframe(self, advanced_dst: bool=False) -> pd.DataFrame:
        """
        Create a DataFrame that merges baseline and scenario waste fractions data based on the implementation year.

        Args:
            baseline_waste_fractions (object): Baseline waste fractions data.
            scenario_waste_fractions (object): Scenario waste fractions data.
            implement_year (int): The year when the scenario waste fractions start being implemented.

        Returns:
            DataFrame: A DataFrame with years as the index and waste fractions data as the columns.
        """
        # Come back to this, waste fractions should already be dataframe for advanced_dst
        if not advanced_dst:
            baseline_waste_fractions = self.baseline_parameters.waste_fractions
            scenario_waste_fractions = self.scenario_parameters[0].waste_fractions
        implement_year = self.scenario_parameters[0].implement_year

        baseline_data = {year: {waste: getattr(baseline_waste_fractions, waste) for waste in baseline_waste_fractions.model_dump()} for year in range(1960, implement_year)}
        scenario_data = {year: {waste: getattr(scenario_waste_fractions, waste) for waste in scenario_waste_fractions.model_dump()} for year in range(implement_year, 2074)}

        df = pd.concat([pd.DataFrame(baseline_data).T, pd.DataFrame(scenario_data).T])

        return df

    def _calculate_net_masses(self, scenario: int=0, advanced_baseline: bool=False, advanced_dst: bool=False) -> WasteMasses:
        """
        Calculate the net masses of different types of waste after diversion.

        Args:
            scenario (int): The scenario number to use (0 for baseline, or the number of the alternative scenario).

        Returns:
            WasteMasses: An instance of WasteMasses containing the net masses of different types of waste.
        """
        if scenario == 0:
            parameters = self.baseline_parameters
        else:
            parameters = self.scenario_parameters.get(scenario-1)
            if parameters is None:
                raise ValueError(f"Scenario '{scenario}' not found in scenario_parameters.")

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
                fraction = parameters.waste_fractions.at[1960, col]
                waste_mass_dict[col] = parameters.waste_mass.iloc[0] * fraction
            parameters.waste_masses = WasteMasses(**waste_mass_dict)

        try:
            combined_diversions = pd.concat(
                [divs.compost, divs.anaerobic, divs.combustion, divs.recycling],
                axis=1
            ).T.groupby(level=0).sum().T
        except:
            diverted = {
                waste: divs.compost.model_dump().get(waste, 0) +
                    divs.anaerobic.model_dump().get(waste, 0) +
                    divs.combustion.model_dump().get(waste, 0) +
                    divs.recycling.model_dump().get(waste, 0)
                for waste in self.waste_types
            }

        try:
            parameters.net_masses = parameters.waste_generated_df - combined_diversions
        except:
            net = {
                waste: parameters.waste_masses.model_dump()[waste] - diverted.get(waste, 0)
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
            scenario: int
    ) -> None:
        
        scenario_parameters = copy.deepcopy(self.baseline_parameters)
        self.scenario_parameters[scenario - 1] = scenario_parameters
        scenario_parameters.div_fractions = new_div_fractions

        # Set new split fractions
        
        scenario_parameters.split_fractions.dumpsite = 1 - new_landfill_pct
        pct_landfill = 1 - scenario_parameters.split_fractions.dumpsite
        scenario_parameters.split_fractions.landfill_w_capture = new_gas_pct * pct_landfill
        scenario_parameters.split_fractions.landfill_wo_capture = (1 - new_gas_pct) * pct_landfill
        scenario_parameters.landfills[0].fraction_of_waste = scenario_parameters.split_fractions.landfill_w_capture
        scenario_parameters.landfills[1].fraction_of_waste = scenario_parameters.split_fractions.landfill_wo_capture
        scenario_parameters.landfills[2].fraction_of_waste = scenario_parameters.split_fractions.dumpsite
        for lf in scenario_parameters.landfills:
            lf.scenario = 1
        scenario_parameters.non_zero_landfills = [lf for lf in scenario_parameters.landfills if lf.fraction_of_waste > 0]
        scenario_parameters.implement_year = implement_year

        # Recalculate div_component_fractions
        waste_fractions = scenario_parameters.waste_fractions
        waste_fractions = WasteFractions(**waste_fractions.iloc[0].to_dict())

        def calculate_component_fractions(waste_fractions: WasteFractions, div_type: str) -> WasteFractions:
            components = self.div_components[div_type]
            filtered_fractions = {waste: getattr(waste_fractions, waste) for waste in components}
            total = sum(filtered_fractions.values())
            normalized_fractions = {waste: fraction / total for waste, fraction in filtered_fractions.items()}
            return WasteFractions(**{waste: normalized_fractions.get(waste, 0) for waste in components})

        scenario_parameters.div_component_fractions = DivComponentFractions(
            compost=calculate_component_fractions(waste_fractions, 'compost'),
            anaerobic=calculate_component_fractions(waste_fractions, 'anaerobic'),
            combustion=calculate_component_fractions(waste_fractions, 'combustion'),
            recycling=calculate_component_fractions(waste_fractions, 'recycling'),
        )
        scenario_parameters.non_compostable_not_targeted_total = sum(
            [self.non_compostable_not_targeted[x] * \
             getattr(scenario_parameters.div_component_fractions.compost, x) for x in self.div_components['compost']])
        if np.isnan(scenario_parameters.non_compostable_not_targeted_total):
            scenario_parameters.non_compostable_not_targeted_total = 0.0
        self._calculate_diverted_masses(scenario=scenario) # This function could be moved to cityparameters class, and then it doesn't need scenario argument

        #scenario_parameters.repopulate_attr_dicts()
        self._check_masses_v2(scenario=scenario)

        if scenario_parameters.input_problems:
            print(f'Invalid new value')
            return

        self._calculate_net_masses(scenario=scenario)
        for w in scenario_parameters.net_masses.index:
            mass = scenario_parameters.net_masses.at[w]
            if mass < 0:
                print(f'Invalid new value')
                return
            
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
            recycling=recycling_wm
        )
        
        try:
            yr_pop = scenario_parameters.year_of_data_pop['baseline']
        except:
            yr_pop = scenario_parameters.year_of_data_pop

        scenario_parameters.divs_df = DivsDF.create_simple(
            baseline_divs=baseline_divs, 
            scenario_divs=scenario_parameters.divs,
            start_year=1960,
            end_year=2073,
            implement_year=implement_year,
            year_of_data_pop=yr_pop, 
            growth_rate_historic=scenario_parameters.growth_rate_historic, 
            growth_rate_future=scenario_parameters.growth_rate_future,
        )

        # combine these two loops maybe...though it still does six things, maybe doesn't matter
        scenario_parameters.repopulate_attr_dicts()
        for i, landfill in enumerate(scenario_parameters.landfills):
            # Might be able to do this more efficienctly...i'm looping over the pre implementation years twice sort of
            landfill.waste_mass_df = LandfillWasteMassDF.create(scenario_parameters.waste_generated_df, scenario_parameters.divs_df, landfill.fraction_of_waste, self.components).df
            landfill.waste_mass_df.loc[:(implement_year-1), :] = self.baseline_parameters.landfills[i].waste_mass_df.loc[:(implement_year-1), :]
            #print(landfill.waste_mass_df)

        #scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in scenario_parameters.landfills:
            landfill.estimate_emissions()
            #print(landfill.emissions)

        self.estimate_diversion_emissions(scenario=scenario)
        self.sum_landfill_emissions(scenario=scenario, simple=True)

    # def implement_dst_changes_advanced(
    #     self,
    #     population: float,
    #     precipitation: float,
    #     new_waste_fractions: WasteFractions,
    #     new_div_fractions: DiversionFractions,
    #     new_landfill_types: List,
    #     new_landfill_open_close_dates: List,
    #     implement_year: float,
    #     scenario: int,
    #     new_baseline: int,
    #     fraction_waste_timeline: Dict,
    #     new_gas_efficiency: Dict, # 0 means no gas capture, blank means figure out the efficiency for me
    #     new_landfill_fracs: Dict = None,
    #     new_landfill_flaring: Dict = None,
    #     new_landfill_cover: Dict = None,
    #     new_landfill_leachate_circulate: Dict = None,
    #     new_landfill_latlons: Dict = None,
    #     new_landfill_areas: Dict = None,
    #     new_covertypes: Dict = None,
    #     new_coverthicknesses: Dict = None,
    #     waste_burning: float = None,
    #     do_fancy_ox: bool = False,
    # ) -> None:
        
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
        #new_baseline: int,
        landfill_split_timeline: Dict,
        new_gas_efficiency: Dict, # 0 means no gas capture, blank means figure out the efficiency for me
        new_landfill_fracs: Dict = None,
        new_landfill_flaring: Dict = None,
        new_landfill_cover: Dict = None,
        leachate_circulate: Dict = None,
        new_landfill_latlons: Dict = None,
        new_landfill_areas: Dict = None,
        new_covertypes: Dict = None,
        new_coverthicknesses: Dict = None,
        waste_burning: Dict = None,
        fancy_ox: Dict = {'baseline': False, 'scenario': False},
        new_waste_mass_per_capita: bool = False,
        depths: Dict = None,
        k_values: Dict = None,
        waste_mass_year: int = None,
        ks_overrides: Dict = None,
        biocover: Dict = {'baseline': 0.0, 'scenario': 0.0},
        oxidation_override: Dict = None,
    ) -> None:
        
        scenario_parameters = copy.deepcopy(self.baseline_parameters)
        self.scenario_parameters[scenario - 1] = scenario_parameters
        scenario_parameters.div_fractions = new_div_fractions
        scenario_parameters.waste_fractions = new_waste_fractions
        scenario_parameters._singapore_k(implement_year=implement_year, advanced_dst=True)
        scenario_parameters.implement_year = implement_year

        pd.set_option('display.max_rows', None)
        
        if new_waste_mass:
            pass
        elif new_waste_mass_per_capita:
            new_waste_mass = {}
            new_waste_mass['baseline'] = new_waste_mass_per_capita * population
            new_waste_mass['scenario'] = new_waste_mass_per_capita * population
        else:
            new_waste_mass = {}
            new_waste_mass['baseline'] = scenario_parameters.waste_mass.iat[0]
            new_waste_mass['scenario'] = scenario_parameters.waste_mass.iat[0]
        scenario_parameters.waste_mass = new_waste_mass

        years = pd.Index(range(1960, 2074))
        waste_mass_series = pd.Series(index=years)
        waste_mass_series.loc[:waste_mass_year-1] = new_waste_mass['baseline']
        waste_mass_series.loc[waste_mass_year:] = new_waste_mass['scenario']

        # Adjust for waste burning
        waste_burned = {}
        wb = None
        if waste_burning['baseline'] > 0:
            waste_burned['baseline'] = waste_burning['baseline'] * waste_mass_series
            waste_mass_series.loc[:waste_mass_year-1] -= waste_burned['baseline'].loc[:waste_mass_year-1]

            # Adjust the waste burning for growth rates to get real time series
            t = waste_mass_series.index.values - scenario_parameters.year_of_data_pop['baseline']

            # Create growth rate array, using growth_rate_historic for years before year_of_data_pop and growth_rate_future after
            growth_rate = np.where(waste_mass_series.index.values < scenario_parameters.year_of_data_pop['baseline'], scenario_parameters.growth_rate_historic, scenario_parameters.growth_rate_future)
            growth_factors = growth_rate ** t

            # Apply growth factors
            waste_burned['baseline'] = waste_burned['baseline'].multiply(growth_factors, axis=0)
            wb = waste_burned['baseline']

        if waste_burning['scenario'] > 0:
            waste_burned['scenario'] = waste_burning['scenario'] * waste_mass_series
            waste_mass_series.loc[waste_mass_year:] -= waste_burned['scenario'].loc[waste_mass_year:]

            # Adjust the waste burning for growth rates to get real time series
            t = waste_mass_series.index.values - scenario_parameters.year_of_data_pop['scenario']

            # Create growth rate array, using growth_rate_historic for years before year_of_data_pop and growth_rate_future after
            growth_rate = np.where(waste_mass_series.index.values < scenario_parameters.year_of_data_pop['scenario'], scenario_parameters.growth_rate_historic, scenario_parameters.growth_rate_future)
            growth_factors = growth_rate ** t

            # Apply growth factors
            waste_burned['scenario'] = waste_burned['scenario'].multiply(growth_factors, axis=0)
            if wb is not None:
                wb.loc[implement_year:] = waste_burned['scenario'].loc[implement_year:]
            else:
                wb = waste_burned['scenario']
                wb.loc[:implement_year-1] = 0
        
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
        waste_masses_df = pd.DataFrame(index=years, columns=new_waste_fractions['baseline'].model_dump().keys())

        # Fill the DataFrame with the calculated waste masses
        for waste in waste_masses_df.columns:
            baseline_frac = new_waste_fractions['baseline'].model_dump()[waste]
            scenario_frac = new_waste_fractions['scenario'].model_dump()[waste]
            
            waste_masses_df.loc[:waste_mass_year-1, waste] = baseline_frac * waste_mass_series.loc[:waste_mass_year-1]
            waste_masses_df.loc[waste_mass_year:, waste] = scenario_frac * waste_mass_series.loc[waste_mass_year:]

        # Assign the DataFrame to scenario_parameters
        scenario_parameters.waste_masses = waste_masses_df

        # Update waste generated
        scenario_parameters.waste_generated_df = WasteGeneratedDF.create_advanced(
            waste_masses_df = waste_masses_df,
            start_year=1960, 
            end_year=2073,
            year_of_data_pop=scenario_parameters.year_of_data_pop['baseline'], 
            growth_rate_historic=scenario_parameters.growth_rate_historic, 
            growth_rate_future=scenario_parameters.growth_rate_future,
            implement_year=waste_mass_year
        ).df

        # Create a DataFrame for fraction_waste_timeline
        fraction_df = pd.DataFrame(landfill_split_timeline).transpose()
        fraction_df.columns = [f'Landfill_{i}' for i in range(fraction_df.shape[1])]
        fraction_df.index.name = 'Year'

        # Set up new landfills
        city_params_dict = self.update_cityparams_dict(scenario_parameters)
        #mcfs = [1, 0.7, 0.4] # Should this include ameliorated?
        #mcf_ameliorated = [0.7, 0.4, 0.1]
        mcf_options = [1, 0.6, 0.4]
        gas_capture_efficiencies = {}
        gas_capture_efficiencies['ameliorated'] = [0.5, 0.3, 0]
        gas_capture_efficiencies['not_ameliorated'] = [0.6, 0.45, 0]
        self.ox_options = {
            'ox_nocap': {'landfill': 0.1, 'controlled_dumpsite': 0.05, 'dumpsite': 0},
            'ox_cap': {'landfill': 0.22, 'controlled_dumpsite': 0.1, 'dumpsite': 0}
        }
        landfill_types = ['landfill', 'controlled_dumpsite', 'dumpsite']
        scenario_parameters.landfills = []
        for i, lf_type in enumerate(new_landfill_types['scenario']):
            # Make the MCF, oxidation, and efficiency vectors
            years = pd.Index(range(1960, 2074))
            mcf = {}
            ox_value = {}
            gas_eff = {}

            # Get MCF
            old_lf_type = new_landfill_types['baseline'][i]
            mcf['baseline'] = mcf_options[old_lf_type]
            mcf['scenario'] = mcf_options[lf_type]

            if (depths['baseline'][i] > 5) and (old_lf_type in (1, 2)):
                mcf['baseline'] = 0.8

            if (depths['scenario'][i] > 5) and (lf_type in (1, 2)):
                mcf['scenario'] = 0.8

            # Handle baseline first
            if i >= len(new_gas_efficiency['baseline']):
                ox_value['baseline'] = 0
                gas_eff['baseline'] = 0
            elif new_gas_efficiency['baseline'][i] == 0.0:
                ox_value['baseline'] = self.ox_options['ox_nocap'][landfill_types[old_lf_type]]
                gas_eff['baseline'] = 0
            # If there is gas capture, use the number or figure it out
            elif new_gas_efficiency['baseline'][i] > 0.0:
                ox_value['baseline'] = self.ox_options['ox_cap'][landfill_types[old_lf_type]]
                gas_eff ['baseline']= new_gas_efficiency['baseline'][i] if new_gas_efficiency['baseline'][i] is not None else gas_capture_efficiencies['not_ameliorated'][old_lf_type]
            else:
                print('invalid gas efficiency value')

            # For scenario, handle no gas capture first
            if new_gas_efficiency['scenario'][i] == 0:
                ox_value['scenario'] = self.ox_options['ox_nocap'][landfill_types[lf_type]]
                gas_eff['scenario'] = 0
            # If there is gas capture, use the number or figure it out
            elif new_gas_efficiency['scenario'][i] > 0.0:
                if (new_landfill_types['scenario'][i] < new_landfill_types['baseline'][i]):
                    ameliorated=True
                    if lf_type == 0:
                        ox_value['scenario'] = 0.18
                    else:
                        ox_value['scenario'] = self.ox_options['ox_cap'][landfill_types[lf_type]]
                    gas_eff ['scenario']= new_gas_efficiency['baseline'][i] if new_gas_efficiency['baseline'][i] is not None else gas_capture_efficiencies['ameliorated'][lf_type]
                else:
                    ameliorated=False
                    ox_value['scenario'] = self.ox_options['ox_cap'][landfill_types[lf_type]]
                    gas_eff ['scenario']= new_gas_efficiency['baseline'][i] if new_gas_efficiency['baseline'][i] is not None else gas_capture_efficiencies['not_ameliorated'][lf_type]
            else:
                print('invalid gas efficiency value')

            if i >= len(new_gas_efficiency['baseline']):
                pass
            elif new_gas_efficiency['baseline'][i] is not None:
                gas_eff['baseline'] = new_gas_efficiency['baseline'][i]
            if new_gas_efficiency['scenario'][i] is not None:
                gas_eff['scenario'] = new_gas_efficiency['scenario'][i]

            # Create pandas Series for each: mcf, ox_value, and gas_eff
            mcf_series = pd.Series(index=years)
            ox_value_series = pd.Series(index=years)
            gas_eff_series = pd.Series(index=years)

            # Assign baseline values before implement_year and scenario values after
            mcf_series.loc[years < implement_year] = mcf['baseline']
            mcf_series.loc[years >= implement_year] = mcf['scenario']
            
            if biocover['baseline'] > 0:
                ox_value['baseline ']= biocover['baseline']
            if biocover['scenario'] > 0:
                ox_value['scenario'] = biocover['scenario']
            ox_value_series.loc[years < implement_year] = ox_value['baseline']
            ox_value_series.loc[years >= implement_year] = ox_value['scenario']
            
            gas_eff_series.loc[years < implement_year] = gas_eff['baseline']
            gas_eff_series.loc[years >= implement_year] = gas_eff['scenario']

            doing_fancy_ox = fancy_ox

            if ks_overrides is not None:
                ks_series = pd.Series([ks_overrides['baseline']] * len(years), index=years)
                ks_series.loc[implement_year:] = ks_overrides['scenario']
                landfill_ks = DecompositionRates(
                    food = ks_series,
                    green = ks_series,
                    wood = ks_series,
                    paper_cardboard = ks_series,
                    textiles = ks_series,
                )
            else:
                landfill_ks = scenario_parameters.ks

            if oxidation_override:
                if oxidation_override['baseline']:
                    ox_value_series.loc[:implement_year-1] = oxidation_override['baseline']
                if oxidation_override['scenario']:
                    ox_value_series.loc[implement_year:] = oxidation_override['scenario']

            new_landfill = Landfill(
                open_date=new_landfill_open_close_dates['scenario'][i][0], 
                close_date=new_landfill_open_close_dates['scenario'][i][1], 
                site_type=landfill_types[lf_type], 
                mcf=mcf_series,
                city_params_dict=city_params_dict,
                city_instance_attrs=scenario_parameters.city_instance_attrs,
                landfill_index=i, 
                #fraction_of_waste=new_landfill_fracs[i], 
                gas_capture=False if new_gas_efficiency['scenario'][i] == 0.0 else True,
                scenario=scenario,
                new_baseline=False,
                gas_capture_efficiency=gas_eff_series,
                flaring=new_landfill_flaring['scenario'][i],
                #leachate_circulate=leachate_circulate['scenario'][i],
                fraction_of_waste_vector=fraction_df[f'Landfill_{i}'],
                advanced=True,
                latlon=new_landfill_latlons['scenario'][i] if doing_fancy_ox else None,
                areas=new_landfill_areas['scenario'][i] if doing_fancy_ox else None,
                cover_types=new_covertypes['scenario'][i] if doing_fancy_ox else None,
                cover_thicknesses=new_coverthicknesses['scenario'][i] if doing_fancy_ox else None,
                oxidation_factor=ox_value_series if not doing_fancy_ox else None,
                fancy_ox=fancy_ox,
                implementation_year=implement_year,
                ks = landfill_ks,
            )
            scenario_parameters.landfills.append(new_landfill)


        # Recalculate div_component_fractions
        # waste_fractions = scenario_parameters.waste_fractions

        # def calculate_component_fractions(waste_fractions: WasteFractions, div_type: str) -> WasteFractions:
        #     components = self.div_components[div_type]
        #     filtered_fractions = {waste: getattr(waste_fractions, waste) for waste in components}
        #     total = sum(filtered_fractions.values())
        #     normalized_fractions = {waste: fraction / total for waste, fraction in filtered_fractions.items()}
        #     return WasteFractions(**{waste: normalized_fractions.get(waste, 0) for waste in waste_fractions.model_fields})

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

        def calculate_component_fractions(waste_fractions: dict, div_type: str, implement_year: int, years) -> pd.DataFrame:
            # Extract the waste types from WasteFractions objects
            baseline_df = pd.DataFrame(waste_fractions['baseline'].model_dump(), index=[0])
            scenario_df = pd.DataFrame(waste_fractions['scenario'].model_dump(), index=[0])

            # Components are the subset of columns we're interested in
            components = list(self.div_components[div_type])
            
            # Filter only the relevant columns (components) for both baseline and scenario
            baseline_df = baseline_df[components]
            scenario_df = scenario_df[components]

            # Normalize each row (for baseline and scenario)
            baseline_normalized = baseline_df.div(baseline_df.sum(axis=1), axis=0).fillna(0)
            scenario_normalized = scenario_df.div(scenario_df.sum(axis=1), axis=0).fillna(0)

            # Create a mask for the years before and after the implement year
            mask = years < implement_year

            # Create a DataFrame with years as the index and assign baseline or scenario fractions
            result_df = pd.DataFrame(index=years, columns=components)
            
            # Assign baseline values for years before implement_year, scenario values after
            result_df.loc[mask, :] = np.tile(baseline_normalized.reindex(columns=result_df.columns).iloc[0].values, (mask.sum(), 1))
            result_df.loc[~mask, :] = np.tile(scenario_normalized.reindex(columns=result_df.columns).iloc[0].values, ((~mask).sum(), 1))
            
            return result_df

        # Call the function for each diversion type
        scenario_parameters.div_component_fractions = DivComponentFractionsDF(
            compost=calculate_component_fractions(scenario_parameters.waste_fractions, 'compost', implement_year, years),
            anaerobic=calculate_component_fractions(scenario_parameters.waste_fractions, 'anaerobic', implement_year, years),
            combustion=calculate_component_fractions(scenario_parameters.waste_fractions, 'combustion', implement_year, years),
            recycling=calculate_component_fractions(scenario_parameters.waste_fractions, 'recycling', implement_year, years),
        )
        scenario_parameters.non_compostable_not_targeted_total = sum(
            [self.non_compostable_not_targeted[x] * \
            getattr(scenario_parameters.div_component_fractions.compost, x) for x in self.div_components['compost']])
        scenario_parameters.non_compostable_not_targeted_total = pd.Series(scenario_parameters.non_compostable_not_targeted_total, index=years)
        if scenario_parameters.non_compostable_not_targeted_total.isna().all():
            scenario_parameters.non_compostable_not_targeted_total = pd.Series(0, index=years)
        self._calculate_diverted_masses(scenario=scenario)
        
        # Split the years for baseline and scenario
        baseline_years = years[years < implement_year]
        scenario_years = years[years >= implement_year]

        scenario_parameters.waste_fractions = pd.concat([
            pd.DataFrame(new_waste_fractions['baseline'].model_dump(), index=baseline_years),
            pd.DataFrame(new_waste_fractions['scenario'].model_dump(), index=scenario_years)
        ])

        #scenario_parameters.repopulate_attr_dicts()
        self._check_masses_v2(scenario=scenario, advanced_baseline=False, advanced_dst=True, implement_year=implement_year)

        if scenario_parameters.input_problems:
            print(f'Invalid new value')
            return

        self._calculate_net_masses(scenario=scenario, advanced_dst=True)
        if (scenario_parameters.net_masses < 0).any().any():
            print(f'Invalid new value')
            return

        # This isn't set up yet for year of data pop scenario and implement year being different
        scenario_parameters.divs_df = DivsDF.implement_advanced(
            divs=scenario_parameters.divs, 
            year_of_data_pop=scenario_parameters.year_of_data_pop['baseline'], 
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
            #print(landfill.waste_mass_df.loc[2028:2032, :])

            #landfill.waste_mass_df.loc[:(implement_year-1), :] = self.baseline_parameters.landfills[i].waste_mass_df.loc[:(implement_year-1), :]
            #landfill.waste_mass_df.to_csv('/Users/hugh/Library/CloudStorage/OneDrive-RMI/Documents/RMI/scratch_paper/new' + str(i) + '.csv')

        #scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in scenario_parameters.landfills:
            landfill.estimate_emissions()

        self.estimate_diversion_emissions(scenario=scenario)
        self.sum_landfill_emissions(scenario=scenario)

        # ADD WASTE BURNING EMISSIONS
        if (waste_burning['baseline'] > 0) or (waste_burning['scenario'] > 0):
            scenario_parameters.waste_burning_emissions = waste_burned * 3.7 * 1000 / 1000 / 1000 # g ch4 / kg waste to ton ch4 / ton waste
            scenario_parameters.waste_burning_emissions = scenario_parameters.waste_burning_emissions.reindex(
                scenario_parameters.total_emissions.index, fill_value=0
            )
            scenario_parameters.total_emissions['total'] += scenario_parameters.waste_burning_emissions
        
        # if waste_burning['scenario'] > 0:
        #     scenario_parameters.waste_burning_emissions = waste_burned['scenario'] * 3.7 * 1000 / 1000 / 1000 # g ch4 / kg waste to ton ch4 / ton waste
        #     scenario_parameters.waste_burning_emissions = scenario_parameters.waste_burning_emissions.reindex(
        #         scenario_parameters.total_emissions.index, fill_value=0
        #     )
        #     scenario_parameters.total_emissions['total'] += scenario_parameters.waste_burning_emissions


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
        new_gas_efficiency: List, # 0 means no gas capture, blank means figure out the efficiency for me
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
        
        scenario_parameters = copy.deepcopy(self.baseline_parameters)
        self.scenario_parameters[scenario - 1] = scenario_parameters
        scenario_parameters.div_fractions = new_div_fractions

        years = pd.Index(range(1960, 2074))
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
        #waste_burned = {}
        if waste_burning > 0:
            waste_burned = pd.DataFrame(waste_burning * new_waste_mass, index=years)
            new_waste_mass -= (waste_burning * new_waste_mass)

            # Adjust the waste burning for growth rates to get real time series
            t = years - scenario_parameters.year_of_data_pop

            # Create growth rate array, using growth_rate_historic for years before year_of_data_pop and growth_rate_future after
            growth_rate = np.where(years < scenario_parameters.year_of_data_pop, scenario_parameters.growth_rate_historic, scenario_parameters.growth_rate_future)
            growth_factors = growth_rate ** t

            # Apply growth factors
            waste_burned = waste_burned.multiply(growth_factors, axis=0)

        # New waste masses
        waste_masses = {}
        waste_masses = pd.DataFrame({col: new_waste_fractions.at[2000, col] * new_waste_mass for col in new_waste_fractions.columns}, index=new_waste_fractions.index)
        scenario_parameters.waste_masses = waste_masses #WasteMasses(**waste_masses)

        # Update waste generated
        scenario_parameters.waste_generated_df = WasteGeneratedDF.create(
            waste_masses,
            1960, 
            2073, 
            scenario_parameters.year_of_data_pop['baseline'], 
            scenario_parameters.growth_rate_historic, 
            scenario_parameters.growth_rate_future
        ).df

        # Create a DataFrame for fraction_waste_timeline
        fraction_df = pd.DataFrame(landfill_split_timeline).transpose()
        fraction_df.columns = [f'Landfill_{i}' for i in range(fraction_df.shape[1])]
        fraction_df.index.name = 'Year'

        # Set up new landfills
        city_params_dict = self.update_cityparams_dict(scenario_parameters)
        #mcfs = [1, 0.7, 0.4] # Should this include ameliorated?
        #mcf_ameliorated = [0.7, 0.4, 0.1]
        mcf_options = [1, 0.6, 0.4]
        #mcfs['ameliorated'] = {}
        #mcf_options['not_ameliorated'] = {}
        #mcfs['ameliorated']['gas_capture'] = [0.18, 0, 0]
        #mcfs['ameliorated']['no_gas_capture'] = [0.1, 0, 0]
        #mcf_options['not_ameliorated']['gas_capture'] = [0.22, 0.1, 0]
        #mcf_options['not_ameliorated']['no_gas_capture'] = [0.1, 0.05, 0]
        gas_capture_efficiencies = {}
        gas_capture_efficiencies['ameliorated'] = [0.5, 0.3, 0]
        gas_capture_efficiencies['not_ameliorated'] = [0.6, 0.45, 0]
        self.ox_options = {
            'ox_nocap': {'landfill': 0.1, 'controlled_dumpsite': 0.05, 'dumpsite': 0},
            'ox_cap': {'landfill': 0.22, 'controlled_dumpsite': 0.1, 'dumpsite': 0}
        }
        landfill_types = ['landfill', 'controlled_dumpsite', 'dumpsite']
        scenario_parameters.landfills = []
        for i, lf_type in enumerate(new_landfill_types):
            # Make the MCF, oxidation, and efficiency vectors
            years = pd.Index(range(1960, 2074))
            mcf = mcf_options[lf_type]
            if (depth > 5) and (lf_type in (1, 2)):
                mcf = 0.8
            # Handle no gas capture first
            if new_gas_efficiency[i] == 0:
                #mcf = mcf_options['not_ameliorated']['no_gas_capture'][lf_type]
                ox_value = self.ox_options['ox_nocap'][landfill_types[lf_type]]
                gas_eff = 0
            # If there is gas capture, use the number or figure it out
            else:
                #mcf = mcf_options['not_ameliorated']['gas_capture'][lf_type]
                ox_value = self.ox_options['ox_cap'][landfill_types[lf_type]]
                gas_eff = new_gas_efficiency[i] if new_gas_efficiency[i] is not None else gas_capture_efficiencies['not_ameliorated'][lf_type]

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
                    food = pd.Series([ks_overrides] * len(years), index=years),
                    green = pd.Series([ks_overrides] * len(years), index=years),
                    wood = pd.Series([ks_overrides] * len(years), index=years),
                    paper_cardboard = pd.Series([ks_overrides] * len(years), index=years),
                    textiles = pd.Series([ks_overrides] * len(years), index=years)
                )
            else:
                landfill_ks = scenario_parameters.ks

            if oxidation_override:
                oxs = [oxidation_override for year in years]
            
            new_landfill = Landfill(
                open_date=new_landfill_open_close_dates[i][0], 
                close_date=new_landfill_open_close_dates[i][1], 
                site_type=landfill_types[lf_type], 
                mcf=pd.Series(mcfs, index=years),
                city_params_dict=city_params_dict,
                city_instance_attrs=scenario_parameters.city_instance_attrs,
                landfill_index=i, 
                #fraction_of_waste=new_landfill_fracs[i], 
                gas_capture = False if new_gas_efficiency[i] == 0 else True,
                scenario=scenario,
                new_baseline=True,
                gas_capture_efficiency=pd.Series(gas_effs, index=years),
                flaring=new_landfill_flaring[i],
                #leachate_circulate=leachate_circulate[i],
                fraction_of_waste_vector=fraction_df[f'Landfill_{i}'],
                advanced=True,
                latlon=new_landfill_latlons[i] if fancy_ox else None,
                areas=new_landfill_areas[i] if fancy_ox else None,
                cover_types=new_covertypes[i] if fancy_ox else None,
                cover_thicknesses=new_coverthicknesses[i] if fancy_ox else None,
                oxidation_factor=pd.Series(oxs, index=years) if not fancy_ox else None,
                fancy_ox=fancy_ox,
                ks = landfill_ks
            )
            scenario_parameters.landfills.append(new_landfill)

        # Recalculate div_component_fractions
        waste_fractions = scenario_parameters.waste_fractions

        def calculate_component_fractions(waste_fractions: WasteFractions, div_type: str) -> WasteFractions:
            components = list(self.div_components[div_type])
            filtered_fractions = waste_fractions.loc[2000, components]
            total = filtered_fractions.sum()
            if total != 0:
                normalized_fractions = filtered_fractions / total
            else:
                normalized_fractions = pd.Series(0.0, index=filtered_fractions.index)
            return WasteFractions(**normalized_fractions.to_dict())

        scenario_parameters.div_component_fractions = DivComponentFractions(
            compost=calculate_component_fractions(waste_fractions, 'compost'),
            anaerobic=calculate_component_fractions(waste_fractions, 'anaerobic'),
            combustion=calculate_component_fractions(waste_fractions, 'combustion'),
            recycling=calculate_component_fractions(waste_fractions, 'recycling'),
        )
        scenario_parameters.non_compostable_not_targeted_total = sum(
            [self.non_compostable_not_targeted[x] * \
            getattr(scenario_parameters.div_component_fractions.compost, x) for x in self.div_components['compost']])
        scenario_parameters.non_compostable_not_targeted_total = pd.Series(scenario_parameters.non_compostable_not_targeted_total, index=years)
        if scenario_parameters.non_compostable_not_targeted_total.isna().all():
            scenario_parameters.non_compostable_not_targeted_total = pd.Series(0, index=years)
        self._calculate_diverted_masses(scenario=scenario) # This function could be moved to cityparameters class, and then it doesn't need scenario argument

        #scenario_parameters.repopulate_attr_dicts()
        self._check_masses_v2(scenario=scenario, advanced_baseline=True)

        if scenario_parameters.input_problems:
            raise ValueError('Invalid new values')

        self._calculate_net_masses(scenario=scenario, advanced_baseline=True)
        if (scenario_parameters.net_masses < 0).any().any():
            raise ValueError('Invalid new values')
            return

        scenario_parameters.divs_df = DivsDF.create_advanced_baseline(
            scenario_parameters.divs, 
            scenario_parameters.year_of_data_pop['baseline'], 
            scenario_parameters.growth_rate_historic, 
            scenario_parameters.growth_rate_future
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

        #scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in scenario_parameters.landfills:
            landfill.estimate_emissions()

        self.estimate_diversion_emissions(scenario=scenario)
        self.sum_landfill_emissions(scenario=scenario)

        # ADD WASTE BURNING EMISSIONS
        if waste_burning > 0:
            scenario_parameters.waste_burning_emissions = waste_burned * 3.7 * 1000 / 1000 / 1000 # g ch4 / kg waste to ton ch4 / ton waste
            scenario_parameters.total_emissions['total'] += scenario_parameters.waste_burning_emissions

    async def adst_prepopulate(
        self,
        latlon: str,
        DB_SERVER_IP: str,
        DB_PORT: int,
        DB_USER: str,
        DB_PASSWORD: str,
        DB_NAME: str,
        DB_SSLMODE: str
    ):
        parameters = CityParameters()
        geolocator = Nominatim(user_agent="karl_dilkington")
        location = geolocator.reverse((latlon[0], latlon[1]), language="en")
        country = location.raw['address'].get('country')
        try:
            iso3 = pycountry.countries.search_fuzzy(country)[0].alpha_3
        except LookupError:
            raise ValueError(f"Country '{country}' not found.")
        region = defaults_2019.region_lookup_iso3.get(iso3)
        if region is None:
            raise ValueError(f"Region for ISO3 code '{iso3}' not found.")

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

        # Ensure the database server is reachable via its IP and port
        try:
            socket.create_connection((DB_SERVER_IP, DB_PORT), timeout=5)
        except socket.error as e:
            raise HTTPException(status_code=500, detail=f"Cannot reach database at {DB_SERVER_IP}:{DB_PORT}: {e}")

        # Connect asynchronously to the PostgreSQL database using asyncpg
        conn = await asyncpg.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            host=DB_SERVER_IP,
            port=DB_PORT,
            ssl=DB_SSLMODE
        )

        # Execute the query with the latitude and longitude from latlon
        rows = await conn.fetch(QUERY_WEATHER, latlon[0], latlon[1])

        # Close the connection
        await conn.close()

        # Convert the asyncpg Record objects into a list of dictionaries
        weather_data = [dict(row) for row in rows][0]

        # Waste fractions
        if iso3 in defaults_2019.waste_fractions_country:
            waste_fractions = defaults_2019.waste_fractions_country.loc[iso3, :]
        else:
            waste_fractions = defaults_2019.waste_fraction_defaults.loc[region, :]

        # Normalize the waste fractions so that they sum to 1.
        waste_fractions = waste_fractions / waste_fractions.sum()
        years = pd.Index(range(1960, 2074))
        waste_fractions_df = pd.DataFrame(
            np.tile(waste_fractions.values, (len(years), 1)),
            index=years,
            columns=waste_fractions.index
        )
        try:
            parameters.precip = weather_data['avg_total_precip']
            parameters.temperature = weather_data['avg_temperature']
        except:
            parameters.precip = 999
            parameters.temperature = 15
        
        parameters.waste_fractions = waste_fractions_df
        parameters._singapore_k(advanced_baseline=True)

        wf_out = waste_fractions_df.iloc[0].to_dict()

        growth_rate = defaults_2019.growth_rate_country[iso3] / 100

        return {
            'temperature': parameters.temperature,
            'precipitation': parameters.precip,
            'waste_fractions': wf_out,
            'degredation_constant_k': float(parameters.ks.food.iat[0]),
            'growth_rate': growth_rate
        }

#%%

# # Initialize the City instance
# city = City(city_name="ExampleCity")

# # Example input parameters
# #country = "Argentina"
# country = "Netherlands"
# population = 872680
# precipitation = 800.0  # in mm/year

# # Initialize baseline scenario
# city.dst_baseline_blank(country, population, precipitation)

# # Access baseline parameters
# baseline = city.baseline_parameters
# print(f"Baseline Population: {baseline.population}")
# print(f"Baseline Precipitation Zone: {baseline.precip_zone}")
# print(f"Baseline Waste Mass: {baseline.waste_mass.iloc[0]} tons/year")

# %%
