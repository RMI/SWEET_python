from pydantic import BaseModel, validator
from typing import List, Dict, Union, Any, Set, Optional
import pandas as pd
import numpy as np
import defaults_2019
import pycountry # What am i using this for...seems dumb
from SWEET_python.class_defs import *
import inspect
import copy
from landfill import Landfill

class CityParameters(BaseModel):
    waste_fractions: WasteFractions
    div_fractions: DiversionFractions
    split_fractions: SplitFractions
    div_component_fractions: DivComponentFractions
    precip: float
    growth_rate_historic: float
    growth_rate_future: float
    waste_per_capita: float
    precip_zone: str
    ks: Optional[DecompositionRates] = None
    gas_capture_efficiency: float
    mef_compost: float
    waste_mass: float
    landfills: Optional[List[Landfill]] = None
    non_zero_landfills: Optional[List[Landfill]] = None
    non_compostable_not_targeted_total: float
    waste_masses: Optional[WasteMasses] = None
    divs: Optional[DivMasses] = None
    year_of_data_pop: Optional[int] = None
    scenario: Optional[int] = 0
    implement_year: Optional[int] = None
    organic_emissions: Optional[pd.DataFrame] = None
    landfill_emissions: Optional[pd.DataFrame] = None
    diversion_emissions: Optional[pd.DataFrame] = None
    total_emissions: Optional[pd.DataFrame] = None
    adjusted_diversion_constituents: bool = False
    input_problems: bool = False
    divs_df: Optional[pd.DataFrame] = None
    waste_generated_df: Optional[WasteGeneratedDF] = None
    city_instance_attrs: Optional[Dict[str, Any]] = None
    population: Optional[int] = None
    temp: Optional[float] = None 

    class Config:
        arbitrary_types_allowed = True

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

    def _singapore_k(self) -> None:
        """
        Calculates and sets k values for the city based on the Singapore method.
        """
        # Start with kc, which accounts for waste composition
        # nb = self.waste_fractions['metal'] + self.waste_fractions['glass'] + self.waste_fractions['plastic'] + self.waste_fractions['other'] + self.waste_fractions['rubber']
        # bs = self.waste_fractions['wood'] + self.waste_fractions['paper_cardboard'] + self.waste_fractions['textiles']
        # bf = self.waste_fractions['food'] + self.waste_fractions['green']

        nb = self.waste_fractions.metal + self.waste_fractions.glass + self.waste_fractions.plastic + self.waste_fractions.other + self.waste_fractions.rubber
        bs = self.waste_fractions.wood + self.waste_fractions.paper_cardboard + self.waste_fractions.textiles
        bf = self.waste_fractions.food + self.waste_fractions.green

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
        
        # invalid_combinations = []
        # # Iterate over bs, bf, nb in increments of 0.01
        # step = 0.01
        # for bs in np.arange(0, 1 + step, step):
        #     for bf in np.arange(0, 1 - bs + step, step):
        #         nb = 1 - bs - bf
        #         if nb < 0 or nb > 1:
        #             continue

        #         bs_idx = int(bs * 8)
        #         bf_idx = int(bf * 8)
        #         nb_idx = int(nb * 8)

        #         # Adjust indices if they are at the boundary
        #         if nb_idx == 8:
        #             nb_idx = 7
        #         if bs_idx == 8:
        #             bs_idx = 7
        #         if bf_idx == 8:
        #             bf_idx = 7

        #         kc = lookup_array[bs_idx, bf_idx, nb_idx]
        #         if kc == 0:
        #             invalid_combinations.append((bs, bf, nb))

        # # Print invalid combinations
        # for combo in invalid_combinations:
        #     print(f"Invalid combination: bs={combo[0]:.2f}, bf={combo[1]:.2f}, nb={combo[2]:.2f}")

        # ft, accounts for temperature
        tmin = 0
        tmax = 55
        topt = 35
        self.temp = 18
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

        self.ks = DecompositionRates(
            food=kc * tf * fm,
            green=kc * tf * fm,
            wood=kc * tf * fm,
            paper_cardboard=kc * tf * fm,
            textiles=kc * tf * fm
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
        self.baseline_parameters = None
        self.scenario_parameters = {}
        self.components = {'food', 'green', 'wood', 'paper_cardboard', 'textiles'}
        self.div_components = {
            'compost': {'food', 'green', 'wood', 'paper_cardboard'},
            'anaerobic': {'food', 'green', 'wood', 'paper_cardboard'},
            'combustion': {'food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'},
            'recycling': {'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'}
        }
        self.waste_types = {'food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'metal', 'glass', 'rubber', 'other'}
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

    def _calculate_divs(self) -> None:
        
        city_parameters = self.baseline_parameters
        city_parameters._singapore_k()

        # Create city-level dataframes
        start_year = 1960
        end_year = 2073
        city_parameters.waste_generated_df = WasteGeneratedDF.create(city_parameters.waste_mass, city_parameters.waste_fractions, start_year, end_year, city_parameters.year_of_data_pop, city_parameters.growth_rate_historic, city_parameters.growth_rate_future)
        
        # if scenario == 0:
        #     self.baseline_parameters = city_parameters
        # else:
        #     self.scenario_parameters[scenario - 1] = city_parameters

        # Update other calculated attributes
        self._calculate_waste_masses()
        self._calculate_diverted_masses()
        city_parameters.divs_df = DivsDF.create(city_parameters.divs, start_year, end_year, city_parameters.year_of_data_pop, city_parameters.growth_rate_historic, city_parameters.growth_rate_future)
        self._calculate_net_masses()

        city_params_dict = self.update_cityparams_dict(city_parameters)

        landfill_w_capture = Landfill(
            open_date=1960, 
            close_date=2073, 
            site_type='landfill', 
            mcf=1, 
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
            mcf=1, 
            city_params_dict=city_params_dict, 
            city_instance_attrs=city_parameters.city_instance_attrs, 
            landfill_index=1, 
            fraction_of_waste=city_parameters.split_fractions.landfill_wo_capture, 
            gas_capture=False
        )
        dumpsite = Landfill(
            open_date=1960, 
            close_date=2073, 
            site_type='dumpsite', 
            mcf=0.4, 
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

        if isinstance(parameters.div_fractions.combustion, float):
            for div in parameters.div_component_fractions.model_fields:
                diverted_masses[div] = {}
                fracs = getattr(parameters.div_component_fractions, div)
                s = sum(fracs.__dict__.values())
                # Make sure the component fractions add up to 1
                if s != 0 and np.abs(1 - s) > 0.01:
                    print(s, 'problems', div)
                for waste in fracs.__fields__:
                    diverted_masses[div][waste] = (
                        parameters.waste_mass *
                        getattr(parameters.div_fractions, div) *
                        getattr(fracs, waste)
                    )
        else:
            for div in ['compost', 'anaerobic', 'recycling']:
                diverted_masses[div] = {}
                fracs = getattr(parameters.div_component_fractions, div)
                s = sum(fracs.__dict__.values())
                # Make sure the component fractions add up to 1
                if s != 0 and np.abs(1 - s) > 0.01:
                    print(s, 'problems', div)
                for waste in fracs.__fields__:
                    diverted_masses[div][waste] = (
                        parameters.waste_mass *
                        getattr(parameters.div_fractions, div) *
                        getattr(fracs, waste)
                    )

            diverted_masses['combustion'] = {}
            fracs = parameters.div_component_fractions.combustion
            s = sum(fracs.__dict__.values())
            # Make sure the component fractions add up to 1
            if s != 0 and np.abs(1 - s) > 0.01:
                print(s, 'problems', div)
            for waste in fracs.__fields__:
                diverted_masses['combustion'][waste] = {}
                for year in parameters.div_fractions.combustion.index:
                    diverted_masses['combustion'][waste][year] = (
                            parameters.waste_mass *
                            parameters.div_fractions.combustion.at[year] *
                            getattr(fracs, waste)
                        )
            diverted_masses['combustion'] = pd.DataFrame(diverted_masses['combustion'])

        # Reduce diverted masses by rejection rates
        for waste in self.div_components['compost']:
            diverted_masses['compost'][waste] *= (
                1 - parameters.non_compostable_not_targeted_total
            ) * (1 - self.unprocessable[waste])
        for waste in self.div_components['combustion']:
            diverted_masses['combustion'][waste] *= (1 - self.combustion_reject_rate)
        for waste in self.div_components['recycling']:
            diverted_masses['recycling'][waste] *= self.recycling_reject_rates[waste]

        if isinstance(parameters.div_fractions.combustion, float):
            divs = DivMasses(
                compost=WasteMasses(**diverted_masses['compost']),
                anaerobic=WasteMasses(**diverted_masses['anaerobic']),
                combustion=WasteMasses(**diverted_masses['combustion']),
                recycling=WasteMasses(**diverted_masses['recycling'])
            )
        else:
            divs = DivMasses(
                compost=WasteMasses(**diverted_masses['compost']),
                anaerobic=WasteMasses(**diverted_masses['anaerobic']),
                combustion=diverted_masses['combustion'],
                recycling=WasteMasses(**diverted_masses['recycling'])
            )

        # Save the results in the correct attribute
        parameters.divs = divs

    def dst_baseline_blank(self, country: str, population: int, precipitation: float) -> None:
        """
        Initializes the baseline scenario with given parameters.

        Args:
            country (str): The country name.
            population (int): Population of the city.
            precipitation (float): Average annual precipitation in mm/year.

        Returns:
            None
        """
        self.country = country
        self.iso3 = pycountry.countries.search_fuzzy(self.country)[0].alpha_3
        self.region = defaults_2019.region_lookup_iso3[self.iso3]
        self.population = population
        self.year_of_data_pop = 2022
        self.precip = precipitation
        self.precip_zone = defaults_2019.get_precipitation_zone(self.precip)

        # Hard coding global urban population growth because don't have specific data
        population_1950 = 751000000
        population_2020 = 4300000000
        population_2035 = 5300000000
        self.growth_rate_historic = (population_2020 / population_1950) ** (1 / (2020 - 1950))
        self.growth_rate_future = (population_2035 / population_2020) ** (1 / (2035 - 2020))

        # Get waste per capita
        if self.iso3 in defaults_2019.msw_per_capita_country:
            self.waste_per_capita = defaults_2019.msw_per_capita_country[self.iso3]
        else:
            self.waste_per_capita = defaults_2019.msw_per_capita_defaults[self.region]
        self.waste_mass = self.waste_per_capita * self.population / 1000 * 365

        # Get waste fractions
        if self.iso3 in defaults_2019.waste_fractions_country:
            waste_fractions = defaults_2019.waste_fractions_country.loc[self.iso3, :]
        else:
            waste_fractions = defaults_2019.waste_fraction_defaults.loc[self.region, :]
        self.waste_fractions = WasteFractions(**(waste_fractions / waste_fractions.sum()).to_dict())

        try:
            self.mef_compost = (0.0055 * self.waste_fractions.food / (self.waste_fractions.food + self.waste_fractions.green) + 
                                0.0139 * self.waste_fractions.green / (self.waste_fractions.food + self.waste_fractions.green)) * 1.1023 * 0.7
        except:
            self.mef_compost = 0

        # k values
        self.ks = defaults_2019.k_defaults[self.precip_zone]

        # WasteMAP is set up to use up to three landfills.
        # Determine how much waste goes to each landfill type.
        try:
            self.split_fractions = SplitFractions(
                dumpsite=defaults_2019.fraction_open_dumped_country.get(self.iso3, defaults_2019.fraction_open_dumped[self.region]),
                landfill_wo_capture=defaults_2019.fraction_landfilled_country.get(self.iso3, defaults_2019.fraction_landfilled[self.region]),
                landfill_w_capture=0
            )
        except:
            if self.region in defaults_2019.landfill_default_regions:
                self.split_fractions = SplitFractions(landfill_w_capture=0, landfill_wo_capture=1, dumpsite=0)
            else:
                self.split_fractions = SplitFractions(landfill_w_capture=0, landfill_wo_capture=0, dumpsite=1)

        # Normalize landfills to sum to 1
        split_total = sum(self.split_fractions.dict().values())
        self.split_fractions = SplitFractions(**{site: frac / split_total for site, frac in self.split_fractions.dict().items()})

        # Instantiate landfill class for each landfill type
        landfill_w_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions.landfill_w_capture, gas_capture=True, scenario=0)
        landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions.landfill_wo_capture, gas_capture=False, scenario=0)
        dumpsite = Landfill(self, 1960, 2073, 'dumpsite', 0.4, fraction_of_waste=self.split_fractions.dumpsite, gas_capture=False, scenario=0)

        non_zero_landfills = [x for x in [landfill_w_capture, landfill_wo_capture, dumpsite] if x.fraction_of_waste > 0]

        # Diversion fractions
        self.div_fractions = DiversionFractions(
            compost=defaults_2019.fraction_composted_country.get(self.iso3, defaults_2019.fraction_composted.get(self.region, 0.0)),
            combustion=defaults_2019.fraction_incinerated_country.get(self.iso3, defaults_2019.fraction_incinerated.get(self.region, 0.0)),
            anaerobic=0,
            recycling=0
        )

        # Normalize diversion fractions to sum to 1 if they are >1
        s = sum(x for x in self.div_fractions.dict().values())
        if s > 1:
            self.div_fractions = DiversionFractions(**{div: frac / s for div, frac in self.div_fractions.dict().items()})
        assert sum(x for x in self.div_fractions.dict().values()) <= 1, 'Diversion fractions sum to more than 1'

        # Calculate waste masses
        self.waste_masses = {waste: frac * self.waste_mass for waste, frac in self.waste_fractions.dict().items()}

        # Calculate waste types of diversions
        self.divs = {
            'compost': {},
            'anaerobic': {},
            'combustion': {},
            'recycling': {}
        }
        self.div_component_fractions = {
            'compost': WasteFractions(**self._calc_compost_vol(self.div_fractions.compost)[1]),
            'anaerobic': WasteFractions(**self._calc_anaerobic_vol(self.div_fractions.anaerobic)[1]),
            'combustion': WasteFractions(**self._calc_combustion_vol(self.div_fractions.combustion)[1]),
            'recycling': WasteFractions(**self._calc_recycling_vol(self.div_fractions.recycling)[1])
        }

        for div, fractions in self.div_component_fractions.items():
            self.divs[div] = {waste: self.waste_mass * self.div_fractions.model_dump()[div] * fractions.model_dump()[waste] for waste in fractions.model_dump()}

        # Fill 0s for waste types that are not included in a diversion type
        for c in self.waste_fractions.model_dump().keys():
            for div in self.divs:
                if c not in self.divs[div]:
                    self.divs[div][c] = 0

        # Adjust diversion masses if, for any waste types, more waste is diverted than generated
        self._check_masses_v2(self.div_fractions, self.div_component_fractions)
        if self.baseline_parameters.input_problems:
            print('input problems')
            return

        # Check that more waste is not diverted than generated
        self.net_masses_after_check = {}
        for waste in self.waste_masses.keys():
            net_mass = self.waste_masses[waste] - sum(self.divs[div][waste] for div in self.divs)
            self.net_masses_after_check[waste] = net_mass

        for waste in self.net_masses_after_check.values():
            if waste < -1:
                print(waste)
            assert waste >= -1, 'Waste diversion is net negative'

        for landfill in non_zero_landfills:
            landfill.estimate_emissions(baseline=True)
    
        self.organic_emissions_baseline = self.estimate_diversion_emissions(baseline=True)
        self.landfill_emissions_baseline, self.diversion_emissions_baseline, self.total_emissions_baseline = self.sum_landfill_emissions(baseline=True)

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

    def sum_landfill_emissions(self, scenario: int) -> None:
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
            landfill_emissions = [x.emissions.map(self.convert_methane_m3_to_ton_co2e) for x in parameters.non_zero_landfills]
        else:
            parameters = self.scenario_parameters[scenario - 1]
            organic_emissions = parameters.organic_emissions
            landfill_emissions = [x.emissions.map(self.convert_methane_m3_to_ton_co2e) for x in parameters.landfills]

        # Concatenate all emissions dataframes
        all_emissions = pd.concat(landfill_emissions, axis=0)

        # Group by the year index and sum the emissions for each year
        summed_landfill_emissions = all_emissions.groupby(all_emissions.index).sum()

        summed_landfill_emissions = summed_landfill_emissions / 28  # Convert from co2e to ch4

        # Update total
        summed_landfill_emissions.drop('total', axis=1, inplace=True)
        summed_landfill_emissions['total'] = summed_landfill_emissions.sum(axis=1)

        # Repeat with addition of diverted waste emissions
        all_emissions = pd.concat([all_emissions, organic_emissions.loc[:, list(self.components)]], axis=0)
        summed_emissions = all_emissions.groupby(all_emissions.index).sum()
        summed_emissions.drop('total', axis=1, inplace=True)
        summed_emissions['total'] = summed_emissions.sum(axis=1)
        summed_emissions /= 28

        summed_diversion_emissions = organic_emissions.loc[:, list(self.components)] / 28
        summed_diversion_emissions['total'] = summed_diversion_emissions.sum(axis=1)

        parameters.landfill_emissions = summed_landfill_emissions
        parameters.diversion_emissions = summed_diversion_emissions
        parameters.total_emissions = summed_emissions

    def _check_masses_v2(self, scenario: int, advanced: bool=False) -> None:
        """
        Adjusts diversion waste type fractions if more of a waste type is being diverted than generated.

        Args:
            scenario (int): Scenario index.
        """
        if scenario == 0:
            parameters = self.baseline_parameters
        else:
            parameters = self.scenario_parameters[scenario-1]

        if not advanced:
            div_fractions = parameters.div_fractions
            div_component_fractions = parameters.div_component_fractions

            components_multiplied_through = {}
            for div in div_component_fractions.model_fields:
                components_multiplied_through[div] = {}
                for waste in getattr(div_component_fractions, div).model_fields:
                    components_multiplied_through[div][waste] = getattr(div_fractions, div) * getattr(getattr(div_component_fractions, div), waste)

            net = {}
            negative_catcher = False
            for waste in parameters.waste_fractions.model_fields:
                s = sum(components_multiplied_through[div].get(waste, 0) for div in div_fractions.model_fields)
                net[waste] = getattr(parameters.waste_fractions, waste) - s
                if net[waste] < -1e-3:
                    negative_catcher = True

            if not negative_catcher:
                #divs = self._divs_from_component_fractions(div_fractions, div_component_fractions, scenario=scenario)
                #parameters.divs = divs
                parameters.adjusted_diversion_constituents = False
                parameters.input_problems = False
                return

            if sum(getattr(div_fractions, div) for div in div_fractions.model_fields) > 1:
                raise CustomError("INVALID_PARAMETERS", f"Diversions sum to {sum(getattr(div_fractions, div) for div in div_fractions.model_fields)}, but they must sum to 1 or less.")
            
            compostables = sum(getattr(parameters.waste_fractions, waste) for waste in ['food', 'green', 'wood', 'paper_cardboard'])
            if div_fractions.compost + div_fractions.anaerobic > compostables:
                raise CustomError("INVALID_PARAMETERS", f"Only food, green, wood, and paper/cardboard can be composted or anaerobically digested. Those waste types sum to {compostables}, but input values of compost and anaerobic digestion sum to {div_fractions.compost + div_fractions.anaerobic}.")

            for div in div_fractions.model_fields:
                fraction = getattr(div_fractions, div)
                s = sum(getattr(parameters.waste_fractions, waste) for waste in self.div_components[div])
                if s < fraction:
                    components = self.div_components[div]
                    values = [getattr(parameters.waste_fractions, x) for x in components]
                    raise CustomError("INVALID_PARAMETERS", f"{div} too high. {div} applies to {components}, which are {values} of total waste--the sum of these is {sum(values)}, so only that much waste can be {div}, but input value was {fraction}.")

            non_combustables = sum(getattr(parameters.waste_fractions, waste) for waste in ['glass', 'metal', 'other'])
            if div_fractions.compost + div_fractions.anaerobic + div_fractions.combustion > (1 - non_combustables):
                s = div_fractions.compost + div_fractions.anaerobic + div_fractions.combustion
                raise CustomError("INVALID_PARAMETERS", f"Glass, metal, and other account for {non_combustables:.3f} of waste, and they can only be recycled. {div_fractions.compost} compost, {div_fractions.anaerobic} anaerobic, and {div_fractions.combustion} incineration were specified, summing to {s}, but only {1 - non_combustables} of waste can be diverted to these diversion types.")

            non_combustion = {}
            combustion_all = {}
            keys_of_interest = ['compost', 'anaerobic', 'recycling']
            for waste in parameters.waste_fractions.model_fields:
                s = sum(components_multiplied_through[div].get(waste, 0) for div in keys_of_interest)
                non_combustion[waste] = s
                combustion_all[waste] = getattr(parameters.waste_fractions, waste) - s

            adjust_non_combustion = False
            for waste, frac in non_combustion.items():
                if frac > getattr(parameters.waste_fractions, waste):
                    adjust_non_combustion = True

            if adjust_non_combustion:
                div_component_fractions_adjusted = DivComponentFractions(**div_component_fractions.model_dump())

                dont_add_to = {waste for waste, frac in parameters.waste_fractions.model_dump().items() if frac == 0}
                problems = [set(waste for waste, frac in non_combustion.items() if frac > getattr(parameters.waste_fractions, waste))]
                dont_add_to.update(problems[0])

                while problems:
                    probs = problems.pop(0)
                    for waste in probs:
                        remove = {}
                        distribute = {}
                        overflow = {}
                        can_be_adjusted = []
                        div_total = sum(getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste) for div in keys_of_interest if waste in getattr(div_component_fractions_adjusted, div).model_fields)
                        div_target = getattr(parameters.waste_fractions, waste)
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

                    new_probs = {waste for waste in parameters.waste_fractions.model_fields if sum(getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste, 0) for div in keys_of_interest) > getattr(parameters.waste_fractions, waste) + 0.001}
                    if new_probs:
                        problems.append(new_probs)
                    dont_add_to.update(new_probs)

                components_multiplied_through = {
                    div: {waste: getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste) for waste in getattr(div_component_fractions_adjusted, div).model_fields}
                    for div in div_component_fractions_adjusted.model_fields
                }

            non_combustion = {}
            combustion_all = {}
            for waste in parameters.waste_fractions.model_fields:
                s = sum(components_multiplied_through[div].get(waste, 0) for div in keys_of_interest)
                non_combustion[waste] = s
                combustion_all[waste] = getattr(parameters.waste_fractions, waste) - s

            adjust_non_combustion = False
            for waste, frac in non_combustion.items():
                if frac > (getattr(parameters.waste_fractions, waste) + 1e-5):
                    adjust_non_combustion = True
                    raise CustomError("INVALID_PARAMETERS", f"Combination of compost, anaerobic digestion, and recycling is too high")

            all_divs = sum(getattr(div_fractions, div) for div in div_fractions.model_fields)

            assert np.abs(div_fractions.recycling - sum(components_multiplied_through['recycling'].values())) < 1e-3

            remainder = sum(fraction for waste_type, fraction in combustion_all.items() if waste_type in self.div_components['combustion'])
            combustion_fraction_of_remainder = div_fractions.combustion / remainder
            if combustion_fraction_of_remainder > (1 + 1e-5):
                non_combustables = [x for x in parameters.waste_fractions.model_fields if x not in self.div_components['combustion']]
                for waste in non_combustables:
                    if getattr(parameters.waste_fractions, waste) == 0:
                        continue
                    new_val = getattr(parameters.waste_fractions, waste) * all_divs
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
                for waste in parameters.waste_fractions.model_fields:
                    s = sum(components_multiplied_through[div].get(waste, 0) for div in keys_of_interest)
                    non_combustion[waste] = s
                    combustion_all[waste] = getattr(parameters.waste_fractions, waste) - s

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

            parameters.div_component_fractions = adjusted_div_component_fractions
            parameters.divs = divs
            parameters.adjusted_diversion_constituents = True
            parameters.input_problems = False
        
        else:
            
            # For this to work, I need to identify during which divset the implement year is. All divsets before that
            # only need to check against pre-implement, all divsets after should check against post-implement, and the
            # divset of the implement year should check against both.

            div_fractions = parameters.div_fractions
            div_component_fractions = parameters.div_component_fractions

            components_multiplied_through = {}
            for div in div_component_fractions.model_fields:
                components_multiplied_through[div] = {}
                for waste in getattr(div_component_fractions, div).model_fields:
                    components_multiplied_through[div][waste] = getattr(div_fractions, div) * getattr(getattr(div_component_fractions, div), waste)

            components_multiplied_through['combustion'] = pd.DataFrame(components_multiplied_through['combustion'])
            unique_divsets = components_multiplied_through['combustion'].drop_duplicates()

            div_component_fractions_adjusted = []
            divs = []

            for i in range(unique_divsets.shape[0]):
                divset = unique_divsets.iloc[i,:]
                components_multiplied_through_dummy = components_multiplied_through.copy()
                components_multiplied_through_dummy['combustion'] = {x: float(divset.at[x]) for x in divset.index}

                net = {}
                negative_catcher = False
                for waste in parameters.waste_fractions.model_fields:
                    s = sum(components_multiplied_through_dummy[div].get(waste, 0) for div in div_fractions.model_fields)
                    net[waste] = getattr(parameters.waste_fractions, waste) - s
                    if net[waste] < -1e-3:
                        negative_catcher = True

                if not negative_catcher:
                    #divs = self._divs_from_component_fractions(div_fractions, div_component_fractions, scenario=scenario)
                    #parameters.divs = divs
                    parameters.adjusted_diversion_constituents = False
                    parameters.input_problems = False
                    return

                if sum(getattr(div_fractions, div) for div in div_fractions.model_fields) > 1:
                    raise CustomError("INVALID_PARAMETERS", f"Diversions sum to {sum(getattr(div_fractions, div) for div in div_fractions.model_fields)}, but they must sum to 1 or less.")
                
                compostables = sum(getattr(parameters.waste_fractions, waste) for waste in ['food', 'green', 'wood', 'paper_cardboard'])
                if div_fractions.compost + div_fractions.anaerobic > compostables:
                    raise CustomError("INVALID_PARAMETERS", f"Only food, green, wood, and paper/cardboard can be composted or anaerobically digested. Those waste types sum to {compostables}, but input values of compost and anaerobic digestion sum to {div_fractions.compost + div_fractions.anaerobic}.")

                for div in div_fractions.model_fields:
                    fraction = getattr(div_fractions, div)
                    s = sum(getattr(parameters.waste_fractions, waste) for waste in self.div_components[div])
                    if s < fraction:
                        components = self.div_components[div]
                        values = [getattr(parameters.waste_fractions, x) for x in components]
                        raise CustomError("INVALID_PARAMETERS", f"{div} too high. {div} applies to {components}, which are {values} of total waste--the sum of these is {sum(values)}, so only that much waste can be {div}, but input value was {fraction}.")

                non_combustables = sum(getattr(parameters.waste_fractions, waste) for waste in ['glass', 'metal', 'other'])
                if div_fractions.compost + div_fractions.anaerobic + div_fractions.combustion > (1 - non_combustables):
                    s = div_fractions.compost + div_fractions.anaerobic + div_fractions.combustion
                    raise CustomError("INVALID_PARAMETERS", f"Glass, metal, and other account for {non_combustables:.3f} of waste, and they can only be recycled. {div_fractions.compost} compost, {div_fractions.anaerobic} anaerobic, and {div_fractions.combustion} incineration were specified, summing to {s}, but only {1 - non_combustables} of waste can be diverted to these diversion types.")

                non_combustion = {}
                combustion_all = {}
                keys_of_interest = ['compost', 'anaerobic', 'recycling']
                for waste in parameters.waste_fractions.model_fields:
                    s = sum(components_multiplied_through_dummy[div].get(waste, 0) for div in keys_of_interest)
                    non_combustion[waste] = s
                    combustion_all[waste] = getattr(parameters.waste_fractions, waste) - s

                adjust_non_combustion = False
                for waste, frac in non_combustion.items():
                    if frac > getattr(parameters.waste_fractions, waste):
                        adjust_non_combustion = True

                if adjust_non_combustion:
                    div_component_fractions_adjusted = DivComponentFractions(**div_component_fractions.model_dump())

                    dont_add_to = {waste for waste, frac in parameters.waste_fractions.model_dump().items() if frac == 0}
                    problems = [set(waste for waste, frac in non_combustion.items() if frac > getattr(parameters.waste_fractions, waste))]
                    dont_add_to.update(problems[0])

                    while problems:
                        probs = problems.pop(0)
                        for waste in probs:
                            remove = {}
                            distribute = {}
                            overflow = {}
                            can_be_adjusted = []
                            div_total = sum(getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste) for div in keys_of_interest if waste in getattr(div_component_fractions_adjusted, div).model_fields)
                            div_target = getattr(parameters.waste_fractions, waste)
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

                        new_probs = {waste for waste in parameters.waste_fractions.model_fields if sum(getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste, 0) for div in keys_of_interest) > getattr(parameters.waste_fractions, waste) + 0.001}
                        if new_probs:
                            problems.append(new_probs)
                        dont_add_to.update(new_probs)

                    components_multiplied_through_dummy = {
                        div: {waste: getattr(div_fractions, div) * getattr(getattr(div_component_fractions_adjusted, div), waste) for waste in getattr(div_component_fractions_adjusted, div).model_fields}
                        for div in div_component_fractions_adjusted.model_fields
                    }

                non_combustion = {}
                combustion_all = {}
                for waste in parameters.waste_fractions.model_fields:
                    s = sum(components_multiplied_through_dummy[div].get(waste, 0) for div in keys_of_interest)
                    non_combustion[waste] = s
                    combustion_all[waste] = getattr(parameters.waste_fractions, waste) - s

                adjust_non_combustion = False
                for waste, frac in non_combustion.items():
                    if frac > (getattr(parameters.waste_fractions, waste) + 1e-5):
                        adjust_non_combustion = True
                        raise CustomError("INVALID_PARAMETERS", f"Combination of compost, anaerobic digestion, and recycling is too high")

                all_divs = sum(getattr(div_fractions, div) for div in div_fractions.model_fields)

                assert np.abs(div_fractions.recycling - sum(components_multiplied_through_dummy['recycling'].values())) < 1e-3

                remainder = sum(fraction for waste_type, fraction in combustion_all.items() if waste_type in self.div_components['combustion'])
                combustion_fraction_of_remainder = div_fractions.combustion / remainder
                if combustion_fraction_of_remainder > (1 + 1e-5):
                    non_combustables = [x for x in parameters.waste_fractions.model_fields if x not in self.div_components['combustion']]
                    for waste in non_combustables:
                        if getattr(parameters.waste_fractions, waste) == 0:
                            continue
                        new_val = getattr(parameters.waste_fractions, waste) * all_divs
                        components_multiplied_through_dummy['recycling'][waste] = new_val
                    
                    available_div = sum(v for k, v in components_multiplied_through_dummy['recycling'].items() if k not in non_combustables)
                    available_div_target = div_fractions.recycling - sum(v for k, v in components_multiplied_through_dummy['recycling'].items() if k in non_combustables)
                    if available_div_target < 0:
                        too_much_frac = (sum(v for k, v in components_multiplied_through_dummy['recycling'].items() if k in non_combustables) - div_fractions.recycling) / sum(v for k, v in components_multiplied_through_dummy['recycling'].items() if k in non_combustables)
                        for key, value in components_multiplied_through_dummy['recycling'].items():
                            if key in non_combustables:
                                components_multiplied_through_dummy['recycling'][key] = value * (1 - too_much_frac)
                            else:
                                components_multiplied_through_dummy['recycling'][key] = 0
                        assert np.abs(div_fractions.recycling - sum(v for v in components_multiplied_through_dummy['recycling'].values())) < 1e-5

                    else:
                        reduce_frac = (available_div - available_div_target) / available_div
                        for key, value in components_multiplied_through_dummy['recycling'].items():
                            if key not in non_combustables:
                                components_multiplied_through_dummy['recycling'][key] = value * (1 - reduce_frac)
                        assert np.abs(div_fractions.recycling - sum(v for v in components_multiplied_through_dummy['recycling'].values())) < 1e-5

                    non_combustion = {}
                    combustion_all = {}
                    for waste in parameters.waste_fractions.model_fields:
                        s = sum(components_multiplied_through_dummy[div].get(waste, 0) for div in keys_of_interest)
                        non_combustion[waste] = s
                        combustion_all[waste] = getattr(parameters.waste_fractions, waste) - s

                    remainder = sum(fraction for waste_type, fraction in combustion_all.items() if waste_type in self.div_components['combustion'])
                    combustion_fraction_of_remainder = div_fractions.combustion / remainder
                    assert combustion_fraction_of_remainder < (1 + 1e-5)
                    if combustion_fraction_of_remainder > 1:
                        combustion_fraction_of_remainder = 1

                for waste in self.div_components['combustion']:
                    components_multiplied_through_dummy['combustion'][waste] = combustion_fraction_of_remainder * combustion_all[waste]

                for d in div_fractions.model_fields:
                    assert np.abs(getattr(div_fractions, d) - sum(components_multiplied_through_dummy[d].values())) < 1e-3
                    for w in components_multiplied_through_dummy[d]:
                        if abs(components_multiplied_through_dummy[d][w]) < 1e-5:
                            components_multiplied_through_dummy[d][w] = 0
                        assert components_multiplied_through_dummy[d][w] >= 0

                adjusted_div_component_fractions = {
                    div: {waste: components_multiplied_through_dummy[div][waste] / getattr(div_fractions, div) if getattr(div_fractions, div) != 0 else 0 for waste in components_multiplied_through_dummy[div]}
                    for div in components_multiplied_through_dummy
                }

                adjusted_div_component_fractions = DivComponentFractions(**adjusted_div_component_fractions)

                divs_adj = self._divs_from_component_fractions(div_fractions, adjusted_div_component_fractions, scenario=scenario)
                divs.append(divs_adj)
                div_component_fractions_adjusted.append(adjusted_div_component_fractions)

            parameters.div_component_fractions = adjusted_div_component_fractions
            parameters.divs = divs
            parameters.adjusted_diversion_constituents = True
            parameters.input_problems = False

    def _divs_from_component_fractions(self, div_fractions: DiversionFractions, div_component_fractions: DivComponentFractions, scenario: int) -> dict:
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

        waste_mass = self.baseline_parameters.waste_mass
        non_compostable_not_targeted_total = parameters.non_compostable_not_targeted_total

        compost_masses = {}
        anaerobic_masses = {}
        combustion_masses = {}
        recycling_masses = {}

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

    def _calculate_net_masses(self, scenario: int=0, advanced: bool=False) -> WasteMasses:
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

        if advanced:
            wastefractions_before = self.baseline_parameters.waste_fractions
            wastefractions_after = parameters.waste_fractions

            unique_divsets = divs.combustion.drop_duplicates()
            incineration_implement_year = unique_divsets.index[1]
            try:
                incineration_end_year = unique_divsets.index[2]
                years_to_check = sorted(list(set(
                    implement_year - 1,
                    implement_year + 1,
                    incineration_implement_year - 1,
                    incineration_implement_year + 1,
                    incineration_end_year - 1,
                    incineration_end_year + 1
                )))
            except:
                incineration_end_year = None
                years_to_check = sorted(list(set(
                    implement_year - 1,
                    implement_year + 1,
                    incineration_implement_year - 1,
                    incineration_implement_year + 1,
                )))



        net_masses = {waste: parameters.waste_masses.model_dump()[waste] - (
                        getattr(divs.compost, waste) +
                        getattr(divs.anaerobic, waste) +
                        getattr(divs.combustion, waste) +
                        getattr(divs.recycling, waste)
                    ) for waste in parameters.waste_fractions.model_dump()}

        net = WasteMasses(**net_masses)

        return net

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

        scenario_parameters.non_zero_landfills = [lf for lf in scenario_parameters.landfills if lf.fraction_of_waste > 0]
        scenario_parameters.implement_year = implement_year

        # Recalculate div_component_fractions
        waste_fractions = scenario_parameters.waste_fractions

        def calculate_component_fractions(waste_fractions: WasteFractions, div_type: str) -> WasteFractions:
            components = self.div_components[div_type]
            filtered_fractions = {waste: getattr(waste_fractions, waste) for waste in components}
            total = sum(filtered_fractions.values())
            normalized_fractions = {waste: fraction / total for waste, fraction in filtered_fractions.items()}
            return WasteFractions(**{waste: normalized_fractions.get(waste, 0) for waste in waste_fractions.model_fields})

        scenario_parameters.div_component_fractions = DivComponentFractions(
            compost=calculate_component_fractions(waste_fractions, 'compost'),
            anaerobic=calculate_component_fractions(waste_fractions, 'anaerobic'),
            combustion=calculate_component_fractions(waste_fractions, 'combustion'),
            recycling=calculate_component_fractions(waste_fractions, 'recycling'),
        )
        scenario_parameters.non_compostable_not_targeted_total = sum(
            [self.non_compostable_not_targeted[x] * \
             getattr(scenario_parameters.div_component_fractions.compost, x) for x in self.div_components['compost']])
        self._calculate_diverted_masses(scenario=scenario) # This function could be moved to cityparameters class, and then it doesn't need scenario argument

        #scenario_parameters.repopulate_attr_dicts()
        self._check_masses_v2(scenario=scenario)

        if scenario_parameters.input_problems:
            print(f'Invalid new value')
            return

        net = self._calculate_net_masses(scenario=scenario)
        for mass in vars(net).values():
            if mass < 0:
                print(f'Invalid new value')
                return

        scenario_parameters.divs_df._dst_implement(
            implement_year=implement_year, 
            scenario_div_masses=scenario_parameters.divs, 
            baseline_div_masses=self.baseline_parameters.divs, 
            start_year=1960, 
            end_year=2073, 
            year_of_data_pop=scenario_parameters.year_of_data_pop, 
            growth_rate_historic=scenario_parameters.growth_rate_historic, 
            growth_rate_future=scenario_parameters.growth_rate_future,
            components=self.components
        )

        # combine these two loops maybe...though it still does six things, maybe doesn't matter
        scenario_parameters.repopulate_attr_dicts()
        for i, landfill in enumerate(scenario_parameters.landfills):
            # Might be able to do this more efficienctly...i'm looping over the pre implementation years twice sort of
            landfill.waste_mass_df = LandfillWasteMassDF.create(scenario_parameters.waste_generated_df.df, scenario_parameters.divs_df, landfill.fraction_of_waste, self.components).df
            landfill.waste_mass_df.loc[:(implement_year-1), :] = self.baseline_parameters.landfills[i].waste_mass_df.loc[:(implement_year-1), :]
            landfill.waste_mass_df.to_csv('/Users/hugh/Library/CloudStorage/OneDrive-RMI/Documents/RMI/scratch_paper/new' + str(i) + '.csv')

        #scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in scenario_parameters.non_zero_landfills:
            landfill.estimate_emissions()

        self.estimate_diversion_emissions(scenario=scenario)
        self.sum_landfill_emissions(scenario=scenario)

    def implement_dst_changes_advanced(
        self,
        population: float,
        precipitation: float,
        new_waste_fractions: WasteFractions,
        new_div_fractions: DiversionFractions,
        new_landfill_types: List,
        new_landfill_open_close_dates: List,
        implement_year: float,
        scenario: int,
        new_baseline: int,
        fraction_waste_timeline: Dict,
        new_gas_efficiency: List,
        new_landfill_fracs: List = None,
        new_landfill_flaring: List = None,
        new_landfill_cover: List = None,
        new_landfill_leachate_circulate: List = None,
    ) -> None:
        
        scenario_parameters = copy.deepcopy(self.baseline_parameters)
        self.scenario_parameters[scenario - 1] = scenario_parameters
        scenario_parameters.div_fractions = new_div_fractions
        scenario_parameters.waste_fractions = new_waste_fractions
        scenario_parameters._singapore_k()

        # REMOVE THIS LATER
        new_waste_mass = scenario_parameters.waste_mass

        # New waste masses
        waste_masses = {waste: frac * new_waste_mass for waste, frac in new_waste_fractions.model_dump().items()}
        scenario_parameters.waste_masses = WasteMasses(**waste_masses)

        # Update waste generated
        scenario_parameters.waste_generated_df.dst_implement_advanced(
            df=self.baseline_parameters.waste_generated_df.df,
            implement_year=implement_year, 
            new_waste_mass=new_waste_mass, 
            new_waste_fractions=new_waste_fractions, 
            year_of_data_pop=scenario_parameters.year_of_data_pop, 
            growth_rate_historic=scenario_parameters.growth_rate_historic, 
            growth_rate_future=scenario_parameters.growth_rate_future
        )

        # Create a DataFrame for fraction_waste_timeline
        fraction_df = pd.DataFrame(fraction_waste_timeline).transpose()
        fraction_df.columns = [f'Landfill_{i}' for i in range(fraction_df.shape[1])]
        fraction_df.index.name = 'Year'

        # Set up new landfills
        city_params_dict = self.update_cityparams_dict(scenario_parameters)
        mcfs = [1, 0.7, 0.4] # Should this include ameliorated? 
        landfill_types = ['landfill', 'controlled_dumpsite', 'dumpsite']
        
        for i, lf_type in enumerate(new_landfill_types):
            if i <= 2:
                new_landfill = copy.deepcopy(self.baseline_parameters.landfills[i])
                new_landfill.close_date = new_landfill_open_close_dates[i][1]
                old_type = new_landfill.site_type
                new_type = landfill_types[lf_type]
                if new_type != old_type:
                    new_landfill.ameliorated = True
                new_landfill.site_type = new_type
                new_landfill.mcf = mcfs[lf_type]
                # not sure if these two are necessary or if they'll be taken care of other ways.
                new_landfill.city_params_dict = city_params_dict
                new_landfill.city_instance_attrs = scenario_parameters.city_instance_attrs
                new_landfill.gas_capture = False if new_gas_efficiency[i] == 0 else True
                new_landfill.scenario = scenario
                new_landfill.new_baseline = new_baseline
                new_landfill.gas_capture_efficiency = new_gas_efficiency[i]
                if new_landfill.flaring:
                    new_landfill.flaring = new_landfill_flaring[i]
                    new_landfill.cover = new_landfill_cover[i]
                    new_landfill.leachate_circulate = new_landfill_leachate_circulate[i]
                new_landfill.fraction_of_waste_vector = fraction_df[f'Landfill_{i}']
                scenario_parameters.landfills[i] = new_landfill
            else:
                new_landfill = Landfill(
                    open_date=new_landfill_open_close_dates[i][0], 
                    close_date=new_landfill_open_close_dates[i][1], 
                    site_type=landfill_types[lf_type], 
                    mcf=mcfs[lf_type],
                    city_params_dict=city_params_dict,
                    city_instance_attrs=scenario_parameters.city_instance_attrs,
                    landfill_index=i, 
                    fraction_of_waste=new_landfill_fracs[i], 
                    gas_capture=False if new_gas_efficiency[i] == 0 else True,
                    scenario=scenario,
                    new_baseline=new_baseline,
                    gas_capture_efficiency=new_gas_efficiency[i],
                    flaring=new_landfill_flaring[i],
                    cover=new_landfill_cover[i],
                    leachate_circulate=new_landfill_leachate_circulate[i],
                    fraction_of_waste_vector=fraction_df[f'Landfill_{i}'],
                    advaned=True
                )
                scenario_parameters.landfills[i] = new_landfill

        scenario_parameters.implement_year = implement_year

        # Recalculate div_component_fractions
        waste_fractions = scenario_parameters.waste_fractions

        def calculate_component_fractions(waste_fractions: WasteFractions, div_type: str) -> WasteFractions:
            components = self.div_components[div_type]
            filtered_fractions = {waste: getattr(waste_fractions, waste) for waste in components}
            total = sum(filtered_fractions.values())
            normalized_fractions = {waste: fraction / total for waste, fraction in filtered_fractions.items()}
            return WasteFractions(**{waste: normalized_fractions.get(waste, 0) for waste in waste_fractions.model_fields})

        scenario_parameters.div_component_fractions = DivComponentFractions(
            compost=calculate_component_fractions(waste_fractions, 'compost'),
            anaerobic=calculate_component_fractions(waste_fractions, 'anaerobic'),
            combustion=calculate_component_fractions(waste_fractions, 'combustion'),
            recycling=calculate_component_fractions(waste_fractions, 'recycling'),
        )
        scenario_parameters.non_compostable_not_targeted_total = sum(
            [self.non_compostable_not_targeted[x] * \
             getattr(scenario_parameters.div_component_fractions.compost, x) for x in self.div_components['compost']])
        self._calculate_diverted_masses(scenario=scenario) # This function could be moved to cityparameters class, and then it doesn't need scenario argument

        #scenario_parameters.repopulate_attr_dicts()
        self._check_masses_v2(scenario=scenario, advanced=True)

        if scenario_parameters.input_problems:
            print(f'Invalid new value')
            return

        net = self._calculate_net_masses(scenario=scenario)
        for mass in vars(net).values():
            if mass < 0:
                print(f'Invalid new value')
                return

        scenario_parameters.divs_df._dst_implement(
            implement_year=implement_year, 
            scenario_div_masses=scenario_parameters.divs, 
            baseline_div_masses=self.baseline_parameters.divs, 
            start_year=1960, 
            end_year=2073, 
            year_of_data_pop=scenario_parameters.year_of_data_pop, 
            growth_rate_historic=scenario_parameters.growth_rate_historic, 
            growth_rate_future=scenario_parameters.growth_rate_future,
            components=self.components
        )

        # combine these two loops maybe...though it still does six things, maybe doesn't matter
        scenario_parameters.repopulate_attr_dicts()
        for i, landfill in enumerate(scenario_parameters.landfills):
            # Might be able to do this more efficienctly...i'm looping over the pre implementation years twice sort of
            landfill.waste_mass_df.dst_implement_advanced(
                waste_generated_df=scenario_parameters.waste_generated_df.df,
                divs_df=scenario_parameters.divs_df, 
                fraction_of_waste_series=landfill.fraction_of_waste_vector,
                waste_types=self.components,
            )

            #landfill.waste_mass_df.loc[:(implement_year-1), :] = self.baseline_parameters.landfills[i].waste_mass_df.loc[:(implement_year-1), :]
            #landfill.waste_mass_df.to_csv('/Users/hugh/Library/CloudStorage/OneDrive-RMI/Documents/RMI/scratch_paper/new' + str(i) + '.csv')

        #scenario_parameters.repopulate_attr_dicts() # does this need to come sooner? Does anything in the above functions rely on the attr dicts?
        for landfill in scenario_parameters.landfills:
            landfill.estimate_emissions()

        self.estimate_diversion_emissions(scenario=scenario)
        self.sum_landfill_emissions(scenario=scenario)


