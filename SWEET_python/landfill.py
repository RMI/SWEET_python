import defaults_2019
from class_defs import *
import pandas as pd
from model_v2 import SWEET
from typing import List, Dict, Union, Any, Set, Optional
import time
import numpy as np


class Landfill:
    def __init__(self,
                 open_date: int, 
                 close_date: int, 
                 site_type: str, 
                 mcf: float,
                 city_params_dict: dict,
                 city_instance_attrs: dict,
                 landfill_index: int = 0, 
                 fraction_of_waste: float = 1, 
                 gas_capture: bool = False, 
                 scenario: int = 0):
        """
        Initializes a Landfill object.

        Args:
            name (str): Name of the city.
            components (set): Components of the city.
            div_components (dict): Diversion components of the city.
            waste_types (set): Waste types in the city.
            unprocessable (dict): Unprocessable waste fractions.
            non_compostable_not_targeted (dict): Non-compostable not targeted fractions.
            combustion_reject_rate (float): Combustion reject rate.
            recycling_reject_rates (dict): Recycling reject rates.
            open_date (int): Opening date of the landfill.
            close_date (int): Closing date of the landfill.
            site_type (str): Type of the landfill site.
            mcf (float): Methane correction factor.
            fraction_of_waste (float, optional): Fraction of the waste for the landfill. Defaults to 1.
            gas_capture (bool, optional): Indicates if gas capture system is present. Defaults to False.
        """
        # Some of these are in the city_instance_attrs, so I can remove them from here
        self.open_date = open_date
        self.close_date = close_date
        self.site_type = site_type
        self.mcf = mcf
        self.fraction_of_waste = fraction_of_waste
        self.gas_capture = gas_capture
        self.landfill_index = landfill_index
        self.scenario = scenario
        self.city_instance_attrs = city_instance_attrs
        self.city_params_dict = city_params_dict

        if self.gas_capture:
            self.gas_capture_efficiency = defaults_2019.gas_capture_efficiency[site_type]
            self.oxidation_factor = defaults_2019.oxidation_factor['with_lfg'][site_type]
        else:
            self.gas_capture_efficiency = 0
            self.oxidation_factor = defaults_2019.oxidation_factor['without_lfg'][site_type]

        self.waste_mass = None
        self.emissions = None
        self.ch4 = None
        self.captured = None

        self.waste_mass_df = LandfillWasteMassDF.create(self.city_params_dict['waste_generated_df'], self.city_params_dict['divs_df'], self.fraction_of_waste, self.city_instance_attrs['components']).df

    # def update_landfill_attrs_dict(self, city_parameters: dict) -> None:
    #     """
    #     Updates the landfill parameters dictionary with new values.

    #     Args:
    #         city_params_dict (dict): The dictionary containing the new values.

    #     Returns:
    #         None
    #     """
    #     landfill_instance_attrs = self.model_dump()
    #     keys_to_remove = ['model']
    #     for key in keys_to_remove:
    #         if key in landfill_instance_attrs:
    #             del landfill_instance_attrs[key]

    #     self.model.landfill_instance_attrs = landfill_instance_attrs

    #     return
    
    def model_dump(self) -> dict:
        """
        Dumps the model attributes into a dictionary.

        Returns:
            dict: Dictionary containing the model attributes.
        """

        d = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        if 'model' in d:
            del d['model']

        return d
    
    def model_dump_for_serialization(self) -> dict:
        def convert_sets_to_lists(data):
            if isinstance(data, dict):
                return {self._convert_key(k): convert_sets_to_lists(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_sets_to_lists(v) for v in data]
            elif isinstance(data, set):
                return list(data)
            elif isinstance(data, pd.DataFrame):
                return data.to_dict(orient='records')
            elif isinstance(data, (np.int64, np.float64)):
                return data.item()
            elif isinstance(data, SWEET):
                return data.model_dump_for_serialization()
            else:
                return data

        d = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        if 'city_instance_attrs' in d:
            d['city_instance_attrs'] = {}
        if 'city_params_dict' in d:
            d['city_params_dict'] = {}
        if 'model' in d:
            d['model'] = {}

        return convert_sets_to_lists(d)

    def _convert_key(self, key):
        if isinstance(key, (np.int64, np.float64)):
            return key.item()
        return key

    def estimate_emissions(self) -> tuple:
        """
        Estimate emissions using an instance of the SWEET class.

        Args:
            baseline (bool, optional): If True, estimates baseline emissions. If False, uses new values. Defaults to True.
        """
        if hasattr(self, 'model'):
            # These are maybe not necessary if I did repopulate already?
            self.model.city_instance_attrs = self.city_instance_attrs
            self.model.city_params_dict = self.city_params_dict
            self.model.landfill_instance_attrs = self.model_dump()
        else:
            self.model = SWEET(
                city_instance_attrs=self.city_instance_attrs,
                city_params_dict=self.city_params_dict,
                landfill_instance_attrs=self.model_dump()
            )
        
        start_time = time.time()
        self.waste_mass, self.emissions, self.ch4, self.captured = self.model.estimate_emissions()
        end_time = time.time()
        #print(f"Time taken to estimate emissions in Landfill: {end_time - start_time} seconds")

    # def _convert_df_to_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
    #     return df.to_dict(orient='records') if df is not None else None

    # def to_dict(self) -> Dict[str, Any]:
    #     return self.model_dump()