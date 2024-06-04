from city_dict import cities_to_run
import defaults_2019
from class_defs import *
import pandas as pd
from model_v2 import SWEET
from typing import List, Dict, Union, Any, Set, Optional

class Landfill:
    def __init__(self, 
                 city_name: str,
                 open_date: int, 
                 close_date: int, 
                 site_type: str, 
                 mcf: float, 
                 landfill_index: int = 0, 
                 fraction_of_waste: float = 1, 
                 gas_capture: bool = False, 
                 scenario: int = 0,
                 ):
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
        self.city_name = city_name
        self.open_date = open_date
        self.close_date = close_date
        self.site_type = site_type
        self.mcf = mcf
        self.fraction_of_waste = fraction_of_waste
        self.gas_capture = gas_capture
        self.landfill_index = landfill_index
        self.scenario = scenario

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

        self.waste_mass_df = LandfillWasteMassDF.create(self.access_city_attribute('waste_generated_df'), self.access_city_attribute('divs_df'), self.fraction_of_waste).df

    def estimate_emissions(self) -> tuple:
        """
        Estimate emissions using an instance of the SWEET class.

        Args:
            baseline (bool, optional): If True, estimates baseline emissions. If False, uses new values. Defaults to True.
        """
        self.model = SWEET(**self.__dict__)
        self.waste_mass, self.emissions, self.ch4, self.captured = self.model.estimate_emissions()

    def access_city_attribute(self, attr_name):
        city = cities_to_run.get(self.city_name)
        if city:
            return getattr(city, attr_name, None)
        return None

    def _convert_df_to_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        return df.to_dict(orient='records') if df is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'open_date': self.open_date,
            'close_date': self.close_date,
            'site_type': self.site_type,
            'mcf': self.mcf,
            'fraction_of_waste': self.fraction_of_waste,
            'gas_capture': self.gas_capture,
            'landfill_index': self.landfill_index,
            'scenario': self.scenario,
            'gas_capture_efficiency': self.gas_capture_efficiency,
            'oxidation_factor': self.oxidation_factor,
            'waste_mass': self._convert_df_to_dict(self.waste_mass),
            'emissions': self._convert_df_to_dict(self.emissions),
            'ch4': self._convert_df_to_dict(self.ch4),
            'captured': self.captured,
            **{k: v for k, v in self.__dict__.items() if k not in ['open_date', 'close_date', 'site_type', 'mcf', 'fraction_of_waste', 'gas_capture', 'landfill_index', 'scenario', 'gas_capture_efficiency', 'oxidation_factor', 'waste_mass', 'emissions', 'ch4', 'captured']}
        }