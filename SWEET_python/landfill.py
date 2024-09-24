import defaults_2019
from class_defs import *
import pandas as pd
from model_v2 import SWEET
from typing import List, Dict, Union, Any, Set, Optional, Tuple
import time
import numpy as np
from calmim_ox import Site, WeatherModel, WeatherProfile, Cover, CoverMaterial, materials, attach_thread


class Landfill:
    def __init__(
            self,
            open_date: int,
            close_date: int,
            site_type: str,
            mcf: pd.Series,
            city_params_dict: dict,
            city_instance_attrs: dict,
            landfill_index: int = 0,
            fraction_of_waste: float = 1,
            gas_capture: bool = False,
            scenario: int = 0,
            new_baseline: int = None,
            gas_capture_efficiency: pd.Series = None,
            flaring: int = None,
            cover: int = None,
            leachate_circulate: int = None,
            fraction_of_waste_vector: pd.DataFrame = None,
            ameliorated: int = None,
            advanced: bool = False,
            oxidation_factor: pd.Series = None,
            latlon: Tuple[float, float] = None,
            area: float = None,
            cover_type: str = None, # remember that these need to be in meters and square meters. 
            cover_thickness: float = None,
            fancy_ox: bool = False,
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
        self.new_baseline = new_baseline
        self.gas_capture_efficiency = gas_capture_efficiency
        self.flaring = flaring
        self.cover = cover
        self.leachate_circulate = leachate_circulate
        self.ameliorated = ameliorated
        self.oxidation_factor = oxidation_factor
        self.advanced = advanced
        self.cover_thickness = cover_thickness
        self.latlon = latlon
        self.area = area
        self.cover_type = cover_type
        self.fraction_of_waste_vector = fraction_of_waste_vector
        self.fancy_ox = fancy_ox

        self.doing_fancy_ox = self.fancy_ox
        if isinstance(self.doing_fancy_ox, dict):
            if self.doing_fancy_ox['baseline'] or self.doing_fancy_ox['scenario']:
                self.doing_fancy_ox = True
            else:
                self.doing_fancy_ox = False
        
        if self.doing_fancy_ox:
            print("Starting fancy oxidation calculations")

            # # Attach thread to JVM
            # attach_thread()
            # print("Thread attached to JVM")

            # Get oxidation potential
            site = Site(lat=self.latlon[0], lon=self.latlon[1])  # Make sure this handles negative longitudes correctly
            weather_profile = WeatherProfile()
            # This might recreate the jpypes every time, gotta think about that.
            weather_model = WeatherModel(site=site, weather_profile=weather_profile)

            # Simulate weather data
            print("Starting weather simulation")
            weather_model.simulate_weather()
            print("Weather simulation completed")

            # THIS NEEDS UPDATING TO WORK
            material = materials[0]

            material.calculate_properties()

            cover = Cover(material=material, site=site, weather_profile=weather_profile, weather_model=weather_model)

            # Calculate the oxidation potential by converting micrograms ch4 / g soil / day to ton ch4 / year
            #self.oxidation_potential = self.ch4_convert_ton_to_m3(cover.calculate_oxidation_rate() * area * cover_thickness * cover.soil_density * 365.25 / 1e6)
            self.oxidation_potential = cover.calculate_oxidation_rate() * area * cover_thickness * cover.soil_density * 365.25 / 1e6

        if self.gas_capture_efficiency is None:
            self.gas_capture_efficiency = defaults_2019.gas_capture_efficiency[site_type]

        # if not self.oxidation_factor:
        #     if self.gas_capture:
        #         self.oxidation_factor = defaults_2019.oxidation_factor['with_lfg'][site_type]
        #     else:
        #         self.oxidation_factor = defaults_2019.oxidation_factor['without_lfg'][site_type]

        # if (self.gas_capture) and (scenario == 0):
        #     self.gas_capture_efficiency = defaults_2019.gas_capture_efficiency[site_type]
        #     self.oxidation_factor = defaults_2019.oxidation_factor['with_lfg'][site_type]
        # else:
        #     self.gas_capture_efficiency = 0
        #     self.oxidation_factor = defaults_2019.oxidation_factor['without_lfg'][site_type]

        self.waste_mass = None
        self.emissions = None
        self.ch4 = None
        self.captured = None

        # if (advanced is True) and (new_baseline is True):
        #     pass
        # elif advanced is True:
        #     # Is this implemented twice?
        #     self.waste_mass_df.dst_implement_advanced(
        #         self.city_params_dict['waste_generated_df']['df'], 
        #         self.city_params_dict['divs_df'], 
        #         self.fraction_of_waste, 
        #         self.city_instance_attrs['components'],
        #         fraction_of_waste_vector
        #     )
        if advanced is True:
            # = self.city_params_dict['net_masses'] * self.fraction_of_waste
            pass
        else:
            # self.waste_mass_df = LandfillWasteMassDF.create(
            #     self.city_params_dict['waste_generated_df'], 
            #     self.city_params_dict['divs_df'], 
            #     self.fraction_of_waste, 
            #     self.city_instance_attrs['components']
            # )
            self.waste_mass_df = self.city_params_dict['net_masses'] * self.fraction_of_waste

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
    
    # def model_dump_for_serialization(self) -> dict:
    #     def convert_sets_to_lists(data):
    #         if isinstance(data, dict):
    #             return {self._convert_key(k): convert_sets_to_lists(v) for k, v in data.items()}
    #         elif isinstance(data, list):
    #             return [convert_sets_to_lists(v) for v in data]
    #         elif isinstance(data, set):
    #             return list(data)
    #         elif isinstance(data, pd.DataFrame):
    #             return data.to_dict(orient='records')
    #         elif isinstance(data, (np.int64, np.float64)):
    #             return data.item()
    #         elif isinstance(data, SWEET):
    #             return data.model_dump_for_serialization()
    #         else:
    #             return data

    #     d = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    #     if 'city_instance_attrs' in d:
    #         d['city_instance_attrs'] = {}
    #     if 'city_params_dict' in d:
    #         d['city_params_dict'] = {}
    #     if 'model' in d:
    #         d['model'] = {}

    #     return convert_sets_to_lists(d)

    @staticmethod
    def ch4_convert_ton_to_m3(ch4: float) -> float:
        """
        Convert CH4 in tons to equivalent volume in m^3.

        Args:
            mass_co2e (float): ch4 in tons.

        Returns:
            float: Equivalent volume of ch4 in cubic meters.
        """
        density_kg_per_m3 = 0.7168
        mass_kg = ch4 * 1000
        volume_m3 = mass_kg / density_kg_per_m3
        return volume_m3

    def _convert_key(self, key):
        if isinstance(key, (np.int64, np.float64)):
            return key.item()
        return key
    
    # def _determine_ox_vector(self) -> pd.DataFrame:
    #     if self.cover_thickness is not None:
    #         return
    #     implementation_year = self.city_params_dict['implementation_year']

    #     # Do the simple DST. Landfill types don't change. 
    #     if not self.advanced:
    #         years = pd.Index(range(1960, 2074))
    #         tag_gas = 'ox_nocap' if not self.gas_capture else 'ox_cap'
    #         ox_value = self.ox_options[tag_gas][self.site_type]
    #         values = [ox_value for year in years]
    #         series = pd.Series(values, index=years)
    #         self.oxidation_factor = series
    #     else:
    #         # Do the advanced DST. Landfill types can change, also have to account for new landfills
    #         years = pd.Index(range(1960, 2074))
    #         tag_gas = 'ox_nocap' if not self.gas_capture else 'ox_cap'
    #         ox_value = self.ox_options[tag_gas][self.site_type]
    #         values = [ox_value for year in years]
    #         series = pd.Series(values, index=years)
    #         self.oxidation_factor = series

    def estimate_emissions(self) -> tuple:
        """
        Estimate emissions using an instance of the SWEET class.

        Args:
            baseline (bool, optional): If True, estimates baseline emissions. If False, uses new values. Defaults to True.
        """

        # if self.cover_thickness is not None:
        #     self.oxidation_factor = 0

        # Oxidation factor for simple DST
        ox_nocap = {'landfill': 0.1, 'controlled_dumpsite': 0.05, 'dumpsite': 0}
        ox_cap = {'landfill': 0.22, 'controlled_dumpsite': 0.1, 'dumpsite': 0}
        if (not self.advanced) or (self.oxidation_factor is None):
            if self.gas_capture:
                self.oxidation_factor = ox_cap[self.site_type]
            else:
                self.oxidation_factor = ox_nocap[self.site_type]

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
        self.waste_mass, self.emissions, self.ch4, self.captured = self.model.estimate_emissions2()
        end_time = time.time()
        #print(f"Time taken to estimate emissions in Landfill: {end_time - start_time} seconds")

        if self.doing_fancy_ox:
            available_ch4 = self.ch4.loc[2023, :].sum() - self.captured.loc[2023, :].sum()
            self.oxidation_factor = self.oxidation_potential / available_ch4
            print(f"Oxidation factor: {self.oxidation_factor}")
            if self.oxidation_factor < 0:
                self.oxidation_factor = 0
            elif self.oxidation_factor > 1:
                self.oxidation_factor = 1
            self.model.landfill_instance_attrs = self.model_dump()
            self.waste_mass, self.emissions, self.ch4, self.captured = self.model.estimate_emissions2()

        #print(f"Time taken to estimate emissions in Landfill: {end_time - start_time} seconds")

    # def _convert_df_to_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
    #     return df.to_dict(orient='records') if df is not None else None

    # def to_dict(self) -> Dict[str, Any]:
    #     return self.model_dump()
