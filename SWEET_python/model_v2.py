"""
This module provides an implementation of the SWEET model for calculating methane emissions from municipal solid waste, 
based on the EPA's SWEET excel model.

The model takes into account waste generation, diversion to various processes (e.g., composting, anaerobic digestion, 
combustion/incineration, recycling), biodegradation rates influenced by precipitation, and landfill methane capture, 
to estimate annual methane emissions.

Workflow:
1. Instantiate a `City` object, which encapsulates city-specific parameters and methods.
2. Within the `City` class, instantiate one or more `Landfill` objects
3. For each `Landfill` instance, instantiate the `SWEET` class.
4. Call the `estimate_emissions` method of the `SWEET` class to run the model and retrieve the results.

Main components:
- `SWEET` class: Represents the main model and its operations. 

Dependencies:
- pandas
- numpy
- SWEET_python.defaults

Authors: Hugh Runyan, Andre Scheinwald
Date: Sep 2023
Version: 0.1
"""

import pandas as pd
import numpy as np
import time
try:
    import defaults_2019
except:
    import SWEET_python.defaults_2019 as defaults_2019


# Based on EPA's SWEET excel model for calculating methane emissions from municipal solid waste 
# (https://globalmethane.org/resources/details.aspx?resourceid=5176)

'''
The way this type of model works is: data inputs determine how much waste is generated annually and of what types,
and how much waste is diverted to composting, anaerobic digestion, combustion/incineration, and recycling facilities
rather than being landfilled. Population growth rates in the past and future are used to estimate how waste generation
changes over time. Research-backed default parameters are used to estimate how much biodegradable matter the different
types of waste contain. Different types of waste also biodegrade at different rates, and these rates are influenced by
average annual precipitation. In the first year of the model, the waste that is not diverted is added to the landfill.
In the second year, the waste from the first year has biodegraded to some extent, creating methane, and new generated
waste is added to the landfill. In the third year, waste from the first year biodegrades another year, and waste from 
the second year does its first year of biodegradation, and new generated waste is added. This process continues until
the final year of the model. 

The model outputs how much methane is produced each year from the cumulative degradation of wastes of different types and
ages. This amount is reduced by the amount of methane captured at the landfill.

Unit: m3 CH4/year
'''

# class SWEET:
#     def __init__(self, city_instance_attrs: dict, city_params_dict: dict, landfill_instance_attrs: dict):
#         """
#         Initializes a SWEET instance.

#         Args:
#             landfill (Landfill): An instance containing landfill-specific parameters.
#             scenario (int): The scenario identifier.
#             city_params (dict): Additional city-specific parameters.
#         """

#         self.landfill_instance_attrs = landfill_instance_attrs
#         self.city_instance_attrs = city_instance_attrs
#         self.city_params_dict = city_params_dict

#     def estimate_emissions(self):
#         """
#         Estimates methane emissions based on the SWEET model. It considers the amount and type of waste generated 
#         annually, how much of that waste is diverted to different facilities, and how waste biodegrades over time 
#         to produce methane. The results are aggregated annually.

#         Returns:
#             tuple: Four pandas DataFrames, respectively containing:
#                 0. Landfilled waste masses for each year and type.
#                 1. Net methane emissions for each year.
#                 2. Methane produced (before capture) for each year and waste type.
#                 3. Amount of methane captured at the landfill for each year.
#         """
#         self.qs = {}
#         self.ms = {}
#         self.captured = {}
#         self.ch4_produced = {}

#         # if self.div_masses is None:
#         #     doing_div_masses = True
#         #     div_masses = {key: {} for key in ["compost", "anaerobic", "combustion", "recycling"]}
#         # else:
#         #     doing_div_masses = False
#         #     div_masses = self.div_masses.model_dump()

#         for year in range(self.landfill_instance_attrs['open_date'], self.landfill_instance_attrs['close_date']):
#             t = year - self.city_params_dict['year_of_data_pop']
#             self.qs[year] = {}
#             self.ms[year] = {}
#             self.ch4_produced[year] = {}

#             # if doing_div_masses:
#             #     for key in ["compost", "anaerobic", "combustion", "recycling"]:
#             #         div_masses[key][year] = {}

#             caps = []
#             growth_rate = self.city_params_dict['growth_rate_historic'] if year < self.city_params_dict['year_of_data_pop'] else self.city_params_dict['growth_rate_future']

#             # if self.scenario == 0:
#             #     divs = self.divs
#             #     fraction_of_waste = self.landfill.fraction_of_waste
#             # else:
#             #     if year >= self.dst_implement_year:
#             #         divs = self.scenario_parameters[self.scenario].divs
#             #         fraction_of_waste = self.landfill.fraction_of_waste[self.landfill.landfill_index] # Need to figure out how to handle multiple fraction of waste values. make a df indexed by year? Or bring back new
#             #     else:
#             #         divs = self.divs
#             #         fraction_of_waste = self.landfill.fraction_of_waste

#             for waste in self.city_instance_attrs['components']:
#                 # self.ms[year][waste] = (
#                 #     self.waste_mass * getattr(self.waste_fractions, waste) -
#                 #     sum(getattr(getattr(divs, key), waste) for key in ["compost", "anaerobic", "combustion", "recycling"])) * \
#                 #     fraction_of_waste * (growth_rate ** t)

#                 # if doing_div_masses:
#                 #     for key in self.divs.model_fields:
#                 #         div_masses[key][year][waste] = getattr(getattr(divs, key), waste) * (growth_rate ** t)

#                 ch4_produced = []
#                 ch4 = []
#                 for y in range(self.landfill_instance_attrs['open_date'], year):
#                     years_back = year - y
#                     ch4_produce = (
#                         self.city_params_dict['ks'][waste] *
#                         defaults_2019.L_0[waste] *
#                         self.landfill_instance_attrs['waste_mass_df'].at[y, waste] *
#                         np.exp(-self.city_params_dict['ks'][waste] * (years_back - 0.5)) *
#                         self.landfill_instance_attrs['mcf']
#                     )
#                     ch4_produced.append(ch4_produce)
#                     ch4_capture = ch4_produce * self.landfill_instance_attrs['gas_capture_efficiency']
#                     caps.append(ch4_capture)
#                     val = (ch4_produce - ch4_capture) * (1 - self.landfill_instance_attrs['oxidation_factor']) + ch4_capture * 0.02
#                     ch4.append(val)

#                 self.qs[year][waste] = sum(ch4)
#                 self.ch4_produced[year][waste] = sum(ch4_produced)

#             self.captured[year] = sum(caps) / 365 / 24

#         self.q_df = pd.DataFrame(self.qs).T
#         self.q_df['total'] = self.q_df.sum(axis=1)
#         self.m_df = pd.DataFrame(self.ms).T
#         self.ch4_df = pd.DataFrame(self.ch4_produced).T

#         # if doing_div_masses:
#         #     for key in ["compost", "anaerobic", "combustion", "recycling"]:
#         #         div_masses[key] = pd.DataFrame(div_masses[key]).T

#         #     div_masses_annual = DivMassesAnnual(
#         #         compost=div_masses['compost'],
#         #         anaerobic=div_masses['anaerobic'],
#         #         combustion=div_masses['combustion'],
#         #         recycling=div_masses['recycling']
#         #     )

#         #     if self.scenario == 0:
#         #         self.div_masses = div_masses_annual
#         #     else:
#         #         self.scenario_parameters[self.scenario].div_masses = div_masses_annual

#         return self.m_df, self.q_df, self.ch4_df, self.captured

pd.set_option('display.max_rows', None)

class SWEET:
    def __init__(self, city_instance_attrs: dict, city_params_dict: dict, landfill_instance_attrs: dict):
        self.landfill_instance_attrs = landfill_instance_attrs
        self.city_instance_attrs = city_instance_attrs
        self.city_params_dict = city_params_dict

    def estimate_emissions(self):
        start_time = time.time()
        open_date = self.landfill_instance_attrs['open_date']
        #close_date = self.landfill_instance_attrs['close_date']
        #advanced = self.landfill_instance_attrs['advanced']
        year_of_data_pop = self.city_params_dict['year_of_data_pop']
        #growth_rate_historic = self.city_params_dict['growth_rate_historic']
        #growth_rate_future = self.city_params_dict['growth_rate_future']
        ks = self.city_params_dict['ks']
        waste_mass_df = self.landfill_instance_attrs['waste_mass_df']
        mcf = self.landfill_instance_attrs['mcf']
        gas_capture_efficiency = self.landfill_instance_attrs['gas_capture_efficiency']
        oxidation_factor = self.landfill_instance_attrs['oxidation_factor']
        components = self.city_instance_attrs['components']

        years = np.arange(1960, 2074)
        t = years - year_of_data_pop
        #growth_rates = np.where(years < year_of_data_pop, growth_rate_historic, growth_rate_future) ** t

        qs = {}
        ch4_produced = {}
        captured = {}

        end_time = time.time()
        print(f"Model setup: {end_time - start_time} seconds")

        start_time = time.time()
        for year in years:
            ch4_year = {}
            ch4_produced_year = {}

            # Get some values I don't need to reget every waste type iteration
            mcf_loop = mcf.loc[np.arange(open_date, year)].values
            if isinstance(gas_capture_efficiency, pd.Series):
                gce_loop = gas_capture_efficiency.at[year]
            else:
                gce_loop = gas_capture_efficiency

            if isinstance(oxidation_factor, pd.Series):
                ox_loop = oxidation_factor.loc[np.arange(open_date, year)].values
            else:
                ox_loop = oxidation_factor

            for waste in components:
                ch4_year[waste] = 0
                ch4_produced_year[waste] = 0

                years_back = year - np.arange(open_date, year)
                # is this the right range of years...verify
                exp_term = np.exp(-ks[waste].loc[np.arange(open_date, year)] * (years_back - 0.5))
                waste_masses = waste_mass_df.loc[open_date:year-1, waste] #.values
                # Make sure the use of decay rate time series makes sense. 
                ch4_produce = ks[waste].loc[np.arange(open_date, year)] * defaults_2019.L_0[waste] * waste_masses * exp_term * mcf_loop
                # This np sum could maybe be replaced with pandas
                ch4_produced_year[waste] = np.sum(ch4_produce)
                ch4_capture = ch4_produce * gce_loop
                if (len(ch4_produce) == 0) and (len(ch4_capture) == 0):
                    ch4_year[waste] = 0
                else:
                    try:
                        ch4_year[waste] = np.sum((ch4_produce - ch4_capture) * (1 - ox_loop) + ch4_capture * 0.02)
                    except:
                        print('break point')
                # else:
                #     ch4_year[waste] = np.sum((ch4_produce - ch4_capture) * (1 - oxidation_factor) + ch4_capture * 0.02)

            qs[year] = ch4_year
            ch4_produced[year] = ch4_produced_year
            if isinstance(gas_capture_efficiency, pd.Series):
                captured[year] = np.sum([ch4_produced_year[w] * gas_capture_efficiency.at[year] for w in components]) # / 365 / 24
            else:
                captured[year] = np.sum([ch4_produced_year[w] * gas_capture_efficiency for w in components]) # / 365 / 24

        end_time = time.time()
        print(f"Model run: {end_time - start_time} seconds")

        start_time = time.time()

        q_df = pd.DataFrame(qs).T
        q_df['total'] = q_df.sum(axis=1)
        ch4_df = pd.DataFrame(ch4_produced).T

        end_time = time.time()
        print(f"Model post-processing: {end_time - start_time} seconds")

        return None, q_df, ch4_df, captured
    

    def estimate_emissions2(self):
        start_time = time.time()

        open_date = self.landfill_instance_attrs['open_date']
        close_date = self.landfill_instance_attrs['close_date']
        year_of_data_pop = self.city_params_dict['year_of_data_pop']
        growth_rate_historic = self.city_params_dict['growth_rate_historic']
        growth_rate_future = self.city_params_dict['growth_rate_future']
        ks = self.city_params_dict['ks']
        waste_mass_df = self.landfill_instance_attrs['waste_mass_df']
        mcf = self.landfill_instance_attrs['mcf']
        gas_capture_efficiency = self.landfill_instance_attrs['gas_capture_efficiency']
        oxidation_factor = self.landfill_instance_attrs['oxidation_factor']
        components = self.city_instance_attrs['components']

        #years = np.arange(1960, 2074)

        # Precompute factors outside of the loop for all years
        #growth_rates = np.where(years < year_of_data_pop, growth_rate_historic, growth_rate_future) ** (years - year_of_data_pop)
        year_range = np.arange(open_date, 2074)

        qs = {}
        ch4_produced = {}
        captured = {}

        end_time = time.time()
        #print(f"Model setup: {end_time - start_time} seconds")

        start_time = time.time()

        # Vectorized calculation for each component
        for waste in components:
            years_back_matrix = year_range[None, :] - year_range[:, None]  # Matrix of (years - year_range)
            mask = years_back_matrix <= 0

            # Precompute exponential decay term for all years at once
            ks_values = ks[waste].loc[year_range].values[:, None]
            exp_term = np.exp(-ks_values * (years_back_matrix - 0.5))

            # Vectorized waste mass, L_0, and MCF computation
            waste_masses = waste_mass_df.loc[open_date:, waste].values[:, None]
            mcf_values = mcf.loc[year_range].values[:, None]
            ch4_produce = ks_values * defaults_2019.L_0[waste] * waste_masses * exp_term * mcf_values
            
            ch4_produce[mask] = 0

            # Sum across years for total methane produced
            ch4_produced[waste] = ch4_produce.sum(axis=0)

            # Gas capture and oxidation factor
            if isinstance(gas_capture_efficiency, pd.Series):
                gas_capture_efficiency_values = gas_capture_efficiency.loc[year_range].values
            else:
                gas_capture_efficiency_values = np.full(len(year_range), gas_capture_efficiency)

            if isinstance(oxidation_factor, pd.Series):
                oxidation_factor_values = oxidation_factor.loc[year_range].values
            else:
                oxidation_factor_values = np.full(len(year_range), oxidation_factor)

            ch4_capture = ch4_produce * gas_capture_efficiency_values

            # Final methane emissions calculation with oxidation
            ch4_year_total = np.sum((ch4_produce - ch4_capture) * (1 - oxidation_factor_values[:, None]) + ch4_capture * 0.02, axis=0)

            # Store results for the current waste component
            qs[waste] = ch4_year_total

            # Total methane captured for each year
            captured_total = ch4_produced[waste] * gas_capture_efficiency_values # / 365 / 24
            captured[waste] = captured_total

        end_time = time.time()
        #print(f"Model run: {end_time - start_time} seconds")

        start_time = time.time()

        # Convert results to DataFrames
        q_df = pd.DataFrame(qs, index=year_range)
        q_df['total'] = q_df.sum(axis=1)
        ch4_df = pd.DataFrame(ch4_produced, index=year_range)
        captured_df = pd.DataFrame(captured, index=year_range)

        end_time = time.time()
        #print(f"Model post-processing: {end_time - start_time} seconds")

        return None, q_df, ch4_df, captured_df