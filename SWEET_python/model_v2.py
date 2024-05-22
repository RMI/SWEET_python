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
from SWEET_python import defaults_2019
from SWEET_python.class_defs import DivMassesAnnual

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

class SWEET:
    def __init__(self, landfill, city, scenario=0):
        """
        Initializes a SWEET instance.

        Args:
            landfill (Landfill): An instance containing landfill-specific parameters.
            city (City): An instance containing city-specific parameters and methods.
        """
        self.landfill = landfill
        self.city = city
        self.scenario = scenario
        self.parameters = self.city.baseline_parameters if scenario == 0 else self.city.scenario_parameters[scenario-1]

    def estimate_emissions(self):
        """
        Estimates methane emissions based on the SWEET model. It considers the amount and type of waste generated 
        annually, how much of that waste is diverted to different facilities, and how waste biodegrades over time 
        to produce methane. The results are aggregated annually.

        Returns:
            tuple: Four pandas DataFrames, respectively containing:
                0. Landfilled waste masses for each year and type.
                1. Net methane emissions for each year.
                2. Methane produced (before capture) for each year and waste type.
                3. Amount of methane captured at the landfill for each year.
        """
        self.qs = {}
        self.ms = {}
        self.captured = {}
        self.ch4_produced = {}

        if self.parameters.div_masses is None:
            doing_div_masses = True
            div_masses = {key: {} for key in ["compost", "anaerobic", "combustion", "recycling"]}
        else:
            doing_div_masses = False
            div_masses = self.parameters.div_masses.model_dump()

        for year in range(self.landfill.open_date, self.landfill.close_date):
            t = year - self.parameters.year_of_data_pop
            self.qs[year] = {}
            self.ms[year] = {}
            self.ch4_produced[year] = {}

            if doing_div_masses:
                for key in ["compost", "anaerobic", "combustion", "recycling"]:
                    div_masses[key][year] = {}

            caps = []
            growth_rate = self.parameters.growth_rate_historic if year < self.parameters.year_of_data_pop else self.parameters.growth_rate_future

            if self.scenario == 0:
                divs = self.parameters.divs
                fraction_of_waste = self.landfill.fraction_of_waste
            else:
                if year >= self.parameters.dst_implement_year:
                    divs = self.city.scenario_parameters[self.scenario].divs
                    fraction_of_waste = self.landfill.fraction_of_waste[self.landfill.landfill_index] # Need to figure out how to handle multiple fraction of waste values. make a df indexed by year? Or bring back new
                else:
                    divs = self.city.baseline_parameters.divs
                    fraction_of_waste = self.landfill.fraction_of_waste

            for waste in self.city.components:
                self.ms[year][waste] = (
                    self.parameters.waste_mass * getattr(self.parameters.waste_fractions, waste) -
                    sum(getattr(getattr(divs, key), waste) for key in ["compost", "anaerobic", "combustion", "recycling"])) * \
                    fraction_of_waste * (growth_rate ** t)

                if doing_div_masses:
                    for key in self.parameters.divs.model_fields:
                        div_masses[key][year][waste] = getattr(getattr(divs, key), waste) * (growth_rate ** t)

                ch4_produced = []
                ch4 = []
                for y in range(self.landfill.open_date, year):
                    years_back = year - y
                    ch4_produce = (
                        getattr(self.parameters.ks, waste) *
                        defaults_2019.L_0[waste] *
                        self.ms[y][waste] *
                        np.exp(-getattr(self.parameters.ks, waste) * (years_back - 0.5)) *
                        self.landfill.mcf
                    )
                    ch4_produced.append(ch4_produce)
                    ch4_capture = ch4_produce * self.landfill.gas_capture_efficiency
                    caps.append(ch4_capture)
                    val = (ch4_produce - ch4_capture) * (1 - self.landfill.oxidation_factor) + ch4_capture * 0.02
                    ch4.append(val)

                self.qs[year][waste] = sum(ch4)
                self.ch4_produced[year][waste] = sum(ch4_produced)

            self.captured[year] = sum(caps) / 365 / 24

        self.q_df = pd.DataFrame(self.qs).T
        self.q_df['total'] = self.q_df.sum(axis=1)
        self.m_df = pd.DataFrame(self.ms).T
        self.ch4_df = pd.DataFrame(self.ch4_produced).T

        if doing_div_masses:
            for key in ["compost", "anaerobic", "combustion", "recycling"]:
                div_masses[key] = pd.DataFrame(div_masses[key]).T

            div_masses_annual = DivMassesAnnual(
                compost=div_masses['compost'],
                anaerobic=div_masses['anaerobic'],
                combustion=div_masses['combustion'],
                recycling=div_masses['recycling']
            )

            if self.scenario == 0:
                self.city.baseline_parameters.div_masses = div_masses_annual
            else:
                self.city.scenario_parameters[self.scenario].div_masses = div_masses_annual

        return self.m_df, self.q_df, self.ch4_df, self.captured