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
from SWEET_python import defaults
#import defaults

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
    def __init__(self, landfill, city):
        """
        Initializes a SWEET instance.

        Args:
            landfill (object): An instance containing landfill-specific parameters.
            city (object): An instance containing city-specific parameters and methods.
        """

        self.landfill = landfill
        self.city = city
        #self.baseline_divs = baseline_divs
        #self.new_divs = new_divs

    def estimate_emissions(self, baseline=True):
        """
        Estimates methane emissions based on the SWEET model. It considers the amount and type of waste generated 
        annually, how much of that waste is diverted to different facilities, and how waste biodegrades over time 
        to produce methane. The results are aggregated annually.

        Args:
            baseline (bool, optional): Whether to use the baseline scenario (default) or an alternative scenario.
                The baseline scenario uses collected data, while the alternative scenario uses user-defined parameters.

        Returns:
            tuple: Four pandas DataFrames, respectively containing:
                0. Landfilled waste masses for each year and type.
                1. Net methane emissions for each year.
                2. Methane produced (before capture) for each year and waste type.
                3. Amount of methane captured at the landfill for each year.
        """

        self.qs = {} # net CH4 emissions
        self.ms = {} # Waste mass
        self.captured = {} # CH4 captured
        self.ch4_produced = {} # CH4 produced (before capture)

        # Check if city has an attribute named 'div_masses'
        # div_masses is used to accumulate how much waste was diverted rather than being landfilled each year.
        # doing_div_masses variables prevent this from being calculated more than once, if more than one landfill is
        # being modeled for the same city.
        doing_div_masses = False
        doing_div_masses_new = False
        if not hasattr(self.city, 'div_masses'):
            doing_div_masses = True
        if doing_div_masses:
            self.city.div_masses = {}
            self.city.div_masses['compost'] = {}
            self.city.div_masses['anaerobic'] = {}
            self.city.div_masses['combustion'] = {}
            self.city.div_masses['recycling'] = {}

        # As in other places in this package, "new" is used to denote a user-defined alternative scenario rather than
        # the baseline parameters, which are determined by collected data.
        if (not baseline) and (not hasattr(self.city, 'div_masses_new')):
            doing_div_masses_new = True
        if doing_div_masses_new:
            self.city.div_masses_new = {}
            self.city.div_masses_new['compost'] = {}
            self.city.div_masses_new['anaerobic'] = {}
            self.city.div_masses_new['combustion'] = {}
            self.city.div_masses_new['recycling'] = {}

        for year in range(self.landfill.open_date, self.landfill.close_date):
            
            # t is the number of years since the first year of data
            t = year - self.city.year_of_data
            #print(t)
            #print(year)
            #t2 = year - self.landfill.open_date
            self.qs[year] = {}
            self.ms[year] = {}
            self.ch4_produced[year] = {}
            
            if doing_div_masses:
                self.city.div_masses['compost'][year] = {}
                self.city.div_masses['anaerobic'][year] = {}
                self.city.div_masses['combustion'][year] = {}
                self.city.div_masses['recycling'][year] = {}

            if doing_div_masses_new:
                self.city.div_masses_new['compost'][year] = {}
                self.city.div_masses_new['anaerobic'][year] = {}
                self.city.div_masses_new['combustion'][year] = {}
                self.city.div_masses_new['recycling'][year] = {}

            caps = []

            # The year population, waste generation, and other similar data was collected is used for calculating
            # how those values change with time. Before the year of data collection, historic growth rates are used
            # to project back in time, and after the year of data collection, future growth rates are used to project
            # forward in time.
            if year < self.city.year_of_data:
                growth_rate = self.city.growth_rate_historic
            else:
                growth_rate = self.city.growth_rate_future

            # "DST" is a term from the WasteMAP website, meaning decision support tool. dst_implement_year is used
            # to change the model parameters to a user-defined alternative scenario after a certain year. This is
            # when variables with "new" in their name are used instead of the baseline parameters.
            if baseline:
                divs = self.city.baseline_divs
                fraction_of_waste = self.landfill.fraction_of_waste
            else:
                if year >= self.city.dst_implement_year:
                    divs = self.city.new_divs
                    fraction_of_waste = self.landfill.fraction_of_waste_new
                else:
                    divs = self.city.baseline_divs
                    fraction_of_waste = self.landfill.fraction_of_waste

            # Calculate how much waste is generated in a given year, and subtract how much is diverted. What remains
            # is added to the landfill.
            for waste in self.city.components:
                self.ms[year][waste] = (
                    self.city.waste_mass * 
                    self.city.waste_fractions[waste] - 
                    divs['compost'][waste] - 
                    divs['anaerobic'][waste] - 
                    divs['combustion'][waste] - 
                    divs['recycling'][waste]) * \
                    fraction_of_waste * \
                    (growth_rate ** t)
                
                if doing_div_masses:
                    self.city.div_masses['compost'][year][waste] = divs['compost'][waste] * (growth_rate ** t)
                    self.city.div_masses['anaerobic'][year][waste] = divs['anaerobic'][waste] * (growth_rate ** t)
                    self.city.div_masses['combustion'][year][waste] = divs['combustion'][waste] * (growth_rate ** t)
                    self.city.div_masses['recycling'][year][waste] = divs['recycling'][waste] * (growth_rate ** t)

                if doing_div_masses_new:
                    self.city.div_masses_new['compost'][year][waste] = divs['compost'][waste] * (growth_rate ** t)
                    self.city.div_masses_new['anaerobic'][year][waste] = divs['anaerobic'][waste] * (growth_rate ** t)
                    self.city.div_masses_new['combustion'][year][waste] = divs['combustion'][waste] * (growth_rate ** t)
                    self.city.div_masses_new['recycling'][year][waste] = divs['recycling'][waste] * (growth_rate ** t)
                
                # Loop through years previous to the current one to calculate methane emissions. ks is the decay rates
                # for different types of waste. L_0 is the fraction of biodegradable matter in different types of waste.
                # mcf is the methane correction factor, which accounts for how much biodegradation occurs in anaerobic
                # conditions rather than aerobic -- only anaerobic produces methane. 
                ch4_produced = []
                ch4 = []
                for y in range(self.landfill.open_date, year):
                    years_back = year - y
                    ch4_produce = self.city.ks[waste] * \
                                    defaults.L_0[waste] * \
                                    self.ms[y][waste] * \
                                    np.exp(-self.city.ks[waste] * \
                                    (years_back - 0.5)) * \
                                    self.landfill.mcf

                # for y in range(t2):
                #     year_back = y + self.landfill.open_date
                #     ch4_produce = self.city.ks[waste] * \
                #           defaults.L_0[waste] * \
                #           self.ms[year_back][waste] * \
                #           np.exp(-self.city.ks[waste] * \
                #           (t2 - y - 0.5)) * \
                #           self.landfill.mcf
                    
                    ch4_produced.append(ch4_produce)
                    ch4_capture = ch4_produce * self.landfill.gas_capture_efficiency
                    caps.append(ch4_capture)
                    val = (ch4_produce - ch4_capture) * (1 - self.landfill.oxidation_factor) + ch4_capture * .02
                    #val = ch4_produce * (1 - sweet_tools_compare.oxidation_factor['without_lfg'][site])
                    ch4.append(val)
                    
                # Sum CH4 for waste from all previous years
                self.qs[year][waste] = sum(ch4)
                self.ch4_produced[year][waste] = sum(ch4_produced)
            
            self.captured[year] = sum(caps) / 365 / 24 # 365 and 24 are unit conversions

        self.q_df = pd.DataFrame(self.qs).T
        self.q_df['total'] = self.q_df.sum(axis=1)
        self.m_df = pd.DataFrame(self.ms).T
        self.ch4_df = pd.DataFrame(self.ch4_produced).T

        if doing_div_masses:
            self.city.div_masses['compost'] = pd.DataFrame(self.city.div_masses['compost']).T
            self.city.div_masses['anaerobic'] = pd.DataFrame(self.city.div_masses['anaerobic']).T
            self.city.div_masses['combustion'] = pd.DataFrame(self.city.div_masses['combustion']).T
            self.city.div_masses['recycling'] = pd.DataFrame(self.city.div_masses['recycling']).T

        if doing_div_masses_new:
            self.city.div_masses_new['compost'] = pd.DataFrame(self.city.div_masses_new['compost']).T
            self.city.div_masses_new['anaerobic'] = pd.DataFrame(self.city.div_masses_new['anaerobic']).T
            self.city.div_masses_new['combustion'] = pd.DataFrame(self.city.div_masses_new['combustion']).T
            self.city.div_masses_new['recycling'] = pd.DataFrame(self.city.div_masses_new['recycling']).T
        
        return self.m_df, self.q_df, self.ch4_df, self.captured
    
    # # This loop is no longer up to date and is not used. It was used to compare the results of this Python version
    # # of SWEET to the original Excel version. 
    # def estimate_emissions_match_excel(self):
        
    #     self.qs = {}
    #     self.ms = {}
    #     self.masses_compost = {}
    #     self.masses_anaerobic = {}
    #     self.q_dfs = {}
    #     self.m_dfs = {}
    #     self.organic_df = {}
    #     self.captured = {}
    #     self.ch4_produced = {}
        
    #     for year in range(self.landfill.open_date, self.landfill.close_date):
            
    #         t = year - self.city.year_of_data
    #         #print(t)
    #         #print(year)
    #         #t2 = year - self.landfill.open_date
    #         self.qs[year] = {}
    #         self.ms[year] = {}
    #         self.ch4_produced[year] = {}
    #         # Loop through years
    #         caps = []
    #         for waste in self.city.components:

    #             # This stuff probs doesn't work anymore, copy the above one
    #             if year < self.city.year_of_data:
    #                 growth_rate = self.city.growth_rate_historic
    #             else:
    #                 growth_rate = self.city.growth_rate_future
    #             if year >= 2023:
    #                 divs = self.city.new_divs
    #             else:
    #                 divs = self.city.divs
    #             if waste == 'paper_cardboard':
    #                 self.ms[year][waste] = (
    #                     self.city.waste_mass * 
    #                     self.city.waste_fractions[waste] - 
    #                     0 - 
    #                     0 - 
    #                     divs['combustion'][waste] - 
    #                     divs['recycling'][waste]) * \
    #                     self.landfill.fraction_of_waste * \
    #                     (growth_rate ** t)
    #             else:
    #                 self.ms[year][waste] = (
    #                     self.city.waste_mass * 
    #                     self.city.waste_fractions[waste] - 
    #                     divs['compost'][waste] - 
    #                     divs['anaerobic'][waste] - 
    #                     divs['combustion'][waste] - 
    #                     divs['recycling'][waste]) * \
    #                     self.landfill.fraction_of_waste * \
    #                     (growth_rate ** t)
                
    #             # Loop through previous years to get methane after decay
                
    #             ch4_produced = []
    #             ch4 = []
    #             for y in range(self.landfill.open_date, year):
    #                 years_back = year - y
    #                 ch4_produce = self.city.ks[waste] * \
    #                                 defaults.L_0[waste] * \
    #                                 self.ms[y][waste] * \
    #                                 np.exp(-self.city.ks[waste] * \
    #                                 (years_back - 0.5)) * \
    #                                 self.landfill.mcf

    #             # for y in range(t2):
    #             #     year_back = y + self.landfill.open_date
    #             #     ch4_produce = self.city.ks[waste] * \
    #             #           defaults.L_0[waste] * \
    #             #           self.ms[year_back][waste] * \
    #             #           np.exp(-self.city.ks[waste] * \
    #             #           (t2 - y - 0.5)) * \
    #             #           self.landfill.mcf
                    
    #                 ch4_produced.append(ch4_produce)
    #                 ch4_capture = ch4_produce * self.landfill.gas_capture_efficiency
    #                 caps.append(ch4_capture)
    #                 val = (ch4_produce - ch4_capture) * (1 - self.landfill.oxidation_factor) + ch4_capture * .02
    #                 #val = ch4_produce * (1 - sweet_tools_compare.oxidation_factor['without_lfg'][site])
    #                 ch4.append(val)
                    
    #             # Sum CH4 for waste from all previous years
    #             self.qs[year][waste] = sum(ch4)
    #             self.ch4_produced[year][waste] = sum(ch4_produced)
                
    #         self.captured[year] = sum(caps) / 365 / 24

    #     self.q_df = pd.DataFrame(self.qs).T
    #     self.q_df['total'] = self.q_df.sum(axis=1)
    #     self.m_df = pd.DataFrame(self.ms).T
    #     self.ch4_df = pd.DataFrame(self.ch4_produced).T
        
    #     return self.m_df, self.q_df, self.ch4_df, self.captured
    
    def estimate_emissions_match_excel(self, baseline=True):
        """
        Estimates methane emissions based on the SWEET model. It considers the amount and type of waste generated 
        annually, how much of that waste is diverted to different facilities, and how waste biodegrades over time 
        to produce methane. The results are aggregated annually.

        Args:
            baseline (bool, optional): Whether to use the baseline scenario (default) or an alternative scenario.
                The baseline scenario uses collected data, while the alternative scenario uses user-defined parameters.

        Returns:
            tuple: Four pandas DataFrames, respectively containing:
                0. Landfilled waste masses for each year and type.
                1. Net methane emissions for each year.
                2. Methane produced (before capture) for each year and waste type.
                3. Amount of methane captured at the landfill for each year.
        """

        self.qs = {} # net CH4 emissions
        self.ms = {} # Waste mass
        self.captured = {} # CH4 captured
        self.ch4_produced = {} # CH4 produced (before capture)

        # Check if city has an attribute named 'div_masses'
        # div_masses is used to accumulate how much waste was diverted rather than being landfilled each year.
        # doing_div_masses variables prevent this from being calculated more than once, if more than one landfill is
        # being modeled for the same city.
        doing_div_masses = False
        doing_div_masses_new = False
        if not hasattr(self.city, 'div_masses'):
            doing_div_masses = True
        if doing_div_masses:
            self.city.div_masses = {}
            self.city.div_masses['compost'] = {}
            self.city.div_masses['anaerobic'] = {}
            self.city.div_masses['combustion'] = {}
            self.city.div_masses['recycling'] = {}

        # As in other places in this package, "new" is used to denote a user-defined alternative scenario rather than
        # the baseline parameters, which are determined by collected data.
        if (not baseline) and (not hasattr(self.city, 'div_masses_new')):
            doing_div_masses_new = True
        if doing_div_masses_new:
            self.city.div_masses_new = {}
            self.city.div_masses_new['compost'] = {}
            self.city.div_masses_new['anaerobic'] = {}
            self.city.div_masses_new['combustion'] = {}
            self.city.div_masses_new['recycling'] = {}

        for year in range(self.landfill.open_date, self.landfill.close_date):
            
            # t is the number of years since the first year of data
            t = year - self.city.year_of_data
            #print(t)
            #print(year)
            #t2 = year - self.landfill.open_date
            self.qs[year] = {}
            self.ms[year] = {}
            self.ch4_produced[year] = {}
            
            if doing_div_masses:
                self.city.div_masses['compost'][year] = {}
                self.city.div_masses['anaerobic'][year] = {}
                self.city.div_masses['combustion'][year] = {}
                self.city.div_masses['recycling'][year] = {}

            if doing_div_masses_new:
                self.city.div_masses_new['compost'][year] = {}
                self.city.div_masses_new['anaerobic'][year] = {}
                self.city.div_masses_new['combustion'][year] = {}
                self.city.div_masses_new['recycling'][year] = {}

            caps = []

            # The year population, waste generation, and other similar data was collected is used for calculating
            # how those values change with time. Before the year of data collection, historic growth rates are used
            # to project back in time, and after the year of data collection, future growth rates are used to project
            # forward in time.
            if year < self.city.year_of_data:
                growth_rate = self.city.growth_rate_historic
            else:
                growth_rate = self.city.growth_rate_future

            # "DST" is a term from the WasteMAP website, meaning decision support tool. dst_implement_year is used
            # to change the model parameters to a user-defined alternative scenario after a certain year. This is
            # when variables with "new" in their name are used instead of the baseline parameters.
            if baseline:
                divs = self.city.baseline_divs
                fraction_of_waste = self.landfill.fraction_of_waste
            else:
                if year >= self.city.dst_implement_year:
                    divs = self.city.new_divs
                    fraction_of_waste = self.landfill.fraction_of_waste_new
                else:
                    divs = self.city.baseline_divs
                    fraction_of_waste = self.landfill.fraction_of_waste

            # Calculate how much waste is generated in a given year, and subtract how much is diverted. What remains
            # is added to the landfill.
            for waste in self.city.components:
                if waste == 'paper_cardboard':
                    self.ms[year][waste] = (
                        self.city.waste_mass * 
                        self.city.waste_fractions[waste] - 
                        0 - 
                        0 - 
                        divs['combustion'][waste] - 
                        divs['recycling'][waste]) * \
                        fraction_of_waste * \
                        (growth_rate ** t)
                else:
                    self.ms[year][waste] = (
                        self.city.waste_mass * 
                        self.city.waste_fractions[waste] - 
                        divs['compost'][waste] - 
                        divs['anaerobic'][waste] - 
                        divs['combustion'][waste] - 
                        divs['recycling'][waste]) * \
                        fraction_of_waste * \
                        (growth_rate ** t)
                
                if doing_div_masses:
                    self.city.div_masses['compost'][year][waste] = divs['compost'][waste] * (growth_rate ** t)
                    self.city.div_masses['anaerobic'][year][waste] = divs['anaerobic'][waste] * (growth_rate ** t)
                    self.city.div_masses['combustion'][year][waste] = divs['combustion'][waste] * (growth_rate ** t)
                    self.city.div_masses['recycling'][year][waste] = divs['recycling'][waste] * (growth_rate ** t)

                if doing_div_masses_new:
                    self.city.div_masses_new['compost'][year][waste] = divs['compost'][waste] * (growth_rate ** t)
                    self.city.div_masses_new['anaerobic'][year][waste] = divs['anaerobic'][waste] * (growth_rate ** t)
                    self.city.div_masses_new['combustion'][year][waste] = divs['combustion'][waste] * (growth_rate ** t)
                    self.city.div_masses_new['recycling'][year][waste] = divs['recycling'][waste] * (growth_rate ** t)
                
                # Loop through years previous to the current one to calculate methane emissions. ks is the decay rates
                # for different types of waste. L_0 is the fraction of biodegradable matter in different types of waste.
                # mcf is the methane correction factor, which accounts for how much biodegradation occurs in anaerobic
                # conditions rather than aerobic -- only anaerobic produces methane. 
                ch4_produced = []
                ch4 = []
                for y in range(self.landfill.open_date, year):
                    years_back = year - y
                    ch4_produce = self.city.ks[waste] * \
                                    defaults.L_0[waste] * \
                                    self.ms[y][waste] * \
                                    np.exp(-self.city.ks[waste] * \
                                    (years_back - 0.5)) * \
                                    self.landfill.mcf

                # for y in range(t2):
                #     year_back = y + self.landfill.open_date
                #     ch4_produce = self.city.ks[waste] * \
                #           defaults.L_0[waste] * \
                #           self.ms[year_back][waste] * \
                #           np.exp(-self.city.ks[waste] * \
                #           (t2 - y - 0.5)) * \
                #           self.landfill.mcf
                    
                    ch4_produced.append(ch4_produce)
                    ch4_capture = ch4_produce * self.landfill.gas_capture_efficiency
                    caps.append(ch4_capture)
                    val = (ch4_produce - ch4_capture) * (1 - self.landfill.oxidation_factor) + ch4_capture * .02
                    #val = ch4_produce * (1 - sweet_tools_compare.oxidation_factor['without_lfg'][site])
                    ch4.append(val)
                    
                # Sum CH4 for waste from all previous years
                self.qs[year][waste] = sum(ch4)
                self.ch4_produced[year][waste] = sum(ch4_produced)
            
            self.captured[year] = sum(caps) / 365 / 24 # 365 and 24 are unit conversions

        self.q_df = pd.DataFrame(self.qs).T
        self.q_df['total'] = self.q_df.sum(axis=1)
        self.m_df = pd.DataFrame(self.ms).T
        self.ch4_df = pd.DataFrame(self.ch4_produced).T

        if doing_div_masses:
            self.city.div_masses['compost'] = pd.DataFrame(self.city.div_masses['compost']).T
            self.city.div_masses['anaerobic'] = pd.DataFrame(self.city.div_masses['anaerobic']).T
            self.city.div_masses['combustion'] = pd.DataFrame(self.city.div_masses['combustion']).T
            self.city.div_masses['recycling'] = pd.DataFrame(self.city.div_masses['recycling']).T

        if doing_div_masses_new:
            self.city.div_masses_new['compost'] = pd.DataFrame(self.city.div_masses_new['compost']).T
            self.city.div_masses_new['anaerobic'] = pd.DataFrame(self.city.div_masses_new['anaerobic']).T
            self.city.div_masses_new['combustion'] = pd.DataFrame(self.city.div_masses_new['combustion']).T
            self.city.div_masses_new['recycling'] = pd.DataFrame(self.city.div_masses_new['recycling']).T
        
        return self.m_df, self.q_df, self.ch4_df, self.captured