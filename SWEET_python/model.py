import pandas as pd
import numpy as np
from SWEET_python import defaults
#import defaults

class SWEET:
    def __init__(self, landfill, city):
        
        self.city = city
        self.landfill = landfill
        #self.baseline_divs = baseline_divs
        #self.new_divs = new_divs

    def estimate_emissions(self, baseline=True):
        
        self.qs = {}
        self.ms = {}
        self.masses_compost = {}
        self.masses_anaerobic = {}
        self.q_dfs = {}
        self.m_dfs = {}
        self.organic_df = {}
        self.captured = {}
        self.ch4_produced = {}
        # Check if city has an attribute named 'div_masses'
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

        if (not baseline) and (not hasattr(self.city, 'div_masses_new')):
            doing_div_masses_new = True
        if doing_div_masses_new:
            self.city.div_masses_new = {}
            self.city.div_masses_new['compost'] = {}
            self.city.div_masses_new['anaerobic'] = {}
            self.city.div_masses_new['combustion'] = {}
            self.city.div_masses_new['recycling'] = {}

        for year in range(self.landfill.open_date, self.landfill.close_date):
            
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
            # Loop through years
            caps = []

            if year < self.city.year_of_data:
                growth_rate = self.city.growth_rate_historic
            else:
                growth_rate = self.city.growth_rate_future

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
                # Loop through previous years to get methane after decay
                
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
                
            self.captured[year] = sum(caps) / 365 / 24

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
    
    
    def estimate_emissions_match_excel(self):
        
        self.qs = {}
        self.ms = {}
        self.masses_compost = {}
        self.masses_anaerobic = {}
        self.q_dfs = {}
        self.m_dfs = {}
        self.organic_df = {}
        self.captured = {}
        self.ch4_produced = {}
        
        for year in range(self.landfill.open_date, self.landfill.close_date):
            
            t = year - self.city.year_of_data
            #print(t)
            #print(year)
            #t2 = year - self.landfill.open_date
            self.qs[year] = {}
            self.ms[year] = {}
            self.ch4_produced[year] = {}
            # Loop through years
            caps = []
            for waste in self.city.components:

                # This stuff probs doesn't work anymore, copy the above one
                if year < self.city.year_of_data:
                    growth_rate = self.city.growth_rate_historic
                else:
                    growth_rate = self.city.growth_rate_future
                if year >= 2023:
                    divs = self.city.new_divs
                else:
                    divs = self.city.divs
                if waste == 'paper_cardboard':
                    self.ms[year][waste] = (
                        self.city.waste_mass * 
                        self.city.waste_fractions[waste] - 
                        0 - 
                        0 - 
                        divs['combustion'][waste] - 
                        divs['recycling'][waste]) * \
                        self.landfill.fraction_of_waste * \
                        (growth_rate ** t)
                else:
                    self.ms[year][waste] = (
                        self.city.waste_mass * 
                        self.city.waste_fractions[waste] - 
                        divs['compost'][waste] - 
                        divs['anaerobic'][waste] - 
                        divs['combustion'][waste] - 
                        divs['recycling'][waste]) * \
                        self.landfill.fraction_of_waste * \
                        (growth_rate ** t)
                
                # Loop through previous years to get methane after decay
                
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
                
            self.captured[year] = sum(caps) / 365 / 24

        self.q_df = pd.DataFrame(self.qs).T
        self.q_df['total'] = self.q_df.sum(axis=1)
        self.m_df = pd.DataFrame(self.ms).T
        self.ch4_df = pd.DataFrame(self.ch4_produced).T
        
        return self.m_df, self.q_df, self.ch4_df, self.captured
    
    # class SWEET:
    #     def __init__(self, city, landfill):
    #         self.city = city
    #         self.landfill = landfill

    #     def estimate_emissions(self):
    #         years = range(self.start_year, self.end_year)
    #         t_values = [year - self.start_year for year in years]

    #         # Create empty DataFrames for storing results
    #         m_df = pd.DataFrame(index=years)
    #         q_df = pd.DataFrame(index=years)
    #         captured = pd.Series(index=years)

    #         for waste in self.city.components:
    #             # Compute 'ms' values for all years at once
    #             m_df[waste] = ((self.city.waste_mass * self.city.waste_fractions[waste] -
    #                             self.city.compost[waste] - self.city.anaerobic[waste] -
    #                             self.city.combustion[waste] - self.city.recycling[waste]) *
    #                             self.landfill.fraction_of_waste * (1.03 ** np.array(t_values)))

    #             # Initialize 'qs' and 'captured' for this waste type
    #             q_df[waste] = 0
    #             captured_for_waste = 0

    #             for y, year in enumerate(years):
    #                 # Compute methane for each previous year
    #                 previous_years = range(y+1)  # y+1 to include current year
    #                 ch4_produce = (self.city.ks[waste] * defaults.L_0[waste] * m_df.loc[years[:y+1], waste] *
    #                             np.exp(-self.city.ks[waste] * (y - np.array(previous_years) - 0.5)) *
    #                             self.landfill.mcf)

    #                 ch4_capture = ch4_produce * self.landfill.gas_capture_efficiency
    #                 captured_for_waste += ch4_capture.sum()
    #                 ch4 = (ch4_produce - ch4_capture) * (1 - self.landfill.oxidation_factor) + ch4_capture * .02
    #                 q_df.at[year, waste] = ch4.sum()

    #             captured[waste] = captured_for_waste / 365 / 24

    #         q_df['total'] = q_df.sum(axis=1)

    #         return m_df, q_df