import defaults
import pandas as pd
import numpy as np
from model import SWEET
import copy

class City:
    def __init__(self, name):
        self.name = name
    
    def load_wb_params(self, row, rmi_db):
        
        #idx = row[0]
        row = row[1]
        
        self.country = row['country_name']
        self.region = defaults.region_lookup[self.country]
        #run_params['region'] = sweet_tools.region_lookup[run_params['country']]
        
        # Population, remove the try except when no duplicates
        self.population = float(row['population_population_number_of_people']) # * (1.03 ** (2010 - 2023))
        try:
            population_1950 = rmi_db.at[self.name, '1950_Population'].iloc[0]
            population_2020 = rmi_db.at[self.name, '2020_Population'].iloc[0]
            population_2035 = rmi_db.at[self.name, '2035_Population'].iloc[0]
        except:
            population_1950 = rmi_db.at[self.name, '1950_Population']
            population_2020 = rmi_db.at[self.name, '2020_Population']
            population_2035 = rmi_db.at[self.name, '2035_Population']
        self.growth_rate_historic = ((population_2020 / population_1950) ** (1 / (2020 - 1950)))
        self.growth_rate_future = ((population_2035 / population_2020) ** (1 / (2035 - 2020)))
        
        # Get waste total
        try:
            self.waste_mass = float(row['total_msw_total_msw_generated_tons_year']) # unit is tons
            self.waste_per_capita = self.waste_mass * 1000 / self.population / 365 #unit is kg/person/day
        except:
            self.waste_mass = float(row['total_msw_total_msw_generated_tons_year'].replace(',', ''))
            self.waste_per_capita = self.waste_mass * 1000 / self.population / 365
        if self.waste_mass != self.waste_mass:
            # Use per capita default
            self.waste_per_capita = defaults.msw_per_capita_defaults[self.region]
            self.waste_mass = self.waste_per_capita * self.population / 1000 * 365
        
        # Subtract mass that is informally collected
        self.informal_fraction = float(row['percent_informal_sector_percent_collected_by_informal_sector_percent']) / 100
        if self.informal_fraction == self.informal_fraction:
            self.waste_mass *= (1 - self.informal_fraction)
        
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
                             ]]
    
        waste_fractions.rename(index={'composition_food_organic_waste_percent': 'food',
                                        'composition_yard_garden_green_waste_percent': 'green',
                                        'composition_wood_percent': 'wood',
                                        'composition_paper_cardboard_percent': 'paper_cardboard',
                                        'composition_plastic_percent': 'plastic',
                                        'composition_metal_percent': 'metal',
                                        'composition_glass_percent': 'glass',
                                        'composition_other_percent': 'other',
                                        'composition_rubber_leather_percent': 'rubber'
                                        }, inplace=True)
        waste_fractions /= 100
        
        # Add zeros where there are no values unless all values are nan
        if waste_fractions.isna().all():
            waste_fractions = defaults.waste_fraction_defaults.loc[self.region, :]
        else:
            waste_fractions.fillna(0, inplace=True)
            waste_fractions['textiles'] = 0
        
        if (waste_fractions.sum() < .9) or (waste_fractions.sum() > 1.1):
            #print('waste fractions do not sum to 1')
            waste_fractions = defaults.waste_fraction_defaults.loc[self.region, :]
    
        self.waste_fractions = waste_fractions.to_dict()
        
        try:
            self.mef_compost = (0.0055 * waste_fractions['food']/(waste_fractions['food'] + waste_fractions['green']) + \
                           0.0139 * waste_fractions['green']/(waste_fractions['food'] + waste_fractions['green'])) * 1.1023 * 0.7 # / 28
                           # Unit is Mg CO2e/Mg of organic waste, wtf, so convert to CH4. Mistake in sweet here
        except:
            self.mef_compost = 0
        
        # Precipitation, remove this try except when there are no duplicates
        try:
            precip = rmi_db.at[self.name, 'total_precipitation(mm)_1970-2000'].iloc[0]
        except:
            precip = rmi_db.at[self.name, 'total_precipitation(mm)_1970-2000']
        self.precip_zone = defaults.get_precipitation_zone(precip)
    
        # depth
        #depth = 10
    
        # k values
        self.ks = defaults.k_defaults[self.precip_zone]
        
        # Model components
        self.components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles'])
        
        # Compost params
        self.compost_components = set(['food', 'green', 'wood', 'paper_cardboard']) # Double check we don't want to include paper
        self.compost_fraction = np.nan_to_num(row['waste_treatment_compost_percent']) / 100
        
        # Anaerobic digestion params
        self.anaerobic_components = set(['food', 'green', 'wood', 'paper_cardboard'])
        self.anaerobic_fraction = np.nan_to_num(row['waste_treatment_anaerobic_digestion_percent']) / 100   
        
        # Combustion params
        self.combustion_components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'])
        combustion_fraction_of_total = (np.nan_to_num(row['waste_treatment_incineration_percent']) + 
                                        np.nan_to_num(row['waste_treatment_advanced_thermal_treatment_percent']))/ 100
        self.combustion_fraction = combustion_fraction_of_total
        
        # Recycling params
        self.recycling_components = set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'])
        self.recycling_fraction = np.nan_to_num(row['waste_treatment_recycling_percent']) / 100
        
        self.gas_capture_percent = np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent']) / 100
        
        self.div_components = {}
        self.div_components['compost'] = self.compost_components
        self.div_components['anaerobic'] = self.anaerobic_components
        self.div_components['combustion'] = self.combustion_components
        self.div_components['recycling'] = self.recycling_components
        
        # landfill_w_capture = np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100
        # landfill_wo_capture = (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
        #                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100
        # dumpsite = (np.nan_to_num(row['waste_treatment_open_dump_percent']) +
        #              np.nan_to_num(row['waste_treatment_unaccounted_for_percent']))/100
        
        # all_waste_paths = landfill_w_capture + \
        #                   landfill_wo_capture + \
        #                   dumpsite + \
        #                   self.compost_fraction + \
        #                   self.anaerobic_fraction + \
        #                   self.combustion_fraction + \
        #                   self.recycling_fraction
        
        # if all_waste_paths > 1.01:
        
        # Determine split between landfill and dump site
        split_fractions = {'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
                           'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
                                       np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
                           'dumpsite': (np.nan_to_num(row['waste_treatment_open_dump_percent']) +
                                        np.nan_to_num(row['waste_treatment_unaccounted_for_percent']))/100}
        
        # Get the total that goes to landfill and dump site combined
        split_total = sum([split_fractions[x] for x in split_fractions.keys()])
        
        if split_total == 0:
            # Set to dump site only if no data. This gets changed later to country lookup for wealthier countries, default to landfill
            split_fractions = {'landfill_w_capture': 0, 'landfill_wo_capture': 0, 'dumpsite': 1}
        else:
            # # Calculate % of waste that goes to landfill and dump site, of waste
            # # going to one or the other
            # if (split_fractions['landfill_w_capture'] > 0) & (split_fractions['landfill_wo_capture'] == 0) & (split_fractions['dumpsite'] == 0):
            #     split_fractions = {'landfill_w_capture': split_fractions['landfill_w_capture'], 
            #                        'landfill_wo_capture': (1 - split_fractions['landfill_w_capture']), 
            #                        'dumpsite': 0}
            # else:
            for site in split_fractions.keys():
                split_fractions[site] /= split_total

        
        self.landfill_w_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=split_fractions['landfill_w_capture'], gas_capture=True)
        self.landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=split_fractions['landfill_wo_capture'], gas_capture=False)
        self.dumpsite = Landfill(self, 1960, 2073, 'dumpsite', 0.4, fraction_of_waste=split_fractions['dumpsite'], gas_capture=False)
        
        self.landfills = [self.landfill_w_capture, self.landfill_wo_capture, self.dumpsite]
        
        self.divs = {}
        self.div_fractions = {}
        self.div_fractions['compost'] = self.compost_fraction
        self.div_fractions['anaerobic'] = self.anaerobic_fraction
        self.div_fractions['combustion'] = self.combustion_fraction
        self.div_fractions['recycling'] = self.recycling_fraction
        self.div_component_fractions = {}
        self.calc_compost_vol()
        self.calc_anaerobic_vol()
        self.calc_combustion_vol()
        self.calc_recycling_vol()

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
        self.waste_masses = {x: self.waste_fractions[x] * self.waste_mass for x in self.waste_fractions.keys()}
        
        self.net_masses_before_check = {}
        for waste in self.waste_masses.keys():
            net_mass = self.waste_masses[waste] - (self.divs['compost'][waste] + self.divs['anaerobic'][waste] + self.divs['combustion'][waste] + self.divs['recycling'][waste])
            self.net_masses_before_check[waste] = net_mass

        self.changed_diversion, self.input_problems = self.check_masses()
    
        self.net_masses_after_check = {}
        for waste in self.waste_masses.keys():
            net_mass = self.waste_masses[waste] - (self.divs['compost'][waste] + self.divs['anaerobic'][waste] + self.divs['combustion'][waste] + self.divs['recycling'][waste])
            self.net_masses_after_check[waste] = net_mass

    # def load_un_habitat_params(self, row, rmi_db, un_recovered_materials):
        
    #     self.country = row[0].split('.')[0]
    #     row = row[1]
        
    #     self.region = defaults.region_lookup[self.country]
    #     #run_params['region'] = sweet_tools.region_lookup[run_params['country']]
        
    #     # Population, remove the try except when no duplicates
    #     self.population = float(row['Population']) # * (1.03 ** (2010 - 2023))
    #     try:
    #         population_1950 = rmi_db.at[self.name, '1950_Population'].iloc[0]
    #         population_2020 = rmi_db.at[self.name, '2020_Population'].iloc[0]
    #         population_2035 = rmi_db.at[self.name, '2035_Population'].iloc[0]
    #     except:
    #         population_1950 = rmi_db.at[self.name, '1950_Population']
    #         population_2020 = rmi_db.at[self.name, '2020_Population']
    #         population_2035 = rmi_db.at[self.name, '2035_Population']
    #     self.growth_rate_historic = ((population_2020 / population_1950) ** (1 / (2020 - 1950)))
    #     self.growth_rate_future = ((population_2035 / population_2020) ** (1 / (2035 - 2020)))
    
    #     # Get waste total
        
    #     self.waste_mass = float(row['MSW generated (t/d)']) # unit is tons/day
    #     self.waste_per_capita = self.waste_mass * 1000 / self.population # unit is kg/person/day

    #     if self.waste_mass != self.waste_mass:
    #         # Use per capita default
    #         self.waste_per_capita = defaults.msw_per_capita_defaults[self.region]
    #         self.waste_mass = self.waste_per_capita * self.population / 1000 * 365
        
    #     # # Collection coverage_stats
    #     # # Don't use these for now, as it seems like WB already adjusted total msw to account for these. 
    #     # coverage_by_area = float(row['waste_collection_coverage_total_percent_of_geographic_area_percent_of_geographic_area']) / 100
    #     # coverage_by_households = float(row['waste_collection_coverage_total_percent_of_households_percent_of_households']) / 100
    #     # coverage_by_pop = float(row['waste_collection_coverage_total_percent_of_population_percent_of_population']) / 100
    #     # coverage_by_waste = float(row['waste_collection_coverage_total_percent_of_waste_percent_of_waste']) / 100
        
    #     # if coverage_by_waste == coverage_by_waste:
    #     #     self.mass *= 
        
    #     # Waste fractions
    #     waste_fractions = row[['Kitchen/canteen (%)', 
    #                          'Garden/park (%)', 
    #                          'Paper/cardboard (%)',
    #                          'Plastic film (%)',
    #                          'Plastics dense (%)',
    #                          'Metals (%)',
    #                          'Glass (%)',
    #                          'Textiles/shoes (%)',
    #                          'Wood (processed) (%)',
    #                          'Special wastes (%)',
    #                          'Composite products (%)',
    #                          'Other (%)'
    #                          ]]
    
    #     waste_fractions.rename(index={'composition_food_organic_waste_percent': 'food',
    #                                     'composition_yard_garden_green_waste_percent': 'green',
    #                                     'composition_wood_percent': 'wood',
    #                                     'composition_paper_cardboard_percent': 'paper_cardboard',
    #                                     'composition_plastic_percent': 'plastic',
    #                                     'composition_metal_percent': 'metal',
    #                                     'composition_glass_percent': 'glass',
    #                                     'composition_other_percent': 'other',
    #                                     'composition_rubber_leather_percent': 'rubber'
    #                                     }, inplace=True)
    #     waste_fractions /= 100
        
    #     # Add zeros where there are no values unless all values are nan
    #     if waste_fractions.isna().all():
    #         waste_fractions = defaults.waste_fraction_defaults.loc[self.region, :]
    #     else:
    #         waste_fractions.fillna(0, inplace=True)
    #         waste_fractions['textiles'] = 0
        
    #     if (waste_fractions.sum() < .9) or (waste_fractions.sum() > 1.1):
    #         #print('waste fractions do not sum to 1')
    #         waste_fractions = defaults.waste_fraction_defaults.loc[self.region, :]
    
    #     self.waste_fractions = waste_fractions.to_dict()
        
    #     try:
    #         self.mef_compost = (0.0055 * waste_fractions['food']/(waste_fractions['food'] + waste_fractions['green']) + \
    #                        0.0139 * waste_fractions['green']/(waste_fractions['food'] + waste_fractions['green'])) * 1.1023 * 0.7 # / 28
    #                        # Unit is Mg CO2e/Mg of organic waste, wtf, so convert to CH4. Mistake in sweet here
    #     except:
    #         self.mef_compost = 0
        
    #     # Precipitation, remove this try except when there are no duplicates
    #     try:
    #         precip = rmi_db.at[self.name, 'total_precipitation(mm)_1970-2000'].iloc[0]
    #     except:
    #         precip = rmi_db.at[self.name, 'total_precipitation(mm)_1970-2000']
    #     self.precip_zone = defaults.get_precipitation_zone(precip)
    
    #     # depth
    #     #depth = 10
    
    #     # k values
    #     self.ks = defaults.k_defaults[self.precip_zone]
        
    #     # Model components
    #     self.components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles'])
        
    #     # Compost params
    #     self.compost_components = set(['food', 'green', 'wood', 'paper_cardboard']) # Double check we don't want to include paper
    #     self.compost_fraction = np.nan_to_num(row['waste_treatment_compost_percent']) / 100
        
    #     # Anaerobic digestion params
    #     self.anaerobic_components = set(['food', 'green', 'wood', 'paper_cardboard'])
    #     self.anaerobic_fraction = np.nan_to_num(row['waste_treatment_anaerobic_digestion_percent']) / 100   
        
    #     # Combustion params
    #     self.combustion_components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'])
    #     combustion_fraction = (np.nan_to_num(row['waste_treatment_incineration_percent']) + 
    #                                     np.nan_to_num(row['waste_treatment_advanced_thermal_treatment_percent']))/ 100
    #     self.combustion_fraction = combustion_fraction
        
    #     # Recycling params
    #     self.recycling_components = set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'])
        
    #     self.recycling_fraction = np.nan_to_num(row['waste_treatment_recycling_percent']) / 100
        
    #     self.gas_capture_percent = np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent']) / 100
        
    #     # landfill_w_capture = np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100
    #     # landfill_wo_capture = (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
    #     #                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100
    #     # dumpsite = (np.nan_to_num(row['waste_treatment_open_dump_percent']) +
    #     #              np.nan_to_num(row['waste_treatment_unaccounted_for_percent']))/100
        
    #     # all_waste_paths = landfill_w_capture + \
    #     #                   landfill_wo_capture + \
    #     #                   dumpsite + \
    #     #                   self.compost_fraction + \
    #     #                   self.anaerobic_fraction + \
    #     #                   self.combustion_fraction + \
    #     #                   self.recycling_fraction
        
    #     # if all_waste_paths > 1.01:
        
    #     # Determine split between landfill and dump site
    #     split_fractions = {'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
    #                        'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
    #                                    np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
    #                        'dumpsite': (np.nan_to_num(row['waste_treatment_open_dump_percent']) +
    #                                     np.nan_to_num(row['waste_treatment_unaccounted_for_percent']))/100}
        
    #     # Get the total that goes to landfill and dump site combined
    #     split_total = sum([split_fractions[x] for x in split_fractions.keys()])
        
    #     if split_total == 0:
    #         # Set to dump site only if no data. This gets changed later to country lookup for wealthier countries, default to landfill
    #         split_fractions = {'landfill_w_capture': 0, 'landfill_wo_capture': 0, 'dumpsite': 1}
    #     else:
    #         # # Calculate % of waste that goes to landfill and dump site, of waste
    #         # # going to one or the other
    #         # if (split_fractions['landfill_w_capture'] > 0) & (split_fractions['landfill_wo_capture'] == 0) & (split_fractions['dumpsite'] == 0):
    #         #     split_fractions = {'landfill_w_capture': split_fractions['landfill_w_capture'], 
    #         #                        'landfill_wo_capture': (1 - split_fractions['landfill_w_capture']), 
    #         #                        'dumpsite': 0}
    #         # else:
    #         for site in split_fractions.keys():
    #             split_fractions[site] /= split_total
    
        
    #     self.landfill_w_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=split_fractions['landfill_w_capture'], gas_capture=True)
    #     self.landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=split_fractions['landfill_wo_capture'], gas_capture=False)
    #     self.dumpsite = Landfill(self, 1960, 2073, 'dumpsite', 0.4, fraction_of_waste=split_fractions['dumpsite'], gas_capture=False)
        
    #     self.landfills = [self.landfill_w_capture, self.landfill_wo_capture, self.dumpsite]
        
    #     self.divs = {}
    #     self.calc_compost_vol()
    #     self.calc_anaerobic_vol()
    #     self.calc_combustion_vol()
    #     self.calc_recycling_vol()
    
    #     for c in self.waste_fractions.keys():
    #         if c not in self.divs['compost'].keys():
    #             self.divs['compost'][c] = 0
    #         if c not in self.divs['anaerobic'].keys():
    #             self.divs['anaerobic'][c] = 0
    #         if c not in self.divs['combustion'].keys():
    #             self.divs['combustion'][c] = 0
    #         if c not in self.divs['recycling'].keys():
    #             self.divs['recycling'][c] = 0
        
    #     # Save waste diverions calculated with default assumptions, and then update them if any components are net negative.
    #     self.divs_before_check = copy.deepcopy(self.divs)
    #     self.waste_masses = {x: self.waste_fractions[x] * self.waste_mass for x in self.waste_fractions.keys()}
        
    #     self.net_masses_before_check = {}
    #     for waste in self.waste_masses.keys():
    #         net_mass = self.waste_masses[waste] - (self.divs['compost'][waste] + self.divs['anaerobic'][waste] + self.divs['combustion'][waste] + self.divs['recycling'][waste])
    #         self.net_masses_before_check[waste] = net_mass
    
    #     self.changed_diversion, self.input_problems = self.check_masses()
    
    #     self.net_masses_after_check = {}
    #     for waste in self.waste_masses.keys():
    #         net_mass = self.waste_masses[waste] - (self.divs['compost'][waste] + self.divs['anaerobic'][waste] + self.divs['combustion'][waste] + self.divs['recycling'][waste])
    #         self.net_masses_after_check[waste] = net_mass
    
    def calc_compost_vol(self):
        # Helps to set to 0 at start
        compost_total = 0
        
        # Total mass of compost
        compost_total = self.compost_fraction * self.waste_mass
        
        # Sum of fraction of waste types that are compostable
        fraction_compostable_types = sum([self.waste_fractions[x] for x in self.compost_components])
        self.unprocessable = {'food': .0192, 'green': .042522, 'wood': .07896, 'paper_cardboard': .12}
        
        if self.compost_fraction != 0:
            compost_waste_fractions = {x: self.waste_fractions[x] / fraction_compostable_types for x in self.compost_components}
            #non_compostable_not_targeted = .1 # I don't know what this means, basically, waste that gets composted that shouldn't have been and isn't compostable?
            non_compostable_not_targeted = {'food': .1, 'green': .05, 'wood': .05, 'paper_cardboard': .1}
            non_compostable_not_targeted_total = sum([non_compostable_not_targeted[x] * \
                                                      compost_waste_fractions[x] for x in self.compost_components])
            self.divs['compost'] = {}
            for waste in self.compost_components:
                self.divs['compost'][waste] = (
                    compost_total * 
                    (1 - non_compostable_not_targeted_total) *
                    compost_waste_fractions[waste] *
                    (1 - self.unprocessable[waste])
                    )
        else:
            self.divs['compost'] = {x: 0 for x in self.compost_components}
            compost_waste_fractions = {x: 0 for x in self.compost_components}
            non_compostable_not_targeted = {'food': 0, 'green': 0, 'wood': 0, 'paper_cardboard': 0}
            non_compostable_not_targeted_total = 0
            
        self.compost_total = compost_total
        self.fraction_compostable_types = fraction_compostable_types
        self.compost_waste_fractions = compost_waste_fractions
        self.div_component_fractions['compost'] = compost_waste_fractions
        self.non_compostable_not_targeted = non_compostable_not_targeted
        self.non_compostable_not_targeted_total = non_compostable_not_targeted_total
    
    def calc_anaerobic_vol(self):
        anaerobic_total = 0
        fraction_anaerobic_types = sum([self.waste_fractions[x] for x in self.anaerobic_components])
        if self.anaerobic_fraction != 0:
            anaerobic_total = self.anaerobic_fraction * self.waste_mass
            #print(anaerobic_total)
            anaerobic_waste_fractions = {x: self.waste_fractions[x] / fraction_anaerobic_types for x in self.anaerobic_components}
            self.divs['anaerobic'] = {x: anaerobic_total * anaerobic_waste_fractions[x] for x in self.anaerobic_components}
        else:
            self.divs['anaerobic'] = {x: 0 for x in self.anaerobic_components}
            anaerobic_waste_fractions = {x: 0 for x in self.anaerobic_components}
        
        self.anaerobic_total = anaerobic_total
        #params['fraction_anaerobic_types'] = fraction_anaerobic_types
        self.anaerobic_waste_fractions = anaerobic_waste_fractions
        self.div_component_fractions['anaerobic'] = anaerobic_waste_fractions
    
    def calc_combustion_vol(self):
        self.combustion_total = self.combustion_fraction * self.waste_mass
        combustion_reject_rate = 0 #.1 I think sweet has an error, the rejected from combustion stuff just disappears
        if self.combustion_fraction != 0:
            self.divs['combustion'] = {x: self.waste_fractions[x] * \
                                  self.combustion_fraction * \
                                  (1 - combustion_reject_rate) * \
                                  self.waste_mass for x in self.combustion_components}
        else:
            self.divs['combustion'] = {x: 0 for x in self.combustion_components}
            
        self.combustion_reject_rate = combustion_reject_rate
        # BELOW IS WRONG, its just how sweet does it.
        fraction_combustion_types = sum([self.waste_fractions[x] for x in self.combustion_components])
        self.combustion_waste_fractions = {x: self.waste_fractions[x] / fraction_combustion_types for x in self.combustion_components}
        self.div_component_fractions['combustion'] = self.combustion_waste_fractions
        #self.combustion_waste_fractions = self.waste_fractions
    
    def calc_recycling_vol(self):
        self.recycling_total = self.recycling_fraction * self.waste_mass
        fraction_recyclable_types = sum([self.waste_fractions[x] for x in self.recycling_components])
        recycling_reject_rates = {'wood': .8, 'paper_cardboard': .775, 'textiles': .99, 
                                  'plastic': .875, 'metal': .955, 'glass': .88, 
                                  'rubber': .78, 'other': .87}
        if self.recycling_fraction != 0:
            recycling_waste_fractions = {x: self.waste_fractions[x] / fraction_recyclable_types for x in self.recycling_components}
            self.divs['recycling'] = {x: self.waste_fractions[x] / \
                              fraction_recyclable_types * \
                              self.recycling_fraction * \
                              (recycling_reject_rates[x]) * \
                              self.waste_mass for x in self.recycling_components}
            #recycling_vol_total = sum([recycling_vol[x] for x in recycling_vol.keys()])
        else:
            self.divs['recycling'] = {x: 0 for x in self.recycling_components}
            recycling_waste_fractions = {x: 0 for x in self.recycling_components}
        
        self.fraction_recyclable_types = fraction_recyclable_types
        self.recycling_reject_rates = recycling_reject_rates
        self.recycling_waste_fractions = recycling_waste_fractions
        self.div_component_fractions['recycling'] = recycling_waste_fractions
        
        #self.divs = pd.DataFrame(self.divs)
    
    # def estimate_diversion_emissions(self):
        
    #     for div in self.divs.keys():
    #         self.divs[div]['ms'] = {}
    #         self.divs[div]['qs'] = {}
            
    #         years = np.arange(2023, 2073)
    #         t_values = years - 2023
            
    #         for waste in self.divs[div].keys():
    #             self.divs[div]['ms'][waste] = {}
                
    #             # for year in range(2023, 2073):
    #             #     t = year - 2023
    #             #     self.divs[div]['ms'][waste][year] = self.divs[div][waste] * (1.03 ** t)
                    
    #             #     if div == 'compost':
    #             #         self.divs[div]['qs'][waste][year] = self.divs[div]['ms'][waste][year] * self.mef_compost
    #             #     elif div == 'anaerobic':
    #             #         self.divs[div]['qs'][waste][year] = self.divs[div]['ms'][waste][year] * defaults.mef_anaerobic * defaults.ch4_to_co2e
    #             #     else:
    #             #         continue
            
                
    #             self.divs[div]['ms'][waste] = pd.Series(self.divs[div][waste] * (1.03 ** t_values), index=years)
                
    #             if div == 'compost':
    #                 self.divs[div]['qs'][waste] = self.divs[div]['ms'][waste] * self.mef_compost
    #             elif div == 'anaerobic':
    #                 self.divs[div]['qs'][waste] = self.divs[div]['ms'][waste] * defaults.mef_anaerobic * defaults.ch4_to_co2e
            
    #         # Create DataFrames from dictionaries
    #         compost_qs_df = pd.DataFrame(self.divs['compost']['qs'])
    #         anaerobic_qs_df = pd.DataFrame(self.divs['anaerobic']['qs'])
            
    #         self.organic_emissions = compost_qs_df.add(anaerobic_qs_df)
    
    def estimate_diversion_emissions(self):
        
        # REMOVE THE MASS CALCULATION PART OF THIS IT SHOULD BE HAPPENING ALREADY IN DIVS
        
        # Define years and t_values 
        years_historic = np.arange(1960, 2016)
        years_future = np.arange(2016, 2073)
        t_values_historic = years_historic - 2016
        t_values_future = years_future - 2016
        
        # Initialize empty DataFrames to hold 'ms' and 'qs' values for each diversion type
        ms_dict = {}
        qs_dict = {}
        
        # Iterate over each diversion type
        for div in self.divs.keys():
            # Create dataframe with years from div dictionary. All values should be the same, no exponential growth yet
            div_data_historic = pd.DataFrame({waste_type: [value] * len(years_historic) for waste_type, value in self.divs[div].items()}, index=years_historic)
            div_data_future = pd.DataFrame({waste_type: [value] * len(years_future) for waste_type, value in self.divs[div].items()}, index=years_future)
            
            # Compute 'ms' values
            #ms = div_data * (1.03 ** (t_values[:, np.newaxis]))
            ms_historic = div_data_historic * (self.growth_rate_historic ** (t_values_historic[:, np.newaxis]))
            ms_future = div_data_future * (self.growth_rate_future ** (t_values_future[:, np.newaxis]))
            ms_dict[div] = pd.concat((ms_historic, ms_future), axis=0)
        
            # Compute 'qs' values based on the diversion type
            if div == 'compost':
                qs = ms_dict[div] * self.mef_compost
            elif div == 'anaerobic':
                qs = ms_dict[div] * defaults.mef_anaerobic * defaults.ch4_to_co2e
            else:
                qs = None
        
            qs_dict[div] = qs
        
        # Store the total organic emissions, only adding compost and anaerobic
        self.organic_emissions = qs_dict['compost'].add(qs_dict['anaerobic'], fill_value=0)

    def sum_landfill_emissions(self):
        self.landfill_emissions = [x.emissions.applymap(self.convert_methane_m3_to_ton_co2e) for x in self.landfills]
        self.landfill_emissions.append(self.organic_emissions.loc[:, list(self.components)])

        # Concatenate all emissions dataframes
        all_emissions = pd.concat(self.landfill_emissions, axis=0)
        
        # Group by the year index and sum the emissions for each year
        summed_emissions = all_emissions.groupby(all_emissions.index).sum()
        
        self.total_emissions = summed_emissions
    
    def check_masses(self):
        #masses = {x: self.waste_fractions[x] * self.waste_mass for x in self.waste_fractions.keys()}
        
        fractions_before = {}
        for div in self.divs.keys():
            fractions_before[div] = sum([x for x in self.divs[div].values()]) / self.waste_mass
        
        
        problems = [set()]
        net_masses = {}
        for waste in self.waste_masses.keys():
            net_mass = self.waste_masses[waste] - sum([self.divs[x][waste] for x in self.divs.keys()])
            net_masses[waste] = net_mass
            if net_mass < 0:
                #print(self.name)
                problems[0].add(waste)
        dont_add_to = problems[0].copy()
        #old_net_masses = copy.deepcopy(net_masses)
        
        for waste in self.waste_masses.keys():
            if self.waste_masses[waste] == 0:
                dont_add_to.add(waste)
        
        if len(problems[0]) == 0:
            return False, False
        
        while problems:
            probs = problems.pop(0)
            for waste in probs:
                deficit = -net_masses[waste]
                total_subtracted = sum([self.divs[x][waste] for x in self.divs.keys()])
                # fractions = [compost_vol[waste] / total_subtracted, 
                #              anaerobic_vol[waste] / total_subtracted, 
                #              combustion_vol[waste] / total_subtracted, 
                #              recycling_vol[waste] / total_subtracted]
                
                fraction_to_fix = deficit / total_subtracted
                # add_back_amounts = {'compost': compost_vol[waste] * fraction_to_fix, 
                #                     'anaerobic': anaerobic_vol[waste] * fraction_to_fix, 
                #                     'combustion': combustion_vol[waste] * fraction_to_fix, 
                #                     'recycling': recycling_vol[waste] * fraction_to_fix}
                
                add_back_amounts = {}
                for div in self.divs.keys():
                    if div == 'compost':
                        if waste in self.unprocessable:
                            add_back_amounts[div] = self.divs[div][waste] * fraction_to_fix / \
                                                    (1 - self.non_compostable_not_targeted_total) / \
                                                    (1 - self.unprocessable[waste])
                        else:
                            assert self.divs[div][waste] == 0, 'Hope this doesnt happen'
                            add_back_amounts[div] = 0
                    elif div == 'combustion':
                        #continue
                        add_back_amounts[div] = self.divs[div][waste] * fraction_to_fix / \
                                                (1 - self.combustion_reject_rate)
                    elif div == 'recycling':
                        if waste in self.recycling_reject_rates:
                            add_back_amounts[div] = self.divs[div][waste] * fraction_to_fix / \
                                                    (self.recycling_reject_rates[waste])
                        else:
                            assert self.divs[div][waste] == 0, 'Hope this doesnt happen'
                            add_back_amounts[div] = 0
                    else:
                        add_back_amounts[div] = self.divs[div][waste] * fraction_to_fix
                        
                    # Don't adjust the amount subtracted by the efficiency losses, this is the important part
                    self.divs[div][waste] -= self.divs[div][waste] * fraction_to_fix
                
                for div in self.divs.keys():
                    amount = add_back_amounts[div]
                    if amount == 0:
                        continue
                    types_to_add_to = [x for x in getattr(self, f"{div}_waste_fractions").keys() if x not in dont_add_to]
                    fraction_of_types_adding_to = sum([getattr(self, f"{div}_waste_fractions")[x] for x in types_to_add_to])
                    
                    if (amount > 0) & (fraction_of_types_adding_to == 0):
                        return True, True
                    
                    for w in types_to_add_to:
                        if div == 'compost':
                            self.divs[div][w] += amount * getattr(self, f"{div}_waste_fractions")[w] / fraction_of_types_adding_to * \
                                            (1 - self.non_compostable_not_targeted_total) * \
                                            (1 - self.unprocessable[w])                                
                        elif div == 'combustion':
                            #continue
                            self.divs[div][w] += amount * getattr(self, f"{div}_waste_fractions")[w] / fraction_of_types_adding_to * \
                                            (1 - self.combustion_reject_rate)
                        elif div == 'recycling':
                            self.divs[div][w] += amount * getattr(self, f"{div}_waste_fractions")[w] / fraction_of_types_adding_to * \
                                            (self.recycling_reject_rates[w])
                        else:
                            self.divs[div][w] += amount * getattr(self, f"{div}_waste_fractions")[w] / fraction_of_types_adding_to
                        
            net_masses = {}
            new_probs = set()
            for waste in self.waste_masses.keys():
                net_mass = self.waste_masses[waste] - sum([self.divs[x][waste] for x in self.divs.keys()])
                net_masses[waste] = net_mass
                if (net_mass < -0.1):
                    new_probs.add(waste)
                    dont_add_to.add(waste)
                    
            if len(new_probs) > 0:
                problems.append(new_probs)
        
        fractions_after = {}
        for div in self.divs.keys():
            fractions_after[div] = sum([x for x in self.divs[div].values()]) / self.waste_mass
            
        # for div in divs.keys():
        #     assert (fractions_before[div] - fractions_after[div]) < .01, 'total diversion fractions should not change'
            
        # original_fractions = {'compost': {}}
        # for waste in self.compost_components:
        #     final_mass_after_adjustment = self.divs['compost'][waste]
        #     input_mass_after_adjustment = final_mass_after_adjustment / \
        #                                   (1 - self.non_compostable_not_targeted_total) / \
        #                                   (1 - self.unprocessable[waste])
        #     try:
        #         original_fractions['compost'][waste] = input_mass_after_adjustment / self.compost_total
        #         #print(run_params['city'], run_params['compost_total'])
        #     except:
        #         original_fractions['compost'][waste] = np.nan
            
        # original_fractions['anaerobic'] = {}
        # for waste in self.anaerobic_components:
        #     final_mass_after_adjustment = self.divs['anaerobic'][waste]
        #     try:
        #         original_fractions['anaerobic'][waste] = final_mass_after_adjustment / self.anaerobic_total
        #     except:
        #         original_fractions['anaerobic'][waste] = np.nan
            
        # original_fractions['combustion'] = {}
        # for waste in self.combustion_components:
        #     final_mass_after_adjustment = self.divs['combustion'][waste]
        #     input_mass_after_adjustment = final_mass_after_adjustment / \
        #                                   (1 - self.combustion_reject_rate)
        #     try:
        #         original_fractions['combustion'][waste] = input_mass_after_adjustment / self.combustion_total
        #     except:
        #         original_fractions['combustion'][waste] = np.nan
            
        # original_fractions['recycling'] = {}
        # for waste in self.recycling_components:
        #     final_mass_after_adjustment = self.divs['recycling'][waste]
        #     input_mass_after_adjustment = final_mass_after_adjustment / \
        #                                   (self.recycling_reject_rates[waste])
        #     try:
        #         original_fractions['recycling'][waste] = input_mass_after_adjustment / self.recycling_total
        #     except:
        #         original_fractions['recycling'][waste] = np.nan
        
        return True, False    
    
    def convert_methane_m3_to_ton_co2e(self, volume_m3):
        density_kg_per_m3 = 0.7168
        mass_kg = volume_m3 * density_kg_per_m3
        mass_ton = mass_kg / 1000
        mass_co2e = mass_ton * 28
        return mass_co2e

    def convert_co2e_to_methane_m3(mass_co2e):
        density_kg_per_m3 = 0.7168
        mass_ton = mass_co2e / 28
        mass_kg = mass_ton * 1000
        volume_m3 = mass_kg / density_kg_per_m3
        return volume_m3

class Landfill:
    def __init__(self, city, open_date, close_date, site_type, mcf, fraction_of_waste=1, gas_capture=False):
        
        self.city = city
        self.open_date = open_date
        self.close_date = close_date
        self.site_type = site_type
        self.mcf = mcf
        self.fraction_of_waste = fraction_of_waste
        self.gas_capture = gas_capture
        if self.gas_capture:
            self.gas_capture_efficiency = defaults.gas_capture_efficiency[site_type]
            self.oxidation_factor = defaults.oxidation_factor['with_lfg'][site_type]
        else:
            self.gas_capture_efficiency = 0
            self.oxidation_factor = defaults.oxidation_factor['without_lfg'][site_type]
        
    def estimate_emissions(self):
        self.model = SWEET(self, self.city)
        #self.waste_mass, self.emissions = self.model.estimate_emissions()
        # This is due to paper coardboard thing
        self.waste_mass, self.emissions, self.ch4, self.captured = self.model.estimate_emissions_match_excel()
        