from . import defaults
#import defaults
import pandas as pd
import numpy as np
from .model import SWEET
#from model import SWEET
import copy
from . import city_manual_baselines
#import city_manual_baselines

class City:
    def __init__(self, name):
        self.name = name
    
    def load_from_database(self, db):
        
        self.country = db.loc[self.name, 'Country'].values[0]
        self.lat = db.loc[self.name, 'Latitude'].values[0]
        self.lon = db.loc[self.name, 'Longitude'].values[0]
        self.waste_mass = db.loc[self.name, 'Waste Generation Rate (tons/year)'].values[0]
        self.data_source = db.loc[self.name, 'Input Data Source'].values[0]
        self.population = db.loc[self.name, 'Population'].values[0]

        self.waste_fractions = {}
        self.waste_fractions['food'] = db.loc[self.name, 'Waste Components: Food (%)'].values[0] / 100
        self.waste_fractions['green'] = db.loc[self.name, 'Waste Components: Green (%)'].values[0] / 100
        self.waste_fractions['wood'] = db.loc[self.name, 'Waste Components: Wood (%)'].values[0] / 100
        self.waste_fractions['paper_cardboard'] = db.loc[self.name, 'Waste Components: Paper and Cardboard (%)'].values[0] / 100
        self.waste_fractions['textiles'] = db.loc[self.name, 'Waste Components: Textiles (%)'].values[0] / 100
        self.waste_fractions['plastic'] = db.loc[self.name, 'Waste Components: Plastic (%)'].values[0] / 100
        self.waste_fractions['metal'] = db.loc[self.name, 'Waste Components: Metal (%)'].values[0] / 100
        self.waste_fractions['glass'] = db.loc[self.name, 'Waste Components: Glass (%)'].values[0] / 100
        self.waste_fractions['rubber'] = db.loc[self.name, 'Waste Components: Rubber/Leather (%)'].values[0] / 100
        self.waste_fractions['other'] = db.loc[self.name, 'Waste Components: Other (%)'].values[0] / 100

        self.div_fractions = {}
        self.div_fractions['compost'] = db.loc[self.name, 'Diversons: Compost (%)'].values[0] / 100
        self.div_fractions['anaerobic'] = db.loc[self.name, 'Diversons: Anaerobic Digestion (%)'].values[0] / 100
        self.div_fractions['combustion'] = db.loc[self.name, 'Diversons: Incineration (%)'].values[0] / 100
        self.div_fractions['recycling'] = db.loc[self.name, 'Diversons: Recycling (%)'].values[0] / 100

        self.precip = db.loc[self.name, 'Average Annual Precipitation (mm/year)'].values[0]
        self.growth_rate_historic = db.loc[self.name, 'Population Growth Rate: Historic (%)'].values[0] / 100 + 1
        self.growth_rate_future = db.loc[self.name, 'Population Growth Rate: Future (%)'].values[0] / 100 + 1
        self.waste_per_capita = db.loc[self.name, 'Waste Generation Rate per Capita (kg/person/day)'].values[0]
        self.informal_fraction = db.loc[self.name, 'Informal Waste Collection Rate (%)'].values[0] / 100

        self.split_fractions = {}
        self.split_fractions['landfill_w_capture'] = db.loc[self.name, 'Percent of Waste to Landfills with Gas Capture (%)'].values[0] / 100
        self.split_fractions['landfill_wo_capture'] = db.loc[self.name, 'Percent of Waste to Landfills without Gas Capture (%)'].values[0] / 100
        self.split_fractions['dumpsite'] = db.loc[self.name, 'Percent of Waste to Dumpsites (%)'].values[0] / 100

        self.div_component_fractions = {}
        for div in self.div_fractions:
            self.div_component_fractions[div] = {}
        self.div_component_fractions['compost']['food'] = db.loc[self.name, 'Diversion Components: Composted Food (% of Total Composted)'].values[0] / 100
        self.div_component_fractions['compost']['green'] = db.loc[self.name, 'Diversion Components: Composted Green (% of Total Composted)'].values[0] / 100
        self.div_component_fractions['compost']['wood'] = db.loc[self.name, 'Diversion Components: Composted Wood (% of Total Composted)'].values[0] / 100
        self.div_component_fractions['compost']['paper_cardboard'] = db.loc[self.name, 'Diversion Components: Composted Paper and Cardboard (% of Total Composted)'].values[0] / 100

        self.div_component_fractions['anaerobic']['food'] = db.loc[self.name, 'Diversion Components: Anaerobically Digested Food (% of Total Digested)'].values[0] / 100
        self.div_component_fractions['anaerobic']['green'] = db.loc[self.name, 'Diversion Components: Anaerobically Digested Green (% of Total Digested)'].values[0] / 100
        self.div_component_fractions['anaerobic']['wood'] = db.loc[self.name, 'Diversion Components: Anaerobically Digested Wood (% of Total Digested)'].values[0] / 100
        self.div_component_fractions['anaerobic']['paper_cardboard'] = db.loc[self.name, 'Diversion Components: Anaerobically Digested Paper and Cardboard (% of Total Digested)'].values[0] / 100

        self.div_component_fractions['combustion']['food'] = db.loc[self.name, 'Diversion Components: Incinerated Food (% of Total Incinerated)'].values[0] / 100
        self.div_component_fractions['combustion']['green'] = db.loc[self.name, 'Diversion Components: Incinerated Green (% of Total Incinerated)'].values[0] / 100
        self.div_component_fractions['combustion']['wood']  = db.loc[self.name, 'Diversion Components: Incinerated Wood (% of Total Incinerated)'].values[0] / 100
        self.div_component_fractions['combustion']['paper_cardboard'] = db.loc[self.name, 'Diversion Components: Incinerated Paper and Cardboard (% of Total Incinerated)'].values[0] / 100
        self.div_component_fractions['combustion']['plastic'] = db.loc[self.name, 'Diversion Components: Incinerated Plastic (% of Total Incinerated)'].values[0] / 100
        self.div_component_fractions['combustion']['rubber'] = db.loc[self.name, 'Diversion Components: Incinerated Rubber/Leather (% of Total Incinerated)'].values[0] / 100
        self.div_component_fractions['combustion']['textiles'] = db.loc[self.name, 'Diversion Components: Incinerated Textiles (% of Total Incinerated)'].values[0] / 100

        self.div_component_fractions['recycling']['wood'] = db.loc[self.name, 'Diversion Components: Recycled Wood (% of Total Recycled)'].values[0] / 100
        self.div_component_fractions['recycling']['paper_cardboard'] = db.loc[self.name, 'Diversion Components: Recycled Paper and Cardboard (% of Total Recycled)'].values[0] / 100
        self.div_component_fractions['recycling']['plastic'] = db.loc[self.name, 'Diversion Components: Recycled Plastic (% of Total Recycled)'].values[0] / 100
        self.div_component_fractions['recycling']['rubber'] = db.loc[self.name, 'Diversion Components: Recycled Rubber/Leather (% of Total Recycled)'].values[0] / 100
        self.div_component_fractions['recycling']['textiles'] = db.loc[self.name, 'Diversion Components: Recycled Textiles (% of Total Recycled)'].values[0] / 100
        self.div_component_fractions['recycling']['glass'] = db.loc[self.name, 'Diversion Components: Recycled Glass (% of Total Recycled)'].values[0] / 100
        self.div_component_fractions['recycling']['metal'] = db.loc[self.name, 'Diversion Components: Recycled Metal (% of Total Recycled)'].values[0] / 100
        self.div_component_fractions['recycling']['other'] = db.loc[self.name, 'Diversion Components: Recycled Other (% of Total Recycled)'].values[0] / 100

        # for waste in self.waste_fractions:
        #     for div in self.div_component_fractions:
        #         if waste not in self.div_component_fractions[div]:
        #             self.div_component_fractions[div][waste] = 0
        
        self.precip_zone = defaults.get_precipitation_zone(self.precip)
    
        # k values
        self.ks = defaults.k_defaults[self.precip_zone]
        
        # Model components
        self.components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles'])
        self.compost_components = set(['food', 'green', 'wood', 'paper_cardboard']) # Double check we don't want to include paper
        self.anaerobic_components = set(['food', 'green', 'wood', 'paper_cardboard'])
        self.combustion_components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'])
        self.recycling_components = set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'])

        self.gas_capture_efficiency = db.loc[self.name, 'Methane Capture Efficiency (%)'].values[0] / 100
        self.landfill_w_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_w_capture'], gas_capture=True)
        self.landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_wo_capture'], gas_capture=False)
        self.dumpsite = Landfill(self, 1960, 2073, 'dumpsite', 0.4, fraction_of_waste=self.split_fractions['dumpsite'], gas_capture=False)
        
        self.landfills = [self.landfill_w_capture, self.landfill_wo_capture, self.dumpsite]
        
        try:
            self.mef_compost = (0.0055 * self.waste_fractions['food']/(self.waste_fractions['food'] + self.waste_fractions['green']) + \
                           0.0139 * self.waste_fractions['green']/(self.waste_fractions['food'] + self.waste_fractions['green'])) * 1.1023 * 0.7 # / 28
                           # Unit is Mg CO2e/Mg of organic waste, wtf, so convert to CH4. Mistake in sweet here
        except:
            self.mef_compost = 0

        self.waste_masses = {}
        for waste in self.waste_fractions:
            self.waste_masses[waste] = self.waste_fractions[waste] * self.waste_mass

        self.divs = {}
        for div, fracs in self.div_component_fractions.items():
            self.divs[div] = {}
            s = sum([x for x in fracs.values()])
            # make sure the component fractions add up to 1
            if (s != 0) and (np.absolute(1 - s) > 0.01):
                print(s, 'problems', div)
            for waste in fracs.keys():
                self.divs[div][waste] = self.waste_mass * self.div_fractions[div] * self.div_component_fractions[div][waste]

        # Set divs to 0 for components that are not in the diversion
        for c in self.waste_fractions:
            if c not in self.divs['compost']:
                self.divs['compost'][c] = 0
            if c not in self.divs['anaerobic']:
                self.divs['anaerobic'][c] = 0
            if c not in self.divs['combustion']:
                self.divs['combustion'][c] = 0
            if c not in self.divs['recycling']:
                self.divs['recycling'][c] = 0

        # Some compost params
        self.unprocessable = {'food': .0192, 'green': .042522, 'wood': .07896, 'paper_cardboard': .12}
        self.non_compostable_not_targeted = {'food': .1, 'green': .05, 'wood': .05, 'paper_cardboard': .1}
        self.non_compostable_not_targeted_total = sum([
            self.non_compostable_not_targeted[x] * \
            self.div_component_fractions['compost'][x] for x in self.compost_components
        ])
        
        # More for other divs
        self.combustion_reject_rate = 0.1
        self.recycling_reject_rates = {
            'wood': .8, 
            'paper_cardboard': .775, 
            'textiles': .99, 
            'plastic': .875, 
            'metal': .955, 
            'glass': .88, 
            'rubber': .78, 
            'other': .87
        }

        # Reduce them by non-compostable and unprocessable and etc rates
        for waste in self.compost_components:
            self.divs['compost'][waste] = (
                self.divs['compost'][waste]  * 
                (1 - self.non_compostable_not_targeted_total) *
                (1 - self.unprocessable[waste])
            )
        for waste in self.combustion_components:
            self.divs['combustion'][waste] = (
                self.divs['combustion'][waste]  * 
                (1 - self.combustion_reject_rate)
            )
        for waste in self.recycling_components:
            self.divs['recycling'][waste] = (
                self.divs['recycling'][waste]  * 
                self.recycling_reject_rates[waste]
            )

        if self.name == 'Kitakyushu':
            self.divs['combustion']['paper_cardboard'] -= 410

        self.net = self.calculate_net_masses(self.divs)
        for waste in self.net.values():
            assert waste >= -1, 'Waste diversion is net negative'

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

    def load_wb_params(self, row, rmi_db):
        
        #idx = row[0]
        row = row[1]
        
        self.country = row['country_name']
        self.region = defaults.region_lookup[self.country]
        #run_params['region'] = sweet_tools.region_lookup[run_params['country']]
        
        # Population, remove the try except when no duplicates
        self.population = float(row['population_population_number_of_people']) # * (1.03 ** (2010 - 2023))
        try:
            population_1950 = rmi_db.at[self.name, 'Population_1950'].iloc[0]
            population_2020 = rmi_db.at[self.name, 'Population_2020'].iloc[0]
            population_2035 = rmi_db.at[self.name, 'Population_2035'].iloc[0]
        except:
            population_1950 = rmi_db.at[self.name, 'Population_1950']
            population_2020 = rmi_db.at[self.name, 'Population_2020']
            population_2035 = rmi_db.at[self.name, 'Population_2035']
        self.growth_rate_historic = ((population_2020 / population_1950) ** (1 / (2020 - 1950)))
        self.growth_rate_future = ((population_2035 / population_2020) ** (1 / (2035 - 2020)))
        
        # lat lon
        try:
            self.lat = rmi_db.at[self.name, 'latitude_original'].values[0]
            self.lon = rmi_db.at[self.name, 'longitude_original'].values[0]
        except:
            self.lat = rmi_db.at[self.name, 'latitude_original']
            self.lon = rmi_db.at[self.name, 'longitude_original']

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
        self.informal_fraction = np.nan_to_num(row['percent_informal_sector_percent_collected_by_informal_sector_percent']) / 100
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
        
        s = sum([x for x in self.waste_fractions.values()])
        self.waste_fractions = {x: self.waste_fractions[x] / s for x in self.waste_fractions.keys()}

        try:
            self.mef_compost = (0.0055 * waste_fractions['food']/(waste_fractions['food'] + waste_fractions['green']) + \
                           0.0139 * waste_fractions['green']/(waste_fractions['food'] + waste_fractions['green'])) * 1.1023 * 0.7 # / 28
                           # Unit is Mg CO2e/Mg of organic waste, wtf, so convert to CH4. Mistake in sweet here
        except:
            self.mef_compost = 0
        
        # Precipitation, remove this try except when there are no duplicates
        try:
            self.precip = rmi_db.at[self.name, 'total_precipitation(mm)_1970-2000'].iloc[0]
        except:
            self.precip = rmi_db.at[self.name, 'total_precipitation(mm)_1970-2000']
        self.precip_zone = defaults.get_precipitation_zone(self.precip)
    
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
        
        # if all_waste_paths > 1.01:
        
        # Determine split between landfill and dump site
        split_fractions = {'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
                           'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
                                                  np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
                           'dumpsite': (np.nan_to_num(row['waste_treatment_open_dump_percent']) +
                                        np.nan_to_num(row['waste_treatment_unaccounted_for_percent']))/100}
        
        # Get the total that goes to landfill and dump site combined
        split_total = sum([x for x in split_fractions.values()])
        
        if split_total == 0:
            # Set to dump site only if no data
            if self.region in defaults.landfill_default_regions:
                split_fractions = {'landfill_w_capture': 0, 'landfill_wo_capture': 1, 'dumpsite': 0}
            else:
                split_fractions = {'landfill_w_capture': 0, 'landfill_wo_capture': 0, 'dumpsite': 1}
        else:
            for site in split_fractions.keys():
                split_fractions[site] /= split_total

        self.split_fractions = split_fractions
        
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

        s = sum(x for x in self.div_fractions.values())
        if  s > 1:
            for div in self.div_fractions:
                self.div_fractions[div] /= s
        assert sum(x for x in self.div_fractions.values()) <= 1, 'Diversion fractions sum to more than 1'
        # Use IPCC defaults if no data
        if s == 0:
            self.div_fractions['compost'] = defaults.fraction_composted[self.region]
            self.div_fractions['combustion'] = defaults.fraction_incinerated[self.region]

        self.div_component_fractions = {}
        self.divs['compost'], self.div_component_fractions['compost'] = self.calc_compost_vol(self.div_fractions['compost'])
        self.divs['anaerobic'], self.div_component_fractions['anaerobic'] = self.calc_anaerobic_vol(self.div_fractions['anaerobic'])
        self.divs['combustion'], self.div_component_fractions['combustion'] = self.calc_combustion_vol(self.div_fractions['combustion'])
        self.divs['recycling'], self.div_component_fractions['recycling'] = self.calc_recycling_vol(self.div_fractions['recycling'])

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
        
        if self.name in city_manual_baselines.manual_cities:
            city_manual_baselines.get_manual_baseline(self)
            self.changed_diversion = True
            self.input_problems = False
            # if self.name == 'Kitakyushu':
            #     self.divs['combustion']['paper_cardboard'] -= 410

            # Inefficiency factors
            self.non_compostable_not_targeted_total = sum([
                self.non_compostable_not_targeted[x] * \
                self.div_component_fractions['compost'][x] for x in self.compost_components
            ])
            
            # Reduce them by non-compostable and unprocessable and etc rates
            for waste in self.compost_components:
                self.divs['compost'][waste] = (
                    self.divs['compost'][waste]  * 
                    (1 - self.non_compostable_not_targeted_total) *
                    (1 - self.unprocessable[waste])
                )
            for waste in self.combustion_components:
                self.divs['combustion'][waste] = (
                    self.divs['combustion'][waste]  * 
                    (1 - self.combustion_reject_rate)
                )
            for waste in self.recycling_components:
                self.divs['recycling'][waste] = (
                    self.divs['recycling'][waste]  * 
                    self.recycling_reject_rates[waste]
                )

        else:
            self.changed_diversion, self.input_problems, self.div_component_fractions, self.divs = self.check_masses(self.div_fractions, self.divs)

        if self.input_problems:
            return

        self.net_masses_after_check = {}
        for waste in self.waste_masses.keys():
            net_mass = self.waste_masses[waste] - (self.divs['compost'][waste] + self.divs['anaerobic'][waste] + self.divs['combustion'][waste] + self.divs['recycling'][waste])
            self.net_masses_after_check[waste] = net_mass

        for waste in self.net_masses_after_check.values():
            if waste < -1:
                print(waste)
            assert waste >= -1, 'Waste diversion is net negative'

        self.new_divs = copy.deepcopy(self.divs)
        self.new_div_fractions = copy.deepcopy(self.div_fractions)
        self.new_div_component_fractions = copy.deepcopy(self.div_component_fractions)
    
    def calc_compost_vol(self, compost_fraction, update=False):
        # Helps to set to 0 at start
        compost_total = 0
        
        # Total mass of compost
        compost_total = compost_fraction * self.waste_mass

        # if update:
        #     compost = {}
        #     for waste in self.compost_components:
        #         compost[waste] = (
        #             compost_total * 
        #             (1 - self.non_compostable_not_targeted_total) *
        #             self.div_component_fractions['compost'][waste] *
        #             (1 - self.unprocessable[waste])
        #         )
        # else:
        # Sum of fraction of waste types that are compostable
        fraction_compostable_types = sum([self.waste_fractions[x] for x in self.compost_components])
        self.unprocessable = {'food': .0192, 'green': .042522, 'wood': .07896, 'paper_cardboard': .12}
        
        if compost_fraction != 0:
            compost_waste_fractions = {x: self.waste_fractions[x] / fraction_compostable_types for x in self.compost_components}
            #non_compostable_not_targeted = .1 # I don't know what this means, basically, waste that gets composted that shouldn't have been and isn't compostable?
            non_compostable_not_targeted = {'food': .1, 'green': .05, 'wood': .05, 'paper_cardboard': .1}
            self.non_compostable_not_targeted_total = sum([non_compostable_not_targeted[x] * \
                                                    compost_waste_fractions[x] for x in self.compost_components])
            compost = {}
            for waste in self.compost_components:
                compost[waste] = (
                    compost_total * 
                    (1 - self.non_compostable_not_targeted_total) *
                    compost_waste_fractions[waste] *
                    (1 - self.unprocessable[waste])
                    )

        else:
            compost = {x: 0 for x in self.compost_components}
            compost_waste_fractions = {x: 0 for x in self.compost_components}
            non_compostable_not_targeted = {'food': 0, 'green': 0, 'wood': 0, 'paper_cardboard': 0}
            self.non_compostable_not_targeted_total = 0
            
        self.compost_total = compost_total
        self.fraction_compostable_types = fraction_compostable_types
        #self.compost_waste_fractions = compost_waste_fractions
        #self.div_component_fractions['compost'] = compost_waste_fractions
        self.non_compostable_not_targeted = non_compostable_not_targeted
        #self.non_compostable_not_targeted_total = non_compostable_not_targeted_total
        
        return compost, compost_waste_fractions
    
    def calc_anaerobic_vol(self, anaerobic_fraction, update=False):
        anaerobic_total = 0
        if update:
            anaerobic = {x: anaerobic_total * self.div_component_fractions['anaerobic'][x] for x in self.anaerobic_components}
        else:
            fraction_anaerobic_types = sum([self.waste_fractions[x] for x in self.anaerobic_components])
            if anaerobic_fraction != 0:
                anaerobic_total = anaerobic_fraction * self.waste_mass
                anaerobic_waste_fractions = {x: self.waste_fractions[x] / fraction_anaerobic_types for x in self.anaerobic_components}
                #self.divs['anaerobic'] = {x: anaerobic_total * anaerobic_waste_fractions[x] for x in self.anaerobic_components}
                anaerobic = {x: anaerobic_total * anaerobic_waste_fractions[x] for x in self.anaerobic_components}
            else:
                #self.divs['anaerobic'] = {x: 0 for x in self.anaerobic_components}
                anaerobic = {x: 0 for x in self.anaerobic_components}
                anaerobic_waste_fractions = {x: 0 for x in self.anaerobic_components}
            
            self.anaerobic_total = anaerobic_total
            #params['fraction_anaerobic_types'] = fraction_anaerobic_types
            #self.anaerobic_waste_fractions = anaerobic_waste_fractions
            #self.div_component_fractions['anaerobic'] = anaerobic_waste_fractions

        return anaerobic, anaerobic_waste_fractions
    
    def calc_combustion_vol(self, combustion_fraction, update=False):
        self.combustion_total = combustion_fraction * self.waste_mass

        if update:
            combustion = {
                x: self.combustion_total * \
                self.div_component_fractions['combustion'][x] * \
                (1 - self.combustion_reject_rate) for x in self.combustion_components
            }
        
        else:
            self.combustion_reject_rate = 0.1 #I think sweet has an error, the rejected from combustion stuff just disappears
            # Remember there's likely a SWEET error here, it just multiplies each waste fraction by combustion fraction, meaning
            # the total doesn't add up to the actual combustion fraction because some waste types are not combustible
            if combustion_fraction != 0:
                # self.divs['combustion'] = {x: self.waste_fractions[x] * \
                #                       combustion_fraction * \
                #                       (1 - combustion_reject_rate) * \
                #                       self.waste_mass for x in self.combustion_components}
                combustion = {
                    x: self.waste_fractions[x] * \
                    combustion_fraction * \
                    (1 - self.combustion_reject_rate) * \
                    self.waste_mass for x in self.combustion_components
                }
            else:
                self.combustion_waste_fractions = {x: 0 for x in self.combustion_components}
                #self.divs['combustion'] = {x: 0 for x in self.combustion_components}
                combustion = {x: 0 for x in self.combustion_components}
                
            self.combustion_reject_rate = self.combustion_reject_rate
            # BELOW IS WRONG, its just how sweet does it.
            fraction_combustion_types = sum([self.waste_fractions[x] for x in self.combustion_components])
            combustion_waste_fractions = {x: self.waste_fractions[x] / fraction_combustion_types for x in self.combustion_components}
            #self.div_component_fractions['combustion'] = self.combustion_waste_fractions

        return combustion, combustion_waste_fractions

    def calc_recycling_vol(self, recycling_fraction, update=False):
        self.recycling_total = recycling_fraction * self.waste_mass

        if update:
            recycling = {
                x: self.recycling_total * \
                self.div_component_fractions['recycling'][x] * \
                recycling_reject_rates[x] for x in self.recycling_components
            }

        else:
            fraction_recyclable_types = sum([self.waste_fractions[x] for x in self.recycling_components])
            recycling_reject_rates = {'wood': .8, 'paper_cardboard': .775, 'textiles': .99, 
                                    'plastic': .875, 'metal': .955, 'glass': .88, 
                                    'rubber': .78, 'other': .87}
            if recycling_fraction != 0:
                recycling_waste_fractions = {x: self.waste_fractions[x] / fraction_recyclable_types for x in self.recycling_components}
                # self.divs['recycling'] = {x: self.waste_fractions[x] / \
                #                   fraction_recyclable_types * \
                #                   recycling_fraction * \
                #                   (recycling_reject_rates[x]) * \
                #                   self.waste_mass for x in self.recycling_components}
                recycling = {
                    x: self.waste_fractions[x] / \
                    fraction_recyclable_types * \
                    recycling_fraction * \
                    (recycling_reject_rates[x]) * \
                    self.waste_mass for x in self.recycling_components
                }
                #recycling_vol_total = sum([recycling_vol[x] for x in recycling_vol.keys()])
            else:
                #self.divs['recycling'] = {x: 0 for x in self.recycling_components}
                recycling = {x: 0 for x in self.recycling_components}
                recycling_waste_fractions = {x: 0 for x in self.recycling_components}
            
            self.fraction_recyclable_types = fraction_recyclable_types
            self.recycling_reject_rates = recycling_reject_rates
            #self.recycling_waste_fractions = recycling_waste_fractions
            #self.div_component_fractions['recycling'] = recycling_waste_fractions
        
        return recycling, recycling_waste_fractions

    def implement_dst_changes(self, new_div_fractions, new_landfill_pct, new_gas_split):
        # Check if new value is valid
        # assert (sum(x for _, x in self.new_div_fractions.items() if x != new_value) + new_value <= 1), \
        #     f'New {diversion_type} value is too large. Total diversion cannot exceed 100%.'
        
        # Set the values
        self.new_div_fractions = new_div_fractions

        # Recalculate the volumes
        self.new_divs['compost'], self.new_div_component_fractions['compost'] = self.calc_compost_vol(self.new_div_fractions['compost'])
        self.new_divs['anaerobic'], self.new_div_component_fractions['anaerobic'] = self.calc_anaerobic_vol(self.new_div_fractions['anaerobic'])
        self.new_divs['combustion'], self.new_div_component_fractions['combustion'] = self.calc_combustion_vol(self.new_div_fractions['combustion'])
        self.new_divs['recycling'], self.new_div_component_fractions['recycling'] = self.calc_recycling_vol(self.new_div_fractions['recycling'])

        # Add zeros for remaining diversion components
        for waste in self.waste_fractions:
            for div, fracs in self.new_divs.items():
                if waste not in fracs:
                    self.new_divs[div][waste] = 0

        self.changed_diversion, self.input_problems, self.new_div_component_fractions, self.new_divs = self.check_masses(self.new_div_fractions, self.new_divs)
        if self.input_problems:
            print(f'Invalid new value')
            return

        net = self.calculate_net_masses(self.new_divs)
        for mass in net.values():
            if mass < 0:
                print(f'Invalid new value')
                return

        # Update the landfill split
        # REMEMBER TO ADJUST THESE FOR 0-100 PERCENTAGE OR 0-1 FRACTION
        # Get the % that is not gas capture
        pct_not_gas = (1 - self.split_fractions['landfill_w_capture'])

        # Set the value
        self.split_fractions_new['landfill_wo_capture'] = new_landfill_pct * pct_not_gas
        self.split_fractions_new['dumpsite'] = (1 - new_landfill_pct) * pct_not_gas

        # Recalculate the volumes
        #landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions_new['landfill_wo_capture'], gas_capture=False)
        dumpsite = Landfill(self, 1960, 2073, 'dumpsite', 0.4, fraction_of_waste=self.split_fractions_new['dumpsite'], gas_capture=False)

        pct_landfill = 1 - self.split_fractions_new['dumpsite']

        self.split_fractions_new['landfill_w_capture'] = new_gas_split * pct_landfill
        self.split_fractions_new['landfill_wo_capture'] = (1 - new_gas_split) * pct_landfill

        assert np.absolute(1 - sum(x for x in self.split_fractions_new.values()) <= .0001)

        landfill_w_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_w_capture'], gas_capture=True)
        landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_wo_capture'], gas_capture=False)

        self.landfills_new = [landfill_w_capture, landfill_wo_capture, dumpsite]

        # Run the model
        for landfill in self.landfills_new:
            landfill.estimate_emissions(baseline=False)

        self.organic_emissions_new = self.estimate_diversion_emissions(baseline=False)
        self.total_emissions_new = self.sum_landfill_emissions(baseline=False)

    def estimate_diversion_emissions(self, baseline=True):
        
        # Define years and t_values.
        # Population and waste data are from 2016. New diversions kick in in 2023.
        years_historic = np.arange(1960, 2016)
        years_middle = np.arange(2016, 2023)
        years_future = np.arange(2023, 2073)
        t_values_historic = years_historic - 2016
        t_values_middle = years_middle - 2016
        t_values_future = years_future - 2016
        
        # Initialize empty DataFrames to hold 'ms' and 'qs' values for each diversion type
        ms_dict = {}
        qs_dict = {}
        
        # Iterate over each diversion type
        for div in self.divs.keys():

            if baseline:
                # Create dataframe with years from div dictionary. All values should be the same, no exponential growth yet
                div_data_historic = pd.DataFrame({waste_type: [value] * len(years_historic) for waste_type, value in self.baseline_divs[div].items()}, index=years_historic)
                div_data_middle = pd.DataFrame({waste_type: [value] * len(years_middle) for waste_type, value in self.baseline_divs[div].items()}, index=years_middle)
                div_data_future = pd.DataFrame({waste_type: [value] * len(years_future) for waste_type, value in self.baseline_divs[div].items()}, index=years_future)
            else:
                # Create dataframe with years from div dictionary. All values should be the same, no exponential growth yet
                div_data_historic = pd.DataFrame({waste_type: [value] * len(years_historic) for waste_type, value in self.baseline_divs[div].items()}, index=years_historic)
                div_data_middle = pd.DataFrame({waste_type: [value] * len(years_middle) for waste_type, value in self.baseline_divs[div].items()}, index=years_middle)
                div_data_future = pd.DataFrame({waste_type: [value] * len(years_future) for waste_type, value in self.new_divs[div].items()}, index=years_future)
            
            # Compute 'ms' values
            #ms = div_data * (1.03 ** (t_values[:, np.newaxis]))
            ms_historic = div_data_historic * (self.growth_rate_historic ** (t_values_historic[:, np.newaxis]))
            ms_middle = div_data_middle * (self.growth_rate_future ** (t_values_middle[:, np.newaxis]))
            ms_future = div_data_future * (self.growth_rate_future ** (t_values_future[:, np.newaxis]))
            ms_dict[div] = pd.concat((ms_historic, ms_middle, ms_future), axis=0)
        
            # Compute 'qs' values based on the diversion type
            if div == 'compost':
                qs = ms_dict[div] * self.mef_compost
            elif div == 'anaerobic':
                qs = ms_dict[div] * defaults.mef_anaerobic * defaults.ch4_to_co2e
            else:
                qs = None
        
            qs_dict[div] = qs
        
        # Store the total organic emissions, only adding compost and anaerobic
        #self.organic_emissions = qs_dict['compost'].add(qs_dict['anaerobic'], fill_value=0)
        return qs_dict['compost'].add(qs_dict['anaerobic'], fill_value=0)

    def sum_landfill_emissions(self, baseline=True):
        if baseline:
            landfills = self.landfills_baseline
            organic_emissions = self.organic_emissions_baseline
        else:
            landfills = self.landfills_new
            organic_emissions = self.organic_emissions_new
            
        landfill_emissions = [x.emissions.applymap(self.convert_methane_m3_to_ton_co2e) for x in landfills]
        landfill_emissions.append(organic_emissions.loc[:, list(self.components)])

        # Concatenate all emissions dataframes
        all_emissions = pd.concat(landfill_emissions, axis=0)
        
        # Group by the year index and sum the emissions for each year
        summed_emissions = all_emissions.groupby(all_emissions.index).sum()
        
        summed_emissions.drop('total', axis=1, inplace=True)
        summed_emissions['total'] = summed_emissions.sum(axis=1)

        return summed_emissions
    
    def check_masses(self, div_fractions, divs):
        div_component_fractions_adjusted = copy.deepcopy(self.div_component_fractions)
        #divs = copy.deepcopy(self.div_component_fractions)
        dont_add_to = set([x for x in self.waste_fractions.keys() if self.waste_fractions[x] == 0])
        
        problems = [set()]
        for waste in self.waste_fractions:
            components = []
            for div in self.divs:
                try:
                    component = div_fractions[div] * self.div_component_fractions[div][waste]
                except:
                    component = 0
                components.append(component)
            s = sum(components)
            #print(div, waste, 'in', self.waste_fractions[waste], 'diverted', s)
            if s > self.waste_fractions[waste]:
                # if div not in problems:
                #     problems[div] = [waste]
                # else:
                #     problems[div].append(waste)
                problems[0].add(waste)

        dont_add_to.update(problems[0])

        if len(problems[0]) == 0:
            #pass
            return False, False, self.div_component_fractions, divs

        removes = {}
        while problems:
            probs = problems.pop(0)
            for waste in probs:
                remove = {}
                distribute = {}
                overflow = {}
                can_be_adjusted = []
                
                div_total = 0
                for div in self.divs.keys():
                    try:
                        component = div_fractions[div] * div_component_fractions_adjusted[div][waste]
                    except:
                        component = 0
                    div_total += component
                div_target = self.waste_fractions[waste]
                diff = div_total - div_target
                #print(waste, div_total, div_target)
                diff = (diff / div_total)
                
                #diff = diffs[waste]
                for div in self.div_component_fractions:
                    if div_fractions[div] == 0:
                        continue
                    distribute[div] = {}
                    try:
                        component = div_component_fractions_adjusted[div][waste]
                    except:
                        continue
                    to_be_removed = diff * component
                    #print(to_be_removed, waste, 'has to be removed from', div)
                    to_distribute_to = [x for x in self.div_components[div] if x not in dont_add_to]
                    to_distribute_to_sum = sum([div_component_fractions_adjusted[div][x] for x in to_distribute_to])
                    if to_distribute_to_sum == 0:
                        overflow[div] = 1
                        continue
                    distributed = 0
                    for w in to_distribute_to:
                        add_amount = to_be_removed * (div_component_fractions_adjusted[div][w] / to_distribute_to_sum )
                        #self.div_component_fractions[div][w] += add_amount
                        if w not in distribute[div]:
                            distribute[div][w] = [add_amount]
                        else:
                            distribute[div][w].append(add_amount)
                        distributed += add_amount
                    #self.div_component_fractions[div][waste] -= 
                    remove[div] = to_be_removed
                    #print('removed', to_be_removed, 'fixing', waste, 'div is', div)
                    removes[waste] = remove
                    can_be_adjusted.append(div)
                    #print(to_be_removed, distributed)
                    
                #for div in overflow:
                    #del distribute[div]
                    #del remove[div]
                    
                for div in overflow:
                    # First, get the amount we were hoping to redistribute away from problem waste component
                    component = div_fractions[div] * div_component_fractions_adjusted[div][waste]
                    to_be_removed = diff * component
                    # Which other diversions can be adjusted instead?
                    to_distribute_to = [x for x in distribute.keys() if waste in self.div_components[x]]
                    to_distribute_to = [x for x in to_distribute_to if x not in overflow]
                    to_distribute_to_sum = sum([div_fractions[x] for x in to_distribute_to])
                    
                    if to_distribute_to_sum == 0:
                        print('aaagh')
                        #print(self.name)
                        return True, True, None, None
                        
                    for d in to_distribute_to:
                        to_be_removed_component = to_be_removed * (div_fractions[d] / to_distribute_to_sum) / div_fractions[d]
                        to_distribute_to_component = [x for x in div_component_fractions_adjusted[d].keys() if x not in dont_add_to]
                        to_distribute_to_sum_component = sum([div_component_fractions_adjusted[d][x] for x in to_distribute_to_component])
                        if to_distribute_to_sum_component == 0:
                            print('grumble')
                            #print(self.name)
                            #to_distribute_to_sum -= self.div_fractions[d]
                            return True, True, None, None
                        #distributed = 0
                        for w in to_distribute_to_component:
                            add_amount = to_be_removed_component * div_component_fractions_adjusted[d][w] / to_distribute_to_sum_component
                            #self.div_component_fractions[div][w] += add_amount
                            if w in distribute[d]:
                                distribute[d][w].append(add_amount)
                            #distributed += add_amount
                        
                    remove[d] += to_be_removed
                    #print('removed', to_be_removed, 'fixing', waste, 'div didnt work is', div, 'going to', d)
            
                for div in distribute:
                    for w in distribute[div]:
                        div_component_fractions_adjusted[div][w] += sum(distribute[div][w])
                
                #print(remove)
                for div in remove:
                    div_component_fractions_adjusted[div][waste] -= remove[div]
                    
                #for div in self.div_component_fractions_adjusted.values():
                    #print(sum(x for x in div.values()))
                    
            if len(probs) > 0: 
                new_probs = set()
                for waste in self.waste_fractions:
                    components = []
                    for div in self.divs:
                        try:
                            component = div_fractions[div] * div_component_fractions_adjusted[div][waste]
                        except:
                            component = 0
                        components.append(component)
                    s = sum(components)
                    #print(waste, s, self.waste_fractions[waste])
                    if s > self.waste_fractions[waste] + 0.001:
                        #print(waste, s, self.waste_fractions[waste])
                        # if div not in problems:
                        #     problems[div] = [waste]
                        # else:
                        #     problems[div].append(waste)
                        new_probs.add(waste)
                    
                if len(new_probs) > 0:
                    problems.append(new_probs)
                dont_add_to.update(new_probs)

        # Calculate diversion amounts with new fractions
        divs = {}
        for div, fracs in div_component_fractions_adjusted.items():
            divs[div] = {}
            s = sum([x for x in fracs.values()])
            # make sure the component fractions add up to 1
            if (s != 0) and (np.absolute(1 - s) > 0.01):
                print(s, 'problems', div)
            for waste in fracs.keys():
                divs[div][waste] = self.waste_mass * div_fractions[div] * div_component_fractions_adjusted[div][waste]

        # Set divs to 0 for components that are not in the diversion
        for c in self.waste_fractions:
            if c not in divs['compost']:
                divs['compost'][c] = 0
            if c not in divs['anaerobic']:
                divs['anaerobic'][c] = 0
            if c not in divs['combustion']:
                divs['combustion'][c] = 0
            if c not in divs['recycling']:
                divs['recycling'][c] = 0

        #net = self.calculate_net_masses(divs)

        # Reduce them by non-compostable and unprocessable and etc rates

        self.non_compostable_not_targeted_total = sum([
            self.non_compostable_not_targeted[x] * \
            div_component_fractions_adjusted['compost'][x] for x in self.compost_components
        ])

        for waste in self.compost_components:
            divs['compost'][waste] = (
                divs['compost'][waste]  * 
                (1 - self.non_compostable_not_targeted_total) *
                (1 - self.unprocessable[waste])
                )
        for waste in self.combustion_components:
            divs['combustion'][waste] = (
                divs['combustion'][waste]  * 
                (1 - self.combustion_reject_rate)
                )
        for waste in self.recycling_components:
            divs['recycling'][waste] = (
                divs['recycling'][waste]  * 
                self.recycling_reject_rates[waste]
                )

        return True, False, div_component_fractions_adjusted, divs
    
    def calculate_net_masses(self, divs):
        net = {}
        for waste in self.waste_fractions:
            net_mass = self.waste_masses[waste] - (divs['compost'][waste] + divs['anaerobic'][waste] + divs['combustion'][waste] + divs['recycling'][waste])
            net[waste] = net_mass

        return net

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

    def export_tables(self, out_path):
        self.total_emissions.to_csv(f'../../data/city_emissions/{self.name}.csv')

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
        
    def estimate_emissions(self, baseline=True):
        if baseline:
            self.model = SWEET(self, self.city, baseline=True)
            # This is due to paper coardboard thing
            #self.waste_mass, self.emissions, self.ch4, self.captured = self.model.estimate_emissions_match_excel()
            self.waste_mass, self.emissions, self.ch4, self.captured = self.model.estimate_emissions()
        else:
            self.model = SWEET(self, self.city, baseline=False)
            self.waste_mass, self.emissions, self.ch4, self.captured = self.model.estimate_emissions()
        