from . import defaults
from . import defaults_2019
#import defaults
import pandas as pd
import numpy as np
from .model import SWEET
#from model import SWEET
import copy
from . import city_manual_baselines
#import city_manual_baselines
from scipy.optimize import curve_fit
#import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

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
        self.year_of_data = db.loc[self.name, 'Year of Data Collection'].values[0]

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
        #self.informal_fraction = db.loc[self.name, 'Informal Waste Collection Rate (%)'].values[0] / 100

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
        
        #self.precip_zone = defaults.get_precipitation_zone(self.precip)
        self.precip_zone = db.loc[self.name, 'Precipitation Zone'].values[0]
    
        # k values
        #self.ks = defaults.k_defaults[self.precip_zone]
        self.ks = {}
        self.ks['food'] = db.loc[self.name, 'k: Food'].values[0]
        self.ks['green'] = db.loc[self.name, 'k: Green'].values[0]
        self.ks['wood'] = db.loc[self.name, 'k: Wood'].values[0]
        self.ks['paper_cardboard'] = db.loc[self.name, 'k: Paper and Cardboard'].values[0]
        self.ks['textiles'] = db.loc[self.name, 'k: Textiles'].values[0]
        
        # Model components
        self.components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles'])
        self.div_components = {}
        self.div_components['compost'] = set(['food', 'green', 'wood', 'paper_cardboard']) # Double check we don't want to include paper
        self.div_components['anaerobic'] = set(['food', 'green', 'wood', 'paper_cardboard'])
        self.div_components['combustion'] = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'])
        self.div_components['recycling'] = set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'])

        self.gas_capture_efficiency = db.loc[self.name, 'Methane Capture Efficiency (%)'].values[0] / 100
        self.landfill_w_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_w_capture'], gas_capture=True)
        self.landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_wo_capture'], gas_capture=False)
        self.dumpsite = Landfill(self, 1960, 2073, 'dumpsite', 0.4, fraction_of_waste=self.split_fractions['dumpsite'], gas_capture=False)
        
        self.landfills = [self.landfill_w_capture, self.landfill_wo_capture, self.dumpsite]
        
        self.mef_compost = db.loc[self.name, 'MEF: Compost'].values[0]

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

        self.divs_loaded = {}
        for div in self.div_fractions:
            self.divs_loaded[div] = {}
        self.divs_loaded['compost']['food'] = db.loc[self.name, 'Diversion Masses: Composted Food (tons)'].values[0]
        self.divs_loaded['compost']['green'] = db.loc[self.name, 'Diversion Masses: Composted Green (tons)'].values[0]
        self.divs_loaded['compost']['wood'] = db.loc[self.name, 'Diversion Masses: Composted Wood (tons)'].values[0]
        self.divs_loaded['compost']['paper_cardboard'] = db.loc[self.name, 'Diversion Masses: Composted Paper and Cardboard (tons)'].values[0]

        self.divs_loaded['anaerobic']['food'] = db.loc[self.name, 'Diversion Masses: Anaerobically Digested Food (tons)'].values[0]
        self.divs_loaded['anaerobic']['green'] = db.loc[self.name, 'Diversion Masses: Anaerobically Digested Green (tons)'].values[0]
        self.divs_loaded['anaerobic']['wood'] = db.loc[self.name, 'Diversion Masses: Anaerobically Digested Wood (tons)'].values[0]
        self.divs_loaded['anaerobic']['paper_cardboard'] = db.loc[self.name, 'Diversion Masses: Anaerobically Digested Paper and Cardboard (tons)'].values[0]

        self.divs_loaded['combustion']['food'] = db.loc[self.name, 'Diversion Masses: Incinerated Food (tons)'].values[0]
        self.divs_loaded['combustion']['green'] = db.loc[self.name, 'Diversion Masses: Incinerated Green (tons)'].values[0]
        self.divs_loaded['combustion']['wood']  = db.loc[self.name, 'Diversion Masses: Incinerated Wood (tons)'].values[0]
        self.divs_loaded['combustion']['paper_cardboard'] = db.loc[self.name, 'Diversion Masses: Incinerated Paper and Cardboard (tons)'].values[0]
        self.divs_loaded['combustion']['plastic'] = db.loc[self.name, 'Diversion Masses: Incinerated Plastic (tons)'].values[0]
        self.divs_loaded['combustion']['rubber'] = db.loc[self.name, 'Diversion Masses: Incinerated Rubber/Leather (tons)'].values[0]
        self.divs_loaded['combustion']['textiles'] = db.loc[self.name, 'Diversion Masses: Incinerated Textiles (tons)'].values[0]

        self.divs_loaded['recycling']['wood'] = db.loc[self.name, 'Diversion Masses: Recycled Wood (tons)'].values[0]
        self.divs_loaded['recycling']['paper_cardboard'] = db.loc[self.name, 'Diversion Masses: Recycled Paper and Cardboard (tons)'].values[0]
        self.divs_loaded['recycling']['plastic'] = db.loc[self.name, 'Diversion Masses: Recycled Plastic (tons)'].values[0]
        self.divs_loaded['recycling']['rubber'] = db.loc[self.name, 'Diversion Masses: Recycled Rubber/Leather (tons)'].values[0]
        self.divs_loaded['recycling']['textiles'] = db.loc[self.name, 'Diversion Masses: Recycled Textiles (tons)'].values[0]
        self.divs_loaded['recycling']['glass'] = db.loc[self.name, 'Diversion Masses: Recycled Glass (tons)'].values[0]
        self.divs_loaded['recycling']['metal'] = db.loc[self.name, 'Diversion Masses: Recycled Metal (tons)'].values[0]
        self.divs_loaded['recycling']['other'] = db.loc[self.name, 'Diversion Masses: Recycled Other (tons)'].values[0]

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
            self.div_component_fractions['compost'][x] for x in self.div_components['compost']
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
        for waste in self.div_components['compost']:
            self.divs['compost'][waste] = (
                self.divs['compost'][waste]  * 
                (1 - self.non_compostable_not_targeted_total) *
                (1 - self.unprocessable[waste])
            )
        for waste in self.div_components['combustion']:
            self.divs['combustion'][waste] = (
                self.divs['combustion'][waste]  * 
                (1 - self.combustion_reject_rate)
            )
        for waste in self.div_components['recycling']:
            self.divs['recycling'][waste] = (
                self.divs['recycling'][waste]  * 
                self.recycling_reject_rates[waste]
            )

        for name, values in self.divs_loaded.items():
            for waste, value in values.items():
                if np.absolute(value - self.divs[name][waste]) > 0.1:
                    print('urgh')
                assert np.absolute(value - self.divs[name][waste]) < 0.1, 'Loaded diversion masses do not match calculated diversion masses'

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


    def load_andre_params(self, row):
        
        #idx = row[0]
        row = row[1]
        self.data_source = row['Data Source']
        self.country = row['country_original']
        self.iso3 = row['iso3_original']
        self.region = defaults.region_lookup[self.country]
        self.year_of_data = row['Year']
        
        # Population, remove the try except when no duplicates
        self.population = float(row['population'])
        population_1950 = row['Population_1950']
        population_2020 = row['Population_2020']
        population_2035 = row['Population_2035']
        self.growth_rate_historic = ((population_2020 / population_1950) ** (1 / (2020 - 1950)))
        self.growth_rate_future = ((population_2035 / population_2020) ** (1 / (2035 - 2020)))

        # # Define the exponential function
        # def exponential(x, a, b):
        #     return a * np.exp(b * x)

        # # Extract data from row
        # years = np.array([1950, 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030, 2035])
        # populations = np.array([row['Population_1950'], row['Population_1955'], row['Population_1960'], 
        #                         row['Population_1965'], row['Population_1970'], row['Population_1975'], 
        #                         row['Population_1980'], row['Population_1985'], row['Population_1990'], 
        #                         row['Population_1995'], row['Population_2000'], row['Population_2005'], 
        #                         row['Population_2010'], row['Population_2015'], row['Population_2020'], 
        #                         row['Population_2025'], row['Population_2030'], row['Population_2035']])

        # # Split data into historic and future
        # historic_years = years[years <= 2020] - 1950
        # future_years = years[years > 2020] - 2025
        # historic_populations = populations[:len(historic_years)]
        # future_populations = populations[len(historic_years):]

        # # Fit the historic data to the exponential function
        # params_historic, _ = curve_fit(exponential, historic_years, historic_populations)
        # a_historic, b_historic = params_historic

        # # Fit the future data to the exponential function
        # params_future, _ = curve_fit(exponential, future_years, future_populations)
        # a_future, b_future = params_future
        
        # self.historic_growth_rate = 100 * (np.exp(b_historic) - 1)
        # self.future_growth_rate = b_future

        # # Predicting using the curve fit model
        # historic_prediction = exponential(np.array(historic_years), *params_historic)
        # future_prediction = exponential(np.array(future_years), *params_future)

        # #%%
        # # Plotting historic data
        # plt.figure(figsize=(10, 5))

        # plt.subplot(1, 2, 1)
        # plt.scatter(historic_years, historic_populations, color='blue', label='Historic Data')
        # plt.plot(historic_years, historic_prediction, color='red', linestyle='dashed', label='Historic Fit')
        # plt.title('Historic Population Growth')
        # plt.legend()

        # # Plotting future data
        # plt.subplot(1, 2, 2)
        # plt.scatter(future_years, future_populations, color='green', label='Future Data')
        # plt.plot(future_years, future_prediction, color='red', linestyle='dashed', label='Future Fit')
        # plt.title('Future Population Growth')
        # plt.legend()

        # plt.tight_layout()
        # plt.show()

        # lat lon
        self.lat = row['latitude_original']
        self.lon = row['longitude_original']

        # Get waste total
        try:
            self.waste_mass_load = float(row['waste (tonnes per year)']) # unit is tons
            self.waste_per_capita = self.waste_mass_load * 1000 / self.population / 365 #unit is kg/person/day
        except:
            self.waste_mass_load = float(row['waste (tonnes per year)'].replace(',', ''))
            self.waste_per_capita = self.waste_mass_load * 1000 / self.population / 365
        if self.waste_mass_load != self.waste_mass_load:
            # Use per capita default
            if self.iso3 in defaults_2019.msw_per_capita_country:
                self.waste_per_capita = defaults_2019.msw_per_capita_country[self.iso3]
            else:
                self.waste_per_capita = defaults.msw_per_capita_defaults[self.region]
            self.waste_mass_load = self.waste_per_capita * self.population / 1000 * 365
        
        # Subtract mass that is informally collected
        #self.informal_fraction = np.nan_to_num(row['percent_informal_sector_percent_collected_by_informal_sector_percent']) / 100
        #self.waste_mass = self.waste_mass_load * (1 - self.informal_fraction)
        self.waste_mass = self.waste_mass_load
        
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
            if self.iso3 in defaults_2019.waste_fractions_country:
                waste_fractions = defaults_2019.waste_fractions_country.loc[self.iso3, :]
            else:
                waste_fractions = defaults.waste_fraction_defaults.loc[self.region, :]
        else:
            waste_fractions.fillna(0, inplace=True)
            waste_fractions['textiles'] = 0
        
        if (waste_fractions.sum() < .98) or (waste_fractions.sum() > 1.02):
            #print('waste fractions do not sum to 1')
            if self.iso3 in defaults_2019.waste_fractions_country:
                waste_fractions = defaults_2019.waste_fractions_country.loc[self.iso3, :]
            else:
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
        
        # Precipitation
        self.precip = row['total_precipitation(mm)_1970-2000']
        self.precip_zone = defaults.get_precipitation_zone(self.precip)
    
        # depth
        #depth = 10
    
        # k values
        self.ks = defaults.k_defaults[self.precip_zone]
        
        # Model components
        self.components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles'])
        
        # Compost params
        self.compost_components = set(['food', 'green', 'wood', 'paper_cardboard']) # Double check we don't want to include paper
        self.compost_fraction = float(row['waste_treatment_compost_percent']) / 100
        
        # Anaerobic digestion params
        self.anaerobic_components = set(['food', 'green', 'wood', 'paper_cardboard'])
        self.anaerobic_fraction = float(row['waste_treatment_anaerobic_digestion_percent']) / 100   
        
        # Combustion params
        self.combustion_components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'])
        value1 = float(row['waste_treatment_incineration_percent'])
        value2 = float(row['waste_treatment_advanced_thermal_treatment_percent'])
        if np.isnan(value1) and np.isnan(value2):
            self.combustion_fraction = np.nan
        else:
            self.combustion_fraction = (np.nan_to_num(value1) + np.nan_to_num(value2)) / 100
        
        # Recycling params
        self.recycling_components = set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'])
        self.recycling_fraction = float(row['waste_treatment_recycling_percent']) / 100
        
        self.gas_capture_percent = np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent']) / 100
        
        self.div_components = {}
        self.div_components['compost'] = self.compost_components
        self.div_components['anaerobic'] = self.anaerobic_components
        self.div_components['combustion'] = self.combustion_components
        self.div_components['recycling'] = self.recycling_components

        # Determine if we need to use defaults. 
        # First case to check: all diversions and landfills are 0. Use defaults.
        landfill_inputs = [
            float(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent']),
            float(row['waste_treatment_controlled_landfill_percent']),
            float(row['waste_treatment_landfill_unspecified_percent']),
            float(row['waste_treatment_open_dump_percent'])
        ]
        all_nan_fill = all(np.isnan(value) for value in landfill_inputs)
        total_fill = sum(0 if np.isnan(x) else x for x in landfill_inputs) / 100
        diversions = [self.compost_fraction, self.anaerobic_fraction, self.combustion_fraction, self.recycling_fraction]
        all_nan_div = all(np.isnan(value) for value in diversions)

        if all_nan_fill and all_nan_div:
            if self.iso3 in defaults_2019.fraction_composted_country:
                self.compost_fraction = defaults_2019.fraction_composted_country[self.iso3]
            elif self.region in defaults_2019.fraction_composted:
                self.compost_fraction = defaults_2019.fraction_composted[self.region]
            else:
                self.compost_fraction = 0.0

            if self.iso3 in defaults_2019.fraction_incinerated_country:
                self.combustion_fraction = defaults_2019.fraction_incinerated_country[self.iso3]
            elif self.region in defaults_2019.fraction_incinerated_country:
                self.combustion_fraction = defaults_2019.fraction_incinerated[self.region]
            else:
                self.combustion_fraction = 0.0

            if self.iso3 in ['CAN', 'CHE', 'DEU']:
                self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 1.0, 'dumpsite': 0.0}
            else:
                if self.iso3 in defaults_2019.fraction_open_dumped_country:
                    self.split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled_country[self.iso3],
                        'dumpsite': defaults_2019.fraction_open_dumped_country[self.iso3]
                    }
                elif self.region in defaults_2019.fraction_open_dumped:
                    self.split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled[self.region],
                        'dumpsite': defaults_2019.fraction_open_dumped[self.region]
                    }
                else:
                    if self.region in defaults.landfill_default_regions:
                        self.split_fractions = {'landfill_w_capture': 0, 'landfill_wo_capture': 1, 'dumpsite': 0}
                    else:
                        self.split_fractions = {'landfill_w_capture': 0, 'landfill_wo_capture': 0, 'dumpsite': 1}

        # Second case to check: all diversions are nan, but landfills are not. Use defaults for diversions if landfills sum to less than 1
        # This assumes that entered data is incomplete. Also, normalize landfills to sum to 1.
        # Caveat: if landfills sum to 1, assume diversions are supposed to be 0. 
        elif all_nan_div and total_fill > .99:
            self.split_fractions = {
                'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
                'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
                                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
                'dumpsite': np.nan_to_num(row['waste_treatment_open_dump_percent'])/100
            }
        elif all_nan_div and total_fill < .99:
            if self.iso3 in defaults_2019.fraction_composted_country:
                self.compost_fraction = defaults_2019.fraction_composted_country[self.iso3]
            elif self.region in defaults_2019.fraction_composted:
                self.compost_fraction = defaults_2019.fraction_composted[self.region]
            else:
                self.compost_fraction = 0.0

            if self.iso3 in defaults_2019.fraction_incinerated_country:
                self.combustion_fraction = defaults_2019.fraction_incinerated_country[self.iso3]
            elif self.region in defaults_2019.fraction_incinerated_country:
                self.combustion_fraction = defaults_2019.fraction_incinerated[self.region]
            else:
                self.combustion_fraction = 0.0

            self.split_fractions = {
                'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
                'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
                                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
                'dumpsite': np.nan_to_num(row['waste_treatment_open_dump_percent'])/100
            }

        # Third case to check: all landfills are nan, but diversions are not. Use defaults for landfills
        elif all_nan_fill:
            if self.iso3 in ['CAN', 'CHE', 'DEU']:
                self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 1.0, 'dumpsite': 0.0}
            else:
                if self.iso3 in defaults_2019.fraction_open_dumped_country:
                    self.split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled_country[self.iso3],
                        'dumpsite': defaults_2019.fraction_open_dumped_country[self.iso3]
                    }
                elif self.region in defaults_2019.fraction_open_dumped:
                    self.split_fractions = {
                        'landfill_w_capture': 0.0,
                        'landfill_wo_capture': defaults_2019.fraction_landfilled[self.region],
                        'dumpsite': defaults_2019.fraction_open_dumped[self.region]
                    }
                else:
                    if self.region in defaults.landfill_default_regions:
                        self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 1.0, 'dumpsite': 0.0}
                    else:
                        self.split_fractions = {'landfill_w_capture': 0.0, 'landfill_wo_capture': 0.0, 'dumpsite': 1.0}
        
        # Fourth case to check: imported non-nan values in both landfills and diversions. Use the values. 
        else:
            self.split_fractions = {
                'landfill_w_capture': np.nan_to_num(row['waste_treatment_sanitary_landfill_landfill_gas_system_percent'])/100,
                'landfill_wo_capture': (np.nan_to_num(row['waste_treatment_controlled_landfill_percent']) + 
                                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']))/100,
                'dumpsite': np.nan_to_num(row['waste_treatment_open_dump_percent'])/100
            }
        
        # Normalize landfills to 1
        split_total = sum([x for x in self.split_fractions.values()])
        for site in self.split_fractions.keys():
            self.split_fractions[site] /= split_total
        
        # Replace diversion NaN values with 0
        attrs = ['compost_fraction', 'anaerobic_fraction', 'combustion_fraction', 'recycling_fraction']
        for attr in attrs:
            if np.isnan(getattr(self, attr)):
                setattr(self, attr, 0.0)

        self.landfill_w_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_w_capture'], gas_capture=True)
        self.landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_wo_capture'], gas_capture=False)
        self.dumpsite = Landfill(self, 1960, 2073, 'dumpsite', 0.4, fraction_of_waste=self.split_fractions['dumpsite'], gas_capture=False)
        
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

        if self.data_source == 'UN Habitat':
            #self.changed_diversion, self.input_problems, self.div_component_fractions, self.divs = self.check_masses_un()
            self.div_component_fractions, self.divs = self.determine_component_fractions_un()
            self.waste_masses = {x: self.waste_fractions[x] * self.waste_mass for x in self.waste_fractions.keys()}
            self.changed_diversion, self.input_problems, self.div_component_fractions, self.divs = self.check_masses(self.div_fractions, self.divs)
        else:
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
            print('input problems')
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
        #self.informal_fraction = np.nan_to_num(row['percent_informal_sector_percent_collected_by_informal_sector_percent']) / 100
        #self.waste_mass *= (1 - self.informal_fraction)
        
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
        fraction_compostable_types = sum([self.waste_fractions[x] for x in self.div_components['compost']])
        self.unprocessable = {'food': .0192, 'green': .042522, 'wood': .07896, 'paper_cardboard': .12}
        
        if compost_fraction != 0:
            compost_waste_fractions = {x: self.waste_fractions[x] / fraction_compostable_types for x in self.div_components['compost']}
            #non_compostable_not_targeted = .1 # I don't know what this means, basically, waste that gets composted that shouldn't have been and isn't compostable?
            non_compostable_not_targeted = {'food': .1, 'green': .05, 'wood': .05, 'paper_cardboard': .1}
            self.non_compostable_not_targeted_total = sum([non_compostable_not_targeted[x] * \
                                                    compost_waste_fractions[x] for x in self.div_components['compost']])
            compost = {}
            for waste in self.div_components['compost']:
                compost[waste] = (
                    compost_total * 
                    (1 - self.non_compostable_not_targeted_total) *
                    compost_waste_fractions[waste] *
                    (1 - self.unprocessable[waste])
                    )

        else:
            compost = {x: 0 for x in self.div_components['compost']}
            compost_waste_fractions = {x: 0 for x in self.div_components['compost']}
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
            anaerobic = {x: anaerobic_total * self.div_component_fractions['anaerobic'][x] for x in self.div_components['anaerobic']}
        else:
            fraction_anaerobic_types = sum([self.waste_fractions[x] for x in self.div_components['anaerobic']])
            if anaerobic_fraction != 0:
                anaerobic_total = anaerobic_fraction * self.waste_mass
                anaerobic_waste_fractions = {x: self.waste_fractions[x] / fraction_anaerobic_types for x in self.div_components['anaerobic']}
                #self.divs['anaerobic'] = {x: anaerobic_total * anaerobic_waste_fractions[x] for x in self.anaerobic_components}
                anaerobic = {x: anaerobic_total * anaerobic_waste_fractions[x] for x in self.div_components['anaerobic']}
            else:
                #self.divs['anaerobic'] = {x: 0 for x in self.anaerobic_components}
                anaerobic = {x: 0 for x in self.div_components['anaerobic']}
                anaerobic_waste_fractions = {x: 0 for x in self.div_components['anaerobic']}
            
            self.anaerobic_total = anaerobic_total
            #params['fraction_anaerobic_types'] = fraction_anaerobic_types
            #self.anaerobic_waste_fractions = anaerobic_waste_fractions
            #self.div_component_fractions['anaerobic'] = anaerobic_waste_fractions

        return anaerobic, anaerobic_waste_fractions
    
    def calc_combustion_vol(self, combustion_fraction, update=False):
        self.combustion_total = combustion_fraction * self.waste_mass

        self.combustion_reject_rate = 0.1 #I think sweet has an error, the rejected from combustion stuff just disappears
        # Remember there's likely a SWEET error here, it just multiplies each waste fraction by combustion fraction, meaning
        # the total doesn't add up to the actual combustion fraction because some waste types are not combustible
        # if combustion_fraction != 0:
        #     # self.divs['combustion'] = {x: self.waste_fractions[x] * \
        #     #                       combustion_fraction * \
        #     #                       (1 - combustion_reject_rate) * \
        #     #                       self.waste_mass for x in self.combustion_components}
        #     combustion = {
        #         x: self.waste_fractions[x] * \
        #         combustion_fraction * \
        #         (1 - self.combustion_reject_rate) * \
        #         self.waste_mass for x in self.combustion_components
        #     }
        # else:
        #     self.combustion_waste_fractions = {x: 0 for x in self.combustion_components}
        #     #self.divs['combustion'] = {x: 0 for x in self.combustion_components}
        #     combustion = {x: 0 for x in self.combustion_components}
            
        fraction_combustion_types = sum([self.waste_fractions[x] for x in self.div_components['combustion']])
        combustion_waste_fractions = {x: self.waste_fractions[x] / fraction_combustion_types for x in self.div_components['combustion']}

        combustion = {x:
            self.waste_mass * 
            combustion_fraction * \
            combustion_waste_fractions[x] * \
            (1 - self.combustion_reject_rate) for x in self.div_components['combustion']
        }

        return combustion, combustion_waste_fractions

    def calc_recycling_vol(self, recycling_fraction, update=False):
        self.recycling_total = recycling_fraction * self.waste_mass

        fraction_recyclable_types = sum([self.waste_fractions[x] for x in self.div_components['recycling']])
        recycling_reject_rates = {'wood': .8, 'paper_cardboard': .775, 'textiles': .99, 
                                'plastic': .875, 'metal': .955, 'glass': .88, 
                                'rubber': .78, 'other': .87}
        if recycling_fraction != 0:
            recycling_waste_fractions = {x: self.waste_fractions[x] / fraction_recyclable_types for x in self.div_components['recycling']}
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
                self.waste_mass for x in self.div_components['recycling']
            }
            #recycling_vol_total = sum([recycling_vol[x] for x in recycling_vol.keys()])
        else:
            #self.divs['recycling'] = {x: 0 for x in self.recycling_components}
            recycling = {x: 0 for x in self.div_components['recycling']}
            recycling_waste_fractions = {x: 0 for x in self.div_components['recycling']}
        
        self.fraction_recyclable_types = fraction_recyclable_types
        self.recycling_reject_rates = recycling_reject_rates
        #self.recycling_waste_fractions = recycling_waste_fractions
        #self.div_component_fractions['recycling'] = recycling_waste_fractions
        
        return recycling, recycling_waste_fractions
    

    def estimate_diversion_emissions(self, baseline=True):
        
        # Define years and t_values.
        # Population and waste data are from 2016. New diversions kick in in 2023.
        years_historic = np.arange(1960, self.year_of_data)
        if baseline:
            years_middle = np.arange(self.year_of_data, 2023)
            years_future = np.arange(2023, 2073)
        else:
            years_middle = np.arange(self.year_of_data, self.dst_implement_year)
            years_future = np.arange(self.dst_implement_year, 2073)

        t_values_historic = years_historic - self.year_of_data
        t_values_middle = years_middle - self.year_of_data
        t_values_future = years_future - self.year_of_data
        
        # Initialize empty DataFrames to hold 'ms' and 'qs' values for each diversion type
        ms_dict = {}
        qs_dict = {}
        
        # Iterate over each diversion type
        for div in ['compost', 'anaerobic']:

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
            organic_emissions = self.organic_emissions_baseline
        else:
            organic_emissions = self.organic_emissions_new

            # for base, new in zip(self.landfills_baseline, self.landfills_new):
            #     combined = base.emissions.copy()
            #     combined.update(new.emissions.copy())
            #     landfill_emissions.append(combined.applymap(self.convert_methane_m3_to_ton_co2e))
        
        landfill_emissions = [x.emissions.map(self.convert_methane_m3_to_ton_co2e) for x in self.landfills]
        landfill_emissions.append(organic_emissions.loc[:, list(self.components)])

        # Concatenate all emissions dataframes
        all_emissions = pd.concat(landfill_emissions, axis=0)
        
        # Group by the year index and sum the emissions for each year
        summed_emissions = all_emissions.groupby(all_emissions.index).sum()
        
        summed_emissions.drop('total', axis=1, inplace=True)
        summed_emissions['total'] = summed_emissions.sum(axis=1)

        summed_emissions /= 28

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
    

    def determine_component_fractions_un(self):

        filepath_un = '../data/data_overview_2022.xlsx'

        un_data_overview = pd.read_excel(filepath_un, sheet_name='Data overview', header=1).loc[:, 'Country':].T
        un_data_overview.columns = un_data_overview.iloc[0, :]
        un_data_overview = un_data_overview.iloc[1:-4, :]

        row = un_data_overview.loc[un_data_overview['City'] == self.name, :]

        divs = {}

        # This just gets filled with zeros
        divs['anaerobic'] = {}
        
        # Compost 
        divs['compost'] = {}
        if self.div_fractions['compost'] < 0.001:
            self.div_fractions['compost'] = 0
        compost_total = self.div_fractions['compost'] * self.waste_mass
        if self.div_fractions['compost'] != 0:
            divs['compost']['food'] = compost_total * self.waste_fractions['food'] / sum([self.waste_fractions[x] for x in ['food', 'green', 'wood']])
            divs['compost']['green'] = compost_total * self.waste_fractions['green'] / sum([self.waste_fractions[x] for x in ['food', 'green', 'wood']])
            divs['compost']['wood'] = compost_total * self.waste_fractions['wood'] / sum([self.waste_fractions[x] for x in ['food', 'green', 'wood']])
            
        else:
            divs['compost'] = {x: 0 for x in self.div_components['compost']}
        
        assert np.absolute(sum([x for x in divs['compost'].values()]) - self.waste_mass * self.div_fractions['compost']) < 1e-3

        divs['recycling'] = {}
        recycling_total = self.div_fractions['recycling'] * self.waste_mass
        self.fraction_recyclable_types = sum([self.waste_fractions[x] for x in self.div_components['recycling']])
        self.recycling_reject_rates = {'wood': .8, 'paper_cardboard': .775, 'textiles': .99, 
                                'plastic': .875, 'metal': .955, 'glass': .88, 
                                'rubber': .78, 'other': .87}
        recovery_rate = (row['total recovered materials (t/d) with rejects'].values[0] * 365) / self.waste_mass

        if self.div_fractions['recycling'] != 0:
            # glass, metal, and other recovered are given directly, and are not in combustion, so they have to be recycling.
            divs['recycling']['wood'] = 0
            divs['recycling']['glass'] = row['Glass recovered (t/d)'].values[0] * 365 + row['Mixed waste (t/d)'].values[0] * self.waste_fractions['glass'] * 365
            divs['recycling']['metal'] = row['Metal recovered (t/d)'].values[0] * 365 + row['Mixed waste (t/d)'].values[0] * self.waste_fractions['metal'] * 365
            divs['recycling']['other'] = row['Other waste (t/d)'].values[0] * 365  + row['Mixed waste (t/d)'].values[0] * self.waste_fractions['other'] * 365
            divs['recycling']['paper_cardboard'] = \
                (row['Paper or Cardboard (t/d)'].values[0] + row['Mixed waste (t/d)'].values[0] * self.waste_fractions['paper_cardboard']) * \
                (self.div_fractions['recycling'] + self.div_fractions['compost']) / recovery_rate * 365                             
            divs['recycling']['plastic'] = \
                (row['Total Plastic recovered (t/d)'].values[0] + row['Mixed waste (t/d)'].values[0] * self.waste_fractions['plastic']) * \
                (self.div_fractions['recycling'] + self.div_fractions['compost']) / recovery_rate * 365                             
            divs['recycling']['textiles'] = \
                (row['Mixed waste (t/d)'].values[0] * self.waste_fractions['textiles']) * \
                (self.div_fractions['recycling'] + self.div_fractions['compost']) / recovery_rate * 365
            divs['recycling']['rubber'] = \
                (row['Mixed waste (t/d)'].values[0] * self.waste_fractions['rubber']) * \
                (self.div_fractions['recycling'] + self.div_fractions['compost']) / recovery_rate * 365
            
            # This one increases recycling. If there is no organic, some mixed waste isn't used
            if sum([x for x in divs['recycling'].values()]) - recycling_total < -10:
                adds = {}
                diff = recycling_total - sum([x for x in divs['recycling'].values()])
                for w in divs['recycling'].keys():
                    adds[w] = diff * (divs['recycling'][w] / sum(x for x in divs['recycling'].values()))
                    
                for w in divs['recycling'].keys():
                    divs['recycling'][w] += adds[w]
            
            
            # This one is to reduce recycling if it's too much. Happens when too much mixed waste ends up in recycling
            if sum([x for x in divs['recycling'].values()]) - recycling_total > 10:
                excess = sum(divs['recycling'].values()) - recycling_total

                limits = {'wood': 0, 
                        'glass': row['Glass recovered (t/d)'].values[0] * 365, 
                        'metal': row['Metal recovered (t/d)'].values[0] * 365, 
                        'other': row['Other waste (t/d)'].values[0] * 365, 
                        'paper_cardboard': row['Paper or Cardboard (t/d)'].values[0], 
                        'plastic': row['Total Plastic recovered (t/d)'].values[0], 
                        'textiles': 0, 
                        'rubber': 0}
            
                if sum([x for x in limits.values()]) - recycling_total > 10:
                    print('cant fix recycling')
                        
                while excess > 0:
                    total_reducible = sum(divs['recycling'][waste] - limit for waste, limit in limits.items())
                    
                    if total_reducible == 0:  # if no category can be reduced anymore
                        print('cant fix recycling')
                        return True, True, None, None
                    
                    reductions = {waste: self.calculate_reduction(divs['recycling'][waste], limit, excess, total_reducible) 
                                for waste, limit in limits.items()}
            
                    # apply reductions and re-calculate excess
                    for waste, reduction in reductions.items():
                        divs['recycling'][waste] -= reduction
                    
                    excess = sum(divs['recycling'].values()) - recycling_total

                    #assert np.absolute(excess) < 1
        else:
            divs['recycling'] = {x: 0 for x in self.div_components['recycling']}
            #self.recycling_waste_fractions = {x: 0 for x in self.div_components['recycling']}
            
        
        assert np.absolute(sum(x for x in divs['recycling'].values()) - recycling_total) < 10

        #mixed_waste = row['Mixed waste (t/d)']
        if self.div_fractions['combustion'] < 0.01:
            self.div_fractions['combustion'] = 0
            
        divs['combustion'] = {} 
        combustion_total = self.div_fractions['combustion'] * self.waste_mass
        # Subtract the recycling from total
        self.combustion_reject_rate = .1 #.1 I think sweet has an error, the rejected from combustion stuff just disappears
        # Food, green, wood are only in combustion, so the recovered must go here. Split proportionally.
        if self.div_fractions['combustion'] != 0:
            divs['combustion']['food'] = (row['Organic waste recovered (t/d)'].values[0] * 365 * \
                                        (self.waste_fractions['food'] / \
                                        (self.waste_fractions['food'] + self.waste_fractions['green'] + self.waste_fractions['wood'])) + \
                                        row['Mixed waste (t/d)'].values[0] * 365 * self.waste_fractions['food']) * \
                                        self.div_fractions['combustion'] / recovery_rate
            divs['combustion']['green'] = (row['Organic waste recovered (t/d)'].values[0] * 365 * \
                                        (self.waste_fractions['green'] / \
                                        (self.waste_fractions['food'] + self.waste_fractions['green'] + self.waste_fractions['wood'])) + \
                                        row['Mixed waste (t/d)'].values[0] * 365 * self.waste_fractions['green']) * \
                                        self.div_fractions['combustion'] / recovery_rate
            divs['combustion']['wood'] = (row['Organic waste recovered (t/d)'].values[0] * 365 * \
                                        (self.waste_fractions['wood'] / \
                                        (self.waste_fractions['food'] + self.waste_fractions['green'] + self.waste_fractions['wood'])) + \
                                        row['Mixed waste (t/d)'].values[0] * 365 * self.waste_fractions['wood'])* \
                                        self.div_fractions['combustion'] / recovery_rate
            divs['combustion']['paper_cardboard'] = \
                (row['Paper or Cardboard (t/d)'].values[0] + row['Mixed waste (t/d)'].values[0] * self.waste_fractions['paper_cardboard']) * \
                self.div_fractions['combustion'] / recovery_rate * 365 + \
                reductions['paper_cardboard']                             
            divs['combustion']['plastic'] = \
                (row['Total Plastic recovered (t/d)'].values[0] + row['Mixed waste (t/d)'].values[0] * self.waste_fractions['plastic']) * \
                self.div_fractions['combustion'] / recovery_rate * 365 + \
                reductions['plastic']                
            divs['combustion']['textiles'] = \
                (row['Mixed waste (t/d)'].values[0] * self.waste_fractions['textiles']) * \
                (self.div_fractions['combustion']) / recovery_rate * 365 + \
                reductions['textiles']                
            divs['combustion']['rubber'] = \
                (row['Mixed waste (t/d)'].values[0] * self.waste_fractions['rubber']) * \
                (self.div_fractions['combustion']) / recovery_rate * 365 + \
                reductions['textiles']                
                
            if sum([x for x in divs['combustion'].values()]) - combustion_total < -10:
                adds = {}
                diff = combustion_total - sum([x for x in divs['combustion'].values()])
                for w in divs['combustion'].keys():
                    adds[w] = diff * (divs['combustion'][w] / sum(x for x in divs['combustion'].values()))
                    
                for w in divs['combustion'].keys():
                    divs['combustion'][w] += adds[w]
                    
        else:
            divs['combustion'] = {x: 0 for x in self.div_components['combustion']}
        
        assert np.absolute(sum([x for x in divs['combustion'].values()]) - combustion_total) < 10
        
        for c in self.waste_fractions.keys():
            if c not in divs['compost'].keys():
                divs['compost'][c] = 0
            if c not in divs['anaerobic'].keys():
                divs['anaerobic'][c] = 0
            if c not in divs['combustion'].keys():
                divs['combustion'][c] = 0
            if c not in divs['recycling'].keys():
                divs['recycling'][c] = 0

        div_component_fractions = {}
        for div, fraction in self.div_fractions.items():
            div_component_fractions[div] = {}
            if fraction == 0:
                for waste in divs[div]:
                    div_component_fractions[div][waste] = 0
            else:
                for waste in divs[div]:
                    div_component_fractions[div][waste] = divs[div][waste] / (self.waste_mass * fraction)

        assert (sum(x for x in div_component_fractions[div].values()) < 1.01)
        assert (sum(x for x in div_component_fractions[div].values()) > 0.98) or \
            (sum(x for x in div_component_fractions[div].values()) == 0)
        
        # Reduce them by non-compostable and unprocessable and etc rates
        self.unprocessable = {'food': .0192, 'green': .042522, 'wood': .07896, 'paper_cardboard': .12}
        self.non_compostable_not_targeted = {'food': .1, 'green': .05, 'wood': .05, 'paper_cardboard': .1}
        self.non_compostable_not_targeted_total = sum([self.non_compostable_not_targeted[x] * \
                                                div_component_fractions['compost'][x] for x in self.div_components['compost']])
        self.non_compostable_not_targeted_total = sum([
            self.non_compostable_not_targeted[x] * \
            div_component_fractions['compost'][x] for x in self.div_components['compost']
        ])

        for waste in self.div_components['compost']:
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

        return div_component_fractions, divs

        # # Save waste diverions calculated with default assumptions, and then update them if any components are net negative.
        # # self.divs_before_check = copy.deepcopy(self.divs)
        # waste_masses = {x: self.waste_fractions[x] * self.waste_mass for x in self.waste_fractions.keys()}
        
        # problem = False
        # net_masses_before_check = {}
        # for waste in waste_masses.keys():
        #     net_mass = waste_masses[waste] - sum(divs[x][waste] for x in divs.keys())
        #     net_masses_before_check[waste] = net_mass
        #     if net_mass < 0:
        #         #print('i want to go home', net_mass)
        #         problem = True
        
        # if problem:     
        #     div_component_fractions_adjusted = copy.deepcopy(div_component_fractions)
        #     dont_add_to = set([x for x in self.waste_fractions.keys() if self.waste_fractions[x] == 0])
            
        #     problems = [set()]
        #     for waste in self.waste_fractions:
        #         components = []
        #         for div in divs:
        #             try:
        #                 component = self.div_fractions[div] * div_component_fractions[div][waste]
        #             except:
        #                 component = 0
        #             components.append(component)
        #         s = sum(components)
        #         if s > self.waste_fractions[waste]:
        #             problems[0].add(waste)

        #     dont_add_to.update(problems[0])

        #     removes = {}
        #     while problems:
        #         probs = problems.pop(0)
        #         for waste in probs:
        #             remove = {}
        #             distribute = {}
        #             overflow = {}
        #             can_be_adjusted = []
                    
        #             div_total = 0
        #             for div in divs.keys():
        #                 try:
        #                     component = self.div_fractions[div] * div_component_fractions_adjusted[div][waste]
        #                 except:
        #                     component = 0
        #                 div_total += component
        #             div_target = self.waste_fractions[waste]
        #             diff = div_total - div_target
        #             diff = (diff / div_total)

        #             for div in div_component_fractions:
        #                 if self.div_fractions[div] == 0:
        #                     continue
        #                 distribute[div] = {}
        #                 try:
        #                     component = div_component_fractions_adjusted[div][waste]
        #                 except:
        #                     continue
        #                 to_be_removed = diff * component
        #                 #print(to_be_removed, waste, 'has to be removed from', div)
        #                 to_distribute_to = [x for x in self.div_components[div] if x not in dont_add_to]
        #                 to_distribute_to_sum = sum([div_component_fractions_adjusted[div][x] for x in to_distribute_to])
        #                 if to_distribute_to_sum == 0:
        #                     overflow[div] = 1
        #                     continue
        #                 distributed = 0
        #                 for w in to_distribute_to:
        #                     add_amount = to_be_removed * (div_component_fractions_adjusted[div][w] / to_distribute_to_sum )
        #                     if w not in distribute[div]:
        #                         distribute[div][w] = [add_amount]
        #                     else:
        #                         distribute[div][w].append(add_amount)
        #                     distributed += add_amount
        #                 remove[div] = to_be_removed
        #                 removes[waste] = remove
        #                 can_be_adjusted.append(div)

                        
        #             for div in overflow:
        #                 # First, get the amount we were hoping to redistribute away from problem waste component
        #                 component = self.div_fractions[div] * div_component_fractions_adjusted[div][waste]
        #                 to_be_removed = diff * component
        #                 # Which other diversions can be adjusted instead?
        #                 to_distribute_to = [x for x in distribute.keys() if waste in self.div_components[x]]
        #                 to_distribute_to = [x for x in to_distribute_to if x not in overflow]
        #                 to_distribute_to_sum = sum([self.div_fractions[x] for x in to_distribute_to])
                        
        #                 if to_distribute_to_sum == 0:
        #                     print('aaagh')
        #                     return True, True, None, None
                            
        #                 for d in to_distribute_to:
        #                     to_be_removed = to_be_removed * (self.div_fractions[d] / to_distribute_to_sum) / self.div_fractions[d]
        #                     to_distribute_to = [x for x in div_component_fractions_adjusted[d].keys() if x not in dont_add_to]
        #                     to_distribute_to_sum = sum([div_component_fractions_adjusted[d][x] for x in to_distribute_to])
        #                     if to_distribute_to_sum == 0:
        #                         print('an error')
        #                         return True, True, None, None
        #                     for w in to_distribute_to:
        #                         add_amount = to_be_removed * div_component_fractions_adjusted[d][w] / to_distribute_to_sum
        #                         if w in distribute[d]:
        #                             distribute[d][w].append(add_amount)
                            
        #                     remove[d] += to_be_removed
                
        #             for div in distribute:
        #                 for w in distribute[div]:
        #                     div_component_fractions_adjusted[div][w] += sum(distribute[div][w])
                            
        #             for div in remove:
        #                 div_component_fractions_adjusted[div][waste] -= remove[div]  
                    
        #         if len(probs) > 0: 
        #             new_probs = set()
        #             for waste in self.waste_fractions:
        #                 components = []
        #                 for div in divs:
        #                     try:
        #                         component = self.div_fractions[div] * div_component_fractions_adjusted[div][waste]
        #                     except:
        #                         component = 0
        #                     components.append(component)
        #                 s = sum(components)
        #                 if s > self.waste_fractions[waste] + 0.01:
        #                     new_probs.add(waste)
                        
        #             if len(new_probs) > 0:
        #                 problems.append(new_probs)
        #             dont_add_to.update(new_probs)
        # else:
        #     return False, False, div_component_fractions, divs

        # # for div in div_component_fractions_adjusted.values():
        # #     assert (sum(x for x in div.values()) < 1.01)
        # #     assert (sum(x for x in div.values()) > 0.98) or \
        # #         (sum(x for x in div.values()) == 0)

        # # Calculate diversion amounts with new fractions
        # divs = {}
        # for div, fracs in div_component_fractions_adjusted.items():
        #     divs[div] = {}
        #     s = sum([x for x in fracs.values()])
        #     # make sure the component fractions add up to 1
        #     if (s != 0) and (np.absolute(1 - s) > 0.01):
        #         print(s, 'problems', div)
        #     for waste in fracs.keys():
        #         divs[div][waste] = self.waste_mass * self.div_fractions[div] * div_component_fractions_adjusted[div][waste]

        # # Set divs to 0 for components that are not in the diversion
        # for c in self.waste_fractions:
        #     if c not in divs['compost']:
        #         divs['compost'][c] = 0
        #     if c not in divs['anaerobic']:
        #         divs['anaerobic'][c] = 0
        #     if c not in divs['combustion']:
        #         divs['combustion'][c] = 0
        #     if c not in divs['recycling']:
        #         divs['recycling'][c] = 0

        # #net = self.calculate_net_masses(divs)

        # # Reduce them by non-compostable and unprocessable and etc rates
        # self.unprocessable = {'food': .0192, 'green': .042522, 'wood': .07896, 'paper_cardboard': .12}
        # self.non_compostable_not_targeted = {'food': .1, 'green': .05, 'wood': .05, 'paper_cardboard': .1}
        # self.non_compostable_not_targeted_total = sum([self.non_compostable_not_targeted[x] * \
        #                                         div_component_fractions_adjusted['compost'][x] for x in self.div_components['compost']])
        # self.non_compostable_not_targeted_total = sum([
        #     self.non_compostable_not_targeted[x] * \
        #     div_component_fractions_adjusted['compost'][x] for x in self.div_components['compost']
        # ])

        # for waste in self.div_components['compost']:
        #     divs['compost'][waste] = (
        #         divs['compost'][waste]  * 
        #         (1 - self.non_compostable_not_targeted_total) *
        #         (1 - self.unprocessable[waste])
        #         )
        # for waste in self.combustion_components:
        #     divs['combustion'][waste] = (
        #         divs['combustion'][waste]  * 
        #         (1 - self.combustion_reject_rate)
        #         )
        # for waste in self.recycling_components:
        #     divs['recycling'][waste] = (
        #         divs['recycling'][waste]  * 
        #         self.recycling_reject_rates[waste]
        #         )

        # return True, False, div_component_fractions_adjusted, divs

    def calculate_reduction(self, value, limit, excess, total_reducible):
        reducible = value - limit  # the amount we can reduce this component by
        reduction = min(reducible, excess * (reducible / total_reducible))  # proportional reduction
        return reduction
    
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

    def implement_dst_changes(self, new_div_fractions, new_landfill_pct, new_gas_split, implement_year):
        # Check if new value is valid
        # assert (sum(x for _, x in self.new_div_fractions.items() if x != new_value) + new_value <= 1), \
        #     f'New {diversion_type} value is too large. Total diversion cannot exceed 100%.'
        
        self.dst_implement_year = implement_year

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

        # Set the value
        self.split_fractions_new['dumpsite'] = 1 - new_landfill_pct

        # Recalculate the volumes
        #landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions_new['landfill_wo_capture'], gas_capture=False)
        #dumpsite = Landfill(self, 2023, 2073, 'dumpsite', 0.4, fraction_of_waste=self.split_fractions_new['dumpsite'], gas_capture=False)
        self.landfills[2].fraction_of_waste_new = self.split_fractions_new['dumpsite']

        pct_landfill = 1 - self.split_fractions_new['dumpsite']

        self.split_fractions_new['landfill_w_capture'] = new_gas_split * pct_landfill
        self.split_fractions_new['landfill_wo_capture'] = (1 - new_gas_split) * pct_landfill

        print(self.split_fractions_new)

        assert np.absolute(1 - sum(x for x in self.split_fractions_new.values()) <= .0001)

        #landfill_w_capture = Landfill(self, 2023, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_w_capture'], gas_capture=True)
        #landfill_wo_capture = Landfill(self, 2023, 2073, 'landfill', 1, fraction_of_waste=self.split_fractions['landfill_wo_capture'], gas_capture=False)
        self.landfills[0].fraction_of_waste_new = self.split_fractions_new['landfill_w_capture']
        self.landfills[1].fraction_of_waste_new = self.split_fractions_new['landfill_wo_capture']

        #self.landfills_new = [landfill_w_capture, landfill_wo_capture, dumpsite]

        # Run the model
        for landfill in self.landfills:
            landfill.estimate_emissions(baseline=False)

        self.organic_emissions_new = self.estimate_diversion_emissions(baseline=False)
        self.total_emissions_new = self.sum_landfill_emissions(baseline=False)

class Landfill:
    def __init__(self, city, open_date, close_date, site_type, mcf, fraction_of_waste=1, gas_capture=False):
        
        self.city = city
        self.open_date = open_date
        self.close_date = close_date
        self.site_type = site_type
        self.mcf = mcf
        self.fraction_of_waste = fraction_of_waste
        self.fraction_of_waste_new = None
        self.gas_capture = gas_capture
        if self.gas_capture:
            self.gas_capture_efficiency = defaults.gas_capture_efficiency[site_type]
            self.oxidation_factor = defaults.oxidation_factor['with_lfg'][site_type]
        else:
            self.gas_capture_efficiency = 0
            self.oxidation_factor = defaults.oxidation_factor['without_lfg'][site_type]
        
    def estimate_emissions(self, baseline=True):
        #if baseline:
        self.model = SWEET(self, self.city)
        # This is due to paper coardboard thing
        #self.waste_mass, self.emissions, self.ch4, self.captured = self.model.estimate_emissions_match_excel()
        self.waste_mass, self.emissions, self.ch4, self.captured = self.model.estimate_emissions(baseline=baseline)
        # else:
        #     self.model = SWEET(self, self.city, baseline=False)
        #     self.waste_mass, self.emissions, self.ch4, self.captured = self.model.estimate_emissions(baseline=False)
        