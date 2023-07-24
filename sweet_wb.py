from sweet_tools_obj import City
import defaults
import pandas as pd

# from fastapi import FastAPI, Query
# from fastapi.encoders import jsonable_encoder
# from fastapi.exceptions import HTTPException
# from fastapi.responses import JSONResponse
# from starlette.responses import RedirectResponse

filepath_wb = 'city_level_data_0_0.csv'
filepath_rmi = 'Merged Waste Dataset Updated.xlsx'
filepath_un = 'data_overview_2022.xlsx'
# Initiate parameter dictionary
params = {}

# Load parameter file
param_file = pd.read_csv(filepath_wb)
rmi_db = pd.read_excel(filepath_rmi, sheet_name=0)
rmi_db = rmi_db[rmi_db['Data Source'] == 'World Bank']
rmi_db.index = rmi_db['City']
un_data_overview = pd.read_excel(filepath_un, sheet_name='Data overview', header=1).loc[:, 'Country':].T
un_data_overview.columns = un_data_overview.iloc[0, :]
un_data_overview = un_data_overview.iloc[1:-4, :]
un_recovered_materials = pd.read_excel(filepath_un, sheet_name='recovered materials', header=1).T
un_recovered_materials.columns = un_recovered_materials.iloc[1, :]
un_recovered_materials = un_recovered_materials.iloc[2:, :]

cities_to_run = {}
# Loop over rows and store sets of parameters
for row in param_file.iterrows():
    try:
        rmi_db.at[row[1]['city_name'], '1950_Population']
    except:
        continue
    #print(row[1]['city_name'])
    city = City(row[1]['city_name'])
    city.load_wb_params(row, rmi_db)
    cities_to_run[city.name] = city
    
# for row in un_data_overview.iterrows():
#     try:
#         rmi_db.at[row[1]['City'], '1950_Population']
#     except:
#         continue
#     #print(row[1]['city_name'])
#     city = City(row[1]['City'])
#     city.load_un_habitat_params(row, rmi_db, un_recovered_materials)
#     cities_to_run[city.name] = city

diversion_adjusted_cities = []
problem_cities = []
for city_name in cities_to_run.keys():
    
    # Load parameters
    city = cities_to_run[city_name]
    
    for landfill in city.landfills:
        landfill.estimate_emissions()
    
    city.estimate_diversion_emissions()
    city.sum_landfill_emissions()
    
    if city.changed_diversion:
        diversion_adjusted_cities.append(city_name)
    if city.input_problems:
        problem_cities.append(city_name)

print('some stuff happened!')

#%%

for row in un_data_overview.iterrows():


    name = row[1]['City']
    print(name)
    country = row[0].split('.')[0]
    row = row[1]
    
    region = defaults.region_lookup[country]
    #run_params['region'] = sweet_tools.region_lookup[run_params['country']]
    
    # Population
    population = float(row['Population']) # * (1.03 ** (2010 - 2023))
    #population_1950 = rmi_db.at[name, '1950_Population'].iloc[0]
    #population_2020 = rmi_db.at[name, '2020_Population'].iloc[0]
    #population_2035 = rmi_db.at[name, '2035_Population'].iloc[0]
    #growth_rate_historic = ((population_2020 / population_1950) ** (1 / (2020 - 1950)))
    #growth_rate_future = ((population_2035 / population_2020) ** (1 / (2035 - 2020)))
    
    # Get waste total
    waste_mass = float(row['MSW generated (t/d)']) * 365 # unit is tons/year
    waste_per_capita = waste_mass * 1000 / population / 365 # unit is kg/person/day
    
    if waste_mass != waste_mass:
        # Use per capita default
        waste_per_capita = defaults.msw_per_capita_defaults[region]
        waste_mass = waste_per_capita * population / 1000 * 365
    
    # # Collection coverage_stats
    # # Don't use these for now, as it seems like WB already adjusted total msw to account for these. 
    # coverage_by_area = float(row['waste_collection_coverage_total_percent_of_geographic_area_percent_of_geographic_area']) / 100
    # coverage_by_households = float(row['waste_collection_coverage_total_percent_of_households_percent_of_households']) / 100
    # coverage_by_pop = float(row['waste_collection_coverage_total_percent_of_population_percent_of_population']) / 100
    # coverage_by_waste = float(row['waste_collection_coverage_total_percent_of_waste_percent_of_waste']) / 100
    
    # if coverage_by_waste == coverage_by_waste:
    #     self.mass *= 
    
    # Waste fractions
    waste_fractions = {'food': row['Kitchen/canteen (%)'][0], 
                       'green': row['Garden/park (%)'][0],
                       'wood': row['Wood (processed) (%)'][0],
                       'paper_cardboard': row['Paper/cardboard (%)'][0],
                       'textiles': row['Textiles/shoes (%)'][0],
                       'plastic': row['Plastic film (%)'][0] + row['Plastics dense (%)'][0],
                       'metal': row['Metals (%)'][0],
                       'glass': row['Glass (%)'][0],
                       'rubber': 0,
                       'other': row['Special wastes (%)'][0] + row['Composite products (%)'][0] + row['Other (%)'][0]
                      }
    
    waste_fractions = pd.DataFrame.from_dict(waste_fractions, orient='index')
    waste_fractions = waste_fractions.transpose()
    
    # Add zeros where there are no values unless all values are nan
    if waste_fractions.isna().all().all():
        print('this shouldnt happen')
        waste_fractions = defaults.waste_fraction_defaults.loc[region, :]
    else:
        waste_fractions.fillna(0, inplace=True)
    
    if (waste_fractions.sum().sum() < .9) or (waste_fractions.sum().sum() > 1.1):
        #print('waste fractions do not sum to 1')
        waste_fractions = defaults.waste_fraction_defaults.loc[region, :]
    
    waste_fractions = waste_fractions.loc[0,:].to_dict()
    
    try:
        mef_compost = (0.0055 * waste_fractions['food']/(waste_fractions['food'] + waste_fractions['green']) + \
                       0.0139 * waste_fractions['green']/(waste_fractions['food'] + waste_fractions['green'])) * 1.1023 * 0.7 # / 28
                       # Unit is Mg CO2e/Mg of organic waste, wtf, so convert to CH4. Mistake in sweet here
    except:
        mef_compost = 0
    
    # Precipitation, remove this try except when there are no duplicates
    # try:
    #     precip = rmi_db.at[name, 'total_precipitation(mm)_1970-2000'].iloc[0]
    # except:
    #     precip = rmi_db.at[name, 'total_precipitation(mm)_1970-2000']
    # precip_zone = defaults.get_precipitation_zone(precip)
    
    # depth
    #depth = 10
    
    # k values
    #ks = defaults.k_defaults[precip_zone]
    
    # Model components
    components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles'])
    
    # Compost params
    organic = row['% recovered out of recoverable organic waste']
    paper_cardboard = row['% recovered out of recoverable paper/cardboard']
    compost_components = set(['food', 'green', 'wood', 'paper_cardboard']) # Double check we don't want to include paper
    # compost_fraction = (organic * \
    #                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood']) + \
    #                     paper_cardboard * waste_fractions['paper_cardboard'] / \
    #                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'] + waste_fractions['paper_cardboard']))
    compost_fraction = 0
    
    # Anaerobic digestion params
    anaerobic_components = set(['food', 'green', 'wood', 'paper_cardboard'])
    anaerobic_fraction = 0
    
    # Recycling
    recycling_components = set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'])
    recovery_rate = row['city recovery rate']
    recycling_fraction = row['recycling rate (recovered minus WtE)']
    
    
    # Combustion params
    combustion_components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'])
    combustion_fraction = recovery_rate - recycling_fraction
    
    # Determine split between landfill and dump site
    split_fractions = {'landfill_w_capture': combustion_fraction,
                       'landfill_wo_capture': row['% of MSW Collected'] - combustion_fraction - recycling_fraction,
                       'dumpsite': 1 - row['% of MSW Collected']}
    
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
        lwc = split_fractions['landfill_wo_capture'] / \
              (split_fractions['landfill_wo_capture'] + split_fractions['dumpsite']) * \
              (1 - split_fractions['landfill_w_capture'])
        split_fractions['dumpsite'] = split_fractions['dumpsite'] / \
                                      (split_fractions['landfill_wo_capture'] + split_fractions['dumpsite']) * \
                                      (1 - split_fractions['landfill_w_capture'])
    
        split_fractions['landfill_wo_capture'] = lwc
    split_total = sum([split_fractions[x] for x in split_fractions.keys()])

    # landfill_w_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=split_fractions['landfill_w_capture'], gas_capture=True)
    # landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=split_fractions['landfill_wo_capture'], gas_capture=False)
    # dumpsite = Landfill(self, 1960, 2073, 'dumpsite', 0.4, fraction_of_waste=split_fractions['dumpsite'], gas_capture=False)
    
    # self.landfills = [self.landfill_w_capture, self.landfill_wo_capture, self.dumpsite]
    
    divs = {}
    # calc_compost_vol()
    # calc_anaerobic_vol()
    
    #divs = calc_recycling_vol(row, divs)
    #divs = calc_combustion_vol(row, divs)
    
    
    mixed_waste = row['Mixed waste (t/d)']

    divs['recycling'] = {}
    recycling_total = recycling_fraction * waste_mass
    fraction_recyclable_types = sum([waste_fractions[x] for x in recycling_components])
    recycling_reject_rates = {'wood': .8, 'paper_cardboard': .775, 'textiles': .99, 
                              'plastic': .875, 'metal': .955, 'glass': .88, 
                              'rubber': .78, 'other': .87}
    if recycling_fraction != 0:
        # glass, metal, and other recovered are given directly, and are not in combustion, so they have to be recycling. 
        divs['recycling']['glass'] = row['Glass recovered (t/d)'] * 365 + mixed_waste * waste_fractions['glass'] * 365
        divs['recycling']['metal'] = row['Metal recovered (t/d)'] * 365 + mixed_waste * waste_fractions['metal'] * 365
        # We know how much other waste was collected. But their term for other includes rubber and textiles. 
        divs['recycling']['other'] = row['Other waste (t/d)'] * 365 * \
                                     (waste_fractions['other'] / \
                                     (waste_fractions['other'] + waste_fractions['rubber'] + waste_fractions['textiles'])) + \
                                     mixed_waste * waste_fractions['other'] * 365
                                     
                                        
        # Remove these three from the total diverted to recycling.
        remaining_recycling = recycling_total - \
                              divs['recycling']['glass'] / recycling_reject_rates['glass'] - \
                              divs['recycling']['metal'] / recycling_reject_rates['metal'] - \
                              divs['recycling']['other'] / recycling_reject_rates['other']
        # Deal with the remaining types proportionally. 
        sum_remaining_types = sum([waste_fractions[x] for x in recycling_components if x not in ['glass', 'metal', 'other']])
        remaining_split = {x: waste_fractions[x] / sum_remaining_types for x in recycling_components if x not in ['glass', 'metal', 'other']}
        #print(sum([x for x in remaining_split.values()]))
        for key, value in remaining_split.items():
            divs['recycling'][key] = remaining_recycling * value * recycling_reject_rates[key]
    else:
        divs['recycling'] = {x: 0 for x in recycling_components}
        recycling_waste_fractions = {x: 0 for x in recycling_components}
        
    
    #mixed_waste = row['Mixed waste (t/d)']
    divs['combustion'] = {} 
    combustion_total = combustion_fraction * waste_mass
    # Subtract the recycling from total
    combustion_reject_rate = 0 #.1 I think sweet has an error, the rejected from combustion stuff just disappears
    # Food, green, wood are only in combustion, so the recovered must go here. Split proportionally.
    if combustion_fraction != 0:
        divs['combustion']['food'] = row['Organic waste recovered (t/d)'] * 365 * \
                                     (waste_fractions['food'] / \
                                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + \
                                     mixed_waste * waste_fractions['food'] * 365
        divs['combustion']['green'] = row['Organic waste recovered (t/d)'] * 365 * \
                                     (waste_fractions['green'] / \
                                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + \
                                     mixed_waste * waste_fractions['green'] * 365
        divs['combustion']['wood'] = row['Organic waste recovered (t/d)'] * 365 * \
                                     (waste_fractions['wood'] / \
                                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + \
                                     mixed_waste * waste_fractions['wood'] * 365
        divs['combustion']['paper_cardboard'] = row['Paper or Cardboard (t/d)'] * 365 + \
                                                mixed_waste * waste_fractions['paper_cardboard'] * 365 -\
                                                divs['recycling']['paper_cardboard']
        divs['combustion']['plastic'] = row['Total Plastic recovered (t/d)'] * 365 + \
                                        mixed_waste * waste_fractions['plastic'] * 365 -\
                                        divs['recycling']['plastic']
        remaining_combustion = combustion_total - \
                               divs['combustion']['food'] / (1 - combustion_reject_rate) - \
                               divs['combustion']['green'] / (1 - combustion_reject_rate) - \
                               divs['combustion']['wood'] / (1 - combustion_reject_rate) - \
                               divs['combustion']['paper_cardboard'] / (1 - combustion_reject_rate) - \
                               divs['combustion']['plastic'] / (1 - combustion_reject_rate)
        
        sum_remaining_types = sum([waste_fractions[x] for x in combustion_components if x not in ['food', 'green', 'wood', 'paper_cardboard', 'plastic']])
        remaining_split = {x: waste_fractions[x] / sum_remaining_types for x in combustion_components if x not in ['food', 'green', 'wood', 'paper_cardboard', 'plastic']}
        #print(sum([x for x in remaining_split.values()]))
        for key, value in remaining_split.items():
            divs['combustion'][key] = remaining_combustion * value * combustion_reject_rate
        
    else:
        divs['combustion'] = {x: 0 for x in combustion_components}
    
    
    breaker = False
    for div in divs.keys():
        d = divs[div]
        for c in d.keys():
            if divs[div][c] < 0:
                breaker = True
                break
        if breaker:
            break
    if breaker:
        break
    
    for c in waste_fractions.keys():
        # if c not in divs['compost'].keys():
        #     divs['compost'][c] = 0
        # if c not in divs['anaerobic'].keys():
        #     divs['anaerobic'][c] = 0
        if c not in divs['combustion'].keys():
            divs['combustion'][c] = 0
        if c not in divs['recycling'].keys():
            divs['recycling'][c] = 0
    
    # Save waste diverions calculated with default assumptions, and then update them if any components are net negative.
    # self.divs_before_check = copy.deepcopy(self.divs)
    waste_masses = {x: waste_fractions[x] * waste_mass for x in waste_fractions.keys()}
    
    net_masses_before_check = {}
    for waste in waste_masses.keys():
        net_mass = waste_masses[waste] - (divs['combustion'][waste] + divs['recycling'][waste])
        net_masses_before_check[waste] = net_mass
    
    #print(net_masses)
    
    # self.changed_diversion, self.input_problems = self.check_masses()
    
    # self.net_masses_after_check = {}
    # for waste in self.waste_masses.keys():
    #     net_mass = self.waste_masses[waste] - (self.divs['compost'][waste] + self.divs['anaerobic'][waste] + self.divs['combustion'][waste] + self.divs['recycling'][waste])
    #     self.net_masses_after_check[waste] = net_mass

#%%

#def calc_recycling_vol(row, divs):
mixed_waste = row['Mixed waste (t/d)']

divs['recycling'] = {}
recycling_total = recycling_fraction * waste_mass
fraction_recyclable_types = sum([waste_fractions[x] for x in recycling_components])
recycling_reject_rates = {'wood': .8, 'paper_cardboard': .775, 'textiles': .99, 
                          'plastic': .875, 'metal': .955, 'glass': .88, 
                          'rubber': .78, 'other': .87}
if recycling_fraction != 0:
    # glass, metal, and other recovered are given directly, and are not in combustion, so they have to be recycling. 
    divs['recycling']['glass'] = row['Glass recovered (t/d)'] * 365 + mixed_waste * waste_fractions['glass'] * 365
    divs['recycling']['metal'] = row['Metal recovered (t/d)'] * 365 + mixed_waste * waste_fractions['metal'] * 365
    # We know how much other waste was collected. But their term for other includes rubber and textiles. 
    divs['recycling']['other'] = row['Other waste (t/d)'] * 365 * \
                                 (waste_fractions['other'] / \
                                 (waste_fractions['other'] + waste_fractions['rubber'] + waste_fractions['textiles'])) + \
                                 mixed_waste * waste_fractions['other'] * 365
                                 
                                    
    # Remove these three from the total diverted to recycling.
    remaining_recycling = recycling_total - \
                          divs['recycling']['glass'] / recycling_reject_rates['glass'] - \
                          divs['recycling']['metal'] / recycling_reject_rates['metal'] - \
                          divs['recycling']['other'] / recycling_reject_rates['other']
    # Deal with the remaining types proportionally. 
    sum_remaining_types = sum([waste_fractions[x] for x in recycling_components if x not in ['glass', 'metal', 'other']])
    remaining_split = {x: waste_fractions[x] / sum_remaining_types for x in recycling_components if x not in ['glass', 'metal', 'other']}
    #print(sum([x for x in remaining_split.values()]))
    for key, value in remaining_split.items():
        divs['recycling'][key] = remaining_recycling * value * recycling_reject_rates[key]
else:
    divs['recycling'] = {x: 0 for x in recycling_components}
    recycling_waste_fractions = {x: 0 for x in recycling_components}
        
    #return divs
    
#def calc_combustion_vol(row, divs):
    mixed_waste = row['Mixed waste (t/d)']
    divs['combustion'] = {} 
    combustion_total = combustion_fraction * waste_mass
    # Subtract the recycling from total
    combustion_reject_rate = 0 #.1 I think sweet has an error, the rejected from combustion stuff just disappears
    # Food, green, wood are only in combustion, so the recovered must go here. Split proportionally.
    if combustion_fraction != 0:
        divs['combustion']['food'] = row['Organic waste recovered (t/d)'] * 365 * \
                                     (waste_fractions['food'] / \
                                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + \
                                     mixed_waste * waste_fractions['food'] * 365
        divs['combustion']['green'] = row['Organic waste recovered (t/d)'] * 365 * \
                                     (waste_fractions['green'] / \
                                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + \
                                     mixed_waste * waste_fractions['green'] * 365
        divs['combustion']['wood'] = row['Organic waste recovered (t/d)'] * 365 * \
                                     (waste_fractions['wood'] / \
                                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + \
                                     mixed_waste * waste_fractions['wood'] * 365
        divs['combustion']['paper_cardboard'] = row['Paper or Cardboard (t/d)'] * 365 + \
                                                mixed_waste * waste_fractions['paper_cardboard'] * 365 -\
                                                divs['recycling']['paper_cardboard']
        divs['combustion']['plastic'] = row['Total Plastic recovered (t/d)'] * 365 + \
                                        mixed_waste * waste_fractions['plastic'] * 365 -\
                                        divs['recycling']['plastic']
        remaining_combustion = combustion_total - \
                               divs['combustion']['food'] / (1 - combustion_reject_rate) - \
                               divs['combustion']['green'] / (1 - combustion_reject_rate) - \
                               divs['combustion']['wood'] / (1 - combustion_reject_rate) - \
                               divs['combustion']['paper_cardboard'] / (1 - combustion_reject_rate) - \
                               divs['combustion']['plastic'] / (1 - combustion_reject_rate)
        
        sum_remaining_types = sum([waste_fractions[x] for x in combustion_components if x not in ['food', 'green', 'wood', 'paper_cardboard', 'plastic']])
        remaining_split = {x: waste_fractions[x] / sum_remaining_types for x in combustion_components if x not in ['food', 'green', 'wood', 'paper_cardboard', 'plastic']}
        #print(sum([x for x in remaining_split.values()]))
        for key, value in remaining_split.items():
            divs['combustion'][key] = remaining_combustion * value * combustion_reject_rate
        
    else:
        divs['combustion'] = {x: 0 for x in combustion_components}
        
    return divs

#%%

# Canberra has 70% recycling, but 47% of waste is food...
# Seattle has 58%, but only 46% or so of waste is non-food. So, it's just one diversion, can't fix it. 
# Bristol has 50% recycling, 25% combustion...same problem. All recyclables are already net 0, nowhere to put the extra recycling.

#city = cities_to_run['Canberra']
#city = cities_to_run['Bristol']
city = cities_to_run['Kanpur']
#city = cities_to_run['Seattle']



city.divs_before_check
city.divs

city.net_masses_before_check
city.net_masses_after_check

city.waste_mass
city.population
city.compost_fraction
city.anaerobic_fraction
city.combustion_fraction
city.recycling_fraction
city.waste_fractions

masses = {x: city.waste_fractions[x] * city.waste_mass for x in city.waste_fractions.keys()}
net_masses = {}
for waste in city.waste_fractions.keys():
    net_masses[waste] = masses[waste] - (city.divs['compost'][waste] + city.divs['anaerobic'][waste] + city.divs['combustion'][waste] + city.divs['recycling'][waste])

x1 = city.divs_before_check
x2 = city.divs
x1 == x2


#%%


city = cities_to_run['Dubai']
# city = City(name)
# for row in param_file.iterrows():
#     if row[1]['city_name'] != 'Dubai':
#         continue
#     else:
#         city = City(row[1]['city_name'])
#         city.load_wb_params(row, rmi_db)

# city.precip = 130
# city.precip_zone = defaults.get_precipitation_zone(city.precip)
# city.ks = defaults.k_defaults[city.precip_zone]

# for landfill in city.landfills:
#     landfill.estimate_emissions()

# city.estimate_diversion_emissions()
# city.sum_landfill_emissions()

# print(city.waste_per_capita)
# city.waste_mass * (city.growth_rate_historic ** (1960 - 2016)) * city.recycling_fraction
city.landfill_w_capture.fraction_of_waste
city.landfill_wo_capture.fraction_of_waste
city.dumpsite.fraction_of_waste

x = city.landfill_w_capture.waste_mass

city.growth_rate_historic
city.growth_rate_future

x2 = city.total_emissions
x3 = city.landfill_w_capture.ch4
x4 = city.divs['recycling']

city.ks

rmi_db.at['Kabul', 'total_precipitation(mm)_1970-2000'].iloc[0]
