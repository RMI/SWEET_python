#%%
from importlib import reload
import sweet_tools_obj 
sweet_tools_obj = reload(sweet_tools_obj)
from sweet_tools_obj import City
import defaults
import pandas as pd
import numpy as np
import copy
import warnings

# from fastapi import FastAPI, Query
# from fastapi.encoders import jsonable_encoder
# from fastapi.exceptions import HTTPException
# from fastapi.responses import JSONResponse
# from starlette.responses import RedirectResponse

# Convert RuntimeWarning into an error
warnings.filterwarnings('error', category=RuntimeWarning)

pth = '/Users/hugh/Library/CloudStorage/OneDrive-RMI/Documents/RMI/WasteMAP/decision_support_tool/python_only_sweet/'

filepath_wb = pth + 'city_level_data_0_0.csv'
filepath_rmi = pth + 'Merged Waste Dataset Updated.xlsx'
filepath_un = pth + 'data_overview_2022.xlsx'
# Initiate parameter dictionary
params = {}

# Load parameter file
param_file = pd.read_csv(filepath_wb)
rmi_db = pd.read_excel(filepath_rmi, sheet_name=0)
rmi_db = rmi_db[rmi_db['Data Source'] == 'World Bank']
rmi_db.index = rmi_db['City_original']
un_data_overview = pd.read_excel(filepath_un, sheet_name='Data overview', header=1).loc[:, 'Country':].T
un_data_overview.columns = un_data_overview.iloc[0, :]
un_data_overview = un_data_overview.iloc[1:-4, :]
un_recovered_materials = pd.read_excel(filepath_un, sheet_name='recovered materials', header=1).T
un_recovered_materials.columns = un_recovered_materials.iloc[1, :]
un_recovered_materials = un_recovered_materials.iloc[2:, :]

#%%

cities_to_run = {}
# Loop over rows and store sets of parameters
#problem_cities = set(['Kanpur', 'Canberra', 'Paris', 'Bristol', 'Naha', 'Toyama', 'Oslo', 'Stockholm', 'Seattle', 'Liege', 'Tadipatri', 'Itanagar', 'Navi Mumbai', 'Milano'])
problem_cities = []
adjusted_cities = []
for row in param_file.iterrows():
    try:
        rmi_db.at[row[1]['city_name'], 'Population_1950']
    except:
        continue
    # if row[1]['city_name'] in problem_cities:
    #     continue
    city = City(row[1]['city_name'])
    print(city.name)
    city.load_wb_params(row, rmi_db)
    
    cities_to_run[city.name] = city
    if city.input_problems:
        problem_cities.append(city.name)
    elif city.changed_diversion:
        adjusted_cities.append(city.name)

for city_name in cities_to_run.keys():
    
    # Load parameters
    city = cities_to_run[city_name]
    
    for landfill in city.landfills:
        landfill.estimate_emissions()
    
    city.estimate_diversion_emissions()
    city.sum_landfill_emissions()

#%%

for city_name, city in cities_to_run.items():
    #print(city_name, city.total_emissions.isna().sum().sum())
    city.total_emissions.to_csv(f'../../data/city_emissions/{city_name}.csv')

print('some stuff happened!')

#%%

for row in un_data_overview.iterrows():
    
    breaker = False

    name = row[1]['City']
    if name in ['Hatyai']:
        continue

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
    #organic = row['% recovered out of recoverable organic waste']
    #paper_cardboard = row['% recovered out of recoverable paper/cardboard']
    compost_components = set(['food', 'green', 'wood', 'paper_cardboard']) # Double check we don't want to include paper
    # compost_fraction = (organic * \
    #                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood']) + \
    #                     paper_cardboard * waste_fractions['paper_cardboard'] / \
    #                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'] + waste_fractions['paper_cardboard']))
    #compost_fraction = (row['Organic waste recovered (t/d)'] * 365) / waste_mass
    
    # Anaerobic digestion params
    anaerobic_components = set(['food', 'green', 'wood', 'paper_cardboard'])
    anaerobic_fraction = 0
    
    # Recycling
    recycling_components = set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'])
    recovery_rate = (row['total recovered materials (t/d) with rejects'] * 365) / waste_mass
    recycling_rate = row['recycling rate (recovered minus WtE)'] # - compost_fraction
    if np.absolute(recovery_rate - recycling_rate) < 0.01:
        recycling_rate = recovery_rate
    compost_total = \
        (row['Organic waste recovered (t/d)'] + \
        row['Mixed waste (t/d)'] * \
        (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) * \
        (recycling_rate / recovery_rate) * 365
    compost_fraction = compost_total / waste_mass
    recycling_fraction = recycling_rate - compost_fraction
    
    # Combustion params
    combustion_components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'])
    combustion_fraction = recovery_rate - recycling_fraction - compost_fraction
    
    
    if (combustion_fraction < 0) and (combustion_fraction > -0.05):
        combustion_fraction = 0
        
    if (combustion_fraction < 0):
        raise ValueError("Combustion is negative, throw computer out window")
        
    # Determine split between landfill and dump site
    split_fractions = {'landfill_w_capture': 0,
                       'landfill_wo_capture': row['% of MSW received by disposal facilities '] * row['% of MSW disposed in controlled disposal facilities '],
                       'dumpsite': 1 - recovery_rate - row['% of MSW received by disposal facilities '] * row['% of MSW disposed in controlled disposal facilities ']}
    
    # Check if dumpsite equation produced a negative
    if name == 'Hoi An':
        t = recovery_rate + row['% of MSW received by disposal facilities '] * row['% of MSW disposed in controlled disposal facilities ']
        
        split_fractions['landfill_wo_capture'] = 0
        split_fractions['dumpsite'] = 0
        
    elif split_fractions['dumpsite'] < -1e-5:
        raise ValueError("Dumpsite equation didn't work")
    
    # Get the total that goes to landfill and dump site combined
    split_total = sum([split_fractions[x] for x in split_fractions.keys()])
    
    if split_total == 0:
        # Set to dump site only if no data
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

    # Check if the sum of the fractions is close to 1
    if not abs(sum(split_fractions.values()) - 1) < 1e-5:
        raise ValueError("The landfill splits do not add up to 1.")

    # landfill_w_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=split_fractions['landfill_w_capture'], gas_capture=True)
    # landfill_wo_capture = Landfill(self, 1960, 2073, 'landfill', 1, fraction_of_waste=split_fractions['landfill_wo_capture'], gas_capture=False)
    # dumpsite = Landfill(self, 1960, 2073, 'dumpsite', 0.4, fraction_of_waste=split_fractions['dumpsite'], gas_capture=False)
    
    # self.landfills = [self.landfill_w_capture, self.landfill_wo_capture, self.dumpsite]
    
    # Simple checks
    assert waste_mass * sum(waste_fractions[x] for x in ['food', 'green', 'wood']) > row['Organic waste recovered (t/d)']
    assert waste_mass * waste_fractions['paper_cardboard'] >row['Paper or Cardboard (t/d)']
    assert waste_mass * waste_fractions['plastic'] > row['Total Plastic recovered (t/d)']
    assert waste_mass * waste_fractions['glass'] > row['Glass recovered (t/d)']
    assert waste_mass * waste_fractions['metal'] > row['Metal recovered (t/d)']
    assert waste_mass * waste_fractions['other'] > row['Other waste (t/d)']
    
    divs = {}
    # calc_compost_vol()
    # calc_anaerobic_vol()
    
    #divs = calc_recycling_vol(row, divs)
    #divs = calc_combustion_vol(row, divs)
    
    # This just gets filled with zeros
    divs['anaerobic'] = {}
    
    # Compost 
    divs['compost'] = {}
    
    if compost_fraction != 0:
        # Compost here is pulled out of recycling. So, all "recycled" organic waste gets composted, the rest combusted. 
        # divs['compost']['food'] = row['Organic waste recovered (t/d)'] * 365 * \
        #                              (waste_fractions['food'] / \
        #                              (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + \
                                       # add other back here maybe, other two have it
        #                              (compost_fraction + recycling_fraction) / recovery_rate 
        # divs['compost']['green'] = row['Organic waste recovered (t/d)'] * 365 * \
        #                              (waste_fractions['green'] / \
        #                              (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + \
        #                              row['Mixed waste (t/d)'] * 365 * waste_fractions['green']) * \
        #                              (compost_fraction + recycling_fraction) / recovery_rate
        # divs['compost']['wood'] = row['Organic waste recovered (t/d)'] * 365 * \
        #                              (waste_fractions['wood'] / \
        #                              (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + 
        #                              row['Mixed waste (t/d)'] * 365 * waste_fractions['wood']) * \
        #                              (compost_fraction + recycling_fraction) / recovery_rate
        
        divs['compost']['food'] = compost_total * waste_fractions['food'] / sum([waste_fractions[x] for x in ['food', 'green', 'wood']])
        divs['compost']['green'] = compost_total * waste_fractions['green'] / sum([waste_fractions[x] for x in ['food', 'green', 'wood']])
        divs['compost']['wood'] = compost_total * waste_fractions['wood'] / sum([waste_fractions[x] for x in ['food', 'green', 'wood']])
        
    else:
        divs['compost'] = {x: 0 for x in compost_components}
    
    assert np.absolute(sum([x for x in divs['compost'].values()]) - waste_mass * compost_fraction) < 1e-3

    divs['recycling'] = {}
    recycling_total = recycling_fraction * waste_mass
    fraction_recyclable_types = sum([waste_fractions[x] for x in recycling_components])
    recycling_reject_rates = {'wood': .8, 'paper_cardboard': .775, 'textiles': .99, 
                              'plastic': .875, 'metal': .955, 'glass': .88, 
                              'rubber': .78, 'other': .87}
    if recycling_fraction != 0:
        # glass, metal, and other recovered are given directly, and are not in combustion, so they have to be recycling.
        divs['recycling']['wood'] = 0
        divs['recycling']['glass'] = row['Glass recovered (t/d)'] * 365 + row['Mixed waste (t/d)'] * waste_fractions['glass'] * 365
        divs['recycling']['metal'] = row['Metal recovered (t/d)'] * 365 + row['Mixed waste (t/d)'] * waste_fractions['metal'] * 365
        divs['recycling']['other'] = row['Other waste (t/d)'] * 365  + row['Mixed waste (t/d)'] * waste_fractions['other'] * 365
        divs['recycling']['paper_cardboard'] = \
            (row['Paper or Cardboard (t/d)'] + row['Mixed waste (t/d)'] * waste_fractions['paper_cardboard']) * \
            (recycling_fraction + compost_fraction) / recovery_rate * 365                             
        divs['recycling']['plastic'] = \
            (row['Total Plastic recovered (t/d)'] + row['Mixed waste (t/d)'] * waste_fractions['plastic']) * \
            (recycling_fraction + compost_fraction) / recovery_rate * 365                             
        divs['recycling']['textiles'] = \
            (row['Mixed waste (t/d)'] * waste_fractions['textiles']) * \
            (recycling_fraction + compost_fraction) / recovery_rate * 365
        divs['recycling']['rubber'] = \
            (row['Mixed waste (t/d)'] * waste_fractions['rubber']) * \
            (recycling_fraction + compost_fraction) / recovery_rate * 365
        
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
                      'glass': row['Glass recovered (t/d)'] * 365, 
                      'metal': row['Metal recovered (t/d)'] * 365, 
                      'other': row['Other waste (t/d)'] * 365, 
                      'paper_cardboard': row['Paper or Cardboard (t/d)'], 
                      'plastic': row['Total Plastic recovered (t/d)'], 
                      'textiles': 0, 
                      'rubber': 0}
        
            if sum([x for x in limits.values()]) - recycling_total > 10:
                print('cant fix recycling')
                      
            while excess > 0:
                total_reducible = sum(divs['recycling'][waste] - limit for waste, limit in limits.items())
                
                if total_reducible == 0:  # if no category can be reduced anymore
                    print('cant fix recycling')
                    break
                
                reductions = {waste: calculate_reduction(divs['recycling'][waste], limit, excess, total_reducible) 
                              for waste, limit in limits.items()}
        
                # apply reductions and re-calculate excess
                for waste, reduction in reductions.items():
                    divs['recycling'][waste] -= reduction
                
                excess = sum(divs['recycling'].values()) - recycling_total

                #assert np.absolute(excess) < 1
    else:
        divs['recycling'] = {x: 0 for x in recycling_components}
        recycling_waste_fractions = {x: 0 for x in recycling_components}
        
    
    assert np.absolute(sum(x for x in divs['recycling'].values()) - recycling_total) < 10

    #mixed_waste = row['Mixed waste (t/d)']
    if combustion_fraction < 0.01:
        combustion_fraction = 0
        
    divs['combustion'] = {} 
    combustion_total = combustion_fraction * waste_mass
    # Subtract the recycling from total
    combustion_reject_rate = 0 #.1 I think sweet has an error, the rejected from combustion stuff just disappears
    # Food, green, wood are only in combustion, so the recovered must go here. Split proportionally.
    if combustion_fraction != 0:
        divs['combustion']['food'] = (row['Organic waste recovered (t/d)'] * 365 * \
                                     (waste_fractions['food'] / \
                                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + \
                                     row['Mixed waste (t/d)'] * 365 * waste_fractions['food']) * \
                                     combustion_fraction / recovery_rate
        divs['combustion']['green'] = (row['Organic waste recovered (t/d)'] * 365 * \
                                     (waste_fractions['green'] / \
                                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + \
                                     row['Mixed waste (t/d)'] * 365 * waste_fractions['green']) * \
                                     combustion_fraction / recovery_rate
        divs['combustion']['wood'] = (row['Organic waste recovered (t/d)'] * 365 * \
                                     (waste_fractions['wood'] / \
                                     (waste_fractions['food'] + waste_fractions['green'] + waste_fractions['wood'])) + \
                                     row['Mixed waste (t/d)'] * 365 * waste_fractions['wood'])* \
                                     combustion_fraction / recovery_rate
        divs['combustion']['paper_cardboard'] = \
            (row['Paper or Cardboard (t/d)'] + row['Mixed waste (t/d)'] * waste_fractions['paper_cardboard']) * \
            combustion_fraction / recovery_rate * 365 + \
            reductions['paper_cardboard']                             
        divs['combustion']['plastic'] = \
            (row['Total Plastic recovered (t/d)'] + row['Mixed waste (t/d)'] * waste_fractions['plastic']) * \
            combustion_fraction / recovery_rate * 365 + \
            reductions['plastic']                
        divs['combustion']['textiles'] = \
            (row['Mixed waste (t/d)'] * waste_fractions['textiles']) * \
            (combustion_fraction) / recovery_rate * 365 + \
            reductions['textiles']                
        divs['combustion']['rubber'] = \
            (row['Mixed waste (t/d)'] * waste_fractions['rubber']) * \
            (combustion_fraction) / recovery_rate * 365 + \
            reductions['textiles']                
            
        if sum([x for x in divs['combustion'].values()]) - combustion_total < -10:
            adds = {}
            diff = combustion_total - sum([x for x in divs['combustion'].values()])
            for w in divs['combustion'].keys():
                adds[w] = diff * (divs['combustion'][w] / sum(x for x in divs['combustion'].values()))
                
            for w in divs['combustion'].keys():
                divs['combustion'][w] += adds[w]
                
    else:
        divs['combustion'] = {x: 0 for x in combustion_components}
    
    
    assert np.absolute(sum([x for x in divs['combustion'].values()]) - combustion_total) < 10
    
    for div in divs.keys():
        d = divs[div]
        for c in d.keys():
            if divs[div][c] < 0:
                breaker = True
                break
        if breaker:
            break
    if breaker:
        print(f'{name} has problems, do manually')
        break
    
    for c in waste_fractions.keys():
        if c not in divs['compost'].keys():
            divs['compost'][c] = 0
        if c not in divs['anaerobic'].keys():
            divs['anaerobic'][c] = 0
        if c not in divs['combustion'].keys():
            divs['combustion'][c] = 0
        if c not in divs['recycling'].keys():
            divs['recycling'][c] = 0
    
    # Save waste diverions calculated with default assumptions, and then update them if any components are net negative.
    # self.divs_before_check = copy.deepcopy(self.divs)
    waste_masses = {x: waste_fractions[x] * waste_mass for x in waste_fractions.keys()}
    
    problem = False
    net_masses_before_check = {}
    for waste in waste_masses.keys():
        net_mass = waste_masses[waste] - sum(divs[x][waste] for x in divs.keys())
        net_masses_before_check[waste] = net_mass
        if net_mass < 0:
            print('i want to go home', net_mass)
            problem = True
    
    if problem:
        
        div_fractions = {
            'compost': compost_fraction, 
            'anaerobic':anaerobic_fraction, 
            'combustion': combustion_fraction,
            'recycling': recycling_fraction
            }
        
        div_components = {
            'compost': compost_components,
            'anaerobic': anaerobic_components,
            'combustion': combustion_components,
            'recycling': recycling_components}
        
        div_component_fractions = {}
        for div, fraction in div_fractions.items():
            div_component_fractions[div] = {}
            if fraction == 0:
                for waste in divs[div]:
                    div_component_fractions[div][waste] = 0
            else:
                for waste in divs[div]:
                    div_component_fractions[div][waste] = divs[div][waste] / (waste_mass * fraction)
                    
            # JUST FORCE THEM TO ALL BE 1.00 BY NORMALIZING?
            assert (sum(x for x in div_component_fractions[div].values()) < 1.01)
            assert (sum(x for x in div_component_fractions[div].values()) > 0.98) or \
                   (sum(x for x in div_component_fractions[div].values()) == 0)
        
        div_component_fractions_adjusted = copy.deepcopy(div_component_fractions)
        dont_add_to = set([x for x in waste_fractions.keys() if waste_fractions[x] == 0])
        
        problems = [set()]
        for waste in waste_fractions:
            components = []
            for div in divs:
                try:
                    component = div_fractions[div] * div_component_fractions[div][waste]
                except:
                    component = 0
                components.append(component)
            s = sum(components)
            if s > waste_fractions[waste]:
                problems[0].add(waste)

        dont_add_to.update(problems[0])

        removes = {}
        while problems:
            probs = problems.pop(0)
            for waste in probs:
                remove = {}
                distribute = {}
                overflow = {}
                can_be_adjusted = []
                
                div_total = 0
                for div in divs.keys():
                    try:
                        component = div_fractions[div] * div_component_fractions_adjusted[div][waste]
                    except:
                        component = 0
                    div_total += component
                div_target = waste_fractions[waste]
                diff = div_total - div_target
                diff = (diff / div_total)

                for div in div_component_fractions:
                    if div_fractions[div] == 0:
                        continue
                    distribute[div] = {}
                    try:
                        component = div_component_fractions_adjusted[div][waste]
                    except:
                        continue
                    to_be_removed = diff * component
                    #print(to_be_removed, waste, 'has to be removed from', div)
                    to_distribute_to = [x for x in div_components[div] if x not in dont_add_to]
                    to_distribute_to_sum = sum([div_component_fractions_adjusted[div][x] for x in to_distribute_to])
                    if to_distribute_to_sum == 0:
                        overflow[div] = 1
                        continue
                    distributed = 0
                    for w in to_distribute_to:
                        add_amount = to_be_removed * (div_component_fractions_adjusted[div][w] / to_distribute_to_sum )
                        if w not in distribute[div]:
                            distribute[div][w] = [add_amount]
                        else:
                            distribute[div][w].append(add_amount)
                        distributed += add_amount
                    remove[div] = to_be_removed
                    removes[waste] = remove
                    can_be_adjusted.append(div)

                    
                for div in overflow:
                    # First, get the amount we were hoping to redistribute away from problem waste component
                    component = div_fractions[div] * div_component_fractions_adjusted[div][waste]
                    to_be_removed = diff * component
                    # Which other diversions can be adjusted instead?
                    to_distribute_to = [x for x in distribute.keys() if waste in div_components[x]]
                    to_distribute_to = [x for x in to_distribute_to if x not in overflow]
                    to_distribute_to_sum = sum([div_fractions[x] for x in to_distribute_to])
                    
                    if to_distribute_to_sum == 0:
                        print('aaagh')
                        print(name)
                        
                    for d in to_distribute_to:
                        to_be_removed = to_be_removed * (div_fractions[d] / to_distribute_to_sum) / div_fractions[d]
                        to_distribute_to = [x for x in div_component_fractions_adjusted[d].keys() if x not in dont_add_to]
                        to_distribute_to_sum = sum([div_component_fractions_adjusted[d][x] for x in to_distribute_to])
                        if to_distribute_to_sum == 0:
                            print(name, 'an error')
                            continue
                        for w in to_distribute_to:
                            add_amount = to_be_removed * div_component_fractions_adjusted[d][w] / to_distribute_to_sum
                            if w in distribute[d]:
                                distribute[d][w].append(add_amount)
                        
                        remove[d] += to_be_removed
            
                for div in distribute:
                    for w in distribute[div]:
                        div_component_fractions_adjusted[div][w] += sum(distribute[div][w])
                        
                for div in remove:
                    div_component_fractions_adjusted[div][waste] -= remove[div]  
                   
            if len(probs) > 0: 
                new_probs = set()
                for waste in waste_fractions:
                    components = []
                    for div in divs:
                        try:
                            component = div_fractions[div] * div_component_fractions_adjusted[div][waste]
                        except:
                            component = 0
                        components.append(component)
                    s = sum(components)
                    if s > waste_fractions[waste] + 0.01:
                        new_probs.add(waste)
                    
                if len(new_probs) > 0:
                    problems.append(new_probs)
                dont_add_to.update(new_probs)
                
    for div in div_component_fractions_adjusted.values():
        assert (sum(x for x in div.values()) < 1.01)
        assert (sum(x for x in div.values()) > 0.98) or \
               (sum(x for x in div.values()) == 0)
    
    # if name == 'Thiruvananthapuram':
    #     break
    
def calculate_reduction(value, limit, excess, total_reducible):
    reducible = value - limit  # the amount we can reduce this component by
    reduction = min(reducible, excess * (reducible / total_reducible))  # proportional reduction
    return reduction

#%%



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
