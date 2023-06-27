import argparse
import os
#import ast
import pandas as pd
import sweet_tools
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="A program that accepts a config file path")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    return args

def load_params_from_python_file(filepath):
    path, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)
    import_path = path.replace(os.sep, '.') + '.' + filename
    config_module = __import__(import_path, fromlist=[filename])
    params = config_module.params
    
    params['region'] = sweet_tools.region_lookup[params['country']]
    
    assert bool(params['waste_per_capita']) != bool(params['waste_mass']), \
        "Can't specify both waste per capita and total in config, one is calculated from the other"
    
    if params['waste_per_capita'] != params['waste_per_capita']:
        params['waste_per_capita'] = sweet_tools.msw_per_capita_defaults[params['region']]
    
    if params['waste_per_capita']:
        params['waste_mass'] = params['waste_per_capita'] * params['population']
    
    # If waste fractions aren't loaded, use defaults
    if not params['waste_fractions']:
        sweet_tools.waste_fraction_defaults.loc[params['region'], :]
    
    # Waste_fractions should equal 1
    fraction_total = sum(params['waste_fractions'])
    assert 0.99 <= fraction_total <= 1.01, "Sum of all values must be between 0.99 and 1.01"
    
    # Calculate methane emission factor for compost
    try:
        params['mef_compost'] = (0.0055 * params['waste_fractions']['food'] / \
                                (params['waste_fractions']['food'] + params['waste_fractions']['green']) + \
                                0.0139 * params['waste_fractions']['green'] / \
                                (params['waste_fractions']['food'] + params['waste_fractions']['green'])) * \
                                1.1023 * 0.7 / 28
        # Unit is Mg CO2e/Mg of organic waste, wtf, so convert to CH4. Mistake in sweet here
    except:
        params['mef_compost'] = 0
    
    # Look up precipitation zone
    params['precip_zone'] = sweet_tools.get_precipitation_zone(params['precip'])
    
    # Look up k decay factors
    params['ks'] = sweet_tools.k_defaults[params['precip_zone']]
    
    mcf_landfill = 1
    mcf_dump = .4
    params['mcfs'] = {'landfill': mcf_landfill, 'dump_site': mcf_dump}
    
    # Get the total that goes to landfill and dump site combined
    split_total = sum([params['split_fractions'][x] for x in params['split_fractions'].keys()])
    
    if split_total == 0:
        # Set to dump site only if no data
        params['split_fractions'] = {'landfill': 0, 'dump_site': 1}
    else:
        # Calculate % of waste that goes to landfill and dump site, of waste
        # going to one or the other
        for site in params['split_fractions'].keys():
            params['split_fractions'][site] /= split_total
    
    # Compost. If mass is specified, calculate fraction
    assert bool(params['compost_fraction']) != bool(params['compost_mass']), \
        "Can't specify both compost mass and fraction, one is calculated from the other" 
    if params['compost_mass']:
        params['compost_fraction'] = params['compost_mass'] / params['waste_mass']
    
    # Anaerobic digestion. If mass is specified, calculate fraction
    assert bool(params['anaerobic_fraction']) != bool(params['anaerobic_mass']), \
        "Can't specify both anaerobic mass and fraction, one is calculated from the other" 
    if params['anaerobic_mass']:
        params['anaerobic_fraction'] = params['anaerobic_mass'] / params['waste_mass']
        
    # Combustion. If mass is specified, calculate fraction
    assert bool(params['combustion_fraction']) != bool(params['combustion_mass']), \
        "Can't specify both combustion mass and fraction, one is calculated from the other" 
    if params['combustion_mass']:
        params['combustion_fraction'] = params['combustion_mass'] / params['waste_mass']
        
    # Recycling. If mass is specified, calculate fraction
    assert bool(params['recycling_fraction']) != bool(params['recycling_mass']), \
        "Can't specify both recycling mass and fraction, one is calculated from the other" 
    if params['recycling_mass']:
        params['recycling_fraction'] = params['recycling_mass'] / params['waste_mass']
    
    return {0: params}

def load_params_from_csv(filepath):
    
    # Initiate parameter dictionary
    params = {}
    
    # Load parameter file
    param_file = pd.read_csv(filepath)
    
    # Loop over rows and store sets of parameters
    for row in param_file.iterrows():
        
        idx = row[0]
        row = row[1]
        
        # Make a set of params for each unique run
        run_params = {}
        
        run_params['city'] = row['city_name']
        run_params['country'] = row['country_name']
        run_params['region'] = sweet_tools.region_lookup[run_params['country']]
        
        # Population
        run_params['population'] = float(row['population_population_number_of_people'])
        run_params['growth_rate'] = 1.03 # Need a lookup for this
    
        # Get waste total
        try:
            waste_mass = float(row['total_msw_total_msw_generated_tons_year'])
        except:
            waste_mass = float(row['total_msw_total_msw_generated_tons_year'].replace(',', ''))
        if waste_mass != waste_mass:
            # Use per capita default
            waste_per_capita = sweet_tools.msw_per_capita_defaults[run_params['region']]
            waste_mass = waste_per_capita * run_params['population']
        
        run_params['waste_mass'] = waste_mass
        
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
            waste_fractions = sweet_tools.waste_fraction_defaults.loc[run_params['region'], :]
        else:
            waste_fractions.fillna(0, inplace=True)
            waste_fractions['textiles'] = 0
        
        if (waste_fractions.sum() < .9) or (waste_fractions.sum() > 1.1):
            #print('waste fractions do not sum to 1')
            waste_fractions = sweet_tools.waste_fraction_defaults.loc[run_params['region'], :]
    
        waste_fractions = waste_fractions.to_dict()
        
        run_params['waste_fractions'] = waste_fractions
        
        try:
            run_params['mef_compost'] = (0.0055 * waste_fractions['food']/(waste_fractions['food'] + waste_fractions['green']) + \
                           0.0139 * waste_fractions['green']/(waste_fractions['food'] + waste_fractions['green'])) * 1.1023 * 0.7 # / 28
                           # Unit is Mg CO2e/Mg of organic waste, wtf, so convert to CH4. Mistake in sweet here
        except:
            run_params['mef_compost'] = 0
        
        # Precipitation
        precip = 1277.8 # mm
        run_params['precip_zone'] = sweet_tools.get_precipitation_zone(precip)
    
        # depth
        #depth = 10
    
        # k values
        run_params['ks'] = sweet_tools.k_defaults[run_params['precip_zone']]
        
        # Model components
        run_params['components'] = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles'])
        
        # Determine split between landfill and dump site
        run_params['sites'] = ['landfill', 'dump_site']
        mcf_landfill = 1
        mcf_dump = .4
        run_params['mcfs'] = {'landfill': mcf_landfill, 'dump_site': mcf_dump}
        split_fractions = {'landfill': np.nan_to_num(row['waste_treatment_controlled_landfill_percent'])/100,
                                'dump_site': (np.nan_to_num(row['waste_treatment_open_dump_percent']) +
                                        np.nan_to_num(row['waste_treatment_landfill_unspecified_percent']) +
                                        np.nan_to_num(row['waste_treatment_unaccounted_for_percent']))/100}
        
        # Get the total that goes to landfill and dump site combined
        split_total = sum([split_fractions[x] for x in split_fractions.keys()])
        
        if split_total == 0:
            # Set to dump site only if no data
            split_fractions = {'landfill': 0, 'dump_site': 1}
        else:
            # Calculate % of waste that goes to landfill and dump site, of waste
            # going to one or the other
            for site in split_fractions.keys():
                split_fractions[site] /= split_total
        run_params['split_fractions'] = split_fractions
        
        # Compost params
        run_params['compost_components'] = set(['food', 'green', 'wood']) # Double check we don't want to include paper
        run_params['compost_fraction'] = np.nan_to_num(row['waste_treatment_compost_percent']) / 100  
        
        # Anaerobic digestion params
        run_params['anaerobic_components'] = set(['food', 'green', 'wood'])
        run_params['anaerobic_fraction'] = np.nan_to_num(row['waste_treatment_anaerobic_digestion_percent']) / 100     
        
        # Combustion params
        run_params['combustion_components'] = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber'])
        combustion_fraction_of_total = (np.nan_to_num(row['waste_treatment_incineration_percent']) + 
                                        np.nan_to_num(row['waste_treatment_advanced_thermal_treatment_percent']))/ 100
        run_params['combustion_fraction'] = combustion_fraction_of_total
        
        # Recycling params
        run_params['recycling_components'] = set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'])
        run_params['recycling_fraction'] = np.nan_to_num(row['waste_treatment_recycling_percent']) / 100
        
        # Each row gets a parameter dictionary key
        params[idx] = run_params
        
    return params

def load_params(config_path):
    _, ext = os.path.splitext(config_path)
    if ext == '.py':
        return load_params_from_python_file(config_path)
    elif ext == '.csv':
        return load_params_from_csv(config_path)
    else:
        raise ValueError(f"Unsupported file type {ext}")