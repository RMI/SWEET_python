import pandas as pd
import numpy as np
import warnings
import defaults


# Convert RuntimeWarning into an error
warnings.filterwarnings('error', category=RuntimeWarning)

filepath_un = 'data_overview_2022.xlsx'

un_data_overview = pd.read_excel(filepath_un, sheet_name='Data overview', header=1).loc[:, 'Country':].T
un_data_overview.columns = un_data_overview.iloc[0, :]
un_data_overview = un_data_overview.iloc[1:-4, :]

for row in un_data_overview.iterrows():
    
    breaker = False

    name = row[1]['City']
    if name in ['Hatyai']:
        continue

    print(name)
    country = row[0].split('.')[0]
    row = row[1]
    
    region = defaults.region_lookup[country]
    population = float(row['Population'])
    
    # Get waste total
    waste_mass = float(row['MSW generated (t/d)']) * 365 # unit is tons/year
    waste_per_capita = waste_mass * 1000 / population / 365 # unit is kg/person/day
    
    if waste_mass != waste_mass:
        print('this shouldnt happen')
        # Use per capita default
        #waste_per_capita = defaults.msw_per_capita_defaults[region]
        #waste_mass = waste_per_capita * population / 1000 * 365
    
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
    
    # Model components
    components = set(['food', 'green', 'wood', 'paper_cardboard', 'textiles'])
    
    # Compost params
    compost_components = set(['food', 'green', 'wood', 'paper_cardboard']) # Double check we don't want to include paper
    
    # Anaerobic digestion params
    anaerobic_components = set(['food', 'green', 'wood', 'paper_cardboard'])
    anaerobic_fraction = 0
    
    # Recycling
    recycling_components = set(['wood', 'paper_cardboard', 'textiles', 'plastic', 'rubber', 'metal', 'glass', 'other'])
    recovery_rate = (row['total recovered materials (t/d) with rejects'] * 365) / waste_mass
    recycling_rate = row['recycling rate (recovered minus WtE)'] # - compost_fraction
    if np.absolute(recovery_rate - recycling_rate) < 0.01:
        recycling_rate = recovery_rate
    # Compost comes from recycling
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
    
    if combustion_fraction < 0.01:
        combustion_fraction = 0
        
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
        for site in split_fractions.keys():
            split_fractions[site] /= split_total

    # Check if the sum of the fractions is close to 1
    if not abs(sum(split_fractions.values()) - 1) < 1e-5:
        raise ValueError("The landfill splits do not add up to 1.")
    
    # Simple checks
    assert waste_mass * sum(waste_fractions[x] for x in ['food', 'green', 'wood']) > row['Organic waste recovered (t/d)']
    assert waste_mass * waste_fractions['paper_cardboard'] >row['Paper or Cardboard (t/d)']
    assert waste_mass * waste_fractions['plastic'] > row['Total Plastic recovered (t/d)']
    assert waste_mass * waste_fractions['glass'] > row['Glass recovered (t/d)']
    assert waste_mass * waste_fractions['metal'] > row['Metal recovered (t/d)']
    assert waste_mass * waste_fractions['other'] > row['Other waste (t/d)']

    # From here on out, variable names are column names from the excel sheet. Out isn't defined.
    out.at[name, 'composition_food_organic_waste'] = waste_fractions['food'] * 100
    out.at[name, 'composition_yard_garden_green_waste_percent'] = waste_fractions['green'] * 100
    out.at[name, 'composition_wood_percent'] = waste_fractions['wood'] * 100
    out.at[name, 'composition_paper_cardboard_percent'] = waste_fractions['paper_cardboard'] * 100
    # Have to add a textiles column
    #out.at[name, ''composition_textiles_percent'] = waste_fractions['textiles'] * 100
    out.at[name, 'composition_plastic_percent'] = waste_fractions['plastic'] * 100
    out.at[name, 'composition_glass_percent'] = waste_fractions['glass'] * 100
    out.at[name, 'composition_metal_percent'] = waste_fractions['metal'] * 100
    out.at[name, 'composition_rubber_leather_percent'] = waste_fractions['rubber'] * 100
    out.at[name, 'composition_other_percent'] = waste_fractions['other'] * 100
    
    out.at[name, 'population'] = population

    out.at[name, 'waste (tonnes per year)'] = waste_mass

    out.at[name, 'waste_treatment_compost_percent'] = compost_fraction * 100
    out.at[name, 'waste_treatment_anaerobic_digestion_percent'] = anaerobic_fraction * 100
    out.at[name, 'waste_treatment_incineration_percent'] = combustion_fraction * 100
    out.at[name, 'waste_treatment_recycling_percent'] = recycling_fraction * 100

    out.at[name, 'waste_treatment_sanitary_landfill_landfill_gas_system_percent'] = split_fractions['landfill_w_capture'] * 100 # is set to 0
    out.at[name, 'waste_treatment_controlled_landfill_percent'] = split_fractions['landfill_wo_capture'] * 100
    out.at[name, 'waste_treatment_open_dump_percent'] = split_fractions['dumpsite'] * 100
    
    # set everything not mentioned here to 0. I think I got everything...
    out.at[name, 'waste_treatment_advanced_thermal_treatment_percent'] = 0
    out.at[name, 'waste_treatment_unaccounted_for_percent'] = 0
    out.at[name, 'waste_treatment_landfill_unspecified_percent'] = 0
    out.at[name, 'waste_treatment_other_percent'] = 0