#%%

import pandas as pd

waste_fraction_defaults = {'Australia and New Zealand' : [0.259, 0.122, 0.065, 0.12, 0.029, 0.083, 0.018, 0.028, 0.0000001, 0.276, 1.0000001],
                            'Caribbean' : [0.469, 0.0000001, 0.024, 0.17, 0.051, 0.099, 0.05, 0.057, 0.019, 0.035, 0.9740001],
                            'Central America' : [0.627, 0, 0.003, 0.126, 0.022, 0.103, 0.027, 0.033, 0, 0.06, 1.001],
                            'Eastern Africa' : [0.444, 0.069, 0.005, 0.104, 0.03, 0.08, 0.026, 0.021, 0.004, 0.217, 1],
                            'Eastern Asia' : [0.403, 0, 0.021, 0.204, 0.01, 0.065, 0.027, 0.043, 0, 0.229, 1.002],
                            'Eastern Europe' : [0.318, 0.024, 0.025, 0.171, 0.031, 0.046, 0.007, 0.018, 0.005, 0.354, 0.999],
                            'Middle Africa' : [0.284, 0, 0, 0.08, 0.013, 0.071, 0.014, 0.011, 0.0000001, 0.527, 1.0000001],
                            "North America" : [0.202, 0.068, 0.041, 0.233, 0.039, 0.158, 0.064, 0.042, 0.016, 0.14, 1.003],
                            "Northern Africa" : [0.504, 0, 0, 0.121, 0.058, 0.138, 0.044, 0.033, 0.0000001, 0.105, 1.0030001],
                            "Northern Europe" : [0.303, 0.052, 0.018, 0.138, 0.032, 0.049, 0.014, 0.043, 0, 0.352, 1.001],
                            "Rest of Oceania" : [0.675, 0.0000001, 0.025, 0.06, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.7600007],
                            "South America" : [0.541, 0.033, 0, 0.124, 0.017, 0.137, 0.02, 0.03, 0.006, 0.091, 0.999],
                            "Central Asia" : [0.3, 0.014, 0.025, 0.247, 0.035, 0.084, 0.008, 0.059, 0, 0.23, 1.002],
                            "South-Eastern Asia" : [0.499, 0.01, 0.008, 0.112, 0.004, 0.102, 0.042, 0.037, 0, 0.186, 1],
                            "Southern Africa" : [0.24, 0, 0, 0.145, 0.055, 0.265, 0.065, 0.09, 0, 0.14, 1],
                            "Southern Europe" : [0.358, 0.014, 0.012, 0.214, 0.028, 0.141, 0.02, 0.035, 0.002, 0.178, 1.002],
                            "Western Africa" : [0.539, 0, 0, 0.075, 0.019, 0.064, 0.027, 0.013, 0, 0.265, 1.002],
                            "Western Asia" : [0.422, 0.032, 0.008, 0.153, 0.03, 0.172, 0.025, 0.034, 0.003, 0.122, 1.001],
                            "Western Europe" : [0.332, 0.027, 0.023, 0.172, 0.059, 0.205, 0.015, 0.014, 0, 0.153, 1],
                            "Southern Asia" : [0.661, 0, 0, 0.092, 0.012, 0.07, 0.009, 0.015, 0.004, 0.139, 1.002]}

waste_fraction_defaults = pd.DataFrame(waste_fraction_defaults).T
waste_fraction_defaults.columns = ['food', 'green', 'wood', 'paper_cardboard', 'textiles', 'plastic', 'metal', 'glass', 'rubber', 'other', 'total']
waste_fraction_defaults = waste_fraction_defaults.drop('total', axis=1)
#waste_fraction_defaults.head()
#waste_fraction_defaults = waste_fraction_defaults.T.to_dict()

region_lookup = {'China': 'Eastern Asia',
                 'Japan': 'Eastern Asia',
                 'Republic of Korea': 'Eastern Asia',
                 'Mongolia': 'Central Asia',
                 'Democratic People\'s Republic of Korea': 'Eastern Asia',
                 'Bangladesh': 'Southern Asia',
                 'India': 'Southern Asia',
                 'Nepal': 'Southern Asia',
                 'Sri Lanka': 'Southern Asia',
                 'Pakistan': 'Southern Asia',
                 'Brunei Darussalam': 'South-Eastern Asia',
                 'Cambodia': 'South-Eastern Asia',
                 'Indonesia': 'South-Eastern Asia',
                 'Lao People\'s Democratic Republic': 'South-Eastern Asia',
                 'Malaysia': 'South-Eastern Asia',
                 'Myanmar': 'South-Eastern Asia',
                 'Philippines': 'South-Eastern Asia',
                 'Singapore': 'South-Eastern Asia',
                 'Thailand': 'South-Eastern Asia',
                 'Vietnam ': 'South-Eastern Asia',
                 'Afghanistan': 'Central Asia',
                 'Kazakhstan': 'Central Asia',
                 'Tajikistan': 'Central Asia',
                 'Turkmenistan': 'Central Asia',
                 'Kyrgyzstan': 'Central Asia',
                 'Uzbekistan': 'Central Asia',
                 'Armenia': 'Western Asia',
                 'Georgia': 'Western Asia',
                 'Azerbaijan': 'Western Asia',
                 'Saudi Arabia': 'Western Asia',
                 'Iran': 'Western Asia',
                 'Iraq': 'Western Asia',
                 'Syrian Arab Republic': 'Western Asia',
                 'Lebanon': 'Western Asia',
                 'Israel': 'Western Asia',
                 'Kuwait': 'Western Asia',
                 'Jordan': 'Western Asia',
                 'Yemen': 'Western Asia',
                 'Oman': 'Western Asia',
                 'United Arab Emirates': 'Western Asia',
                 'Qatar': 'Western Asia',
                 'Bahrain': 'Western Asia',
                 'Egypt': 'Northern Africa',
                 'Libya': 'Northern Africa',
                 'Tunisia': 'Northern Africa',
                 'Algeria': 'Northern Africa',
                 'Morocco': 'Northern Africa',
                 'Sudan': 'Northern Africa',
                 'South Sudan': 'Northern Africa',
                 'Western Sahara': 'Northern Africa',
                 'Eritrea': 'Eastern Africa',
                 'Ethiopia': 'Eastern Africa',
                 'Somalia': 'Eastern Africa',
                 'Djibouti': 'Eastern Africa',
                 'Uganda': 'Eastern Africa',
                 'Kenya': 'Eastern Africa',
                 'Rwanda': 'Eastern Africa',
                 'Burundi': 'Eastern Africa',
                 'Tanzaia': 'Eastern Africa',
                 'Mauritanina': 'Western Africa',
                 'Senegal': 'Western Africa',
                 'The Gambia': 'Western Africa',
                 'Guinne-Bissau': 'Western Africa',
                 'Guinea': 'Western Africa',
                 'Sierra Leone': 'Western Africa',
                 'Liberia': 'Western Africa',
                 'Cote d\'Ivoire': 'Western Africa',
                 'Ghana': 'Western Africa',
                 'Togo': 'Western Africa',
                 'Benin': 'Western Africa',
                 'Nigeria': 'Western Africa',
                 'Mali': 'Middle Africa',
                 'Burkina Faso': 'Middle Africa',
                 'Niger': 'Middle Africa',
                 'Chad': 'Middle Africa',
                 'Central African Republic': 'Middle Africa',
                 'Cameroon': 'Middle Africa',
                 'Republic of Congo': 'Middle Africa',
                 'Democratic Republic of Congo': 'Middle Africa',
                 'Angola': 'Middle Africa',
                 'Zambia': 'Middle Africa',
                 'Gabon': 'Middle Africa',
                 'Malawi': 'Middle Africa',
                 'Namibia': 'Southern Africa',
                 'Botswana': 'Southern Africa',
                 'Zimbabwe': 'Southern Africa',
                 'Mozambique': 'Southern Africa',
                 'Madagascar': 'Southern Africa',
                 'South Africa': 'Southern Africa',
                 'Lesotho': 'Southern Africa',
                 'Eswatini': 'Southern Africa',
                 'Comoros': 'Southern Africa',
                 'Bulgaria': 'Eastern Europe',
                 'Croatioa': 'Eastern Europe',
                 'Serbia': 'Eastern Europe',
                 'Montenegro': 'Eastern Europe',
                 'Kosovo': 'Eastern Europe',
                 'Bosnia and Herzegovina': 'Eastern Europe',
                 'Albania': 'Eastern Europe',
                 'Czechia': 'Eastern Europe',
                 'Slovakia': 'Eastern Europe',
                 'Slovenia': 'Eastern Europe',
                 'Romania': 'Eastern Europe',
                 'Russian Federation': 'Eastern Europe',
                 'Ukraine': 'Eastern Europe',
                 'Belarus': 'Eastern Europe',
                 'Latvia': 'Eastern Europe',
                 'Lithuania': 'Eastern Europe',
                 'Estonia': 'Eastern Europe',
                 'Poland': 'Eastern Europe',
                 'Hungary': 'Eastern Europe',
                 'Republic of Moldova': 'Eastern Europe',
                 'Denmark': 'Northern Europe',
                 'Finland': 'Northern Europe',
                 'Sweden': 'Northern Europe',
                 'Norway': 'Northern Europe',
                 'Iceland': 'Northern Europe',
                 'Cyprus': 'Southern Europe',
                 'Greece': 'Southern Europe',
                 'Italy': 'Southern Europe',
                 'Malta': 'Southern Europe',
                 'Spain': 'Southern Europe',
                 'Portugal': 'Southern Europe',
                 'Turkey': 'Southern Europe',
                 'Austria': 'Western Europe',
                 'Belgium': 'Western Europe',
                 'Netherlands': 'Western Europe',
                 'France': 'Western Europe',
                 'Germany': 'Western Europe',
                 'Ireland': 'Western Europe',
                 'Luxembourg': 'Western Europe',
                 'Switzerland': 'Western Europe',
                 'United Kingdom': 'Western Europe',
                 'Bahamas': 'Caribbean',
                 'Cuba': 'Caribbean',
                 'Dominican Republic': 'Caribbean',
                 'St. Lucia': 'Caribbean',
                 'Costa Rica': 'Central America',
                 'Guatemala': 'Central America',
                 'Hondruas': 'Central America',
                 'Nicarauga': 'Central America',
                 'Belize': 'Central America',
                 'Panama': 'Central America',
                 'El Salvador': 'Central America',
                 'Argentina': 'South America',
                 'Bolivia': 'South America',
                 'Brazil': 'South America',
                 'Chile': 'South America',
                 'Colombia': 'South America',
                 'Ecuador': 'South America',
                 'Paraguay': 'South America',
                 'Peru': 'South America',
                 'Uruguay': 'South America',
                 'Venezuela': 'South America',
                 'Canada': 'North America',
                 'Mexico': 'North America',
                 'United States of America': 'North America',
                 'American Samoa': 'North America',
                 'Australia': 'Australia and New Zealand',
                 'New Zealand': 'Australia and New Zealand',
                 'Fiji': 'Rest of Oceania',
                 'Papua New Guinea': 'Rest of Oceania',
                 'Bhutan': 'Southern Asia',
                 'Côte d’Ivoire': 'Western Africa',
                 'Congo, Dem. Rep.': 'Middle Africa',
                 'Congo, Rep.': 'Middle Africa',
                 'Czech Republic': 'Eastern Europe',
                 'Egypt, Arab Rep.': 'Northern Africa',
                 'Micronesia, Fed. Sts.': 'Rest of Oceania',
                 'Gambia, The': 'Western Africa',
                 'Equatorial Guinea': 'Middle Africa',
                 'Honduras': 'Central America',
                 'Croatia': 'Eastern Europe',
                 'Haiti': 'Caribbean',
                 'Isle of Man': 'Western Europe',
                 'Iran, Islamic Rep.': 'Western Asia',
                 'Kyrgyz Republic': 'Central Asia',
                 'Kiribati': 'Rest of Oceania',
                 'Korea, Rep.': 'Eastern Asia',
                 'Lao PDR': 'South-Eastern Asia',
                 'Moldova': 'Eastern Europe',
                 'Maldives': 'Southern Asia',
                 'Marshall Islands': 'Rest of Oceania',
                 'Macedonia, FYR': 'Eastern Europe',
                 'Northern Mariana Islands': 'Rest of Oceania',
                 'Mauritania': 'Western Africa',
                 'Nicaragua': 'Central America',
                 'Palau': 'Rest of Oceania',
                 'West Bank and Gaza': 'Western Asia',
                 'Solomon Islands': 'Rest of Oceania',
                 'Slovak Republic': 'Eastern Europe',
                 'Timor-Leste': 'South-Eastern Asia',
                 'Tonga': 'Rest of Oceania',
                 'Tuvalu': 'Rest of Oceania',
                 'Tanzania': 'Eastern Africa',
                 'United States': 'North America',
                 'Venezuela, RB': 'South America',
                 'Vietnam': 'South-Eastern Asia',
                 'Vanuatu': 'Rest of Oceania',
                 'Samoa': 'Rest of Oceania',
                 'Yemen, Rep.': 'Western Asia',
                }


msw_per_capita_defaults = {'Australia and New Zealand' : 1.643835616438360,
                           'Caribbean' : 2.136986301369860,
                           'Central America' : 1.589041095890410,
                           'Central Asia' : 0.931506849315068,
                           'Eastern Africa' : 0.794520547945205,
                           'Eastern Asia' : 1.315068493150680,
                           'Eastern Europe' : 1.013698630136990,
                           'Middle Africa' : 0.520547945205479,
                           'North America' : 2.630136986301370,
                           "Northern Africa" : 1.123287671232880,
                           "Northern Europe" : 1.315068493150680,
                           "Rest of Oceania" : 0.931506849315068,
                           "South America" : 1.178082191780820,
                           "South-Eastern Asia" : 1.260273972602740,
                           "Southern Africa" : 0.904109589041096,
                           "Southern Asia" : 1.369863013698630,
                           "Southern Europe" : 1.287671232876710,
                           "Western Africa" : 0.493150684931507,
                           "Western Asia" : 1.890410958904110,
                           "Western Europe" : 1.616438356164380,
                            }

k_defaults = {'Dry':            {'food': .1,  'diapers': .1,  'green': .05, 'paper_cardboard': .02,  'textiles': .02,  'wood': .01,  'rubber': .01},
              'Moderately Dry': {'food': .18, 'diapers': .18, 'green': .09, 'paper_cardboard': .036, 'textiles': .036, 'wood': .018, 'rubber': .018},
              'Moderately Wet': {'food': .26, 'diapers': .26, 'green': .12, 'paper_cardboard': .048, 'textiles': .048, 'wood': .024, 'rubber': .024},
              'Wet':            {'food': .34, 'diapers': .34, 'green': .15, 'paper_cardboard': .06,  'textiles': .06,  'wood': .03,  'rubber': .03},
              'Very Wet':       {'food': .4,  'diapers': .4,  'green': .17, 'paper_cardboard': .07,  'textiles': .07,  'wood': .035, 'rubber': .035}
             }

#k_defaults = pd.DataFrame(k_defaults).T

# Function to get precipitation zone from annual rainfall
def get_precipitation_zone(rainfall):
    # Unit is mm
    if rainfall < 500:
        return "Dry"
    elif rainfall < 1000:
        return "Moderately Dry"
    elif rainfall < 1500:
        return "Moderately Wet"
    elif rainfall < 2000:
        return "Wet"
    else:
        return "Very Wet"
    
L_0 = {'food': 70, 'diapers': 112, 'green': 93, 'paper_cardboard': 186, 
       'textiles': 112, 'wood': 200, 'rubber': 100}

oxidation_factor = {'with_lfg':{'landfill': 0.22, 'controlled_dump_site': 0.1, 
                                'dump_site': 0, 'remediated_to_landfill': 0.18}, 
                    'without_lfg':{'landfill': 0.1, 'controlled_dump_site': 0.05, 
                                   'dump_site': 0, 'remediated_to_landfill': 0.1}}

#mef_compost = 0.005876238822222 # Unit is Mg CO2e/Mg of organic waste, wtf
mef_anaerobic = 0.26/1000*1.1023 # Unit is Mg CH4/Mg organic waste...wtf 
ch4_to_co2e = 28

def calc_compost_vol(run_params):
    # Helps to set to 0 at start
    compost_total = 0
    
    # Total mass of compost
    compost_total = run_params['compost_fraction'] * run_params['waste_mass']
    
    # Sum of fraction of waste types that are compostable
    fraction_compostable_types = sum([run_params['waste_fractions'][x] for x in run_params['compost_components']])
    run_params['unprocessable'] = {'food': .0192, 'green': .042522, 'wood': .07896, 'paper_cardboard': .12}
    
    if run_params['compost_fraction'] != 0:
        compost_waste_fractions = {x: run_params['waste_fractions'][x] / fraction_compostable_types for x in run_params['compost_components']}
        #non_compostable_not_targeted = .1 # I don't know what this means, basically, waste that gets composted that shouldn't have been and isn't compostable?
        non_compostable_not_targeted = {'food': .1, 'green': .05, 'wood': .05, 'paper_cardboard': .1}
        non_compostable_not_targeted_total = sum([non_compostable_not_targeted[x] * \
                                                  compost_waste_fractions[x] for x in run_params['compost_components']])
    
        compost_vol = {}
        for waste in run_params['compost_components']:
            compost_vol[waste] = (
                compost_total * 
                (1 - non_compostable_not_targeted_total) *
                compost_waste_fractions[waste] *
                (1 - run_params['unprocessable'][waste])
                )
    else:
        compost_vol = {x: 0 for x in run_params['compost_components']}
        compost_waste_fractions = {x: 0 for x in run_params['compost_components']}
        non_compostable_not_targeted = {'food': 0, 'green': 0, 'wood': 0, 'paper_cardboard': 0}
        non_compostable_not_targeted_total = 0
        
    run_params['compost_total'] = compost_total
    run_params['fraction_compostable_types'] = fraction_compostable_types
    run_params['compost_waste_fractions'] = compost_waste_fractions
    run_params['non_compostable_not_targeted'] = non_compostable_not_targeted
    run_params['non_compostable_not_targeted_total'] = non_compostable_not_targeted_total
    
    return run_params, compost_vol

def calc_anaerobic_vol(params):
    anaerobic_total = 0
    fraction_anaerobic_types = sum([params['waste_fractions'][x] for x in params['anaerobic_components']])
    if params['anaerobic_fraction'] != 0:
        anaerobic_total = params['anaerobic_fraction'] * params['waste_mass']
        #print(anaerobic_total)
        anaerobic_waste_fractions = {x: params['waste_fractions'][x] / fraction_anaerobic_types for x in params['anaerobic_components']}
        anaerobic_vol = {x: anaerobic_total * anaerobic_waste_fractions[x] for x in params['anaerobic_components']}
    else:
        anaerobic_vol = {x: 0 for x in params['anaerobic_components']}
        anaerobic_waste_fractions = {x: 0 for x in params['anaerobic_components']}
    
    params['anaerobic_total'] = anaerobic_total
    #params['fraction_anaerobic_types'] = fraction_anaerobic_types
    params['anaerobic_waste_fractions'] = anaerobic_waste_fractions
    
    return params, anaerobic_vol

# def calc_combustion_vol(params):
#     params['combustion_total'] = params['combustion_fraction'] * params['waste_mass']
#     fraction_combustion_types = sum([params['waste_fractions'][x] for x in params['combustion_components']])
#     combustion_reject_rate = 0 #.1 I think sweet has an error, the rejected from combustion stuff just disappears
#     if params['combustion_fraction'] != 0:
#         combustion_waste_fractions = {x: params['waste_fractions'][x] / fraction_combustion_types for x in params['combustion_components']}
#         combustion_vol = {x: combustion_waste_fractions[x] * \
#                              params['combustion_fraction'] * \
#                              (1 - combustion_reject_rate) * \
#                              params['waste_mass'] for x in params['combustion_components']}
#         #combustion_vol_total = sum([combustion_vol[x] for x in combustion_vol.keys()])
#     else:
#         combustion_vol = {x: 0 for x in params['combustion_components']}
#         combustion_waste_fractions = {x: 0 for x in params['combustion_components']}
        
#     params['fraction_combustion_types'] = fraction_combustion_types
#     params['combustion_waste_fractions'] = combustion_waste_fractions
#     params['combustion_reject_rate'] = combustion_reject_rate
        
#     return params, combustion_vol

def calc_combustion_vol(params):
    params['combustion_total'] = params['combustion_fraction'] * params['waste_mass']
    combustion_reject_rate = 0 #.1 I think sweet has an error, the rejected from combustion stuff just disappears
    if params['combustion_fraction'] != 0:
        combustion_vol = {x: params['waste_fractions'][x] * \
                             params['combustion_fraction'] * \
                             (1 - combustion_reject_rate) * \
                             params['waste_mass'] for x in params['combustion_components']}
    else:
        combustion_vol = {x: 0 for x in params['combustion_components']}
        
    params['combustion_reject_rate'] = combustion_reject_rate
        
    return params, combustion_vol

def calc_recycling_vol(params):
    params['recycling_total'] = params['recycling_fraction'] * params['waste_mass']
    fraction_recyclable_types = sum([params['waste_fractions'][x] for x in params['recycling_components']])
    recycling_reject_rates = {'wood': .8, 'paper_cardboard': .775, 'textiles': .99, 
                              'plastic': .875, 'metal': .955, 'glass': .88, 
                              'rubber': .78, 'other': .87}
    if params['recycling_fraction'] != 0:
        recycling_waste_fractions = {x: params['waste_fractions'][x] / fraction_recyclable_types for x in params['recycling_components']}
        recycling_vol = {x: params['waste_fractions'][x] / \
                         fraction_recyclable_types * \
                         params['recycling_fraction'] * \
                         (recycling_reject_rates[x]) * \
                         params['waste_mass'] for x in params['recycling_components']}
        #recycling_vol_total = sum([recycling_vol[x] for x in recycling_vol.keys()])
    else:
        recycling_vol = {x: 0 for x in params['recycling_components']}
        recycling_waste_fractions = {x: 0 for x in params['recycling_components']}
    
    params['fraction_recyclable_types'] = fraction_recyclable_types
    params['recycling_reject_rates'] = recycling_reject_rates
    params['recycling_waste_fractions'] = recycling_waste_fractions
    
    return params, recycling_vol

def convert_methane_m3_to_ton_co2e(volume_m3):
    density_kg_per_m3 = 0.7168
    mass_kg = volume_m3 * density_kg_per_m3
    mass_ton = mass_kg / 1000
    mass_co2e = mass_ton * 28
    return mass_co2e
    