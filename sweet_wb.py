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

# Initiate parameter dictionary
params = {}

# Load parameter file
param_file = pd.read_csv(filepath_wb)
rmi_db = pd.read_excel(filepath_rmi, sheet_name=0)
rmi_db = rmi_db[rmi_db['Data Source'] == 'World Bank']
rmi_db.index = rmi_db['City']

cities_to_run = {}
# Loop over rows and store sets of parameters
for row in param_file.iterrows():
    try:
        rmi_db.at[row[1]['city_name'], '1950_Population']
    except:
        continue
    city = City(row[1]['city_name'])
    city.load_wb_params(row, rmi_db)
    cities_to_run[city.name] = city
    
runs = {}
diversion_adjusted_cities = []
problem_cities = []
for city_name in cities_to_run.keys():
    
    # Load parameters
    city = cities_to_run[city_name]
    
    for landfill in city.landfills:
        landfill.estimate_emissions()
    
    city.estimate_diversion_emissions()
    city.sum_landfill_emissions()

print('some stuff happened!')

# #%%

# city = cities_to_run['Dubai']

# city.waste_per_capita
# sum([x for x in city.waste_fractions.values()])
# city.landfill_w_capture.fraction_of_waste
# city.landfill_wo_capture.fraction_of_waste
# city.dumpsite.fraction_of_waste

# city.compost_fraction
# city.anaerobic_fraction
# city.combustion_fraction
# city.recycling_fraction

# city.waste_mass * city.recycling_fraction

# city.recycling_waste_fractions
# sum([x for x in city.recycling_waste_fractions.values()])

# q = city.total_emissions

# m1 = city.landfill_w_capture.waste_mass
# m2 = city.landfill_wo_capture.waste_mass

# city.landfill_w_capture.fraction_of_waste * city.waste_mass * city.waste_fractions['food']

# q2 = city.landfill_wo_capture.emissions

# ch4_tab1 = city.landfill_w_capture.ch4
# ch4_tab2 = city.landfill_wo_capture.ch4

# r = city.divs['recycling']
# c = city.landfill_w_capture.captured

#%%


name = 'Dubai'
city = City(name)
for row in param_file.iterrows():
    if row[1]['city_name'] != 'Dubai':
        continue
    else:
        city = City(row[1]['city_name'])
        city.load_wb_params(row, rmi_db)

city.precip = 130
city.precip_zone = defaults.get_precipitation_zone(city.precip)
city.ks = defaults.k_defaults[city.precip_zone]

for landfill in city.landfills:
    landfill.estimate_emissions()

city.estimate_diversion_emissions()
city.sum_landfill_emissions()

print(city.waste_per_capita)
city.waste_mass * (city.growth_rate_historic ** (1960 - 2016)) * city.recycling_fraction
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
