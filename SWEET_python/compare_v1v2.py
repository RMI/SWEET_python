#%%

import pandas as pd
import numpy as np
from SWEET_python.city_params import City as CityNew
from SWEET_python.sweet_tools_obj import City as CityOld


#%%

filepath_db = '../../WasteMAP/data/cities_for_map.csv'
db_file = pd.read_csv(filepath_db)
db_file.index = db_file['City']

#%%

cities_to_run_new = {}
cities_to_run_old = {}
adjusted_cities = []
cities = db_file['City'].unique()

# Run the new code
for city_name in cities:
    city_new = CityNew(city_name)
    city_new.load_from_csv(db_file)
    for landfill in city_new.baseline_parameters.non_zero_landfills:
        landfill.estimate_emissions()
    city_new.organic_emissions_baseline = city_new.estimate_diversion_emissions(scenario=city_new.baseline_parameters.scenario)
    city_new.landfill_emissions_baseline, city_new.diversion_emissions_baseline, city_new.total_emissions_baseline = city_new.sum_landfill_emissions()
    cities_to_run_new[city_new.name] = city_new
    #break

# Run the old code
for city_name in cities:
    city_old = CityOld(city_name)
    city_old.load_from_database(db_file)
    for landfill in city_old.non_zero_landfills:
        landfill.estimate_emissions(baseline=True)
    city_old.organic_emissions_baseline = city_old.estimate_diversion_emissions(baseline=True)
    city_old.landfill_emissions_baseline, city_old.diversion_emissions_baseline, city_old.total_emissions_baseline = city_old.sum_landfill_emissions(baseline=True)
    cities_to_run_old[city_old.name] = city_old
    #break

#%%

# Compare the results
for city_name in cities:
    city_new = cities_to_run_new[city_name]
    city_old = cities_to_run_old[city_name]

    new_emissions = city_new.total_emissions_baseline['total'].loc[2025]
    old_emissions = city_old.total_emissions_baseline['total'].loc[2025]
    
    difference = np.abs(new_emissions - old_emissions)
    percentage_difference = difference / old_emissions * 100

    if percentage_difference > 1:
        print(f"Discrepancy found in {city_name}:")
        print(f"New emissions: {new_emissions}")
        print(f"Old emissions: {old_emissions}")
        print(f"Percentage difference: {percentage_difference}")
    #else:
        #print(f"{city_name}: Emissions match within 1%")

# %%
