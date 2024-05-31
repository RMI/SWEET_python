#%%

import pandas as pd
import numpy as np
import time
from SWEET_python.city_params import City as CityNew
from SWEET_python.sweet_tools_obj import City as CityOld

# Load the city database
filepath_db = '../../WasteMAP/data/cities_for_map.csv'
db_file = pd.read_csv(filepath_db)
db_file.index = db_file['City']

# Initialize dictionaries to store city instances
cities_to_run_new = {}
cities_to_run_old = {}
adjusted_cities = []
cities = db_file['City'].unique()

# Run the new code and measure the time
start_time_new = time.time()
for city_name in cities:
    city_new = CityNew(city_name)
    city_new.load_from_csv(db_file)
    for landfill in city_new.baseline_parameters.non_zero_landfills:
        landfill.estimate_emissions()
    city_new.estimate_diversion_emissions(scenario=0)
    city_new.sum_landfill_emissions(scenario=0)
    cities_to_run_new[city_new.name] = city_new
end_time_new = time.time()
time_taken_new = end_time_new - start_time_new

# Run the old code and measure the time
start_time_old = time.time()
for city_name in cities:
    city_old = CityOld(city_name)
    city_old.load_from_database(db_file)
    for landfill in city_old.non_zero_landfills:
        landfill.estimate_emissions(baseline=True)
    city_old.organic_emissions_baseline = city_old.estimate_diversion_emissions(baseline=True)
    city_old.landfill_emissions_baseline, city_old.diversion_emissions_baseline, city_old.total_emissions_baseline = city_old.sum_landfill_emissions(baseline=True)
    cities_to_run_old[city_old.name] = city_old
end_time_old = time.time()
time_taken_old = end_time_old - start_time_old

# Print the time taken for each version
print(f"Time taken for new version: {time_taken_new} seconds")
print(f"Time taken for old version: {time_taken_old} seconds")

#%%

# Compare the results
for city_name in cities:
    city_new = cities_to_run_new[city_name]
    city_old = cities_to_run_old[city_name]

    new_emissions = city_new.baseline_parameters.total_emissions['total'].loc[2025]
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
