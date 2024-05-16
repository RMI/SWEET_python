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
    city = CityNew(city_name)
    city.load_from_csv(db_file)
    cities_to_run_new[city.name] = city
    for landfill in city.baseline_parameters.non_zero_landfills:
        landfill.estimate_emissions()
    city.organic_emissions_baseline = city.estimate_diversion_emissions(scenario=city.baseline_parameters.scenario)
    city.landfill_emissions_baseline, city.diversion_emissions_baseline, city.total_emissions_baseline = city.sum_landfill_emissions()

# Run the old code
for city_name in cities:
    city = CityOld(city_name)
    city.load_from_database(db_file)
    cities_to_run_old[city.name] = city
    for landfill in city.non_zero_landfills:
        landfill.estimate_emissions(baseline=True)
    city.organic_emissions_baseline = city.estimate_diversion_emissions(baseline=True)
    city.landfill_emissions_baseline, city.diversion_emissions_baseline, city.total_emissions_baseline = city.sum_landfill_emissions(baseline=True)

#%%

# Compare the results
for city_name in cities:
    city_new = cities_to_run_new[city_name]
    city_old = cities_to_run_old[city_name]

    new_emissions = city_new.total_emissions_baseline['total']
    old_emissions = city_old.total_emissions_baseline['total']
    
    difference = np.abs(new_emissions - old_emissions)
    percentage_difference = difference / old_emissions * 100

    if percentage_difference > 1:
        print(f"Discrepancy found in {city_name}:")
        print(f"New emissions: {new_emissions}")
        print(f"Old emissions: {old_emissions}")
        print(f"Percentage difference: {percentage_difference}%")
    else:
        print(f"{city_name}: Emissions match within 1%")

#%%
