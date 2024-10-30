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
    start_time = time.time()
    city_new = CityNew(city_name)
    end_time = time.time()
    print(f"Time taken to initialize CityNew: {end_time - start_time} seconds")
    start_time = time.time()
    city_new.load_csv_new(db_file)
    end_time = time.time()
    print(f"Time taken to load data into CityNew: {end_time - start_time} seconds")
    start_time = time.time()
    city_new._calculate_divs()
    end_time = time.time()
    print(f"Time taken to calculate divs in CityNew: {end_time - start_time} seconds")
    start_time = time.time()
    for landfill in city_new.baseline_parameters.non_zero_landfills:
        landfill.estimate_emissions()
    end_time = time.time()
    print(f"Time taken to estimate emissions in CityNew: {end_time - start_time} seconds")
    start_time = time.time()
    city_new.estimate_diversion_emissions(scenario=0)
    end_time = time.time()
    print(f"Time taken to estimate diversion emissions in CityNew: {end_time - start_time} seconds")
    start_time = time.time()
    city_new.sum_landfill_emissions(scenario=0)
    end_time = time.time()
    print(f"Time taken to sum landfill emissions in CityNew: {end_time - start_time} seconds")
    cities_to_run_new[city_new.city_name] = city_new
    break
end_time_new = time.time()
time_taken_new = end_time_new - start_time_new
print(time_taken_new)
print(city_new.baseline_parameters.total_emissions['total'].loc[2035])

#%%

# Run the old code and measure the time
start_time_old = time.time()
for city_name in cities:
    city_old = CityOld(city_name)
    city_old.load_from_database(db_file)
    start_time = time.time()
    for landfill in city_old.non_zero_landfills:
        landfill.estimate_emissions(baseline=True)
    end_time = time.time()
    #print(f"Time taken to estimate emissions in CityOld: {end_time - start_time} seconds")
    city_old.organic_emissions_baseline = city_old.estimate_diversion_emissions(baseline=True)
    city_old.landfill_emissions_baseline, city_old.diversion_emissions_baseline, city_old.total_emissions_baseline = city_old.sum_landfill_emissions(baseline=True)
    cities_to_run_old[city_old.name] = city_old
    break
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

    new_emissions = city_new.baseline_parameters.total_emissions['total']
    old_emissions = city_old.total_emissions_baseline['total']

    years = new_emissions.index

    for year in years:
        if year == 2073:
            continue
        new_emissions_year = new_emissions.loc[year]
        old_emissions_year = old_emissions.loc[year]
        
        difference = np.abs(new_emissions_year - old_emissions_year)
        #if difference > 1:
            #print('blurgh')
        percentage_difference = difference / old_emissions_year * 100

        if percentage_difference > 1:
            print(f"Discrepancy found in {city_name} for year {year}:")
            print(f"New emissions: {new_emissions_year}")
            print(f"Old emissions: {old_emissions_year}")
            print(f"Percentage difference: {percentage_difference:.2f}%")
        else:
            pass
            #print(f"{city_name} for year {year}: Emissions match within 1% (difference: {percentage_difference:.2f}%)")

    break


# %%
