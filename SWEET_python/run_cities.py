#from city_dict import cities_to_run
from city_params import City
import pandas as pd
import warnings

# Convert RuntimeWarning into an error
warnings.filterwarnings('error', category=RuntimeWarning)

#%%

filepath_db = 'cities_for_map.csv'
db_file = pd.read_csv(filepath_db)
db_file.index = db_file['City']

#%%

# This reads the baseline parameters into the model and calculates emissions

cities_to_run = {}
adjusted_cities = []
cities = db_file['City'].unique()

for city_name in cities:
    city = City(city_name)
    cities_to_run[city.city_name] = city
    city.load_from_csv(db_file)
    city._calculate_divs()
    for landfill in city.baseline_parameters.non_zero_landfills:
        landfill.estimate_emissions()
    city.estimate_diversion_emissions(scenario=0)
    city.sum_landfill_emissions(scenario=0)

def get_cities_to_run():
    global cities_to_run
    return cities_to_run

#%%