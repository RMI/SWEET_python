#%%
from importlib import reload
import sweet_tools_obj 
sweet_tools_obj = reload(sweet_tools_obj)
from sweet_tools_obj import City
import defaults
import pandas as pd
import numpy as np
#import copy
import warnings
import sys
# I need to reorganize the github repo for better module importing
sys.path.append('/Users/hugh/Library/CloudStorage/OneDrive-RMI/Documents/RMI/WasteMAP/map/visualizations')
from per_capita import *

from fastapi import FastAPI, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
from fastapi import Path

# Convert RuntimeWarning into an error
warnings.filterwarnings('error', category=RuntimeWarning)

#%%

# This loads tables with baseline parameters

pth = '/Users/hugh/Library/CloudStorage/OneDrive-RMI/Documents/RMI/WasteMAP/decision_support_tool/python_only_sweet/'

filepath_wb = pth + 'city_level_data_0_0.csv'
filepath_rmi = pth + 'Merged Waste Dataset Updated.xlsx'
filepath_un = pth + 'data_overview_2022.xlsx'

param_file = pd.read_csv(filepath_wb)
rmi_db = pd.read_excel(filepath_rmi, sheet_name=0)
rmi_db = rmi_db[rmi_db['Data Source'] == 'World Bank']
rmi_db.index = rmi_db['City_original']
un_data_overview = pd.read_excel(filepath_un, sheet_name='Data overview', header=1).loc[:, 'Country':].T
un_data_overview.columns = un_data_overview.iloc[0, :]
un_data_overview = un_data_overview.iloc[1:-4, :]


#%%

# This reads the baseline parameters into the model and calculates emissions

cities_to_run = {}
problem_cities = []
adjusted_cities = []
for row in param_file.iterrows():
    try:
        rmi_db.at[row[1]['city_name'], 'Population_1950']
    except:
        continue
    city = City(row[1]['city_name'])
    print(city.name)
    city.load_wb_params(row, rmi_db)
    
    cities_to_run[city.name] = city
    if city.input_problems:
        problem_cities.append(city.name)
    elif city.changed_diversion:
        adjusted_cities.append(city.name)

    for landfill in city.landfills:
        landfill.estimate_emissions()
    
    city.estimate_diversion_emissions()
    city.sum_landfill_emissions()

#%%

# This outputs emissions to csv files, output format will change in the future, mostly a placeholder

# for city_name, city in cities_to_run.items():
#     #print(city_name, city.total_emissions.isna().sum().sum())
#     city.total_emissions.to_csv(f'../../data/city_emissions/{city_name}.csv')

print('some stuff happened!')

#%%

# Load country-level stuff

pth = '../../map/visualizations/'

# This cell won't work yet, sorry, need to add some stuff to the Github repo and change paths. 
lut = LUT('/Users/hugh/Library/CloudStorage/OneDrive-RMI/Documents/RMI/GIS_Projects/country_code_lut.csv')

edgar = DataBase('EDGAR', lut, load_finished_path=f"{pth}edgar.csv")
unfccc = unfccc = DataBase('UNFCCC', lut, load_finished_path=f"{pth}unfccc.csv")

edgar.make_map(load_finished_path=f"{pth}edgar_map.geojson")

# This is some ugly wrangling, fix later
edgar.map_df.index = edgar.map_df['iso_a3']
edgar.map_df.drop(index=edgar.map_df.index[edgar.map_df.index.isna()], inplace=True)
edgar.map_df = edgar.map_df[edgar.map_df['name'] != 'Somaliland']
edgar.map_df = edgar.map_df[edgar.map_df['name'] != 'N. Cyprus']
edgar.make_edgar_pc_table(lut)
unfccc.make_unfccc_pc_table()

#%%

# Make a dictionary with country values from emissions datasets
db = {}
datasets = [edgar, unfccc]

for country in lut.countries:
    db[country] = Country(country, lut, *datasets)


#%%

# Some example API calls. 

service = FastAPI(
    title="RMI SWEET API",
    description="RMI SWEET Waste Model API",
)


# redirect the default URL to the OpenAPI docs
@service.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


# health check endpoint to make sure the service is up and running
@service.get("/v1/health", responses={200: {"description": "HealthCheck"}})
def get_health():
    return {"healthy": True}

# countries endpoint 
@service.get(
    "/v1/country_emissions/{country}",
    responses={
        200: {
            "description": "Emission estimates for a country",
        }
    },
    response_class=JSONResponse,
    description="Returns a list of all available countries",
)
def get_country_emissions():
    return JSONResponse(content=jsonable_encoder(db[country]))


# Specific city endpoint 
@service.get(
    "/v1/city_emissions/{city}",
    responses={
        200: {"description": "Emission estimates for a city"},
    },
)
def get_country_emissions():
    return JSONResponse(content=jsonable_encoder(cities_to_run[city].total_emissions))

# Change diversion value
@service.get(
    "/v1/city_emissions/change_diversion",
    responses={
        200: {"description": "Change diversion value and recalculate emissions"},
    },
)
def change_diversion_value(city: str = Query(..., description="Name of the city"),
                           diversion_type: str = Query(..., description="Type of diversion"),
                           new_value: float = Query(..., description="New value for diversion")):
    city_instance = cities_to_run[city]
    city_instance.change_diversion(diversion_type, new_value)
    return JSONResponse(content=jsonable_encoder(city_instance.total_emissions))

