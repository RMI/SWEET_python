from sweet_tools_obj import City
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
from sweet_tools import region_lookup, msw_per_capita_defaults


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
    "/v1/countries",
    responses={
        200: {
            "content": {"appliciation/json": {}},
            "description": "A list of countries",
        }
    },
    response_class=JSONResponse,
    description="Returns a list of all available countries",
)
def get_countries():
    return JSONResponse(content=jsonable_encoder(region_lookup))


# specific country endpoint
@service.get(
    "/v1/countries/{country}",
    responses={
        200: {"description": "Returns the default parameters for a given country"}
    },
)
def get_country(country: str):
    if country not in region_lookup.keys():
        raise HTTPException(status_code=404, detail="Country not found")

    continent = region_lookup[country]
    params = msw_per_capita_defaults[continent]
    # for example
    resp = {'country': country, 'params': params}
    return JSONResponse(content=jsonable_encoder(resp))


# Specific city endpoint 
@service.get(
    "/v1/cities/{city}",
    responses={
        200: {"description": "Returns the default parameters for a given city"}
    },
)
def get_city(name: str):
    raise HTTPException(status_code=404, detail="City not found")
    # Implement this...


@service.get(
    "/v1/cities",
    responses={
        200: {"description": "Returns a list of cities"}
    },    
)
def get_cities():
    # Implement this...
    return  ['Barcelona', 'Boston', 'Birmingham','Some other city']

@service.get(
    "/v1/change_compost_percent",
    responses={
        200: {"description": "Change compost percent for a given city"}
    },
)
def change_compost_percent(city_name: str, compost_percent: float):
    # Ok so on startup I load the cities database, then this way I can change stuff. 
    cities_to_run[city_name].compost_fraction = compost_percent


filepath = 'city_level_data_0_0.csv'

# Initiate parameter dictionary
params = {}

# Load parameter file
param_file = pd.read_csv(filepath)

cities_to_run = {}
# Loop over rows and store sets of parameters
for row in param_file.iterrows():
    city = City(row[1]['city_name'])
    city.load_wb_params(row)
    cities_to_run[city.name] = city
    
runs = {}
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
