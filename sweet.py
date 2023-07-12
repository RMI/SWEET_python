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
