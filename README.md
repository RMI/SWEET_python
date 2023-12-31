# SWEET_python
This repo contains code for a Python port of the methane emissions portions of the US Environmental Protection Agency's Solid Waste Emissions Estimation Tool (SWEET). SWEET is a first-order decay model for estimating methane emissions from municipal waste, similar to the IPCC Waste Model from the 2006 report. This python port does not contain the portions SWEET for calculating non-methane emissions. The original Excel model is available at https://globalmethane.org/resources/details.aspx?resourceid=5176. A user manual is also available at that site. 


# Installation
After following these steps, SWEET_python can be imported:
1) Use an existing python environment or create one with conda or venv. e.g. `conda create -n "sweet"`
2) Clone this repo (in a terminal, write `git clone https://github.com/RMI/SWEET_python.git`)
3) cd into the SWEET_python directory, then write `pip install -e .` to install in editable mode, which allows editing the code. Updates to the code in the repo should be automatically reflected in the installed package. If they are not, reinstall it. 


# Usage
You will have to write your own code to import your data files. Examples are in SWEET_python/sweet_tools_obj.py—the load_from_database method illustrates the many different parameters that can be specified. For many parameters, default values are available. These are stored in the defaults_2019.py file, and the sweet_tools_obj.py file contains many examples of accessing them. The code for the model itself is in model.py. Models are generally run as part of a Landfill instance—the Landfill class is defined at the bottom of sweet_tools_obj.py

The standard way to run the model is shown by this code:

```
from SWEET_python.sweet_tools_obj import City
city = City(city_name)
city.load_from_database(db_file) # Replace this with your own data loading code
for landfill in city.non_zero_landfills:
    landfill.estimate_emissions(baseline=True)

city.organic_emissions_baseline = city.estimate_diversion_emissions(baseline=True)
city.total_emissions_baseline = city.sum_landfill_emissions(baseline=True)
```

The minimum information required to run the model is:
1) The country the city is in. This is used for some default parameters, and will look up a global region, used for other defaults.
2) Population or total waste generated per year. Default waste generation values are per capita, so city population is needed to calculate total waste generation if those data are not available.
3) Population growth rate, both historic (before the current year) and future (after current year). This is used to determine how waste generation changes backwards and forwards in time. Growth rates are percent growth rate + 1, e.g. 5% growth is represented by 1.05. Historic growth rates are above 1, not below, i.e. if population grew by an average of 5% per year from 1950 to the current date, the historic growth rate would be 1.05 not .95
4) Mean annual precipitation in mm. Precipitation is used to determine waste decay rates.

The SWEET model is designed to operate at the city level, and allows multiple landfills per city. If you would like to model only a single landfill, that can be done by creating only a single Landfill instance, directing all waste to it with the fraction_of_waste attribute, and setting all the diversion values to 0—no waste is composted, anaerobically digested, combusted/incinerated, or recycled, it all goes to the landfill. Country, population or waste generation, population growth rate, and precipitation are still required. 
