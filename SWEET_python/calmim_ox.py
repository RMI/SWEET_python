from dataclasses import dataclass
import jpype
import jpype.imports
from jpype.types import *
import numpy as np
import math
import os

@dataclass
class Site:
    lat: float
    lon: float
    avg_annual_air_temp: float = 0.0
    wind_strength: float = 0.0

    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

@dataclass
class CoverMaterial:
    name: str
    sand_percentage: float
    silt_percentage: float
    clay_percentage: float
    is_soil: bool
    organic_matter_percentage: float
    bulk_density: float
    saturated_conductivity: float
    moist33: float
    moist1500: float
    beta: float
    porosity33: float
    nonsoil_flag: int
    alpha: float = None
    air_entry_potential: float = None
    theta_residual: float = None
    max_ox_rate: float = 400 # micrograms per gram per day

    def calculate_properties(self):
        # Calculate alpha
        self.alpha = self.saturated_conductivity * 360.0 / 0.0102 / 10.0 * 0.0011824 + 0.014675
        
        # Calculate air entry potential
        self.air_entry_potential = -1.0 / abs(self.alpha)
        
        # Calculate theta residual
        x3 = -1.05976 + 0.0650437 * self.silt_percentage
        x5 = -2.21391 + 8.92268 * self.moist33 / self.bulk_density
        z6 = 0.12867 - 0.492412 * x3 + 0.787425 * x5 - 0.235254 * x3 * x5
        self.theta_residual = max(0.161487 + 0.101111 * z6, 0.01)

    # def get_rain():
    #     # Path to JAR file
    #     jar_path = 'ARS_GlobalRainSIM.Jar'

    #     # Start the JVM
    #     if not jpype.isJVMStarted():
    #         jpype.startJVM(classpath=[jar_path])

    #     # Import the RainSIM class directly
    #     RainSIM = jpype.JClass('RainSIM')

    #     # Instantiate the RainSIM class
    #     rain_sim = RainSIM()

    #     # Call the getRain method (assuming the method signature is as follows)
    #     lat = 40.7128
    #     lon = -74.0060
    #     min_val = 0.0
    #     max_val = 100.0

    #     # The method likely returns an array or list, so let's call it
    #     rain_data = rain_sim.getRain(JDouble(lat), JDouble(lon), JDouble(min_val), JDouble(max_val))

    #     # Convert the Java array to a Python list for easier manipulation
    #     rain_data_list = list(rain_data)

    #     print("Rain data:", rain_data_list)

    #     # Shutdown the JVM
    #     #jpype.shutdownJVM()


# Initialize the materials
materials = [
    CoverMaterial("sand", 93.6, 3.06, 3.34, True, 1.0, 1.627, 0.002703, 0.091, 0.039, 2.013529, 0.386, 0),
    CoverMaterial("sandy clay", 47.2, 3.7, 49.1, True, 1.0, 1.444, 1.76E-4, 0.231, 0.113, 10.208511, 0.455, 0),
    CoverMaterial("loamy sand", 82.74, 9.66, 7.6, True, 1.0, 1.584, 0.00176, 0.116, 0.051, 2.856075, 0.402, 0),
    CoverMaterial("sandy loam", 65.6, 22.5, 11.9, True, 1.0, 1.533, 7.63E-4, 0.139, 0.062, 4.288265, 0.422, 0),
    CoverMaterial("silty loam", 21.8, 62.7, 15.5, True, 1.0, 1.278, 1.37E-4, 0.276, 0.125, 6.857649, 0.518, 0),
    CoverMaterial("loam", 42.9, 39.5, 17.6, True, 1.0, 1.395, 2.92E-4, 0.223, 0.102, 6.263467, 0.474, 0),
    CoverMaterial("sandy clay loam", 60.1, 11.3, 28.6, True, 1.0, 1.471, 3.56E-4, 0.2, 0.095, 7.421883, 0.445, 0),
    CoverMaterial("silty clay loam", 9.0, 55.0, 36.0, True, 1.0, 1.239, 5.17E-5, 0.321, 0.151, 10.357549, 0.532, 0),
    CoverMaterial("clay loam", 34.7, 30.3, 35.0, True, 1.0, 1.354, 1.25E-4, 0.264, 0.125, 8.943137, 0.489, 0),
    CoverMaterial("silty clay", 9.3, 43.89, 46.81, True, 1.0, 1.279, 3.05E-5, 0.312, 0.15, 13.001208, 0.517, 0),
    CoverMaterial("clay", 10.0, 25.0, 65.0, True, 1.0, 1.262, 3.38E-5, 0.351, 0.172, 14.203435, 0.524, 0),
    CoverMaterial("silt", 7.96, 63.62, 5.41, True, 1.0, 1.156, 1.06E-4, 0.332, 0.149, 6.715722, 0.564, 0),
    CoverMaterial("rocks - pebbles", 93.6, 3.06, 3.34, True, 1.0, 1.627, 0.002703, 0.091, 0.039, 2.013529, 0.386, 0),
    CoverMaterial("rocks - boulders (large)", 93.6, 3.06, 3.34, True, 1.0, 1.627, 0.002703, 0.091, 0.039, 2.013529, 0.386, 0),
    CoverMaterial("geomembrane (hdpe)", 10.0, 25.0, 65.0, False, 1.0, 1.262, 3.38E-5, 0.351, 0.172, 14.203435, 0.524, 0),
    CoverMaterial("geomembrane (ldpe)", 10.0, 25.0, 65.0, False, 1.0, 1.262, 3.38E-5, 0.351, 0.172, 14.203435, 0.524, 0),
    CoverMaterial("geomembrane (edpm)", 10.0, 25.0, 65.0, False, 1.0, 1.262, 3.38E-5, 0.351, 0.172, 14.203435, 0.524, 0),
    CoverMaterial("geotextile (woven)", 10.0, 25.0, 65.0, False, 1.0, 1.262, 3.38E-5, 0.351, 0.172, 14.203435, 0.524, 0),
    CoverMaterial("adc foundry sands", 90.0, 5.0, 5.0, True, 0.1, 1.627, 0.003703, 0.1, 0.04, 3.0, 0.39, 0),
    CoverMaterial("adc dredged materials", 7.96, 63.62, 5.41, True, 1.0, 1.156, 1.06E-4, 0.332, 0.149, 6.715722, 0.564, 0),
    CoverMaterial("adc ash", 21.8, 62.7, 15.5, True, 0.0, 1.278, 1.37E-4, 0.276, 0.125, 6.857649, 0.518, 0),
    CoverMaterial("adc contaminated soils (clay)", 10.0, 25.0, 65.0, True, 1.0, 1.262, 3.38E-5, 0.351, 0.172, 14.203435, 0.524, 0),
    CoverMaterial("adc contaminated soils (sand)", 90.0, 5.0, 5.0, True, 0.0, 1.627, 0.003703, 0.1, 0.04, 3.0, 0.39, 0),
    CoverMaterial("adc contaminated soils (general)", 42.9, 39.5, 17.6, True, 1.0, 1.395, 2.92E-4, 0.223, 0.102, 6.263467, 0.474, 0),
    CoverMaterial("adc composted organic materials", 42.9, 39.5, 17.6, True, 5.0, 1.395, 2.92E-4, 0.223, 0.102, 6.263467, 0.474, 0),
    CoverMaterial("adc tire shreds [small <2 in (50 mm)]", 90.0, 5.0, 5.0, True, 0.0, 1.08, 0.0513, 0.19, 0.06, 2.0, 0.51, 1),
    CoverMaterial("adc tire shreds [large >2 in (50 mm)]", 90.0, 5.0, 5.0, True, 0.0, 1.627, 0.003703, 0.1, 0.04, 3.0, 0.39, 0),
    CoverMaterial("adc spray applied cement products", 10.0, 25.0, 65.0, False, 1.0, 1.262, 3.38E-5, 0.351, 0.172, 14.203435, 0.524, 3),
    CoverMaterial("adc spray applied foams", 10.0, 25.0, 65.0, False, 1.0, 1.262, 3.38E-5, 0.351, 0.172, 14.203435, 0.524, 4),
    CoverMaterial("adc wood chips (all)", 90.0, 5.0, 5.0, True, 0.0, 1.08, 0.0813, 0.22, 0.06, 2.0, 0.33, 4),
    CoverMaterial("adc sludge", 10.0, 80.0, 10.0, True, 1.0, 1.2, 0.00106, 0.452, 0.15, 8.7, 0.564, 5),
    CoverMaterial("adc temporary tarp", 10.0, 25.0, 65.0, False, 1.0, 1.262, 2.38E-6, 0.351, 0.172, 14.203435, 0.524, 0),
    CoverMaterial("adc energy resource exploration and production wastes", 42.9, 39.5, 17.6, True, 5.0, 1.395, 2.92E-4, 0.223, 0.102, 6.263467, 0.474, 6)
]

#%%

# # Determine the base directory (current directory where this script is run)
# launch_dir = os.path.dirname(__file__)

# # Determine if the script is running from the WasteMAP directory or the SWEET_python directory
# if os.path.basename(launch_dir) == 'SWEET_python':
#     # We're already in the SWEET_python directory, so the JARs should be here
#     jar_dir = launch_dir
# elif os.path.basename(launch_dir) == 'WasteMAP':
#     # We're in the WasteMAP directory, the JARs should be one directory up and then in SWEET_python/SWEET_python
#     jar_dir = os.path.abspath(os.path.join(launch_dir, '../SWEET_python/SWEET_python'))
# else:
#     print('unexpected directory structure')

# # Construct the full paths to the JAR files
# jar_paths = [
#     os.path.join(jar_dir, 'ARS_GlobalRainSIM.Jar'),
#     os.path.join(jar_dir, 'GlobalTempSim10.Jar')
# ]

# # Start the JVM if not already started, with the constructed classpath
# if not jpype.isJVMStarted():
#     jpype.startJVM(classpath=jar_paths)

# # Start the JVM with the classpath for both .jar files
# #jpype.startJVM(classpath=['ARS_GlobalRainSIM.Jar', 'GlobalTempSim10.Jar'])

# # Import the Java classes from the JAR files
# RainSIM = jpype.JClass('RainSIM')
# TempSim = jpype.JClass('TempSim')

@dataclass
class WeatherProfile:
    weather_data: np.ndarray
    weather_solar_hourly: np.ndarray
    soil_temp_add: np.ndarray
    surface_temp: np.ndarray
    rel_humidity: np.ndarray
    time_evap: np.ndarray

    def __init__(self):
        self.weather_data = np.zeros((9, 366))
        self.weather_solar_hourly = np.zeros((366, 24))
        self.soil_temp_add = np.zeros((366, 24))
        self.surface_temp = np.zeros((366, 1442))
        self.rel_humidity = np.zeros((366, 1442))
        self.time_evap = np.zeros((366, 1442))

@dataclass
class Site:
    lat: float
    lon: float
    avg_annual_air_temp: float = 0.0
    wind_strength: float = 0.0

    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

class WeatherModel:
    days = 366
    hours = 24
    max, min, precip, evap, dew, d_solar, total_precip, delta_temp, avg_temp = range(9)

    def __init__(self, site: Site, weather_profile: WeatherProfile):
        self.weather_profile = weather_profile
        self.site = site
        self.weather_data = np.zeros((9, self.days))
        self.weather_holder = np.zeros((2, self.days))
        self.solar_abs = np.zeros((self.days, self.hours))
        # self.rain_sim = RainSIM()
        # self.temp_sim = TempSim()

    def simulate_rain_only(self):
        print("RainSIM Started")
        
        # Fetch rain data
        rain_data = self.rain_sim.getRain(self.site.lat, self.site.lon, 0.0, 100.0)
        
        # Log the length to understand the size
        print(f"Length of rain_data: {len(rain_data)}")
        
        # Trim the padding. index 399 is cumulative, if i need that. 
        if len(rain_data) > 366:
            rain_data = rain_data[:366]
        
        # Assign the truncated or full rain data to the weather data array
        self.weather_data[self.precip] = rain_data
        
        print("Rain Simulation Completed")
        #self.weather_profile.weather_data = self.weather_data

    def simulate_weather(self):
        # Determine the base directory (current directory where this script is run)
        launch_dir = os.path.dirname(__file__)

        # Determine if the script is running from the WasteMAP directory or the SWEET_python directory
        if os.path.basename(launch_dir) == 'SWEET_python':
            # We're already in the SWEET_python directory, so the JARs should be here
            jar_dir = launch_dir
        elif os.path.basename(launch_dir) == 'WasteMAP':
            # We're in the WasteMAP directory, the JARs should be one directory up and then in SWEET_python/SWEET_python
            jar_dir = os.path.abspath(os.path.join(launch_dir, '../SWEET_python/SWEET_python'))
        else:
            print('unexpected directory structure')

        # Construct the full paths to the JAR files
        jar_paths = [
            os.path.join(jar_dir, 'ARS_GlobalRainSIM.Jar'),
            os.path.join(jar_dir, 'GlobalTempSim10.Jar')
        ]

        # Start the JVM if not already started, with the constructed classpath
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=jar_paths)

        # Start the JVM with the classpath for both .jar files
        #jpype.startJVM(classpath=['ARS_GlobalRainSIM.Jar', 'GlobalTempSim10.Jar'])

        # Import the Java classes from the JAR files
        RainSIM = jpype.JClass('RainSIM')
        TempSim = jpype.JClass('TempSim')

        print("RainSIM Started")
        
        # Fetch rain data
        rain_data = RainSIM.getRain(self.site.lat, self.site.lon, 0.0, 100.0)
        
        # Log the length to understand the size
        print(f"Length of rain_data: {len(rain_data)}")
        
        # Trim the padding. index 399 is cumulative, if i need that. 
        if len(rain_data) > 366:
            rain_data = rain_data[:366]
        
        # Assign the truncated or full rain data to the weather data array
        self.weather_data[self.precip] = rain_data
        
        print("Rain Simulation Completed")

        print("Temperature Simulation Started")
        self.weather_holder = TempSim.getDailyTemps(self.site.lat, self.site.lon, False)

        print(f"Length of temp data: {len(self.weather_holder[0])}")
        # Adjust the length to remove padding
        if len(self.weather_holder[0]) > 366:
            self.weather_holder = [data[:366] for data in self.weather_holder]

        self.weather_data[self.max] = self.weather_holder[0]
        self.weather_data[self.min] = self.weather_holder[1]
        print("Temperature Simulation Completed")
        #self.simulate_rain_only()
        self.process_monthly_to_daily_weather_data()
        self.weather_profile.weather_data = self.weather_data

        if jpype.isJVMStarted():
            jpype.shutdownJVM()

    def process_monthly_to_daily_weather_data(self):
        avg_temp_loop = 0.0
        avg_temp = 0.0
        rain_total = 0.0
        for day in range(self.days):
            self.sec_temp(self.weather_data[self.max][day], self.weather_data[self.min][day], day)
            delta_t = self.weather_data[self.max][day] - self.weather_data[self.min][day]
            rain_total += self.weather_data[self.precip][day]
            self.weather_data[self.delta_temp][day] = delta_t
            self.weather_data[self.total_precip][day] = rain_total
            avg_temp_loop += 1
            self.weather_data[self.avg_temp][day] = (self.weather_data[self.max][day] + self.weather_data[self.min][day]) / 2.0
            avg_temp += self.weather_data[self.avg_temp][day]
        avg_temp /= self.days
        self.site.avg_annual_air_temp = avg_temp
        self.calc_solar_data(self.site.lat, self.site.lon, 100.0)

    # rh here seems weird...maybe i need [2]
    def sec_temp(self, max_air, min_air, day):
        amplitude = (max_air - min_air) / 2.0
        avg = (max_air + min_air) / 2.0
        for t in range(1441):
            self.weather_profile.surface_temp[day][t] = avg - amplitude * math.sin(t * 7.27E-5 * 60.0)
            self.weather_profile.rel_humidity[day][t] = self.rh(min_air, self.weather_profile.surface_temp[day][t])

    def rh(self, min_temp, current_temp):
        eo = 0.6108 * math.exp(17.27 * current_temp / (current_temp + 273.3))
        eos = 0.6108 * math.exp(17.27 * min_temp / (min_temp + 273.3))
        eo = max(eo, 0.0)
        eos = max(eos, 0.0)
        rh = eos / eo
        rh = min(rh, 1.0)
        return 100.0 * rh

    def calc_solar_data(self, lat, lon, alt):
        radian_lat = lat * math.pi / 180.0
        radian_lon = lon * math.pi / 180.0
        for day in range(self.days):
            lc = self.get_lc(radian_lon)
            et = self.get_et(day)
            solar_dec = self.get_solar_declination(day)
            solar_noon = 12.0 - lc - et
            half_day_length = self.calc_half_day_length(solar_dec, radian_lat)
            sunrise = solar_noon - half_day_length
            sunset = solar_noon + half_day_length
            for hour in range(self.hours):
                today_rain = self.weather_profile.weather_data[self.precip][day]
                if day <= 1:
                    previous_day_rain = self.weather_profile.weather_data[self.precip][365]
                else:
                    previous_day_rain = self.weather_profile.weather_data[self.precip][day - 1]
                atmos_trans = 0.7
                if today_rain > 1.0:
                    atmos_trans = 0.3 if previous_day_rain > 1.0 else 0.4
                elif today_rain == 0.0 and previous_day_rain > 1.0:
                    atmos_trans = 0.6
                if abs(radian_lat / math.pi * 180.0) < 60.0:
                    delta = self.weather_profile.weather_data[self.delta_temp][day]
                    if delta <= 10.0 and delta != 0.0:
                        atmos_trans /= (11.0 - delta)
                zangle = self.zenith(radian_lat, solar_dec, solar_noon, hour)
                pa = 101.0 * math.exp(-1.0 * alt / 8200.0)
                m = pa / 101.3 / math.cos(zangle)

                # Handle potential overflow by limiting `m`
                try:
                    sp = 1360.0 * math.pow(atmos_trans, m)
                except OverflowError:
                    sp = 0.0  # Set sp to 0 to avoid overflow

                if hour < sunrise or hour > sunset:
                    sp = 0.0

                # Handle potential overflow by limiting `m`
                try:
                    sd = sd = 0.3 * (1.0 - math.pow(atmos_trans, m)) * math.cos(zangle)
                except OverflowError:
                    sd = 0.0  # Set sp to 0 to avoid overflow

                if hour < sunrise or hour > sunset:
                    sd = 0.0
                sd *= 1360.0
                sb = sp * math.cos(zangle)
                if hour < sunrise or hour > sunset:
                    sb = 0.0
                st = sb + sd
                st = max(st, 0.0)
                albedo = 0.15
                la = self.sb((self.weather_profile.weather_data[self.max][day] + self.weather_profile.weather_data[self.min][day]) / 2.0)
                rabs = (1.0 - albedo) * st + 0.17000000000000004 * la
                self.solar_abs[day][hour] = rabs
                self.weather_profile.weather_solar_hourly[day][hour] = st

    def get_et(self, day_of_year):
        et_calc = (279.575 + 0.9856 * day_of_year) * math.pi / 180.0
        temp1 = (-104.7 * math.sin(et_calc) + 596.2 * math.sin(2.0 * et_calc) + 4.3 * math.sin(3.0 * et_calc)
                 - 12.7 * math.sin(4.0 * et_calc) - 429.3 * math.cos(et_calc) - 2.0 * math.cos(2.0 * et_calc)
                 + 19.3 * math.cos(3.0 * et_calc))
        return temp1 / 3600.0

    def get_lc(self, radian_lon):
        return radian_lon / 360.0 * 24.0

    def calc_half_day_length(self, solar_dec, radian_lat):
        temp3 = math.cos(math.pi / 2) - math.sin(radian_lat) * math.sin(solar_dec)
        temp3 /= math.cos(solar_dec) * math.cos(radian_lat)
        return math.acos(temp3) * 180.0 / math.pi / 15.0

    def get_solar_declination(self, day_of_year):
        temp2 = 278.97 + 0.9856 * day_of_year + 1.9165 * math.sin((356.6 + 0.9856 * day_of_year) * math.pi / 180.0)
        temp2 = math.sin(temp2 * math.pi / 180.0)
        return math.asin(0.39785 * temp2)

    def sb(self, airtemp):
        return 5.67E-8 * math.pow(airtemp + 273.16, 4.0)

    def zenith(self, radian_lat, solar_dec, solar_noon, hour):
        temp = (math.sin(radian_lat) * math.sin(solar_dec) + 
                math.cos(radian_lat) * math.cos(solar_dec) * 
                math.cos(15.0 * (hour - solar_noon) * math.pi / 180.0))
        return math.acos(temp)

    def sec_evap(self, max_evap, min_evap, day):
        for t in range(1441):
            self.weather_profile.time_evap[day][t] = max_evap * math.sin(t / 1440.0 * math.pi)
            if self.weather_profile.time_evap[day][t] < min_evap:
                self.weather_profile.time_evap[day][t] = min_evap

# # Example of how to use the WeatherModel
# site = Site(lat=40.7128, lon=-74.0060)  # Example lat/lon for New York City
# weather_profile = WeatherProfile()
# weather_model = WeatherModel(site=site, weather_profile=weather_profile)

# # Simulate weather data
# weather_model.simulate_weather()

# %%

@dataclass
class Cover:
    material: CoverMaterial
    site: Site
    weather_profile: WeatherProfile
    weather_model: WeatherModel

    def __post_init__(self):
        self.soil_density = self.material.bulk_density
        self.total_porosity = self.material.porosity33
        self.saturated_conductivity = self.material.saturated_conductivity
        self.temperature_vector = self.weather_model.weather_data[self.weather_model.avg_temp]
        self.rainfall_vector = self.weather_model.weather_data[self.weather_model.precip]
        self.temperature = np.mean(self.temperature_vector)
        self.rainfall = np.mean(self.rainfall_vector)

        self.water_potential = self.calculate_water_potential(self.material.moist33)
        #self.water_content = self.calculate_water_content(self.water_potential)
        self.o2_concentration = self.calculate_o2_concentration()

    def calculate_water_potential(self, moisture):
        ws = 1.0 - self.soil_density / 2.65
        b = self.material.beta
        pe = self.material.air_entry_potential
        water_potent = pe * (ws / moisture) ** b
        return min(water_potent, pe)

    def calculate_water_content(self, p):
        ws = 1.0 - self.soil_density / 2.65
        b = self.material.beta
        if p < self.material.air_entry_potential:
            b1 = 1.0 / b
            w = ws * (self.material.air_entry_potential / p) ** b1
        else:
            w = ws
        return w

    def calculate_o2_concentration(self):
        pres = 101.3
        r = 8.3143
        oxygen_lower = 5 # THIS NEEDS TO CHANGE BASED ON DAILY INT FINAL
        oxygen_upper = 20
        temp_lower = 25 # THIS TOO
        temp_upper = self.temperature
        # da = 1.77E-5 * (101.3 / pres) * math.pow(1.0 + temp / 273.16, 1.75)
        # fg = self.total_porosity - self.water_content
        # df = da * 0.9 * math.pow(fg, 1.7)
        # o2_conc = 100.0 * df * r * (temp + 273.16) / (pres * 1000.0 * 32.0)
        co_lower = oxygen_lower / 100 * pres * 1000 * 32 / (r * (temp_lower + 273.16))
        co_upper = oxygen_upper / 100 * pres * 1000 * 32 / (r * (temp_upper + 273.16))
        o2_conc_lower = 100 * co_lower * r * (temp_lower + 273.16) / (pres * 1000 * 32)
        o2_conc_upper = 100 * co_upper * r * (temp_upper + 273.16) / (pres * 1000 * 32)
        o2_conc = (o2_conc_lower + o2_conc_upper) / 2.0
        return o2_conc

    def calculate_oxidation_rate(self):
        reference_ch4oxrate = self.material.max_ox_rate
        #oxidation_rate = -1.0 * reference_ch4oxrate * self.soil_density / 86400.0 * 2.54
        oxidation_rate = reference_ch4oxrate * self.soil_density / 86400.0 * 2.54

        kurtfraction = (self.temperature - 27.6) / 9.59
        tempfrac = 1.05 * math.exp(-0.5 * kurtfraction * kurtfraction)

        calc1 = (self.water_potential + 754.1) / -151.84
        moistfrac = 0.852 / (1.0 + math.exp(calc1))

        o2correction = self.o2_concentration / 20.0
        frac = o2correction * tempfrac * moistfrac

        return oxidation_rate * frac
    
#%%

# # Example usage:
# site = Site(lat=40.7128, lon=-74.0060)  # New York City
# weather_profile = WeatherProfile()
# weather_model = WeatherModel(site=site, weather_profile=weather_profile)

# # Simulate weather data
# weather_model.simulate_weather()

# material = materials[0]

# material.calculate_properties()

# cover = Cover(material=material, site=site, weather_profile=weather_profile, weather_model=weather_model)
# oxidation_rate = cover.calculate_oxidation_rate()

# print(f"Oxidation Rate: {oxidation_rate}")

# Shut down the JVM after all operations that require it are completed.
# if jpype.isJVMStarted():
#     jpype.shutdownJVM()

#%%

