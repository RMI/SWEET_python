"""
This module provides an implementation of the SWEET model for calculating methane emissions from municipal solid waste,
based on the EPA's SWEET excel model.

The model takes into account waste generation, diversion to various processes (e.g., composting, anaerobic digestion,
combustion/incineration, recycling), biodegradation rates influenced by precipitation, and landfill methane capture,
to estimate annual methane emissions.

Workflow:
1. Instantiate a `City` object, which encapsulates city-specific parameters and methods.
2. Within the `City` class, instantiate one or more `Landfill` objects
3. For each `Landfill` instance, instantiate the `SWEET` class.
4. Call the `estimate_emissions` method of the `SWEET` class to run the model and retrieve the results.

Main components:
- `SWEET` class: Represents the main model and its operations.

Dependencies:
- pandas
- numpy
- SWEET_python.defaults

Authors: Hugh Runyan, Andre Scheinwald
Date: Sep 2023
Version: 0.1
"""

import pandas as pd
import numpy as np
import time
import calendar
import SWEET_python.defaults_2019 as defaults_2019
pd.set_option("display.max_rows", None)


# Based on EPA's SWEET excel model for calculating methane emissions from municipal solid waste
# (https://globalmethane.org/resources/details.aspx?resourceid=5176)

"""
The way this type of model works is: data inputs determine how much waste is generated annually and of what types,
and how much waste is diverted to composting, anaerobic digestion, combustion/incineration, and recycling facilities
rather than being landfilled. Population growth rates in the past and future are used to estimate how waste generation
changes over time. Research-backed default parameters are used to estimate how much biodegradable matter the different
types of waste contain. Different types of waste also biodegrade at different rates, and these rates are influenced by
average annual precipitation. In the first year of the model, the waste that is not diverted is added to the landfill.
In the second year, the waste from the first year has biodegraded to some extent, creating methane, and new generated
waste is added to the landfill. In the third year, waste from the first year biodegrades another year, and waste from 
the second year does its first year of biodegradation, and new generated waste is added. This process continues until
the final year of the model. 

The model outputs how much methane is produced each year from the cumulative degradation of wastes of different types and
ages. This amount is reduced by the amount of methane captured at the landfill.

Unit: m3 CH4/year
"""


class SWEET:
    def __init__(
        self,
        city_instance_attrs: dict,
        city_params_dict: dict,
        landfill_instance_attrs: dict,
    ):
        self.landfill_instance_attrs = landfill_instance_attrs
        self.city_instance_attrs = city_instance_attrs
        self.city_params_dict = city_params_dict

    def estimate_emissions2(self):
        start_time = time.time()

        open_date = self.landfill_instance_attrs["open_date"]
        close_date = self.landfill_instance_attrs["close_date"]
        year_of_data_pop = self.city_params_dict["year_of_data_pop"]
        growth_rate_historic = self.city_params_dict["growth_rate_historic"]
        growth_rate_future = self.city_params_dict["growth_rate_future"]
        ks = self.landfill_instance_attrs["ks"]
        waste_mass_df = self.landfill_instance_attrs["waste_mass_df"]
        mcf = self.landfill_instance_attrs["mcf"]
        gas_capture_efficiency = self.landfill_instance_attrs["gas_capture_efficiency"]
        oxidation_factor = self.landfill_instance_attrs["oxidation_factor"]
        components = self.city_instance_attrs["components"]
        flare_efficiency = self.landfill_instance_attrs["flaring"]

        # Precompute factors outside of the loop for all years
        year_range = np.arange(open_date, 2074)
        if flare_efficiency is None:
            flare_efficiency = pd.Series([1 for x in year_range], index=year_range)
        elif isinstance(flare_efficiency, dict):
            flare_efficiency = pd.Series(flare_efficiency)

        qs = {}
        ch4_produced = {}
        captured = {}
        waste_in_place_dict = {}

        end_time = time.time()
        # print(f"Model setup: {end_time - start_time} seconds")

        start_time = time.time()

        # Vectorized calculation for each component
        for waste in components:
            years_back_matrix = (
                year_range[None, :] - year_range[:, None]
            )  # Matrix of (years - year_range)
            mask = years_back_matrix <= 0

            # Precompute exponential decay term for all years at once
            ks_values = ks[waste].loc[year_range].values[:, None]
            exp_term = np.exp(-ks_values * (years_back_matrix - 0.5))

            # FIXED: Vectorized waste mass, L_0, and MCF computation
            waste_masses = waste_mass_df.loc[open_date:, waste].values[:, None]  # FIXED: removed reference to undefined 'year'
            mcf_values = mcf.loc[year_range].values[:, None]
            ch4_produce = (
                ks_values
                * defaults_2019.L_0[waste]
                * waste_masses
                * exp_term
                * mcf_values
            )
            ch4_produce[mask] = 0
            ch4_produced[waste] = ch4_produce.sum(axis=0)

            waste_in_place = waste_masses * exp_term
            waste_in_place[mask] = 0
            waste_in_place_total = waste_in_place.sum(axis=0)

            # Gas capture and oxidation factor
            if isinstance(gas_capture_efficiency, pd.Series):
                gas_capture_efficiency_values = gas_capture_efficiency.loc[
                    year_range
                ].values
            else:
                gas_capture_efficiency_values = np.full(
                    len(year_range), gas_capture_efficiency
                )

            if isinstance(oxidation_factor, pd.Series):
                oxidation_factor_values = oxidation_factor.loc[year_range].values
            else:
                oxidation_factor_values = np.full(len(year_range), oxidation_factor)

            ch4_capture = (
                ch4_produce
                * gas_capture_efficiency_values
                * flare_efficiency.loc[year_range].values
            )

            # Final methane emissions calculation with oxidation
            ch4_year_total = np.sum(
                (ch4_produce - ch4_capture) * (1 - oxidation_factor_values[:, None])
                + ch4_capture * 0.02,
                axis=0,
            )

            # Store results for the current waste component
            qs[waste] = ch4_year_total

            # Total methane captured for each year
            captured_total = (
                ch4_produced[waste] * gas_capture_efficiency_values
            )  # / 365 / 24
            captured[waste] = captured_total
            waste_in_place_dict[waste] = waste_in_place_total

        end_time = time.time()
        # print(f"Model run: {end_time - start_time} seconds")

        start_time = time.time()

        # Convert results to DataFrames
        q_df = pd.DataFrame(qs, index=year_range)
        q_df["total"] = q_df.sum(axis=1)
        ch4_df = pd.DataFrame(ch4_produced, index=year_range)
        captured_df = pd.DataFrame(captured, index=year_range)
        waste_in_place_df = pd.DataFrame(waste_in_place_dict, index=year_range)

        end_time = time.time()
        # print(f"Model post-processing: {end_time - start_time} seconds")

        q_df.to_csv("q_df.csv")

        return waste_in_place_df, q_df, ch4_df, captured_df
    

    def estimate_emissions_monthly(self):
        open_date = self.landfill_instance_attrs["open_date"] or 1990
        close_date = self.landfill_instance_attrs["close_date"]
        ks = self.landfill_instance_attrs["ks"]
        waste_mass_df = self.landfill_instance_attrs["waste_mass_df"]
        mcf = self.landfill_instance_attrs["mcf"]
        gas_capture_efficiency = self.landfill_instance_attrs["gas_capture_efficiency"]
        oxidation_factor = self.landfill_instance_attrs["oxidation_factor"]
        components = list(self.city_instance_attrs["components"])
        flare_efficiency = self.landfill_instance_attrs["flaring"]

        # Monthly date range
        start_date = pd.Timestamp(f'{int(open_date)}-01-01')
        end_date = pd.Timestamp('2073-12-31')
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        n_months = len(monthly_dates)
        years = monthly_dates.year
        # Month length in years (variable by month and leap years)
        month_days = monthly_dates.days_in_month.values
        year_days = np.array([366 if calendar.isleap(y) else 365 for y in years])
        month_fracs = month_days / year_days

        # Expand annual factors to monthly arrays
        def expand_to_months(annual_series_or_value):
            # Supports dict (year->value), pd.Series (indexed by year), or scalar
            if isinstance(annual_series_or_value, dict):
                annual_series_or_value = pd.Series(annual_series_or_value)
            if isinstance(annual_series_or_value, pd.Series):
                return annual_series_or_value.reindex(years, fill_value=0).values
            else:
                return np.full(n_months, annual_series_or_value)

        gas_capture_values = expand_to_months(gas_capture_efficiency)
        oxidation_values = expand_to_months(oxidation_factor)
        flare_values = expand_to_months(flare_efficiency if flare_efficiency is not None else 0.98)

        # Setup mask for valid emission-addition month pairs (emit month >= add month)
        mask_valid = np.tri(n_months, n_months, k=0, dtype=bool)

        # Prepare result arrays
        ch4_df = np.zeros((n_months, len(components)))
        waste_in_place_df = np.zeros_like(ch4_df)
        captured_df = np.zeros_like(ch4_df)
        emissions_df = np.zeros_like(ch4_df)

        for i, waste in enumerate(components):
            # Get constants
            L0 = defaults_2019.L_0[waste]

            # Monthly series
            annual_mass = waste_mass_df.get(waste, pd.Series(0, index=waste_mass_df.index)) \
                .reindex(years, fill_value=0).values
            # Distribute annual mass by actual days per month (sums to annual)
            waste_mass_monthly = annual_mass * month_fracs
            ks_monthly = ks[waste].reindex(years, fill_value=0).values
            mcf_monthly = mcf.reindex(years, fill_value=0).values

            # Prepare decay using integral of k over variable month lengths
            cum_k_ext = np.concatenate(([0.0], np.cumsum(ks_monthly * month_fracs)))
            # Use cumulative up to the start of each month for emit/add so i==j yields zero integral
            cum_emit = cum_k_ext[:-1]
            cum_add = cum_k_ext[:-1]
            integral = cum_emit[:, None] - cum_add[None, :]
            decay = np.exp(-integral)
            decay[~mask_valid] = 0

            # Vectorized CH4 production (emission-time k)
            waste_input = waste_mass_monthly[None, :]
            k_emit = ks_monthly[:, None]
            mcf_matrix = mcf_monthly[:, None]
            dt_emit = month_fracs[:, None]
            ch4_matrix = (L0 * waste_input * decay) * (k_emit * dt_emit) * mcf_matrix
            ch4_total = ch4_matrix.sum(axis=1)

            # Vectorized waste in place
            waste_matrix = waste_input * decay
            wip_total = waste_matrix.sum(axis=1)

            # Capture and emissions
            capture = ch4_total * gas_capture_values * flare_values
            emissions = (ch4_total - capture) * (1 - oxidation_values)

            # Store in result arrays
            ch4_df[:, i] = ch4_total
            waste_in_place_df[:, i] = wip_total
            captured_df[:, i] = capture
            emissions_df[:, i] = emissions

        # Wrap up as DataFrames
        columns = components
        index = monthly_dates
        ch4_df = pd.DataFrame(ch4_df, columns=columns, index=index)
        waste_in_place_df = pd.DataFrame(waste_in_place_df, columns=columns, index=index)
        captured_df = pd.DataFrame(captured_df, columns=columns, index=index)
        q_df = pd.DataFrame(emissions_df, columns=columns, index=index)
        q_df["total"] = q_df.sum(axis=1)

        q_df.to_csv("q_df.csv")

        return waste_in_place_df, q_df, ch4_df, captured_df
