"""
Singapore k-value calculation using the method from Wang et al (2024).

https://doi.org/10.1038/s41893-024-01307-9
Lab of Professor Xunchang Fei, Nanyang Technological University
College of Engineering, Singapore.
"""

import numpy as np
import pandas as pd

from SWEET_python.class_defs import DecompositionRates


# Minimum decomposition rate (1/yr). IPCC (2019 refinement) Vol. 5 Table 3.3
# gives a bulk-MSW default of 0.05/yr for dry boreal/temperate conditions, with
# a range whose low end is 0.04/yr. We floor k here because the Wang et al.
# (2024) temperature factor drives k toward 0 for cold/dry sites (tf -> 0 at
# ambient -10 C), which produces physically implausible near-zero methane
# generation and an emissions factor that is hypersensitive to tiny temperature
# differences. 0.04 is the lowest defensible bulk default, so no real site
# should decay slower than this.
K_MIN_PER_YEAR = 0.04


def _build_lookup_array():
    """Build the 3D composition lookup array (bs, bf, nb) for kc values."""
    lookup_array = np.zeros((8, 8, 8))

    lookup_array[0, 0, 7] = 0.3  # lower left corner
    lookup_array[0, 0, 6] = 0.3  # this is all the bottom row
    lookup_array[1, 0, 6] = 0.3
    lookup_array[1, 0, 5] = 0.3
    lookup_array[2, 0, 6] = 0.3
    lookup_array[2, 0, 5] = 0.3
    lookup_array[2, 0, 4] = 0.3
    lookup_array[3, 0, 4] = 0.3
    lookup_array[3, 0, 3] = 0.3
    lookup_array[4, 0, 4] = 0.4
    lookup_array[4, 0, 3] = 0.5
    lookup_array[4, 0, 2] = 0.5
    lookup_array[5, 0, 2] = 0.5
    lookup_array[5, 0, 1] = 0.5
    lookup_array[6, 0, 2] = 0.3
    lookup_array[6, 0, 1] = 0.1
    lookup_array[6, 0, 0] = 0.1
    lookup_array[7, 0, 0] = 0.1  # lower right corner

    lookup_array[0, 1, 6] = 0.3  # second row from bottom
    lookup_array[0, 1, 5] = 0.3
    lookup_array[1, 1, 5] = 0.3
    lookup_array[1, 1, 4] = 0.3
    lookup_array[2, 1, 4] = 0.3
    lookup_array[2, 1, 3] = 0.3
    lookup_array[3, 1, 3] = 0.3
    lookup_array[3, 1, 2] = 0.3
    lookup_array[4, 1, 2] = 0.5
    lookup_array[4, 1, 1] = 0.1
    lookup_array[5, 1, 1] = 0.1
    lookup_array[5, 1, 0] = 0.1
    lookup_array[6, 1, 0] = 0.1

    lookup_array[0, 2, 6] = 0.3
    lookup_array[0, 2, 5] = 0.3
    lookup_array[0, 2, 4] = 0.3
    lookup_array[1, 2, 4] = 0.3
    lookup_array[1, 2, 3] = 0.3
    lookup_array[2, 2, 4] = 0.5
    lookup_array[2, 2, 3] = 0.7
    lookup_array[2, 2, 2] = 0.7
    lookup_array[3, 2, 2] = 0.7
    lookup_array[3, 2, 1] = 0.7
    lookup_array[4, 2, 2] = 0.5
    lookup_array[4, 2, 1] = 0.1
    lookup_array[4, 2, 0] = 0.1
    lookup_array[5, 2, 0] = 0.1
    lookup_array[6, 2, 0] = 0.1

    lookup_array[0, 3, 4] = 0.3
    lookup_array[0, 3, 3] = 0.3
    lookup_array[1, 3, 3] = 0.3
    lookup_array[1, 3, 2] = 0.3
    lookup_array[2, 3, 2] = 0.7
    lookup_array[2, 3, 1] = 0.7
    lookup_array[3, 3, 1] = 0.7
    lookup_array[3, 3, 0] = 0.7
    lookup_array[4, 3, 0] = 0.1

    lookup_array[0, 4, 4] = 0.3
    lookup_array[0, 4, 3] = 0.3
    lookup_array[0, 4, 2] = 0.3
    lookup_array[1, 4, 2] = 0.3
    lookup_array[1, 4, 1] = 0.5
    lookup_array[2, 4, 2] = 0.5
    lookup_array[2, 4, 1] = 0.5
    lookup_array[2, 4, 0] = 0.5
    lookup_array[3, 4, 0] = 0.5
    lookup_array[4, 4, 0] = 0.5

    lookup_array[0, 5, 3] = 0.7  # doesn't exist i think
    lookup_array[0, 5, 2] = 0.7
    lookup_array[0, 5, 1] = 0.7
    lookup_array[1, 5, 1] = 0.7
    lookup_array[1, 5, 0] = 0.7
    lookup_array[2, 5, 0] = 0.5

    # Placeholder value; confirm whether this case needs a real lookup.
    lookup_array[1, 5, 2] = 0.5

    lookup_array[0, 6, 2] = 0.6
    lookup_array[0, 6, 1] = 0.5
    lookup_array[0, 6, 0] = 0.5
    lookup_array[1, 6, 0] = 0.5
    lookup_array[2, 6, 0] = 0.6

    lookup_array[0, 7, 0] = 0.5

    return lookup_array


def _compute_kc(waste_fractions, lookup_array, advanced_baseline, advanced_dst,
                for_trace_reported_projections):
    """Compute the composition factor kc and biofraction bf from waste fractions."""
    if for_trace_reported_projections:
        nb = (
            waste_fractions["metal"]
            + waste_fractions["glass"]
            + waste_fractions["plastic"]
            + waste_fractions["other"]
            + waste_fractions["rubber"]
        )
        bs = (
            waste_fractions["wood"]
            + waste_fractions["paper_cardboard"]
            + waste_fractions["textiles"]
        )
        bf = (
            waste_fractions["food"]
            + waste_fractions["green"]
        )

        bs_idx = int(bs * 8)
        bf_idx = int(bf * 8)
        nb_idx = int(nb * 8)

        if nb_idx == 8:
            nb_idx = 7
        if bs_idx == 8:
            bs_idx = 7
        if bf_idx == 8:
            bf_idx = 7

        kc = lookup_array[bs_idx, bf_idx, nb_idx]
        if kc == 0:
            print("Invalid value for k")

    elif advanced_dst:
        nb = {}
        bs = {}
        bf = {}

        nb["baseline"] = (
            waste_fractions["baseline"].metal
            + waste_fractions["baseline"].glass
            + waste_fractions["baseline"].plastic
            + waste_fractions["baseline"].other
            + waste_fractions["baseline"].rubber
        )
        bs["baseline"] = (
            waste_fractions["baseline"].wood
            + waste_fractions["baseline"].paper_cardboard
            + waste_fractions["baseline"].textiles
        )
        bf["baseline"] = (
            waste_fractions["baseline"].food
            + waste_fractions["baseline"].green
        )

        nb["scenario"] = (
            waste_fractions["scenario"].metal
            + waste_fractions["scenario"].glass
            + waste_fractions["scenario"].plastic
            + waste_fractions["scenario"].other
            + waste_fractions["scenario"].rubber
        )
        bs["scenario"] = (
            waste_fractions["scenario"].wood
            + waste_fractions["scenario"].paper_cardboard
            + waste_fractions["scenario"].textiles
        )
        bf["scenario"] = (
            waste_fractions["scenario"].food
            + waste_fractions["scenario"].green
        )

        bs_idx = {}
        bf_idx = {}
        nb_idx = {}

        bs_idx["baseline"] = int(bs["baseline"] * 8)
        bf_idx["baseline"] = int(bf["baseline"] * 8)
        nb_idx["baseline"] = int(nb["baseline"] * 8)

        bs_idx["scenario"] = int(bs["scenario"] * 8)
        bf_idx["scenario"] = int(bf["scenario"] * 8)
        nb_idx["scenario"] = int(nb["scenario"] * 8)

        if nb_idx["baseline"] == 8:
            nb_idx["baseline"] = 7
        if bs_idx["baseline"] == 8:
            bs_idx["baseline"] = 7
        if bf_idx["baseline"] == 8:
            bf_idx["baseline"] = 7

        if nb_idx["scenario"] == 8:
            nb_idx["scenario"] = 7
        if bs_idx["scenario"] == 8:
            bs_idx["scenario"] = 7
        if bf_idx["scenario"] == 8:
            bf_idx["scenario"] = 7

        kc = {}
        kc["baseline"] = lookup_array[
            bs_idx["baseline"], bf_idx["baseline"], nb_idx["baseline"]
        ]
        if kc["baseline"] == 0.0:
            print("Invalid value for k")

        kc["scenario"] = lookup_array[
            bs_idx["scenario"], bf_idx["scenario"], nb_idx["scenario"]
        ]
        if kc["scenario"] == 0.0:
            print("Invalid value for k")

    elif advanced_baseline:
        nb = {}
        bs = {}
        bf = {}

        nb = (
            waste_fractions.at[2000, "metal"]
            + waste_fractions.at[2000, "glass"]
            + waste_fractions.at[2000, "plastic"]
            + waste_fractions.at[2000, "other"]
        )
        bs = (
            waste_fractions.at[2000, "wood"]
            + waste_fractions.at[2000, "paper_cardboard"]
            + waste_fractions.at[2000, "textiles"]
        )
        bf = (
            waste_fractions.at[2000, "food"]
            + waste_fractions.at[2000, "green"]
        )

        bs_idx = {}
        bf_idx = {}
        nb_idx = {}

        bs_idx = int(bs * 8)
        bf_idx = int(bf * 8)
        nb_idx = int(nb * 8)

        if nb_idx == 8:
            nb_idx = 7
        if bs_idx == 8:
            bs_idx = 7
        if bf_idx == 8:
            bf_idx = 7

        kc = {}
        kc = lookup_array[bs_idx, bf_idx, nb_idx]
        if kc == 0:
            print("Invalid value for k")

    else:
        nb = (
            waste_fractions.at[2025, "metal"]
            + waste_fractions.at[2025, "glass"]
            + waste_fractions.at[2025, "plastic"]
            + waste_fractions.at[2025, "other"]
            + waste_fractions.at[2025, "rubber"]
        )
        bs = (
            waste_fractions.at[2025, "wood"]
            + waste_fractions.at[2025, "paper_cardboard"]
            + waste_fractions.at[2025, "textiles"]
        )
        bf = (
            waste_fractions.at[2025, "food"]
            + waste_fractions.at[2025, "green"]
        )

        bs_idx = int(bs * 8)
        bf_idx = int(bf * 8)
        nb_idx = int(nb * 8)

        if nb_idx == 8:
            nb_idx = 7
        if bs_idx == 8:
            bs_idx = 7
        if bf_idx == 8:
            bf_idx = 7

        kc = lookup_array[bs_idx, bf_idx, nb_idx]
        if kc == 0:
            print("Invalid value for k")

    return kc, bf


def _compute_temperature_factor(temperature):
    """Compute the temperature factor (tf) for decomposition rate."""
    tmin = 0
    tmax = 55
    topt = 35
    # The Ratkowsky response is only fitted for landfill temperatures in
    # [tmin, tmax]. Outside that range it extrapolates nonsensically: below
    # tmin the squared (t - tmin) term makes tf spuriously positive AND
    # *increasing* as the site gets colder (colder modeled as faster decay).
    # Clamp the input to the valid domain so we evaluate at the nearest
    # boundary instead of extrapolating; with t = temperature + 10 this is
    # equivalent to clamping ambient temperature to [-10 C, 45 C]. tf is 0 at
    # both boundaries, and the k floor (K_MIN_PER_YEAR) downstream then keeps
    # cold sites physical. np.clip (unlike max/min) preserves NaN, so a missing
    # temperature stays NaN -> NaN k rather than being silently floored; the
    # caller (_singapore_k) logs which asset is missing a temperature.
    t = temperature + 10  # landfill is warmer than ambient
    t = np.clip(t, tmin, tmax)

    num = (t - tmax) * (t - tmin) ** 2
    denom = (topt - tmin) * (
        (topt - tmin) * (t - topt) - (topt - tmax) * (topt + tmin - 2 * t)
    )

    if denom != 0:
        tf = num / denom
    else:
        print("Invalid value for temperature factor")
        tf = 0.0

    return float(tf)


def _compute_moisture_factor(precip):
    """Compute the moisture factor (fm) based on precipitation."""
    # read more on this to make sure it handles dumpsites correctly.
    if precip < 500:
        fm = 0.1
    elif precip >= 500 and precip < 1000:
        fm = 0.3
    elif precip >= 1000 and precip < 1500:
        fm = 0.5
    elif precip >= 1500 and precip < 2000:
        fm = 0.8
    else:
        fm = 1

    return fm


def _create_series(kc, tf, fm, implement_year=None, advanced_baseline=False,
                   advanced_dst=False):
    """Create a time series of k values from 1990 to 2050."""
    years = pd.Series(index=range(1990, 2051))
    if advanced_dst:
        baseline_series = kc["baseline"] * tf * fm
        years.loc[: implement_year - 1] = baseline_series

        scenario_series = kc["scenario"] * tf * fm
        years.loc[implement_year:] = scenario_series
    elif advanced_baseline:
        baseline_series = kc * tf * fm
        years.loc[:] = baseline_series
    else:
        baseline_series = kc * tf * fm
        years.loc[:] = baseline_series

    # Floor the decomposition rate at the IPCC dry-boreal low-end default.
    # Cold/dry sites can otherwise drive k toward 0 (tf -> 0 near ambient
    # -10 C); clip leaves any genuine NaN entries untouched. See K_MIN_PER_YEAR.
    years = years.clip(lower=K_MIN_PER_YEAR)

    return years


def compute_singapore_k(waste_fractions, temperature, precip,
                        advanced_baseline=False, advanced_dst=False,
                        implement_year=None,
                        for_trace_reported_projections=False):
    """
    Calculate k values using the method from Wang et al (2024).

    https://doi.org/10.1038/s41893-024-01307-9
    Lab of Professor Xunchang Fei, Nanyang Technological University
    College of Engineering, Singapore.

    Args:
        waste_fractions: Waste fraction data (DataFrame, Series, or dict
            depending on the calculation mode).
        temperature (float): Ambient temperature in degrees Celsius.
        precip (float): Annual precipitation in mm.
        advanced_baseline (bool): Flag to indicate if advanced baseline
            calculations are needed.
        advanced_dst (bool): Flag to indicate if advanced diversion scenario
            calculations are needed.
        implement_year (int): Year when the diversion scenario is implemented.
        for_trace_reported_projections (bool): Flag for trace reported
            projection calculations.

    Returns:
        tuple: (ks, bf) where ks is a DecompositionRates instance and bf is
            the biofraction value.
    """
    lookup_array = _build_lookup_array()

    kc, bf = _compute_kc(
        waste_fractions, lookup_array, advanced_baseline, advanced_dst,
        for_trace_reported_projections,
    )

    tf = _compute_temperature_factor(temperature)
    fm = _compute_moisture_factor(precip)

    if advanced_dst or advanced_baseline:
        vals = _create_series(
            kc, tf, fm, implement_year,
            advanced_baseline=advanced_baseline,
            advanced_dst=advanced_dst,
        )
    else:
        vals = _create_series(kc, tf, fm)

    ks = DecompositionRates(
        food=vals, green=vals, wood=vals, paper_cardboard=vals, textiles=vals
    )

    return ks, bf
