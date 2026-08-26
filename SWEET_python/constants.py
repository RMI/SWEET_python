"""Canonical modeling window shared across SWEET_python, the Climate TRACE
waste methane pipeline, and the WasteMAP backend.

Waste deposited before MODEL_START_YEAR is assumed to be zero everywhere:
sites with earlier reported opening years keep their true opening year in
source data, but models treat deposition as starting in MODEL_START_YEAR.
Export and display windows (Climate TRACE submissions, WasteMAP charts) are
filters over this window, never separate modeling horizons.

MODEL_START_YEAR moved 1990 -> 1970 on 2026-08-26. The cutoff is a truncation of
the decay tail, not a neutral choice: methane emitted today comes from decades of
accumulated stock, so zeroing deposition before the cutoff understates every site
with a long landfilling history -- most severely in cold/dry climates, where the
IPCC k values are lowest and the tail is longest. Measured on the 08_24_26 run,
Russia deposited as much municipal waste before 1990 as it did 1990-2021, and
restoring the earlier stock raises its national FOD by ~20%. 1970 is chosen over
1950 because the population series backing the waste projection (WPP2024, via
pops_yearly.csv) is credible per-year that far back while per-capita generation
before ~1970 is not, and because the residual tail before 1970 is small at every
k in defaults_2019.

CHANGING THIS CONSTANT REQUIRES A MATCHING pops_yearly.csv. The waste series is
population-driven, and city_params._population_series_from_pop_data returns None
unless the table carries EVERY column from MODEL_START_YEAR onward -- which
silently drops every country back to the frozen-CAGR growth scalars. Regenerate
with diagnostic_scripts/generate_pops_yearly.py and upload to blob
static_data/pops_yearly.csv BEFORE the constant lands in a run.
"""

MODEL_START_YEAR: int = 1970
MODEL_END_YEAR: int = 2050
