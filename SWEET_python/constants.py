"""Canonical modeling window shared across SWEET_python, the Climate TRACE
waste methane pipeline, and the WasteMAP backend.

Waste deposited before MODEL_START_YEAR is assumed to be zero everywhere:
sites with earlier reported opening years keep their true opening year in
source data, but models treat deposition as starting in MODEL_START_YEAR.
Export and display windows (Climate TRACE submissions, WasteMAP charts) are
filters over this window, never separate modeling horizons.
"""

MODEL_START_YEAR: int = 1990
MODEL_END_YEAR: int = 2050
