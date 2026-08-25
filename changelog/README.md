# SWEET_python Changelog

All notable changes to **SWEET_python** are documented here. SWEET_python is a
Python port of the methane-emissions portions of the US EPA's Solid Waste
Emissions Estimation Tool (SWEET) — a first-order-decay model of landfill
emissions. It is the modeling library that the WasteMAP backend API depends on,
and this folder tracks notable changes to it.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The project does not publish semantic version tags, so releases are tracked by
**calendar month**: each entry file covers a one-month span and is named
`YYYY-MM.md`.

## Entries

Newest first:

- [2026-08](2026-08.md) — Single-site adst gains optional `depth` (deep-dump MCF bump) and `k_override` (caller-supplied decomposition rate) inputs, restoring the last two site-DST levers; `/sdst` flaring efficiency reaches the model again after a variable-name bug silently forced flare destruction to 0.98; annual model applies cover oxidation by emission year not deposit year, fixing biocover having no effect on closed landfills (WasteMAP #719); `City.sdst_v1_5` custom-site path holds the scenario equal to the baseline before the implementation year even when composition changes (was back-dating the new composition onto pre-implementation deposits) (model-output change); MCF consolidated into a new `SWEET_python.mcf` module and both dump types moved to the IPCC uncategorised-SWDS 0.6 (open dumps up from 0.4, controlled dumps down from 0.7), with a supplied depth now selecting the deep/shallow category (model-output change)
- [2026-07](2026-07.md) — All ten waste types eligible for combustion (metal/glass/other added); methane-only model treats combustion as landfill diversion (model-output change)
- [2026-06](2026-06.md) — New single-site and city-level ADST modeling modules, min-cost max-flow rewrite of the city DST diversion allocator, physical-k fix for cold/dry sites, no more spurious negative food-waste mass
- [2026-05](2026-05.md) — SDST models from a landfill's actual open year (1950–2050), Central Asia/Afghanistan disposal-default fix, auto-Jira issue tooling, professional-comment cleanup
- [2026-04](2026-04.md) — SDST baseline/scenario oxidation hardening against pandas dtype bugs, new PM2.5/PM10 particulate emissions from flared methane
- [2026-03](2026-03.md) — TRACE pipeline integration for the City model: caller-supplied weather, weather-lookup retries, waste-window/reference-year fixes
- [2026-02](2026-02.md) — Singapore (Wang et al. 2024) k-value subpackage, hardened data-file/credential loading for Climate TRACE and Azure, pd.NA gas-capture crash fix
- [2026-01](2026-01.md) — City-parameter loaders realigned with TRACE data, advanced DST per-scenario growth, biocover oxidation floors, new-landfill open/close windowing

## Categories

Each entry groups changes under the standard Keep a Changelog headings. Only
headings with content are shown:

- **Added** — new features, models, modules, or capabilities.
- **Changed** — changes to existing behaviour, modeling, or structure.
- **Deprecated** — features still present but slated for removal.
- **Removed** — features, dependencies, or artifacts that were deleted.
- **Fixed** — bug fixes.
- **Security** — vulnerability fixes or hardening.

## Conventions

- Each bullet links to the pull request (`#NN`) or, where no PR exists, the
  commit it came from.
- Jira ticket references (e.g. `WP-NNN`) are preserved where they appear in the
  source history.
- Entries reference the public repository at
  <https://github.com/RMI/SWEET_python>.

## Adding a new entry

1. If a file for the current month does not exist, copy the heading structure
   from the most recent month and name it `YYYY-MM.md`.
2. Add your change under the appropriate category heading, newest items at the
   top, with a link to the merging PR.
3. Add (or update) the month's line in the **Entries** list above.

> **Note:** The entries for January–June 2026 were reconstructed from the git
> history of the repository and may be less granular than entries written at
> merge time. Going forward, update the relevant month's file as part of each
> PR.
