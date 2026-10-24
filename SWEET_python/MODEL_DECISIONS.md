# SWEET's model decisions

Status: started 2026-10-22, from the SWEET entries in Climate TRACE's list; extended 2026-10-23 after a
review of the code. `tests/test_model_decisions.py` checks that each entry's comments, fields and dates
agree with the code. It cannot check what the prose claims; a reviewer does that.

This is the list of choices SWEET makes that change a number a tool built on it reports, where someone
could reasonably have chosen differently. Government modelling guidance calls this an assumptions log;
HM Treasury's Aqua Book is the usual reference. The format follows Climate TRACE's
[`docs/MODEL_DECISIONS.md`](https://github.com/RMI/RMI_Climate_TRACE_Waste_Methane/blob/main/docs/MODEL_DECISIONS.md),
so the lists read alike.

**Who keeps what.** SWEET serves several tools, and they do not all make the same choice, so each entry
says which **paths** apply it and how. SWEET says what a decision is, why, and how sure we are. Each app
that uses SWEET keeps a list of its own decisions, links to the entries here that its paths reach, and
rates how much each moves its own numbers. That is why entries here carry no impact rating.

**This file ships with the package**, so an app reads the list of the release it pins:
`SWEET_python.model_decisions.load()`.

**Which branch.** This list describes `main`. The advanced DST's development branch, `am-waste-flow`,
adds per-site decay and oxidation overrides, lets a capture year be blank (the type's default), routes
a city's unassigned waste to "other landfills" (#90) and models a site's outside waste (#65), all in
`adst-city`. It also still sets `combustion_reject_rate` to 0.1. Those notes join the entries when that
branch reaches `main`.

Where the code applies a decision, a one-line comment names it:

```python
# decision: mcf-by-type-and-depth
```

## How to read an entry

- **The heading** is the entry's id. An id is never renamed and never reused. Apps link to it.
- **The sentence under it** says what SWEET does.
- **Kind** is one of four: *fallback* (try A, then B), *fixed value* (a number we chose),
  *simplification* (we do the simple thing, and a better version is known) or *method choice*
  (one approach over another defensible one).
- **Why** gives the reason in one sentence.
- **Options, best first** is for fallbacks only.
- **Paths** names the entry points that apply it and what each does. A path not named does not apply
  it. The paths are:

  | Path | Entry point | Used by |
  |---|---|---|
  | `city-model` | `City.load_csv_new`, `City.load_andre_params` | WasteMAP's cities table and city DST baseline |
  | `city-dst-scenario` | `City.implement_dst_changes_simple*` | WasteMAP's city DST scenarios |
  | `custom-location` | `City.dst_baseline_blank` | WasteMAP's Custom Location |
  | `site-tool` | `City.sdst_v1_5`, `City.sdst_prepopulate` | WasteMAP's single-site tool |
  | `ct-site` | `City.site_only_estimate_trace`, `City.citysite_estimate_trace` | Climate TRACE's site estimates |
  | `adst-site` | `advanced_dst.run_advanced_dst` | the single-site advanced DST |
  | `adst-city` | `advanced_dst_city.run_advanced_dst_city` | WasteMAP's advanced DST |

- **Confidence** is how sure we are the choice is right: *high*, backed by measurement or a published
  standard used as intended; *medium*, reasoned but not measured; *low*, a placeholder or a guess.
- **Better** says what a more careful version would do.
- **Value** names the setting that holds the number, and says where the number is only a literal.
- **Known problem** names a confirmed bug that makes the code do something other than the decision. The
  bug gets fixed; the note stays until it is.
- **Code** lists the files that carry the comment, with the number of places when it is more than one.
- **Reviewed** is the date someone last checked the entry against the code.
- **Docs** links longer reasoning, or a measured effect.

## Rules

- **In:** anything SWEET chooses that changes a reported number: fallbacks, thresholds, default values,
  simplifications, and choices between two methods. **Out:** plumbing, and bugs, which get fixed rather
  than recorded; a confirmed bug in a decision's code is noted under Known problem until it is fixed.
- **A pull request that changes a decision updates its entry in the same pull request**, and the
  changelog says how much the numbers move. The pull-request template asks.
- **An app's list links here** rather than copying an entry. When an app's own code applies a SWEET
  decision a further way, that part is the app's decision, in the app's list.
- **A replaced decision moves to "Retired"** with the date and the entry that replaced it, and is never
  deleted.
- **Text stays under 40 words per field.** The test enforces this for the sentence, Why, Better, Known
  problem, each option and each path.

## At a glance

| Decision | Section | Kind | Confidence |
|---|---|---|---|
| [`site-type-fill`](#site-type-fill) | Inputs | fallback | low |
| [`waste-composition`](#waste-composition) | Inputs | fallback | medium |
| [`all-city-waste-is-collected`](#all-city-waste-is-collected) | Inputs | simplification | low |
| [`population-growth`](#population-growth) | Inputs | fallback | medium |
| [`intake-follows-population`](#intake-follows-population) | Inputs | simplification | low |
| [`waste-before-1970-is-zero`](#waste-before-1970-is-zero) | Inputs | simplification | medium |
| [`intake-ends-before-close-year`](#intake-ends-before-close-year) | Inputs | simplification | medium |
| [`waste-per-capita-default`](#waste-per-capita-default) | Inputs | fallback | low |
| [`city-shares-constant`](#city-shares-constant) | Inputs | simplification | low |
| [`open-year-unknown`](#open-year-unknown) | Inputs | fallback | low |
| [`site-tool-prefill-defaults`](#site-tool-prefill-defaults) | Inputs | fixed value | low |
| [`multi-city-site-split`](#multi-city-site-split) | Inputs | method choice | low |
| [`combustion-reject`](#combustion-reject) | Diversion | fixed value | medium |
| [`other-treatment-shares-landfilled`](#other-treatment-shares-landfilled) | Diversion | simplification | low |
| [`diversion-defaults`](#diversion-defaults) | Diversion | fallback | low |
| [`city-treatment-buckets`](#city-treatment-buckets) | Diversion | method choice | low |
| [`diversion-eligibility`](#diversion-eligibility) | Diversion | fixed value | medium |
| [`diversion-rejects`](#diversion-rejects) | Diversion | fixed value | low |
| [`diversion-emission-factors`](#diversion-emission-factors) | Diversion | fixed value | low |
| [`diversion-split`](#diversion-split) | Diversion | method choice | medium |
| [`food-waste-prevention`](#food-waste-prevention) | Diversion | method choice | medium |
| [`mcf-by-type-and-depth`](#mcf-by-type-and-depth) | Decay model | fixed value | medium |
| [`decay-rate-from-climate`](#decay-rate-from-climate) | Decay model | method choice | medium |
| [`methane-potential-by-waste-type`](#methane-potential-by-waste-type) | Decay model | fixed value | high |
| [`oxidation`](#oxidation) | Decay model | fallback | medium |
| [`gas-capture-efficiency`](#gas-capture-efficiency) | Decay model | fallback | medium |
| [`flaring-uniform`](#flaring-uniform) | Decay model | simplification | medium |
| [`monthly-from-decay-model`](#monthly-from-decay-model) | Decay model | simplification | medium |
| [`inert-waste-types`](#inert-waste-types) | Decay model | simplification | low |
| [`missing-climate`](#missing-climate) | Decay model | fallback | low |
| [`particulate-factors`](#particulate-factors) | Outputs | fixed value | low |

## Where the paths disagree

Tracing each decision through every path, on 2026-10-22, found these differences. Some may be
deliberate; none is decided yet. Each names the entry it belongs to.

- **Flaring** (`flaring-uniform`): the city paths set no flaring, which the annual engine reads as
  1.0, so their captured gas is wholly destroyed. Every other path uses 0.98.
- **Decay rate** (`decay-rate-from-climate`): the waste mix is sampled in 2025 (city paths), 2000
  (`ct-site`, the site tool's prefill) or the implement year (advanced DST). Rubber counts as
  non-degradable on the city paths and the advanced DST, and is left out of the mix altogether in
  the 2000 sample. A typed rate skips the 0.04 floor.
- **First year** (`waste-before-1970-is-zero`): `ct-site` and the city paths start in 1970; the site
  tool and both advanced DST engines model back to 1950. A site opened before 1970 gets earlier
  waste in the site tool unless its caller raises the opening year first.
- **Unknown close year** (`intake-ends-before-close-year`): `ct-site` and the site tool close it in
  2050, taking no waste that year; the advanced DST has no unknown close year: it requires one, and
  2051 keeps a site open.
- **Site type** (`site-type-fill`): a city with no split of its own (What a Waste and other city
  sources give one for many cities) takes the IPCC 2019 Refinement's country, then regional,
  open-dump/landfill fractions (Tables 2A.1 and 2.1, Updated) before the rule; the code marks them for
  review. IPCC gives Central Asia and Rest of Oceania no fractions, so those places go straight to the
  rule (until 2026-10-23 they were filled with 100% dumpsite).
  Canada, Switzerland and Germany are hard-coded in `load_andre_params`.
- **Gas capture** (`gas-capture-efficiency`): the city paths capture a fixed 60% and ignore a
  city's reported efficiency; the city DST's converted dump captures 0.3 where the type default is 0.45.
- **What counts as capturing** (`oxidation`): `ct-site` uses Climate TRACE's capture-system flag, the
  site tool and advanced DST use an efficiency above 0, and the city paths use the capture bucket.
- **MCF** (`mcf-by-type-and-depth`): a city's controlled-landfill share is modelled as a landfill
  (1.0); a site typed controlled dump gets 0.6.
- **No population series** (`population-growth`): the city model uses the city's rates, a Custom
  Location world urban rates, `ct-site` Climate TRACE's rates, and the site tool asks the user.
- **Two engines** (`monthly-from-decay-model`): `ct-site` runs the monthly engine; every other path
  runs the annual one. They differ in timing, and in which year's decay rate and MCF apply when those
  change over time.
- **Diversion defaults** (`diversion-defaults`): a Custom Location always takes its country's
  composting and incineration shares; a city takes them only when it reports no diversion and its
  landfill shares add up to under 99%.
- **Over-drawn diversion** (`diversion-split`): the city paths re-split by min-cost max-flow; `adst-city`
  refuses the request.
- **Food prevention** (`food-waste-prevention`): the city DST keeps the baseline's decay rate and compost
  factor; `adst-city` recomputes both from the reduced mix.
- **Missing climate** (`missing-climate`): the advanced DST assumes 10 °C, the site-tool prefill 20 °C
  and 1,000 mm.
- **Unknown opening year** (`open-year-unknown`): `ct-site` uses 1970; the site-tool prefill 2010 for a
  listed site and 1970 for a location; the advanced DST requires a year.
- **Dead tags:** one tag each for `waste-before-1970-is-zero` and `intake-ends-before-close-year` sits
  in `City.site_only_estimate`, which nothing calls.

## Inputs

### `site-type-fill`

A site or city with no recorded type is assumed a sanitary landfill or an open dump, from its country.

- **Kind:** fallback
- **Why:** Site type sets MCF, oxidation and the gas-capture default, and most sites and many cities record no type.
- **Options, best first:**
  1. `recorded`: the type the caller supplies.
  2. `sanitary-by-country`: sanitary landfill if high income, in a listed region, or China, Russia, Mexico, Brazil or Costa Rica. Never Mongolia or Yemen.
  3. `dumpsite`: open dump otherwise, including a missing or unplaceable country. Never a controlled dump.
- **Paths:**
  - `ct-site`: Climate TRACE fills the type before the call, with its own income groups; SWEET applies the rule again to a blank type.
  - `city-model`, `custom-location`: only for a city with no split of its own, after IPCC 2019's country and then regional open-dump/landfill fractions (Tables 2A.1, 2.1).
  - `site-tool`: the prefill applies it to a location or a blank type, and makes an unrecognised type a dumpsite; the run takes the type it is given.
- **Confidence:** low
- **Better:** Classify each site from its area, intake or imagery, not its country's income, and use one fallback order on every path.
- **Value:** `SWEET_python.site_type_defaults`: `LANDFILL_DEFAULT_COUNTRIES`, `DUMPSITE_COUNTRIES`, `LANDFILL_DEFAULT_REGIONS`
- **Code:** `SWEET_python/site_type_defaults.py`
- **Reviewed:** 2026-10-22

### `waste-composition`

A site or city with no usable composition takes its country's mix, else its region's. The same mix is used in every year.

- **Kind:** fallback
- **Why:** Few sites or cities report their composition, and the country mix is the nearest measured value.
- **Options, best first:**
  1. `reported`: the caller's own mix.
  2. `country`: the country's row in SWEET's composition table, when its plastic share is above zero.
  3. `region`: the regional default.
- **Paths:**
  - `ct-site`, `custom-location`: always `country`, else `region`. A Climate TRACE site never supplies its own mix.
  - `city-model`: the city's reported mix, unless all blank or summing outside 0.98–1.02 (partial blanks become 0). The city DST baseline renormalises its published mix and never falls back.
  - `site-tool`: the caller's mix, except over the years a Climate TRACE series covers, which keep that series' mix (or split a total by the caller's). The location prefill uses `country`, else `region`.
  - `adst-site`, `adst-city`: the caller's mix, which may change by year.
- **Confidence:** medium
- **Better:** Use a site's own composition when reported (Climate TRACE issue #146), and let the mix change over time.
- **Value:** `SWEET_python.defaults_2019`: `WASTE_FRACTIONS_COUNTRY`, `WASTE_FRACTION_DEFAULTS`
- **Code:** `SWEET_python/defaults_2019.py`, `SWEET_python/city_params.py`
- **Reviewed:** 2026-10-22

### `all-city-waste-is-collected`

Every tonne a city generates is diverted or disposed of. None is left uncollected, and informal collection is not subtracted.

- **Kind:** simplification
- **Why:** Neither the cities table nor SWEET holds a collection-coverage figure for a city.
- **Paths:**
  - `city-model`: the waste mass is the generated tonnage, else the collected tonnage, else a per-person default. The informal-sector subtraction is commented out.
  - `custom-location`: a per-person default times population, all of it collected.
- **Confidence:** low
- **Better:** Scale by collection coverage, as Climate TRACE's country totals do with What a Waste 3.0. Uncollected waste is mostly burned or scattered and makes far less methane.
- **Code:** `SWEET_python/city_params.py`
- **Reviewed:** 2026-10-22
- **Docs:** [WasteMAP `docs/future_plans/uncollected_city_waste.md`](https://github.com/RMI/WasteMAP/blob/main/docs/future_plans/uncollected_city_waste.md)

### `population-growth`

Waste grows with the country's UN population each year. A big UN-listed city keeps its own two UN rates. With no usable series, fixed rates apply.

- **Kind:** fallback
- **Why:** Population is the main driver of how much waste a place generates over time, and a big city can outgrow its country.
- **Options, best first:**
  1. `un-city-rates`: a UN-listed city of 300,000 or more, with plausible rates, compounds its own historic and future UN rates.
  2. `un-population`: the country's UN World Population Prospects 2024 population, by year, from SWEET's table.
  3. `constant-rate`: fixed historic and future rates, when there is no usable yearly series.
- **Paths:**
  - `city-model`, `city-dst-scenario`: all three; `constant-rate` is the city's own rates. `city_population_trajectory` gives the same answer to a caller that stores it.
  - `custom-location`: `un-population`, else world urban-population rates.
  - `site-tool`: `un-population`, else an error asking for a rate. A rate the user sets compounds both ways; −100% is meant to hold waste flat.
  - `ct-site`: the series Climate TRACE hands in, gaps held at the nearest year, else Climate TRACE's rates. An anchor year outside 1970–2050 takes the rates.
- **Confidence:** medium
- **Better:** Use one fallback on every path, and UN city trajectories rather than two rates for listed cities.
- **Known problem:** in `site-tool`, a −100% rate holds only the baseline flat; the scenario's waste goes to zero or infinity, because its growth factors skip the zero-rate guard.
- **Value:** `SWEET_python/pops_yearly.csv`; `city_params`: `UN_LISTED_CITY_MIN_POPULATION`, `PLAUSIBLE_CITY_GROWTH`, `FALLBACK_GROWTH_RATE_HISTORIC`, `FALLBACK_GROWTH_RATE_FUTURE`
- **Code:** `SWEET_python/class_defs.py`
- **Reviewed:** 2026-10-22

### `intake-follows-population`

A site's intake in other years is its anchor year's intake scaled by population, so waste per person never changes. Several years' reports, from 2010 on, are averaged first.

- **Kind:** simplification
- **Why:** Population is the one driver available for every country and year.
- **Paths:**
  - `ct-site`: as above, at the latest report year. With no dated row, a several-row site is anchored at 2020; a one-row site at last year by the system clock, or the year before closing.
  - `city-model`, `city-dst-scenario`, `custom-location`: the same constant-per-person rule for a city's generated waste.
  - `site-tool`: grows a blank site the same way, and scales a Climate TRACE site's series by one ratio.
- **Confidence:** low
- **Better:** Let waste per person change with income, as What a Waste projects, and place an averaged report at its middle year.
- **Code:** `SWEET_python/city_params.py` (2 places)
- **Reviewed:** 2026-10-22

### `waste-before-1970-is-zero`

No waste is deposited before 1970. A site that opened earlier is modelled as opening in 1970.

- **Kind:** simplification
- **Why:** SWEET's population-driven deposition history starts in 1970, and most earlier waste has decayed by the reported years.
- **Paths:**
  - `ct-site`: an earlier open year becomes 1970.
  - `city-model`, `city-dst-scenario`, `custom-location`: every series starts in 1970.
- **Confidence:** medium
- **Better:** Start each site at its real open year, back to 1950, as the site tool and the advanced DST engines already do.
- **Value:** `SWEET_python.constants.MODEL_START_YEAR`. The site tool and the advanced DST refuse open years before `constants.MODEL_YEAR_MIN` (1950); neither clamps.
- **Code:** `SWEET_python/city_params.py` (3 places)
- **Reviewed:** 2026-10-22

### `intake-ends-before-close-year`

A site takes its last waste in the year before its close year, not in the close year itself.

- **Kind:** simplification
- **Why:** A close year is when operations stop, so a full year of waste then would overstate it.
- **Paths:**
  - `ct-site`: a site with no close year is closed in 2050, so takes no waste that year.
  - `site-tool`: the same rule; years after 2050 are refused, so an open site also closes in 2050.
  - `adst-site`, `adst-city`: the same rule; 2051 means never closing, so an open site takes 2050's waste. `adst-city` gives a closed site's share to the open ones.
- **Confidence:** medium
- **Better:** Treat an unknown close year as never closing on every path.
- **Value:** `SWEET_python.dst_common.NEVER_CLOSES` (2051) in the advanced DST
- **Code:** `SWEET_python/city_params.py` (4 places)
- **Reviewed:** 2026-10-22

### `waste-per-capita-default`

A city or place with no waste tonnage generates its country's per-person default, else its region's, times its population.

- **Kind:** fallback
- **Why:** Every city needs a tonnage, and the IPCC 2019 Refinement tabulates waste per person by country and region.
- **Options, best first:**
  1. `reported`: the city's generated, else collected, tonnage.
  2. `country`: the country's rate (IPCC 2019 Refinement Table 2A.1, some rows from other sources) times population.
  3. `region`: the regional rate times population.
- **Paths:**
  - `city-model`: the rate is read as the population year's.
  - `custom-location`: the rate times the population of 2022.
  - `ct-site`: dates it 2019; unreachable, because Climate TRACE drops sites with no tonnage before calling SWEET.
- **Confidence:** low
- **Better:** Let waste per person change with income and year, as What a Waste projects, instead of one figure for any year.
- **Value:** `SWEET_python.defaults_2019`: `MSW_PER_CAPITA_COUNTRY`, `MSW_PER_CAPITA_DEFAULTS`
- **Code:** `SWEET_python/city_params.py` (2 places)
- **Reviewed:** 2026-10-23

### `city-shares-constant`

A city's diversion shares and its landfill, dump and capture split are one number each, used in every year from 1970 to 2050.

- **Kind:** simplification
- **Why:** The cities table reports one year's treatment shares, and nothing records how they changed.
- **Paths:**
  - `city-model`, `custom-location`: in every year.
  - `city-dst-scenario`: the baseline's shares until the implementation year.
- **Confidence:** low
- **Better:** Hold a treatment at zero before the year a city started it. A city that began incinerating recently is otherwise modelled as burning since 1970, which cuts its waste in place and today's methane.
- **Code:** `SWEET_python/city_params.py`
- **Reviewed:** 2026-10-23

### `open-year-unknown`

A site with no usable opening year opens in 1970 on `ct-site`. The site-tool prefill offers 2010 for a listed site and 1970 for a map location.

- **Kind:** fallback
- **Why:** Not recorded; 1970 is the model's first year, so it assumes the longest history.
- **Options, best first:**
  1. `recorded`: the year the caller supplies.
  2. `model-start`: 1970.
  3. `prefill-2010`: 2010, for a listed site in the site-tool prefill.
- **Paths:**
  - `ct-site`: `model-start`, including for "NS", "NO SABE" and similar entries.
  - `site-tool`: the prefill's `prefill-2010` or `model-start`; the run takes the year it is given.
- **Confidence:** low
- **Better:** One rule on every path, and estimate a site's age from its area or intake.
- **Code:** `SWEET_python/city_params.py` (2 places)
- **Reviewed:** 2026-10-23

### `site-tool-prefill-defaults`

The site-tool prefill fills gaps with fixed values: 10,000 t a year, a waste year of 2024 (listed site) or 2025 (location), a close year of 2050, and growth anchored on last year by the system clock.

- **Kind:** fixed value
- **Why:** Not recorded; the form needs a starting value for every field.
- **Paths:**
  - `site-tool`: the prefill only; the run takes what the form sends. A failed weather lookup is `missing-climate`.
- **Confidence:** low
- **Better:** One waste year for both cases, the model year from the caller rather than the clock, and unknown fields left blank.
- **Code:** `SWEET_python/city_params.py`
- **Reviewed:** 2026-10-23

### `multi-city-site-split`

A site several cities send waste to is split into one share per city, weighted by each city's latest population times its average share sent, and renormalised each year.

- **Kind:** method choice
- **Why:** Climate TRACE links some sites, Brazil's, to several cities and needs each city's part.
- **Paths:**
  - `ct-site`: in `City.citysite_estimate_trace` only. The site's total does not change, only its split between cities.
- **Confidence:** low
- **Better:** Use each city's reported share by year where it exists, and its tonnage rather than its population.
- **Code:** `SWEET_python/city_params.py`
- **Reviewed:** 2026-10-23

## Diversion

### `combustion-reject`

None of the waste sent to incineration is landfilled unburned, so incinerated waste makes no methane.

- **Kind:** fixed value
- **Why:** Incinerator ash is nearly all mineral: the EU's Industrial Emissions Directive holds its organic carbon under 3%, and Japan landfills under 1% of its waste unburned. The old 0.1 had no source.
- **Paths:**
  - `city-model`: on the reported, else the default, incineration share (see `diversion-defaults`).
  - `city-dst-scenario`: on the scenario's chosen share.
  - `custom-location`: on the country's default share.
  - `adst-city`: on the combustion pathway, and on a facility marked `combusts`, which would deposit that share of its intake.
  - `ct-site`: multiplies an incineration share hard-coded to 0, so it changes nothing.
- **Confidence:** medium
- **Better:** Where a plant reports the unburned organic content of its bottom ash, use that.
- **Value:** `SWEET_python.constants.COMBUSTION_REJECT_RATE`: 0, and 0.1 through `sweet/2026.10.5`
- **Code:** `SWEET_python/constants.py`
- **Reviewed:** 2026-10-22

### `other-treatment-shares-landfilled`

A city's treatment shares reported as "other" or "unaccounted for" get no destination. The landfill split is scaled to 100%, so that waste is landfilled.

- **Kind:** simplification
- **Why:** SWEET has no pathway for them, and What a Waste 2.0 records mechanical-biological treatment under "other".
- **Paths:**
  - `city-model`: in `load_andre_params`. The city DST then reads the split the cities table publishes.
- **Confidence:** low
- **Better:** Route mechanical-biological treatment by what it puts out (recycling, combustion, compost, residue), and treat unaccounted waste through collection coverage.
- **Code:** `SWEET_python/city_params.py`
- **Reviewed:** 2026-10-22
- **Docs:** [WasteMAP `docs/future_plans/mbt_needs_a_destination.md`](https://github.com/RMI/WasteMAP/blob/main/docs/future_plans/mbt_needs_a_destination.md)

### `diversion-defaults`

A city or place with no treatment data takes its country's, else its region's, composting and incineration shares, and none for digestion or recycling.

- **Kind:** fallback
- **Why:** A city with no treatment data most likely treats its waste as its country does; Hugh, 2026-10-23.
- **Options, best first:**
  1. `reported`: the city's own shares.
  2. `country`: the country's composting and incineration shares.
  3. `region`: the region's.
  4. `none`: no diversion.
- **Paths:**
  - `custom-location`: `country`, else `region`, else `none`.
  - `city-model`: `reported`; with every diversion blank and landfill shares under 99%, `country`, else `region`, else `none`. Before this release the defaults never applied.
- **Confidence:** low
- **Better:** Review the default tables, which the code marks as needing review: Palestine's 69% incineration, for one, is doubtful.
- **Value:** `SWEET_python.defaults_2019`: `FRACTION_COMPOSTED_COUNTRY`, `FRACTION_COMPOSTED`, `FRACTION_INCINERATED_COUNTRY`, `FRACTION_INCINERATED`
- **Code:** `SWEET_python/city_params.py` (2 places)
- **Reviewed:** 2026-10-23

### `city-treatment-buckets`

A city's reported treatment shares map onto three buckets: sanitary landfill with gas collection is "with capture"; controlled landfill and unspecified landfill are "without capture"; advanced thermal treatment counts as incineration.

- **Kind:** method choice
- **Why:** A controlled landfill is engineered, so it takes a sanitary landfill's MCF; an unspecified landfill is assumed to be one too, not typed by the country's rule.
- **Paths:**
  - `city-model`: in `load_andre_params`; diversion shares above 100% are scaled down. The city DST reads the split the cities table publishes.
- **Confidence:** low
- **Better:** Model an unspecified landfill by the country's site-type rule. In a country SWEET's rule calls dumps, the "without capture" bucket gives about 50% more methane than a dump.
- **Code:** `SWEET_python/city_params.py`
- **Reviewed:** 2026-10-23

### `diversion-eligibility`

Each diversion takes only some waste types: composting and digestion take food, green, wood and paper; recycling takes everything but food and green; incineration takes all ten.

- **Kind:** fixed value
- **Why:** Not recorded; it follows what each treatment can process.
- **Paths:**
  - `city-model`, `city-dst-scenario`, `custom-location`, `adst-city`: a diversion draws on these types in proportion to the mix.
- **Confidence:** medium
- **Better:** Let incineration take a burnable mix. A share now burns metal and glass in proportion, so it removes fewer degradables than its size suggests.
- **Value:** `SWEET_python.constants.DIVERSION_ELIGIBILITY`, which `City.div_components` copies and `dst_allocation.DEFAULT_ELIGIBILITY` reads
- **Code:** `SWEET_python/constants.py`
- **Reviewed:** 2026-10-23

### `diversion-rejects`

Part of what is sent to composting or recycling goes back to landfill as its own material; digestion keeps everything. Incineration's reject is `combustion-reject`.

- **Kind:** fixed value
- **Why:** Not recorded; composting turns away contaminants and unprocessable material, and recycling has process losses.
- **Paths:**
  - `city-model`, `city-dst-scenario`, `custom-location`, `adst-city`: composting keeps (1 − not targeted) × (1 − unprocessable) of each type; recycling keeps a fixed share of each.
- **Confidence:** low
- **Better:** Source each rate. Rename the attribute `City.recycling_reject_rates`, which holds the share kept (80% of wood, 77.5% of paper), not the share rejected.
- **Value:** `SWEET_python.constants`: `NON_COMPOSTABLE_NOT_TARGETED`, `UNPROCESSABLE_SHARE`, `RECYCLING_YIELD`
- **Code:** `SWEET_python/constants.py`
- **Reviewed:** 2026-10-23

### `diversion-emission-factors`

Composting emits a CO2e factor per tonne of food and green waste, turned into methane with a GWP of 28. Digestion emits 0.26 kg of methane per short ton; recycling and incineration emit none.

- **Kind:** fixed value
- **Why:** The compost factor comes from EPA WARM via the SWEET spreadsheet. The spreadsheet notes that digestion's leaves out plant leakage.
- **Paths:**
  - `city-model`, `city-dst-scenario`, `custom-location`, `adst-city`: composting and digestion emissions. Everywhere else ×28 and ÷28 cancel.
- **Confidence:** low
- **Better:** Use a sourced composting factor; IPCC 2006's 4 kg of methane per tonne is at least ten times SWEET's. Add digestion leakage. GWP 28 is AR5's; Climate TRACE publishes with AR6's.
- **Value:** `SWEET_python.defaults_2019`: `compost_emission_factor`, `MEF_ANAEROBIC`; `SWEET_python.constants.GWP100_CH4`
- **Code:** `SWEET_python/city_params.py`, `SWEET_python/defaults_2019.py`
- **Reviewed:** 2026-10-23

### `diversion-split`

When a diversion over-draws a waste type, the city paths re-split composting, digestion and recycling by min-cost max-flow, favouring types fewer treatments take, then burn a uniform share of the rest.

- **Kind:** method choice
- **Why:** A proportional split can ask for more of a type than a city generates.
- **Paths:**
  - `city-model`, `city-dst-scenario`, `custom-location`: proportional, then the solver when a type is over-drawn.
  - `adst-city`: proportional; an over-drawn type is refused as an error.
- **Confidence:** medium
- **Better:** One rule on every path. The split jumps from proportional to the solver's as a slider crosses the boundary.
- **Code:** `SWEET_python/city_params.py`
- **Reviewed:** 2026-10-23

### `food-waste-prevention`

Prevented food is never generated: from the implementation year food falls by the prevented share, the total shrinks, and diversion shares apply to the smaller stream.

- **Kind:** method choice
- **Why:** Prevention means the food is never made, so it leaves before any treatment.
- **Paths:**
  - `city-dst-scenario`: keeps the baseline's decay rate and compost factor.
  - `adst-city`: recomputes both from the reduced mix, with the share clamped to 0–1.
- **Confidence:** medium
- **Better:** One rule for whether prevention changes the decay rate and the compost factor.
- **Code:** `SWEET_python/city_params.py`, `SWEET_python/advanced_dst_city.py`
- **Reviewed:** 2026-10-23

## Decay model

### `mcf-by-type-and-depth`

MCF is 1.0 at a sanitary landfill and 0.6 at either kind of dump. A dump with a known depth gets 0.8 if deeper than 5 m, else 0.4.

- **Kind:** fixed value
- **Why:** These are the IPCC 2006 defaults. Depth is unknown at nearly every dump, so dumps take IPCC's value for an uncategorised site.
- **Paths:**
  - `ct-site`, `site-tool`, `adst-site`, `adst-city`: by type and the depth given. Climate TRACE sends 10 m, 2 m or none.
  - `city-model`, `city-dst-scenario`, `custom-location`: the same values, never with a depth: `MCF_MANAGED_ANAEROBIC` for the two landfill buckets, `MCF_UNCATEGORISED` for the dumpsite bucket.
- **Confidence:** medium
- **Better:** Record depth at dumps, and send the city buckets through `mcf_for_site` so a change to the table reaches them.
- **Value:** `SWEET_python.mcf`: `MCF_BY_TYPE`, `DEEP_SITE_DEPTH_M`
- **Code:** `SWEET_python/mcf.py`
- **Reviewed:** 2026-10-22

### `decay-rate-from-climate`

Each site or city gets one decay rate from its temperature, rainfall and waste mix (Wang et al. 2024), at least 0.04 a year, used for every waste type in every year.

- **Kind:** method choice
- **Why:** It uses the site's own climate rather than IPCC's four climate zones.
- **Paths:**
  - `city-model`, `custom-location`: the waste mix in 2025.
  - `city-dst-scenario`: keeps the baseline's rate, even when the scenario changes the mix.
  - `ct-site`: the waste mix in 2000.
  - `adst-site`: the mix at the implement year, the scenario's own after it. A typed `k_override` replaces the rate and is not floored.
  - `adst-city`: one city-wide rate from the mix at the implement year, for every landfill.
  - `site-tool`: takes the caller's rate, not floored; its prefill computes one from the mix in 2000.
- **Confidence:** medium
- **Better:** Use a separate rate for each waste type, as the IPCC model does, and one rule for which year's mix to use.
- **Value:** `SWEET_python.singapore_k.singapore_k`: `K_MIN_PER_YEAR`, `MOISTURE_FACTOR_BANDS`, `LANDFILL_T_OPT_C`
- **Code:** `SWEET_python/singapore_k/singapore_k.py`
- **Reviewed:** 2026-10-22

### `methane-potential-by-waste-type`

Each degradable waste type has a fixed methane potential: food 70, green 93, wood 200, paper 186 and textiles 112 m³ per tonne. Emissions start half a year after deposit.

- **Kind:** fixed value
- **Why:** These follow IPCC 2006's default carbon contents, with half the carbon decomposing and half the gas being methane.
- **Paths:**
  - `city-model`, `city-dst-scenario`, `custom-location`, `site-tool`, `adst-site`, `adst-city`: in the annual engine.
  - `ct-site`: in the monthly engine.
- **Confidence:** high
- **Better:** Use measured degradable carbon where a country or site reports it.
- **Value:** `SWEET_python.defaults_2019.L_0`
- **Code:** `SWEET_python/model_v2.py` (2 places)
- **Reviewed:** 2026-10-22

### `oxidation`

A site's oxidation is its reported value by year, else a default set by its type and whether it captures gas.

- **Kind:** fallback
- **Why:** Oxidation is rarely reported, and cover quality tends to follow site type and gas management.
- **Options, best first:**
  1. `reported`: the reported or user-stated value.
  2. `type-default`: without capture 0.10, 0.05 and 0 for sanitary, controlled dump and dumpsite. With capture 0.22, 0.10 and 0.
- **Paths:**
  - `ct-site`: `reported` by year, else `type-default` chosen by Climate TRACE's capture-system flag.
  - `site-tool`: a nonzero user value, else `type-default` chosen by an efficiency above 0. A dump upgraded to a landfill with capture gets 0.18; biocover sets a floor.
  - `adst-site`, `adst-city`: `type-default` chosen by an efficiency above 0, an upgraded dump 0.18, with biocover's floor.
  - `city-model`, `city-dst-scenario`, `custom-location`: `type-default`, chosen by disposal bucket.
- **Confidence:** medium
- **Better:** Choose the default the same way on every path, and let the city tools use reported values.
- **Value:** `SWEET_python.oxidation`: `OX_NO_CAPTURE`, `OX_WITH_CAPTURE`, `OX_REMEDIATED_TO_LANDFILL`; Climate TRACE reads them as `defaults_2019.oxidation_factor`
- **Code:** `SWEET_python/city_params.py` (2 places)
- **Reviewed:** 2026-10-22

### `gas-capture-efficiency`

A site captures its reported efficiency, else a site-type default if it has a capture system, else nothing.

- **Kind:** fallback
- **Why:** Few sites report an efficiency, but many report having a system.
- **Options, best first:**
  1. `reported`: the reported or user-stated efficiency.
  2. `type-default`: 0.6 at a sanitary landfill, 0.45 at a controlled dump, 0 at a dumpsite, for a site with a capture system.
  3. `none`: zero, when nothing says the site has a system.
- **Paths:**
  - `ct-site`: all three; Climate TRACE decides whether a site has a system.
  - `site-tool`: the caller's value, which it must give.
  - `adst-site`, `adst-city`: the caller's value, by year where it gives years; none given means 0.
  - `city-model`: 0.6 on the share sent to landfills with capture; the cities table's efficiency is read but never used.
  - `custom-location`: 0.6 on a capture share, but the default split has none, so a Custom Location captures nothing.
  - `city-dst-scenario`: "add gas capture" sets 0.6 on both landfill shares, including the one without capture, and 0.3 on a dump made a controlled dump, from the year after implementation.
- **Confidence:** medium
- **Better:** Use one table and rule on every path, and let the city model read each city's reported efficiency.
- **Value:** `SWEET_python.gas_capture`: `GAS_CAPTURE_EFFICIENCY`; the city DST's converted dump, `CONVERTED_DUMP_CAPTURE_EFFICIENCY`
- **Code:** `SWEET_python/city_params.py`
- **Reviewed:** 2026-10-22

### `flaring-uniform`

Captured gas is burned at 0.98 efficiency in every year, unless the caller states a value.

- **Kind:** simplification
- **Why:** No source records flaring efficiency per site.
- **Paths:**
  - `ct-site`: 0.98 in every year.
  - `site-tool`, `adst-site`, `adst-city`: the caller's value, else 0.98.
  - `city-model`, `city-dst-scenario`, `custom-location`: none is set, and the annual engine reads that as 1.0: captured gas is wholly destroyed.
- **Confidence:** medium
- **Better:** Read flare efficiency per site where operators report it, and give the city paths 0.98 so no path treats flaring as perfect.
- **Value:** `SWEET_python.constants`: `DEFAULT_FLARE_EFFICIENCY`; the city paths' 1.0 is `FLARE_EFFICIENCY_WHEN_UNSET`
- **Code:** `SWEET_python/city_params.py` (2 places)
- **Reviewed:** 2026-10-22

### `monthly-from-decay-model`

Monthly values come from SWEET's monthly decay model, with each year's waste spread by month length and no seasonal cycle.

- **Kind:** simplification
- **Why:** Nothing measures how a landfill's methane varies through the year.
- **Paths:**
  - `ct-site`: the only path that runs the monthly engine. Its diversion emissions are spread at a twelfth a month.
- **Confidence:** medium
- **Better:** Make the monthly and annual engines agree on timing, and on which year's decay rate and MCF apply.
- **Code:** `SWEET_python/model_v2.py`
- **Reviewed:** 2026-10-22

### `inert-waste-types`

Only food, green, wood, paper and textiles make methane. Rubber is inert, and nappies are counted as "other", which is inert.

- **Kind:** simplification
- **Why:** Not recorded; SWEET's ten waste types have no nappies type.
- **Paths:**
  - `ct-site`, `custom-location`: rubber and "other" make no methane, and the country mixes they use fold nappies into "other".
  - `city-model`, `city-dst-scenario`, `site-tool`, `adst-site`, `adst-city`: rubber and "other" make no methane; nappies are folded in wherever a country mix fills a gap.
- **Confidence:** low
- **Better:** Give nappies a methane potential; `L_0` already carries 112 for them, unused. Nappies are 6.9% of France's country mix, about 14% of its methane potential.
- **Value:** `SWEET_python.defaults_2019.L_0` (its `diapers` and `rubber` values are unused)
- **Code:** `SWEET_python/defaults_2019.py`
- **Reviewed:** 2026-10-23

### `missing-climate`

A missing climate value takes a fallback: missing rainfall the wettest moisture class, a missing advanced-DST temperature 10 °C, and a failed site-tool weather lookup 1,000 mm and 20 °C.

- **Kind:** fallback
- **Why:** Not recorded.
- **Options, best first:**
  1. `reported`: the caller's or the weather table's value.
  2. `fallback`: the values above.
- **Paths:**
  - `city-model`, `custom-location`, `ct-site`: missing rainfall gets a moisture factor of 1.0, ten times a dry site's.
  - `adst-site`, `adst-city`: 10 °C when no temperature is sent.
  - `site-tool`: the prefill's 1,000 mm and 20 °C.
- **Confidence:** low
- **Better:** Use the nearest weather cell, one default pair everywhere, or refuse to run without a climate.
- **Code:** `SWEET_python/singapore_k/singapore_k.py`, `SWEET_python/city_params.py`
- **Reviewed:** 2026-10-23

## Outputs

### `particulate-factors`

PM2.5 and PM10 are each 0.00024 kg per m³ of methane destroyed by flaring.

- **Kind:** fixed value
- **Why:** Flaring is the one particulate source a landfill model can see.
- **Paths:**
  - `ct-site`: the monthly engine reports them. The annual engine, on every other path, reports none.
- **Confidence:** low
- **Better:** Cite a flare emission factor, and give PM2.5 and PM10 their own.
- **Value:** `SWEET_python.model_v2`: `PM2P5_KG_PER_M3_CH4_FLARED`, `PM10_KG_PER_M3_CH4_FLARED`
- **Code:** `SWEET_python/model_v2.py`
- **Reviewed:** 2026-10-22

## Retired

None yet.
