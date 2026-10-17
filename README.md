# SWEET_python
This repo contains code for a Python port of the methane emissions portions of the US Environmental Protection Agency's Solid Waste Emissions Estimation Tool (SWEET). SWEET is a first-order decay model for estimating methane emissions from municipal waste, similar to the IPCC Waste Model from the 2006 report. This python port does not contain the portions SWEET for calculating non-methane emissions. The original Excel model is available at https://globalmethane.org/resources/details.aspx?resourceid=5176. A user manual is also available at that site. 


# Installation
After following these steps, SWEET_python can be imported:
1) Use an existing python environment or create one with conda or venv. e.g. `conda create -n "sweet"`
2) Clone this repo (in a terminal, write `git clone https://github.com/RMI/SWEET_python.git`)
3) cd into the SWEET_python directory, then write `pip install -e .` to install in editable mode, which allows editing the code. Updates to the code in the repo should be automatically reflected in the installed package. If they are not, reinstall it. 


# Tests

The test suite lives in `tests/` and runs with pytest:

```
pip install -r requirements.txt "pytest>=8,<10"
pip install --no-deps -e .
pytest
```

`--no-deps` on the editable install is the point of the two lines above it:
`setup.py`'s `install_requires` comes from `requirements.in`, the abstract deps,
so without it pip re-resolves what `requirements.txt` just pinned and the local
environment stops matching CI. The install step in
`.github/workflows/tests.yml` is the same three commands for the same reason.
(The `pip install -e .` in Installation above has no lockfile to protect and
should resolve its dependencies.)

Every test in the suite is hermetic — no database, no network, no environment
variables — and the whole thing takes a few seconds.

CI (`.github/workflows/tests.yml`) runs `pytest -m "not integration"` on Python
3.12 for every push and pull request. If you add a test that does need a live
database or network access, mark it `@pytest.mark.integration` so it stays out of
that job, following the same fast/integration split as WasteMAP's
`backend/tests/fast` and `backend/tests/integration`. There are no such tests
today; the first one should get a CI job of its own rather than a database being
bolted onto the fast job.

This is complementary to WasteMAP's CI, not a replacement for it: WasteMAP
installs a SWEET_python branch with the same name as the WasteMAP branch under
test, so a model change here and its WasteMAP consumer are already exercised as a
pair over there. What that pairing does *not* do is run this repo's own tests —
that is the gap this workflow fills. Tests of this package's internals belong
here, in the repo that owns them.

# Usage

SWEET has no single entry point: each caller uses the path built for it. The two apps
that call it are the working examples — WasteMAP's API (`backend/sweet_launch.py` in
[RMI/WasteMAP](https://github.com/RMI/WasteMAP)) and the
[Climate TRACE waste-methane pipeline](https://github.com/RMI/RMI_Climate_TRACE_Waste_Methane).
Where things are:

- **`SWEET_python/city_params.py`** — `City`, which sets up a city's waste, diversion
  and landfills and runs them. Its main entry points: `load_csv_new` and
  `load_andre_params` (a row of WasteMAP's cities table); `dst_baseline_blank` (a
  Custom Location: a country, a population and the climate);
  `cityparams_obj_for_blank_site` and `sdst_v1_5` (WasteMAP's single-site tool); and
  `site_only_estimate_trace` (one Climate TRACE site).
- **`SWEET_python/landfill.py`** — `Landfill`: one disposal site (its type, open and
  close years, gas capture and cover) and `estimate_emissions`, which runs the model
  for it.
- **`SWEET_python/model_v2.py`** — `SWEET`, the first-order-decay engine.
- **`SWEET_python/advanced_dst.py`** and **`advanced_dst_city.py`** — the advanced DST,
  where the caller supplies waste mass and composition as yearly series directly:
  `run_advanced_dst` for one site, `run_advanced_dst_city` for a city and its
  landfills.
- **`SWEET_python/defaults_2019.py`** — default parameters by country and region (waste
  per person, composition, disposal split), alongside `mcf.py` (methane correction
  factors), `population.py` (UN population, for growing waste year by year) and
  `dst_common.py` (shared by the DST paths).

# Decision comments

Some lines carry a comment like `# decision: mcf-by-type-and-depth`. Each marks a modelling
choice that changes the numbers Climate TRACE publishes: a default value, a fallback, or a
simplification. The id names an entry in
[`docs/MODEL_DECISIONS.md`](https://github.com/RMI/RMI_Climate_TRACE_Waste_Methane/blob/main/docs/MODEL_DECISIONS.md)
in the Climate TRACE pipeline repo. The entry says what the choice is, why it was made, and how
sure we are of it. The list lives there, not here, because its ratings are about Climate TRACE's
published numbers. SWEET keeps no copy.

Seven decisions are tagged in SWEET.

- **Four are shared with WasteMAP's tools**, so changing one moves WasteMAP's numbers too:
  - MCF, in `mcf.py`
  - the guess for a site with no recorded type, in `site_type_defaults.py`
  - waste composition, in `defaults_2019.waste_composition_for`
  - population growth, in `class_defs.growth_factors_for_years`
- **Three are in the Climate TRACE paths** of `city_params.py`: gas capture, oxidation and
  flaring.

To change a tagged line:

- Open a pull request in the Climate TRACE repo, from a branch with the same name as yours here,
  that updates the entry. A replaced decision moves to that list's "Retired" section with the date.
- Keep the comment on the line it marks. Rename or remove it only when the entry changes too.

The Climate TRACE pipeline's test checks these comments against its list. It reads them from the
SWEET release named in that repo's `sweet_python.ref`. A release that drops or renames a comment
therefore fails that repo's CI when the release is pinned, not this repo's.

# Releases

SWEET publishes dated releases. A release is an annotated git tag named
`sweet/YYYY.MM.N` on a commit from `main`, with a matching
[GitHub Release](https://github.com/RMI/SWEET_python/releases).

## What the name means

- **`sweet/`** — a release of this repository. The other tags here are other projects'
  records of which SWEET commit they used: `wastemap/YYYY.MM.DD` (a wastemap.earth
  deploy), `submission/MM_DD_YY` (a Climate TRACE submission) and the older
  `v2026.08.17`. They are not SWEET releases, and the release workflow does not count
  them.
- **`YYYY.MM`** — the year and month, in UTC, that the release was cut. Not the year of
  any data or method in it.
- **`N`** — counts from 0 within the month: `sweet/2026.10.0`, `sweet/2026.10.1`, then
  `sweet/2026.11.0`.

A later name always contains everything in an earlier one: the workflow will not release
a commit that lacks the newest release, so moving a pin to a newer release can never take
a fix away. The name says nothing about how big the change is; the release notes and the
changelog do.

## A release is never moved or reused

Once `sweet/2026.10.0` exists, it points at the same commit for good. Nothing moves,
deletes or re-creates a release tag, because an app pinned to it has to keep installing
exactly the code it was tested with. A release with a mistake in it is fixed by merging
the fix to `main` and cutting the next release.

## Cutting a release

1. Merge what should be in it to `main`, with its changelog entry (see
   [`changelog/README.md`](changelog/README.md)).
2. On GitHub, open **Actions → Release → Run workflow**, leave "Use workflow from" on
   `main`, and run it. The defaults release the tip of `main` under the next free name
   for this month. The two inputs are for when you need something else:
   - **ref** — what to release: a branch, tag or commit sha. It has to be on `main`.
   - **version** — a name of your own, e.g. `sweet/2026.10.3`. It has to be unused and
     sort after the newest release.
3. Before anything is tagged, the workflow refuses a commit `main` does not contain, a
   commit that is already a release, a commit that lacks the newest release, a name that
   is taken, and a commit whose tests fail (the same `pytest -m "not integration"` as
   CI, run against that commit). Then it tags the commit, pushes the tag and publishes
   the GitHub Release, whose notes list the pull requests merged since the previous
   release and link to that month's changelog file. The run's summary page shows the
   tag, the commit and the links.

If the tag gets pushed but the GitHub Release step fails, do not re-run the workflow:
the name is taken now, so it will refuse. The run's summary page gives the
`gh release create` command that finishes the job.

## How an app asks for a release

An app that uses SWEET names the release it wants in a small pin file in its own
repository (each app documents its own) and installs that tag:

```
pip install "git+https://github.com/RMI/SWEET_python.git@sweet/2026.10.0"
```

A copy installed this way knows which commit it is: pip builds it from a git checkout,
and `setup.py` writes that commit into the package, so `SWEET_python.__version__` returns
it (`VERSION_SOURCE == "build"`). An editable install or a path dependency has no such
stamp; for those, the app sets `SWEET_GIT_SHA` to the commit it installed (both apps do),
and a local checkout falls back to reading its `.git`. Nothing reports a commit it
cannot vouch for: with none of the three, `__version__` is `"unknown"`.

- On an app's **feature branch**, the pin may name a SWEET branch instead, to try a
  model change before it is released.
- On an app's **`main` and production line**, the pin names a release or an exact
  commit sha, never a branch. A branch moves: the next merge here would change what the
  app runs without anyone changing the app.

In a running app, `SWEET_python.__version__` is the commit it installed, not the release
name (the package docstring says why). To turn a commit back into a release name:
`git tag --points-at <sha> --list 'sweet/*'`.

The pin matters most for WasteMAP's API, which calls SWEET at request time for its
decision-support tools (`/sdst`, `/adst`, `/cdst`): whatever SWEET its image was built
with is live the moment it deploys, even while the map's precomputed numbers came from an
older model. WasteMAP's
[`RELEASES.md`](https://github.com/RMI/WasteMAP/blob/main/RELEASES.md) covers its side.
