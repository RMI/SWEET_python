#!/usr/bin/env python3
"""Run the same landfills through the SWEET Excel workbook and through SWEET_python.

This checks the decay engine, not our default values. For each case below the
script fills a fresh copy of the workbook, lets Excel recalculate, and then reads
back every value Excel actually used: waste by stream and year, k per stream, L0,
MCF, collection efficiency by year, oxidation and flare destruction. Those exact
values go into both of our engines (``SWEET.estimate_emissions2``, the annual
one, and ``SWEET.estimate_emissions_monthly``, the one the TRACE pipeline runs),
so any gap left is in how the engines compute, not in what they were given.

Excel cannot take a site's tonnage year by year: it builds waste from population
x generation rate x share collected, grown at one rate. A case can therefore
paste its own yearly waste straight over the formulas in ``'I. Waste'``, and its
own yearly collection efficiency over ``'CH4 Collect'``. That happens only in a
throwaway copy; the template is never written to.

Needs Microsoft Excel and ``xlwings`` (``uv pip install xlwings``), so it runs on
a Mac with Excel, never in CI. The first run brings up a macOS prompt asking to
let Python control Excel. Excel is sandboxed, so the working copies go in its own
container folder, which it can always open.

    python scripts/excel_engine_check.py --workbook ~/.../SWEET_Version_4.0.6_EN.xlsm
    python scripts/excel_engine_check.py --workbook ... --cases landfill_gas
    python scripts/excel_engine_check.py --workbook ... --write-fixtures

``--write-fixtures`` saves each case's Excel inputs and outputs as a small CSV
under ``tests/fixtures/excel_sweet_4_0_6/``. ``tests/test_excel_engine_parity.py``
replays those with no Excel, so CI keeps the engines honest from then on.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from SWEET_python import defaults_2019
from SWEET_python.landfill import CH4_DENSITY_KG_PER_M3
from SWEET_python.model_v2 import SWEET

REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO / "tests" / "fixtures" / "excel_sweet_4_0_6"
OUT_DIR = REPO / "scripts" / "excel_engine_check_out"
# Excel is sandboxed on macOS: a file outside its container makes it ask for
# access, which a hidden instance cannot show, so it hangs.
EXCEL_WORK_DIR = Path.home() / "Library/Containers/com.microsoft.Excel/Data/sweet_excel_check"

# The workbook's degradable streams, in its column order, and our names for them.
STREAMS = {"food": "food", "green": "green", "wood": "wood", "paper": "paper_cardboard", "textiles": "textiles"}
ALL_STREAMS = ["food", "green", "wood", "paper", "textiles", "plastic", "metal", "glass", "tires", "other"]

# Every row-per-year range in the workbook runs 1960-2120, 161 rows.
EXCEL_FIRST_YEAR = 1960
EXCEL_LAST_YEAR = 2120
# Our engines stop at MODEL_END_YEAR; compare only years both cover.
COMPARE_LAST_YEAR = 2050
HOURS_PER_YEAR = 8760  # the workbook's own constant, leap years included

# Years where the workbook itself is wrong, so a gap there is not ours.
KNOWN_EXCEL_DEFECTS = {
    # 'I. CH4'!C64 (site #1, food, 2015) holds row 65's formula, so the
    # workbook's 2015 food methane is really 2016's. Every other cell in the
    # column, and the other sites and scenarios, are right.
    2015: "I. CH4!C64 holds the 2016 formula",
}


def _year_range(col: str, first_row: int) -> str:
    """The cells holding 1960-2120 in column ``col``, from ``first_row`` down."""
    return f"{col}{first_row}:{col}{first_row + EXCEL_LAST_YEAR - EXCEL_FIRST_YEAR}"


# --- Cases -------------------------------------------------------------------

COMPOSITION = {
    "food": 0.50, "green": 0.10, "wood": 0.03, "paper": 0.12, "textiles": 0.03,
    "plastic": 0.10, "metal": 0.03, "glass": 0.03, "tires": 0.01, "other": 0.05,
}

# The city-level inputs every case shares. Region only feeds defaults the
# composition overwrites; precipitation is set per case because it picks k.
CITY = {
    ("General Information", "C4"): "Engine check",
    ("General Information", "C5"): "Test",
    ("General Information", "C6"): "Northern Africa",
    ("General Information", "C7"): 500_000,
    ("General Information", "C8"): 0,
    ("General Information", "C9"): 2024,
    ("General Information", "C13"): 20,
    ("General Information", "C16"): 1.0,
    ("General Information", "C17"): 0,
    ("General Information", "C18"): 0.03,
    ("General Information", "C19"): 0.02,
    ("General Information", "C20"): 0.9,
    ("General Information", "C21"): 0,
}
CITY.update({("General Information", f"C{29 + i}"): COMPOSITION[s] for i, s in enumerate(ALL_STREAMS)})


def _site(open_year, close_year, site_type, depth, gas=False, gas_start=None, gas_to_energy=False):
    """Site #1's front-panel cells in 'Landfills and Dumpsites', baseline column C."""
    return {
        ("Landfills and Dumpsites", "C13"): "Site 1",
        ("Landfills and Dumpsites", "C14"): open_year,
        ("Landfills and Dumpsites", "C15"): close_year,
        ("Landfills and Dumpsites", "C16"): 100_000,  # only a share weight with one site
        ("Landfills and Dumpsites", "C17"): site_type,
        ("Landfills and Dumpsites", "C19"): depth,
        ("Landfills and Dumpsites", "C20"): "Yes" if gas else "No",
        ("Landfills and Dumpsites", "C21"): gas_start if gas else None,
        ("Landfills and Dumpsites", "C22"): "Yes" if gas_to_energy else "No",
    }


def _irregular_waste() -> dict[int, dict[str, float]]:
    """Uneven yearly tonnage 1978-2034: a ramp, a dip, a jump and a taper."""
    rng = np.random.default_rng(7)
    waste = {}
    for year in range(1978, 2035):
        base = 20_000 + 2_500 * (year - 1978)
        if 1995 <= year <= 1999:
            base *= 0.4
        if year >= 2015:
            base *= 1.8
        total = base * rng.uniform(0.85, 1.15)
        waste[year] = {s: total * COMPOSITION[s] for s in ALL_STREAMS}
    return waste


# Each case: precipitation (sets the workbook's k zone), front-panel site cells,
# and optional overrides pasted over the workbook's formulas.
CASES = {
    "dump_shallow_dry": dict(
        precip=400, site=_site(1990, 2030, "Dumpsite", 3),
    ),
    "controlled_deep_wet": dict(
        precip=1700, site=_site(1985, 2025, "Controlled Dumpsite", 10),
    ),
    "controlled_deep_gas": dict(
        precip=1200, site=_site(1980, 2035, "Controlled Dumpsite", 12, gas=True, gas_start=2010),
    ),
    "landfill_gas": dict(
        precip=1200, site=_site(1980, 2040, "Landfill", 20, gas=True, gas_start=2005),
    ),
    "landfill_closed_very_wet_gte": dict(
        precip=2500, site=_site(1975, 2010, "Landfill", 25, gas=True, gas_start=2000, gas_to_energy=True),
    ),
    # Pasted waste and a capture efficiency that steps up, as a real site's would.
    "landfill_override_waste_and_capture": dict(
        precip=800,
        site=_site(1978, 2034, "Landfill", 15, gas=True, gas_start=2008),
        waste=_irregular_waste(),
        collection={**{y: 0.3 for y in range(2008, 2015)}, **{y: 0.6 for y in range(2015, 2030)},
                    **{y: 0.75 for y in range(2030, EXCEL_LAST_YEAR + 1)}},
    ),
}


# --- Excel side --------------------------------------------------------------

def prepare_template(workbook: Path, out: Path) -> Path:
    """A copy of the workbook Excel can open and fill without asking anything.

    Excel asks about macros every time it opens the .xlsm, even hidden, and
    asks for a password to unlock a sheet; both questions land on whoever is
    at the screen. The macros are only reset and navigation buttons, and the
    locks only stop typing over formulas, so the copy drops both: the VBA part
    and its references go, the workbook becomes a plain .xlsx, and each
    sheet's <sheetProtection> element is removed. Formulas and values are
    untouched. Only the copy is changed.
    """
    with zipfile.ZipFile(workbook) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            if item.filename == "xl/vbaProject.bin":
                continue
            data = src.read(item.filename)
            if item.filename == "[Content_Types].xml":
                text = data.decode("utf-8")
                text = re.sub(r'<Override PartName="/xl/vbaProject\.bin"[^>]*/>', "", text)
                text = text.replace(
                    "application/vnd.ms-excel.sheet.macroEnabled.main+xml",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
                )
                data = text.encode("utf-8")
            elif item.filename == "xl/_rels/workbook.xml.rels":
                data = re.sub(rb'<Relationship [^>]*Target="vbaProject\.bin"[^>]*/>', b"", data)
            elif item.filename.startswith("xl/worksheets/") and item.filename.endswith(".xml"):
                data = re.sub(rb"<sheetProtection [^>]*/>", b"", data)
            dst.writestr(item, data)
    return out


def _check_pasted(book, case: dict, years: list[int]) -> None:
    """Stop if Excel did not keep what was pasted, rather than compare the wrong thing."""
    if "waste" in case:
        got = np.array(book.sheets["I. Waste"].range(_year_range("E", 7)).value, dtype=float)
        want = np.array([case["waste"].get(y, {}).get("food", 0.0) for y in years])
        if not np.allclose(got, want, rtol=1e-9):
            raise RuntimeError("Excel did not keep the pasted waste in 'I. Waste'")
    if "collection" in case:
        got = np.array(book.sheets["CH4 Collect"].range(_year_range("C", 7)).value, dtype=float)
        want = np.array([case["collection"].get(y, 0.0) for y in years])
        if not np.allclose(got, want, rtol=1e-9):
            raise RuntimeError("Excel did not keep the pasted collection efficiency in 'CH4 Collect'")


def run_excel(app, template: Path, name: str, case: dict, keep: bool) -> pd.DataFrame:
    """Fill a fresh copy of the workbook with one case and read back what it used."""
    EXCEL_WORK_DIR.mkdir(parents=True, exist_ok=True)
    path = EXCEL_WORK_DIR / f"{name}{template.suffix}"
    shutil.copyfile(template, path)
    book = app.books.open(str(path))
    try:
        # Recalculating the 161-year array formulas after every write is slow
        # enough that a later command times out; calculate once, at the end.
        app.calculation = "manual"
        cells ={**CITY, ("General Information", "C12"): case["precip"], **case["site"]}
        for (sheet, cell), value in cells.items():
            book.sheets[sheet].range(cell).value = value

        years = list(range(EXCEL_FIRST_YEAR, EXCEL_LAST_YEAR + 1))
        # The template has no sheet locks (see prepare_template); a write to a
        # locked cell would be dropped without an error.
        if "waste" in case:
            rows = [[case["waste"].get(y, {}).get(s, 0.0) for s in ALL_STREAMS] for y in years]
            book.sheets["I. Waste"].range("E7").value = rows
        if "collection" in case:
            book.sheets["CH4 Collect"].range("C7").options(transpose=True).value = [
                case["collection"].get(y, 0.0) for y in years
            ]
        app.calculate()
        _check_pasted(book, case, years)

        def column(sheet, col, first_row):
            values = book.sheets[sheet].range(_year_range(col, first_row)).value
            return np.array([np.nan if v is None or isinstance(v, str) else v for v in values], dtype=float)

        ch4 = book.sheets["I. CH4"]
        emit = book.sheets["CH4 Emit"]
        k = dict(zip(STREAMS, ch4.range("C5:G5").value))
        l0 = dict(zip(STREAMS, ch4.range("C6:G6").value))
        mcf = ch4.range("E4").value
        ox_with, ox_without = emit.range("C3").value, emit.range("C4").value
        destruction = emit.range("C5").value
        gas_start = book.sheets["Landfills and Dumpsites"].range("C21").value or 0

        frame = pd.DataFrame({"year": years})
        waste_cols = {"food": "E", "green": "F", "wood": "G", "paper": "H", "textiles": "I"}
        for stream, col in waste_cols.items():
            frame[f"waste_{stream}"] = column("I. Waste", col, 7)
        for stream in STREAMS:
            frame[f"k_{stream}"] = k[stream]
            frame[f"l0_{stream}"] = l0[stream]
        frame["mcf"] = mcf
        frame["collection_efficiency"] = column("CH4 Collect", "C", 7)
        # 'CH4 Emit'!D10 picks the oxidation by comparing the year to the gas
        # start-up year, whether or not the site has gas collection.
        frame["oxidation"] = np.where(frame["year"] < gas_start, ox_without, ox_with)
        frame["destruction_efficiency"] = destruction
        frame["excel_generated_m3"] = column("I. CH4", "H", 9)
        frame["excel_generated_adjusted_m3"] = column("I. CH4", "I", 9)
        frame["excel_emitted_m3"] = column("CH4 Emit", "D", 10) * HOURS_PER_YEAR
        frame.insert(0, "case", name)
        if keep:
            book.save()
        return frame
    finally:
        book.close()
        if not keep:
            path.unlink(missing_ok=True)


# --- Python side -------------------------------------------------------------

def run_python(excel: pd.DataFrame) -> pd.DataFrame:
    """Run both engines on exactly the values Excel used; yearly m3 CH4.

    Excel's columns start in 1960; our engines start at the site's first year of
    waste, which every case keeps at or after MODEL_START_YEAR (1970).
    """
    frame = excel.set_index("year")
    deposited = frame[[f"waste_{s}" for s in STREAMS]].sum(axis=1)
    open_year = int(deposited[deposited > 0].index.min())
    years = frame.index[frame.index <= COMPARE_LAST_YEAR]
    modelled = years[years >= open_year]

    for stream, ours in STREAMS.items():
        excel_l0 = frame[f"l0_{stream}"].iloc[0]
        if not np.isclose(defaults_2019.L_0[ours], excel_l0):
            raise ValueError(f"L0 for {stream}: Excel {excel_l0}, SWEET_python {defaults_2019.L_0[ours]}")

    landfill = {
        "open_date": open_year,
        "close_date": None,
        "ks": {ours: frame.loc[years, f"k_{s}"].copy() for s, ours in STREAMS.items()},
        "waste_mass_df": pd.DataFrame({ours: frame.loc[years, f"waste_{s}"] for s, ours in STREAMS.items()}),
        "mcf": frame.loc[years, "mcf"],
        "gas_capture_efficiency": frame.loc[years, "collection_efficiency"],
        "oxidation_factor": frame.loc[years, "oxidation"],
        "flaring": frame.loc[years, "destruction_efficiency"],
    }
    city = {"components": list(STREAMS.values())}
    # The annual engine reads these but no longer uses them: waste comes in whole.
    params = {"year_of_data_pop": None, "growth_rate_historic": None, "growth_rate_future": None}

    # Each engine gets its own ks dict: the monthly one reassigns its entries.
    _, emitted_a, generated_a, _ = SWEET(city, params, dict(landfill, ks=dict(landfill["ks"]))).estimate_emissions2()
    _, emitted_m, generated_m, _ = SWEET(city, params, dict(landfill, ks=dict(landfill["ks"]))).estimate_emissions_monthly()

    out = pd.DataFrame(index=pd.Index(modelled, name="year"))
    out["annual_generated_m3"] = generated_a.sum(axis=1)
    out["annual_emitted_m3"] = emitted_a["total"]
    out["monthly_generated_m3"] = generated_m.groupby(generated_m.index.year).sum().sum(axis=1)
    out["monthly_emitted_m3"] = emitted_m["total"].groupby(emitted_m.index.year).sum()
    return out.reset_index()


# --- Comparison --------------------------------------------------------------

def excel_emitted_with_slip_oxidised(frame: pd.DataFrame) -> pd.Series:
    """Excel's emissions, re-done the way our engines treat flare slip.

    Excel ('CH4 Emit'!D10) emits the gas the flare fails to destroy as is:
    generated x (1 - capture) x (1 - oxidation) + generated x capture x (1 - DE).
    Ours oxidise that slip too: (generated - generated x capture x DE) x (1 - oxidation).
    The difference is generated x capture x (1 - DE) x oxidation; take it off
    Excel's number and the engines should agree exactly.
    """
    slip = frame["excel_generated_m3"] * frame["collection_efficiency"] * (1 - frame["destruction_efficiency"])
    return frame["excel_emitted_m3"] - slip * frame["oxidation"]


def compare(excel: pd.DataFrame) -> pd.DataFrame:
    """Excel and both engines side by side, one row per year the site is modelled."""
    ours = run_python(excel)
    side = excel.merge(ours, on="year", how="inner")
    side["excel_emitted_slip_oxidised_m3"] = excel_emitted_with_slip_oxidised(side)
    side["excel_defect"] = side["year"].map(KNOWN_EXCEL_DEFECTS).fillna("")

    def pct(ours_col, excel_col):
        return np.where(side[excel_col] > 0, 100 * (side[ours_col] / side[excel_col] - 1), np.nan)

    for engine in ("annual", "monthly"):
        for quantity in ("generated", "emitted"):
            side[f"{engine}_{quantity}_pct"] = pct(f"{engine}_{quantity}_m3", f"excel_{quantity}_m3")
    side["annual_emitted_unexplained_pct"] = pct("annual_emitted_m3", "excel_emitted_slip_oxidised_m3")
    return side


def summarise(side: pd.DataFrame) -> dict:
    """Cumulative and worst-year gaps, over years with real emissions."""
    row = {"case": side["case"].iloc[0]}
    # Ignore the first sliver of a site's life, where both sides are near zero
    # and a percentage means nothing, and years where Excel is known to be wrong.
    live = side[
        (side["excel_generated_m3"] > 0.01 * side["excel_generated_m3"].max())
        & (side["excel_defect"] == "")
    ]
    for engine in ("annual", "monthly"):
        for quantity in ("generated", "emitted"):
            ours, theirs = live[f"{engine}_{quantity}_m3"], live[f"excel_{quantity}_m3"]
            row[f"{engine}_{quantity}_total_pct"] = 100 * (ours.sum() / theirs.sum() - 1)
            row[f"{engine}_{quantity}_worst_year_pct"] = live[f"{engine}_{quantity}_pct"].abs().max()
    row["annual_emitted_unexplained_worst_year_pct"] = live["annual_emitted_unexplained_pct"].abs().max()
    for year in KNOWN_EXCEL_DEFECTS:
        hit = side[side["year"] == year]
        if len(hit):
            row[f"excel_{year}_defect_pct"] = float(hit["annual_generated_pct"].iloc[0])
    row["excel_emitted_t_2025"] = float(
        side.loc[side["year"] == 2025, "excel_emitted_m3"].iloc[0] * CH4_DENSITY_KG_PER_M3 / 1000
    )
    return row


FIXTURE_COLUMNS = (
    ["year"]
    + [f"waste_{s}" for s in STREAMS]
    + [f"k_{s}" for s in STREAMS]
    + [f"l0_{s}" for s in STREAMS]
    + ["mcf", "collection_efficiency", "oxidation", "destruction_efficiency",
       "excel_generated_m3", "excel_emitted_m3"]
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workbook", required=True, type=Path, help="SWEET .xlsm (only read; copied first)")
    parser.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    parser.add_argument("--write-fixtures", action="store_true", help=f"save Excel results to {FIXTURE_DIR.relative_to(REPO)}")
    parser.add_argument("--keep-workbooks", action="store_true", help=f"keep each filled workbook in {EXCEL_WORK_DIR}")
    args = parser.parse_args(argv)

    import xlwings as xw

    OUT_DIR.mkdir(exist_ok=True)
    EXCEL_WORK_DIR.mkdir(parents=True, exist_ok=True)
    template = prepare_template(args.workbook.expanduser(), EXCEL_WORK_DIR / "template.xlsx")

    app = xw.App(visible=False, add_book=False)
    app.display_alerts = False
    app.screen_updating = False
    try:
        sides = []
        for name in args.cases:
            print(f"{name} ...", flush=True)
            excel = run_excel(app, template, name, CASES[name], args.keep_workbooks)
            if args.write_fixtures:
                FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
                fixture = excel[(excel["year"] <= COMPARE_LAST_YEAR)][FIXTURE_COLUMNS]
                fixture.to_csv(FIXTURE_DIR / f"{name}.csv", index=False, float_format="%.10g")
            sides.append(compare(excel))
    finally:
        # quit() leaves a hidden instance running on macOS; kill() ends only
        # the one this script started, not an Excel the user has open.
        app.kill()
        template.unlink(missing_ok=True)

    side = pd.concat(sides, ignore_index=True)
    side.to_csv(OUT_DIR / "by_year.csv", index=False)
    summary = pd.DataFrame([summarise(s) for s in sides])
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(summary.set_index("case").T)
    print(f"\nWrote {OUT_DIR / 'by_year.csv'} and {OUT_DIR / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
