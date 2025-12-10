"""
Quick consistency check between SWEET.estimate_emissions2 (annual) and
SWEET.estimate_emissions_monthly (monthly). We build a simple synthetic
landfill input, run both methods, aggregate the monthly results to annual,
and report the maximum absolute difference per column.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys

def _prefer_local_sweet_python() -> None:
    """
    Ensure a local checkout of SWEET_python takes precedence over the git-installed package.
    """

    candidates = []

    env_path = os.getenv("SWEET_PYTHON_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    backend_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            backend_dir / "SWEET_python",
            backend_dir.parent / "SWEET_python",
            backend_dir.parent.parent / "SWEET_python",
        ]
    )

    for location in candidates:
        if not location:
            continue

        try:
            resolved = location.expanduser().resolve()
        except FileNotFoundError:
            continue

        if resolved.is_dir():
            if resolved.name == "SWEET_python" and (resolved / "__init__.py").is_file():
                package_root = resolved.parent
            elif (resolved / "SWEET_python/__init__.py").is_file():
                package_root = resolved
            else:
                continue

            if str(package_root) not in sys.path:
                sys.path.insert(0, str(package_root))
            return


_prefer_local_sweet_python()
print("SWEET_python path preference applied", file=sys.stdout, flush=True)

from SWEET_python.model_v2 import SWEET


def _years(start: int = 1990, end: int = 2050) -> np.ndarray:
    return np.arange(start, end + 1)


def _series(years: np.ndarray, value: float) -> pd.Series:
    return pd.Series(np.full(len(years), value, dtype=float), index=years)


def build_fixture():
    years = _years()
    compare_through = 2050
    components = ["food", "green", "wood", "paper_cardboard", "textiles"]

    # Constant annual waste mass by component (tons/year) — moderate values
    waste_mass_df = pd.DataFrame(
        {
            "food": 1.0e5,
            "green": 2.0e4,
            "wood": 1.5e4,
            "paper_cardboard": 4.0e4,
            "textiles": 1.0e4,
        },
        index=years,
    )

    # Component-specific k values (per year)
    ks = {
        "food": _series(years, 0.05),
        "green": _series(years, 0.045),
        "wood": _series(years, 0.035),
        "paper_cardboard": _series(years, 0.03),
        "textiles": _series(years, 0.025),
    }

    city_instance_attrs = {"components": components}
    city_params_dict = {
        "year_of_data_pop": 2025,
        "growth_rate_historic": 0.02,
        "growth_rate_future": 0.02,
    }

    landfill_instance_attrs = {
        "open_date": 1990,
        "close_date": compare_through,
        "ks": ks,
        "waste_mass_df": waste_mass_df,
        "mcf": _series(years, 1.0),
        "gas_capture_efficiency": _series(years, 0.0),
        "oxidation_factor": _series(years, 0.0),
        "flaring": _series(years, 1.0),
    }

    model = SWEET(
        city_instance_attrs=city_instance_attrs,
        city_params_dict=city_params_dict,
        landfill_instance_attrs=landfill_instance_attrs,
    )
    return model


def run_check(tol_pct: float = 1.0) -> bool:
    """
    Run consistency check between annual and monthly models.
    
    Args:
        tol_pct: Maximum allowed percent difference (default 1.0%).
                 Small differences are expected due to:
                 - Variable month lengths in monthly model
                 - Different treatment of first-year emissions
    """
    model = build_fixture()

    _, annual_q_df, _, _ = model.estimate_emissions2()
    _, monthly_q_df, _, _ = model.estimate_emissions_monthly()

    # Restrict comparison to years 1991-2050 (exclude first year where annual is 0)
    annual_q_df = annual_q_df.loc[(annual_q_df.index >= 1991) & (annual_q_df.index <= 2050)]

    # Aggregate monthly results to annual
    monthly_annual = monthly_q_df.resample("YE").sum()
    monthly_annual.index = monthly_annual.index.year
    monthly_annual = monthly_annual.loc[(monthly_annual.index >= 1991) & (monthly_annual.index <= 2050)]
    monthly_annual = monthly_annual.loc[annual_q_df.index]

    # Align columns and compute absolute differences
    monthly_annual = monthly_annual[annual_q_df.columns]
    diff = (monthly_annual - annual_q_df).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_diff = (diff / annual_q_df.replace(0, np.nan)) * 100.0

    max_diff_by_col = diff.max()
    max_diff = max_diff_by_col.max()

    max_pct_by_col = pct_diff.max()
    max_pct = max_pct_by_col.max()

    print("Max abs diff by column:")
    print(max_diff_by_col)
    print(f"Global max abs diff: {max_diff}")
    print("\nMax percent diff by column (relative to annual):")
    print(max_pct_by_col)
    print(f"Global max percent diff: {max_pct}%")

    # Debug: show first few years ratios for selected columns
    sample_cols = ["food", "total"]
    ratios = monthly_annual[sample_cols] / annual_q_df[sample_cols]

    # Quick side-by-side debug for early years
    debug_years = annual_q_df.index[:5]
    print("\nDebug (first 5 years, annual vs monthly-agg):")
    for col in ["food", "total"]:
        print(f"{col}:")
        print(
            pd.DataFrame(
                {
                    "annual": annual_q_df.loc[debug_years, col],
                    "monthly_agg": monthly_annual.loc[debug_years, col],
                    "ratio": ratios.loc[debug_years, col],
                }
            )
        )

    print("\nSample ratios (monthly/annual) first 5 years:")
    print(ratios.head())

    # Debug: sum comparison overall
    print("\nTotal sums comparison (all years):")
    print("annual totals:", annual_q_df.sum())
    print("monthly aggregated totals:", monthly_annual.sum())

    # Use percent difference for tolerance check
    print(f"\nMax percent diff: {max_pct:.4f}%")
    return bool(max_pct <= tol_pct)


if __name__ == "__main__":
    ok = run_check()
    if ok:
        print("PASS: monthly and annual q_df match within tolerance.")
    else:
        print("FAIL: discrepancies exceed tolerance.")

