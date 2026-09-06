from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
OLD = ROOT / "5_Test" / "previous_mainline" / "reports" / "reach_class_light_constraint_summary.csv"
NEW = ROOT / "5_Test" / "20260608_1" / "reports" / "main_model" / "reach_class_light_constraint_summary.csv"
OUT = ROOT / "5_Test" / "20260608_1" / "reports" / "reproducibility_comparison.csv"


def main() -> None:
    old = pd.read_csv(OLD, encoding="utf-8-sig").iloc[0]
    new = pd.read_csv(NEW, encoding="utf-8-sig").iloc[0]
    fields = [
        "validation_stations",
        "class_median_NSElog",
        "class_median_KGE",
        "class_median_absPBIAS",
        "class_good_count",
        "class_median_alpha",
        "class_mean_alpha",
        "distance_to_mass_reduction_pct_vs_alpha0",
        "alpha0_median_NSElog",
        "global01_median_NSElog",
        "station_adaptive_median_NSElog",
        "mass_alpha1_median_NSElog",
    ]
    rows = []
    for field in fields:
        old_value = old[field]
        new_value = new[field]
        try:
            delta = float(new_value) - float(old_value)
        except Exception:
            delta = ""
        rows.append({"metric": field, "original_previous_mainline": old_value, "clean_20260608_1": new_value, "delta": delta})
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(OUT)


if __name__ == "__main__":
    main()
