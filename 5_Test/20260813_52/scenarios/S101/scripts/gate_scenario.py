from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["comid", "q_site", "year", "month", "fold_id"]
FOLDS = [
    (2012, 2013, "fit_2006_2011_eval_2012_2013"),
    (2014, 2015, "fit_2006_2013_eval_2014_2015"),
    (2016, 2018, "fit_2006_2015_eval_2016_2018"),
]


def main() -> None:
    scenario = ROOT.name
    excluded = set(pd.read_csv(ROOT / "inputs" / "excluded_stations.csv", encoding="utf-8-sig").q_site.astype(str))
    panel = pd.read_parquet(ROOT / "inputs" / "parent_indata.parquet")
    oof = pd.read_parquet(ROOT / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet")
    expected_parts = []
    for start, end, fold_id in FOLDS:
        p = panel.loc[
            panel.year.between(start, end) & panel.Q_obsv_cfs.notna() & panel.Q_obsv_cfs.gt(0),
            ["comid", "q_site", "year", "month"],
        ].copy()
        p["q_site"] = p.q_site.astype(str)
        p["fold_id"] = fold_id
        expected_parts.append(p)
    expected = pd.concat(expected_parts, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    actual = oof[KEY].sort_values(KEY).reset_index(drop=True)
    excluded_rows = int(oof.q_site.astype(str).isin(excluded).sum())
    protected_rows = int(oof.q_site.astype(str).eq("石角站").sum())
    passed = (
        len(panel) == 46920 and panel.comid.nunique() == 230 and len(oof) == len(expected)
        and actual.equals(expected) and excluded_rows == 0 and protected_rows > 0
    )
    payload = {
        "terminal": "FILTERED_INPUT_AND_OOF_POPULATION_PASS" if passed else "FILTERED_INPUT_AND_OOF_POPULATION_FAILURE",
        "scenario_id": scenario,
        "oof_rows": len(oof),
        "expected_oof_rows": len(expected),
        "oof_stations": int(oof.q_site.nunique()),
        "excluded_rows": excluded_rows,
        "protected_shijiao_rows": protected_rows,
        "forcing_rows": len(panel),
        "reaches": int(panel.comid.nunique()),
    }
    (ROOT / "reports" / "reproduction_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
