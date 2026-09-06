from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


ROOT = Path(__file__).resolve().parents[1]
FOLDS = [
    "fit_2006_2011_eval_2012_2013",
    "fit_2006_2013_eval_2014_2015",
    "fit_2006_2015_eval_2016_2018",
]


def read_fixed(scenario: str, fold: str) -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "outputs" / scenario / "blocked_folds" / fold / "reports"
        / "monthly_bayes_seasonal_hysteresis_fixed_parameters.csv",
        encoding="utf-8-sig",
    )


def main() -> None:
    assert_sparrow_runtime()
    rows = []
    for fold in FOLDS:
        paths = {
            scenario: ROOT / "outputs" / scenario / "blocked_folds" / fold / "reports"
            / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.parquet"
            for scenario in ("A0", "A1")
        }
        frames = {scenario: pd.read_parquet(path) for scenario, path in paths.items()}
        a0, a1 = frames["A0"], frames["A1"]
        keys = ["comid", "q_site", "year", "month", "split"]
        paired = a0[keys + ["upstream_positive_input_equivalent_cfs"]].merge(
            a1[keys + ["upstream_positive_input_equivalent_cfs"]],
            on=keys,
            suffixes=("_a0", "_a1"),
            validate="one_to_one",
        )
        x0 = paired.upstream_positive_input_equivalent_cfs_a0.to_numpy(float)
        x1 = paired.upstream_positive_input_equivalent_cfs_a1.to_numpy(float)
        if not np.allclose(x0, x1, atol=0, rtol=0):
            raise RuntimeError(f"Unscaled forcing differs between A0/A1 in {fold}")
        raw0, raw1 = np.log1p(0.35 * x0), np.log1p(x1)

        std = {}
        for scenario in ("A0", "A1"):
            standardization = pd.read_csv(
                ROOT / "outputs" / scenario / "blocked_folds" / fold / "reports" / "design_standardization.csv",
                encoding="utf-8-sig",
            ).set_index("feature")
            mean = float(standardization.loc["log_qcalc", "training_mean"])
            scale = float(standardization.loc["log_qcalc", "training_std"])
            std[scenario] = ((raw0 if scenario == "A0" else raw1) - mean) / scale

        coef = {}
        for scenario in ("A0", "A1"):
            fixed = read_fixed(scenario, fold).set_index("parameter")
            coef[scenario] = float(fixed.loc["log_qcalc", "coefficient_standardized"])
        rows.append(
            {
                "fold_id": fold,
                "paired_rows": int(len(paired)),
                "unscaled_input_max_abs_difference": float(np.max(np.abs(x0 - x1))),
                "raw_log1p_max_abs_difference": float(np.max(np.abs(raw1 - raw0))),
                "raw_log1p_mean_abs_difference": float(np.mean(np.abs(raw1 - raw0))),
                "standardized_log1p_max_abs_difference": float(np.max(np.abs(std["A1"] - std["A0"]))),
                "standardized_log1p_mean_abs_difference": float(np.mean(np.abs(std["A1"] - std["A0"]))),
                "a0_log_qcalc_fixed_coefficient": coef["A0"],
                "a1_log_qcalc_fixed_coefficient": coef["A1"],
                "coefficient_difference_a1_minus_a0": float(coef["A1"] - coef["A0"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "reports" / "scale_transform_and_coefficient_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "rows": int(len(out)),
        "folds": FOLDS,
        "a0_a1_unscaled_input_exact": bool(out.unscaled_input_max_abs_difference.eq(0).all()),
        "raw_and_standardized_differences_quantified": True,
        "fixed_coefficient_differences_quantified": True,
    }
    (ROOT / "reports" / "scale_transform_and_coefficient_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
