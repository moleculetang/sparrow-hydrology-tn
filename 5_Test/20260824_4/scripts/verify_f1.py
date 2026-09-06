from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260824_4")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
OLD_CACHE = Path(r"E:\SPARROW\5_Test\20260820_19\cache\temporal")
MUS = (12, 36, 60, 96, 144, 240)
MODELS = tuple(sorted([f"S0_mu_{m:03d}m" for m in MUS] + [f"S1_tau_012m_mu_{m:03d}m" for m in MUS]))
KEYS = ["station_key", "year", "month", "fold_id", "layer"]


def main() -> None:
    pred = pd.read_parquet(OUT / "f1_temporal_oof_predictions.parquet")
    par = pd.read_parquet(OUT / "f1_fold_parameters.parquet")
    gates = pd.read_parquet(OUT / "f1_paired_simultaneous_gates.parquet")
    synthetic = json.loads((REPORTS / "f1_synthetic_identifiability.json").read_text(encoding="utf-8"))
    decision = json.loads((REPORTS / "f1_decision.json").read_text(encoding="utf-8"))
    counts = pred.groupby(["model_id", "layer", "arm"], observed=True).size()
    expected_groups = 12 * 2 * 4
    key_contract = bool(len(counts) == expected_groups and counts.eq(3895).all())
    duplicate_contract = not pred.duplicated(["model_id", "arm", *KEYS]).any()
    time_contract = set(pred["year"].unique()) == {2018, 2019, 2020, 2021}

    max_abs = 0.0
    max_rel = 0.0
    rows = []
    gaussian = pred.loc[pred["arm"].eq("GAUSSIAN_PARENT")]
    for model_id in MODELS:
        old = pd.read_parquet(OLD_CACHE / f"{model_id}__predictions.parquet")
        old = old.loc[old["mechanism"].eq("H1_GLOBAL"), [*KEYS, "pred_tn_mg_l"]]
        new = gaussian.loc[gaussian["model_id"].eq(model_id), [*KEYS, "pred_tn_mg_l"]]
        joined = new.merge(old, on=KEYS, suffixes=("_new", "_old"), validate="one_to_one")
        diff = np.abs(joined["pred_tn_mg_l_new"] - joined["pred_tn_mg_l_old"])
        rel = diff / np.maximum(np.abs(joined["pred_tn_mg_l_old"]), 1.0)
        rows.append({"model_id": model_id, "rows": len(joined), "max_abs": float(diff.max()), "max_relative": float(rel.max())})
        max_abs = max(max_abs, float(diff.max()))
        max_rel = max(max_rel, float(rel.max()))
    reproduction = pd.DataFrame(rows)
    reproduction.to_parquet(OUT / "f1_gaussian_parent_reproduction.parquet", index=False)

    hinge = par.loc[par["arm"].eq("GAUSSIAN_CQ_HINGE")]
    beta_summary = hinge.groupby(["model_id", "layer"], observed=True).agg(
        beta_low_median=("beta_low", "median"),
        beta_high_median=("beta_high", "median"),
        beta_low_min=("beta_low", "min"),
        beta_low_max=("beta_low", "max"),
        beta_high_min=("beta_high", "min"),
        beta_high_max=("beta_high", "max"),
        successful_folds=("success", "sum"),
    ).reset_index()
    beta_summary.to_parquet(OUT / "f1_hinge_parameter_stability.parquet", index=False)
    max_t_valid = bool(
        gates["bootstrap_standard_error"].gt(0).all()
        and gates["max_t_critical_value"].between(1.0, 10.0).all()
        and np.isfinite(gates["simultaneous_ci95_upper"]).all()
    )
    optimizer = bool(par["success"].all())
    boundary = bool(not par["eta_boundary"].groupby(par["model_id"]).sum().ge(2).any() and not par["beta_boundary"].groupby(par["model_id"]).sum().ge(2).any())
    report = {
        "status": "PASS"
        if all(
            [
                key_contract,
                duplicate_contract,
                time_contract,
                # F1 independently refits the same frozen readout at tighter
                # optimizer tolerances, so historical reproduction is judged
                # at the registered numerical tolerance rather than bitwise.
                max_rel <= 1e-6,
                synthetic["status"] == "PASS",
                optimizer,
                boundary,
                max_t_valid,
                decision["status"] == "F1_MEAN_LAYER_TEMPORAL_UPGRADE_SUPPORTED",
            ]
        )
        else "FAIL",
        "checks": {
            "96_model_layer_arm_groups": key_contract,
            "3895_keys_per_group": bool(counts.eq(3895).all()),
            "no_duplicate_keys": duplicate_contract,
            "OOF_years_are_2018_2021": time_contract,
            "gaussian_parent_max_abs_reproduction_mg_l": max_abs,
            "gaussian_parent_max_relative_reproduction": max_rel,
            "synthetic_gate": synthetic["status"],
            "all_optimizers_success": optimizer,
            "no_ensemble_boundary_confounding": boundary,
            "studentized_max_t_fields_valid": max_t_valid,
            "formal_decision": decision["status"],
        },
        "scientific_boundary": "The Gaussian C-Q hinge is a low-dimensional Q72-driven mean-layer upgrade. It does not identify a terrestrial, groundwater or aquatic reaction process and does not establish station-blind spatial transfer.",
        "TN_2022_values_read": False,
    }
    (REPORTS / "f1_independent_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report["status"] != "PASS":
        raise RuntimeError(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
