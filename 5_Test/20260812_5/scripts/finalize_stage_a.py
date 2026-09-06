from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    gate = json.loads((ROOT / "terminal_gate.json").read_text(encoding="utf-8"))
    reproduction = json.loads((ROOT / "logs" / "A0_exact_reproduction_gate.json").read_text(encoding="utf-8"))
    required = [
        "README.md", "experiment_contract.md", "terminal_gate.json",
        "inputs/scenarios/I0_indata.parquet", "inputs/topology/topology_edges.csv",
        "inputs/canonical_signal_registry.csv", "inputs/confirmed_evaluable_nearest_downstream_paths.csv",
        "scripts/components/q72_input_scale_component.py", "scripts/run_stage_a.py",
        "scripts/evaluate_stage_a.py", "scripts/tests/test_stage_a_contract.py",
        "outputs/A0/q72_three_fold_oof_predictions.parquet",
        "outputs/A1/q72_three_fold_oof_predictions.parquet",
        "reports/scenario_metrics.csv", "reports/fold_metrics.csv",
        "reports/station_metrics.csv", "reports/topology_path_metrics.csv",
        "reports/legacy_lowflow_metrics.csv", "reports/feature_accounting_audit.csv",
        "reports/scale_transform_and_coefficient_audit.csv",
        "reports/scale_transform_and_coefficient_audit.json",
        "scripts/audit_scale_transform.py",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    audit = {
        "required_files_present": not missing,
        "a0_exact_reproduction_pass": bool(reproduction.get("pass")),
        "promotion_gates_pass": bool(gate.get("all_promotion_gates_pass")),
        "legacy_evaluable_targets_27": gate.get("legacy_evaluable_targets") == 27,
        "terminal_correct": gate.get("terminal") == "EMPIRICAL_035_SCALE_REMOVED_WITH_PREDICTIVE_EQUIVALENCE",
        "scale_transform_and_coefficients_quantified": bool(
            (ROOT / "reports" / "scale_transform_and_coefficient_audit.csv").is_file()
            and (ROOT / "reports" / "scale_transform_and_coefficient_audit.json").is_file()
        ),
        "manifest_file_count": None,
    }
    audit["all_pass"] = all(value for key, value in audit.items() if key != "manifest_file_count")
    metrics = pd.read_csv(ROOT / "reports" / "scenario_metrics.csv", encoding="utf-8-sig").set_index("scenario")
    a0, a1 = metrics.loc["A0"], metrics.loc["A1"]
    readme = f"""# 20260812_5 — Q72 empirical 0.35 scale removal

## Terminal result

```text
EMPIRICAL_035_SCALE_REMOVED_WITH_PREDICTIVE_EQUIVALENCE
```

The undocumented `0.35` has been removed from the forcing-only network-input predictor. A1 retains the unscaled upstream positive-input-equivalent volume rate and its fold-pure climatology as statistical covariates. Neither is called simulated streamflow, connectivity, recharge or loss.

## Reproduction and population

- A0 reproduced `20260812_3/I0`: 8,738/8,738 keys, observed maximum difference 0, prediction maximum difference 0.
- 46,920 reach-month forcing rows, 230 reaches and 204 months per reach were retained.
- OOF fold rows remain 2,434, 2,582 and 3,722.
- 28 historical registry members were audited; 27 are evaluable. Tianhe remains the pre-existing non-evaluable member and was not added.

## Metrics

| scenario | raw NSE | log-NSE | PBIAS | log-RMSE |
|---|---:|---:|---:|---:|
| A0 fixed 0.35 | {a0.raw_nse:.9f} | {a0.log_nse:.9f} | {a0.pbias_pct:+.4f}% | {a0.log_rmse:.9f} |
| A1 no 0.35 | {a1.raw_nse:.9f} | {a1.log_nse:.9f} | {a1.pbias_pct:+.4f}% | {a1.log_rmse:.9f} |

A1 minus A0:

- raw NSE: {a1.raw_nse-a0.raw_nse:+.9f};
- log-NSE: {a1.log_nse-a0.log_nse:+.9f};
- high-flow log-RMSE: {gate['highflow_logrmse_change_pct']:+.4f}%;
- large-station log-RMSE: {gate['large_station_logrmse_change_pct']:+.4f}%;
- 21-path downstream MAE: {gate['path_mean_downstream_mae_change_pct']:+.4f}%;
- absolute PBIAS: {gate['abs_pbias_change_points']:+.4f} percentage points.

All pre-registered prediction and accounting gates passed. This establishes that `0.35` is an unnecessary empirical scale in this predictor path; it does not establish that the unscaled input equals runoff.

## Transform and coefficient audit

The A0/A1 unscaled forcing arrays are exactly identical. Removing the scalar changes
`log1p` before standardization, as expected: the maximum raw difference is `1.04982`.
After fold-specific standardization, the maximum difference is only `0.15876鈥?.16168`
and the mean absolute difference is `0.03567鈥?.03677`. The fitted standardized
`log_qcalc` coefficient also changes rather than being hidden by a rename:

| fold | A0 coefficient | A1 coefficient | A1鈭扐0 |
|---|---:|---:|---:|
| 2012鈥?013 | -0.166065 | -0.059724 | +0.106341 |
| 2014鈥?015 | -0.098532 | -0.041845 | +0.056686 |
| 2016鈥?018 | -0.011408 | -0.057591 | -0.046182 |

The complete paired audit is `reports/scale_transform_and_coefficient_audit.csv`.

## Reproduce

```powershell
conda --no-plugins run -n sparrow python scripts/tests/test_stage_a_contract.py
conda --no-plugins run -n sparrow python scripts/run_stage_a.py all
conda --no-plugins run -n sparrow python scripts/evaluate_stage_a.py
conda --no-plugins run -n sparrow python scripts/audit_scale_transform.py
conda --no-plugins run -n sparrow python scripts/finalize_stage_a.py
```

All code and data needed for these runs are inside this directory. `20260812_3` was used read-only only for the A0 equality gate.
"""
    # README is a maintained frozen narrative. Do not overwrite it during a
    # manifest-only refinalization; the computed values above are still used
    # by the terminal gate and machine-readable reports.
    # Freeze the audit before hashing.  The manifest excludes only itself; no
    # file is rewritten after its hash enters the manifest.
    pre_manifest_files = [path for path in ROOT.rglob("*") if path.is_file() and path.name != "input_manifest.json"]
    audit["manifest_file_count"] = len(pre_manifest_files)
    (ROOT / "reports" / "completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit["all_pass"]:
        raise RuntimeError(audit)
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "input_manifest.json":
            rows.append({
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "root": str(ROOT),
        "terminal": gate["terminal"],
        "missing_required": missing,
        "a0_exact_reproduction": reproduction,
        "file_count": len(rows),
        "files": rows,
    }
    (ROOT / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if audit["manifest_file_count"] != len(rows):
        raise RuntimeError(f"Manifest population changed during freeze: {audit['manifest_file_count']} != {len(rows)}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
