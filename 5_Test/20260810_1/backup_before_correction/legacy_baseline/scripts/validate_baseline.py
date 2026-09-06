from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
CONFIG = json.loads((RUN / "config.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).replace(" ", "").replace("\u3000", "")
    return text.replace("(", "（").replace(")", "）").strip()


def relaxed(value: object) -> str:
    text = re.sub(r"_\d+$", "", norm_name(value)).replace("（重复）", "")
    return text[:-1] if text.endswith("站") else text


def main() -> None:
    panel = pd.read_parquet(RUN / "inputs" / "indata.parquet")
    selected = pd.read_csv(RUN / "inputs" / "source_metadata" / "same_reach_selection_audit.csv", encoding="utf-8-sig")
    monthly = pd.read_csv(RUN / "inputs" / "source_metadata" / "selected_station_monthly_observations.csv", encoding="utf-8-sig")
    policy = pd.read_csv(RUN / "inputs" / "source_snapshot" / "registry" / "model_exclusion_policy.csv", encoding="utf-8-sig")
    gate = json.loads((RUN / "reports" / "model_audit" / "gate.json").read_text(encoding="utf-8"))
    build = json.loads((RUN / "inputs" / "source_metadata" / "input_build_summary.json").read_text(encoding="utf-8"))
    oof = pd.read_parquet(RUN / "outputs" / "q72_three_fold_oof_predictions.parquet")

    excluded = set()
    for row in policy.itertuples(index=False):
        excluded.add(relaxed(row.canonical_station))
        excluded.update(relaxed(name) for name in str(row.aliases).split("|") if str(name).strip())
    active_names = set(panel["q_site"].dropna().astype(str))
    forbidden_active = sorted(name for name in active_names if relaxed(name) in excluded)
    active_obs = panel[panel["Q_obsv_cfs"].notna()].copy()
    active_counts = active_obs.groupby("q_site").size()

    if CONFIG["cohort"] == "full_2006_2022":
        ordinary = active_counts[active_counts.index != "石角站"]
        cohort_rule = bool(ordinary.eq(204).all() and int(active_counts.get("石角站", 0)) == 192)
    else:
        grouped = monthly.groupby("station_name").agg(months=("Q_obsv_cfs", "size"), training=("year", lambda s: int((s <= 2015).sum())))
        cohort_rule = bool(grouped["months"].ge(12).all() and grouped["training"].ge(1).all())

    selected_only = selected[selected["selected_for_reach"].astype(str).str.casefold().isin({"true", "1"})]
    required = [
        RUN / "config.json",
        RUN / "inputs" / "covariate_backbone.parquet",
        RUN / "inputs" / "indata.parquet",
        RUN / "inputs" / "topology" / "topology_edges.csv",
        RUN / "inputs" / "source_snapshot" / "discharge" / "complete_2010_2022",
        RUN / "inputs" / "source_snapshot" / "discharge" / "DischargeData_2006_2009.xlsx",
        RUN / "inputs" / "source_snapshot" / "registry" / "model_exclusion_policy.csv",
        RUN / "inputs" / "source_snapshot" / "spatial" / "PRB水文站_全部.shp",
        RUN / "inputs" / "source_snapshot" / "spatial" / "reaches_topology.shp",
        RUN / "inputs" / "source_snapshot" / "spatial" / "reach_catchments.shp",
        RUN / "scripts" / "build_updated_input.py",
        RUN / "scripts" / "run_q72_baseline.py",
        RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py",
        RUN / "outputs" / "q72_three_fold_oof_predictions.parquet",
        RUN / "reports" / "model_audit" / "model_report.md",
        RUN / "inputs_manifest" / "provenance_manifest.csv",
        RUN / "README.md",
    ]
    if CONFIG["include_noncomplete_source"]:
        required.append(RUN / "inputs" / "source_snapshot" / "discharge" / "noncomplete_2010_2022")

    checks = {
        "runtime_exact_conda_sparrow": RUNTIME["environment_name"] == "sparrow",
        "all_required_local_artifacts_exist": all(path.exists() for path in required),
        "input_panel_key_unique": not bool(panel.duplicated(["comid", "year", "month"]).any()),
        "input_panel_shape_230_reaches_x_204_months": len(panel) == 46920,
        "active_observations_positive_finite": bool(np.isfinite(active_obs["Q_obsv_cfs"]).all() and active_obs["Q_obsv_cfs"].gt(0).all()),
        "all_policy_exclusions_absent": not forbidden_active,
        "protected_shijiao_present": "石角站" in active_names,
        "cohort_coverage_rule_passed": cohort_rule,
        "one_selected_station_per_reach": not bool(selected_only.duplicated("reach_id").any()),
        "same_reach_maximum_flow_rank_selected": bool(selected_only["selection_rank"].eq(1).all()),
        "input_build_fixed_exclusions_absent": bool(build["fixed_exclusions_absent"]),
        "three_blocked_folds_complete": oof["fold_id"].nunique() == 3,
        "oof_key_unique": not bool(oof.duplicated(["fold_id", "station_name", "reach_id", "year", "month"]).any()),
        "oof_year_boundary_2012_2018": bool(oof["year"].between(2012, 2018).all()),
        "oof_predictions_positive_finite": bool(np.isfinite(oof["predicted_cfs"]).all() and oof["predicted_cfs"].gt(0).all()),
        "q72_parent_code_hashes_exact": bool(gate["parent_q72_code_hashes_exact"]),
        "q72_model_gate_passed": bool(gate["passed"]),
        "source_inventory_paths_are_local": bool(pd.read_csv(RUN / "reports" / "input_audit" / "discharge_file_inventory.csv", encoding="utf-8-sig")["relative_path"].str.startswith("inputs\\source_snapshot\\").all()),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "run_id": RUN.name,
        "cohort": CONFIG["cohort"],
        "checks": checks,
        "passed_count": int(sum(checks.values())),
        "check_count": int(len(checks)),
        "passed": all(checks.values()),
        "decision": "FREEZE_AS_NEW_Q72_BASELINE" if all(checks.values()) else "DO_NOT_FREEZE",
        "forbidden_active_stations": forbidden_active,
        "active_station_count": int(len(active_names)),
        "active_observation_months": int(len(active_obs)),
        "oof_station_count": int(oof["station_name"].nunique()),
    }
    (RUN / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_rows = []
    for root_name in ["outputs", "reports", "logs"]:
        for path in sorted((RUN / root_name).rglob("*")):
            if path.is_file():
                artifact_rows.append({"role": root_name, "path": str(path.relative_to(RUN)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    for path in [RUN / "README.md", RUN / "validation.json"]:
        artifact_rows.append({"role": "root", "path": str(path.relative_to(RUN)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(artifact_rows).to_csv(RUN / "inputs_manifest" / "final_artifact_manifest.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
