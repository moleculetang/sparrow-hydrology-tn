from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260815_6")
P5 = Path(r"E:\SPARROW\5_Test\20260815_5")
P4 = Path(r"E:\SPARROW\5_Test\20260815_4")
LOCAL_PATH = P5 / "outputs" / "selected_delivery_model_reach_month_1961_2022.parquet"
DELIVERY_DECISION = P5 / "reports" / "delivery_structure_decision.json"
SOURCE_DECISION = P4 / "reports" / "source_structure_decision.json"
PARENT_AUDIT = P5 / "reports" / "completion_audit.json"
TOPOLOGY = Path(r"E:\SPARROW\5_Test\20260814_1\inputs\topology\topology_edges.csv")
S4_SCRIPT = P4 / "scripts" / "run_stage4.py"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def load_stage4():
    spec = importlib.util.spec_from_file_location("stage4_routing_core", S4_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load routing core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


s4 = load_stage4()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow" or any(os.environ.get(k) != "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]):
        raise RuntimeError("sparrow runtime and one-thread limits required")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not json.loads(PARENT_AUDIT.read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("20260815_5 did not pass")
    parents = [LOCAL_PATH, DELIVERY_DECISION, SOURCE_DECISION, PARENT_AUDIT, TOPOLOGY, S4_SCRIPT]
    start = {str(p): sha256(p) for p in parents}
    dump(REPORTS / "parent_hashes_start.json", start)
    local = pd.read_parquet(LOCAL_PATH)
    source_decision = json.loads(SOURCE_DECISION.read_text(encoding="utf-8"))
    delivery_decision = json.loads(DELIVERY_DECISION.read_text(encoding="utf-8"))
    reach_ids = np.array(sorted(local.reach_id.unique()), dtype=int)
    order, downstream, terminal = s4.topology_operators(reach_ids)
    ridx = {rid: i for i, rid in enumerate(reach_ids)}
    routed_columns = [
        ("local_tn_release_kg_n", "routed_tn_kg_n"),
        ("local_tn_gt1y_kg_n", "routed_tn_gt1y_kg_n"),
        ("local_tn_gt5y_kg_n", "routed_tn_gt5y_kg_n"),
        ("local_tn_gt10y_kg_n", "routed_tn_gt10y_kg_n"),
        ("local_tn_pre1961_kg_n", "routed_tn_pre1961_kg_n"),
        ("local_tn_post1961_kg_n", "routed_tn_post1961_kg_n"),
        ("local_tn_post1961_age_moment_month_kg_n", "routed_tn_post1961_age_moment_month_kg_n"),
        ("local_tn_age_lower_bound_moment_month_kg_n", "routed_tn_age_lower_bound_moment_month_kg_n"),
        ("quick_tn_release_kg_n", "routed_quick_tn_kg_n"),
        ("base_tn_release_kg_n", "routed_base_tn_kg_n"),
    ]
    rows = []
    month_audits = []
    max_recursion_abs = 0.0
    max_terminal_abs = 0.0
    max_terminal_rel = 0.0
    for (year, month), block in local.groupby(["year", "month"], sort=True):
        b = block.set_index("reach_id").loc[reach_ids]
        values = np.column_stack([b[src].to_numpy(float) for src, _ in routed_columns] + [(b.q_local_total_mm * b.catchment_area_km2 * 1000.0).to_numpy(float)])
        local_values = values.copy()
        for rid in order:
            if rid in downstream:
                down, frac = downstream[rid]
                values[ridx[down]] += values[ridx[rid]] * frac
        out = pd.DataFrame({"reach_id": reach_ids, "year": int(year), "month": int(month)})
        for i, (_, target) in enumerate(routed_columns):
            out[target] = values[:, i]
        out["routed_water_volume_m3"] = values[:, -1]
        out["terminal_tree_id"] = out.reach_id.map(terminal).astype(int)
        out["tn_concentration_proxy_mg_l"] = np.divide(out.routed_tn_kg_n * 1000.0, out.routed_water_volume_m3, out=np.full(len(out), np.nan), where=out.routed_water_volume_m3.to_numpy() > 0)
        for label, mass in [("gt1y", "routed_tn_gt1y_kg_n"), ("gt5y", "routed_tn_gt5y_kg_n"), ("gt10y", "routed_tn_gt10y_kg_n")]:
            out[f"fraction_memory_{label}"] = np.divide(out[mass], out.routed_tn_kg_n, out=np.zeros(len(out)), where=out.routed_tn_kg_n.to_numpy() > 0)
        out["fraction_pre1961_equilibrium"] = np.divide(out.routed_tn_pre1961_kg_n, out.routed_tn_kg_n, out=np.zeros(len(out)), where=out.routed_tn_kg_n.to_numpy() > 0)
        out["post1961_mean_cohort_age_month"] = np.divide(out.routed_tn_post1961_age_moment_month_kg_n, out.routed_tn_post1961_kg_n, out=np.full(len(out), np.nan), where=out.routed_tn_post1961_kg_n.to_numpy() > 0)
        out["all_history_mean_cohort_age_lower_bound_month"] = np.divide(out.routed_tn_age_lower_bound_moment_month_kg_n, out.routed_tn_kg_n, out=np.full(len(out), np.nan), where=out.routed_tn_kg_n.to_numpy() > 0)
        out["source_model_id"] = source_decision["selected_source_model_id"]
        out["delivery_model_id"] = delivery_decision["selected_delivery_model_id"]
        out["effective_tn_delivery_mu_month"] = delivery_decision["selected_effective_tn_delivery_mu_month"]
        out["routing_model_id"] = "R0_same_month_mass_conserving"
        out["point_source_tn_status"] = "not_available_not_fabricated"
        rows.append(out)
        terminal_idx = np.array([ridx[r] for r in sorted(set(terminal.values()))], dtype=int)
        local_total = local_values[:, 0].sum(dtype=np.longdouble)
        terminal_total = values[terminal_idx, 0].sum(dtype=np.longdouble)
        err = float(terminal_total - local_total)
        rel = float(abs(err) / max(abs(float(local_total)), 1.0))
        max_terminal_abs = max(max_terminal_abs, abs(err))
        max_terminal_rel = max(max_terminal_rel, rel)
        month_audits.append({"year": int(year), "month": int(month), "local_total_kg_n": float(local_total), "terminal_total_kg_n": float(terminal_total), "error_kg_n": err, "relative_error": rel})
    routed = pd.concat(rows, ignore_index=True).sort_values(["reach_id", "year", "month"])
    routed.to_parquet(OUT / "r0_routed_tn_1961_2022.parquet", index=False)
    pd.DataFrame({"reach_id": reach_ids, "terminal_tree_id": [terminal[int(r)] for r in reach_ids]}).to_csv(OUT / "reach_terminal_tree_mapping.csv", index=False)
    audit = {"scenario_id": "20260815_6", "rows": len(routed), "reaches": int(routed.reach_id.nunique()), "months": int(len(routed[["year", "month"]].drop_duplicates())), "full_terminal_tree_count": len(set(terminal.values())), "max_terminal_mass_abs_error_kg_n": max_terminal_abs, "max_terminal_mass_relative_error": max_terminal_rel, "R1_status": "not_run_no_reliable_tau_r", "point_source_tn_status": "not_available_not_fabricated"}
    dump(REPORTS / "network_mass_balance_audit.json", {"summary": audit, "monthly": month_audits})
    dump(REPORTS / "routing_decision.json", {"routing_model_id": "R0_same_month_mass_conserving", "R1_status": "not_run_no_reliable_tau_r", "new_fitted_parameter_count": 0, "locked_2022_used_for_selection": False})
    end = {str(p): sha256(p) for p in parents}
    dump(REPORTS / "parent_hashes_end.json", end)
    if start != end:
        raise RuntimeError("parent changed during stage 6")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
