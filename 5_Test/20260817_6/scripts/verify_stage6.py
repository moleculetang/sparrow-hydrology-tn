from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"E:\SPARROW\5_Test\20260817_6")
completion = json.loads((ROOT / "reports" / "completion_audit.json").read_text(encoding="utf-8"))
manifest = json.loads((ROOT / "reports" / "final_completion_manifest.json").read_text(encoding="utf-8"))
lock = json.loads((ROOT / "final_lock.json").read_text(encoding="utf-8"))
audit = pd.read_parquet(ROOT / "outputs" / "requirement_by_requirement_audit.parquet")
checks = {
    "completion": completion["pass"],
    "all_hard_checks": audit["pass"].all() and len(audit) == completion["hard_checks"],
    "no_core_import": not completion["core_imported"] and not manifest["audit_core_imported"],
    "final_manifest": manifest["pass"] and manifest["status"] == "complete",
    "final_lock": lock["status"] == "frozen_complete" and len(lock["canonical_output_sha256"]) == 6,
    "no_point_identification": not manifest["scientific_decision"]["effective_delivery_time_point_identified"],
    "eta_diagnostic_only": manifest["eta_boundary_confounding_role"] == "diagnostic_only_not_hard_gate",
    "time_boundaries": manifest["OOF_evaluation_years"] == [2018, 2019, 2020, 2021] and manifest["retrospective_locked_year"] == 2022,
}
checks = {key: bool(value) for key, value in checks.items()}
result = {"scenario_id": "20260817_6", "pass": all(checks.values()), "checks": checks}
(ROOT / "reports" / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result["pass"]:
    raise SystemExit(1)
