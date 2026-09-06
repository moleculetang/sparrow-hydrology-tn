"""Create the final delivery lock for the formal and sensitivity products."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_38"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest_path = ROOT / "5_Test" / "20260828_31" / "program_manifest.json"
    formal_lock_path = ROOT / "5_Test" / "20260828_37" / "locks" / "program_completion_lock.json"
    sensitivity_lock_path = RUN / "locks" / "forced_2025_sensitivity_product_lock.json"
    crosscheck_path = RUN / "reports" / "dual_product_crosscheck.json"
    catalog_path = RUN / "reports" / "dual_product_catalog.md"
    manifest = read(manifest_path)
    formal = read(formal_lock_path)
    sensitivity = read(sensitivity_lock_path)
    crosscheck = read(crosscheck_path)
    checks = {
        "manifest_dual_product_status": manifest.get("status") == "COMPLETE_DUAL_PRODUCT_1961_2024_FORMAL_AND_1961_2025_SENSITIVITY",
        "formal_status": formal.get("status") == "COMPLETE_VERIFIED_READY_FOR_TN_1961_2024_WITH_2025_SENSITIVITY",
        "sensitivity_status": sensitivity.get("status") == "PASS_NUMERICALLY_1961_2025_SENSITIVITY_ONLY",
        "common_period_identity": crosscheck.get("status") == "PASS_DUAL_PRODUCT_ISOLATION_AND_COMMON_PERIOD_IDENTITY",
        "catalog_exists": catalog_path.is_file(),
    }
    formal_paths = {
        "reach_daily": ROOT / "5_Test" / "20260828_35" / "outputs" / "tn_hydrology_reach_daily.parquet",
        "reach_monthly": ROOT / "5_Test" / "20260828_35" / "outputs" / "tn_hydrology_reach_monthly.parquet",
        "reservoir_daily": ROOT / "5_Test" / "20260828_35" / "outputs" / "tn_hydrology_reservoir_daily.parquet",
        "reservoir_monthly": ROOT / "5_Test" / "20260828_35" / "outputs" / "tn_hydrology_reservoir_monthly.parquet",
        "reservoir_static": ROOT / "5_Test" / "20260828_35" / "outputs" / "tn_hydrology_reservoir_static_metadata.parquet",
    }
    sensitivity_paths = {
        "forcing": RUN / "outputs" / "daily_hbv_forcing_1961_2025_sensitivity.parquet",
        "reach_daily": RUN / "outputs" / "tn_hydrology_reach_daily.parquet",
        "reach_monthly": RUN / "outputs" / "tn_hydrology_reach_monthly.parquet",
        "reservoir_daily": RUN / "outputs" / "tn_hydrology_reservoir_daily.parquet",
        "reservoir_monthly": RUN / "outputs" / "tn_hydrology_reservoir_monthly.parquet",
        "reservoir_static": RUN / "outputs" / "tn_hydrology_reservoir_static_metadata.parquet",
    }
    checks["formal_hashes_current"] = all(sha256(path) == formal["files"][key] for key, path in formal_paths.items())
    checks["sensitivity_hashes_current"] = all(sha256(path) == sensitivity["files"][key] for key, path in sensitivity_paths.items())
    status = "COMPLETE_DUAL_PRODUCT_DELIVERY" if all(checks.values()) else "FAIL_DUAL_PRODUCT_DELIVERY"
    report = {
        "stage": "20260828_38",
        "status": status,
        "checks": checks,
        "formal_product": "1961-2024 TN-ready",
        "sensitivity_product": "1961-2025 PET_EXTENSION_CONFOUNDED sensitivity only",
        "files": {
            "program_manifest": sha256(manifest_path),
            "formal_lock": sha256(formal_lock_path),
            "sensitivity_lock": sha256(sensitivity_lock_path),
            "crosscheck_report": sha256(crosscheck_path),
            "catalog": sha256(catalog_path),
            "finalizer_code": sha256(Path(__file__)),
        },
    }
    report_path = RUN / "reports" / "dual_product_delivery.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["files"]["delivery_report"] = sha256(report_path)
    lock_path = RUN / "locks" / "dual_product_delivery_lock.json"
    lock_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "COMPLETE_DUAL_PRODUCT_DELIVERY":
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
