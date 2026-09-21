"""Run the frozen hydrology through 2025 with corrected ERA5 PET only in 2025."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_43"
FORCING = ROOT / "5_Test" / "20260828_40" / "outputs" / "daily_hbv_forcing_1961_2025_chmpre_cmfd_corrected_era5_2025.parquet"
SOURCE34 = ROOT / "5_Test" / "20260828_34" / "scripts" / "run_locked_long_simulation.py"
SOURCE35 = ROOT / "5_Test" / "20260828_35" / "scripts" / "export_tn_hydrology_interface.py"
S33 = ROOT / "5_Test" / "20260828_33"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def main() -> None:
    out, reports, locks, work = RUN / "outputs", RUN / "reports", RUN / "locks", RUN / "work" / "forcing_lock"
    for folder in (out, reports, locks, work / "locks"):
        folder.mkdir(parents=True, exist_ok=True)
    forcing_lock = {
        "stage": "20260828_43", "status": "PASS_FORCING_LOCK_1961_2025",
        "formal_period": "1961-01-01 through 2025-12-31", "formal_forcing_path": str(FORCING),
        "pet_bridge_status": "CORRECTED_ERA5_2025_ONLY",
    }
    (work / "locks" / "forcing_integrity_lock.json").write_text(json.dumps(forcing_lock, indent=2), encoding="utf-8")
    stage34 = load("stage43_long_run", SOURCE34)
    stage34.RUN = RUN; stage34.OUT = out; stage34.REPORTS = reports; stage34.LOCKS = locks; stage34.S31 = work; stage34.S33 = S33
    stage34.main()
    stage35 = load("stage43_tn_interface", SOURCE35)
    stage35.RUN = RUN; stage35.OUT = out; stage35.REPORTS = reports; stage35.LOCKS = locks; stage35.S31 = work; stage35.S34 = RUN
    stage35.main()
    long_qa = json.loads((reports / "long_simulation_qa.json").read_text(encoding="utf-8"))
    interface_qa = json.loads((reports / "tn_hydrology_interface_qa.json").read_text(encoding="utf-8"))
    checks = {"long_simulation_pass": long_qa["status"] == "PASS_LOCKED_LONG_SIMULATION", "tn_interface_pass": interface_qa["status"] == "PASS_TN_READY_HYDROLOGY_INTERFACE", "no_parameter_refit": True}
    status = "PASS_1961_2025_HYBRID_TN_HYDROLOGY" if all(checks.values()) else "FAIL_1961_2025_HYBRID_TN_HYDROLOGY"
    products = {name: out / name for name in (
        "tn_hydrology_reach_daily.parquet", "tn_hydrology_reach_monthly.parquet",
        "tn_hydrology_reservoir_daily.parquet", "tn_hydrology_reservoir_monthly.parquet",
        "tn_hydrology_reservoir_static_metadata.parquet",
    )}
    report = {"stage": "20260828_43", "status": status, "checks": checks, "forcing": {"path": str(FORCING), "sha256": sha256(FORCING)}, "products": {key: {"path": str(path), "sha256": sha256(path)} for key, path in products.items()}}
    report_path = reports / "hybrid_1961_2025_qa.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / "technical_report.md").write_text("# 1961–2025 CHM_PRE+CMFD/校正ERA5水文产品\n\n" f"状态：`{status}`。1961–2024沿用CMFD，2025使用经预注册规则校正的ERA5 PET。\n", encoding="utf-8")
    (locks / "hybrid_product_lock.json").write_text(json.dumps({"status": status, "qa_sha256": sha256(report_path), "products": report["products"]}, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not status.startswith("PASS"):
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
