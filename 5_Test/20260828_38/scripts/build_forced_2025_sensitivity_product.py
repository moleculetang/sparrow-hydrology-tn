"""Build a separately labelled 1961-2025 hydrology sensitivity product.

This stage deliberately does not promote the failed 2025 PET bridge.  It
reuses the frozen Stage 34/35 implementations in an isolated output directory
and records the known bridge failure in the authoritative Stage 38 report.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_38"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
WORK = RUN / "work"
HISTORICAL = ROOT / "5_Test" / "20260828_29" / "outputs" / "daily_hbv_forcing_1961_2024.parquet"
EXTENSION = ROOT / "5_Test" / "20260828_30" / "outputs" / "daily_hbv_forcing_2025_era5_sensitivity.parquet"
BRIDGE_QA = ROOT / "5_Test" / "20260828_30" / "reports" / "era5_pet_harmonization_qa.json"
STAGE34_SCRIPT = ROOT / "5_Test" / "20260828_34" / "scripts" / "run_locked_long_simulation.py"
STAGE35_SCRIPT = ROOT / "5_Test" / "20260828_35" / "scripts" / "export_tn_hydrology_interface.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_combined_forcing() -> tuple[Path, dict[str, object]]:
    bridge = json.loads(BRIDGE_QA.read_text(encoding="utf-8"))
    if bridge.get("status") != "PET_EXTENSION_CONFOUNDED":
        raise RuntimeError("Stage 38 is authorized only for the registered confounded sensitivity branch")
    historical = pd.read_parquet(HISTORICAL)
    extension = pd.read_parquet(EXTENSION)
    historical["date"] = pd.to_datetime(historical["date"])
    extension["date"] = pd.to_datetime(extension["date"])
    historical["pet_fao56_raw_mm_day"] = historical["pet_fao56_mm_day"]
    historical["pet_factor"] = 1.0
    historical["precip_source"] = "CHM_PRE_V2_daily"
    historical["pet_source"] = "CMFD_V2_0_03HR_FAO56"
    historical["pet_bias_corrected"] = False
    historical["forcing_extension_flag"] = False
    if list(historical.columns) != list(extension.columns):
        missing = sorted(set(historical.columns) - set(extension.columns))
        extra = sorted(set(extension.columns) - set(historical.columns))
        raise RuntimeError(f"Historical and 2025 forcing schemas differ: missing={missing}, extra={extra}")
    extension = extension[historical.columns]
    combined = pd.concat([historical, extension], ignore_index=True)
    combined = combined.sort_values(["reach_id", "date"], kind="mergesort").reset_index(drop=True)
    dates = pd.date_range("1961-01-01", "2025-12-31", freq="D")
    checks = {
        "rows_exact": len(combined) == len(dates) * 230 == 5_460_430,
        "keys_unique": not combined.duplicated(["date", "reach_id"]).any(),
        "date_range_exact": combined.date.min() == dates.min() and combined.date.max() == dates.max(),
        "reach_ids_exact": np.array_equal(np.sort(combined.reach_id.unique()), np.arange(1, 231)),
        "historical_prefix_source_hash_preserved": sha256(HISTORICAL) == bridge["hashes"]["historical_1961_2024"],
        "extension_source_hash_preserved": sha256(EXTENSION) == bridge["hashes"]["extension_2025"],
        "forcing_finite": bool(np.isfinite(combined.select_dtypes(include=[np.number]).to_numpy(float)).all()),
        "precipitation_pet_nonnegative": bool(combined[["precipitation_daily_mm", "pet_fao56_mm_day"]].ge(0).all().all()),
        "only_2025_flagged_extension": bool(
            combined.loc[combined.date.dt.year.le(2024), "forcing_extension_flag"].eq(False).all()
            and combined.loc[combined.date.dt.year.eq(2025), "forcing_extension_flag"].eq(True).all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    path = OUT / "daily_hbv_forcing_1961_2025_sensitivity.parquet"
    combined.to_parquet(path, index=False, compression="zstd")
    return path, {"checks": checks, "rows": len(combined), "sha256": sha256(path)}


def run_reused_stages(forcing_path: Path) -> None:
    adapter = WORK / "sensitivity_forcing_lock"
    (adapter / "locks").mkdir(parents=True, exist_ok=True)
    # Stage 34's frozen implementation uses the exact status token solely to
    # select the end date.  This is an internal execution adapter, not the
    # authoritative product decision; Stage 38 remains sensitivity-only.
    adapter_lock = {
        "stage": "20260828_38_adapter",
        "status": "PASS_FORCING_LOCK_1961_2025",
        "formal_period": "1961-01-01 through 2025-12-31",
        "formal_forcing_path": str(forcing_path),
        "pet_bridge_status": "PET_EXTENSION_CONFOUNDED",
        "product_class": "sensitivity_only",
    }
    (adapter / "locks" / "forcing_integrity_lock.json").write_text(
        json.dumps(adapter_lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stage34 = load_module("stage34_sensitivity", STAGE34_SCRIPT)
    stage34.RUN = RUN
    stage34.OUT = OUT
    stage34.REPORTS = REPORTS
    stage34.LOCKS = LOCKS
    stage34.S31 = adapter
    stage34.main()

    stage35 = load_module("stage35_sensitivity", STAGE35_SCRIPT)
    stage35.RUN = RUN
    stage35.OUT = OUT
    stage35.REPORTS = REPORTS
    stage35.LOCKS = LOCKS
    stage35.S31 = adapter
    stage35.S34 = RUN
    stage35.main()


def finalize(forcing_path: Path, forcing_audit: dict[str, object]) -> None:
    interface_qa = json.loads((REPORTS / "tn_hydrology_interface_qa.json").read_text(encoding="utf-8"))
    long_qa = json.loads((REPORTS / "long_simulation_qa.json").read_text(encoding="utf-8"))
    product_paths = {
        "forcing": forcing_path,
        "reach_daily": OUT / "tn_hydrology_reach_daily.parquet",
        "reach_monthly": OUT / "tn_hydrology_reach_monthly.parquet",
        "reservoir_daily": OUT / "tn_hydrology_reservoir_daily.parquet",
        "reservoir_monthly": OUT / "tn_hydrology_reservoir_monthly.parquet",
        "reservoir_static": OUT / "tn_hydrology_reservoir_static_metadata.parquet",
    }
    checks = {
        "forcing_grid_pass": all(forcing_audit["checks"].values()),
        "long_simulation_numeric_pass": long_qa.get("status") == "PASS_LOCKED_LONG_SIMULATION",
        "tn_interface_numeric_pass": interface_qa.get("status") == "PASS_TN_READY_HYDROLOGY_INTERFACE",
        "all_products_exist": all(path.is_file() for path in product_paths.values()),
        "formal_product_untouched": (ROOT / "5_Test" / "20260828_35" / "outputs" / "tn_hydrology_reach_daily.parquet").is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "stage": "20260828_38",
        "status": "PASS_NUMERICALLY_1961_2025_SENSITIVITY_ONLY",
        "product_class": "sensitivity_only_not_formal",
        "period": "1961-01-01 through 2025-12-31",
        "known_failed_gate": {
            "metric": "2024 monthly PET correlation",
            "observed": 0.916165152165145,
            "required": 0.98,
            "bridge_status": "PET_EXTENSION_CONFOUNDED",
        },
        "checks": checks,
        "forcing_audit": forcing_audit,
        "rows": interface_qa["rows"],
        "products": {key: {"path": str(path), "sha256": sha256(path)} for key, path in product_paths.items()},
        "interpretation": "The frozen hydrology and reservoir equations remain conservative through 2025, but the 2025 PET source bridge did not pass its preregistered correlation gate. Use this product for sensitivity/exploration only; use Stage 35 for formal TN experiments.",
    }
    report_path = REPORTS / "forced_2025_sensitivity_product_qa.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    technical = (
        "# 1961–2025 强制延伸水文敏感性产品\n\n"
        "状态：`PASS_NUMERICALLY_1961_2025_SENSITIVITY_ONLY`。\n\n"
        "该产品使用与1961–2024正式产品完全相同的冻结DYN2P/Q72快慢流结构、参数、历史初始状态和R2水库算子。"
        "2025仅接入CHM_PRE V2降水与经过月尺度乘法校正的ERA5-Land PET；没有重新拟合任何参数。\n\n"
        "数值上，快流、慢流、direct流、库存、水库来源示踪和水龄矩继续闭合，因此可以用于2025探索性TN敏感性计算。"
        "但2024桥接验证的月PET相关系数为0.9162，低于预注册0.98，因此该产品不能替代1961–2024正式产品，也不能称为通过门禁的2025水文验证。\n"
    )
    (REPORTS / "technical_report.md").write_text(technical, encoding="utf-8")
    lock = {
        "stage": "20260828_38",
        "status": report["status"],
        "product_class": report["product_class"],
        "files": {key: value["sha256"] for key, value in report["products"].items()},
        "qa": sha256(report_path),
        "contract": sha256(RUN / "experiment_contract.json"),
        "runner_code": sha256(Path(__file__)),
        "technical_report": sha256(REPORTS / "technical_report.md"),
    }
    (LOCKS / "forced_2025_sensitivity_product_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    for directory in (OUT, REPORTS, LOCKS, WORK):
        directory.mkdir(parents=True, exist_ok=True)
    forcing_path, forcing_audit = build_combined_forcing()
    run_reused_stages(forcing_path)
    finalize(forcing_path, forcing_audit)


if __name__ == "__main__":
    main()
