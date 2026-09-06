from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_32"
REPORT = RUN / "reports"
DECISION31 = TEST / "20260823_31" / "reports" / "stage31_decision.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    parent = json.loads(DECISION31.read_text(encoding="utf-8"))
    prerequisite = {
        "stage31_numerical_pass": bool(parent["numerical_pass"]),
        "stage31_total_flow_gates_pass": bool(all(parent["total_flow_gates"].values())),
        "stage31_path_gates_pass": bool(all(parent["path_gates"].values())),
        "stage31_spatial_pass_false": bool(not parent["spatial_pass"]),
    }
    triggered_before_moran = bool(all(prerequisite.values()))
    if triggered_before_moran:
        raise RuntimeError("Stage32 prerequisites unexpectedly passed; a preregistered Moran audit/model implementation is required before fitting")
    decision = {
        "stage": "20260823_32",
        "status": "NOT_TRIGGERED",
        "prerequisites": prerequisite,
        "moran_test_run": False,
        "river_network_residual_fit": False,
        "reason": "20260823_31 failed registered path-transfer gates (station delayed-fraction ranking and peak-month distance); a network residual may not be used to rescue a failed process/path hypothesis.",
        "successor": "20260823_33 diagnostic-only state analysis trigger audit",
    }
    (REPORT / "stage32_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_32 河网残差触发审计\n\n"
        "状态：`NOT_TRIGGERED`。`_31` 的延迟比例站间排序与峰值月份门未通过，因此不允许用河网高斯残差挽救该过程假设，也不执行 Moran 筛选。\n",
        encoding="utf-8",
    )
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "stage31_decision_sha256": sha256(DECISION31),
    }, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
