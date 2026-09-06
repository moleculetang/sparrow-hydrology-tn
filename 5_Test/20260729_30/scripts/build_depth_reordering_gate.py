from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
PARENT = ROOT / "5_Test" / "20260729_29"
PARENT_GATE = (
    PARENT / "reports" / "spatial_recharge_control_audit" / "gate.json"
)
CORRELATIONS = (
    PARENT / "reports" / "spatial_recharge_control_audit"
    / "attribute_bfi_spearman.csv"
)
REACH_AUDIT = (
    PARENT / "outputs" / "reach_recharge_spatial_control_candidate.parquet"
)
REPORT = RUN / "reports" / "depth_reordering_gate"
OUTPUTS = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
LOGS = RUN / "logs"
CONTRACT = RUN / "experiment_contract.md"
LITERATURE = RUN / "literature_basis.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, role: str, state: str) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "role": role,
        "semantic_state": state,
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256(path),
    }


def main() -> None:
    for directory in (REPORT, OUTPUTS, MANIFEST, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent["authorized_next_action"] != (
        "RETURN_TO_LITERATURE_AND_ATTRIBUTE_GAP_REVIEW"
    ):
        raise RuntimeError("Parent gate does not authorize this review")
    correlations = pd.read_csv(CORRELATIONS)
    reaches = pd.read_parquet(REACH_AUDIT).copy()
    indexed = correlations.set_index(["subset", "scale", "attribute"])
    depth = indexed.loc[
        ("all_97", "upstream_aw", "depth_to_bedrock_m")
    ]
    current = indexed.loc[
        ("all_97", "upstream_aw", "vertical_share_current")
    ]
    stable_depth = indexed.loc[
        (
            "lower_bfi_filter_sensitivity",
            "upstream_aw",
            "depth_to_bedrock_m",
        )
    ]
    gain = float(depth["spearman"] - current["spearman"])

    sorted_share = np.sort(
        reaches["vertical_share_current"].to_numpy(float)
    )
    order = np.argsort(
        reaches["depth_to_bedrock_m"].to_numpy(float),
        kind="mergesort",
    )
    candidate = np.empty(len(reaches), dtype=float)
    candidate[order] = sorted_share
    reaches["vertical_share_depth_reordered"] = candidate
    distribution_preserved = bool(np.array_equal(
        sorted_share, np.sort(candidate)
    ))
    authorized = bool(
        float(depth["spearman"]) > 0
        and gain >= 0.05
        and float(depth["bootstrap_95_lower"]) > 0
        and float(stable_depth["spearman"]) > 0
        and float(depth["leave_one_out_positive_fraction"]) >= 0.90
        and distribution_preserved
    )
    output = OUTPUTS / "reach_depth_reordered_vertical_share.parquet"
    reaches[[
        "reach_id", "depth_to_bedrock_m", "vertical_share_current",
        "vertical_share_depth_reordered",
    ]].to_parquet(output, index=False)

    checks = {
        "parent_authorization": True,
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "reach_count_230": (
            len(reaches) == 230 and reaches["reach_id"].nunique() == 230
        ),
        "bedrock_complete": bool(
            reaches["depth_to_bedrock_m"].notna().all()
        ),
        "depth_spearman_positive": float(depth["spearman"]) > 0,
        "depth_gain_at_least_0_05": gain >= 0.05,
        "depth_bootstrap_lower_positive": (
            float(depth["bootstrap_95_lower"]) > 0
        ),
        "stable_subset_positive": float(stable_depth["spearman"]) > 0,
        "leave_one_out_positive_at_least_90pct": (
            float(depth["leave_one_out_positive_fraction"]) >= 0.90
        ),
        "candidate_distribution_exactly_preserved": (
            distribution_preserved
        ),
        "zero_fitted_parameters": True,
        "no_forbidden_period_or_management": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    decision = (
        "AUTHORIZE_DEPTH_RANK_REORDERED_RECHARGE_TEST"
        if authorized
        else "DO_NOT_AUTHORIZE_DEPTH_RANK_REORDERED_RECHARGE_TEST"
    )
    next_action = (
        "RUN_ONE_DEPTH_RANK_REORDERED_RECHARGE_MODEL"
        if authorized
        else "WAIT_FOR_NEW_STATIC_HYDRAULIC_ATTRIBUTES"
    )
    gate = {
        "run_id": "20260729_30",
        "phase": "depth_rank_reordering_evidence_gate",
        "created_utc": utc_now(),
        "checks": checks,
        "evidence_metrics": {
            "current_upstream_vertical_share_spearman": float(
                current["spearman"]
            ),
            "depth_upstream_spearman": float(depth["spearman"]),
            "depth_gain_over_current": gain,
            "depth_bootstrap_95_lower": float(
                depth["bootstrap_95_lower"]
            ),
            "depth_bootstrap_95_upper": float(
                depth["bootstrap_95_upper"]
            ),
            "depth_lower_filter_sensitivity_spearman": float(
                stable_depth["spearman"]
            ),
            "depth_leave_one_out_positive_fraction": float(
                depth["leave_one_out_positive_fraction"]
            ),
        },
        "candidate_authorized": authorized,
        "decision": decision,
        "authorized_next_action": next_action,
        "parameters_calibrated": False,
        "candidate_distribution_preserved": distribution_preserved,
        "period_2019_2022_read": False,
        "management_fluxes_read": False,
        "runtime_identity": RUNTIME_IDENTITY,
        "passed": all(checks.values()),
        "series_terminal": False,
    }
    gate_path = REPORT / "gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = f"""# 基岩深度单项补给排序门禁

## 结论

`{decision}`

- 当前上游垂向份额Spearman：{float(current['spearman']):.3f}；
- 上游基岩深度Spearman：{float(depth['spearman']):.3f}；
- 增益：{gain:+.3f}；
- bootstrap 95%区间：[{float(depth['bootstrap_95_lower']):.3f},
  {float(depth['bootstrap_95_upper']):.3f}]；
- 低BFI滤波敏感性子集：{float(stable_depth['spearman']):.3f}；
- 逐站删除正号比例：
  {float(depth['leave_one_out_positive_fraction']):.1%}；
- 垂向份额边际分布精确保留：{distribution_preserved}。

该证据只授权一次可证伪的模型测试，不把基岩深度等同于渗透率。

下一步：`{next_action}`
"""
    report_path = REPORT / "technical_report.md"
    report_path.write_text(report, encoding="utf-8")
    sources = [
        PARENT_GATE, CORRELATIONS, REACH_AUDIT, CONTRACT, LITERATURE,
        RUN / "scripts" / "runtime_guard.py",
        RUN / "scripts" / "build_depth_reordering_gate.py",
        RUN / "scripts" / "validate_depth_reordering_gate.py",
    ]
    products = [output, gate_path, report_path]
    provenance = {
        "run_id": "20260729_30",
        "created_utc": utc_now(),
        "sources": [
            record(path, "depth_gate_source", "reported_or_derived")
            for path in sources
        ],
        "products": [
            record(path, "depth_gate_product", "derived")
            for path in products
        ],
        "period_2019_2022_read": False,
        "management_fluxes_read": False,
    }
    (MANIFEST / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        provenance["sources"] + provenance["products"]
    ).to_csv(
        MANIFEST / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (LOGS / "runtime_log.md").write_text(
        "# Runtime log\n\n"
        "`conda --no-plugins run -n sparrow python "
        "E:\\SPARROW\\5_Test\\20260729_30\\scripts\\"
        "build_depth_reordering_gate.py`\n\n"
        f"Decision: `{decision}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "decision": decision,
        "authorized_next_action": next_action,
        "current_spearman": float(current["spearman"]),
        "depth_spearman": float(depth["spearman"]),
        "gain": gain,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
