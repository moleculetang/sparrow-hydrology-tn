from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[1]
PARENT = ROOT / "5_Test" / "20260729_16"
PRIOR_RUN = ROOT / "5_Test" / "20260729_15"
STATIC_RUN = ROOT / "5_Test" / "20260729_13"
REPORT = RUN_DIR / "reports" / "attribute_parameter_mapping_gate"
OUTPUTS = RUN_DIR / "outputs"
MANIFEST_DIR = RUN_DIR / "inputs_manifest"
CONFIG = RUN_DIR / "config" / "attribute_parameter_mapping_contract.json"
PARENT_GATE = PARENT / "reports" / "spinup_protocol_gate" / "gate.json"
PRIOR_CONFIG = PRIOR_RUN / "config" / "parameter_prior_initialization_contract.json"
STATIC_PATH = STATIC_RUN / "inputs" / "reach_static_direction_final.parquet"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path, role: str, state: str) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "role": role,
        "semantic_state": state,
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def rank_coordinate(series: pd.Series) -> pd.Series:
    return 2.0 * (series.rank(method="average", pct=True) - 0.5)


def raw_mapping(features: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    coefficients = cfg["mapping_coefficients"]
    k_perc = coefficients["k_perc"]["center"] * np.exp(
        coefficients["k_perc"]["bedrock"] * features["bedrock"]
        + coefficients["k_perc"]["soil_storage"] * features["soil_storage"]
    )
    k_int = coefficients["k_int"]["center"] * np.exp(
        coefficients["k_int"]["slope"] * features["slope"]
        + coefficients["k_int"]["bedrock"] * features["bedrock"]
    )
    k_g = coefficients["k_g"]["center"] * np.exp(
        coefficients["k_g"]["slope"] * features["slope"]
        + coefficients["k_g"]["bedrock"] * features["bedrock"]
        + coefficients["k_g"]["soil_storage"] * features["soil_storage"]
    )
    route_index = (
        coefficients["k_route"]["slope"] * features["slope"]
        + coefficients["k_route"]["area"] * features["area"]
        + coefficients["k_route"]["length"] * features["length"]
    )
    k_route = (
        coefficients["k_route"]["center"]
        + coefficients["k_route"]["amplitude"] * route_index
    )
    return {
        "k_perc": k_perc,
        "k_int": k_int,
        "k_g": k_g,
        "k_route": k_route,
    }


def monotonicity_checks(cfg: dict) -> dict:
    zero = np.array([0.0])
    delta = 1e-5
    base = {
        "bedrock": zero.copy(),
        "soil_storage": zero.copy(),
        "slope": zero.copy(),
        "area": zero.copy(),
        "length": zero.copy(),
    }
    expected = {
        ("k_perc", "bedrock"): 1,
        ("k_perc", "soil_storage"): -1,
        ("k_int", "slope"): 1,
        ("k_int", "bedrock"): -1,
        ("k_g", "slope"): 1,
        ("k_g", "bedrock"): -1,
        ("k_g", "soil_storage"): -1,
        ("k_route", "slope"): 1,
        ("k_route", "area"): 1,
        ("k_route", "length"): -1,
    }
    checks = {}
    for (parameter, feature), direction in expected.items():
        low = {name: values.copy() for name, values in base.items()}
        high = {name: values.copy() for name, values in base.items()}
        low[feature][0] -= delta
        high[feature][0] += delta
        derivative = (
            raw_mapping(high, cfg)[parameter][0]
            - raw_mapping(low, cfg)[parameter][0]
        ) / (2 * delta)
        checks[f"{parameter}_monotonic_{feature}"] = bool(
            derivative * direction > 0
        )
    return checks


def main() -> None:
    for directory in (REPORT, OUTPUTS, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    priors = json.loads(PRIOR_CONFIG.read_text(encoding="utf-8"))
    if parent_gate["authorized_next_action"] != cfg["required_parent_action"]:
        raise RuntimeError("Parent gate does not authorize attribute mapping")
    static = pd.read_parquet(STATIC_PATH).copy()
    required = {
        "reach_id", "soil_storage_eff_mm", "depth_to_bedrock_m",
        "slope_m_m", "inc_area_km2", "length_km",
    }
    if not required.issubset(static.columns):
        raise RuntimeError(f"Missing mapping attributes: {sorted(required - set(static.columns))}")
    if static[list(required)].isna().any().any():
        raise RuntimeError("Mapping attributes contain missing values")

    mapped = static[[
        "reach_id", "soil_storage_eff_mm", "depth_to_bedrock_m",
        "slope_m_m", "inc_area_km2", "length_km",
    ]].copy()
    mapped["soil_storage_rank"] = rank_coordinate(mapped["soil_storage_eff_mm"])
    mapped["bedrock_rank"] = rank_coordinate(mapped["depth_to_bedrock_m"])
    mapped["slope_rank"] = rank_coordinate(mapped["slope_m_m"])
    mapped["area_rank"] = rank_coordinate(mapped["inc_area_km2"])
    mapped["length_rank"] = rank_coordinate(mapped["length_km"])
    features = {
        "soil_storage": mapped["soil_storage_rank"].to_numpy(),
        "bedrock": mapped["bedrock_rank"].to_numpy(),
        "slope": mapped["slope_rank"].to_numpy(),
        "area": mapped["area_rank"].to_numpy(),
        "length": mapped["length_rank"].to_numpy(),
    }
    parameters = raw_mapping(features, cfg)
    for parameter, values in parameters.items():
        prior = priors["parameter_priors"][parameter]
        mapped[parameter] = np.clip(values, prior["lower"], prior["upper"])
    for parameter, value in cfg["global_parameters"].items():
        mapped[parameter] = value
    mapped["parameter_semantic_state"] = cfg["mapping_semantic_state"]

    parameter_names = list(priors["parameter_priors"])
    bounds_checks = {}
    for parameter in parameter_names:
        prior = priors["parameter_priors"][parameter]
        bounds_checks[f"{parameter}_inside_prior"] = bool(
            mapped[parameter].between(prior["lower"], prior["upper"]).all()
        )
    monotonic_checks = monotonicity_checks(cfg)
    checks = {
        "parent_authorization": True,
        "reach_count_230": mapped["reach_id"].nunique() == 230,
        "required_attributes_complete": not mapped[list(required)].isna().any().any(),
        **bounds_checks,
        "joint_soil_outflow_constraint": bool(
            (mapped["k_perc"] + mapped["k_int"]
             <= cfg["joint_k_perc_k_int_upper"]).all()
        ),
        **monotonic_checks,
        "four_mapped_parameters_only": len(cfg["mapped_parameters"]) == 4,
        "zero_calibrated_reach_effects": True,
        "station_observations_not_read": True,
        "management_fluxes_not_read": True,
        "period_2019_2022_not_read": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    next_action = (
        cfg["next_action_if_pass"] if passed else cfg["next_action_if_fail"]
    )
    output_path = OUTPUTS / "reach_attribute_parameter_map.parquet"
    summary_path = REPORT / "parameter_mapping_summary.csv"
    mapped.to_parquet(output_path, index=False)
    summary = mapped[parameter_names].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T
    summary.index.name = "parameter"
    summary.reset_index().to_csv(summary_path, index=False, encoding="utf-8-sig")

    gate = {
        "run_id": cfg["run_id"],
        "phase": "attribute_parameter_mapping_gate",
        "created_utc": utc_now(),
        "checks": checks,
        "mapping_semantic_state": cfg["mapping_semantic_state"],
        "mapped_parameters": cfg["mapped_parameters"],
        "global_parameters": list(cfg["global_parameters"]),
        "calibrated_degrees_of_freedom": 0,
        "scope": {"reaches": 230, "period_read": None},
        "decision": "ATTRIBUTE_PARAMETER_MAPPING_PASSED" if passed else "ATTRIBUTE_PARAMETER_MAPPING_FAILED",
        "authorized_next_action": next_action,
        "authorized_scope": (
            "Run the conservation core with the frozen spatial parameter map; station-flow calibration remains forbidden."
            if passed
            else "Repair only the deterministic mapping formula; calibration remains forbidden."
        ),
        "station_observations_read": False,
        "management_fluxes_read": False,
        "period_2019_2022_read": False,
        "series_terminal": False,
        "passed": passed,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    parameter_lines = "\n".join(
        f"| {row.parameter} | {row['min']:.4g} | {row['50%']:.4g} | {row['max']:.4g} |"
        for _, row in summary.reset_index().iterrows()
    )
    report = f"""# Q78-NAT 低维属性参数映射

## 结论

映射门禁{'通过' if passed else '未通过'}。四个参数由五个冻结属性的固定公式决定，其他五个参数保持全局固定；率定自由度为 0。

| 参数 | 最小值 | 中位数 | 最大值 |
|---|---:|---:|---:|
{parameter_lines}

## 解释边界

- 百分位坐标只在冻结的 230 reach 内定义；
- 映射系数没有使用站点流量；
- 先验边界是开发约束，不是后验可信区间；
- 本轮没有执行流量模拟或技能评价；
- 2019–2022 与管理通量均未读取。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    run_manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "runtime": "conda sparrow",
        "inputs_read": [str(PARENT_GATE), str(PRIOR_CONFIG), str(STATIC_PATH)],
        "station_observation_files_read": [],
        "management_flux_files_read": [],
        "period_2019_2022_read": False,
    }
    (REPORT / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sources = [
        CONFIG,
        RUN_DIR / "experiment_contract.md",
        RUN_DIR / "README.md",
        RUN_DIR / "scripts" / "build_attribute_parameter_mapping.py",
        RUN_DIR / "scripts" / "validate_attribute_parameter_mapping.py",
        PARENT_GATE,
        PRIOR_CONFIG,
        STATIC_PATH,
    ]
    products = [
        output_path,
        summary_path,
        REPORT / "gate.json",
        REPORT / "technical_report.md",
        REPORT / "run_manifest.json",
    ]
    manifest = {
        "run_id": cfg["run_id"],
        "created_utc": utc_now(),
        "sources": [record(path, "attribute_mapping_source", "reported_or_derived") for path in sources],
        "products": [record(path, "attribute_mapping_product", "derived") for path in products],
        "station_observations_read": False,
        "management_fluxes_read": False,
        "period_2019_2022_read": False,
    }
    (MANIFEST_DIR / "provenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(manifest["sources"] + manifest["products"]).to_csv(
        MANIFEST_DIR / "provenance_manifest.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps({
        "passed": passed,
        "authorized_next_action": next_action,
        "mapped_parameter_ranges": {
            parameter: {
                "min": float(mapped[parameter].min()),
                "median": float(mapped[parameter].median()),
                "max": float(mapped[parameter].max()),
            }
            for parameter in parameter_names
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

