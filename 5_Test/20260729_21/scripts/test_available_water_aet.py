from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[1]
ENGINE = ROOT / "5_Test" / "20260729_20" / "scripts" / "test_available_water_aet.py"
BASE_MAP = ROOT / "5_Test" / "20260729_17" / "outputs" / "reach_attribute_parameter_map.parquet"
DERIVED_MAP = RUN_DIR / "inputs" / "reach_attribute_parameter_map_gamma_0_5.parquet"


def main() -> None:
    cfg = json.loads(
        (RUN_DIR / "config" / "available_water_aet_contract.json").read_text(
            encoding="utf-8"
        )
    )
    DERIVED_MAP.parent.mkdir(parents=True, exist_ok=True)
    parameter_map = pd.read_parquet(BASE_MAP)
    parameter_map["gamma_ET"] = cfg["gamma_ET_override"]
    parameter_map["parameter_semantic_state"] = (
        "aet_prior_lower_bound_test_not_calibrated"
    )
    parameter_map.to_parquet(DERIVED_MAP, index=False)

    spec = importlib.util.spec_from_file_location("available_water_aet_engine", ENGINE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the frozen `_20` AET engine")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.RUN_DIR = RUN_DIR
    engine.REPORT = RUN_DIR / "reports" / "available_water_aet_gate"
    engine.OUTPUTS = RUN_DIR / "outputs"
    engine.MANIFEST_DIR = RUN_DIR / "inputs_manifest"
    engine.CONFIG = RUN_DIR / "config" / "available_water_aet_contract.json"
    engine.PARENT_GATE = (
        ROOT / "5_Test" / "20260729_20"
        / "reports" / "available_water_aet_gate" / "gate.json"
    )
    engine.PARAMETER_MAP = DERIVED_MAP
    engine.main()

    manifest_path = engine.MANIFEST_DIR / "provenance_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"].append(
        engine.record(ENGINE, "frozen_available_water_aet_engine", "reported")
    )
    manifest["sources"].append(
        engine.record(BASE_MAP, "base_attribute_parameter_map", "reported_or_derived")
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(manifest["sources"] + manifest["products"]).to_csv(
        engine.MANIFEST_DIR / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()

