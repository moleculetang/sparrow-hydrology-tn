from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260818_1")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main() -> None:
    temporal = pd.read_parquet(OUT / "readout_level_predictions.parquet")
    spatial = pd.read_parquet(OUT / "spatial_holdout_predictions.parquet")
    skills = pd.read_parquet(OUT / "station_blind_spatial_skill.parquet")
    audit = json.loads((REPORTS / "completion_audit.json").read_text(encoding="utf-8"))
    checks = {
        "parent_hashes_unchanged": bool(audit["parent_hashes_unchanged"]),
        "twelve_models": bool(temporal.model_id.nunique() == 12),
        "three_temporal_layers": set(temporal.layer) == {"P0", "P1", "P2"},
        "oof_4097_each": bool(temporal.groupby(["model_id", "layer"]).size().eq(4097).all()),
        "spatial_two_layers": set(spatial.layer) == {"P0", "P1"},
        "heldout_station_effect_zero": bool(spatial.station_effect.abs().max() == 0.0),
        "two_spatial_schemes": set(spatial.spatial_scheme) == {"LOSO", "LOTO"},
        "skill_rows": bool(len(skills) == 24),
        "no_2022": bool(temporal.year.max() == 2021 and spatial.year.max() == 2021),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {"status": status, "checks": checks}
    (REPORTS / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if status != "PASS":
        raise RuntimeError(result)


if __name__ == "__main__":
    main()
