from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
INPUTS = RUN / "inputs"
MANIFESTS = RUN / "inputs_manifest"
REPORTS = RUN / "reports" / "data_readiness_gate"
EXCLUDED = {"劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    gate = json.loads((REPORTS / "gate.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (MANIFESTS / "provenance_manifest.json").read_text(encoding="utf-8")
    )
    locked = json.loads(
        (MANIFESTS / "locked_confirmation_manifest.json").read_text(encoding="utf-8")
    )
    static = pd.read_parquet(INPUTS / "reach_static.parquet")
    forcing = pd.read_parquet(INPUTS / "reach_month_forcing_2006_2018.parquet")
    management = pd.read_parquet(
        INPUTS / "management_flux_scenario_2006_2018.parquet"
    )
    observations = pd.read_parquet(
        INPUTS / "station_observation_2006_2018.parquet"
    )

    hash_failures: list[str] = []
    for record in provenance["sources"] + provenance["products"]:
        path = Path(record["path"])
        if not path.exists() or sha256(path) != record["sha256"]:
            hash_failures.append(str(path))

    sector_sum = management[
        [
            "hswud_domestic_loc_cfs",
            "hswud_irrigation_loc_cfs",
            "hswud_manufacturing_loc_cfs",
            "hswud_thermal_loc_cfs",
        ]
    ].sum(axis=1)
    checks = {
        "gate_passed": bool(gate["passed"]),
        "source_and_product_hashes_match": len(hash_failures) == 0,
        "static_230_unique_reaches": (
            len(static) == 230 and static["reach_id"].nunique() == 230
        ),
        "forcing_2006_2018_only": (
            len(forcing) == 35880
            and int(forcing["year"].min()) == 2006
            and int(forcing["year"].max()) == 2018
        ),
        "management_2006_2018_only": (
            len(management) == 35880
            and int(management["year"].min()) == 2006
            and int(management["year"].max()) == 2018
        ),
        "observations_2006_2018_only": (
            int(observations["year"].min()) == 2006
            and int(observations["year"].max()) == 2018
        ),
        "fixed_exclusions_absent": EXCLUDED.isdisjoint(
            set(observations["station_name"])
        ),
        "shijiao_present": bool((observations["station_name"] == "石角站").any()),
        "legacy_slope_not_exposed_as_zero": (
            static["slope_m_m"].isna().all()
            and static["slope_status"].eq("missing_requires_dem_rebuild").all()
        ),
        "soil_complete": static["soil_storage_eff_mm"].notna().all(),
        "bedrock_complete": static["depth_to_bedrock_m"].notna().all(),
        "hswud_gross_total_recomputed_from_four_sectors": np.allclose(
            management["hswud_gross_total_recomputed_cfs"],
            sector_sum,
            rtol=1e-12,
            atol=1e-12,
        ),
        "hswud_bad_source_total_not_exposed": (
            "hswud_total_loc_cfs" not in management.columns
        ),
        "locked_observations_not_copied": (
            locked["observations_copied"] is False
            and gate["locked_observations_loaded"] is False
        ),
        "next_action_is_slope_repair": (
            gate["authorized_next_action"] == "REPAIR_REACH_SLOPE"
        ),
        "formal_man_is_blocked": (
            gate["gates"]["q78_man_formal_data_ready"] is False
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = bool(all(checks.values()))
    payload = {
        "run_id": RUN.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": "conda sparrow environment executable",
        "checks": checks,
        "hash_failures": hash_failures,
        "passed": passed,
    }
    (REPORTS / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"LEDGER_VALIDATION passed={passed} checks={len(checks)}")
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit(f"Ledger validation failed: {failed}; hashes={hash_failures}")


if __name__ == "__main__":
    main()
