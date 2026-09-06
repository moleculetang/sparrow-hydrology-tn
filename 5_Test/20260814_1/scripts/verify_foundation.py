from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from runtime_guard import assert_sparrow_runtime


assert_sparrow_runtime()
ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    non_gis = read_json(ROOT / "reports" / "foundation_non_gis_gate.json")
    gis = read_json(ROOT / "reports" / "groundwater_input_gate.json")
    scenario = read_json(ROOT / "scenario_contract.json")
    metrics = read_json(ROOT / "metrics_contract.json")
    dev = pd.read_parquet(ROOT / "inputs" / "development_indata_2006_2018.parquet", columns=["year"])
    state = pd.read_parquet(ROOT / "inputs" / "state_forcing_2006_2022_no_locked_observations.parquet", columns=["year", "Q_obsv_cfs"])
    monthly = pd.read_parquet(ROOT / "inputs" / "spatial" / "groundwater_catchment_monthly_2005_2018.parquet", columns=["year"])
    required = [
        "reports/h0_reproduction_audit.json", "reports/baseline_feature_transform_audit.json",
        "reports/local_interface_audit.json", "reports/tail_registry_audit.json",
        "reports/groundwater_input_gate.json", "reports/environment.json",
        "reports/parameter_manifest.json", "reports/spinup_contract.json",
        "reports/instrumentation_neutrality.json", "reports/interface_schema.json",
        "inputs/registries/tail_event_registry.parquet", "inputs/registries/tail_month_registry.parquet",
        "inputs/registries/wet_dry_registry.parquet", "outputs/fixed_branch_local_states.parquet",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    checks = {
        "non_gis_gate": bool(non_gis.get("pass")),
        "gis_gate": bool(gis.get("pass")),
        "development_input_max_year": int(dev.year.max()),
        "groundwater_values_max_year": int(monthly.year.max()),
        "state_forcing_locked_observation_nonnull": int(state.loc[state.year >= 2019, "Q_obsv_cfs"].notna().sum()),
        "scenario_contract_parsed": bool(scenario),
        "metrics_contract_parsed": bool(metrics),
        "required_missing": missing,
    }
    checks["pass"] = bool(checks["non_gis_gate"] and checks["gis_gate"] and checks["development_input_max_year"] == 2018 and checks["groundwater_values_max_year"] == 2018 and checks["state_forcing_locked_observation_nonnull"] == 0 and not missing)
    target = ROOT / "reports" / "foundation_gate.json"
    target.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    if not checks["pass"]:
        raise RuntimeError(f"Foundation verification failed: {checks}")


if __name__ == "__main__":
    main()
