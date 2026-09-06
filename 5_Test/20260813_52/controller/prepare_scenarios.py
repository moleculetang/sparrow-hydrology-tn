from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / "20260813_50"
SCENARIO_ROOT = ROOT / "scenarios"

GROUPS = {
    "G_shared": {"天生桥站", "泸江站"},
    "G_oof_only": {"富宁_2", "河步（二）站", "六陈水库（渠道）站"},
    "G_2019_only": {
        "八达（坝上）站", "百色（三）站", "北流站", "岔江站", "峨山站",
        "官良站", "榕峰（三）站", "天峨站", "严洞站", "尤家寨站",
    },
}
SCENARIOS = {
    "S000": set(),
    "S100": GROUPS["G_shared"],
    "S010": GROUPS["G_oof_only"],
    "S001": GROUPS["G_2019_only"],
    "S110": GROUPS["G_shared"] | GROUPS["G_oof_only"],
    "S101": GROUPS["G_shared"] | GROUPS["G_2019_only"],
    "S011": GROUPS["G_oof_only"] | GROUPS["G_2019_only"],
    "S111": GROUPS["G_shared"] | GROUPS["G_oof_only"] | GROUPS["G_2019_only"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if SCENARIO_ROOT.exists():
        raise RuntimeError(f"Refuse to overwrite existing scenario root: {SCENARIO_ROOT}")
    source_input = PARENT / "inputs" / "parent_indata.parquet"
    source = pd.read_parquet(source_input)
    if len(source) != 46920 or source.comid.nunique() != 230:
        raise RuntimeError("Parent forcing identity gate failed")
    if "石角站" in set().union(*GROUPS.values()):
        raise RuntimeError("Protected 石角站 entered an exclusion group")

    station_rows = []
    for group, stations in GROUPS.items():
        for station in sorted(stations):
            station_rows.append({"group_id": group, "q_site": station})
    pd.DataFrame(station_rows).to_csv(ROOT / "station_group_registry.csv", index=False, encoding="utf-8-sig")

    scenario_rows = []
    for scenario, excluded in SCENARIOS.items():
        work = SCENARIO_ROOT / scenario
        (work / "inputs" / "topology").mkdir(parents=True)
        (work / "scripts" / "components").mkdir(parents=True)
        (work / "reports").mkdir(parents=True)
        shutil.copy2(PARENT / "inputs" / "topology" / "topology_edges.csv", work / "inputs" / "topology" / "topology_edges.csv")
        for filename in ["runtime_guard.py", "run_prior_semantics_repair.py"]:
            shutil.copy2(PARENT / "scripts" / filename, work / "scripts" / filename)
        shutil.copy2(ROOT / "controller" / "run_2019_2022_validation_generic.py", work / "scripts" / "run_2019_2022_validation.py")
        shutil.copy2(ROOT / "controller" / "gate_scenario.py", work / "scripts" / "gate_scenario.py")
        shutil.copy2(
            PARENT / "scripts" / "components" / "q72_prior_semantics_component.py",
            work / "scripts" / "components" / "q72_prior_semantics_component.py",
        )
        filtered = source.copy()
        mask = filtered.q_site.astype(str).isin(excluded)
        positive_masked = int((mask & filtered.Q_obsv_cfs.notna() & filtered.Q_obsv_cfs.gt(0)).sum())
        filtered.loc[mask, "Q_obsv_cfs"] = np.nan
        filtered.to_parquet(work / "inputs" / "parent_indata.parquet", index=False)
        registry = pd.DataFrame({"scenario_id": scenario, "q_site": sorted(excluded)})
        registry.to_csv(work / "inputs" / "excluded_stations.csv", index=False, encoding="utf-8-sig")
        for station in sorted(excluded):
            scenario_rows.append({"scenario_id": scenario, "q_site": station})
        contract = {
            "scenario_id": scenario,
            "excluded_station_count": len(excluded),
            "excluded_stations": sorted(excluded),
            "protected_stations": ["石角站"],
            "source_rows": len(source),
            "filtered_rows": len(filtered),
            "reaches": int(filtered.comid.nunique()),
            "positive_observations_masked": positive_masked,
            "source_input_sha256": sha256(source_input),
            "filtered_input_sha256": sha256(work / "inputs" / "parent_indata.parquet"),
        }
        (work / "scenario_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(scenario_rows).to_csv(ROOT / "scenario_exclusion_registry.csv", index=False, encoding="utf-8-sig")
    payload = {
        "terminal": "EIGHT_SCENARIO_WORKSPACES_PREPARED",
        "scenario_count": len(SCENARIOS),
        "scenario_ids": list(SCENARIOS),
        "source_sha256": sha256(source_input),
    }
    (ROOT / "preparation_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
