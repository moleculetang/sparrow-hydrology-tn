from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent.parent
REPORTS = RUN / "reports"
INPUTS = RUN / "inputs"
FOCUS = "灵渠（三）站"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    policy_path = INPUTS / "source_metadata" / "station_screening_policy.csv"
    panel_path = INPUTS / "indata.parquet"
    files_path = REPORTS / "input_preprocessing" / "discharge_files_all.csv"
    excluded_path = REPORTS / "input_preprocessing" / "excluded_active_calibration_stations.csv"
    policy = pd.read_csv(policy_path, encoding="utf-8-sig")
    panel = pd.read_parquet(panel_path)
    sources = pd.read_csv(files_path, encoding="utf-8-sig")
    excluded = pd.read_csv(excluded_path, encoding="utf-8-sig")
    focus_policy = policy.loc[policy["station_name"].eq(FOCUS)]
    if len(focus_policy) != 1 or not bool(focus_policy["exclude_before_training"].iloc[0]):
        raise AssertionError("Lingqu（三）must be explicitly excluded in this run policy")
    station_columns = [c for c in ["station_name", "q_site", "station_norm"] if c in panel.columns]
    present_in_panel = {c: int(panel[c].astype(str).eq(FOCUS).sum()) for c in station_columns}
    present_in_excluded = int(excluded.astype(str).apply(lambda col: col.eq(FOCUS)).any(axis=1).sum())
    current_root = ROOT / "1_Inputs" / "DischargeData"
    path_column = "path" if "path" in sources.columns else "file_path"
    source_paths = [Path(x) for x in sources[path_column].dropna().astype(str)] if path_column in sources.columns else []
    used_current_root = bool(source_paths) and all(str(p).startswith(str(current_root)) for p in source_paths)
    source_groups = sorted(sources["source_group"].dropna().astype(str).unique().tolist()) if "source_group" in sources.columns else []
    gate = {
        "run_id": RUN.name,
        "phase": "fresh_latest_discharge_input_excluding_lingqu_san",
        "policy_excludes_lingqu_san": True,
        "lingqu_san_rows_in_model_panel": present_in_panel,
        "lingqu_san_is_recorded_in_excluded_station_report": present_in_excluded > 0,
        "all_discharge_input_paths_under_current_DischargeData": used_current_root,
        "discharge_path_column": path_column,
        "discharge_source_groups_seen": source_groups,
        "input_panel_sha256": sha256(panel_path),
        "policy_sha256": sha256(policy_path),
    }
    gate["passed"] = bool(
        all(v == 0 for v in present_in_panel.values())
        and gate["lingqu_san_is_recorded_in_excluded_station_report"]
        and used_current_root
    )
    (REPORTS / "latest_input_lingqu_exclusion_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Latest DischargeData and Lingqu Exclusion Gate",
        "",
        f"- Current `DischargeData` paths used: {used_current_root}",
        f"- Source groups: {', '.join(source_groups)}",
        f"- Lingqu（三） rows in model panel: {present_in_panel}",
        f"- Lingqu（三） recorded in excluded-station report: {present_in_excluded > 0}",
        f"- Gate: {'PASS' if gate['passed'] else 'FAIL'}",
    ]
    (REPORTS / "latest_input_lingqu_exclusion_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not gate["passed"]:
        raise SystemExit("Latest-input/Lingqu-exclusion gate failed")


if __name__ == "__main__":
    main()
