from __future__ import annotations

import calendar
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
BASE = RUN / "inputs" / "baseline" / "indata.parquet"
TOPOLOGY = RUN / "inputs" / "topology" / "topology_edges.csv"
OUT = RUN / "inputs" / "scenarios"
TABLES = RUN / "reports" / "tables"
M3S_TO_CFS = 35.3146667


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def downstream_ids(value: object) -> list[int]:
    if pd.isna(value) or not str(value).strip():
        return []
    out = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if token:
            out.append(int(float(token)))
    return out


def upstream_sets(topo: pd.DataFrame) -> dict[int, set[int]]:
    reverse: dict[int, set[int]] = defaultdict(set)
    reaches = set(topo["reach_id"].astype(int))
    for row in topo[["reach_id", "downstream_reach"]].itertuples(index=False):
        for downstream in downstream_ids(row.downstream_reach):
            reverse[int(downstream)].add(int(row.reach_id))
    result: dict[int, set[int]] = {}
    for reach in reaches:
        seen = {reach}
        stack = [reach]
        while stack:
            current = stack.pop()
            for parent in reverse.get(current, set()):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        result[reach] = seen
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(BASE).sort_values(["comid", "year", "month"]).reset_index(drop=True)
    topo = pd.read_csv(TOPOLOGY, encoding="utf-8-sig")
    if len(base) != 46920 or base["comid"].nunique() != 230:
        raise RuntimeError(f"Frozen panel gate failed: {base.shape}")
    if base.duplicated(["comid", "year", "month"]).any():
        raise RuntimeError("Reach-month keys are not unique")
    if int(base["Q_obsv_cfs"].notna().sum()) != 21440:
        raise RuntimeError("Observed row gate failed")
    upstream = upstream_sets(topo)
    area = topo.set_index("reach_id")["inc_area_km2"].astype(float)
    declared = topo.set_index("reach_id")["tot_area_km2"].astype(float)
    closure_rows = []
    for reach, members in upstream.items():
        summed = float(area.reindex(sorted(members)).sum())
        target = float(declared.loc[reach])
        closure_rows.append({
            "reach_id": reach, "upstream_reach_count": len(members),
            "sum_incremental_area_km2": summed, "declared_total_area_km2": target,
            "relative_error": abs(summed - target) / max(target, 1e-12),
        })
    closure = pd.DataFrame(closure_rows).sort_values("reach_id")
    closure.to_csv(TABLES / "explicit_upstream_area_closure.csv", index=False, encoding="utf-8-sig")
    if closure["relative_error"].max() > 1e-8:
        raise RuntimeError(f"Upstream area closure failed: {closure['relative_error'].max()}")

    work = base.copy()
    days = np.array([calendar.monthrange(int(y), int(m))[1] for y, m in zip(work["year"], work["month"])])
    seconds = days * 86400.0
    local_net_volume = np.maximum(work["PPT"].to_numpy(float) - work["AET"].to_numpy(float), 0.0) / 1000.0
    local_net_volume *= work["IncAreaKm2"].to_numpy(float) * 1_000_000.0
    local_threshold_volume = np.maximum(work["PPT"].to_numpy(float) - 0.8 * work["PET"].to_numpy(float), 0.0) / 1000.0
    local_threshold_volume *= work["IncAreaKm2"].to_numpy(float) * 1_000_000.0
    work["_local_net_volume_m3"] = local_net_volume
    work["_local_threshold_volume_m3"] = local_threshold_volume

    key_to_position = {
        (int(r.comid), int(r.year), int(r.month)): int(i)
        for i, r in enumerate(work[["comid", "year", "month"]].itertuples(index=False))
    }
    upstream_net = np.zeros(len(work), dtype=float)
    upstream_threshold = np.zeros(len(work), dtype=float)
    for reach, members in upstream.items():
        for year in range(2006, 2023):
            for month in range(1, 13):
                target = key_to_position[(reach, year, month)]
                sources = [key_to_position[(member, year, month)] for member in members]
                upstream_net[target] = float(work.iloc[sources]["_local_net_volume_m3"].sum())
                upstream_threshold[target] = float(work.iloc[sources]["_local_threshold_volume_m3"].sum())
    work["explicit_upstream_net_cfs"] = upstream_net / seconds * M3S_TO_CFS
    work["explicit_upstream_threshold_cfs"] = upstream_threshold / seconds * M3S_TO_CFS
    work = work.drop(columns=["_local_net_volume_m3", "_local_threshold_volume_m3"])

    baseline_approx = work["Q_calc_cfs"].to_numpy(float)
    explicit_qcalc = 0.35 * work["explicit_upstream_net_cfs"].to_numpy(float)
    comparable = explicit_qcalc > 1e-12
    comparison = pd.DataFrame({
        "comid": work.loc[comparable, "comid"].to_numpy(int),
        "year": work.loc[comparable, "year"].to_numpy(int),
        "month": work.loc[comparable, "month"].to_numpy(int),
        "legacy_qcalc_cfs": baseline_approx[comparable],
        "explicit_qcalc_cfs": explicit_qcalc[comparable],
    })
    comparison["ratio_legacy_to_explicit"] = comparison["legacy_qcalc_cfs"] / comparison["explicit_qcalc_cfs"]
    comparison["relative_abs_difference"] = np.abs(comparison["legacy_qcalc_cfs"] - comparison["explicit_qcalc_cfs"]) / comparison["explicit_qcalc_cfs"]
    comparison.to_parquet(TABLES / "legacy_vs_explicit_upstream_rows.parquet", index=False)

    paths = {}
    for scenario in ["R3_00", "R3_10", "R3_01", "R3_11"]:
        frame = work.copy()
        if scenario in {"R3_01", "R3_11"}:
            frame["Q_calc_cfs"] = explicit_qcalc
            frame["Q_ma_cfs"] = frame.groupby("comid")["Q_calc_cfs"].transform("mean")
            frame["MAFlowUcfs"] = frame["Q_ma_cfs"]
        path = OUT / f"{scenario}_indata.parquet"
        frame.to_parquet(path, index=False)
        paths[scenario] = {"path": str(path), "sha256": sha256(path), "rows": len(frame)}

    summary = {
        "runtime": RUNTIME,
        "baseline_sha256": sha256(BASE),
        "topology_sha256": sha256(TOPOLOGY),
        "panel_rows": len(base),
        "observed_rows": int(base["Q_obsv_cfs"].notna().sum()),
        "oof_positive_rows_2012_2018": int((base["Q_obsv_cfs"].gt(0) & base["year"].between(2012, 2018)).sum()),
        "max_upstream_area_relative_error": float(closure["relative_error"].max()),
        "legacy_to_explicit_ratio_median": float(comparison["ratio_legacy_to_explicit"].median()),
        "legacy_to_explicit_ratio_p95": float(comparison["ratio_legacy_to_explicit"].quantile(0.95)),
        "legacy_to_explicit_relative_abs_p95": float(comparison["relative_abs_difference"].quantile(0.95)),
        "scenario_inputs": paths,
    }
    (RUN / "reports" / "r3_input_build.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

