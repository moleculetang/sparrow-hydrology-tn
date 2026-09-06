from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT.parent
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
S000 = TEST / "20260813_52" / "scenarios" / "S000" / "inputs" / "parent_indata.parquet"
S111 = TEST / "20260813_54" / "inputs" / "parent_indata.parquet"
DESIGN = TEST / "20260823_7" / "outputs" / "final_fit" / "fit_artifacts" / "design_matrix"
COORD = TEST / "20260809_3" / "reports" / "all_station_coordinate_based_spatial_classification.csv"
REGISTRY = TEST.parent / "1_Inputs" / "DischargeData" / "registry" / "discharge_station_year_registry.json"
SEED = 20260823


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_station(value: object) -> str:
    text = str(value).strip().replace("（", "(").replace("）", ")")
    text = re.sub(r"\(重复\)$", "", text)
    text = re.sub(r"_\d+$", "", text)
    return text[:-1] if text.endswith("站") else text


def population_audit() -> dict[str, object]:
    rows = []
    for label, path in [("S000", S000), ("S111", S111)]:
        frame = pd.read_parquet(path)
        obs = frame[pd.to_numeric(frame["Q_obsv_cfs"], errors="coerce").gt(0)].copy()
        rows.append(
            {
                "population": label,
                "panel_rows": len(frame),
                "observed_rows": len(obs),
                "observed_stations": obs["q_site"].nunique(),
                "observed_reaches": obs["comid"].nunique(),
                "oof_2012_2018_rows": len(obs[obs["year"].between(2012, 2018)]),
                "oof_2012_2018_stations": obs.loc[obs["year"].between(2012, 2018), "q_site"].nunique(),
                "retrospective_2019_2022_rows": len(obs[obs["year"].between(2019, 2022)]),
                "retrospective_2019_2022_stations": obs.loc[obs["year"].between(2019, 2022), "q_site"].nunique(),
                "sha256": sha256(path),
            }
        )
    out = pd.DataFrame(rows)
    out.to_parquet(OUTPUTS / "population_audit.parquet", index=False)
    return {row["population"]: row for row in rows}


def map_audit() -> dict[str, object]:
    manifest = json.loads((DESIGN / "design_matrix_manifest.json").read_text(encoding="utf-8"))
    columns = pd.read_csv(DESIGN / "design_matrix_columns.csv", encoding="utf-8-sig")
    group_counts = columns["parameter_block"].value_counts(dropna=False).to_dict()
    names = columns["parameter"].astype(str)
    explicit_time = names[names.str.contains(r"year|trend|drift|time_state", case=False, regex=True)].tolist()
    payload = {
        "objective": "Gaussian-prior ridge/MAP linear least squares",
        "solver": "numpy.linalg.lstsq",
        "convex": True,
        "rows": manifest.get("observation_rows"),
        "columns": int(len(columns)),
        "augmented_rank": manifest.get("rank"),
        "condition_number": manifest.get("condition_number"),
        "feature_group_counts": group_counts,
        "explicit_chronological_trend_columns": explicit_time,
        "optimizer_change_needed_for_same_objective": False,
        "interpretation": "persistent drift is a model/forcing/error-structure issue, not evidence of a local optimum"
    }
    dump(REPORTS / "map_solver_audit.json", payload)
    return payload


def fixed_effect_trend(frame: pd.DataFrame, model_id: str, period: str) -> dict[str, object]:
    work = frame.copy()
    work["residual"] = np.log1p(work["actual"].clip(lower=0)) - np.log1p(work["predict"].clip(lower=0))
    work["year_center"] = work["year"] - work["year"].mean()
    controls = pd.get_dummies(
        work[["q_site", "month"]].astype({"q_site": str, "month": str}),
        drop_first=True,
        dtype=float,
    )
    if "fold_id" in work:
        fold = pd.get_dummies(work["fold_id"].astype(str), prefix="fold", drop_first=True, dtype=float)
        controls = pd.concat([controls, fold], axis=1)
    x = np.column_stack([np.ones(len(work)), work["year_center"].to_numpy(float), controls.to_numpy(float)])
    y = work["residual"].to_numpy(float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    annual = work.assign(adjusted=y - x[:, 2:] @ beta[2:]).groupby("year")["adjusted"].mean()
    annual = annual - annual.mean()
    residual = y - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]), float)
    for positions in work.groupby("q_site", sort=False).indices.values():
        pos = np.asarray(positions, dtype=int)
        score = x[pos].T @ residual[pos]
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    slope_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    return {
        "period": period,
        "model_id": model_id,
        "rows": len(work),
        "stations": work["q_site"].nunique(),
        "log_residual_slope_per_year": float(beta[1]),
        "slope_ci95_lower": float(beta[1] - 1.96 * slope_se),
        "slope_ci95_upper": float(beta[1] + 1.96 * slope_se),
        "J_year": float(np.sqrt(np.mean(annual.to_numpy(float) ** 2))),
    }


def nonstationarity_audit() -> list[dict[str, object]]:
    paths = {
        "P0_PROCESS": TEST / "20260823_4" / "outputs" / "P0_PROCESS" / "oof.parquet",
        "P1_GLOBAL": TEST / "20260823_4" / "outputs" / "P1_GLOBAL" / "oof.parquet",
        "P2_CURRENT": TEST / "20260823_3" / "outputs" / "main" / "oof.parquet",
    }
    rows = []
    for model_id, path in paths.items():
        frame = pd.read_parquet(path)
        rows.append(fixed_effect_trend(frame, model_id, "development_oof_2012_2018"))
    locked = pd.read_parquet(TEST / "20260823_7" / "outputs" / "locked_retrospective_predictions_2019_2022.parquet")
    rows.append(fixed_effect_trend(locked, "P2_CURRENT", "retrospective_2019_2022"))
    p0 = locked.copy()
    p0["predict"] = p0["routed_quick_cfs"].clip(lower=0) + p0["routed_base_cfs"].clip(lower=0)
    rows.append(fixed_effect_trend(p0, "P0_PROCESS", "retrospective_2019_2022"))
    pd.DataFrame(rows).to_parquet(OUTPUTS / "residual_nonstationarity_audit.parquet", index=False)
    return rows


def discharge_archive_audit() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    summary = pd.DataFrame(registry["station_summary"])
    summary["station_key"] = summary["station"].map(norm_station)
    coord = pd.read_csv(COORD, encoding="utf-8-sig")
    coord["station_key_normalized"] = coord["station"].map(norm_station)
    coord["archive_exact_or_alias_match"] = coord["station_key_normalized"].isin(set(summary["station_key"]))
    selected_reaches = set(pd.read_parquet(S000).loc[lambda x: x["Q_obsv_cfs"].gt(0), "comid"].astype(int))
    coord["assigned_reach_int"] = pd.to_numeric(coord["assigned_reach_id"], errors="coerce").astype("Int64")
    coord["selected_S000_reach"] = coord["assigned_reach_int"].isin(selected_reaches)
    coord["potential_unseen_reach"] = coord["assigned_reach_int"].notna() & ~coord["selected_S000_reach"]
    coord["spatially_reliable"] = coord["coordinate_spatial_class"].eq("SPATIALLY_CONSISTENT_MAIN_REACH")
    summary["archive_2010_2022_year_count"] = pd.to_numeric(summary["archive_2010_2022_year_count"], errors="coerce")
    years = summary.groupby("station_key")["archive_2010_2022_year_count"].max()
    coord["archive_years_2010_2022"] = coord["station_key_normalized"].map(years)
    coord["usable_unseen_reach_candidate"] = (
        coord["potential_unseen_reach"]
        & coord["spatially_reliable"]
        & coord["archive_exact_or_alias_match"]
        & coord["archive_years_2010_2022"].fillna(0).ge(3)
    )
    coord.to_parquet(OUTPUTS / "discharge_archive_independence_audit.parquet", index=False)
    payload = {
        "registry_station_entities": int(len(summary)),
        "normalized_registry_station_entities": int(summary["station_key"].nunique()),
        "coordinate_rows": int(len(coord)),
        "coordinate_archive_matches": int(coord["archive_exact_or_alias_match"].sum()),
        "S000_observed_reaches": int(len(selected_reaches)),
        "potential_unseen_reach_rows_before_quality_gates": int(coord["potential_unseen_reach"].sum()),
        "usable_unseen_reach_candidate_rows": int(coord["usable_unseen_reach_candidate"].sum()),
        "usable_unseen_reaches": sorted(coord.loc[coord["usable_unseen_reach_candidate"], "assigned_reach_int"].dropna().astype(int).unique().tolist()),
        "same_reach_or_alias_records_are_spatially_independent": False,
        "decision": "additional records may support replication/quality checks; only explicitly usable unseen reaches can be called spatially independent"
    }
    dump(REPORTS / "discharge_archive_independence_audit.json", payload)
    return payload


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    result = {
        "population": population_audit(),
        "map": map_audit(),
        "nonstationarity": nonstationarity_audit(),
        "discharge_archive": discharge_archive_audit(),
        "stage_status": "STAGE8_COMPLETE_READY_FOR_CANDIDATES",
    }
    dump(REPORTS / "stage8_decision.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
