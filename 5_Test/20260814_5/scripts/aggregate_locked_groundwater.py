from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from runtime_guard import assert_sparrow_runtime


assert_sparrow_runtime()
Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT.parent / "20260814_1"
LOCK = ROOT / "model_lock.json"
WEIGHTS = STAGE1 / "inputs" / "spatial" / "groundwater_grid_overlap_weights.parquet"
MANIFEST = STAGE1 / "reports" / "groundwater_source_manifest.csv"
OUT = ROOT / "outputs" / "groundwater_catchment_monthly_2019_2022.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5 * weights.sum(), side="left")])


def aggregate() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    locked = manifest[manifest.year.between(2019, 2022)].sort_values(["year", "month"])
    if len(locked) != 48:
        raise RuntimeError("Expected exactly 48 locked groundwater rasters")
    weights = pd.read_parquet(WEIGHTS)
    minr, maxr = int(weights.raster_row.min()), int(weights.raster_row.max())
    minc, maxc = int(weights.raster_col.min()), int(weights.raster_col.max())
    local_r = weights.raster_row.to_numpy(int) - minr
    local_c = weights.raster_col.to_numpy(int) - minc
    groups = {int(rid): np.asarray(idx, int) for rid, idx in weights.groupby("reach_id").groups.items()}
    rows = []
    for item in locked.itertuples(index=False):
        with Image.open(Path(item.path)) as image:
            crop = np.asarray(image.crop((minc, minr, maxc + 1, maxr + 1)), dtype=float)
        values = crop[local_r, local_c]
        valid = np.isfinite(values) & (values > -1e30)
        for rid, idx in groups.items():
            base = weights.loc[idx, "base_weight"].to_numpy(float)
            good = valid[idx]
            valid_fraction = float(base[good].sum())
            mean = median = np.nan
            if valid_fraction >= 0.95 and good.any():
                ww = base[good] / valid_fraction
                vv = values[idx][good]
                mean = float(np.sum(ww * vv))
                median = weighted_median(vv, ww)
            rows.append({
                "reach_id": rid,
                "year": int(item.year),
                "month": int(item.month),
                "groundwater_area_mean_m": mean,
                "groundwater_area_weighted_median_m": median,
                "valid_area_fraction": valid_fraction,
                "static_weight_sum": float(base.sum()),
                "area_method": "frozen_dense8_pixel_footprint_Albers_exact_intersection",
            })
    frame = pd.DataFrame(rows).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    if len(frame) != 11040 or frame[["reach_id", "year", "month"]].duplicated().any():
        raise RuntimeError("Locked groundwater row/key gate failed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    return frame


def descriptive_diagnostics(gw: pd.DataFrame) -> dict[str, object]:
    states = pd.read_parquet(
        STAGE1 / "outputs" / "fixed_branch_local_states.parquet",
        columns=["branch_id", "comid", "year", "month", "gw_response_state_end_mm"],
    )
    states = states[(states.branch_id == "main") & states.year.between(2019, 2022)].copy()
    joined = gw.merge(
        states,
        left_on=["reach_id", "year", "month"],
        right_on=["comid", "year", "month"],
        validate="one_to_one",
    ).sort_values(["reach_id", "year", "month"])
    summaries = {}
    for label, field in {
        "area_mean": "groundwater_area_mean_m",
        "area_weighted_median": "groundwater_area_weighted_median_m",
    }.items():
        x = joined[["reach_id", "year", "month", field, "gw_response_state_end_mm"]].copy()
        x["dh"] = x.groupby("reach_id")[field].diff()
        x["ds"] = x.groupby("reach_id").gw_response_state_end_mm.diff()
        valid = x.dropna(subset=["dh", "ds"])
        corr = valid.groupby("reach_id").apply(
            lambda g: g.dh.corr(g.ds) if g.dh.nunique() > 1 and g.ds.nunique() > 1 else np.nan,
            include_groups=False,
        )
        sign_hit = float((np.sign(valid.dh) == np.sign(valid.ds)).mean())
        hclim = joined.groupby("month")[field].mean()
        sclim = joined.groupby("month").gw_response_state_end_mm.mean()
        peak_h, peak_s = int(hclim.idxmax()), int(sclim.idxmax())
        circular = min(abs(peak_h - peak_s), 12 - abs(peak_h - peak_s))
        summaries[label] = {
            "valid_change_pairs": int(len(valid)),
            "median_catchment_change_correlation": float(corr.median()),
            "fraction_catchments_positive_change_correlation": float((corr > 0).mean()),
            "change_direction_hit_rate": sign_hit,
            "seasonal_peak_month_groundwater": peak_h,
            "seasonal_peak_month_main_state": peak_s,
            "seasonal_peak_circular_month_error": int(circular),
        }
    mean_direction = np.sign(summaries["area_mean"]["median_catchment_change_correlation"])
    median_direction = np.sign(summaries["area_weighted_median"]["median_catchment_change_correlation"])
    return {
        "role": "locked_period_descriptive_only_not_structure_selection",
        "groundwater_status": "non_identifying",
        "reason": "no_single_production_candidate_passed_development_flow_gate",
        "aggregation_sensitivity": summaries,
        "mean_median_direction_consistent": bool(mean_direction == median_direction),
    }


def main() -> None:
    report_path = ROOT / "reports" / "locked_groundwater_descriptive.json"
    if report_path.exists() or OUT.exists():
        raise RuntimeError("Locked groundwater values have already been aggregated; repeat access is forbidden")
    if not LOCK.exists():
        raise RuntimeError("model_lock.json must exist before locked groundwater values are read")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if not lock.get("lock_written_before_locked_value_access"):
        raise RuntimeError("Invalid model lock")
    frame = aggregate()
    report = descriptive_diagnostics(frame)
    report.update({
        "rows": int(len(frame)),
        "reaches": int(frame.reach_id.nunique()),
        "months": int(frame[["year", "month"]].drop_duplicates().shape[0]),
        "minimum_valid_area_fraction": float(frame.valid_area_fraction.min()),
        "output_sha256": sha256(OUT),
        "pass": bool(len(frame) == 11040 and frame.reach_id.nunique() == 230),
    })
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
