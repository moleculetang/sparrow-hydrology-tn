from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_28"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
REGISTRY = ROOT / "1_Inputs" / "DischargeData" / "registry" / "stage2_result_revised_sync_operations.csv"
COVERAGE = TEST / "20260823_14" / "outputs" / "final_user_locked_station_coverage.parquet"
OBSERVATIONS = TEST / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
BRIDGE = TEST / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def station_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().replace(" ", "").replace("\u3000", "")
    text = text.replace("(", "（").replace(")", "）")
    for suffix, replacement in [("_2", "（二）"), ("_3", "（三）"), ("_4", "（四）")]:
        text = re.sub(re.escape(suffix) + r"$", replacement, text)
    return re.sub(r"站$", "", text)


def authoritative_files(selected: set[str]) -> pd.DataFrame:
    files = pd.read_csv(REGISTRY, encoding="utf-8-sig")
    files["station_norm_key"] = files.station_norm.map(station_key)
    files["exists"] = files.target_path.map(lambda value: Path(str(value)).is_file())
    files = files[
        files.model_input_status.eq("ACTIVE")
        & files.target_group.isin(["complete_2010_2022", "noncomplete_2010_2022"])
        & files.year.between(2010, 2022)
        & files.station_norm_key.isin(selected)
        & files.exists
    ].copy()
    files = files.sort_values(["station_norm_key", "year", "supplement_priority", "target_path"])
    files = files.groupby(["station_norm_key", "year"], as_index=False).tail(1)
    return files


def read_daily(files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    errors: list[dict[str, object]] = []
    for record in files.itertuples(index=False):
        path = Path(str(record.target_path))
        try:
            raw = pd.read_csv(path)
            raw.columns = [str(column).strip().lower() for column in raw.columns]
            if "day" not in raw or not set(MONTH_NAMES).issubset(raw.columns):
                raise ValueError(f"Unexpected daily CSV columns: {raw.columns.tolist()}")
            day = pd.to_numeric(raw.day, errors="coerce")
            for month, column in enumerate(MONTH_NAMES, start=1):
                values = pd.to_numeric(raw[column], errors="coerce")
                dates = pd.to_datetime({
                    "year": np.full(len(raw), int(record.year)),
                    "month": np.full(len(raw), month),
                    "day": day,
                }, errors="coerce")
                part = pd.DataFrame({
                    "station_norm": str(record.station_norm_key), "date": dates,
                    "year": int(record.year), "month": month, "q_m3_s": values,
                    "source_path": str(path),
                })
                rows.append(part[part.date.notna()])
        except Exception as exc:
            errors.append({"station_norm": str(record.station_norm_key), "year": int(record.year), "path": str(path), "error": repr(exc)})
    if not rows:
        raise RuntimeError("No authoritative daily streamflow could be read")
    daily = pd.concat(rows, ignore_index=True)
    daily.loc[~np.isfinite(daily.q_m3_s) | daily.q_m3_s.lt(0), "q_m3_s"] = np.nan
    daily = daily.sort_values(["station_norm", "date"]).drop_duplicates(["station_norm", "date"], keep="last")
    return daily.reset_index(drop=True), pd.DataFrame(errors)


def contiguous_segments(values: np.ndarray) -> list[np.ndarray]:
    valid = np.isfinite(values) & (values >= 0)
    positions = np.flatnonzero(valid)
    if not len(positions):
        return []
    cuts = np.flatnonzero(np.diff(positions) > 1) + 1
    return [part for part in np.split(positions, cuts) if len(part)]


def lh_single(q: np.ndarray, alpha: float) -> np.ndarray:
    quick = np.zeros(len(q), dtype=float)
    for i in range(1, len(q)):
        quick[i] = alpha * quick[i - 1] + 0.5 * (1.0 + alpha) * (q[i] - q[i - 1])
        quick[i] = float(np.clip(quick[i], 0.0, q[i]))
    return q - quick


def lyne_hollick(q: np.ndarray, alpha: float) -> np.ndarray:
    first = lh_single(q, alpha)
    second = lh_single(first[::-1], alpha)[::-1]
    third = lh_single(second, alpha)
    return np.clip(third, 0.0, q)


def eckhardt(q: np.ndarray, recession_a: float, bfi_max: float) -> np.ndarray:
    base = np.zeros(len(q), dtype=float)
    if not len(q):
        return base
    base[0] = min(q[0], bfi_max * q[0])
    denominator = 1.0 - recession_a * bfi_max
    for i in range(1, len(q)):
        value = ((1.0 - bfi_max) * recession_a * base[i - 1] + (1.0 - recession_a) * bfi_max * q[i]) / denominator
        base[i] = float(np.clip(value, 0.0, q[i]))
    return base


def ukih(q: np.ndarray, block_days: int = 5, factor: float = 0.9) -> np.ndarray:
    if len(q) < block_days * 3:
        return np.full(len(q), np.nan)
    block_positions = []
    for start in range(0, len(q), block_days):
        stop = min(start + block_days, len(q))
        local = int(np.argmin(q[start:stop])) + start
        block_positions.append(local)
    turning = [block_positions[0]]
    for i in range(1, len(block_positions) - 1):
        left, center, right = block_positions[i - 1:i + 2]
        if q[center] < factor * min(q[left], q[right]):
            turning.append(center)
    turning.append(block_positions[-1])
    turning = np.array(sorted(set(turning)), dtype=int)
    base = np.interp(np.arange(len(q)), turning, q[turning])
    return np.clip(base, 0.0, q)


def separate_station(part: pd.DataFrame) -> pd.DataFrame:
    q = part.q_m3_s.to_numpy(float)
    result = part[["station_norm", "date", "year", "month", "q_m3_s"]].copy()
    method_values: dict[str, list[np.ndarray]] = {"lh": [], "eckhardt": [], "ukih": []}
    for alpha in [0.90, 0.925, 0.95]:
        arr = np.full(len(q), np.nan)
        for idx in contiguous_segments(q):
            arr[idx] = lyne_hollick(q[idx], alpha)
        method_values["lh"].append(arr)
    for recession_a in [0.95, 0.98, 0.995]:
        for bfi_max in [0.50, 0.65, 0.80]:
            arr = np.full(len(q), np.nan)
            for idx in contiguous_segments(q):
                arr[idx] = eckhardt(q[idx], recession_a, bfi_max)
            method_values["eckhardt"].append(arr)
    arr = np.full(len(q), np.nan)
    for idx in contiguous_segments(q):
        arr[idx] = ukih(q[idx])
    method_values["ukih"].append(arr)
    method_medians = []
    all_variants = []
    for method, variants in method_values.items():
        stack = np.vstack(variants)
        median = np.nanmedian(stack, axis=0)
        result[f"delayed_{method}_m3_s"] = median
        method_medians.append(median)
        all_variants.extend(variants)
    method_stack = np.vstack(method_medians)
    variant_stack = np.vstack(all_variants)
    result["delayed_proxy_median_m3_s"] = np.nanmedian(method_stack, axis=0)
    result["delayed_proxy_lower_m3_s"] = np.nanmin(variant_stack, axis=0)
    result["delayed_proxy_upper_m3_s"] = np.nanmax(variant_stack, axis=0)
    return result


def monthly_proxy(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["valid"] = daily.q_m3_s.notna()
    fields = ["delayed_lh_m3_s", "delayed_eckhardt_m3_s", "delayed_ukih_m3_s", "delayed_proxy_median_m3_s", "delayed_proxy_lower_m3_s", "delayed_proxy_upper_m3_s"]
    aggregations = {field: (field, "sum") for field in fields}
    monthly = daily.groupby(["station_norm", "year", "month"], as_index=False).agg(
        total_flow_volume_proxy=("q_m3_s", "sum"), valid_days=("valid", "sum"), calendar_days=("date", "size"), **aggregations,
    )
    monthly["daily_coverage"] = monthly.valid_days / monthly.calendar_days
    monthly["proxy_usable"] = monthly.daily_coverage.ge(0.90) & monthly.total_flow_volume_proxy.gt(0)
    for field in fields:
        name = field.replace("_m3_s", "_fraction")
        monthly[name] = np.divide(
            monthly[field], monthly.total_flow_volume_proxy,
            out=np.full(len(monthly), np.nan), where=monthly.total_flow_volume_proxy.to_numpy(float) > EPS,
        )
        monthly.loc[~monthly.proxy_usable, name] = np.nan
    monthly["proxy_role"] = "HYDROGRAPH_SEPARATION_DELAYED_FLOW_PROXY_NOT_OBSERVED_GROUNDWATER"
    monthly["period"] = np.where(monthly.year.le(2018), "development_2010_2018", "frozen_check_2019_2022")
    return monthly


def circular_month_distance(a: int, b: int) -> int:
    difference = abs(int(a) - int(b))
    return min(difference, 12 - difference)


def baseline_metrics(merged: pd.DataFrame, period: str) -> dict[str, float | int]:
    work = merged[(merged.period.eq(period)) & merged.delayed_proxy_median_fraction.notna()].copy()
    station = work.groupby(["station_norm", "reach_id"], as_index=False).agg(
        proxy_bfi=("delayed_proxy_median_m3_s", "sum"), model_slow=("support_delayed_volume_m3", "sum"),
        observed_total=("total_flow_volume_proxy", "sum"), model_total=("support_total_volume_m3", "sum"),
    )
    station["proxy_fraction"] = station.proxy_bfi / station.observed_total.clip(lower=EPS)
    station["model_fraction"] = station.model_slow / station.model_total.clip(lower=EPS)
    station["absolute_fraction_error"] = (station.model_fraction - station.proxy_fraction).abs()
    correlations, peak_distances = [], []
    for _, part in work.groupby("station_norm"):
        climatology = part.groupby("month", as_index=False).agg(proxy=("delayed_proxy_median_fraction", "mean"), model=("support_delayed_fraction", "mean"))
        if len(climatology) >= 9 and climatology.proxy.std() > EPS and climatology.model.std() > EPS:
            correlations.append(float(climatology.proxy.corr(climatology.model)))
            peak_distances.append(circular_month_distance(climatology.loc[climatology.proxy.idxmax(), "month"], climatology.loc[climatology.model.idxmax(), "month"]))
    rho = spearmanr(station.proxy_fraction, station.model_fraction, nan_policy="omit").statistic if len(station) >= 3 else np.nan
    return {
        "station_count": int(len(station)), "month_count": int(len(work)),
        "proxy_bfi_median": float(station.proxy_fraction.median()),
        "model_delayed_fraction_median": float(station.model_fraction.median()),
        "absolute_fraction_error_median": float(station.absolute_fraction_error.median()),
        "fraction_with_absolute_error_le_0_25": float(station.absolute_fraction_error.le(0.25).mean()),
        "station_spearman": float(rho),
        "monthly_seasonal_correlation_median": float(np.nanmedian(correlations)) if correlations else np.nan,
        "peak_month_distance_median": float(np.nanmedian(peak_distances)) if peak_distances else np.nan,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    registered = pd.read_parquet(COVERAGE)
    observed_keys = pd.read_parquet(OBSERVATIONS, columns=["station_norm", "reach_id"]).drop_duplicates()
    coverage = observed_keys.merge(
        registered[["station_norm", "reach_id", "station_type", "downstream_fraction_on_reach"]],
        on=["station_norm", "reach_id"], how="left", validate="one_to_one",
    )
    if coverage[["station_type", "downstream_fraction_on_reach"]].isna().any().any():
        raise RuntimeError("Authoritative 105-station key could not be joined to Gauge support metadata")
    coverage["station_norm"] = coverage.station_norm.map(station_key)
    selected = set(coverage.station_norm)
    files = authoritative_files(selected)
    files.to_parquet(OUT / "authoritative_daily_file_manifest.parquet", index=False)
    daily_raw, errors = read_daily(files)
    errors.to_parquet(OUT / "daily_read_errors.parquet", index=False)
    ordinary = set(coverage.loc[coverage.station_type.eq("ordinary_river_gauge"), "station_norm"])
    daily_raw = daily_raw[daily_raw.station_norm.isin(ordinary)].copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        separated = pd.concat([separate_station(part) for _, part in daily_raw.groupby("station_norm", sort=False)], ignore_index=True)
    separated.to_parquet(OUT / "daily_hydrograph_separation.parquet", index=False)
    proxy = monthly_proxy(separated)
    proxy = proxy.merge(coverage[["station_norm", "reach_id", "station_type", "downstream_fraction_on_reach"]], on="station_norm", how="inner", validate="many_to_one")
    proxy.to_parquet(OUT / "monthly_hydrograph_separation_proxy.parquet", index=False)

    hydro = pd.read_parquet(BRIDGE)
    support = hydro[[
        "reach_id", "year", "month", "month_seconds",
        "local_fast_volume_m3", "local_delayed_volume_m3", "local_total_volume_m3",
        "routed_fast_volume_m3", "routed_delayed_volume_m3", "routed_total_volume_m3",
    ]].merge(coverage[["station_norm", "reach_id", "downstream_fraction_on_reach"]], on="reach_id", how="inner", validate="many_to_one")
    fraction = support.downstream_fraction_on_reach.clip(0, 1).to_numpy(float)
    support["support_fast_volume_m3"] = support.routed_fast_volume_m3 - support.local_fast_volume_m3 + fraction * support.local_fast_volume_m3
    support["support_delayed_volume_m3"] = support.routed_delayed_volume_m3 - support.local_delayed_volume_m3 + fraction * support.local_delayed_volume_m3
    support["support_total_volume_m3"] = support.support_fast_volume_m3 + support.support_delayed_volume_m3
    support["support_fast_m3_s"] = support.support_fast_volume_m3 / support.month_seconds
    support["support_delayed_m3_s"] = support.support_delayed_volume_m3 / support.month_seconds
    support["support_total_m3_s"] = support.support_total_volume_m3 / support.month_seconds
    support["support_delayed_fraction"] = np.divide(
        support.support_delayed_volume_m3, support.support_total_volume_m3,
        out=np.zeros(len(support)), where=support.support_total_volume_m3.to_numpy(float) > EPS,
    )
    support.to_parquet(OUT / "gauge_support_hydrology.parquet", index=False)

    merged = proxy.merge(support[[
        "station_norm", "reach_id", "year", "month", "support_fast_volume_m3",
        "support_delayed_volume_m3", "support_total_volume_m3", "support_delayed_fraction",
    ]], on=["station_norm", "reach_id", "year", "month"], how="inner", validate="one_to_one")
    usable = merged[merged.proxy_usable & merged.delayed_proxy_median_fraction.notna()].copy()
    usable.to_parquet(OUT / "q72_vs_separation_proxy_monthly.parquet", index=False)
    metrics = {
        "development_2010_2018": baseline_metrics(usable, "development_2010_2018"),
        "frozen_check_2019_2022": baseline_metrics(usable, "frozen_check_2019_2022"),
    }
    fractions = proxy.filter(regex="_fraction$").to_numpy(float)
    finite_fractions = fractions[np.isfinite(fractions)]
    closure = support.support_total_m3_s - support.support_fast_m3_s - support.support_delayed_m3_s
    audit = {
        "stage": "20260823_28", "selected_station_count": int(len(selected)),
        "ordinary_station_count": int(len(ordinary)), "active_daily_file_count": int(len(files)),
        "daily_station_count": int(separated.station_norm.nunique()), "daily_row_count": int(len(separated)),
        "valid_proxy_station_count": int(usable.station_norm.nunique()), "valid_proxy_month_count": int(len(usable)),
        "read_error_count": int(len(errors)),
        "fraction_min": float(np.min(finite_fractions)), "fraction_max": float(np.max(finite_fractions)),
        "support_closure_max_abs_m3_s": float(np.max(np.abs(closure))),
        "baseline_metrics": metrics,
    }
    gates = contract["hard_gates"]
    audit["hard_gate_pass"] = bool(
        audit["valid_proxy_station_count"] >= int(gates["minimum_valid_proxy_station_count"])
        and audit["fraction_min"] >= float(gates["fraction_min"]) - 1e-12
        and audit["fraction_max"] <= float(gates["fraction_max"]) + 1e-12
        and audit["support_closure_max_abs_m3_s"] <= float(gates["support_closure_max_abs_m3_s"])
    )
    (REPORT / "stage28_separation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit["hard_gate_pass"]:
        raise RuntimeError(f"Stage28 hard gate failed: {audit}")
    dev = metrics["development_2010_2018"]
    check = metrics["frozen_check_2019_2022"]
    report = f"""# 20260823_28 逐日流量分割与Gauge支持审计\n\n## 结论\n\n`PASS`。三方法分割得到{audit['valid_proxy_station_count']}个有效自然站、{audit['valid_proxy_month_count']}个有效站月，可进入后续联合MAP作为软约束。\n\n该分割是水文过程代理，不是实测地下水。\n\n## 当前Q72快慢路径诊断\n\n### 2010–2018 development\n\n- 代理慢流比例中位数：{dev['proxy_bfi_median']:.3f}\n- Q72延迟流比例中位数：{dev['model_delayed_fraction_median']:.3f}\n- 绝对差中位数：{dev['absolute_fraction_error_median']:.3f}\n- 绝对差≤0.25站比例：{dev['fraction_with_absolute_error_le_0_25']:.1%}\n- 站际Spearman：{dev['station_spearman']:.3f}\n- 月季节相关中位数：{dev['monthly_seasonal_correlation_median']:.3f}\n- 峰月差中位数：{dev['peak_month_distance_median']:.1f}月\n\n### 2019–2022 frozen check\n\n- 代理/Q72延迟流比例中位数：{check['proxy_bfi_median']:.3f} / {check['model_delayed_fraction_median']:.3f}\n- 绝对差中位数：{check['absolute_fraction_error_median']:.3f}\n\n## Gauge支持\n\nGauge支持流量使用完整上游贡献加本Reach沿程比例，不再默认每个测站都位于Reach出口。该算子只定义观测支持，不改变导出给TN的Reach水量。\n"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    integrity = {
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "station_lock_sha256": sha256(COVERAGE), "observation_key_sha256": sha256(OBSERVATIONS), "bridge_sha256": sha256(BRIDGE),
        "proxy_sha256": sha256(OUT / "monthly_hydrograph_separation_proxy.parquet"),
        "support_sha256": sha256(OUT / "gauge_support_hydrology.parquet"),
    }
    (REPORT / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
