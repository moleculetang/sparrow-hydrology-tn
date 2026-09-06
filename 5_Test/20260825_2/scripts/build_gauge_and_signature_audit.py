"""Rebuild Gauge support and development-only Q-derived signatures.

The 2019-2022 observations are physically separated into a locked file before
any later fitting.  No retrospective metric is computed here.  Hydrograph
separation is a response signature only and never a groundwater observation.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260825_2"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
COVERAGE = ROOT / "5_Test" / "20260823_14" / "outputs" / "final_user_locked_station_coverage.parquet"
STATION_MATCH = ROOT / "5_Test" / "20260823_14" / "reports" / "station_reach_match.csv"
OLD_DAILY = ROOT / "5_Test" / "20260823_28" / "outputs" / "daily_hydrograph_separation.parquet"
CHM = ROOT / "5_Test" / "20260823_35" / "outputs" / "daily_reach_forcing_2006_2022.parquet"
BRIDGE = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
EPS = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def terminal_map(topo: pd.DataFrame) -> dict[int, int]:
    downstream = {
        int(row.reach_id): None if pd.isna(row.downstream_reach) else int(row.downstream_reach)
        for row in topo.itertuples()
    }
    result: dict[int, int] = {}
    for reach in downstream:
        current = reach
        seen: set[int] = set()
        while downstream[current] is not None:
            if current in seen:
                raise RuntimeError(f"Topology cycle at Reach {current}")
            seen.add(current)
            current = int(downstream[current])
        result[reach] = current
    return result


def upstream_operator(reaches: np.ndarray, topo: pd.DataFrame) -> csr_matrix:
    index = {int(reach): i for i, reach in enumerate(reaches)}
    downstream = {
        int(row.reach_id): None if pd.isna(row.downstream_reach) else int(row.downstream_reach)
        for row in topo.itertuples()
    }
    rows: list[int] = []
    cols: list[int] = []
    for source in reaches:
        current: int | None = int(source)
        seen: set[int] = set()
        while current is not None:
            if current in seen:
                raise RuntimeError(f"Topology cycle at Reach {current}")
            seen.add(current)
            rows.append(index[current])
            cols.append(index[int(source)])
            current = downstream[current]
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(reaches), len(reaches)))


def lh_single(q: np.ndarray, alpha: float) -> np.ndarray:
    quick = np.zeros(len(q), dtype=float)
    for i in range(1, len(q)):
        quick[i] = alpha * quick[i - 1] + 0.5 * (1.0 + alpha) * (q[i] - q[i - 1])
        quick[i] = float(np.clip(quick[i], 0.0, q[i]))
    return q - quick


def lyne_hollick(q: np.ndarray, alpha: float) -> np.ndarray:
    first = lh_single(q, alpha)
    second = lh_single(first[::-1], alpha)[::-1]
    return np.clip(lh_single(second, alpha), 0.0, q)


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
    positions = []
    for start in range(0, len(q), block_days):
        stop = min(start + block_days, len(q))
        positions.append(start + int(np.argmin(q[start:stop])))
    turning = [positions[0]]
    for i in range(1, len(positions) - 1):
        left, center, right = positions[i - 1 : i + 2]
        if q[center] < factor * min(q[left], q[right]):
            turning.append(center)
    turning.append(positions[-1])
    turning = np.asarray(sorted(set(turning)), dtype=int)
    return np.clip(np.interp(np.arange(len(q)), turning, q[turning]), 0.0, q)


def contiguous_segments(dates: pd.DatetimeIndex, q: np.ndarray) -> list[np.ndarray]:
    valid_positions = np.flatnonzero(np.isfinite(q) & (q > 0))
    if not len(valid_positions):
        return []
    parts: list[list[int]] = [[int(valid_positions[0])]]
    for previous, current in zip(valid_positions[:-1], valid_positions[1:]):
        same_position_run = int(current) == int(previous) + 1
        one_day = dates[int(current)] - dates[int(previous)] == pd.Timedelta(days=1)
        if same_position_run and one_day:
            parts[-1].append(int(current))
        else:
            parts.append([int(current)])
    return [np.asarray(part, dtype=int) for part in parts]


def separate(part: pd.DataFrame) -> pd.DataFrame:
    part = part.sort_values("date").copy()
    dates = pd.DatetimeIndex(part.date)
    q = part.q_m3_s.to_numpy(float)
    variants: dict[str, list[np.ndarray]] = {"lh": [], "eckhardt": [], "ukih": []}
    segments = contiguous_segments(dates, q)
    for alpha in (0.90, 0.925, 0.95):
        values = np.full(len(q), np.nan)
        for positions in segments:
            values[positions] = lyne_hollick(q[positions], alpha)
        variants["lh"].append(values)
    for recession_a in (0.95, 0.98, 0.995):
        for bfi_max in (0.50, 0.65, 0.80):
            values = np.full(len(q), np.nan)
            for positions in segments:
                values[positions] = eckhardt(q[positions], recession_a, bfi_max)
            variants["eckhardt"].append(values)
    values = np.full(len(q), np.nan)
    for positions in segments:
        values[positions] = ukih(q[positions])
    variants["ukih"].append(values)
    for method, method_variants in variants.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            part[f"base_{method}_m3_s"] = np.nanmedian(np.vstack(method_variants), axis=0)
    part["segment_count"] = len(segments)
    return part


def correlation_at_lag(dates: pd.DatetimeIndex, values: np.ndarray, lag: int) -> float:
    pairs_x: list[np.ndarray] = []
    pairs_y: list[np.ndarray] = []
    for positions in contiguous_segments(dates, values):
        if len(positions) <= lag:
            continue
        x = np.log1p(values[positions[:-lag]])
        y = np.log1p(values[positions[lag:]])
        pairs_x.append(x)
        pairs_y.append(y)
    if not pairs_x:
        return float("nan")
    x = np.concatenate(pairs_x)
    y = np.concatenate(pairs_y)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 30 and np.std(x) > EPS and np.std(y) > EPS else float("nan")


def signature_rows(part: pd.DataFrame) -> list[dict[str, object]]:
    station = str(part.station_norm.iloc[0])
    reach = int(part.reach_id.iloc[0])
    tree = int(part.terminal_tree.iloc[0])
    part = part.sort_values("date").copy()
    part["season"] = pd.Categorical(
        np.select(
            [part.date.dt.month.isin([12, 1, 2]), part.date.dt.month.isin([3, 4, 5]), part.date.dt.month.isin([6, 7, 8])],
            ["DJF", "MAM", "JJA"],
            default="SON",
        ),
        categories=["DJF", "MAM", "JJA", "SON"],
    )
    rows: list[dict[str, object]] = []

    def add(group_type: str, group_value: str, signature: str, value: float, n: int, method: str = "Q-derived") -> None:
        rows.append(
            {
                "station_norm": station,
                "reach_id": reach,
                "terminal_tree": tree,
                "period": "development_2010_2018",
                "group_type": group_type,
                "group_value": group_value,
                "signature": signature,
                "value": value,
                "n_days": int(n),
                "method": method,
                "claim_role": "Q_DERIVED_RESPONSE_SIGNATURE_NOT_OBSERVED_GROUNDWATER",
            }
        )

    valid = part.q_m3_s.notna() & part.q_m3_s.gt(0)
    q = part.loc[valid, "q_m3_s"].to_numpy(float)
    if len(q):
        for probability in (0.05, 0.10, 0.50, 0.90, 0.95):
            add("overall", "all", f"flow_quantile_{probability:.2f}_m3_s", float(np.quantile(q, probability)), len(q), "empirical_FDC")
        add("overall", "all", "log_flow_lag1_correlation", correlation_at_lag(pd.DatetimeIndex(part.date), part.q_m3_s.to_numpy(float), 1), len(q), "contiguous_log1p_Q")
        add("overall", "all", "log_flow_lag7_correlation", correlation_at_lag(pd.DatetimeIndex(part.date), part.q_m3_s.to_numpy(float), 7), len(q), "contiguous_log1p_Q")
        add("overall", "all", "log_flow_lag30_correlation", correlation_at_lag(pd.DatetimeIndex(part.date), part.q_m3_s.to_numpy(float), 30), len(q), "contiguous_log1p_Q")
    for group_type, groups in (
        ("overall", [("all", part)]),
        ("year", [(str(year), group) for year, group in part.groupby(part.date.dt.year)]),
        ("season", [(str(season), group) for season, group in part.groupby("season", observed=True)]),
    ):
        for group_value, group in groups:
            total = float(group.q_m3_s.sum(skipna=True))
            n = int(group.q_m3_s.notna().sum())
            if total <= 0:
                continue
            method_values = []
            for method in ("lh", "eckhardt", "ukih"):
                base = float(group[f"base_{method}_m3_s"].sum(skipna=True))
                value = base / total
                method_values.append(value)
                add(group_type, group_value, f"bfi_{method}", value, n, f"{method}_volume_fraction")
            add(group_type, group_value, "bfi_three_method_median", float(np.median(method_values)), n, "median_of_three_volume_BFI")
    part["previous_q_m3_s"] = part.q_m3_s.shift()
    recession = part.loc[
        part.q_m3_s.notna()
        & part.previous_q_m3_s.notna()
        & part.q_m3_s.gt(0)
        & part.previous_q_m3_s.gt(0)
        & part.q_m3_s.lt(part.previous_q_m3_s)
        & part.precip_support_mm.le(1.0)
        & part.date.diff().eq(pd.Timedelta(days=1))
    ].copy()
    if len(recession) >= 30:
        slope = np.log(recession.q_m3_s.to_numpy(float) / recession.previous_q_m3_s.to_numpy(float))
        slope = slope[np.isfinite(slope) & (slope < 0)]
        if len(slope):
            median_slope = float(np.median(slope))
            add("overall", "all", "recession_log_ratio_median", median_slope, len(slope), "Q_decline_with_support_P_le_1mm")
            add("overall", "all", "recession_tau_days", float(-1.0 / median_slope), len(slope), "Q_decline_with_support_P_le_1mm")
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    topo = pd.read_csv(TOPOLOGY)
    terminal = terminal_map(topo)
    reaches = np.sort(topo.reach_id.astype(int).unique())
    upstream = upstream_operator(reaches, topo)

    coverage = pd.read_parquet(COVERAGE)
    coverage = coverage.loc[coverage.selected_for_model & coverage.station_type.eq("ordinary_river_gauge")].copy()
    coverage = coverage.drop_duplicates("station_norm")
    matches = pd.read_csv(STATION_MATCH, encoding="utf-8-sig")
    matches = matches.loc[matches.used].drop_duplicates("station_norm")
    coverage = coverage.merge(matches[["station_norm", "x", "y", "best_line_src_id"]], on="station_norm", how="left", validate="one_to_one")
    coverage["terminal_tree"] = coverage.reach_id.astype(int).map(terminal).astype(int)

    raw = pd.read_parquet(OLD_DAILY, columns=["station_norm", "date", "q_m3_s"])
    raw["date"] = pd.to_datetime(raw.date)
    original_missing_q = int((~np.isfinite(raw.q_m3_s)).sum())
    original_nonpositive_q = int((np.isfinite(raw.q_m3_s) & raw.q_m3_s.le(0)).sum())
    raw.loc[~np.isfinite(raw.q_m3_s) | raw.q_m3_s.le(0), "q_m3_s"] = np.nan
    raw = raw.merge(coverage[["station_norm", "reach_id", "terminal_tree", "downstream_fraction_on_reach"]], on="station_norm", how="inner", validate="many_to_one")
    development = raw.loc[raw.date.dt.year.between(2010, 2018)].copy()
    retrospective = raw.loc[raw.date.dt.year.between(2019, 2022)].copy()
    development.to_parquet(OUT / "daily_discharge_development_2010_2018.parquet", index=False)
    retrospective.to_parquet(OUT / "daily_discharge_locked_retrospective_2019_2022.parquet", index=False)

    area = pd.read_parquet(BRIDGE, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reaches)
    if area.catchment_area_km2.isna().any():
        raise RuntimeError("Missing local catchment area")
    local_area = area.catchment_area_km2.to_numpy(float)
    support_matrix = np.asarray(upstream[coverage.reach_id.to_numpy(int) - 1].todense(), dtype=float)
    for station_index, row in enumerate(coverage.itertuples()):
        local_index = int(row.reach_id) - 1
        support_matrix[station_index, local_index] = float(row.downstream_fraction_on_reach)
    support_area = support_matrix @ local_area
    coverage["topology_support_area_km2"] = support_area

    precipitation = pd.read_parquet(CHM, columns=["reach_id", "date", "precipitation_daily_mm"])
    precipitation["date"] = pd.to_datetime(precipitation.date)
    precipitation = precipitation.loc[precipitation.date.dt.year.between(2010, 2018)]
    p_matrix = precipitation.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(columns=reaches)
    if p_matrix.isna().any().any():
        raise RuntimeError("Development CHM precipitation is incomplete")
    local_p_volume_m3 = p_matrix.to_numpy(float) * local_area[None, :] * 1000.0
    support_p_volume_m3 = local_p_volume_m3 @ support_matrix.T
    support_p_mm = support_p_volume_m3 / (support_area[None, :] * 1000.0)
    p_long = pd.DataFrame(
        {
            "date": np.repeat(p_matrix.index.to_numpy(), len(coverage)),
            "station_norm": np.tile(coverage.station_norm.to_numpy(), len(p_matrix)),
            "precip_support_volume_m3": support_p_volume_m3.reshape(-1),
            "precip_support_mm": support_p_mm.reshape(-1),
        }
    )
    development = development.merge(p_long, on=["station_norm", "date"], how="left", validate="one_to_one")
    if development.precip_support_mm.isna().any():
        raise RuntimeError("Gauge-support precipitation join is incomplete")

    development["precip_volume_on_valid_q_days_m3"] = np.where(
        development.q_m3_s.notna(), development.precip_support_volume_m3, np.nan
    )
    annual = development.assign(year=development.date.dt.year)
    annual = annual.groupby(["station_norm", "year"], as_index=False).agg(
        valid_q_days=("q_m3_s", lambda values: int(values.notna().sum())),
        q_volume_m3=("q_m3_s", lambda values: float(values.dropna().sum() * 86400.0)),
        p_volume_m3=("precip_volume_on_valid_q_days_m3", lambda values: float(values.sum(skipna=True))),
    )
    annual["runoff_coefficient"] = annual.q_volume_m3 / annual.p_volume_m3.clip(lower=EPS)
    usable_annual = annual.loc[annual.valid_q_days.ge(300)].copy()
    runoff = usable_annual.groupby("station_norm", as_index=False).agg(
        runoff_coefficient_median=("runoff_coefficient", "median"),
        runoff_coefficient_min=("runoff_coefficient", "min"),
        runoff_coefficient_max=("runoff_coefficient", "max"),
        usable_runoff_years=("year", "nunique"),
    )
    coverage = coverage.merge(runoff, on="station_norm", how="left", validate="one_to_one")
    coverage["official_area_km2"] = np.nan
    coverage["official_to_topology_area_ratio"] = np.nan
    coverage["official_area_status"] = "NOT_AVAILABLE_IN_AUTHORITATIVE_LOCAL_GAUGE_METADATA"
    coverage["coordinate_pass"] = coverage.best_line_distance_m.le(5000)
    coverage["runoff_coefficient_pass"] = coverage.runoff_coefficient_median.between(0.02, 1.5)
    coverage["topology_representative"] = coverage.coordinate_pass & coverage.runoff_coefficient_pass & coverage.usable_runoff_years.ge(3)
    coverage["representativeness_reason"] = np.select(
        [~coverage.coordinate_pass, coverage.runoff_coefficient_median.isna(), ~coverage.runoff_coefficient_pass, coverage.usable_runoff_years.lt(3)],
        ["coordinate_outside_5km", "insufficient_runoff_audit", "runoff_coefficient_outside_0p02_1p5", "fewer_than_3_complete_runoff_years"],
        default="coordinate_and_runoff_support_pass",
    )
    coverage.to_parquet(OUT / "gauge_representativeness_audit.parquet", index=False)
    usable_annual.to_parquet(OUT / "gauge_annual_runoff_coefficient_audit.parquet", index=False)

    valid_stations = set(coverage.loc[coverage.topology_representative, "station_norm"])
    development_valid = development.loc[development.station_norm.isin(valid_stations)].copy()
    separated_parts = []
    for station, part in development_valid.groupby("station_norm", sort=False):
        separated_parts.append(separate(part))
    separated = pd.concat(separated_parts, ignore_index=True)
    separated.to_parquet(OUT / "development_q_separation_diagnostics.parquet", index=False)
    signatures: list[dict[str, object]] = []
    for station, part in separated.groupby("station_norm", sort=False):
        signatures.extend(signature_rows(part))
    signature_frame = pd.DataFrame(signatures)
    signature_frame.to_parquet(OUT / "q_derived_response_signatures_development.parquet", index=False)

    tree_counts = coverage.loc[coverage.topology_representative].groupby("terminal_tree").size().sort_values(ascending=False)
    valid_count = int(coverage.topology_representative.sum())
    audit = {
        "stage": "20260825_2",
        "selected_ordinary_gauges": int(len(coverage)),
        "topology_representative_gauges": valid_count,
        "coordinate_pass_gauges": int(coverage.coordinate_pass.sum()),
        "runoff_coefficient_pass_gauges": int(coverage.runoff_coefficient_pass.sum()),
        "official_area_available_gauges": int(coverage.official_area_km2.notna().sum()),
        "official_area_limitation": "Official drainage area was not present in the authoritative local Gauge metadata; topology area is therefore checked independently by coordinate and observed-runoff plausibility, not declared validated by area ratio.",
        "observed_terminal_trees": int(len(tree_counts)),
        "largest_tree_station_count": int(tree_counts.iloc[0]) if len(tree_counts) else 0,
        "largest_tree_station_fraction": float(tree_counts.iloc[0] / valid_count) if valid_count else 1.0,
        "development_rows": int(len(development)),
        "locked_retrospective_rows": int(len(retrospective)),
        "original_missing_q_rows": original_missing_q,
        "nonpositive_q_rows_set_missing": original_nonpositive_q,
        "signature_stations": int(signature_frame.station_norm.nunique()),
        "signature_rows": int(len(signature_frame)),
        "filters_cross_2018_boundary": False,
        "daily_fraction_likelihood_authorized": False,
    }
    audit["support_gate_pass"] = bool(
        valid_count >= 80
        and audit["observed_terminal_trees"] >= 6
        and audit["largest_tree_station_fraction"] < 0.85
    )
    audit["status"] = "PASS_GAUGE_AND_SIGNATURE_SUPPORT" if audit["support_gate_pass"] else "DATA_OR_TOPOLOGY_BLOCKED"
    (REPORT / "gauge_representativeness_and_signature_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "signature_definition_contract.json").write_text(
        json.dumps(
            {
                "period": "development_2010_2018 only",
                "nonpositive_q": "missing",
                "segment_boundaries": "missing dates and unavailable dates reset every filter; 2019 is in a physically separate file",
                "BFI": "annual, season and overall volume fractions from Lyne-Hollick, Eckhardt and UKIH plus their median",
                "FDC": "empirical daily Q quantiles",
                "memory": "contiguous log1p(Q) lag correlations",
                "recession": "declining contiguous Q pairs with Gauge-support CHM precipitation <=1 mm/day",
                "role": "soft envelope and evaluation signature only; never a daily-fraction likelihood or observed groundwater",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (REPORT / "discharge_split_integrity.json").write_text(
        json.dumps(
            {
                "source_sha256": sha256(OLD_DAILY),
                "development_sha256": sha256(OUT / "daily_discharge_development_2010_2018.parquet"),
                "locked_retrospective_sha256": sha256(OUT / "daily_discharge_locked_retrospective_2019_2022.parquet"),
                "development_years": [2010, 2018],
                "retrospective_years": [2019, 2022],
                "retrospective_metrics_computed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if not audit["support_gate_pass"]:
        raise RuntimeError(audit)


if __name__ == "__main__":
    main()
