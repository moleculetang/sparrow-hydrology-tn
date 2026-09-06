from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
BASE_PRED = RUN / "reports" / "main_model" / "reach_class_selected_predictions_long.csv"
TOPO_PATH = ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
REPORT = RUN / "reports" / "layered_upstream_transfer"
FIG = RUN / "figure" / "layered_upstream_transfer"
LOGS = RUN / "logs"
DAILY_LOG = ROOT / "5_Test" / "20260611.log"
EPS = 1.0e-6


KEY_STATIONS = [
    "博罗（二）站",
    "河源站",
    "龙川站",
    "枫树坝水库（坝下二）站",
    "平山（三）站",
    "石角站",
    "高要站",
    "梧州（二）站",
]


def ensure_dirs() -> None:
    for p in [REPORT, FIG, LOGS]:
        p.mkdir(parents=True, exist_ok=True)


def split_name(year: int) -> str:
    if 2006 <= year <= 2015:
        return "train"
    if 2016 <= year <= 2018:
        return "inner"
    if 2019 <= year <= 2022:
        return "strict"
    return "unused"


def setup_plot() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def nse_log(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    if mask.sum() < 3:
        return np.nan
    lo = np.log(obs[mask] + EPS)
    lp = np.log(pred[mask] + EPS)
    denom = np.sum((lo - lo.mean()) ** 2)
    return np.nan if denom <= 0 else float(1.0 - np.sum((lp - lo) ** 2) / denom)


def kge_2012(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) < 3 or np.std(obs) <= 0 or np.std(pred) <= 0 or np.mean(obs) == 0:
        return np.nan
    r = float(np.corrcoef(obs, pred)[0, 1])
    beta = float(np.mean(pred) / np.mean(obs))
    cv_obs = float(np.std(obs) / np.mean(obs))
    cv_pred = float(np.std(pred) / np.mean(pred))
    gamma = cv_pred / cv_obs if cv_obs > 0 else np.nan
    if not np.isfinite(r) or not np.isfinite(beta) or not np.isfinite(gamma):
        return np.nan
    return float(1.0 - math.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2))


def pbias(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0 or np.sum(obs) == 0:
        return np.nan
    return float(100.0 * np.sum(pred - obs) / np.sum(obs))


def rmse_log(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((np.log(pred[mask] + EPS) - np.log(obs[mask] + EPS)) ** 2)))


def corr_log(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    if mask.sum() < 3:
        return np.nan
    lo = np.log(obs[mask] + EPS)
    lp = np.log(pred[mask] + EPS)
    if np.std(lo) <= 0 or np.std(lp) <= 0:
        return np.nan
    return float(np.corrcoef(lo, lp)[0, 1])


def is_good(nselog: float, kge_val: float, pbias_val: float, n: int) -> bool:
    return bool(
        n >= 24
        and np.isfinite(nselog)
        and np.isfinite(kge_val)
        and np.isfinite(pbias_val)
        and nselog >= 0.65
        and kge_val >= 0.50
        and abs(pbias_val) <= 25.0
    )


def metric_rows(frame: pd.DataFrame, pred_col: str, label: str, period: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for site, part in frame.groupby("q_site", sort=True):
        obs = part["Q_obsv_cfs"].to_numpy(float)
        pred = part[pred_col].to_numpy(float)
        ns = nse_log(obs, pred)
        kg = kge_2012(obs, pred)
        pb = pbias(obs, pred)
        rows.append(
            {
                "period": period,
                "model": label,
                "q_site": site,
                "reach_id": int(part["reach_id"].iloc[0]),
                "reach_class": str(part["reach_class"].iloc[0]),
                "layer_id": int(part["layer_id"].iloc[0]),
                "has_upstream_sim": int(part["has_upstream_sim"].iloc[0]),
                "upstream_station_count": int(part["upstream_station_count"].iloc[0]),
                "n": int(np.isfinite(pred).sum()),
                "NSElog": ns,
                "KGE_2012": kg,
                "PBIAS_pct": pb,
                "abs_PBIAS_pct": abs(pb) if np.isfinite(pb) else np.nan,
                "RMSElog": rmse_log(obs, pred),
                "Rlog": corr_log(obs, pred),
                "good": is_good(ns, kg, pb, int(np.isfinite(pred).sum())),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, model, group_name), part in metrics.groupby(["period", "model", "summary_group"], sort=False):
        rows.append(
            {
                "period": period,
                "model": model,
                "group": group_name,
                "station_count": int(part["q_site"].nunique()),
                "median_NSElog": float(part["NSElog"].median()),
                "median_KGE": float(part["KGE_2012"].median()),
                "median_absPBIAS": float(part["abs_PBIAS_pct"].median()),
                "median_RMSElog": float(part["RMSElog"].median()),
                "median_Rlog": float(part["Rlog"].median()),
                "good_count": int(part["good"].sum()),
            }
        )
    return pd.DataFrame(rows)


def parse_reach_list(value: object) -> list[int]:
    if pd.isna(value):
        return []
    out: list[int] = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(float(token)))
        except ValueError:
            pass
    return out


def load_topology() -> dict[int, list[int]]:
    topo = pd.read_csv(TOPO_PATH, encoding="utf-8-sig")
    upstream: dict[int, list[int]] = {}
    for row in topo[["reach_id", "upstream_reaches"]].itertuples(index=False):
        upstream[int(row.reach_id)] = parse_reach_list(row.upstream_reaches)
    return upstream


def upstream_set(reach_id: int, upstream: dict[int, list[int]]) -> set[int]:
    seen: set[int] = set()
    stack = list(upstream.get(int(reach_id), []))
    while stack:
        rid = int(stack.pop())
        if rid in seen:
            continue
        seen.add(rid)
        stack.extend(upstream.get(rid, []))
    return seen


def build_station_network(pred: pd.DataFrame, upstream: dict[int, list[int]]) -> pd.DataFrame:
    stations = pred[["q_site", "reach_id", "reach_class"]].drop_duplicates("q_site").sort_values("q_site")
    station_reaches = {str(r.q_site): int(r.reach_id) for r in stations.itertuples(index=False)}
    upstream_cache = {rid: upstream_set(rid, upstream) for rid in set(station_reaches.values())}

    def is_upstream_of(a: int, b: int) -> bool:
        return int(a) in upstream_cache.get(int(b), set())

    rows: list[dict[str, object]] = []
    for target, target_reach in station_reaches.items():
        candidates = [
            (site, rid)
            for site, rid in station_reaches.items()
            if site != target and rid in upstream_cache.get(target_reach, set())
        ]
        immediate: list[tuple[str, int]] = []
        for site, rid in candidates:
            has_downstream_candidate = False
            for other_site, other_rid in candidates:
                if other_site == site:
                    continue
                if rid == other_rid or is_upstream_of(rid, other_rid):
                    has_downstream_candidate = True
                    break
            if not has_downstream_candidate:
                immediate.append((site, rid))
        if not immediate:
            rows.append(
                {
                    "target_site": target,
                    "target_reach_id": target_reach,
                    "upstream_site": "",
                    "upstream_reach_id": "",
                    "upstream_type": "none",
                }
            )
        else:
            for site, rid in sorted(immediate, key=lambda x: (x[1], x[0])):
                rows.append(
                    {
                        "target_site": target,
                        "target_reach_id": target_reach,
                        "upstream_site": site,
                        "upstream_reach_id": rid,
                        "upstream_type": "immediate_upstream_simulated_station",
                    }
                )
    return pd.DataFrame(rows)


def assign_layers(network: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    stations = sorted(pred["q_site"].astype(str).unique())
    parents: dict[str, set[str]] = {s: set() for s in stations}
    children: dict[str, set[str]] = {s: set() for s in stations}
    for row in network[network["upstream_type"].eq("immediate_upstream_simulated_station")].itertuples(index=False):
        up = str(row.upstream_site)
        down = str(row.target_site)
        parents[down].add(up)
        children[up].add(down)

    indegree = {s: len(parents[s]) for s in stations}
    queue = deque(sorted([s for s, d in indegree.items() if d == 0]))
    layer = {s: 0 for s in stations}
    visited: list[str] = []
    while queue:
        cur = queue.popleft()
        visited.append(cur)
        for nxt in sorted(children[cur]):
            layer[nxt] = max(layer[nxt], layer[cur] + 1)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    for station in stations:
        if station not in visited:
            layer[station] = max([layer.get(p, 0) for p in parents[station]] + [0]) + 1

    inv_rows: list[dict[str, object]] = []
    station_meta = pred[["q_site", "reach_id", "reach_class"]].drop_duplicates("q_site").set_index("q_site")
    for station in stations:
        meta = station_meta.loc[station]
        inv_rows.append(
            {
                "q_site": station,
                "reach_id": int(meta["reach_id"]),
                "reach_class": str(meta["reach_class"]),
                "layer_id": int(layer[station]),
                "upstream_station_count": int(len(parents[station])),
                "immediate_upstream_stations": ";".join(sorted(parents[station])),
            }
        )
    return pd.DataFrame(inv_rows).sort_values(["layer_id", "q_site"]).reset_index(drop=True)


def lag_by_site(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("q_site", sort=False)[col].shift(1).fillna(df[col])


def add_upstream_features(panel: pd.DataFrame, upstream_pred_col: str) -> pd.DataFrame:
    out = panel.copy()
    for col in [
        "Qmain_cfs",
        "Q72_cfs",
        "Q78_mass_cfs",
        "upstream_qmain_sum_cfs",
        "upstream_q72_sum_cfs",
        "upstream_q78_sum_cfs",
        upstream_pred_col,
    ]:
        out[f"log_{col}"] = np.log(out[col].clip(lower=EPS))
    out["log_up_qmain_to_target_qmain"] = np.log((out["upstream_qmain_sum_cfs"] + EPS) / (out["Qmain_cfs"] + EPS)).clip(-4, 4)
    out["log_up_q78_to_target_q78"] = np.log((out["upstream_q78_sum_cfs"] + EPS) / (out["Q78_mass_cfs"] + EPS)).clip(-4, 4)
    out["log_up_q78_to_up_q72"] = np.log((out["upstream_q78_sum_cfs"] + EPS) / (out["upstream_q72_sum_cfs"] + EPS)).clip(-4, 4)
    out["log_up_qcorr_to_target_qmain"] = np.log((out[upstream_pred_col] + EPS) / (out["Qmain_cfs"] + EPS)).clip(-4, 4)
    out["log_up_qcorr_to_up_qmain"] = np.log((out[upstream_pred_col] + EPS) / (out["upstream_qmain_sum_cfs"] + EPS)).clip(-4, 4)
    out["month_sin"] = np.sin(2 * np.pi * out["month"].astype(float) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"].astype(float) / 12.0)
    for col in [
        "log_up_qmain_to_target_qmain",
        "log_up_q78_to_target_q78",
        "log_up_q78_to_up_q72",
        "log_up_qcorr_to_target_qmain",
        "log_up_qcorr_to_up_qmain",
    ]:
        out[f"{col}_lag1"] = lag_by_site(out, col)
    return out


def design_matrix(panel: pd.DataFrame, feature_mode: str, class_names: list[str]) -> tuple[np.ndarray, list[str]]:
    gate = panel["has_upstream_sim"].astype(float).to_numpy()
    selected_by_mode = {
        "cascade_core": [
            "log_up_qcorr_to_target_qmain",
            "log_up_qcorr_to_up_qmain",
            "log_up_q78_to_up_q72",
        ],
        "cascade_core_lag": [
            "log_up_qcorr_to_target_qmain",
            "log_up_qcorr_to_up_qmain",
            "log_up_q78_to_up_q72",
            "log_up_qcorr_to_target_qmain_lag1",
            "log_up_qcorr_to_up_qmain_lag1",
            "log_up_q78_to_up_q72_lag1",
        ],
        "cascade_with_baseline_gap": [
            "log_up_qmain_to_target_qmain",
            "log_up_q78_to_target_q78",
            "log_up_qcorr_to_target_qmain",
            "log_up_qcorr_to_up_qmain",
            "log_up_q78_to_up_q72",
        ],
    }
    selected = selected_by_mode[feature_mode]
    cols: list[np.ndarray] = []
    names: list[str] = []
    for name in selected:
        cols.append(panel[name].to_numpy(dtype=float) * gate)
        names.append(f"global__{name}")
    for cls in class_names:
        cmask = panel["reach_class"].eq(cls).astype(float).to_numpy()
        for name in selected:
            cols.append(panel[name].to_numpy(dtype=float) * gate * cmask)
            names.append(f"{cls}__{name}")
    return np.vstack(cols).T, names


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(np.isfinite(std) & (std > 1.0e-8), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = standardize_fit(x)
    xs = standardize_apply(x, mean, std)
    xtx = xs.T @ xs
    beta = np.linalg.solve(xtx + lam * np.eye(xs.shape[1]), xs.T @ y)
    return beta, mean, std


def aggregate_upstream(
    pred: pd.DataFrame,
    network: pd.DataFrame,
    qcorr_col: str,
) -> pd.DataFrame:
    network_use = network[network["upstream_type"].eq("immediate_upstream_simulated_station")].copy()
    upstream_map: dict[str, list[str]] = defaultdict(list)
    for row in network_use.itertuples(index=False):
        upstream_map[str(row.target_site)].append(str(row.upstream_site))
    lookup = {
        (str(r.q_site), int(r.year), int(r.month)): (
            float(r.Q_pred_cfs),
            float(r.Q72_pred_cfs),
            float(r.Q78_mass_cfs),
            float(getattr(r, qcorr_col)),
        )
        for r in pred.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for r in pred.itertuples(index=False):
        ups = upstream_map.get(str(r.q_site), [])
        up_qmain = 0.0
        up_q72 = 0.0
        up_q78 = 0.0
        up_qcorr = 0.0
        available = 0
        for usite in ups:
            vals = lookup.get((usite, int(r.year), int(r.month)))
            if vals is None:
                continue
            available += 1
            up_qmain += vals[0]
            up_q72 += vals[1]
            up_q78 += vals[2]
            up_qcorr += vals[3]
        rows.append(
            {
                "q_site": str(r.q_site),
                "year": int(r.year),
                "month": int(r.month),
                "upstream_station_count": int(len(ups)),
                "upstream_available_count": int(available),
                "upstream_qmain_sum_cfs": up_qmain,
                "upstream_q72_sum_cfs": up_q72,
                "upstream_q78_sum_cfs": up_q78,
                "upstream_qcorr_sum_cfs": up_qcorr,
                "has_upstream_sim": int(available > 0),
            }
        )
    return pd.DataFrame(rows)


def build_feature_panel(pred: pd.DataFrame, network: pd.DataFrame, qcorr_col: str) -> pd.DataFrame:
    agg = aggregate_upstream(pred, network, qcorr_col)
    keep = [
        "q_site",
        "reach_id",
        "reach_class",
        "year",
        "month",
        "split",
        "Q_obsv_cfs",
        "Q_pred_cfs",
        "Q72_pred_cfs",
        "Q78_mass_cfs",
        qcorr_col,
        "layer_id",
    ]
    panel = pred[keep].merge(agg, on=["q_site", "year", "month"], how="left")
    panel = panel.rename(
        columns={
            "Q_pred_cfs": "Qmain_cfs",
            "Q72_pred_cfs": "Q72_cfs",
            "Q78_mass_cfs": "Q78_mass_cfs",
            "upstream_qcorr_sum_cfs": "upstream_qcorr_sum_cfs",
        }
    )
    panel["log_obs"] = np.log(panel["Q_obsv_cfs"].clip(lower=EPS))
    panel["log_qmain"] = np.log(panel["Qmain_cfs"].clip(lower=EPS))
    panel["baseline_residual_log"] = panel["log_obs"] - panel["log_qmain"]
    panel = panel.sort_values(["q_site", "year", "month"]).reset_index(drop=True)
    return add_upstream_features(panel, "upstream_qcorr_sum_cfs")


def apply_cascade(
    pred_input: pd.DataFrame,
    network: pd.DataFrame,
    class_names: list[str],
    feature_mode: str,
    beta: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip_value: float,
) -> pd.DataFrame:
    out = pred_input.copy()
    out["Q_layered_transfer_cfs"] = out["Q_pred_cfs"].astype(float)
    out["layered_transfer_correction_log_raw"] = 0.0
    out["layered_transfer_correction_log"] = 0.0
    for layer in sorted(out["layer_id"].unique()):
        if int(layer) == 0:
            continue
        feature_panel = build_feature_panel(
            out.rename(columns={"Q_layered_transfer_cfs": "current_qcorr_cfs"}),
            network,
            "current_qcorr_cfs",
        )
        layer_mask = feature_panel["layer_id"].eq(layer)
        if not layer_mask.any():
            continue
        x, _ = design_matrix(feature_panel.loc[layer_mask].copy(), feature_mode, class_names)
        corr_raw = standardize_apply(x, mean, std) @ beta
        corr = np.clip(corr_raw, -clip_value, clip_value)
        keys = feature_panel.loc[layer_mask, ["q_site", "year", "month"]].copy()
        keys["corr_raw"] = corr_raw
        keys["corr"] = corr
        out = out.merge(keys, on=["q_site", "year", "month"], how="left")
        use = out["corr"].notna()
        out.loc[use, "layered_transfer_correction_log_raw"] = out.loc[use, "corr_raw"].astype(float)
        out.loc[use, "layered_transfer_correction_log"] = out.loc[use, "corr"].astype(float)
        out.loc[use, "Q_layered_transfer_cfs"] = np.exp(
            np.log(out.loc[use, "Q_pred_cfs"].clip(lower=EPS)) + out.loc[use, "corr"].astype(float)
        )
        out = out.drop(columns=["corr_raw", "corr"])
    return out


def evaluate_panel(pred: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "inner":
        frame = pred[pred["year"].between(2016, 2018)].copy()
    elif period == "strict":
        frame = pred[pred["year"].between(2019, 2022)].copy()
    else:
        frame = pred[pred["year"].between(2006, 2022)].copy()
    return pd.concat(
        [
            metric_rows(frame, "Q_pred_cfs", "Q_main", period),
            metric_rows(frame, "Q_layered_transfer_cfs", "Q_layered_transfer", period),
        ],
        ignore_index=True,
    )


def select_model(pred: pd.DataFrame, network: pd.DataFrame, class_names: list[str]) -> tuple[dict[str, object], pd.DataFrame]:
    modes = ["cascade_core", "cascade_core_lag", "cascade_with_baseline_gap"]
    lambdas = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    clip_values = [0.35, 0.50, 0.70]
    pred0 = pred.copy()
    pred0["qmain_seed_cfs"] = pred0["Q_pred_cfs"].astype(float)
    base_panel = build_feature_panel(pred0, network, "qmain_seed_cfs")
    train_mask = base_panel["year"].between(2006, 2015) & base_panel["has_upstream_sim"].eq(1)
    y_train = base_panel.loc[train_mask, "baseline_residual_log"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    candidates: list[tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]] = []
    base_inner = evaluate_panel(pred.assign(Q_layered_transfer_cfs=pred["Q_pred_cfs"]), "inner")
    base_good_all = int(base_inner[(base_inner["model"].eq("Q_main"))]["good"].sum())
    base_up = base_inner[(base_inner["model"].eq("Q_main")) & (base_inner["has_upstream_sim"].eq(1))]
    base_up_nse = float(base_up["NSElog"].median())
    for mode in modes:
        x_train, names = design_matrix(base_panel.loc[train_mask].copy(), mode, class_names)
        for lam in lambdas:
            beta, mean, std = fit_ridge(x_train, y_train, lam)
            for clip in clip_values:
                pred_inner = apply_cascade(pred, network, class_names, mode, beta, mean, std, clip)
                metrics = evaluate_panel(pred_inner, "inner")
                new = metrics[metrics["model"].eq("Q_layered_transfer")].copy()
                new_up = new[new["has_upstream_sim"].eq(1)].copy()
                row = {
                    "feature_mode": mode,
                    "lambda": lam,
                    "clip_log": clip,
                    "feature_count": len(names),
                    "inner_good_all": int(new["good"].sum()),
                    "inner_good_upstream_connected": int(new_up["good"].sum()),
                    "inner_median_NSElog_all": float(new["NSElog"].median()),
                    "inner_median_NSElog_upstream_connected": float(new_up["NSElog"].median()),
                    "delta_inner_median_NSElog_upstream_connected": float(new_up["NSElog"].median() - base_up_nse),
                    "inner_median_absPBIAS_all": float(new["abs_PBIAS_pct"].median()),
                }
                rows.append(row)
                candidates.append((row, beta, mean, std))
    selection = pd.DataFrame(rows)
    eligible = selection[selection["inner_good_all"].ge(base_good_all - 1)].copy()
    if eligible.empty:
        eligible = selection.copy()
    eligible = eligible.sort_values(
        [
            "inner_good_all",
            "inner_median_NSElog_upstream_connected",
            "inner_median_NSElog_all",
            "inner_median_absPBIAS_all",
        ],
        ascending=[False, False, False, True],
    )
    best_row = eligible.iloc[0].to_dict()
    idx = selection[
        selection["feature_mode"].eq(best_row["feature_mode"])
        & selection["lambda"].eq(best_row["lambda"])
        & selection["clip_log"].eq(best_row["clip_log"])
    ].index[0]
    _, beta, mean, std = candidates[int(idx)]
    best = {
        **best_row,
        "beta": beta,
        "mean": mean,
        "std": std,
    }
    return best, selection


def fit_final(pred: pd.DataFrame, network: pd.DataFrame, class_names: list[str], best: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    seed = pred.copy()
    seed["qmain_seed_cfs"] = seed["Q_pred_cfs"].astype(float)
    panel = build_feature_panel(seed, network, "qmain_seed_cfs")
    train_mask = panel["year"].between(2006, 2018) & panel["has_upstream_sim"].eq(1)
    x, names = design_matrix(panel.loc[train_mask].copy(), str(best["feature_mode"]), class_names)
    y = panel.loc[train_mask, "baseline_residual_log"].to_numpy(dtype=float)
    beta, mean, std = fit_ridge(x, y, float(best["lambda"]))
    return beta, mean, std, names


def add_summary_groups(metrics: pd.DataFrame) -> pd.DataFrame:
    all_part = metrics.copy()
    all_part["summary_group"] = "all_stations"
    up_part = metrics[metrics["has_upstream_sim"].eq(1)].copy()
    up_part["summary_group"] = "upstream_connected_stations"
    return pd.concat([all_part, up_part], ignore_index=True)


def compare_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["model"].eq("Q_main")].copy()
    new = metrics[metrics["model"].eq("Q_layered_transfer")].copy()
    comp = base.merge(
        new,
        on=["period", "q_site", "reach_id", "reach_class", "layer_id", "has_upstream_sim", "upstream_station_count"],
        suffixes=("_baseline", "_layered"),
    )
    comp["delta_NSElog"] = comp["NSElog_layered"] - comp["NSElog_baseline"]
    comp["delta_KGE"] = comp["KGE_2012_layered"] - comp["KGE_2012_baseline"]
    comp["delta_absPBIAS"] = comp["abs_PBIAS_pct_layered"] - comp["abs_PBIAS_pct_baseline"]
    return comp.sort_values(["period", "has_upstream_sim", "delta_NSElog"], ascending=[True, False, False])


def write_audits(network: pd.DataFrame, inventory: pd.DataFrame, selection: pd.DataFrame) -> None:
    network.to_csv(REPORT / "upstream_station_network.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(REPORT / "station_layer_inventory.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(REPORT / "model_selection_inner_validation.csv", index=False, encoding="utf-8-sig")
    duplication = pd.DataFrame(
        [
            {
                "feature_name": "upstream_qcorr_sum_cfs",
                "feature_type": "predicted_upstream_flow_after_clean_model_correction",
                "uses_observed_flow": "no",
                "uses_validation_observed_flow": "no",
                "already_in_Q72": "no_directly",
                "already_in_Q78": "no_directly",
                "interpreted_as_water_volume": "yes_predicted_flow_only",
                "allowed_in_clean_model": "yes",
                "risk_level": "medium",
                "note": "This is predicted flow generated by the model cascade, not observed residual.",
            },
            {
                "feature_name": "log_up_qcorr_to_up_qmain",
                "feature_type": "predicted_correction_signal",
                "uses_observed_flow": "no",
                "uses_validation_observed_flow": "no",
                "already_in_Q72": "no",
                "already_in_Q78": "no",
                "interpreted_as_water_volume": "no",
                "allowed_in_clean_model": "yes",
                "risk_level": "low_to_medium",
                "note": "Represents upstream predicted correction strength, not an added hydrologic process.",
            },
            {
                "feature_name": "log_up_q78_to_up_q72",
                "feature_type": "model_branch_gap",
                "uses_observed_flow": "no",
                "uses_validation_observed_flow": "no",
                "already_in_Q72": "branch_output",
                "already_in_Q78": "branch_output",
                "interpreted_as_water_volume": "no",
                "allowed_in_clean_model": "yes",
                "risk_level": "low",
                "note": "Uses disagreement between existing model branches; it does not add PPT/AET/groundwater variables again.",
            },
            {
                "feature_name": "raw_upstream_residual",
                "feature_type": "observed_error",
                "uses_observed_flow": "yes",
                "uses_validation_observed_flow": "would_if_used",
                "already_in_Q72": "not_applicable",
                "already_in_Q78": "not_applicable",
                "interpreted_as_water_volume": "ambiguous",
                "allowed_in_clean_model": "no",
                "risk_level": "forbidden",
                "note": "Not used. Raw residual mixes local runoff, groundwater, reservoir, topology and observation errors.",
            },
            {
                "feature_name": "PPT/AET/PET/dry_stress/SAS/TTD",
                "feature_type": "hydrologic_process_features",
                "uses_observed_flow": "no",
                "uses_validation_observed_flow": "no",
                "already_in_Q72": "yes",
                "already_in_Q78": "partly_yes",
                "interpreted_as_water_volume": "process_state",
                "allowed_in_clean_model": "no_for_this_transfer_layer",
                "risk_level": "high_duplication_risk",
                "note": "Not used in this transfer layer to avoid double-counting process information.",
            },
        ]
    )
    duplication.to_csv(REPORT / "duplication_audit.csv", index=False, encoding="utf-8-sig")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in show.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def make_figures(final_pred: pd.DataFrame, comp: pd.DataFrame, summary: pd.DataFrame) -> None:
    setup_plot()
    strict_comp = comp[comp["period"].eq("strict")].copy()
    up = strict_comp[strict_comp["has_upstream_sim"].eq(1)].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(strict_comp["NSElog_baseline"], strict_comp["NSElog_layered"], s=38, c=strict_comp["layer_id"], cmap="viridis", edgecolor="white", linewidth=0.4)
    lo = np.nanmin([strict_comp["NSElog_baseline"].min(), strict_comp["NSElog_layered"].min(), -1.0])
    hi = np.nanmax([strict_comp["NSElog_baseline"].max(), strict_comp["NSElog_layered"].max(), 1.0])
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1)
    ax.set_xlabel("Q_main NSElog")
    ax.set_ylabel("Layered transfer NSElog")
    ax.set_title("Strict Validation: Baseline vs Layered Transfer")
    fig.tight_layout()
    fig.savefig(FIG / "strict_nselog_scatter_all.png", dpi=240)
    fig.savefig(FIG / "strict_nselog_scatter_all.pdf")
    plt.close(fig)

    if not up.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(up["NSElog_baseline"], up["NSElog_layered"], s=48, c=up["upstream_station_count"], cmap="plasma", edgecolor="white", linewidth=0.4)
        lo = np.nanmin([up["NSElog_baseline"].min(), up["NSElog_layered"].min(), -1.0])
        hi = np.nanmax([up["NSElog_baseline"].max(), up["NSElog_layered"].max(), 1.0])
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=1)
        ax.set_xlabel("Q_main NSElog")
        ax.set_ylabel("Layered transfer NSElog")
        ax.set_title("Upstream-Connected Stations")
        fig.tight_layout()
        fig.savefig(FIG / "strict_nselog_scatter_upstream_connected.png", dpi=240)
        fig.savefig(FIG / "strict_nselog_scatter_upstream_connected.pdf")
        plt.close(fig)

    strict_summary = summary[summary["period"].eq("strict")].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = strict_summary["model"] + "\n" + strict_summary["group"]
    ax.bar(labels, strict_summary["good_count"], color=["#4C78A8", "#F58518", "#72B7B2", "#54A24B"][: len(strict_summary)])
    ax.set_ylabel("Good stations")
    ax.set_title("Strict Validation Good Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "strict_good_count_summary.png", dpi=240)
    fig.savefig(FIG / "strict_good_count_summary.pdf")
    plt.close(fig)

    final_pred["date"] = pd.to_datetime({"year": final_pred["year"].astype(int), "month": final_pred["month"].astype(int), "day": 1})
    for site in KEY_STATIONS:
        sub = final_pred[final_pred["q_site"].eq(site)].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(sub["date"], sub["Q_obsv_cfs"], color="black", lw=1.3, label="Observed")
        ax.plot(sub["date"], sub["Q_pred_cfs"], color="#4C78A8", lw=1.1, label="Q_main")
        ax.plot(sub["date"], sub["Q_layered_transfer_cfs"], color="#F58518", lw=1.1, label="Layered transfer")
        ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"), color="0.9", alpha=0.7, zorder=-1)
        ax.set_title(site)
        ax.set_ylabel("Q (cfs)")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"hydrograph_{site}.png", dpi=220)
        plt.close(fig)


def write_docs(best: dict[str, object], summary: pd.DataFrame, comp: pd.DataFrame, names: list[str]) -> None:
    strict_summary = summary[summary["period"].eq("strict")].copy()
    key = comp[(comp["period"].eq("strict")) & (comp["q_site"].isin(KEY_STATIONS))].copy()
    key = key[
        [
            "q_site",
            "reach_id",
            "layer_id",
            "upstream_station_count",
            "NSElog_baseline",
            "NSElog_layered",
            "delta_NSElog",
            "KGE_2012_baseline",
            "KGE_2012_layered",
            "PBIAS_pct_baseline",
            "PBIAS_pct_layered",
        ]
    ]
    improved = int((comp[(comp["period"].eq("strict"))]["delta_NSElog"] > 0).sum())
    worsened = int((comp[(comp["period"].eq("strict"))]["delta_NSElog"] < 0).sum())
    method = f"""# {RUN.name} Layered Upstream Information Transfer

Generated: {datetime.now().isoformat(timespec='seconds')}

## Purpose

This experiment tests whether cleaner upstream predictions can be propagated downstream through the station topology.

It is a clean predictive experiment. Strict validation never uses upstream observed flow, target observed flow, or observed upstream residuals as simulation inputs.

## Model

Baseline:

```text
Q_main = Q72 high-skill hydrologic feature model + Q78_mass light mass-conserving pull
```

Layered transfer:

```text
log(Q_layered_i,t)
= log(Q_main_i,t)
  + clip(X_i,t beta, -c, c)
```

where `X_i,t` is built from predicted upstream flow only:

```text
log(sum Qcorr_up / Qmain_target)
log(sum Qcorr_up / sum Qmain_up)
log(sum Q78_up / sum Q72_up)
optional lag-1 terms and reach_class interactions
```

The cascade is evaluated from upstream to downstream:

```text
layer 0: Qcorr = Qmain
layer k: use already predicted upstream Qcorr from lower layers
```

## Residual Boundary

This experiment does not transfer raw residuals:

```text
r_up,t = log(Qobs_up,t) - log(Qmain_up,t)
```

Raw residuals are forbidden because they mix local runoff, groundwater, reservoir operations, topology mismatch and observation error. They would also duplicate hydrologic process variables that are already inside Q72/Q78.

The only transferred signal is predicted upstream correction, derived without validation observed flow.

## Selected Inner-Validation Configuration

- feature mode: `{best['feature_mode']}`
- lambda: `{best['lambda']}`
- correction clip: `{best['clip_log']}`
- final feature count: `{len(names)}`

## Strict 2019-2022 Summary

{markdown_table(strict_summary)}

## Strict Key Stations

{markdown_table(key)}

## Strict Improvement Counts

- improved stations by NSElog: {improved}
- worsened stations by NSElog: {worsened}

## Outputs

- `station_layer_inventory.csv`
- `upstream_station_network.csv`
- `duplication_audit.csv`
- `model_selection_inner_validation.csv`
- `layered_transfer_predictions_long.csv`
- `station_metrics_comparison.csv`
- `experiment_summary.csv`
- hydrographs and scatter plots under `figure/layered_upstream_transfer`
"""
    (REPORT / "method.md").write_text(method, encoding="utf-8")
    (LOGS / "run_log.md").write_text(method, encoding="utf-8")
    readme = f"""# {RUN.name}

Clean layered upstream information transfer experiment.

See `reports/layered_upstream_transfer/method.md`.
"""
    (RUN / "README.md").write_text(readme, encoding="utf-8")
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write("\n\n" + method + "\n")


def main() -> None:
    ensure_dirs()
    pred = pd.read_csv(BASE_PRED, encoding="utf-8-sig")
    pred = pred[pred["year"].between(2006, 2022)].copy()
    pred["split2"] = pred["year"].map(split_name)
    pred = pred[pred["split2"].isin(["train", "inner", "strict"])].copy()
    pred["split"] = pred["split2"]
    upstream = load_topology()
    network = build_station_network(pred, upstream)
    inventory = assign_layers(network, pred)
    pred = pred.merge(inventory[["q_site", "layer_id", "upstream_station_count"]], on="q_site", how="left")
    pred["has_upstream_sim"] = pred["upstream_station_count"].fillna(0).astype(int).gt(0).astype(int)
    class_names = sorted(pred["reach_class"].astype(str).unique())

    best, selection = select_model(pred, network, class_names)
    beta, mean, std, names = fit_final(pred, network, class_names, best)
    final_pred = apply_cascade(pred, network, class_names, str(best["feature_mode"]), beta, mean, std, float(best["clip_log"]))
    final_pred.to_csv(REPORT / "layered_transfer_predictions_long.csv", index=False, encoding="utf-8-sig")

    inner_metrics = evaluate_panel(final_pred, "inner")
    strict_metrics = evaluate_panel(final_pred, "strict")
    metrics = pd.concat([inner_metrics, strict_metrics], ignore_index=True)
    comp = compare_metrics(metrics)
    summary = summarize_metrics(add_summary_groups(metrics))
    metrics.to_csv(REPORT / "station_metrics_all_models.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(REPORT / "station_metrics_comparison.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORT / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    write_audits(network, inventory, selection)
    make_figures(final_pred, comp, summary)
    write_docs(best, summary, comp, names)
    result = {
        "run": RUN.name,
        "selected": {k: v for k, v in best.items() if k not in {"beta", "mean", "std"}},
        "strict_summary": summary[summary["period"].eq("strict")].to_dict(orient="records"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
