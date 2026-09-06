from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


assert_sparrow_runtime()
ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT.parent / "20260814_1"
KEY = ["comid", "q_site", "year", "month", "fold_id"]
FOLDS = [
    ("fit_2006_2011_eval_2012_2013", 2011),
    ("fit_2006_2013_eval_2014_2015", 2013),
    ("fit_2006_2015_eval_2016_2018", 2015),
]
EPS = 1e-9
NBOOT = 10000
SEED = 20260814


def metric(obs: pd.Series, pred: pd.Series) -> dict[str, float]:
    o = np.asarray(obs, float)
    p = np.asarray(pred, float)
    good = np.isfinite(o) & np.isfinite(p) & (o > 0) & (p > 0)
    o, p = o[good], p[good]
    lo, lp = np.log(o), np.log(p)
    return {
        "n": int(len(o)),
        "raw_nse": float(1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)),
        "log_nse": float(1 - np.sum((lp - lo) ** 2) / np.sum((lo - lo.mean()) ** 2)),
        "pbias_pct": float(100 * np.sum(p - o) / np.sum(o)),
        "log_rmse": float(np.sqrt(np.mean((lp - lo) ** 2))),
    }


def terminal_map() -> dict[int, int]:
    topo = pd.read_csv(STAGE1 / "inputs" / "topology" / "topology_edges.csv")
    down = {
        int(r.reach_id): (
            None
            if pd.isna(r.downstream_reach) or str(r.downstream_reach).strip() == ""
            else int(float(str(r.downstream_reach).split(",")[0]))
        )
        for r in topo.itertuples(index=False)
    }
    result: dict[int, int] = {}
    for rid in down:
        seen: set[int] = set()
        cur = rid
        while down.get(cur) is not None and cur not in seen:
            seen.add(cur)
            cur = int(down[cur])
        result[rid] = cur
    return result


def build_lowflow(h0: pd.DataFrame) -> pd.DataFrame:
    dev = pd.read_parquet(
        STAGE1 / "inputs" / "development_indata_2006_2018.parquet",
        columns=["comid", "q_site", "year", "month", "Q_obsv_cfs"],
    )
    dev = dev[dev.Q_obsv_cfs.notna() & dev.Q_obsv_cfs.gt(0)].copy()
    dev.q_site = dev.q_site.astype(str)
    rows = []
    for fold, train_end in FOLDS:
        q20 = (
            dev[dev.year <= train_end]
            .groupby("q_site")
            .Q_obsv_cfs.quantile(0.20)
            .rename("train_q20_cfs")
        )
        part = h0[h0.fold_id.eq(fold)].merge(q20, on="q_site", validate="many_to_one")
        rows.append(part[part.actual <= part.train_q20_cfs][KEY + ["actual", "train_q20_cfs"]].copy())
    return pd.concat(rows, ignore_index=True)


def attach(model: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    return registry.merge(model[KEY + ["predict"]], on=KEY, validate="one_to_one")


def cluster_mse(frame: pd.DataFrame, obs_col: str = "actual") -> pd.Series:
    x = frame.copy()
    x["sq"] = (
        np.log(x.predict.clip(lower=EPS)) - np.log(x[obs_col].clip(lower=EPS))
    ) ** 2
    return x.groupby("q_site").sq.mean()


def paired_ci(cand: pd.Series, base: pd.Series) -> dict[str, float | int | bool]:
    common = cand.index.intersection(base.index)
    c = cand.loc[common].to_numpy(float)
    b = base.loc[common].to_numpy(float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(common), size=(NBOOT, len(common)))
    diff = np.sqrt(c[idx].mean(axis=1)) - np.sqrt(b[idx].mean(axis=1))
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return {
        "clusters": int(len(common)),
        "point_difference": float(np.sqrt(c.mean()) - np.sqrt(b.mean())),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "stable_improvement": bool(hi < 0),
    }


def event_metrics(
    model: pd.DataFrame, tail_months: pd.DataFrame, events: pd.DataFrame
) -> tuple[float, float]:
    tail = tail_months.merge(model[KEY + ["predict"]], on=KEY, validate="one_to_one")
    peaks = events.rename(
        columns={
            "peak_year": "year",
            "peak_month": "month",
            "peak_observed_cfs": "peak_observed",
        }
    )
    peaks = peaks.merge(
        model[KEY + ["predict"]].rename(columns={"predict": "peak_predict"}),
        on=KEY,
        validate="one_to_one",
    )
    tail = tail.merge(
        peaks[["event_id", "peak_observed", "peak_predict"]],
        on="event_id",
        validate="many_to_one",
    )
    tail["shape_sq"] = (
        np.log((tail.predict + EPS) / (tail.peak_predict + EPS))
        - np.log((tail.observed_cfs + EPS) / (tail.peak_observed + EPS))
    ) ** 2
    volume = tail.groupby(["event_id", "q_site"], as_index=False).agg(
        obs=("observed_cfs", "sum"), pred=("predict", "sum")
    )
    volume["abs_log_error"] = np.log((volume.pred + EPS) / (volume.obs + EPS)).abs()
    return float(np.sqrt(tail.shape_sq.mean())), float(volume.abs_log_error.mean())


def spatial_difference(
    cand: pd.Series, base: pd.Series, station_to_terminal: dict[str, int]
) -> float:
    common = cand.index.intersection(base.index)
    table = pd.DataFrame({"cand": cand.loc[common], "base": base.loc[common]})
    table["terminal"] = [station_to_terminal.get(str(s), -1) for s in table.index]
    block = table.groupby("terminal")[["cand", "base"]].mean()
    return float(np.sqrt(block.cand.mean()) - np.sqrt(block.base.mean()))


def main() -> None:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    h0 = pd.read_parquet(STAGE1 / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet")
    cand = pd.read_parquet(ROOT / "outputs" / "conditional_grid_oof.parquet")
    for x in (h0, cand):
        x.q_site = x.q_site.astype(str)
    if len(cand) != 7755 or cand[KEY].duplicated().any():
        raise RuntimeError("conditional OOF key gate failed")
    if set(map(tuple, cand[KEY].to_numpy())) != set(map(tuple, h0[KEY].to_numpy())):
        raise RuntimeError("conditional OOF keys do not match H0")

    lowreg = build_lowflow(h0)
    tails = pd.read_parquet(STAGE1 / "inputs" / "registries" / "tail_month_registry.parquet")
    events = pd.read_parquet(STAGE1 / "inputs" / "registries" / "tail_event_registry.parquet")
    tails.q_site = tails.q_site.astype(str)
    events.q_site = events.q_site.astype(str)
    tmap = terminal_map()
    station_comid = h0.groupby("q_site").comid.first().astype(int)
    station_terminal = {s: tmap.get(int(r), int(r)) for s, r in station_comid.items()}

    models = {"H0_hybrid": h0, "conditional_grid": cand}
    summary_rows, fold_rows, losses = [], [], {}
    for name, model in models.items():
        overall = metric(model.actual, model.predict)
        low = attach(model, lowreg)
        tail = attach(
            model,
            tails.rename(columns={"observed_cfs": "actual"})[
                KEY + ["event_id", "tail_lag", "actual"]
            ],
        )
        low_mse, tail_mse = cluster_mse(low), cluster_mse(tail)
        losses[name] = {"low": low_mse, "tail": tail_mse}
        shape, volume = event_metrics(model, tails, events)
        summary_rows.append(
            {
                "model_id": name,
                **overall,
                "lowflow_log_rmse": float(np.sqrt(low_mse.mean())),
                "tail_log_rmse": float(np.sqrt(tail_mse.mean())),
                "tail_shape_rmse": shape,
                "event_volume_abs_log_error": volume,
            }
        )
        for fold, _ in FOLDS:
            mf = model[model.fold_id.eq(fold)]
            lf = low[low.fold_id.eq(fold)]
            tf = tail[tail.fold_id.eq(fold)]
            fold_rows.append(
                {
                    "model_id": name,
                    "fold_id": fold,
                    **metric(mf.actual, mf.predict),
                    "lowflow_log_rmse": metric(lf.actual, lf.predict)["log_rmse"],
                    "tail_log_rmse": metric(tf.actual, tf.predict)["log_rmse"],
                }
            )

    summary = pd.DataFrame(summary_rows).set_index("model_id")
    fold = pd.DataFrame(fold_rows)
    base, row = summary.loc["H0_hybrid"], summary.loc["conditional_grid"]
    sf = fold[fold.model_id.eq("conditional_grid")].set_index("fold_id")
    bf = fold[fold.model_id.eq("H0_hybrid")].set_index("fold_id")
    low_folds = int((sf.lowflow_log_rmse < bf.lowflow_log_rmse).sum())
    tail_folds = int((sf.tail_log_rmse < bf.tail_log_rmse).sum())
    low_ci = paired_ci(losses["conditional_grid"]["low"], losses["H0_hybrid"]["low"])
    tail_ci = paired_ci(losses["conditional_grid"]["tail"], losses["H0_hybrid"]["tail"])
    low_sp = spatial_difference(
        losses["conditional_grid"]["low"], losses["H0_hybrid"]["low"], station_terminal
    )
    tail_sp = spatial_difference(
        losses["conditional_grid"]["tail"], losses["H0_hybrid"]["tail"], station_terminal
    )
    low_ci["terminal_point_difference"] = low_sp
    tail_ci["terminal_point_difference"] = tail_sp
    low_ci["spatial_direction_consistent"] = bool(
        np.sign(low_ci["point_difference"]) == np.sign(low_sp) or abs(low_sp) < 1e-12
    )
    tail_ci["spatial_direction_consistent"] = bool(
        np.sign(tail_ci["point_difference"]) == np.sign(tail_sp) or abs(tail_sp) < 1e-12
    )

    checks = {
        "raw_nse_noninferior": bool(row.raw_nse - base.raw_nse >= -0.005),
        "log_nse_noninferior": bool(row.log_nse - base.log_nse >= -0.005),
        "pbias_guard": bool(abs(row.pbias_pct) <= abs(base.pbias_pct) + 3.0),
        "lowflow_improves_2_of_3": bool(low_folds >= 2),
        "tail_improves_2_of_3": bool(tail_folds >= 2),
        "lowflow_station_ci": bool(low_ci["stable_improvement"]),
        "tail_station_ci": bool(tail_ci["stable_improvement"]),
        "lowflow_terminal_direction": bool(low_ci["spatial_direction_consistent"]),
        "tail_terminal_direction": bool(tail_ci["spatial_direction_consistent"]),
        "tail_shape_nonworsening": bool(row.tail_shape_rmse <= base.tail_shape_rmse + 1e-12),
        "event_volume_nonworsening": bool(
            row.event_volume_abs_log_error <= base.event_volume_abs_log_error + 1e-12
        ),
    }
    qualified = bool(all(checks.values()))
    decision = {
        "performance_type": "conditional_parameter_oof_after_structure_selection",
        "development_qualified": qualified,
        "next_action": "run_groundwater_then_sas_ablation" if qualified else "stop_search_retain_H0",
        "checks": checks,
        "deltas": {
            "raw_nse": float(row.raw_nse - base.raw_nse),
            "log_nse": float(row.log_nse - base.log_nse),
            "abs_pbias_percentage_points": float(abs(row.pbias_pct) - abs(base.pbias_pct)),
            "lowflow_log_rmse": float(row.lowflow_log_rmse - base.lowflow_log_rmse),
            "tail_log_rmse": float(row.tail_log_rmse - base.tail_log_rmse),
            "tail_shape_rmse": float(row.tail_shape_rmse - base.tail_shape_rmse),
            "event_volume_abs_log_error": float(
                row.event_volume_abs_log_error - base.event_volume_abs_log_error
            ),
        },
        "improved_folds": {"lowflow": low_folds, "tail": tail_folds},
        "paired_bootstrap": {"lowflow": low_ci, "tail": tail_ci},
        "pass": True,
    }
    summary.reset_index().to_csv(reports / "conditional_metric_summary.csv", index=False, encoding="utf-8-sig")
    fold.to_csv(reports / "conditional_fold_metrics.csv", index=False, encoding="utf-8-sig")
    (reports / "conditional_flow_gate.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
