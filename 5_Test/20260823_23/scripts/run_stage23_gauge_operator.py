from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, eye, vstack
from scipy.sparse.linalg import lsqr


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_23"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
sys.path.insert(0, str(TEST / "20260823_17" / "scripts"))
from regionalization import ATTRIBUTES, base_attribute_matrix, station_metrics, summary_metrics  # noqa: E402


OBS = TEST / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
COVERAGE = TEST / "20260823_14" / "outputs" / "final_user_locked_station_coverage.parquet"
FLOW = TEST / "20260823_21" / "outputs" / "monthly_network_native_hydrology.parquet"
PARENT_STATE = TEST / "20260823_15" / "final_outputs" / "model_and_assimilation_state_audit.parquet"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
SMOOTH_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
RIDGE_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assemble_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = pd.read_parquet(OBS)
    obs["q_site"] = obs.station_norm.astype(str)
    flow = pd.read_parquet(FLOW)
    flow["quick_fraction"] = flow.routed_network_native_quick_cfs / flow.routed_network_native_total_cfs.clip(lower=EPS)
    cols = ["comid", "year", "month", "routed_network_native_total_cfs", "quick_fraction"]
    panel = obs.merge(flow[cols].rename(columns={"comid": "reach_id"}), on=["reach_id", "year", "month"], validate="many_to_one")
    coverage = pd.read_parquet(COVERAGE)
    coverage["q_site"] = coverage.station_norm.astype(str)
    coverage = coverage[coverage.selected_for_model].copy()
    # The observation registry already carries snap_distance_m; avoid suffixing it during the support merge.
    keep = ["q_site", "reach_id", "downstream_fraction_on_reach", "best_line_distance_m", "line_catchment_override"]
    panel = panel.merge(coverage[keep], on=["q_site", "reach_id"], validate="many_to_one")
    attrs = pd.read_parquet(ATTRIBUTES).sort_values("comid").reset_index(drop=True)
    return panel, attrs, coverage


def support_attribute_matrix(
    attrs: pd.DataFrame, support: pd.DataFrame, fit_sites: list[str], scaling: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str], tuple[np.ndarray, np.ndarray]]:
    reach_attr, reach_names = base_attribute_matrix(attrs)
    reach_lookup = {int(r): i for i, r in enumerate(attrs.comid)}
    base = reach_attr[np.array([reach_lookup[int(r)] for r in support.reach_id], int)]
    fraction = support.downstream_fraction_on_reach.fillna(0.999).clip(0.001, 0.999).to_numpy(float)
    extra = np.column_stack([
        np.log(fraction / (1.0 - fraction)),
        np.log1p(support.snap_distance_m.fillna(0).clip(lower=0).to_numpy(float)),
        np.log1p(support.best_line_distance_m.fillna(0).clip(lower=0).to_numpy(float)),
        support.line_catchment_override.fillna(False).astype(float).to_numpy(),
    ])
    names = reach_names + ["logit_downstream_fraction", "log_snap_distance", "log_best_line_distance", "line_catchment_override"]
    raw = np.column_stack([base, extra])
    if scaling is None:
        mean, std = raw.mean(axis=0), raw.std(axis=0)
        std = np.where(std > EPS, std, 1.0)
    else:
        mean, std = scaling
    return (raw - mean) / std, names, (mean, std)


def dynamic_matrix(frame: pd.DataFrame, scaling: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    raw = np.column_stack([
        np.log1p(frame.routed_network_native_total_cfs.clip(lower=0).to_numpy(float)),
        frame.quick_fraction.fillna(0).to_numpy(float),
        np.sin(2 * np.pi * frame.month.to_numpy(float) / 12),
        np.cos(2 * np.pi * frame.month.to_numpy(float) / 12),
    ])
    if scaling is None:
        mean, std = raw.mean(axis=0), raw.std(axis=0)
        std = np.where(std > EPS, std, 1.0)
    else:
        mean, std = scaling
    z = (raw - mean) / std
    return np.column_stack([np.ones(len(frame)), z]), (mean, std)


def graph_edges(attrs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    topo = pd.read_csv(TOPOLOGY)
    index = {int(r): i for i, r in enumerate(attrs.comid)}
    ei, ej = [], []
    for row in topo.itertuples():
        if pd.isna(row.downstream_reach):
            continue
        ei.append(index[int(row.reach_id)])
        ej.append(index[int(row.downstream_reach)])
    return np.array(ei, int), np.array(ej, int)


class GaugeOperator:
    def __init__(self, attrs: pd.DataFrame, smooth: float, ridge: float, attribute_precision: float = 1.0):
        self.attrs = attrs
        self.smooth = float(smooth)
        self.ridge = float(ridge)
        self.attribute_precision = float(attribute_precision)
        self.nr = len(attrs)
        self.reach_index = {int(r): i for i, r in enumerate(attrs.comid)}
        self.ei, self.ej = graph_edges(attrs)
        self.dynamic_scaling = None
        self.attribute_scaling = None
        self.attribute_names = None
        self.theta = None

    def design(self, frame: pd.DataFrame, fit_scaling: bool) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
        support = frame[["q_site", "reach_id", "downstream_fraction_on_reach", "snap_distance_m", "best_line_distance_m", "line_catchment_override"]].drop_duplicates("q_site")
        support = support.set_index("q_site").loc[frame.q_site].reset_index()
        dyn, dscale = dynamic_matrix(frame, None if fit_scaling else self.dynamic_scaling)
        attr, names, ascale = support_attribute_matrix(
            self.attrs, support, list(frame.q_site.unique()), None if fit_scaling else self.attribute_scaling
        )
        if fit_scaling:
            self.dynamic_scaling, self.attribute_scaling, self.attribute_names = dscale, ascale, names
        na = attr.shape[1]
        n = len(frame)
        # Parameter order: 5 global effects; na*5 attribute effects; nr*5 structured node effects.
        rows, cols, data = [], [], []
        for k in range(5):
            rows.extend(np.arange(n))
            cols.extend(np.full(n, k))
            data.extend(dyn[:, k])
            block = dyn[:, k, None] * attr
            rr = np.repeat(np.arange(n), na)
            cc = 5 + k * na + np.tile(np.arange(na), n)
            rows.extend(rr)
            cols.extend(cc)
            data.extend(block.ravel())
            reach_pos = frame.reach_id.astype(int).map(self.reach_index).to_numpy(int)
            rows.extend(np.arange(n))
            cols.extend(5 + 5 * na + reach_pos * 5 + k)
            data.extend(dyn[:, k])
        x = coo_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(n, 5 + na * 5 + self.nr * 5)).tocsr()
        base = np.log1p(frame.routed_network_native_total_cfs.to_numpy(float))
        y = np.log1p(frame.Q_obsv_cfs.to_numpy(float)) - base
        return x, y, dyn

    def fit(self, frame: pd.DataFrame) -> dict:
        x, y, _ = self.design(frame, fit_scaling=True)
        na = len(self.attribute_names)
        p = x.shape[1]
        penalty_parts, target_parts = [x], [y]
        # Attribute Gaussian prior.
        rows = np.arange(na * 5)
        attr_pen = coo_matrix((np.full(na * 5, np.sqrt(self.attribute_precision)), (rows, 5 + rows)), shape=(na * 5, p)).tocsr()
        penalty_parts.append(attr_pen)
        target_parts.append(np.zeros(na * 5))
        # Reach ridge prior.
        rows = np.arange(self.nr * 5)
        reach_pen = coo_matrix((np.full(self.nr * 5, np.sqrt(self.ridge)), (rows, 5 + na * 5 + rows)), shape=(self.nr * 5, p)).tocsr()
        penalty_parts.append(reach_pen)
        target_parts.append(np.zeros(self.nr * 5))
        # River-network graph differences.
        n_edge_rows = len(self.ei) * 5
        rr, cc, dd = [], [], []
        for e, (i, j) in enumerate(zip(self.ei, self.ej)):
            for k in range(5):
                row = e * 5 + k
                rr.extend([row, row])
                cc.extend([5 + na * 5 + i * 5 + k, 5 + na * 5 + j * 5 + k])
                dd.extend([np.sqrt(self.smooth), -np.sqrt(self.smooth)])
        graph_pen = coo_matrix((dd, (rr, cc)), shape=(n_edge_rows, p)).tocsr()
        penalty_parts.append(graph_pen)
        target_parts.append(np.zeros(n_edge_rows))
        aug_x = vstack(penalty_parts, format="csr")
        aug_y = np.concatenate(target_parts)
        result = lsqr(aug_x, aug_y, atol=1e-10, btol=1e-10, iter_lim=10000, show=False)
        self.theta = result[0]
        return {
            "istop": int(result[1]), "iterations": int(result[2]), "residual_norm": float(result[3]),
            "normal_equation_residual": float(result[4]), "condition_estimate": float(result[6]),
            "coefficient_norm": float(result[8]),
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x, _, _ = self.design(frame, fit_scaling=False)
        delta = x @ self.theta
        return np.maximum(np.expm1(np.log1p(frame.routed_network_native_total_cfs.to_numpy(float)) + delta), 0.0)

    def parameter_fields(self, outlet_support: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        na = len(self.attribute_names)
        global_effect = self.theta[:5]
        gamma = self.theta[5 : 5 + na * 5].reshape(5, na).T
        u = self.theta[5 + na * 5 :].reshape(self.nr, 5)
        attr, _, _ = support_attribute_matrix(self.attrs, outlet_support, [], self.attribute_scaling)
        effects = global_effect + attr @ gamma + u
        names = ["intercept", "log_reach_flow", "quick_fraction", "month_sin", "month_cos"]
        reach = pd.DataFrame({"reach_id": self.attrs.comid.to_numpy(int)})
        for k, name in enumerate(names):
            reach[name] = effects[:, k]
            reach[f"structured_residual_{name}"] = u[:, k]
        gamma_frame = pd.DataFrame(gamma, index=self.attribute_names, columns=names).reset_index(names="attribute")
        return reach, gamma_frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_fitting":
        raise RuntimeError("Stage 23 contract not pre-registered")
    panel, attrs, coverage = assemble_panel()
    train = panel[panel.year.le(2014)].copy()
    validation = panel[panel.year.between(2015, 2018)].copy()
    rows, solver_rows = [], []
    for smooth in SMOOTH_GRID:
        for ridge in RIDGE_GRID:
            model = GaugeOperator(attrs, smooth, ridge)
            info = model.fit(train)
            validation["Q_pred_cfs"] = model.predict(validation)
            metrics = summary_metrics(validation, "Q_pred_cfs")
            rows.append({"smooth_precision": smooth, "ridge_precision": ridge, **metrics})
            solver_rows.append({"phase": "hyperparameter", "smooth_precision": smooth, "ridge_precision": ridge, **info})
    grid = pd.DataFrame(rows).sort_values(["station_mean_RMSE_log", "smooth_precision", "ridge_precision"])
    best = grid.iloc[0]
    smooth, ridge = float(best.smooth_precision), float(best.ridge_precision)
    final = GaugeOperator(attrs, smooth, ridge)
    final_info = final.fit(panel[panel.year.le(2018)].copy())
    solver_rows.append({"phase": "full_2006_2018", "smooth_precision": smooth, "ridge_precision": ridge, **final_info})

    check = panel[panel.year.ge(2019) & panel.selected_for_four_group_check].copy()
    check["Q_GAUGE_OPERATOR_cfs"] = final.predict(check)
    check["Q_REACH_FLOW_cfs"] = check.routed_network_native_total_cfs
    parent = pd.read_parquet(PARENT_STATE)[["station_norm", "year", "month", "Q_MAP_cfs"]]
    parent["q_site"] = parent.station_norm.astype(str)
    check = check.merge(parent[["q_site", "year", "month", "Q_MAP_cfs"]], on=["q_site", "year", "month"], validate="one_to_one")
    temporal_rows, station_parts = [], []
    for model_name, col in [("LOCAL_STATION_MAP_UPPER_BOUND", "Q_MAP_cfs"), ("UNIFIED_GAUGE_OPERATOR", "Q_GAUGE_OPERATOR_cfs"), ("CONSERVING_REACH_FLOW", "Q_REACH_FLOW_cfs")]:
        temporal_rows.append({"model": model_name, **summary_metrics(check, col)})
        s = station_metrics(check, col)
        s.insert(0, "model", model_name)
        station_parts.append(s)
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_parquet(OUT / "locked_2019_2022_metrics.parquet", index=False)
    pd.concat(station_parts, ignore_index=True).to_parquet(OUT / "locked_2019_2022_station_metrics.parquet", index=False)
    check[["q_site", "reach_id", "year", "month", "Q_obsv_cfs", "Q_MAP_cfs", "Q_GAUGE_OPERATOR_cfs", "Q_REACH_FLOW_cfs"]].to_parquet(
        OUT / "locked_2019_2022_predictions.parquet", index=False
    )

    outlet = attrs[["comid"]].rename(columns={"comid": "reach_id"}).copy()
    outlet["q_site"] = "OUTLET_DEFAULT"
    outlet["downstream_fraction_on_reach"] = 0.999
    outlet["snap_distance_m"] = 0.0
    outlet["best_line_distance_m"] = 0.0
    outlet["line_catchment_override"] = False
    reach_params, gamma = final.parameter_fields(outlet)
    reach_params.to_parquet(OUT / "reach_outlet_observation_map5_parameters.parquet", index=False)
    gamma.to_parquet(OUT / "observation_attribute_parameters.parquet", index=False)
    np.savez_compressed(
        OUT / "gauge_operator_parameters.npz", theta=final.theta,
        dynamic_mean=final.dynamic_scaling[0], dynamic_std=final.dynamic_scaling[1],
        attribute_mean=final.attribute_scaling[0], attribute_std=final.attribute_scaling[1],
    )
    grid.to_parquet(OUT / "gauge_operator_hyperparameter_grid.parquet", index=False)
    pd.DataFrame(solver_rows).to_parquet(OUT / "gauge_operator_solver_audit.parquet", index=False)

    upper = temporal[temporal.model.eq("LOCAL_STATION_MAP_UPPER_BOUND")].iloc[0]
    gauge = temporal[temporal.model.eq("UNIFIED_GAUGE_OPERATOR")].iloc[0]
    gates = {
        "pooled_NSE_decline_le_0_02": bool(upper.NSE - gauge.NSE <= 0.02),
        "station_median_NSE_decline_le_0_03": bool(upper.station_median_NSE - gauge.station_median_NSE <= 0.03),
        "log_RMSE_increase_le_0_03": bool(gauge.RMSE_log - upper.RMSE_log <= 0.03),
        "absolute_PBIAS_le_5_pct": bool(abs(gauge.PBIAS_pct) <= 5.0),
    }
    decision = {
        "stage": "20260823_23",
        "status": "GAUGE_OPERATOR_TEMPORAL_RETENTION_PASS" if all(gates.values()) else "GAUGE_OPERATOR_TEMPORAL_RETENTION_FAIL",
        "selected_smooth_precision": smooth, "selected_ridge_precision": ridge,
        "full_solver": final_info, "reach_flow_layer_modified": False,
        "hypothetical_outlet_parameter_count": int(len(reach_params)),
        "free_station_identity_columns": 0,
        "known_gauge_history_conditioning": True,
        "temporal_retention_gates": gates,
        "external_four_stations_read": False,
        "tn_interface": "CONSERVING_REACH_FLOW_ONLY",
    }
    (REPORT / "stage23_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_23 unified gauge observation operator\n\n"
        + f"Status: `{decision['status']}`.\n\n## Development-only hyperparameter validation\n\n"
        + grid.to_markdown(index=False) + "\n\n## Locked 2019-2022 gauge check\n\n"
        + temporal.to_markdown(index=False) + "\n\n"
        + "The Gauge layer does not alter Reach flow. TN uses only the conserving Reach-flow product. "
        + "The next stage must test the operator with each target station's history removed.\n",
        encoding="utf-8",
    )
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "reach_flow_sha256": sha256(FLOW), "gauge_parameters_sha256": sha256(OUT / "gauge_operator_parameters.npz"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(temporal.to_string(index=False))


if __name__ == "__main__":
    main()
