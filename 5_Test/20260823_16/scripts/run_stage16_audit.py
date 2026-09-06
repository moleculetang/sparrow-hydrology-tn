from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_16"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
PARENT_INPUT = TEST / "20260813_54" / "inputs" / "parent_indata.parquet"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
OBS = TEST / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
MAP = TEST / "20260823_15" / "final_outputs" / "gauged_map_parameters.npz"
MAP_LOCK = TEST / "20260823_15" / "final_reports" / "four_group_training_lock.json"
SHP = ROOT / "0_reach_topology" / "results" / "vectors" / "reaches_topology.shp"
DBF = SHP.with_suffix(".dbf")

MAP5 = ["intercept", "log_q72_total", "q72_quick_fraction", "month_sin", "month_cos"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_dbf(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[4:8])[0]
    header_len = struct.unpack("<H", raw[8:10])[0]
    record_len = struct.unpack("<H", raw[10:12])[0]
    fields = []
    pos = 32
    offset = 1
    while pos < header_len - 1 and raw[pos] != 0x0D:
        desc = raw[pos : pos + 32]
        name = desc[:11].split(b"\x00")[0].decode("ascii", "ignore")
        length = int(desc[16])
        fields.append((name, offset, length))
        offset += length
        pos += 32
    records: list[dict[str, str]] = []
    for i in range(n):
        rec = raw[header_len + i * record_len : header_len + (i + 1) * record_len]
        if not rec or rec[0:1] == b"*":
            records.append({})
            continue
        records.append({name: rec[start : start + length].decode("gb18030", "ignore").strip() for name, start, length in fields})
    return records


def read_polyline_centroids(path: Path) -> list[tuple[float, float]]:
    raw = path.read_bytes()
    pos = 100
    centroids: list[tuple[float, float]] = []
    while pos + 8 <= len(raw):
        _, words = struct.unpack(">2i", raw[pos : pos + 8])
        content = raw[pos + 8 : pos + 8 + words * 2]
        pos += 8 + words * 2
        if len(content) < 44 or struct.unpack("<i", content[:4])[0] not in (3, 5, 13, 15):
            centroids.append((np.nan, np.nan))
            continue
        n_parts, n_points = struct.unpack("<2i", content[36:44])
        point_start = 44 + 4 * n_parts
        pts = np.array([struct.unpack("<2d", content[point_start + 16*i : point_start + 16*(i+1)]) for i in range(n_points)])
        if len(pts) == 0:
            centroids.append((np.nan, np.nan))
        elif len(pts) == 1:
            centroids.append(tuple(pts[0]))
        else:
            seg = pts[1:] - pts[:-1]
            length = np.sqrt(np.square(seg).sum(axis=1))
            mids = (pts[1:] + pts[:-1]) / 2
            if length.sum() > 0:
                xy = np.average(mids, axis=0, weights=length)
            else:
                xy = pts.mean(axis=0)
            centroids.append((float(xy[0]), float(xy[1])))
    return centroids


def terminal_and_distance(topo: pd.DataFrame, length: dict[int, float]) -> tuple[dict[int, int], dict[int, float]]:
    downstream = {
        int(row.reach_id): (None if pd.isna(row.downstream_reach) else int(row.downstream_reach))
        for row in topo.itertuples()
    }
    terminal: dict[int, int] = {}
    distance: dict[int, float] = {}
    for reach in downstream:
        seen: set[int] = set()
        current = reach
        dist = 0.0
        while downstream.get(current) is not None:
            if current in seen:
                raise RuntimeError(f"Cycle detected at Reach {current}")
            seen.add(current)
            dist += float(length.get(current, 0.0))
            current = int(downstream[current])
        terminal[reach] = current
        distance[reach] = dist
    return terminal, distance


def graph_basis(topo: pd.DataFrame, reaches: np.ndarray, n_basis: int = 16) -> tuple[np.ndarray, np.ndarray]:
    index = {int(r): i for i, r in enumerate(reaches)}
    rows, cols = [], []
    for row in topo.itertuples():
        if pd.isna(row.downstream_reach):
            continue
        i, j = index[int(row.reach_id)], index[int(row.downstream_reach)]
        rows.extend([i, j])
        cols.extend([j, i])
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(reaches), len(reaches)))
    lap = laplacian(adj, normed=True)
    vals, vecs = eigsh(lap, k=min(n_basis + 8, len(reaches) - 2), which="SM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    keep = np.where(vals > 1e-8)[0][:n_basis]
    vals, vecs = vals[keep], vecs[:, keep]
    for j in range(vecs.shape[1]):
        pivot = int(np.argmax(np.abs(vecs[:, j])))
        if vecs[pivot, j] < 0:
            vecs[:, j] *= -1
    return vals, vecs


def moran(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, float)
    z = values - values.mean()
    s0 = weights.sum()
    den = float(z @ z)
    return float(len(z) / s0 * (z @ weights @ z) / den) if s0 > 0 and den > 0 else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_candidate_fitting":
        raise RuntimeError("Stage 16 contract was not registered before fitting")

    parent = pd.read_parquet(PARENT_INPUT)
    static = parent.sort_values(["comid", "year", "month"]).groupby("comid", as_index=False).first()
    climate = parent.groupby("comid", as_index=False).agg(
        PPT_mean=("PPT", "mean"), AET_mean=("AET", "mean"), PET_mean=("PET", "mean"),
        PPT_cv=("PPT", lambda x: float(np.std(x) / max(np.mean(x), 1e-12))),
    )
    reaches = np.sort(static.comid.astype(int).unique())
    if len(reaches) != 230:
        raise RuntimeError(f"Expected 230 Reaches, found {len(reaches)}")

    records = read_dbf(DBF)
    centroids = read_polyline_centroids(SHP)
    if len(records) != len(centroids):
        raise RuntimeError("SHP/DBF record mismatch")
    geometry = []
    for rec, (lon, lat) in zip(records, centroids):
        if rec and rec.get("reach_id", "").strip():
            geometry.append({"comid": int(float(rec["reach_id"])), "longitude": lon, "latitude": lat})
    geometry = pd.DataFrame(geometry).drop_duplicates("comid")

    topo = pd.read_csv(TOPOLOGY)
    if set(reaches) != set(topo.reach_id.astype(int)):
        raise RuntimeError("Topology and parent Reach sets differ")
    length = dict(zip(static.comid.astype(int), static.LENGTHKM.fillna(0).astype(float)))
    terminal, distance = terminal_and_distance(topo, length)
    eigenvalues, eigenvectors = graph_basis(topo, reaches)

    attributes = static[["comid", "IncAreaKm2", "CumAreaKm2", "LENGTHKM", "SLOPE", "MaxElSmoCm", "Hydroseq", "TermFlag"]].copy()
    attributes = attributes.merge(climate, on="comid", validate="one_to_one").merge(geometry, on="comid", how="left", validate="one_to_one")
    attributes["terminal_reach"] = attributes.comid.astype(int).map(terminal).astype(int)
    attributes["distance_to_terminal_km"] = attributes.comid.astype(int).map(distance)
    for j in range(eigenvectors.shape[1]):
        lookup = dict(zip(reaches, eigenvectors[:, j]))
        attributes[f"graph_eigen_{j+1:02d}"] = attributes.comid.astype(int).map(lookup)
    attributes.to_parquet(OUT / "reach_regionalization_attributes.parquet", index=False)

    obs = pd.read_parquet(OBS)
    obs["q_site"] = obs.station_norm.astype(str)
    dev = obs[obs.year.le(2018)].copy()
    stations = sorted(dev.q_site.unique())
    if len(stations) != 105 or dev.reach_id.nunique() != 105:
        raise RuntimeError("105-station/105-Reach identity lock failed")
    arrays = np.load(MAP)
    beta = arrays["beta"]
    expected = 18 + len(stations) * 5
    if len(beta) != expected:
        raise RuntimeError(f"Expected {expected} MAP coefficients, found {len(beta)}")
    local = beta[18:].reshape(len(stations), 5)
    station_reach = dev.groupby("q_site").reach_id.first().astype(int)
    params = pd.DataFrame(local, columns=MAP5)
    params.insert(0, "q_site", stations)
    params["reach_id"] = params.q_site.map(station_reach)
    counts = dev.groupby("q_site").agg(n_months=("Q_obsv_cfs", "size"), mean_Q_cfs=("Q_obsv_cfs", "mean"))
    params = params.join(counts, on="q_site").merge(
        attributes.rename(columns={"comid": "reach_id"}), on="reach_id", validate="one_to_one"
    )

    coords = params[["longitude", "latitude"]].to_numpy(float)
    dist2 = np.square(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
    np.fill_diagonal(dist2, np.inf)
    knn = np.zeros((len(params), len(params)), float)
    for i in range(len(params)):
        knn[i, np.argsort(dist2[i])[:5]] = 1.0
    knn = np.maximum(knn, knn.T)
    reach_to_station = {int(r): i for i, r in enumerate(params.reach_id)}
    graph_w = np.zeros_like(knn)
    for row in topo.itertuples():
        if pd.isna(row.downstream_reach):
            continue
        a, b = reach_to_station.get(int(row.reach_id)), reach_to_station.get(int(row.downstream_reach))
        if a is not None and b is not None:
            graph_w[a, b] = graph_w[b, a] = 1.0
    moran_rows = []
    for name in MAP5:
        moran_rows.append({
            "coefficient": name,
            "sd": float(params[name].std(ddof=0)),
            "min": float(params[name].min()),
            "max": float(params[name].max()),
            "moran_I_euclidean_5nn": moran(params[name].to_numpy(float), knn),
            "moran_I_direct_network_neighbors": moran(params[name].to_numpy(float), graph_w),
        })
    moran_frame = pd.DataFrame(moran_rows)
    params.to_parquet(OUT / "local_map5_parameter_audit.parquet", index=False)
    moran_frame.to_parquet(OUT / "map5_spatial_signal_audit.parquet", index=False)
    pd.DataFrame({"basis": [f"graph_eigen_{i+1:02d}" for i in range(len(eigenvalues))], "eigenvalue": eigenvalues}).to_parquet(
        OUT / "river_network_basis_audit.parquet", index=False
    )

    inputs = {str(p): sha256(p) for p in [PARENT_INPUT, TOPOLOGY, OBS, MAP, MAP_LOCK, SHP, DBF, RUN / "experiment_contract.json"]}
    (REPORT / "input_hashes.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    audit = {
        "stage": "20260823_16",
        "status": "PASS",
        "reach_count": int(len(attributes)),
        "station_count": int(len(params)),
        "distinct_training_reaches": int(params.reach_id.nunique()),
        "map5_coefficient_count": int(len(local.ravel())),
        "free_station_identity_allowed_in_final": False,
        "geometry_missing_reaches": int(attributes[["longitude", "latitude"]].isna().any(axis=1).sum()),
        "terminal_tree_count": int(attributes.terminal_reach.nunique()),
        "graph_basis_count": int(len(eigenvalues)),
        "candidate_fitting_performed": False,
        "authorized_successor": "20260823_17"
    }
    (REPORT / "stage16_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 20260823_16 audit result", "", "Status: `PASS`", "",
        f"- 230 Reach attributes complete; geometry missing: {audit['geometry_missing_reaches']}.",
        f"- 105 development stations map one-to-one to {audit['distinct_training_reaches']} Reaches.",
        "- Existing local MAP has exactly 105 × 5 free effects; the promoted model may not retain station-ID effects.",
        "", "## Spatial signal", "", moran_frame.to_markdown(index=False), "",
        "These statistics are diagnostic only. Candidate selection begins in 20260823_17 under the prewritten contract."
    ]
    (REPORT / "technical_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(moran_frame.to_string(index=False))


if __name__ == "__main__":
    main()
