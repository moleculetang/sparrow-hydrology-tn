"""Historical readout only; P1/P2 here are not modern process-family P1/P2."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

WATER_EPS = 1e-12

ETA_LAMBDA = 1.0

STATION_LAMBDA = 12.0

ETA_BOUNDARY_TOL = 0.01

def station_effects(log_residual: np.ndarray, station_index: np.ndarray, n_stations: int) -> np.ndarray:
    sums = np.bincount(station_index, weights=log_residual, minlength=n_stations)
    counts = np.bincount(station_index, minlength=n_stations).astype(float)
    return -sums / (counts + STATION_LAMBDA)

def concentration(
    frame: pd.DataFrame,
    eta: np.ndarray,
    point_column: str | None = None,
) -> np.ndarray:
    q = frame.routed_quick_tn_kg_n.to_numpy(float)
    g = frame.routed_gw_tn_kg_n.to_numpy(float)
    point = np.zeros(len(frame), dtype=float) if point_column is None else frame[point_column].fillna(0.0).to_numpy(float)
    water = frame.routed_water_volume_m3.to_numpy(float)
    return np.divide(
        (eta[0] * q + eta[1] * g + point) * 1000.0,
        water,
        out=np.zeros_like(water),
        where=water > WATER_EPS,
    )

def fit_readout(
    train: pd.DataFrame,
    layer: str,
    point_column: str | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, object]]:
    if layer not in {"P1", "P2"}:
        raise ValueError(layer)
    observed_log = np.log1p(train.tn_mg_l.to_numpy(float))
    levels = sorted(train.station_key.astype(str).unique())
    lookup = {station: i for i, station in enumerate(levels)}
    station_index = train.station_key.astype(str).map(lookup).to_numpy(int)

    def components(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.log1p(np.maximum(concentration(train, eta, point_column), 0.0)) - observed_log
        effects = (
            station_effects(raw, station_index, len(levels))
            if layer == "P2"
            else np.zeros(len(levels), dtype=float)
        )
        return raw + effects[station_index], effects

    def residual(eta: np.ndarray) -> np.ndarray:
        data_residual, effects = components(eta)
        pieces = [data_residual, np.sqrt(ETA_LAMBDA) * (eta - 1.0)]
        if layer == "P2":
            pieces.append(np.sqrt(STATION_LAMBDA) * effects)
        return np.concatenate(pieces)

    result = least_squares(
        residual,
        x0=np.array([0.8, 0.8]),
        bounds=(np.zeros(2), np.ones(2)),
        method="trf",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
    )
    _, effects = components(result.x)
    effect_map = {station: float(effects[i]) for i, station in enumerate(levels)}
    diagnostic = {
        "layer": layer,
        "eta_quick": float(result.x[0]),
        "eta_gw": float(result.x[1]),
        "success": bool(result.success),
        "status": int(result.status),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "station_effect_count": len(levels) if layer == "P2" else 0,
        "eta_boundary": bool(np.any(result.x <= ETA_BOUNDARY_TOL) or np.any(result.x >= 1.0 - ETA_BOUNDARY_TOL)),
    }
    return result.x, effect_map, diagnostic
