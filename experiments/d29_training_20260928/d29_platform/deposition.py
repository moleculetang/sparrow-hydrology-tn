"""TN-independent, mass-conserving monthly deposition to daily interface.

Calendar conversion acts on already integrated noleap *month masses*. It does
not reinterpret the original flux as a Gregorian daily flux.
"""
from __future__ import annotations

import calendar
import numpy as np

COMPONENTS = ("drynhx", "drynoy", "wetnhx", "wetnoy")
LAND_CLASSES = ("cropland", "noncropland", "impervious", "water", "unknown")
CLCD_TO_CLASS = np.array([4, 0, 1, 1, 1, 3, 1, 1, 2, 1], dtype=np.int8)


def distribute_month(mass: np.ndarray, rain: np.ndarray, year: int, month: int, *, mode='H1_rain'):
    """Return days x reaches x four components, and explicit unresolved cells.

    In default H1_rain mode positive wet mass with missing or zero precipitation is left NaN. Dry mass
    remains usable. No epsilon, cross-month movement or implicit zero filling.
    monthly_uniform_reference is a separately declared timing scenario applied
    to every month, not a hidden repair of selected dry-H1 cells.
    """
    mass = np.asarray(mass, dtype=np.float64)
    rain = np.asarray(rain, dtype=np.float64)
    ndays = calendar.monthrange(int(year), int(month))[1]
    if mass.ndim != 2 or mass.shape[1] != 4 or rain.shape != (ndays, mass.shape[0]):
        raise ValueError("Monthly mass / Gregorian rain axes differ")
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("Monthly deposition must be finite nonnegative kg N")
    if np.any(rain[np.isfinite(rain)] < 0):
        raise ValueError("Negative H1 rainfall")
    if mode=='monthly_uniform_reference':
        # An independently named monthly-information scenario, applied to ALL
        # wet months. It does not repair H1 rainfall or invent observed wet days.
        out=np.broadcast_to(mass[None,:,:]/ndays,(ndays,*mass.shape)).copy()
        out[-1] += mass-out.sum(axis=0)
        if np.any(out<0):raise ArithmeticError('NEGATIVE_UNIFORM_DEPOSITION')
        return out,[]
    if mode!='H1_rain':raise ValueError('UNKNOWN_DEPOSITION_DAILY_MODE')
    out = np.full((ndays, mass.shape[0], 4), np.nan, dtype=np.float64)
    out[:, :, :2] = mass[None, :, :2] / ndays
    out[-1, :, :2] += mass[:, :2] - out[:, :, :2].sum(axis=0)
    conflicts = []
    for r in range(mass.shape[0]):
        valid = bool(np.isfinite(rain[:, r]).all())
        total = float(rain[:, r].sum()) if valid else np.nan
        for c in (2, 3):
            if mass[r, c] == 0:
                out[:, r, c] = 0.0
            elif not valid or total <= 0:
                conflicts.append(dict(reach_index=r, component=COMPONENTS[c],
                                      unresolved_kg_n=float(mass[r, c]),
                                      reason="missing_month_precipitation" if not valid else "zero_month_precipitation"))
            else:
                out[:, r, c] = mass[r, c] * (rain[:, r] / total)
                last = np.flatnonzero(rain[:, r] > 0)[-1]
                out[last, r, c] += mass[r, c] - out[:, r, c].sum()
    finite = out[np.isfinite(out)]
    if np.any(finite < 0):
        raise ArithmeticError("Daily disaggregation introduced negative mass")
    return out, conflicts


def distribute_class_month(mass: np.ndarray, rain: np.ndarray, year: int, month: int):
    """Apply the same calendar to reach x five-class x component month mass.

    Keeps classes separate and assigns no class to a physical process. Callers
    must select their declared soil/other domains rather than sum all classes.
    """
    mass = np.asarray(mass, dtype=np.float64)
    rain = np.asarray(rain, dtype=np.float64)
    if mass.ndim != 3 or mass.shape[1:] != (5, 4):
        raise ValueError("Expected reach x five land classes x four components")
    daily, flat_conflicts = distribute_month(mass.reshape(-1, 4), np.repeat(rain, 5, axis=1), year, month)
    conflicts = []
    for gap in flat_conflicts:
        flat_index = gap.pop("reach_index")
        conflicts.append(dict(reach_index=flat_index // 5, land_class=LAND_CLASSES[flat_index % 5], **gap))
    return daily.reshape(daily.shape[0], mass.shape[0], 5, 4), conflicts


def normalized_class_area(counts: np.ndarray, exact: np.ndarray, pixel_area=900.0):
    """Same pixel-centre and overcoverage rule as the frozen reach product.

    counts shape (..., 10) retains exact CLCD codes. Unknown includes nodata and
    uncovered polygon edges; impervious is kept apart from soil-support land.
    """
    counts = np.asarray(counts)
    exact = np.asarray(exact, dtype=np.float64)
    if counts.shape[:-1] != exact.shape or counts.shape[-1] != 10:
        raise ValueError("Classification axes differ")
    if np.any(counts < 0) or not np.isfinite(exact).all() or np.any(exact < 0):
        raise ValueError("Negative counts/areas or nonfinite exact areas")
    raw = counts.astype(np.float64) * pixel_area
    sampled = raw.sum(axis=-1)
    scale = np.ones_like(exact)
    np.divide(exact, sampled, out=scale, where=sampled > exact)
    raw *= np.minimum(scale, 1.0)[..., None]
    raw[..., 0] += np.maximum(exact - raw.sum(axis=-1), 0.0)
    grouped = np.stack([raw[..., CLCD_TO_CLASS == k].sum(axis=-1) for k in range(5)], axis=-1)
    return grouped, raw
