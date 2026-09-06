"""SAS sparrow_graphs.sas replacement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, cos, floor, pi
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass
class GraphsResult:
    outputs: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _split_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    if isinstance(value, (list, tuple)):
        tokens: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                tokens.extend(item.split())
            else:
                tokens.append(str(item))
        return tokens
    return str(value).split()


def _first_token(value: object) -> str:
    tokens = _split_tokens(value)
    return tokens[0] if tokens else ""


def _is_yes(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() == "YES"


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _nanmax(values: np.ndarray | list[float]) -> float:
    try:
        return float(np.nanmax(values))
    except ValueError:
        return float("nan")


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _graph_results(
    df: pd.DataFrame,
    xvar: str,
    yvar: str,
    imagename: str,
    graphtype: str,
    graphtitle: str,
    graphtitle2: str,
    output_dir: Path,
    outputs: dict[str, Path],
    warnings: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        warnings.append("matplotlib is required for graph outputs.")
        return

    x = df[xvar].to_numpy(dtype=float)
    y = df[yvar].to_numpy(dtype=float)

    min_x = np.nanmin(x) if x.size else np.nan
    min_y = np.nanmin(y) if y.size else np.nan
    minval = np.nanmin([min_x, min_y])
    minx = min_x
    meany = np.nanmean(y) if y.size else np.nan
    sdevy = np.nanstd(y, ddof=1) if y.size else np.nan

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=25, c="black")

    graphtype = graphtype.upper()
    max_x = _nanmax(x)
    max_y = _nanmax(y)
    maxval = _nanmax([max_x, max_y])
    if np.isnan(maxval):
        maxval = minval

    if graphtype == "ONETOONE":
        ax.plot([minval, maxval], [minval, maxval], color="black")
    elif graphtype == "ZERO":
        ax.plot([minx, max_x], [0, 0], color="black")
    elif graphtype == "PPLOT":
        intercept = meany + minx * sdevy
        x_end = max_x
        ax.plot([minx, x_end], [intercept, intercept + sdevy * (x_end - minx)], color="black")

    if graphtitle:
        ax.set_title(graphtitle)
    if graphtitle2:
        ax.set_title(graphtitle + "\n" + graphtitle2 if graphtitle else graphtitle2)

    fig.tight_layout()
    outpath = output_dir / f"{imagename}.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    outputs[imagename] = outpath


def _mappoints(
    basemap: pd.DataFrame,
    basemapid: str,
    events: pd.DataFrame,
    eventsx: str,
    eventsy: str,
    eventsresponse: str,
    eventsresponselevels: list[float],
    eventsmarkercolors: list[str],
    eventsmarkersizes: list[float],
    eventsmarkerrotates: list[int],
    output_dir: Path,
    outputs: dict[str, Path],
    warnings: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        warnings.append("matplotlib is required for graph outputs.")
        return

    base = basemap.copy()
    if "segment" in base.columns:
        segment_col = "segment"
    else:
        segment_col = None

    base = base.sort_index()
    rows = []
    last_segment = None
    last_id = None
    for _, row in base.iterrows():
        seg = row.get(segment_col) if segment_col else None
        bid = row.get(basemapid)
        if last_segment is not None and (seg != last_segment or bid != last_id):
            rows.append({"x": row.get("x"), "y": np.nan})
        rows.append({"x": row.get("x"), "y": row.get("y")})
        last_segment = seg
        last_id = bid
    base_plot = pd.DataFrame(rows)

    events_df = events[[eventsx, eventsy, eventsresponse]].copy()
    events_df = events_df.rename(columns={eventsx: "x", eventsy: "y", eventsresponse: "resp"})

    minx = np.nanmin(base_plot["x"]) if not base_plot.empty else np.nan
    maxx = np.nanmax(base_plot["x"]) if not base_plot.empty else np.nan
    miny = np.nanmin(base_plot["y"]) if not base_plot.empty else np.nan
    maxy = np.nanmax(base_plot["y"]) if not base_plot.empty else np.nan

    e_minx = np.nanmin(events_df["x"]) if not events_df.empty else minx
    e_maxx = np.nanmax(events_df["x"]) if not events_df.empty else maxx
    e_miny = np.nanmin(events_df["y"]) if not events_df.empty else miny
    e_maxy = np.nanmax(events_df["y"]) if not events_df.empty else maxy

    x1 = max(floor(e_minx - 0.5), minx)
    x2 = min(ceil(0.5 + e_maxx + 0.5), maxx)
    y1 = max(floor(e_miny - 0.5), miny)
    y2 = min(ceil(0.5 + e_maxy + 0.5), maxy)

    lat_mid = (y1 + y2) / 2
    distratio = 1 / cos(2 * pi * (lat_mid) / 360)
    aspectratio = distratio * (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 1

    nlevels = len(eventsresponselevels) + 1
    def _level(val: float) -> int:
        for i, level in enumerate(eventsresponselevels, start=1):
            if val < level:
                return i
        return nlevels

    markers = []
    for val in events_df["resp"].to_numpy(dtype=float):
        level = _level(val)
        color = eventsmarkercolors[level - 1]
        size = eventsmarkersizes[level - 1]
        rotate = eventsmarkerrotates[level - 1]
        marker = "v" if int(rotate) == 180 else "^"
        markers.append((color, size, marker))

    fig, ax = plt.subplots()
    ax.plot(base_plot["x"], base_plot["y"], color="black", linewidth=0.5)

    for (color, size, marker), (_, row) in zip(markers, events_df.iterrows()):
        ax.scatter(row["x"], row["y"], c=color, s=size * 10, marker=marker)

    ax.set_aspect(aspectratio)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    ax.axis("off")

    footnotes = _map_footnotes(
        eventsresponselevels, eventsmarkercolors, eventsmarkersizes, eventsmarkerrotates
    )
    fig.text(0.5, 0.01, footnotes[0], ha="center", fontsize=8)
    fig.text(0.5, 0.0, footnotes[1], ha="center", fontsize=8)

    fig.tight_layout()
    outpath = output_dir / "ResidualsMap.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    outputs["ResidualsMap"] = outpath


def _map_footnotes(
    levels: list[float],
    colors: list[str],
    sizes: list[float],
    rotates: list[int],
) -> tuple[str, str]:
    maxsize = max(sizes) if sizes else 8
    parts = []
    nlevels = len(levels) + 1
    for i in range(1, nlevels + 1):
        if i == nlevels:
            text = f">= {levels[i-2]}" if levels else ">= 0"
        elif i > 1:
            text = f"{levels[i-2]} to {levels[i-1]}"
        else:
            text = f"< {levels[i-1]}"
        parts.append(text)
    foot1 = "  ".join(parts)
    foot2 = "(+) under-predict  (-) over-predict"
    return foot1, foot2


def graph_resids(
    resids: pd.DataFrame,
    config: Mapping[str, object],
    *,
    basemap: pd.DataFrame | None = None,
) -> GraphsResult:
    warnings: list[str] = []
    errors: list[str] = []
    outputs: dict[str, Path] = {}

    output_dir = _ensure_dir(config.get("home_results") or ".")

    lat = _first_token(config.get("lat"))
    lon = _first_token(config.get("lon"))
    if_gis = _is_yes(config.get("if_gis"))
    gis_file = str(config.get("gis_file") or "").strip()

    plotdat = resids.copy()
    mapresids = plotdat
    if lat and lon and lat in plotdat.columns and lon in plotdat.columns:
        mapresids = plotdat[(plotdat[lat].notna()) & (plotdat[lon].notna()) & (plotdat["map_resid"].notna())]
    else:
        mapresids = plotdat.iloc[0:0]

    nobs = len(plotdat)

    sd_norm_weight = plotdat["norm_weight"].std(ddof=1) if "norm_weight" in plotdat.columns else np.nan
    if_same = bool(sd_norm_weight == 0 or np.isnan(sd_norm_weight))

    _graph_results(
        plotdat,
        "ln_predict",
        "ln_actual",
        "PredictVsObserved",
        "ONETOONE",
        f"Predicted Relative to Observed Flux at {nobs} Sites",
        "",
        output_dir,
        outputs,
        warnings,
    )
    _graph_results(
        plotdat,
        "ln_predict",
        "ln_resid",
        "PredictVsResiduals",
        "ZERO",
        f"Predicted Relative to Residual Flux at {nobs} Sites",
        "",
        output_dir,
        outputs,
        warnings,
    )

    if not if_same:
        _graph_results(
            plotdat,
            "ln_predict",
            "weighted_ln_resid",
            "PredictVsWghtResids",
            "ZERO",
            f"Predicted Relative to Weighted Residual Flux at {nobs} Sites",
            "",
            output_dir,
            outputs,
            warnings,
        )
        _graph_results(
            plotdat,
            "ln_pred_yield",
            "weighted_ln_resid",
            "YieldVsWghtResids",
            "ZERO",
            f"Predicted Yield Relative to Weighted Residual Flux at {nobs} Sites",
            "",
            output_dir,
            outputs,
            warnings,
        )
    else:
        _graph_results(
            plotdat,
            "ln_pred_yield",
            "ln_resid",
            "YieldVsResiduals",
            "ZERO",
            f"Predicted Yield Relative to Residual Flux at {nobs} Sites",
            "",
            output_dir,
            outputs,
            warnings,
        )

    _graph_results(
        plotdat,
        "z_map_resid",
        "map_resid",
        "ProbabilityPlot",
        "PPLOT",
        "Probability Plot of Residuals",
        "(Residuals are shown in natural logarithm units)",
        output_dir,
        outputs,
        warnings,
    )

    if if_gis and gis_file and basemap is not None and not mapresids.empty:
        _mappoints(
            basemap=basemap,
            basemapid="fips",
            events=mapresids,
            eventsx=lon,
            eventsy=lat,
            eventsresponse="map_resid",
            eventsresponselevels=[-1.5, 0.0, 1.5],
            eventsmarkercolors=["red", "orange", "green", "blue"],
            eventsmarkersizes=[10, 6, 6, 10],
            eventsmarkerrotates=[180, 180, 0, 0],
            output_dir=output_dir,
            outputs=outputs,
            warnings=warnings,
        )
    else:
        if if_gis:
            warnings.append(
                "Residuals map not produced; missing lat/lon, gis_file, or basemap data."
            )

    return GraphsResult(outputs=outputs, warnings=warnings, errors=errors)
