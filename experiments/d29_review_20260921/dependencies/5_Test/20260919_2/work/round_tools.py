"""Model-side helpers shared by Phase 0 and Phase 1.

WHY THIS FILE EXISTS (a structural deviation from the plan's section 6 file list)
-------------------------------------------------------------------------------
`event_window_mask` answers "which (day, reach) cells are inside an event's peak
window".  Phase 0 asks what the kernel does inside that window; Phase 1 asks
whether the capped days fall inside it.  If each phase carried its own copy, the
two would be answering *almost* the same question -- and the previous round's B1
defect was exactly a window bound that drifted between two implementations of one
recipe (a `searchsorted` defaulting to `side='left'` searched 6 days where the
observation searched 7).  A shared definition is the fix, so this module is added
and the addition is registered in `reports/实际方法与偏离.md`.

The window itself is the frozen one, unchanged: `[t_start, t_end + 1d]` with BOTH
ends inclusive (`20260918_4/work/stage_a_fingerprints.py:145`, mirrored at
`eventlib.base_peak`).
"""
import numpy as np
import pandas as pd
import torch

import common20 as C
import eventlib as E


def station_reach_map(model):
    """station_key -> reach index, from the SAME `c['ri']` the forward uses.

    Taking the map from a stored table instead would silently decouple this
    diagnostic from the branch `daily_boundary` actually computes on.
    """
    meta = pd.read_parquet(C.PEER / 'data/prediction_calendar.parquet')
    meta = meta[meta.year.le(C.END_YEAR)].copy()
    with torch.no_grad():
        c, _rec, _w = model.daily_metadata(meta)
    ri = c['ri'].numpy().astype(np.int64)
    return {str(sk): int(r) for sk, r in zip(meta.station_key.to_numpy(), ri)}


def event_window_mask(model, ev, nd, nr, reach_of=None):
    """(nd, nr) boolean: the event's own reach, on the days of its peak window."""
    dates = np.asarray(pd.to_datetime(model.data.dates).values.astype('datetime64[D]'))
    mask = np.zeros((nd, nr), bool)
    ts, te = E.event_windows(ev)
    reach_of = station_reach_map(model) if reach_of is None else reach_of
    for i, sk in enumerate(ev.station_key.values):
        r = reach_of[str(sk)]
        lo = np.datetime64(ts.iloc[i], 'D')
        hi = np.datetime64(te.iloc[i] + pd.Timedelta(days=1), 'D')
        mask[(dates >= lo) & (dates <= hi), r] = True
    return mask, reach_of, dates


def capped_diagnostics(kappa, mask, ev, reach_of, dates):
    """Where the cap actually binds, and whether it binds inside event windows.

    The plan's section 10 makes this the falsifier of the registered prediction
    P1: if `C_peak` ever rises, the rise must be traced to events that DO have a
    capped day, and reported per event.  So the capped cells are located here,
    once, on the same grid the amplitude budget uses.
    """
    k = np.asarray(kappa, float)
    capped = k < 1.0 - 1e-12
    ts, te = E.event_windows(ev)
    inside = capped & mask
    per_event = []
    d = pd.Series(dates)
    for i, sk in enumerate(ev.station_key.values):
        r = reach_of[str(sk)]
        lo = np.datetime64(ts.iloc[i], 'D')
        hi = np.datetime64(te.iloc[i] + pd.Timedelta(days=1), 'D')
        sel = (dates >= lo) & (dates <= hi)
        per_event.append(dict(
            station_key=str(sk), event_id=str(ev.event_id.values[i]),
            n_window_days=int(sel.sum()),
            n_capped_days=int(capped[sel, r].sum()),
            min_kappa_in_window=float(k[sel, r].min()) if sel.sum() else None))
    pe = pd.DataFrame(per_event)
    return dict(
        n_domain_cells=int(k.size), n_capped_cells=int(capped.sum()),
        share_capped_cells=float(capped.mean()),
        n_capped_cells_inside_event_windows=int(inside.sum()),
        share_of_capped_inside_windows=float(inside.sum() / capped.sum())
        if capped.sum() else None,
        min_kappa=float(k.min()),
        n_reaches_with_any_capped_cell=int(capped.any(axis=0).sum()),
        per_event=per_event,
        n_events_with_a_capped_day=int((pe.n_capped_days > 0).sum()),
        n_events=int(len(pe)))


def cap_headroom(full, mask):
    """How far the cap is from binding, INSIDE the event windows.

    The cap is `E = min(av*p, R)`, so it binds exactly when `R < av*p`.  The ratio
    `R / (av*p)` on the window cells is therefore the headroom, and its minimum is
    the load-bearing number: if it exceeds 1 everywhere, no window cell can be
    capped at that point, which is a statement about the state and the data rather
    than about the grid.

    `full` is `closures_r.scan_r`'s 7-tuple: (fast, slow, av, p_raw, kappa, R, praw).
    """
    av, praw, R = np.asarray(full[2], float), np.asarray(full[6], float), np.asarray(full[5], float)
    efull = av * praw
    sel = mask & (efull > 0)
    if not sel.sum():
        return dict(n_window_cells=int(mask.sum()), n_cells_with_output=0,
                    min_headroom=None, median_headroom=None,
                    note='no window cell has a positive unconstrained output')
    h = (R[sel] / efull[sel])
    return dict(n_window_cells=int(mask.sum()), n_cells_with_output=int(sel.sum()),
                min_headroom=float(h.min()), q05_headroom=float(np.percentile(h, 5)),
                median_headroom=float(np.median(h)),
                frac_below_1=float((h < 1.0).mean()),
                note=('R / (av*p) on the event-window cells; < 1 means the cap binds. '
                      'This is the quantity that decides whether the state can act on '
                      'an event at all.'))


def event_table_for(rep, ev):
    """`eventlib.build_event_table` on a replay frame, with the mass-column gate."""
    t = E.build_event_table(rep[['station_key', 'date', 'p']], ev)
    E.assert_no_mass_columns(t)
    return t
