"""The three-layer forward and the per-layer budget, carried over from
`20260919_2/work/phase0_amplitude_budget.py` with `C` -> `common22`.

CARRIED OVER WITH ONE EXECUTABLE DELTA, AND THAT IS MEASURED RATHER THAN ASSERTED
--------------------------------------------------------------------------------
Compared against the read-only `20260919_3\\work\\layers21.py` (211 lines, sha256
`95daccdb9f812f540e48c8497ed25dbd43c4773499c56a4455bd6785d61d85b9`), this file is NOT
byte-identical: `diff -u` exits 1, and EVERY one of its hunks falls inside this
docstring.  Strip both docstrings and `difflib.unified_diff` leaves exactly TWO lines --
`-import common21 as C` / `+import common22 as C` -- which is the WHOLE executable
delta; the function bodies are line-for-line the same.  THIS PARAGRAPH QUOTES NO NUMBER
ABOUT THIS FILE -- not its sha256, not its line count, not the raw hunk count -- because
every one of those changes the moment this paragraph changes, so quoting them would
write a number that is false by construction.  (Two earlier versions got this wrong;
both passes are registered as D17 in `reports\\实际方法与偏离.md`.)  This paragraph and the
one below replace a docstring that described the PREVIOUS round's intervention
(`Predictor.hazard`); leaving it would have been a false statement about this round, so
the prose changed and the code did not.

NOTHING FROZEN IS EDITED.  Every primitive is CALLED through the module that owns it
(`SM.Transport.apply`, `SM.RiverN.apply`, `SM.boundary_mass`), so the frozen kernels
run exactly as registered.  This round's single intervention is the `Xi` factor that
`common22.install_kernel` binds inside the land-phase hazard-to-`prob` conversion; the
call sites, the signatures and this module are untouched by it.  `SM.Transport` here
resolves to `common22`'s rebound `TransportMC` (via the star-import chain
`campaign_model -> temporal_model -> hf_model -> structure_model`), which is why the
rebind has to reach all six modules.

        L1  local[t,r]                 the station reach's OWN land-phase output
        L2  inlet[t,r] + local[t,r]    all mass arriving at the station reach
        L3  boundary_mass(...)         after the boundary/observation operator

and each layer's CONCENTRATION is `1000 * kg / water_m3_day`.  A_L1 / A_L2 / A_L3
are concentration ratios, not mass-flux ratios -- the round's discipline is that
every criterion lands on a concentration.

`L1` READS THE MODULATED KERNEL; `L3` ALSO READS A SECOND, UNMODULATED HAZARD
-----------------------------------------------------------------------------
`L1 = fast + slow + human_mass(t)` comes straight from `SM.Transport.apply`, so `Xi`
acts on it directly.  `L3` goes on through `RiverN.apply` and `boundary_mass`, and
`routing.py:130` computes `mass = inlet*exp(-vf*h*f) + f*local*exp(-.5*vf*h*f)` with
`c['h'] = data.h_month[mid[ti],ri]` (`temporal_model.py:69`) -- a SECOND hazard taken
from the frozen monthly table and NOT modulated.  So `L3`'s concentration depends on
both the modulated land-phase `h` and the unmodulated `h_month`.  That is a legitimate
design choice, but it means a flat `L3` response is NOT evidence that the land-phase
modulation did nothing, and the report must say so.

THE SD GATE IS ddof=0, AND THIS IS A DELIBERATE CHOICE
------------------------------------------------------
`eventlib.station_daily_sd:215` uses pandas' default `ddof=1`; the J2 criterion path
in `20260919_1/work/phase1_fingerprints.py:518` uses `ddof=0`.  The two disagree, so
`station_sd_gate` computes BOTH and returns both, and the round registers which one
carries the criterion.  Silently inheriting one of them is how a baseline becomes
unreproducible.
"""
import numpy as np
import pandas as pd
import torch

import common22 as C
import eventlib as E
import structure_model as SM


def forward_layers(model, tag, meta=None):
    """Mirror of `StructureEndpoints.daily_boundary`, keeping the three layers."""
    x = C.parameters(tag)
    if meta is None:
        meta = pd.read_parquet(C.PEER / 'data/prediction_calendar.parquet')
        meta = meta[meta.year.le(C.END_YEAR)].copy()
    with torch.no_grad():
        t = torch.tensor(x)
        c, record, weights = model.daily_metadata(meta)
        h_, s_, f_, k_ = model.flux_parameters(t[:30])
        fast, slow = SM.Transport.apply(h_, s_, f_, k_, model)
        local = fast + slow + model.human_mass(t)
        inlet, official, releases = SM.RiverN.apply(local, t[2], model.daily_data,
                                                    'monthly', True)
        mass = SM.boundary_mass(inlet, official, releases, local, t[2], c)
    ti = c['ti'].numpy().astype(np.int64)
    ri = c['ri'].numpy().astype(np.int64)
    L = np.asarray(local.numpy(), float)
    I = np.asarray(inlet.numpy(), float)
    M = np.asarray(mass.numpy(), float)
    V = np.asarray(c['water'].numpy(), float)
    df = pd.DataFrame(dict(
        station_key=meta.station_key.to_numpy()[record.numpy()],
        date=model.data.dates[ti],
        water_m3_day=V,
        L1_kg=L[ti, ri],
        L2_kg=I[ti, ri] + L[ti, ri],
        L3_kg=M))
    for lay in ('L1', 'L2', 'L3'):
        df['p' + lay] = 1000.0 * df[lay + '_kg'] / df.water_m3_day
    # `assert_no_mass_columns` guards the table that reaches a CRITERION: the pL* are
    # the only columns a criterion reads.  The *_kg columns are kept because this frame
    # is the layer-budget EVIDENCE and the budget is a mass ledger, which the round
    # discipline exempts explicitly.
    E.assert_no_mass_columns(df[['station_key', 'date', 'pL1', 'pL2', 'pL3']])
    return df, dict(c=c, ti=ti, ri=ri, local=L, inlet=I, mass=M, water=V,
                    record=record, meta=meta, x=x,
                    f_from_hazard=np.asarray(f_.numpy(), float))


def amp_of(layers, ev, lay):
    tbl = E.build_event_table(
        layers[['station_key', 'date', 'p' + lay]].rename(columns={'p' + lay: 'p'}), ev)
    fin = np.isfinite(tbl.c_base) & np.isfinite(tbl.c_peak)
    r = (tbl.c_peak / tbl.c_base)[fin]
    return tbl, fin, r


def layer_budget(layers, ev):
    """Per-layer C_base / C_peak / ratio on the frozen events."""
    out = {}
    for lay in ('L1', 'L2', 'L3'):
        tbl, fin, r = amp_of(layers, ev, lay)
        out[lay] = dict(
            n_events=int(fin.sum()),
            c_base_median=float(tbl.c_base[fin].median()),
            c_peak_median=float(tbl.c_peak[fin].median()),
            amp_ratio_median=float(r.median()),
            amp_ratio_mean=float(r.mean()),
            amp_ratio_sd=float(r.std()),
            n_peak_min=int(tbl.n_peak.min()), n_peak_max=int(tbl.n_peak.max()),
            n_base_unique=[int(v) for v in sorted(tbl.n_base.unique())])
    return out


def station_sd_gate(layers, mask, lays=('L1', 'L2', 'L3')):
    """|log(SD_pred_s / SD_obs_s)| per station, at BOTH ddof conventions.

    WHICH LAYER CARRIES THE GATE
    ---------------------------
    **L3.**  The registered statistic (`phase1_fingerprints.py:518`) takes `p` from an
    arm replay, and arm `P` is `monthfirst` -- the frozen calendar.  Checked rather than
    assumed: `phase1_replay_s1_monthfirst.parquet::p` against the frozen anchor
    `20260916_2/outputs/C0_s1/daily_station_mass_water.parquet::concentration_mg_l`
    agrees at `max|d| = 0` over all 169,476 rows, i.e. arm P IS the frozen kernel.
    The model's `p` is the observation-operator output, which is this module's `pL3`
    (`pL1` is the station reach's own land-phase concentration, upstream of routing).
    The first version of this function defaulted to `L1` and produced 1.3155 instead of
    the published 0.7941; all three layers are reported so that choice is visible.

    THE `ddof` INCONSISTENCY DISSOLVES HERE
    ---------------------------------------
    The plan registered a conflict between `eventlib.station_daily_sd` (`ddof=1`) and
    `phase1_fingerprints.py:518` (`ddof=0`).  For THIS statistic the choice cannot matter:
    `sd_pred` and `sd_obs` are grouped over the same eligible days, so both carry the
    same `sqrt(n_s/(n_s-1))` factor and it cancels in `sd_pred/sd_obs`.  `e_s` is
    therefore ddof-invariant, which the two reported columns confirm by agreeing to all
    16 digits.  Both are still emitted, so the claim is checkable rather than asserted.

    THE REGISTERED BASIS, AND WHY THE FIRST VERSION WAS WRONG
    ---------------------------------------------------------
    `20260919_1/work/phase1_fingerprints.py:486-520` is the definition the stored
    `0.7941489653630477` was computed under, and it is explicit at :500:

        "per-station sd of the station-day MEAN TN, ddof=0, over the ELIGIBLE days"

    The panel is 4-HOURLY (57,284 rows over 12,152 eligible station-days), so the
    observations must be COLLAPSED TO STATION-DAY MEANS (:488) before the sd is taken.
    Merging the raw 4-hourly rows onto the eligible grid measures a *within-day plus
    between-day* dispersion on the observation side against a *between-day* dispersion
    on the model side -- two different quantities, which inflated the baseline to 1.345.
    """
    m = mask[mask.eligible][['station_key', 'date']].copy()
    m['date'] = m.date.astype('datetime64[ns]')
    panel = pd.read_parquet(E.PANEL)
    panel = panel.assign(date=E.as_day(panel.monitoring_time.dt.tz_localize(None)))
    daily = (panel.groupby(['station_key', 'date'], as_index=False)['TN'].mean()
             .rename(columns={'TN': 'obs_tn'}))
    obs_e = m.merge(daily, on=['station_key', 'date'], how='left')
    n_missing = int(obs_e.obs_tn.isna().sum())
    if n_missing:
        # :495-497 raises here; an eligible day with no observation means the two
        # panels are not the same set and every sd below would be over a different n.
        raise SystemExit('ELIGIBLE_DAY_WITHOUT_OBSERVATION %d' % n_missing)
    out = dict(gating_layer='L3', n_obs_rows=int(len(obs_e)),
               n_panel_rows=int(len(panel)),
               panel_rows_per_station_day=float(len(panel) / max(1, len(daily))),
               obs_definition='per-station sd of the station-day MEAN TN, ddof=0, '
                              'over the ELIGIBLE days (phase1_fingerprints.py:500)',
               sd_obs={'ddof0': {str(k): float(v) for k, v in
                                 obs_e.groupby('station_key').obs_tn.std(ddof=0).items()},
                       'ddof1': {str(k): float(v) for k, v in
                                 obs_e.groupby('station_key').obs_tn.std(ddof=1).items()}})
    for lay in lays:
        cmp = m.merge(layers[['station_key', 'date', 'p' + lay]],
                      on=['station_key', 'date'], how='left', validate='one_to_one')
        if int(cmp['p' + lay].isna().sum()):
            raise SystemExit('ARM_DROPPED_AN_ELIGIBLE_DAY %s' % lay)
        node = {}
        for d in (0, 1):
            sm = pd.DataFrame(dict(
                obs=obs_e.groupby('station_key').obs_tn.std(ddof=d),
                mdl=cmp.groupby('station_key')['p' + lay].std(ddof=d))).dropna()
            sm = sm[(sm.obs > 0) & (sm.mdl > 0)]
            e = np.abs(np.log(sm.mdl / sm.obs))
            node['ddof%d' % d] = dict(
                n_stations=int(len(e)),
                median_e=float(e.median()), mean_e=float(e.mean()),
                stations={str(k): float(v) for k, v in e.items()},
                sd_pred={str(k): float(v) for k, v in sm.mdl.items()},
                sd_obs={str(k): float(v) for k, v in sm.obs.items()},
                ratio_mdl_over_obs_median=float((sm.mdl / sm.obs).median()))
        node['ddof0']['gate_threshold_70pct'] = 0.70 * node['ddof0']['median_e']
        kk = sorted(set(node['ddof0']['stations']) & set(node['ddof1']['stations']))
        dev = max(abs(node['ddof0']['stations'][k] - node['ddof1']['stations'][k])
                  for k in kk)
        # `sd_pred` must MOVE with ddof, otherwise the factor was never applied and the
        # invariance below would be a statement about nothing.
        moved = max(abs(node['ddof0']['sd_pred'][k] - node['ddof1']['sd_pred'][k])
                    for k in kk)
        node['ddof_note'] = ('sd_pred and sd_obs are grouped over the SAME eligible days, '
                            'so both carry sqrt(n_s/(n_s-1)) and it cancels in their '
                            'ratio; e_s is therefore ddof-invariant, up to rounding')
        node['ddof0_vs_ddof1_max_abs_diff'] = float(dev)
        node['sd_pred_moves_with_ddof_max_abs_diff'] = float(moved)
        node['ddof_invariant'] = bool(dev <= 1e-12)
        node['ddof_choice_is_inert'] = bool(dev <= 1e-12 and moved > 0.0)
        out[lay] = node
    out['gating'] = out['L3']
    return out


def median_of_stored_e_s():
    """The SAME quantity as recomputed above, read from the artifact that recorded it.

    `20260919_1/reports/phase1_fingerprints.json` stores 15 per-station values under
    `J2_station_sd_log_distance.per_start.C0_s1.e_s_P`.  NO stored scalar holds their
    median, so the plan's registered `0.7941489653630477` is a DERIVED number; Phase 0
    re-derives it and reports both rather than trusting the prose.
    """
    fp = C.read_json(C.PEER.parent / '20260919_1/reports/phase1_fingerprints.json')
    node = fp['J2_station_sd_log_distance']['per_start']['C0_s1']['e_s_P']
    v = np.asarray([float(x) for x in node.values()], float) if isinstance(node, dict) \
        else np.asarray(node, float)
    return dict(source='20260919_1\\reports\\phase1_fingerprints.json'
                       '::J2_station_sd_log_distance.per_start.C0_s1.e_s_P',
                n=len(v), median=float(np.median(v)), mean=float(np.mean(v)),
                values=sorted(float(x) for x in v))
