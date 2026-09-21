"""F0.2 -- the 116-station error phenotype table. Zero fits, diagnosis only.

Reads only published artefacts of the frozen parent round: the per-tag
`_PUB_evaluated.parquet` files hold the 2024 observed/predicted station-months and
`data/station_registry.parquet` holds the reach assignment. No model is built, no
parameter is fitted, and no FCT8 result is touched -- this is the table the round
reads *before* F2, so that "what kind of error each station has" is on record
independently of whether FCT8 helps.

Four classes, one per station, assigned by frozen thresholds (written into the
output json so they cannot be re-chosen after seeing FCT8):

    SUPPORT      n_months < 8, or the reach carries more than one station
    LEVEL        |bias| / sd_obs
    AMPLITUDE    |log(sd_pred / sd_obs)|
    TIMING       Pearson r of the monthly series

SUPPORT is checked first and is a statement about what the data can bear, not about
the model; the remaining three are ranked by their excess over threshold and the
largest wins. Every station keeps all three deficiencies in the table either way,
so the class is a ranking and never hides a second failing axis.

The `y`/`p` columns are re-derived here into mean, sd, bias, r, NSE and RMSE rather
than read from `station_metrics.parquet`, so this table is an independent check of
that file as well as a phenotype.

Output: reports/f0_phenotype.parquet, reports/f0_phenotype.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
PARENT = RUN.parent / '20260917_5'
EVAL_YEAR = 2024

#: primary baseline for the FCT8 contrast, plus the monthly-only arm for context
TAGS = {'D': 'T24_G_D_H1_s0', 'M': 'T24_G_M_H1_s0'}

# frozen before any FCT8 result exists
T_LEVEL, T_AMPLITUDE, T_TIMING = 0.50, 0.50, 0.50
T_SUPPORT_MONTHS, T_SUPPORT_SHARED = 8, 2


def sha(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def station_axes(frame):
    """Per-station monthly error axes, computed from the (y, p) pairs only."""
    rows = []
    for key, g in frame.groupby('station_key'):
        y = g.y.to_numpy(float)
        p = g.p.to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        n = len(y)
        if n < 2:
            rows.append(dict(station_key=key, n_months=n, cohort=g.cohort.iloc[0],
                             usable=False))
            continue
        my, mp = y.mean(), p.mean()
        sy, sp = y.std(ddof=1), p.std(ddof=1)
        bias = mp - my
        r = float(np.corrcoef(y, p)[0, 1]) if sy > 0 and sp > 0 else np.nan
        rows.append(dict(
            station_key=key, n_months=n, cohort=g.cohort.iloc[0],
            hf_group=g.hf_group.iloc[0], has_daily_training=bool(g.has_daily_training.iloc[0]),
            training_location=bool(g.training_location.iloc[0]), usable=True,
            mean_obs=my, mean_pred=mp, sd_obs=sy, sd_pred=sp, bias=bias,
            # scale-free deficiencies; sd_obs == 0 is reported as an absent axis
            level=float(bias / sy) if sy > 0 else np.nan,
            log_sd_ratio=float(np.log(sp / sy)) if sy > 0 and sp > 0 else np.nan,
            r=r,
            RMSE=float(np.sqrt(np.mean((p - y) ** 2))),
            NSE=float(1 - np.sum((p - y) ** 2) / np.sum((y - my) ** 2)) if np.sum((y - my) ** 2) > 0 else np.nan,
        ))
    return pd.DataFrame(rows)


def classify(tab):
    """One class per station, SUPPORT first, then the largest threshold excess."""
    z_level = tab['level'].abs() / T_LEVEL
    z_amp = tab['log_sd_ratio'].abs() / T_AMPLITUDE
    z_time = (T_TIMING - tab['r']).clip(lower=0) / T_TIMING
    support = (tab.n_months < T_SUPPORT_MONTHS) | (tab.same_reach_stations >= T_SUPPORT_SHARED)
    stacked = np.vstack([z_level.fillna(0), z_amp.fillna(0), z_time.fillna(0)])
    pick = np.array(['LEVEL', 'AMPLITUDE', 'TIMING'])[np.argmax(stacked, axis=0)]
    tab['excess_level'], tab['excess_amplitude'], tab['excess_timing'] = z_level, z_amp, z_time
    tab['axes_over_threshold'] = [
        '+'.join(a for a, v in (('LEVEL', l), ('AMPLITUDE', m), ('TIMING', t))
                 if np.isfinite(v) and v >= 1.0) or 'none'
        for l, m, t in zip(z_level, z_amp, z_time)]
    tab['phenotype'] = np.where(support, 'SUPPORT', pick)
    return tab


def main():
    reg = pd.read_parquet(RUN / 'data/station_registry.parquet')
    shared = reg.groupby('global_reach_id').station_key.nunique().rename('same_reach_stations')
    reach_of = reg.groupby('station_key').global_reach_id.first()

    tables, out = {}, {}
    for arm, tag in TAGS.items():
        src = PARENT / 'reports' / ('%s_PUB_evaluated.parquet' % tag)
        frame = pd.read_parquet(src)
        frame = frame[frame.year == EVAL_YEAR]
        tab = station_axes(frame)
        tab['tag'] = tag
        tab['arm'] = arm
        tab['global_reach_id'] = tab.station_key.map(reach_of)
        tab['same_reach_stations'] = tab.global_reach_id.map(shared).fillna(0).astype(int)
        tab = classify(tab)
        tables[arm] = tab
        primary = tab[tab.usable]
        out[arm] = dict(
            tag=tag, source=str(src.relative_to(RUN.parent)), source_sha256=sha(src),
            stations=int(tab.station_key.nunique()), usable=int(len(primary)),
            months_total=int(tab.n_months.sum()),
            phenotype_counts=primary.phenotype.value_counts().to_dict(),
            axes_over_threshold_counts=primary.axes_over_threshold.value_counts().to_dict(),
            deficit_medians=dict(
                level_abs=float(primary['level'].abs().median()),
                log_sd_ratio_abs=float(primary['log_sd_ratio'].abs().median()),
                r=float(primary['r'].median()), NSE=float(primary['NSE'].median()),
                n_months=float(primary.n_months.median())),
            crosstab_phenotype_by_cohort=pd.crosstab(primary.cohort, primary.phenotype).to_dict(),
        )
        print('\n=== %s (%s) : %d stations ===' % (arm, tag, len(tab)))
        print(primary.phenotype.value_counts().to_string())
        print('  median |level| %.3f   median |log sd ratio| %.3f   median r %.3f   median NSE %.3f'
              % (primary['level'].abs().median(), primary['log_sd_ratio'].abs().median(),
                 primary.r.median(), primary.NSE.median()))
        print('  stations on a shared reach: %d' % int((tab.same_reach_stations >= 2).sum()))
        print(pd.crosstab(primary.cohort, primary.phenotype).to_string())

    primary = tables['D']
    primary.to_parquet(RUN / 'reports/f0_phenotype.parquet', index=False)
    payload = dict(
        status='F0_PHENOTYPE_DIAGNOSIS_ONLY', evaluation_year=EVAL_YEAR, zero_fits=True,
        note=('Diagnosis only, no per-station model. `y`/`p` are re-derived into mean, sd, bias, r, '
              'NSE and RMSE here rather than read from station_metrics.parquet, so this table is '
              'also an independent recomputation of that file.'),
        thresholds=dict(T_LEVEL=T_LEVEL, T_AMPLITUDE=T_AMPLITUDE, T_TIMING=T_TIMING,
                        T_SUPPORT_MONTHS=T_SUPPORT_MONTHS, T_SUPPORT_SHARED=T_SUPPORT_SHARED),
        classification_rule=('SUPPORT if n_months < T_SUPPORT_MONTHS or the reach carries '
                            '>= T_SUPPORT_SHARED stations; otherwise the largest of '
                            '|level|/T_LEVEL, |log_sd_ratio|/T_AMPLITUDE, (T_TIMING-r)+/T_TIMING.'),
        support_axis_note=('`same_reach_stations` counts registered stations sharing a reach. It is '
                           'the plan\'s "irreducible lower bound from same-reach stations": those '
                           'stations see the same modelled water, so their error cannot be driven to '
                           'zero by any parameterisation of this model.'),
        flow_class_availability=('flow_class exists only in reports/flow_diagnostics.parquet and '
                                 'covers the 15 HF stations, not the 116-station monthly panel, so no '
                                 'flow axis is claimed here.'),
        water_confidence=('No measured-vs-simulated water flag exists in this round\'s registries; '
                          'the water axes are the .npy domain arrays only. Recorded as an absent '
                          'axis rather than filled with a proxy.'),
        arms=out)
    (RUN / 'reports/f0_phenotype.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True, default=str),
        encoding='utf-8')
    print('\nwrote reports/f0_phenotype.parquet (%d rows) and f0_phenotype.json' % len(primary))


if __name__ == '__main__':
    sys.exit(main())
