"""Independent scoring of the FCT8 contrast. No import from `scripts/`.

Why this file exists. `finalize_global.py` scores every registered path, but its
cross-path contrasts are written for the mainline's graph. The single question
this round was built to ask -- does decoupling the month level from the within-
month timing improve TN concentration? -- is answered here.

Like `20260917_5/work/audit_arms.py`, and for the same reason, this module imports
no part of `scripts/`: not `campaign_model`, not `hf_metrics`, not the metric
module. NSE, r, RMSE, log-RMSE and bias are written here from their textbook
formulas, the observed/predicted pairs are rebuilt from the parquets, and the
round's own `reports/station_metrics.parquet` is then read as a third party to
compare against. Reuse for comparability, independence for trust, and the
agreement between the two is itself a reported result.

The contrast is unusually clean, and that is worth stating before the numbers:
candidate and baseline train on the SAME fold, on byte-identical labels, over the
same years, and are evaluated on the same stations and the same evaluation year.
Only the model kind differs. There is no hydrology swap and therefore none of the
water-confounding the bridge round had to carry.

Two readings are kept apart, per the round's pre-registration:

* the FIRST criterion is the paired median `median_i[NSE_fct8_i - NSE_base_i]`.
  The difference of order statistics `median(NSE_fct8) - median(NSE_base)` is
  reported descriptively ONLY -- 20260917_5 measured it manufacturing a positive
  headline while the typical station got worse.
* the magnitude tier is not the bottleneck question and the two are never merged.

Spatial dependence is not assumed away. `reports/spatial_blocks.json` is read as
data; the admission gate it records was frozen before any FCT8 result existed. If
it is not admitted, the station bootstrap below is marked pseudo-replicated and
the leave-one-block-out jackknife plus the per-block effects carry the inference.

Run: PYTHONIOENCODING=utf-8 python -B work/audit_fct8.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / 'reports'
PARENT = RUN.parent / '20260917_5'
EVAL_YEAR = 2024
TRAIN_YEARS = [2021, 2022, 2023]
KEY = ['station_key', 'year', 'month']

#: (baseline fold, FCT8 fold, objective, label)
PAIRS = [('T24_G_D_H1', 'T24_G_D_H1_FCT8', 'D', 'daily-labelled'),
         ('T24_G_M_H1', 'T24_G_M_H1_FCT8', 'M', 'monthly-labelled')]

# The round's own eligibility gate, kept as literals so a pipeline change cannot
# silently widen or narrow what this audit calls eligible.
MIN_DAILY = 30
MIN_DAILY_MONTHS = 3
MIN_MONTHLY = 8

# §6.3, pre-registered in `reports/gate0_preflight.json` before the fits.
P_DNSE, P_DR, P_RMSE_RATIO, P_IMPROVED = 0.10, 0.05, 0.95, 0.60
P_SD_DECLINE, P_BIAS_RATIO = 0.90, 1.05
# The three-tier magnitude reading, kept separate from the promotion gate.
TIER_VERY_SMALL, TIER_SMALL = 0.05, 0.10

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = '') -> None:
    print('  [%s] %s%s' % ('OK ' if condition else 'FAIL', label,
                           ('  ' + detail) if detail else ''))
    if not condition:
        FAILURES.append(label)


def score(y: np.ndarray, p: np.ndarray, scale: str, months: int = 0) -> dict:
    """Textbook metrics. Written here, not imported; see the module docstring."""
    n = len(y)
    ybar = float(y.mean()) if n else np.nan
    den = float(((y - ybar) ** 2).sum()) if n else np.nan
    if scale == 'daily':
        eligible = n >= MIN_DAILY and months >= MIN_DAILY_MONTHS
    else:
        eligible = n >= MIN_MONTHLY
    sy, sp = float(y.std()), float(p.std())
    return dict(
        n=n, eligible=bool(eligible),
        NSE=float(1.0 - ((p - y) ** 2).sum() / den) if eligible and den > 0 else np.nan,
        r=float(np.corrcoef(y, p)[0, 1]) if n >= 3 and sy > 1e-12 and sp > 1e-12 else np.nan,
        RMSE=float(np.sqrt(np.mean((p - y) ** 2))) if n else np.nan,
        logRMSE=float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))) if n else np.nan,
        bias=float((p - y).mean()) if n else np.nan,
        observed_sd=sy, predicted_sd=sp,
        sd_ratio=float(sp / sy) if sy > 1e-12 else np.nan)


def per_station(frame: pd.DataFrame, scale: str) -> pd.DataFrame:
    rows = []
    for key, g in frame.groupby('station_key'):
        rows.append(dict(station_key=key, **score(
            g.y.to_numpy(float), g.p.to_numpy(float), scale,
            months=int(g.month.nunique()) if 'month' in g else 0)))
    return pd.DataFrame(rows)


def month_thresholds(labels: pd.DataFrame) -> pd.Series:
    cal = labels[labels.year.isin(TRAIN_YEARS)]
    counts = cal.groupby('station_key').tn_mg_l.size()
    return cal.groupby('station_key').tn_mg_l.quantile(.90).where(counts >= MIN_MONTHLY)


def build_frames(labels: pd.DataFrame, days: pd.DataFrame, root_of, tag: str) -> dict:
    products = pd.read_parquet(root_of(tag) / 'statistical_products.parquet')
    monthly = labels[labels.year.eq(EVAL_YEAR)][KEY + ['tn_mg_l', 'p90']].merge(
        products[products.year.eq(EVAL_YEAR)][KEY + ['MATCH_mg_l', 'FW_mg_l']],
        on=KEY, validate='one_to_one', how='inner')
    d = pd.read_parquet(root_of(tag) / 'daily_station_mass_water.parquet')
    daily = days[['station_key', 'date', 'y', 'month']].merge(
        d[['station_key', 'date', 'concentration_mg_l']],
        on=['station_key', 'date'], validate='one_to_one', how='inner')
    return dict(monthly_all=monthly.rename(columns={'tn_mg_l': 'y', 'MATCH_mg_l': 'p'}),
                monthly_fw=monthly.rename(columns={'tn_mg_l': 'y', 'FW_mg_l': 'p'}),
                daily=daily.rename(columns={'concentration_mg_l': 'p'}))


def select_arm(tags: list[str]) -> dict:
    """Mirror `finalize_global.py`: lowest objective among physically legal fits."""
    valid, selected = {}, {}
    for tag in tags:
        path = RUN / 'outputs' / tag / 'audit.json'
        audit = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if audit.get('status') != 'AUDITED_FIT' or not audit.get('physical_reasonable'):
            continue
        valid[tag] = audit
        fold = tag.rsplit('_s', 1)[0]
        if fold not in selected or audit['objective'] < valid[selected[fold]]['objective']:
            selected[fold] = tag
    return dict(valid=valid, selected=selected)


def tier(delta: float) -> str:
    if not np.isfinite(delta):
        return 'UNDEFINED'
    a = abs(delta)
    if a < TIER_VERY_SMALL:
        return 'VERY_SMALL'
    if a < TIER_SMALL:
        return 'SMALL'
    return 'LARGE'


def block_jackknife(delta_by_station: dict, group_of: dict) -> dict:
    """Leave-one-block-out jackknife of the paired median.

    The unit left out is the hydrologic block, not the station: dropping one
    station and dropping a whole tributary are different claims about dependence,
    and only the second is honest here.
    """
    keys = np.array(sorted(delta_by_station))
    d = np.array([delta_by_station[k] for k in keys], float)
    blocks = {}
    for k in keys:
        blocks.setdefault(group_of.get(k, 'UNASSIGNED'), []).append(k)
    full = float(np.median(d))
    out = {}
    for name, members in sorted(blocks.items()):
        mask = ~np.isin(keys, members)
        out[name] = dict(stations=len(members), n_remaining=int(mask.sum()),
                         median_without=float(np.median(d[mask])) if mask.any() else np.nan,
                         shift=float(np.median(d[mask]) - full) if mask.any() else np.nan)
    shifts = [v['shift'] for v in out.values() if np.isfinite(v['shift'])]
    return dict(full_median=full, n_blocks=len(blocks), blocks=out,
                worst_shift=float(max(shifts, key=abs)) if shifts else np.nan,
                shift_range=[float(min(shifts)), float(max(shifts))] if shifts else None,
                note='jackknife over hydrologic blocks, not stations; a large worst_shift '
                     'means a single block carries the headline')


def main() -> None:
    started = time.time()
    protocol = json.loads((RUN / 'configs/protocol.json').read_text(encoding='utf-8')) \
        if (RUN / 'configs/protocol.json').exists() else {}
    draws = int(protocol.get('bootstrap', 1000))
    seed = int(protocol.get('seed', 1729))
    blocks_doc = json.loads((OUT / 'spatial_blocks.json').read_text(encoding='utf-8'))
    admission = blocks_doc['admission']
    report = dict(
        stage='20260918_1/work/audit_fct8',
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        bootstrap=draws, seed=seed, eval_year=EVAL_YEAR,
        thresholds=dict(paired_dNSE=P_DNSE, paired_dr=P_DR, rmse_ratio=P_RMSE_RATIO,
                        improved_fraction=P_IMPROVED, sd_decline=P_SD_DECLINE,
                        bias_ratio=P_BIAS_RATIO),
        spatial_admission=admission)

    print('spatial admission: bootstrap %s (n_components %d, largest share %.3f)'
          % ('ADMITTED' if admission['component_cluster_bootstrap_admitted'] else 'NOT ADMITTED',
             admission['n_nonempty_components'], admission['largest_share']), flush=True)

    labels = pd.read_parquet(RUN / 'data/heldout_labels/monthly_original.parquet')
    labels['p90'] = labels.station_key.map(month_thresholds(labels))
    days = pd.read_parquet(RUN / 'data/heldout_labels/T24_G_evaluation_BJT_days.parquet')
    n_panel = labels[labels.year.eq(EVAL_YEAR)].station_key.nunique()
    check(n_panel == 116, 'the 2024 evaluation panel carries the full registered cohort',
          '%d stations, %d station-months' % (n_panel, int(labels.year.eq(EVAL_YEAR).sum())))

    def root_of(tag: str) -> Path:
        return RUN / 'outputs' / tag

    arms = {}
    for base_fold, fct8_fold, obj, lab in PAIRS:
        b = select_arm(['%s_s0' % base_fold, '%s_s1' % base_fold])
        f = select_arm(['%s_s0' % fct8_fold, '%s_s1' % fct8_fold])
        # The baseline is reused exactly as published; if selection does not land on
        # the registered s0 this is a different comparison and must be said out loud.
        if b['selected'].get(base_fold) != '%s_s0' % base_fold:
            print('  [note] baseline selection for %s is %s, not the registered s0'
                  % (base_fold, b['selected'].get(base_fold)), flush=True)
        arms[base_fold] = dict(objective=obj, label=lab, base=b, fct8=f,
                               base_tag=b['selected'].get(base_fold),
                               fct8_tag=f['selected'].get(fct8_fold))
    report['arms'] = {k: dict(objective=v['objective'], label=v['label'],
                              base_tag=v['base_tag'], fct8_tag=v['fct8_tag'],
                              base_starts=sorted(v['base']['valid']),
                              fct8_starts=sorted(v['fct8']['valid']))
                      for k, v in arms.items()}
    for k, v in arms.items():
        print('  %-12s base %-18s fct8 %s' % (k, v['base_tag'], v['fct8_tag']), flush=True)

    # ---- A. per-station criteria table, both arms, both scales
    table, station_rows = [], []
    block_l1 = {s: b for b, m in blocks_doc['level1']['members'].items() for s in m}
    block_l2 = {s: b for b, m in blocks_doc['level2']['members'].items() for s in m}
    report['spatial_blocks_used'] = dict(level1_blocks=blocks_doc['level1']['n_components'],
                                         level2_blocks=blocks_doc['level2']['n_blocks'])
    for base_fold, fct8_fold, obj, lab in PAIRS:
        for arm, tag in [('D29_BE', arms[base_fold]['base_tag']),
                         ('FCT8', arms[base_fold]['fct8_tag'])]:
            if tag is None:
                continue
            frames = build_frames(labels, days, root_of, tag)
            for scale, frame in [('monthly_PUB', frames['monthly_all']),
                                 ('monthly_FW', frames['monthly_fw']),
                                 ('daily', frames['daily'])]:
                if frame.empty:
                    continue
                sites = per_station(frame, scale)
                sites.insert(0, 'tag', tag)
                sites.insert(1, 'arm', arm)
                sites.insert(2, 'fold', base_fold)
                sites.insert(3, 'scale', scale)
                station_rows.append(sites)
                table.append(dict(tag=tag, arm=arm, fold=base_fold, scale=scale,
                                  objective=obj, label=lab, stations=len(sites),
                                  eligible=int(sites.eligible.sum()),
                                  NSE_median=float(sites[sites.eligible].NSE.median()),
                                  r_median=float(sites[sites.eligible].r.median()),
                                  sd_ratio_median=float(sites.sd_ratio.median())))
    sites_table = pd.concat(station_rows, ignore_index=True)
    sites_table.to_parquet(OUT / 'fct8_station_metrics.parquet', index=False)
    pd.DataFrame(table).to_csv(OUT / 'fct8_criteria.csv', index=False)
    report['criteria'] = table

    # ---- B. the paired contrast
    contrasts = []
    for base_fold, fct8_fold, obj, lab in PAIRS:
        a = arms[base_fold]
        if a['fct8_tag'] is None:
            print('  [note] %s has no FCT8 fit yet; contrast skipped' % fct8_fold, flush=True)
            continue
        for scale in ['monthly_PUB', 'monthly_FW', 'daily']:
            x = sites_table[(sites_table.tag.eq(a['base_tag'])) & sites_table.scale.eq(scale)]
            y = sites_table[(sites_table.tag.eq(a['fct8_tag'])) & sites_table.scale.eq(scale)]
            if x.empty or y.empty:
                continue
            both = x.merge(y, on='station_key', suffixes=('_b', '_c'), validate='one_to_one')
            ok = (both.NSE_b.notna() & both.NSE_c.notna()).to_numpy()
            if not ok.any():
                continue
            nb = both.NSE_b.to_numpy(float)[ok]
            nc = both.NSE_c.to_numpy(float)[ok]
            rb = both.r_b.to_numpy(float)[ok]
            rc = both.r_c.to_numpy(float)[ok]
            keys = both.station_key.to_numpy()[ok]
            d = nc - nb
            # |log SD ratio| per station, and the mean absolute bias per station
            logsd_b = np.abs(np.log(both.sd_ratio_b.to_numpy(float)[ok]))
            logsd_c = np.abs(np.log(both.sd_ratio_c.to_numpy(float)[ok]))
            abias_b = np.abs(both.bias_b.to_numpy(float)[ok])
            abias_c = np.abs(both.bias_c.to_numpy(float)[ok])
            rmse_b, rmse_c = both.RMSE_b.to_numpy(float)[ok], both.RMSE_c.to_numpy(float)[ok]
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(d), size=(draws, len(d)))
            boot = np.median(d[idx], axis=1)
            row = dict(
                fold=base_fold, objective=obj, label=lab, scale=scale, stations=int(ok.sum()),
                # first criterion: paired
                median_delta_NSE_i=float(np.median(d)),
                # descriptive only; the difference of order statistics
                delta_NSE_median=float(np.median(nc) - np.median(nb)),
                pooled_median_unstable=bool(
                    np.isfinite(np.median(d))
                    and np.sign(np.median(nc) - np.median(nb)) != np.sign(np.median(d))),
                median_delta_r_i=float(np.nanmedian(rc - rb)),
                improved_fraction=float((d > 0).mean()),
                rmse_ratio_station_mean=float(np.nanmean(rmse_c) / np.nanmean(rmse_b)),
                rmse_ratio_pooled=float(np.sqrt(np.nanmean(rmse_c ** 2))
                                        / np.sqrt(np.nanmean(rmse_b ** 2))),
                sd_decline_ratio=float(np.median(logsd_c) / np.median(logsd_b))
                if np.median(logsd_b) > 0 else np.nan,
                negative_r_fraction_baseline=float(np.nanmean(rb < 0)),
                negative_r_fraction_fct8=float(np.nanmean(rc < 0)),
                mean_abs_bias_baseline=float(np.nanmean(abias_b)),
                mean_abs_bias_fct8=float(np.nanmean(abias_c)),
                bias_ratio=float(np.nanmean(abias_c) / np.nanmean(abias_b))
                if np.nanmean(abias_b) > 0 else np.nan,
                tier=tier(float(np.median(d))),
                # station bootstrap is pseudo-replicated when the admission gate fails
                station_bootstrap_ci95=[float(np.quantile(boot, .025)),
                                        float(np.quantile(boot, .975))],
            )
            row['promotion'] = dict(
                paired_dNSE_ge=bool(row['median_delta_NSE_i'] >= P_DNSE),
                paired_dr_ge=bool(row['median_delta_r_i'] >= P_DR),
                rmse_ratio_le=bool(row['rmse_ratio_station_mean'] <= P_RMSE_RATIO),
                improved_ge=bool(row['improved_fraction'] >= P_IMPROVED),
                sd_decline_ge=bool(np.isfinite(row['sd_decline_ratio'])
                                   and row['sd_decline_ratio'] <= P_SD_DECLINE),
                negative_r_not_worse=bool(row['negative_r_fraction_fct8']
                                          <= row['negative_r_fraction_baseline'] + 1e-12),
                bias_ok=bool(np.isfinite(row['bias_ratio']) and row['bias_ratio'] <= P_BIAS_RATIO))
            row['promoted'] = bool(all(row['promotion'].values()))
            # spatial inference: jackknife over blocks, not stations
            dmap = dict(zip(keys, d))
            row['jackknife_level1'] = block_jackknife(dmap, block_l1)
            row['jackknife_level2'] = block_jackknife(dmap, block_l2)
            row['per_block_effect_level1'] = {
                name: dict(stations=len(m), median_delta_NSE_i=float(np.median(
                    [dmap[s] for s in m if s in dmap])))
                for name, m in blocks_doc['level1']['members'].items()
                if any(s in dmap for s in m)}
            contrasts.append(row)
    report['contrasts'] = contrasts

    print('\n[contrasts]  first criterion is the PAIRED median')
    for c in contrasts:
        flag = '  <-- POOLED UNSTABLE' if c['pooled_median_unstable'] else ''
        print('  %-12s %-11s n=%3d  paired dNSE %+.4f  pooled %+.4f  dr %+.4f  '
              'RMSEratio %.4f  improved %.1f%%  tier %s%s'
              % (c['fold'], c['scale'], c['stations'], c['median_delta_NSE_i'],
                 c['delta_NSE_median'], c['median_delta_r_i'],
                 c['rmse_ratio_station_mean'], 100 * c['improved_fraction'], c['tier'], flag))
    print('\n[spatial]  jackknife over hydrologic blocks')
    for c in contrasts:
        j1, j2 = c['jackknife_level1'], c['jackknife_level2']
        print('  %-12s %-11s Level1 %2d blocks worst shift %+.4f   '
              'Level2 %2d blocks worst shift %+.4f'
              % (c['fold'], c['scale'], j1['n_blocks'], j1['worst_shift'],
                 j2['n_blocks'], j2['worst_shift']))

    # ---- C. promotion verdict
    main_row = next((c for c in contrasts
                     if c['fold'] == 'T24_G_D_H1' and c['scale'] == 'monthly_PUB'), None)
    report['headline'] = main_row
    report['headline_choice'] = dict(
        fold='T24_G_D_H1', scale='monthly_PUB',
        reason='the main experiment named in the plan: the daily-labelled fold, which is '
               'the only one whose training signal can see within-month timing at all. '
               'The M fold is the negative control and is reported beside it, never merged.')
    if main_row:
        print('\n[promotion]  T24_G_D_H1 monthly, all seven required')
        for name, held in main_row['promotion'].items():
            print('      %-22s %s' % (name, 'HOLDS' if held else 'fails'))
        print('  promoted: %s' % main_row['promoted'])

    # ---- D. checks
    print('\n[checks]')
    pub_path = OUT / 'station_metrics.parquet'
    if pub_path.exists():
        theirs = pd.read_parquet(pub_path)
        worst, compared = 0.0, 0
        for row in sites_table[sites_table.scale.eq('monthly_PUB')].itertuples():
            other = theirs[theirs.tag.eq(row.tag) & theirs.scale.eq('monthly_PUB')
                           & theirs.station_key.eq(row.station_key)]
            if len(other) != 1 or not np.isfinite(row.NSE) or not np.isfinite(other.NSE.iloc[0]):
                continue
            worst = max(worst, abs(row.NSE - float(other.NSE.iloc[0])))
            compared += 1
        report['independent_vs_pipeline'] = dict(compared=compared, max_abs_delta_NSE=worst)
        check(worst < 1e-9,
              'independent implementation agrees with the pipeline\'s station metrics',
              '%d station-scores, max|dNSE| %.3e' % (compared, worst))

    # The baseline must reproduce its published metrics.  The pre-registered
    # tolerance is 1e-9 (plan section 10, hard gate 5), and it is a tolerance rather
    # than exact equality because this module re-derives the sums in its own row
    # order: `((p-y)**2).sum()` over the same station's months is the same real
    # number but not the same sequence of float64 additions.  Demanding 0.0 here
    # would be measuring summation order, and the measured 2.8e-14 leaves five
    # orders of margin against the registered bound.
    pub_parent = PARENT / 'reports' / 'station_metrics.parquet'
    if pub_parent.exists():
        old = pd.read_parquet(pub_parent)
        mine = sites_table[sites_table.arm.eq('D29_BE')]
        worst, compared = 0.0, 0
        for row in mine.itertuples():
            other = old[old.tag.eq(row.tag) & old.scale.eq(row.scale)
                        & old.station_key.eq(row.station_key)]
            if len(other) != 1 or not np.isfinite(row.NSE) or not np.isfinite(other.NSE.iloc[0]):
                continue
            worst = max(worst, abs(row.NSE - float(other.NSE.iloc[0])))
            compared += 1
        report['baseline_reproduction'] = dict(
            compared=compared, max_abs_delta_NSE=worst, tolerance=1e-9,
            note='re-derived sums differ from the pipeline only in float64 addition '
                 'order; the registered bound is 1e-9')
        check(worst <= 1e-9,
              'the reused baseline reproduces the published per-station metrics',
              '%d station-scores, max|dNSE| %.3e (bound 1e-9)' % (compared, worst))

    n_fct8 = sum(1 for a in arms.values() if a['fct8_tag'])
    check(n_fct8 == len(PAIRS), 'every fold has a fitted FCT8 path to contrast',
          '%d/%d' % (n_fct8, len(PAIRS)))

    report['status'] = 'PASS_FCT8_AUDIT' if not FAILURES else 'INCOMPLETE_FCT8_AUDIT'
    report['failures'] = FAILURES
    report['seconds'] = time.time() - started
    (OUT / 'fct8_arms.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('\n%s  %.1fs  -> reports/fct8_arms.json' % (report['status'], report['seconds']))
    for name in FAILURES:
        print('    - %s' % name)


if __name__ == '__main__':
    main()
