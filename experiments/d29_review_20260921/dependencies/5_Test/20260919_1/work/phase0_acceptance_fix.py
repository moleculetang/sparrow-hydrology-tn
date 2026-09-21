"""Phase 0 -- fix the OPTIMIZER CONTINUATION condition; KEEP the ENDPOINT absolute gate.

THE ONE SENTENCE THIS FILE EXISTS TO MAKE IMPOSSIBLE TO MISREAD
---------------------------------------------------------------
There are now TWO thresholds, they do NOT have the same value, and only one of
them is the scientific acceptance:

    optimizer continuation (F-b):   E_M / S_M <= 1e-6      <-- CHANGED, relative
    final scientific acceptance (F-c): E_M <= 1e-6 kg      <-- UNCHANGED, absolute

The registered line was `max(source_label_sum_errors.values()) <= 1e-6`, in
ABSOLUTE kg, and it is the FIFTH conjunct of `fit_worker.py:70`; it is what
rejected C2.  Writing "1e-6 一字未改，因此没有放宽" would be false: with
S_M ~ 1.55e8 kg the continuation condition becomes ~1.55e8 times looser.  That
number is printed below and must appear next to the claim, in the report.

WHY THIS IS PHASE 0 AND NOT A TIDY-UP
-------------------------------------
The stop was not a stop at an unconverged interior point.  The optimizer REACHED
the optimum and the optimum was REJECTED: both C2 starts end with a last trace
row that is `physical_legal: false` at an objective *below* C0's converged value,
with 1128 / 1131 rejected points that differ from the accepted ones in exactly
one channel (M).  So the delivered `objective` 2.0168 / 1.5161 is the last LEGAL
save point, not the configuration's optimum, and "C2 is worse than C0" is a
reading of the criterion's output rather than of the calendar.

WHAT THIS FILE WRITES, AND IN WHAT ORDER
----------------------------------------
Every quantity in `reports/预注册_判据与门槛.md` is frozen and hashed here, BEFORE
`phase1_replay_calendar.py` runs and before any uniform_daily reading exists:

  * S_M, uniquely defined (S2.4) and evaluated on the frozen table
  * the J1-J5 criterion FORMS, the numeric thresholds, and the DIRECTION predictions
  * the column names -- two different quantities both named amplitude_ratio exist in
    this family and must never share a name again
  * the observation-side fingerprints beta_obs / alpha_obs (from the registered
    lineage, arm A primaries) and A_obs / SD_obs,s (computed here, observation
    only, no model in the loop)
  * the single common eligible mask (S3.6), so no arm can drop rows on its own

The no-op proof (S2.5) is asserted, not narrated: the endpoint audit arithmetic on
the monthfirst arm must reproduce `outputs/C0_s{,0,1}/source_label_audit.json`
EXACTLY -- float equality, not allclose -- because a criterion change that moved a
monthfirst reading would not be a no-op.
"""
import json
import pickle
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common19 as C

R = C.ROUND
PEER = C.PEER
PREV = Path(r'E:\SPARROW\5_Test\20260918_4')
OBS_PANEL = Path(r'E:\SPARROW\5_Test\20260918_1\reports\ammonia_TN_quality_only.parquet')
FINGERPRINTS = PREV / 'reports/stage_a_fingerprints.json'
EVENTS = PREV / 'data/stage_a_events.parquet'

KIND = 'D29_BE'
PRIMARY_ARM = 'A:routed:D3'          # the model runs MODELLED hydrology -> routed events
SECONDARY_ARM = 'B:observed:D3'
START_YEAR, END_YEAR = 2021, 2024     # FULL24 has no hydrology in 2025: the day axis stops
                                      # at 2024-12-31 and map_observations raises past it.
N_BOOT = 2000
SEED = 20260917
NOISE_BAND = 0.0529                   # 2 x the _5 null sd 0.02647 -- a REFERENCE ONLY,
                                      # the baseline for this round is recomputed here (S4.4)
J5_MAX_DEGRADATION = 0.005


# ---------------------------------------------------------------------------
# 1. the stop cause, re-verified rather than quoted
# ---------------------------------------------------------------------------

def trace_summary(tag):
    p = PEER / 'work/jobs' / tag / 'trace.jsonl'
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    legal = [r for r in rows if r['physical_legal']]
    out = dict(tag=tag, n_rows=len(rows), n_legal=len(legal),
               n_illegal=len(rows) - len(legal),
               min_objective=float(min(r['objective'] for r in rows)),
               min_objective_legal=float(min(r['objective'] for r in legal)),
               last_row=rows[-1],
               last_legal=legal[-1])
    return out


def rejection_channels(tag):
    """The per-rejection channel breakdown, from the checkpoint state itself.

    fit_worker.py:71 appends `rejected_physical_points` entries carrying
    `local` and the full six-channel `source_errors` dict, so the question
    "was every rejection the M channel?" is answerable from the record rather
    than from the sentence in status.json.
    """
    cp = PEER / 'work/jobs' / tag / 'checkpoints'
    best = None
    for f in sorted(cp.glob('*.pkl')):
        try:
            o = pickle.loads(f.read_bytes())
        except Exception:
            continue
        if isinstance(o, dict) and o.get('rejected_physical_points'):
            if best is None or len(o['rejected_physical_points']) > len(best[1]['rejected_physical_points']):
                best = (f, o)
    if best is None:
        raise SystemExit('NO_CHECKPOINT_WITH_REJECTIONS ' + tag)
    f, o = best
    rej = o['rejected_physical_points']
    chans = sorted(rej[0]['source_errors'])
    over = collections.Counter()
    for r in rej:
        over[tuple(sorted(k for k, v in r['source_errors'].items() if v > 1e-6))] += 1
    return dict(checkpoint=f.name, n_rejected=len(rej), channels=chans,
                over_channel_pattern={str(k): v for k, v in over.items()},
                max_per_channel={k: float(max(r['source_errors'][k] for r in rej)) for k in chans},
                max_local=float(max(r['local'] for r in rej)),
                n_local_over_1e6=int(sum(1 for r in rej if r['local'] > 1e-6)),
                last_rejected=rej[-1]['source_errors']['M'],
                last_rejected_call=int(rej[-1]['call']),
                last_rejected_objective=float(rej[-1]['objective']))


# ---------------------------------------------------------------------------
# 2. the two-layer criterion, with the magnitude of the change printed
# ---------------------------------------------------------------------------

def criterion_readings(M_errors, S_M):
    return dict(
        f_b_continuation_ratio_max=float(max(M_errors) / S_M),
        f_b_limit=1e-6,
        f_b_passes=bool(max(M_errors) / S_M <= 1e-6),
        f_c_endpoint_absolute_kg_max=float(max(M_errors)),
        f_c_limit_kg=1e-6,
        f_c_passes=bool(max(M_errors) <= 1e-6))


def main():
    ledgers = {}
    audits = {}
    S = {}

    # ---- 1. stop cause -----------------------------------------------------
    traces = {t: trace_summary(t) for t in ('C0_s0', 'C0_s1', 'C2_s0', 'C2_s1')}
    rejects = {t: rejection_channels(t) for t in ('C2_s0', 'C2_s1')}
    stop_reason = {}
    for t in ('C2_s0', 'C2_s1'):
        st = json.loads((PEER / 'work/jobs' / t / 'status.json').read_text(encoding='utf-8'))
        stop_reason[t] = dict(status=st['status'], stop_reason=st['stop_reason'],
                              calls=int(st['calls']), best_objective=float(st['best']['objective']),
                              best_call=int(st['best']['call']),
                              process_alive_is_stale_snapshot=bool(st['process']['alive']))
    S['stop_cause'] = dict(traces=traces, rejections=rejects, status=stop_reason)

    # the single load-bearing sentence, restated from measurement
    for t in ('C2_s0', 'C2_s1'):
        r = rejects[t]
        assert set(r['over_channel_pattern']) == {"('M',)"}, r['over_channel_pattern']
        assert r['n_local_over_1e6'] == 0
        assert traces[t]['last_row']['physical_legal'] is False
        assert traces[t]['last_row']['objective'] < traces['C0_s1']['min_objective_legal']
    print('stop cause: the OPTIMUM itself was rejected; every rejection is the M channel',
          flush=True)

    # ---- 2. F-a: diagnostic only ------------------------------------------
    fa = json.loads((PEER / 'reports/tag_roundoff_diagnostic.json').read_text(encoding='utf-8'))
    S['f_a_diagnostic_only'] = dict(
        source='20260916_2/reports/tag_roundoff_diagnostic.json',
        training_modified=fa['training_modified'], n_records=len(fa['records']),
        records=[dict(tag=r['tag'], M_original=r['original_errors']['M'],
                      M_subtractive=r['subtractive_errors']['M'],
                      largest_inventory_kg=r['largest_inventory_kg']) for r in fa['records']],
        note=('an algebraically equivalent re-sum was TRIED and did NOT repair the M '
              'residual (s0 3.6806e-6 -> 3.5912e-6, s1 3.1590e-6 -> 3.4571e-6); this is '
              'reported as a diagnostic and carries no repair promise'))

    # ---- 3. S_M, uniquely frozen ------------------------------------------
    for tag in ('C0_s0', 'C0_s1'):
        m = C.build(tag, 'monthfirst')
        x = C.parameters(tag)
        a = m.ledger(x)
        rr = np.asarray(m.data.pilot_indices)
        ledgers[tag] = a
        audits[tag] = (rr, float(np.max(np.abs(np.asarray(a['M'])[:, rr]))))
        print('%s ledger: S_M=%.10f kg  E_M=%.6e' % (tag, audits[tag][1],
                                                     a['source_label_sum_errors']['M']), flush=True)

    S_M = min(audits[t][1] for t in audits)
    S_M_from = min(audits, key=lambda t: audits[t][1])
    assert S_M > 0
    S['s_m_freeze'] = dict(
        definition="S_M := max_{t, r in pilot_indices} |a['M'][t, r]| on a frozen monthfirst reference solution",
        why_this_support=("source_label_sum_errors is generated by "
                          "campaign_model.py:100-107 as np.max(abs(v.sum(-1) - a[name][:, rr])) "
                          "with rr = data.pilot_indices, so the scale must be taken on the SAME "
                          "pilot_indices or the scale and the residual are on different axes"),
        pilot_indices=[int(i) for i in audits['C0_s1'][0]],
        pilot_global_reach_ids=[int(i) + 1 for i in audits['C0_s1'][0]],
        n_pilot_indices=int(len(audits['C0_s1'][0])),
        values_kg={t: audits[t][1] for t in audits},
        selected=S_M, selected_from=S_M_from,
        selection_rule='the smaller of the two, so the direction is strict',
        frozen_before='any uniform_daily replay (this file runs before phase1_replay_calendar.py)')
    S['two_layer_criterion'] = dict(
        f_b=dict(name='optimizer continuation', form='E_M / S_M <= 1e-6',
                 S_M_kg=S_M, limit=1e-6,
                 implied_absolute_allowance_kg=1e-6 * S_M,
                 relaxation_factor_vs_absolute=float(1e-6 * S_M / 1e-6)),
        f_c=dict(name='final scientific acceptance', form='E_M <= 1e-6 kg',
                 limit_kg=1e-6, unchanged=True, source='fit_worker.py:70 fifth conjunct'),
        honesty=(('F-b and F-c are NOT the same threshold.  With S_M = %.6e kg the continuation '
                  'condition permits up to %.4f kg of M residual, i.e. %.4e times the absolute '
                  '1e-6 kg.  F-b therefore stops constraining float64 M residuals at all; the '
                  'strictness of the acceptance is carried entirely by F-c.') %
                 (S_M, 1e-6 * S_M, 1e-6 * S_M / 1e-6)))
    print('S_M = %.10f kg  ->  F-b allows %.4f kg (%.4e x the absolute gate)'
          % (S_M, 1e-6 * S_M, 1e-6 * S_M / 1e-6), flush=True)

    # what F-b would have done to the rejected points
    S['f_b_effect_on_the_rejected_points'] = {
        t: dict(n_rejected=rejects[t]['n_rejected'],
                max_M_kg=rejects[t]['max_per_channel']['M'],
                max_M_over_S_M=float(rejects[t]['max_per_channel']['M'] / S_M),
                would_become_legal=bool(rejects[t]['max_per_channel']['M'] / S_M <= 1e-6),
                last_rejected_M_over_S_M=float(rejects[t]['last_rejected'] / S_M))
        for t in rejects}

    # ---- 4. no-op proof on the monthfirst arm -----------------------------
    noop = {}
    for tag in ('C0_s0', 'C0_s1'):
        stored = json.loads((PEER / 'outputs' / tag / 'source_label_audit.json').read_text(encoding='utf-8'))
        got = ledgers[tag]['source_label_sum_errors']
        mismatch = {k: (stored['max_errors'][k], float(got[k]))
                    for k in stored['max_errors'] if float(got[k]) != float(stored['max_errors'][k])}
        noop[tag] = dict(
            stored_passes=bool(stored['passes']),
            channels_reproduced_exactly=(len(mismatch) == 0),
            mismatches=mismatch,
            stored_M=float(stored['max_errors']['M']), recomputed_M=float(got['M']),
            old_absolute_gate_passes=bool(got['M'] <= 1e-6),
            new_relative_gate_passes=bool(got['M'] / S_M <= 1e-6),
            reading_unchanged=bool(got['M'] <= 1e-6 and got['M'] / S_M <= 1e-6),
            form='float equality per channel, never allclose')
        if mismatch:
            raise SystemExit('NOOP_PROOF_FAILED %s %s' % (tag, mismatch))
    S['no_op_proof'] = dict(
        claim='the two-layer criterion leaves the monthfirst arm bitwise unchanged',
        method=('re-run model.ledger(x) on the frozen parameters and compare every one of the '
                'six source_label_sum_errors channels to outputs/<tag>/source_label_audit.json '
                'by float equality'),
        per_tag=noop,
        holds=bool(all(v['reading_unchanged'] and v['channels_reproduced_exactly']
                       for v in noop.values())))
    print('no-op proof: monthfirst endpoint audit reproduced exactly, verdict unchanged',
          flush=True)

    # ---- 5. the network_balance misattribution ----------------------------
    add = json.loads((PEER / 'reports/additive_channel_validation.json').read_text(encoding='utf-8'))
    S['network_balance_correction'] = dict(
        claimed_by_the_plan=dict(monthfirst=-8.940696716308594e-07,
                                 uniform_daily=-8.940696716308594e-06),
        where_those_numbers_actually_come_from=dict(
            file='20260916_2/reports/additive_channel_validation.json',
            producer='20260916_2/scripts/validate_additive_channel.py',
            configuration=('human=TRUE, design=data/designs/T24_G_D.json, x=m.initial(1) with '
                           'x[2]=0. -- a SYNTHETIC fixture, not a frozen C0/C2 solution')),
        measured_on_the_frozen_solutions_kg={
            t: float(ledgers[t]['network_balance_kg']) for t in ledgers},
        status='RECORDED_NOT_ASSERTED either way',
        correction=('the plan quoted this pair while describing the C0/C2 stop.  It is not a '
                    'C0/C2 reading.  The frozen C0 solutions give the values above, all below '
                    'one ulp of the network scale (~2.9e10-3.3e10 kg, ulp ~3.8e-6 kg).  Both '
                    'readings agree on the point that matters: this quantity is NOT the stop '
                    'cause, which is the fifth conjunct.'))

    # ---- 6. observation-side freeze ---------------------------------------
    fp = json.loads(FINGERPRINTS.read_text(encoding='utf-8'))
    f1 = fp['results']['F1'][PRIMARY_ARM]['T_interevent']
    f3 = fp['results']['F3'][PRIMARY_ARM]['alpha_gap_le_30']
    ev = pd.read_parquet(EVENTS)
    ok = ev[(ev.arm == PRIMARY_ARM) & (ev.obs_status == 'OK')].copy()
    A_obs = float(np.median(ok.obs_ratio.to_numpy(float)))
    years = sorted(set(pd.to_datetime(ev.t_start).dt.year) | set(pd.to_datetime(ev.t_end).dt.year))

    panel = pd.read_parquet(OBS_PANEL)
    panel['monitoring_time'] = pd.to_datetime(panel.monitoring_time)
    panel['date'] = panel.monitoring_time.dt.tz_localize(None).dt.normalize()
    daily = (panel.groupby(['station_key', 'date'], as_index=False)['TN'].mean()
             .rename(columns={'TN': 'obs_tn'}))
    daily = daily[(daily.date.dt.year >= START_YEAR) & (daily.date.dt.year <= END_YEAR)]
    sd_obs = daily.groupby('station_key')['obs_tn'].std(ddof=0)

    S['observation_side_freeze'] = dict(
        primary_arm=PRIMARY_ARM, secondary_arm=SECONDARY_ARM,
        why=('the model runs MODELLED hydrology (H1/FULL24), so events delineated from the '
             'routed water are the matched counterpart; arm B is kept as a sensitivity'),
        beta_obs=dict(value=float(f1['point']), n_units=int(f1['n_units']),
                      n_clusters=int(f1['n_clusters']), p=float(f1['p_two_sided']),
                      specification=f1['specification'],
                      control_note=f1['control_note'],
                      source='20260918_4/reports/stage_a_fingerprints.json'),
        alpha_obs=dict(value=float(f3['point']), n_units=int(f3['n_units']),
                       n_clusters=int(f3['n_clusters']), p=float(f3['p_two_sided']),
                       variant='alpha_gap_le_30 (the variant the lineage registers as primary)',
                       source='20260918_4/reports/stage_a_fingerprints.json'),
        A_obs=dict(value=A_obs, definition='median of obs_ratio = obs_tn_peak / obs_tn_base',
                   n=int(len(ok)), arm=PRIMARY_ARM,
                   computed_here=True, not_from_5=True),
        SD_obs_s=dict(values={k: float(v) for k, v in sd_obs.items()},
                      definition=('per-station sd over days of the station-day MEAN TN, '
                                  '2021-01-01..2024-12-31, from the 4h panel; ddof=0'),
                      matched_to='C_sim,s,d = the model daily concentration on the same station-day'),
        event_years_present=years,
        no_2025_event_exists=bool(max(years) <= END_YEAR),
        n_2025_events_excluded=int(sum(1 for y in years if y > END_YEAR)),
        note_2025=('the frozen event set contains NO 2025 event: the year set is %s.  The '
                   '2025-hydrology exclusion therefore removes nothing and is reported as '
                   'vacuous rather than performed silently.' % years))

    # ---- 7. the common eligible mask (S3.6) ------------------------------
    dates = pd.date_range('%d-01-01' % START_YEAR, '%d-12-31' % END_YEAR, freq='D')
    stations = sorted(daily.station_key.unique())
    grid = pd.MultiIndex.from_product([stations, dates], names=['station_key', 'date']).to_frame(index=False)
    grid['observed'] = grid.merge(daily.assign(observed=True), on=['station_key', 'date'],
                                  how='left')['observed'].fillna(False).to_numpy()
    # the model predicts every day of the window for every station in the calendar, so the
    # model side of eligibility is the frozen prediction calendar filtered to year <= 2024.
    cal = pd.read_parquet(PEER / 'data/prediction_calendar.parquet')
    model_stations = set(cal[cal.year.le(END_YEAR)].station_key.unique())
    grid['model_covered'] = grid.station_key.isin(model_stations)
    grid['eligible'] = grid.observed & grid.model_covered
    mask_path = R / 'data/phase1_eligible_mask.parquet'
    grid.to_parquet(mask_path, index=False)
    mask_sha = C.sha(mask_path)
    S['eligible_set'] = dict(
        path=str(mask_path.relative_to(R)), sha256=mask_sha,
        rule=('eligible(station, day) = the station-day MEAN TN exists in the 4h panel AND the '
              'frozen prediction calendar covers that station-day; 2021-01-01..2024-12-31'),
        producer='work/phase0_acceptance_fix.py (Phase 0, i.e. BEFORE any replay exists)',
        n_days=len(dates), n_stations=len(stations), n_cells=int(len(grid)),
        n_eligible=int(grid.eligible.sum()),
        n_eligible_per_station={k: int(v) for k, v in
                                grid[grid.eligible].groupby('station_key').size().items()},
        note=('one mask, computed once, asserted by phase1_fingerprints.py before any criterion; '
              'no arm may drop a row on its own'))
    # the event-level gate: an event enters J1/J3/J4 only if it has an observation
    S['eligible_set']['event_level'] = dict(
        rule=("arm == %s AND obs_status == 'OK'; every event in the lineage has t_start/t_end "
              'year <= 2024 so no 2025 exclusion applies' % PRIMARY_ARM),
        n_events_total=int(len(ev)), n_events_arm=int((ev.arm == PRIMARY_ARM).sum()),
        n_events_eligible=int(len(ok)),
        n_dropped_insufficient=int((ev[ev.arm == PRIMARY_ARM].obs_status == 'INSUFFICIENT_4H_RECORDS').sum()))
    print('eligible mask: %d cells, %d eligible' % (len(grid), int(grid.eligible.sum())), flush=True)

    # ---- 8. the pre-registration document --------------------------------
    md = pre_registration(S, S_M)
    pre_path = R / 'reports/预注册_判据与门槛.md'
    pre_path.parent.mkdir(parents=True, exist_ok=True)
    pre_path.write_text(md, encoding='utf-8')
    pre_sha = C.sha(pre_path)
    S['pre_registration'] = dict(path='reports/预注册_判据与门槛.md', sha256=pre_sha,
                                 written_before_any_replay=True)

    S['phase'] = 'phase0_acceptance_fix'
    S['n_fits'] = 0
    S['fit_worker_calls'] = 0
    S['targets_written'] = False
    C.write_json(R / 'reports/phase0_acceptance.json', S)
    print('PRE_REGISTRATION_SHA', pre_sha, flush=True)
    print('PHASE0_WRITTEN', flush=True)


def pre_registration(S, S_M):
    o = S['observation_side_freeze']
    L = []
    a = L.append
    a('# 预注册：判据、门槛与观测指纹（20260919_1）\n')
    a('> 本文件由 `work/phase0_acceptance_fix.py` 在 **任何回放之前** 写出并哈希。')
    a('> 事后任何改动必须逐条登记在 `实际方法与偏离.md`（`when` / `n_fits_dispatched` / 原文保留）。')
    a('> 冻结 sha256 记录在 `reports/phase0_acceptance.json` 的 `pre_registration.sha256`。\n')
    a('## 0 本轮能裁决的 / 不能裁决的\n')
    a('**能裁决：** `monthfirst` vs within-month uniform redistribution。')
    a('**不能裁决：** 真实源时间。`uniform_daily` 只是"去除月首人工同步"的零假设日历，')
    a('不是真实施肥日历（SWAT+/HYPE 用源特异的施用日期/持续时间/日输入）。\n')
    a('**干预同时改了两条通道**：`structure_model.py` 把 `demand`（作物需求）**与** `source` 一起除以 `nd`。')
    a('⇒ 结论只能表述为"**月内重分配 源与需求**"的合成效应，**不得**写成"只改了源日历"。\n')

    a('## 1 两层验收：两个不同的阈值（不得合并叙述）\n')
    a('| # | 名称 | 形式 | 值 | 状态 |')
    a('|---|---|---|---|---|')
    a('| **F-b** | 优化器**继续条件** | $E_M / S_M \\le 10^{-6}$ | `S_M = %.10f kg` | **已改**（绝对 → 相对）|' % S_M)
    a('| **F-c** | **最终科学验收** | $E_M \\le 10^{-6}\\ \\mathrm{kg}$ | `1e-6 kg` | **未改** |')
    a('')
    a('$$S_M := \\max_{t,\\ r\\in \\text{pilot\\_indices}} \\left|a[\\texttt{M}][t, r]\\right|'
      '\\quad\\text{在冻结的 } \\texttt{%s} \\text{ 参考解上求值}$$' % S['s_m_freeze']['selected_from'])
    a('')
    a('* `pilot_indices = %s`（`len = %d`）。**必须取在同一组 `pilot_indices` 上**：'
      % (S['s_m_freeze']['pilot_indices'], S['s_m_freeze']['n_pilot_indices']))
    a('  `source_label_sum_errors` 的生成式（`campaign_model.py:100-107`）是')
    a('  `np.max(abs(v.sum(-1) - a[name][:, rr]))`，`rr = data.pilot_indices`。')
    a('* 两起点各报一次，**取较小者**（严格方向）：`C0_s1 = %.10f kg`、`C0_s0 = %.10f kg`。'
      % (S['s_m_freeze']['values_kg']['C0_s1'], S['s_m_freeze']['values_kg']['C0_s0']))
    a('* **P / U / 两个起点全部共用这一个标量**；文件中不留任何"二选一"。')
    a('')
    a('### 1.1 必须并列报出的量级后果\n')
    a('$$10^{-6}\\cdot S_M = %.4f\\ \\mathrm{kg}$$' % (1e-6 * S_M))
    a('')
    a('即继续条件比原来的绝对门**宽 %.4e 倍**。' % (1e-6 * S_M / 1e-6))
    a('⇒ **诚实表述**：F-b 之后的继续条件**对 float64 量级的 `M` 残差基本不再阻断**；')
    a('**科学验收的严格性完全由 F-c 的绝对门承担**。绝不可写"`1e-6` 一字未改，因此没有放宽"。\n')
    a('### 1.2 F-b 对历史被拒点的作用（实测）\n')
    a('| 臂 | 被拒点数 | 最大 `M` (kg) | `M / S_M` | F-b 下是否合法 |')
    a('|---|---|---|---|---|')
    for t, v in S['f_b_effect_on_the_rejected_points'].items():
        a('| `%s` | %d | %.6e | %.4e | %s |' % (t, v['n_rejected'], v['max_M_kg'],
                                                v['max_M_over_S_M'], v['would_become_legal']))
    a('')
    a('⇒ 两点：① 被拒点**只**在 `M` 通道越界（其余通道 ≤ 6.9e-8，`local` ≤ 1.9e-7）；')
    a('② 它们在最优点上被拒（最后一行 trace 即最优点，`physical_legal = false`）。')
    a('⇒ 交付的 `objective` 是**最后一个合法保存点**，不是该配置的最优值。')
    a('**把 `2.0168 vs 1.2075` 读成"日历更差"，是在读判据的产物。**\n')
    a('### 1.3 F-a 只作诊断\n')
    a('等价算术重排**已试过且未修复** `M` 残差（`tag_roundoff_diagnostic.json`）：')
    for r in S['f_a_diagnostic_only']['records']:
        a('* `%s`：%.6e → %.6e kg' % (r['tag'], r['M_original'], r['M_subtractive']))
    a('')
    a('⇒ 报告必须写明这一点，否则读者会以为 Phase 0 只是"换个求和方式"。\n')
    a('### 1.4 no-op 证明（硬闸门）\n')
    a('`monthfirst` 两臂在两层判据下读数**必须逐位不变**：`model.ledger(x)` 的六个')
    a('`source_label_sum_errors` 通道与 `outputs/<tag>/source_label_audit.json` **逐通道 float 相等**')
    a('（不是 `allclose`）。实测：')
    for tag, v in S['no_op_proof']['per_tag'].items():
        a('* `%s`：逐通道完全复现 = `%s`，`M = %.6e kg`，旧绝对门 `%s`，新相对门 `%s`'
          % (tag, v['channels_reproduced_exactly'], v['recomputed_M'],
             v['old_absolute_gate_passes'], v['new_relative_gate_passes']))
    a('')

    a('## 2 判据 J1–J5：统一为"与观测指纹的距离"\n')
    a('| # | 判据 | 形式 | 通过条件 |')
    a('|---|---|---|---|')
    a('| **J1** | 事件振幅比 | $G_{\\text{event}} = 1 - \\dfrac{|A_U - A_{\\text{obs}}|}{|A_P - A_{\\text{obs}}|}$ | $G_{\\text{event}} \\ge 0.5$ |')
    a('| **J2** | 站点振幅比 | $e_s = \\left|\\log \\dfrac{SD_{pred,s}}{SD_{obs,s}}\\right|$，$\\Delta e_s = e_{U,s} - e_{P,s}$ | 站点等权中位 $\\Delta e_s < 0$，且配对 station-level permutation 通过 |')
    a('| **J3** | F1 前期积累 | $D_{F1,U} < D_{F1,P}$，其中 $D_{F1} = |\\beta^{sim} - \\beta^{obs}|$ | **距离下降**，不是 $\\beta_U > \\beta_P$ |')
    a('| **J4** | F3 连续事件耗竭 | $D_{F3,U} < D_{F3,P}$，其中 $D_{F3} = |\\alpha^{sim} - \\alpha^{obs}|$ | **距离下降**，不是 $\\alpha_U < \\alpha_P$ |')
    a('| **J5** | 月尺度不损伤 | 月 NSE / $R^2$ 的退化量 | $\\le %.3f$（绝对值）|' % J5_MAX_DEGRADATION)
    a('')
    a('**J3 的模拟侧绝不能用 `obs_tn_base` 去控制**：模型侧回归是')
    a('$\\Delta C^{sim} \\sim T_{interevent} + C^{sim}_{base}$ —— 用**模型自己的**基线，')
    a('否则观测信息被重新塞进模拟指标。J4 同理，判据落在**对数比截距**，**不形成质量**。')
    a('**J2 不看 `SD_pred/SD_obs` 本身** —— 它跨过 1 以后方向会反转。\n')

    a('## 3 数值门槛\n')
    a('* **基线在本谱系自算**（`P` 支即基线），**不得引用 `_5` 的数**。')
    a('  `_5` 参照（仅量级）：`amplitude_underestimated 0.23309`、`p = 0.0`、`null_sd 0.02647`、`null_ci95 ±0.04619`。')
    a('* **噪声带**：任何改善必须超过 $2 \\times$ 零分布 SD $= %.4f$。' % NOISE_BAND)
    a('* **起点散布带**：**日历效应必须大于两起点之间的散布**，否则判 `PARTIAL`（不得 `QUALIFY`）。')
    a('* **J5**：月尺度退化 $\\le %.3f$（绝对值）。' % J5_MAX_DEGRADATION)
    a('* **bootstrap**：`n_boot = %d`，`seed = %d`。' % (N_BOOT, SEED))
    a('  **重采样单位**：J1/J3/J4 为**事件**，按其**所属站点聚类**重采样（站点是可得的最粗依赖；')
    a('  同一站的事件序列相关）。J2 为**站点配对** permutation。**任何情况下都不按行重采样。**')
    a('  同时报出"按事件聚类"作为敏感性读数，判决取**保守组合**（两者须一致）。\n')

    a('## 4 方向预测（**先写死，再看结果**）\n')
    a('本轮预测：$G_{\\text{event}} \\ge 0.5$；$\\Delta e_s$ 中位 $< 0$；$D_{F1}$ 与 $D_{F3}$ **都下降**。\n')

    a('## 5 列名（两个同名 `amplitude_ratio` 必须分开）\n')
    a('| 本轮固定列名 | 定义 | 生产者 | 已知读数 |')
    a('|---|---|---|---|')
    a('| `event_amp_ratio_*`（J1） | 事件级 `TN_peak / TN_base` | `_5/reports/phase1_events.parquet` | 观测中位 **1.31899** vs 模拟 **1.05280** |')
    a('| `station_sd_ratio_*`（J2） | 站点级 sd 家族 | `20260918_1/scripts/hf_metrics.py::metric`（line 3）| `_2/reports/station_metrics.csv`（daily、15 站）：monthfirst 中位 **0.485957** |')
    a('')
    a('两者名字相同、量纲不同、量级差 2–3 倍。报告里每次出现都带完整路径。\n')

    a('## 6 观测指纹（冻结常数，Phase 1 不得重新估计）\n')
    a('主臂 `%s`；次臂 `%s` 仅作敏感性。' % (o['primary_arm'], o['secondary_arm']))
    a('理由：%s。\n' % o['why'])
    a('| 量 | 值 | 出处 / 定义 |')
    a('|---|---|---|')
    a('| $\\beta_{obs}$ | **%.16f** | F1 `T_interevent` 系数，n_units=%d，clusters=%d，p=%.4f |'
      % (o['beta_obs']['value'], o['beta_obs']['n_units'], o['beta_obs']['n_clusters'], o['beta_obs']['p']))
    a('| $\\alpha_{obs}$ | **%.16f** | F3 `alpha_gap_le_30`（谱系注册主变体），n_units=%d，p=%.4f |'
      % (o['alpha_obs']['value'], o['alpha_obs']['n_units'], o['alpha_obs']['p']))
    a('| $A_{obs}$ | **%.16f** | `median(obs_ratio)`，n=%d，臂 `%s`，**本谱系自算** |'
      % (o['A_obs']['value'], o['A_obs']['n'], o['A_obs']['arm']))
    a('| $SD_{obs,s}$ | 见 `phase0_acceptance.json`（15 站逐个） | 站-日**均值** TN 的标准差（`ddof=0`），2021-01-01..2024-12-31 |')
    a('')
    a('$\\beta_{obs}$ / $\\alpha_{obs}$ 取自 `20260918_4/reports/stage_a_fingerprints.json`（本谱系注册件）；')
    a('$A_{obs}$ / $SD_{obs,s}$ 由本文件自算（**纯观测，无模型入环**）。\n')

    a('## 7 共同 eligible set（技术锁 2）\n')
    a('**先算、落盘、哈希，再对 P/U 同时计算；严禁两臂各自掉行。**\n')
    a('* 掩码：`%s`，sha256 `%s`' % (S['eligible_set']['path'], S['eligible_set']['sha256']))
    a('* 规则：%s' % S['eligible_set']['rule'])
    a('* 规模：%d 站 × %d 日 = %d 格，其中 **%d** 格 eligible'
      % (S['eligible_set']['n_stations'], S['eligible_set']['n_days'],
         S['eligible_set']['n_cells'], S['eligible_set']['n_eligible']))
    a('* 事件层：%s' % S['eligible_set']['event_level']['rule'])
    a('  %d 事件中臂内 %d 个，eligible **%d** 个，因 4h 记录不足剔除 **%d** 个。'
      % (S['eligible_set']['event_level']['n_events_total'],
         S['eligible_set']['event_level']['n_events_arm'],
         S['eligible_set']['event_level']['n_events_eligible'],
         S['eligible_set']['event_level']['n_dropped_insufficient']))
    a('* **2025**：%s' % o['note_2025'])
    a('  `max(years) = %d`，被剔除的 2025 事件数 = **%d**。\n'
      % (max(o['event_years_present']), o['n_2025_events_excluded']))

    a('## 8 判决规则\n')
    a('| 结局 | 条件 | 动作 |')
    a('|---|---|---|')
    a('| `QUALIFY_FOR_REFIT` | J1–J5 达标（明显改善且月尺度不损伤） | **不升格为生产日历**；只授权 Phase 2 两日历重新拟合 |')
    a('| `CLOSE` | J1–J5 几乎无作用 | 只关闭窄命题："`monthfirst` artifact **不是当前校准状态下**事件缺陷的重要直接来源"。**不得**外推成"所有 source-calendar 问题全部关闭" |')
    a('| `PARTIAL` | 混合结果 | **不再添加新机制**；逐条登记差多少 |')
    a('| `BLOCKED` | 硬闸门任一失败 | **立即停止，不读任何事件指纹**；登记为装置问题，**不得**当作日历效应的证据 |')
    a('')
    a('`PROMOTE → uniform_daily 升为新的基础源日历` **已被删除**；取代它的是 `QUALIFY_FOR_REFIT`。\n')

    a('## 9 与本轮最大限制\n')
    a('**Phase 1 通过只授权 Phase 2，不是日历有效的证据。**')
    a('固定参数回放的收益可以在重新拟合后消失（日尺度暴露实验的教训：拟合重跑**回吐 +0.0036**）。\n')
    return ''.join(L)


if __name__ == '__main__':
    main()
