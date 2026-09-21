"""Freeze inputs and audit actual data; never import a legacy runner."""
from __future__ import annotations

from common import ROOT, RUNTIME, atomic_json, atomic_parquet, sha256, utc_now, memory_guard
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RUN = ROOT / '5_Test/20260905_1'
OUT, REPORT = RUN / 'outputs', RUN / 'reports'
HYDRO = {p: ROOT / f'5_Test/{s}/outputs' for p, s in
         [('formal', '20260828_35'), ('sensitivity', '20260828_38')]}
SOURCE = {'formal': ROOT / '5_Test/20260824_12/outputs/monthly_source_forcing_1961_2024.parquet',
          'sensitivity': ROOT / '5_Test/20260904_2/outputs/monthly_source_forcing_1961_2025_sensitivity.parquet'}
OBS = ROOT / '5_Test/20260824_18/outputs/tn_observations_audited.parquet'
POS = ROOT / '5_Test/20260824_12/outputs/tn_observations_primary_2016_2024.parquet'
OBS25 = ROOT / '0_water_quality/data/preprocess/model_ready/tn_station_month_2025_prb_sensitivity.parquet'
TOPO = ROOT / '5_Test/20260814_1/inputs/topology/topology_edges.csv'
FEATURE = ROOT / '5_Test/20260826_15/outputs/multiscale_static_features_raw.parquet'


def freeze_inputs():
    paths = [OBS, POS, OBS25, TOPO, FEATURE, *SOURCE.values(),
             ROOT / '5_Test/20260902_3/reports/feature_registry.json']
    for directory in HYDRO.values():
        paths.extend(directory / f'tn_hydrology_{name}.parquet' for name in
                     ['reach_daily', 'reach_monthly', 'reservoir_daily', 'reservoir_static_metadata'])
    for relative in [
        '20260904_3/scripts/unified_tn_core.py', '20260904_4/scripts/unified_fit.py',
        '20260904_4/scripts/block_refine.py', '20260831_2/scripts/reservoir_tn_core.py',
        '20260824_46/scripts/stage46_models.py', '20260824_41/scripts/l0_v2_core.py',
        '20260904_7/reports/f25_training_report.json',
        '20260904_7/outputs/f25_parameters.parquet',
        '20260904_7/outputs/f25_station_predictions_2016_2025.parquet',
        '20260904_7/outputs/f25_reach_monthly_1961_2025.parquet',
        '20260818_3/reports/wwtp_evidence_decision.json',
        '20260818_3/outputs/wwtp_model_gate_matrix.parquet',
        '20260818_3/outputs/wwtp_performance_metrics.parquet',
        '20260818_5/reports/remaining_process_diagnostic_decision.json',
        '20260831_3/reports/stage3_objective_decision.json']:
        paths.append(ROOT / '5_Test' / relative)
    rows = []
    for path in paths:
        entry = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
        if path.suffix == '.parquet':
            p = pq.ParquetFile(path)
            entry.update(rows=p.metadata.num_rows, columns=p.schema_arrow.names)
        rows.append(entry)
    target = REPORT / 'input_manifest.json'
    if target.exists():
        previous = json.loads(target.read_text(encoding='utf-8'))
        if {r['path']: r['sha256'] for r in previous['files']} != {r['path']: r['sha256'] for r in rows}:
            raise RuntimeError('Frozen parent inputs changed; do not overwrite the manifest')
    else:
        atomic_json({'created_utc': utc_now(), 'files': rows}, target)
    return rows


def observations():
    old = pd.read_parquet(OBS).loc[lambda x: x.formal_river_channel].copy()
    pos = pd.read_parquet(POS)[['station_key', 'reach_id', 'downstream_fraction_on_reach']]
    pos = pos.drop_duplicates()
    if pos.duplicated(['station_key', 'reach_id']).any():
        raise RuntimeError('Inconsistent within-reach station fractions')
    old = old.merge(pos, on=['station_key', 'reach_id'], how='left', validate='many_to_one')
    old['quality_flags'] = '[]'
    old['observation_product'] = 'formal_2016_2024'
    new = pd.read_parquet(OBS25)
    new['observation_product'] = 'sensitivity_2025'
    new['prepared_source_file'] = new.source_file
    new['prepared_source_row'] = pd.NA
    new['source_provenance_tier'] = '2025_product_source_file_and_section_code'
    new['raw_workbook_record_link_status'] = 'see_2025_source_section_code'
    new['quality_flags'] = new.quality_flags.map(lambda v: v if isinstance(v, str) else json.dumps(list(v)))
    columns = ['station', 'station_key', 'reach_id', 'terminal_tree_id', 'year', 'month',
               'tn_mg_l', 'downstream_fraction_on_reach', 'lon', 'lat', 'observation_product',
               'quality_flags', 'prepared_source_file', 'prepared_source_row',
               'source_provenance_tier', 'raw_workbook_record_link_status']
    frame = pd.concat([old[columns], new[columns]], ignore_index=True)
    keys = ['station_key', 'year', 'month']
    assert not frame.duplicated(keys).any()
    assert frame[['reach_id', 'year', 'month', 'tn_mg_l', 'downstream_fraction_on_reach']].notna().all().all()
    assert frame.tn_mg_l.ge(0).all() and np.isfinite(frame.tn_mg_l).all()
    assert frame.downstream_fraction_on_reach.between(0, 1).all()
    frame['primary_gate'] = ~frame.reach_id.eq(17) & ~frame.terminal_tree_id.eq(163)
    frame['primary_exclusion_reason'] = np.select([frame.reach_id.eq(17), frame.terminal_tree_id.eq(163)],
        ['unresolved_Baipenzhu_station_position', 'open_lake_domain'], default='')
    frame = frame.sort_values(keys).reset_index(drop=True)
    frame['observation_id'] = np.arange(len(frame), dtype=np.int64)
    atomic_parquet(frame, OUT / 'observations.parquet')
    folds = []
    for test in [2020, 2021, 2022, 2023, 2024, 2025]:
        role = np.where(frame.year.between(2016, test - 1), 'train',
                        np.where(frame.year.eq(test), 'test', 'outside_fold'))
        item = frame[['observation_id']].copy()
        item['fold_id'], item['role'] = f'T{test}', role
        item['evaluation_use'] = 'development' if test <= 2023 else ('confirmation' if test == 2024 else 'sensitivity')
        folds.append(item)
    atomic_parquet(pd.concat(folds), OUT / 'temporal_folds.parquet')
    # Deterministic allocation based only on observation availability and topology.
    count = frame.loc[frame.year.le(2024)].groupby(['terminal_tree_id', 'reach_id']).size().reset_index(name='n')
    counts, groups = np.zeros(5, dtype=int), []
    for tree, block in count.groupby('terminal_tree_id', sort=True):
        local = np.zeros(5, dtype=int)
        for row in block.sort_values(['n', 'reach_id'], ascending=[False, True]).itertuples(index=False):
            f = min(range(5), key=lambda i: (local[i], counts[i], i))
            groups.append({'reach_id': int(row.reach_id), 'terminal_tree_id': int(tree), 'spatial_fold': f,
                           'observed_months_2016_2024': int(row.n)})
            local[f] += row.n
            counts[f] += row.n
    atomic_parquet(pd.DataFrame(groups), OUT / 'reach_folds.parquet')
    by_year = frame.groupby('year').agg(rows=('observation_id', 'size'), stations=('station_key', 'nunique'),
                                         primary_rows=('primary_gate', 'sum')).reset_index()
    atomic_parquet(by_year, OUT / 'observation_coverage.parquet')
    return {'rows_F24': int(frame.year.le(2024).sum()), 'rows_F25': len(frame),
            'stations': int(frame.station_key.nunique()), 'reaches': int(frame.reach_id.nunique()),
            'primary_F24': int((frame.primary_gate & frame.year.le(2024)).sum()),
            'primary_F25': int(frame.primary_gate.sum()), 'duplicate_station_months': 0,
            'monthly_sample_statistic': 'not_confirmed_no_sampling_day_in_prepared_CSV',
            'same_reach_multiple_stations': int((frame[['reach_id', 'station_key']].drop_duplicates().groupby('reach_id').size()>1).sum()),
            'spatial_fold_observation_counts': counts.tolist()}, frame


def digest_numeric(frame, columns):
    return {c: hashlib.sha256(np.asarray(frame[c], dtype=np.float64).tobytes()).hexdigest() for c in columns}


def hydro_audit(product):
    directory = HYDRO[product]
    m = pd.read_parquet(directory / 'tn_hydrology_reach_monthly.parquet').sort_values(['month', 'reach_id'])
    m['month'] = pd.to_datetime(m.month)
    assert not m.duplicated(['month', 'reach_id']).any()
    count = 768 if product == 'formal' else 780
    assert len(m) == count * 230
    components = ['routed_fast_response_m3_s', 'routed_slow_response_m3_s', 'routed_direct_response_m3_s']
    q = m.routed_total_m3_s.to_numpy()
    monthly_error = float(np.max(np.abs(q - m[components].sum(axis=1))))
    columns = ['date', 'reach_id', *components, 'routed_total_m3_s', 'local_fast_response_m3_s',
               'local_slow_response_m3_s', 'percolation_to_lower_mm_day',
               'upper_response_storage_mm', 'lower_slow_storage_mm', 'tmean_c', 'temperature_semantics']
    d = pd.read_parquet(directory / 'tn_hydrology_reach_daily.parquet', columns=columns).sort_values(['date', 'reach_id'])
    dates = pd.DatetimeIndex(pd.to_datetime(d.date.drop_duplicates()))
    expected = pd.date_range('1961-01-01', '2024-12-31' if product == 'formal' else '2025-12-31')
    assert dates.equals(expected) and len(d) == len(dates) * 230
    assert np.array_equal(d.reach_id.to_numpy().reshape(-1,230), np.broadcast_to(np.arange(1,231),(len(dates),230)))
    day_error = float(np.max(np.abs(d.routed_total_m3_s.to_numpy()-d[components].sum(axis=1))))
    numeric_flow = [*components, 'routed_total_m3_s', 'local_fast_response_m3_s', 'local_slow_response_m3_s',
                    'percolation_to_lower_mm_day', 'upper_response_storage_mm', 'lower_slow_storage_mm']
    assert np.isfinite(d[numeric_flow].to_numpy()).all() and d[numeric_flow].min().min() >= -1e-10
    area = m.loc[m.month.eq(m.month.min())].catchment_area_km2.to_numpy()
    slow = d.local_slow_response_m3_s.to_numpy().reshape(-1,230) * 86.4 / area
    lower = d.lower_slow_storage_mm.to_numpy().reshape(-1,230)
    perc = d.percolation_to_lower_mm_day.to_numpy().reshape(-1,230)
    continuity = lower[1:] - lower[:-1] - perc[1:] + slow[1:]
    exact_available = lower + slow
    legacy_available = np.vstack([lower[:1], lower[:-1]]) + perc
    exact_release = np.divide(slow, exact_available, out=np.zeros_like(slow), where=exact_available>1e-12)
    old_release = np.divide(slow, legacy_available, out=np.zeros_like(slow), where=legacy_available>1e-12)
    monthly_digest = digest_numeric(m.loc[m.month.dt.year.le(2024)], ['routed_total_m3_s', 'local_fast_response_m3_s', 'local_slow_response_m3_s', 'tmean_c'])
    daily_digest = digest_numeric(d.iloc[:23376*230], ['routed_total_m3_s','local_fast_response_m3_s','local_slow_response_m3_s','percolation_to_lower_mm_day','upper_response_storage_mm','lower_slow_storage_mm','tmean_c'])
    result = {'daily_rows':len(d), 'monthly_rows':len(m), 'daily_component_max_error':day_error,
              'monthly_component_max_error':monthly_error, 'zero_monthly_flow_rows':int((q==0).sum()),
              'lower_water_continuity_max_abs_mm':float(np.max(np.abs(continuity))),
              'first_day_old_vs_exact_release_probability_max_abs':float(np.max(np.abs(old_release[0]-exact_release[0]))),
              'after_first_day_probability_max_abs':float(np.max(np.abs(old_release[1:]-exact_release[1:]))),
              'temperature': {'finite_fraction':float(np.isfinite(d.tmean_c).mean()),
                              'min':float(d.tmean_c.min()), 'max':float(d.tmean_c.max()),
                              'semantics':d.temperature_semantics.unique().tolist(),
                              'lineage_status':'requires_forcing_history_review_before_biochemical_candidate'},
              'overlap_monthly_sha256':monthly_digest,'overlap_daily_sha256':daily_digest}
    assert day_error < 1e-9 and monthly_error < 1e-9
    assert np.max(np.abs(continuity)) < 1e-7
    zero = m.loc[q==0, ['month','reach_id']].copy()
    zero['product'] = product
    atomic_parquet(zero, OUT / f'{product}_zero_flow_months.parquet')
    del d, lower, slow, perc, continuity, exact_available, legacy_available, exact_release, old_release
    gc.collect()
    r = pd.read_parquet(directory / 'tn_hydrology_reservoir_daily.parquet')
    denominator = r.storage_m3 + r.total_release_m3
    release_fraction = np.divide(r.total_release_m3, denominator, out=np.zeros(len(r)), where=denominator>1e-12)
    assert np.isfinite(release_fraction).all() and release_fraction.min()>=0 and release_fraction.max()<=1+1e-12
    assert r.storage_m3.min()>=0 and r.total_release_m3.min()>=0
    assert not r.duplicated(['date','reservoir_entity_id']).any()
    inactive_wet = ~r.enabled & r.total_release_m3.gt(0)
    result['reservoir'] = {'rows':len(r),'entities':int(r.reservoir_entity_id.nunique()),
        'release_fraction_min':float(release_fraction.min()),'release_fraction_max':float(release_fraction.max()),
        'inactive_wet_rows':int(inactive_wet.sum()),
        'inactive_wet_fraction_max_deviation_from_passthrough':float(np.max(np.abs(release_fraction[inactive_wet]-1))),
        'inactive_dry_rows':int((~r.enabled & r.total_release_m3.eq(0)).sum()),
        'recorded_mass_error_max_m3':float(r.mass_balance_error_m3.abs().max())}
    result['memory'] = memory_guard()
    return result


def source_audit():
    summary, overlaps = {}, {}
    quantities = ['fertilizer_kg_n','manure_kg_n','cropland_bnf_kg_n','atmospheric_deposition_kg_n','crop_demand_kg_n']
    for product,path in SOURCE.items():
        f = pd.read_parquet(path)
        f = f.loc[f.calendar_scenario.eq('CENTRAL')].sort_values(['year','month','reach_id'])
        assert not f.duplicated(['year','month','reach_id']).any()
        assert f[quantities].notna().all().all() and (f[quantities]>=0).all().all()
        summary[product]={'rows':len(f),'year_min':int(f.year.min()),'year_max':int(f.year.max()),
                          'source_and_demand_totals_kg_n':f[quantities].sum().to_dict(),
                          'month_demand_gt_current_input_count':int((f.crop_demand_kg_n>f[quantities[:-1]].sum(axis=1)).sum()),
                          'negative_missing_amount_count':0}
        overlaps[product]=digest_numeric(f.loc[f.year.le(2024)],quantities)
    summary['formal_sensitivity_overlap_exact'] = overlaps['formal']==overlaps['sensitivity']
    return summary


def code_and_history_audit():
    fit = (ROOT/'5_Test/20260904_4/scripts/unified_fit.py').read_text(encoding='utf-8')
    carrier = (ROOT/'5_Test/20260831_2/scripts/reservoir_tn_core.py').read_text(encoding='utf-8')
    core = (ROOT/'5_Test/20260904_3/scripts/unified_tn_core.py').read_text(encoding='utf-8')
    findings = [
        {'id':'crossfold_start','confirmed': 'h22_transfer_head_t3_gamma.parquet' in fit,
         'severity':'must_repair','meaning':'same gamma_start path can seed earlier temporal folds; new starts must be fold-local'},
        {'id':'fixed_vf_kkt','confirmed':'torch.tensor(self.fixed_v_f' in carrier,
         'severity':'must_repair','meaning':'fixed-vf raw gradient does not verify physical-coordinate stationarity'},
        {'id':'monthly_redistribution','confirmed':'fast_layers[:, month] * self.fast_day_weight' in carrier,
         'severity':'quantify_reference_discrepancy','meaning':'monthly load is redistributed by water; actual daily mobilization timing is not retained'},
        {'id':'first_day_lower_available','confirmed':'lower_end[:1], lower_end[:-1]' in core,
         'severity':'repair_quantify_small_effect','meaning':'first-day denominator should use lower_end+slow release'},
        {'id':'predam_water_contract','confirmed':'outlet_water[:, reach] = pre_dam' in core,
         'severity':'boundary_audit_not_yet_bug','meaning':'internal pre-dam volume differs from official reach outflow; compare matching control volumes'},
        {'id':'gamma_prior_dimension','confirmed':'torch.mean((gamma / GAMMA_PRIOR_SD).square())' in fit,
         'severity':'declare_prior_semantics','meaning':'mean-square penalty is not sum of independent Gaussian coefficient penalties'}]
    assert all(x['confirmed'] for x in findings)
    atomic_json(findings, REPORT/'code_findings.json')
    g=pd.read_parquet(ROOT/'5_Test/20260818_3/outputs/wwtp_model_gate_matrix.parquet')
    decision=json.loads((ROOT/'5_Test/20260818_3/reports/wwtp_evidence_decision.json').read_text(encoding='utf-8'))
    decision['checked_against_gate_table']={'models':len(g),'P2_station_delta_median':float(g.P2_station_point_delta.median()),
        'P1_station_delta_median':float(g.P1_station_point_delta.median()),
        'clear_failure_count':int(g.clear_failure.sum()),'station_effect_shrunk_count':int(g.station_effect_shrunk.sum()),
        'P1_LOTO_point_delta_median':float(g.P1_LOTO_point_delta.median())}
    atomic_json(decision,REPORT/'point_source_history_recheck.json')
    return findings


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    started=utc_now()
    files=freeze_inputs()
    print('INPUT_HASHES_FROZEN',len(files),flush=True)
    obs,frame=observations()
    print('OBSERVATIONS',json.dumps(obs),flush=True)
    hydro={}
    for product in HYDRO:
        hydro[product]=hydro_audit(product)
        print('HYDRO_AUDITED',product,json.dumps(hydro[product]),flush=True)
    overlap={level:hydro['formal'][f'overlap_{level}_sha256']==hydro['sensitivity'][f'overlap_{level}_sha256'] for level in ['daily','monthly']}
    source=source_audit()
    findings=code_and_history_audit()
    features=json.loads((ROOT/'5_Test/20260902_3/reports/feature_registry.json').read_text(encoding='utf-8'))['H22_features'][:7]
    x=pd.read_parquet(FEATURE,columns=['reach_id',*features]).sort_values('reach_id')
    assert np.array_equal(x.reach_id,np.arange(1,231)) and np.isfinite(x[features]).all().all()
    atomic_parquet(x,OUT/'h7_raw_features.parquet')
    # Re-hash only metadata/code files here. Large frozen data are checked again by consumers.
    for entry in files:
        if entry['bytes']<10*1024*1024:
            assert sha256(entry['path'])==entry['sha256']
    result={'stage':'20260905_1','status':'INPUT_AUDIT_COMPLETE_NUMERICAL_REPAIRS_PENDING',
            'started_utc':started,'finished_utc':utc_now(),'runtime':RUNTIME,
            'observations':obs,'hydrology':hydro,'hydrology_overlap_exact':overlap,'sources':source,
            'confirmed_code_findings':[f['id'] for f in findings],'h7_fields':features,
            'temperature_update':'full interface contains nonconstant air temperature; provenance must be checked before declaring temperature unavailable or fit-ready',
            'pending':['daily_N_reference_and_full_ledger','vf_physical_gradient_profile','matching_water_N_control_volumes',
                       'old_structure_refit','objective_capacity_factorial','conditional_structure_trials','nested_spatial_validation','F24_F25_fits_and_exports'],
            'memory':memory_guard()}
    atomic_json(result,REPORT/'input_audit.json')
    print('AUDIT_COMPLETE',str(REPORT/'input_audit.json'),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        atomic_json({'status':'FAILED','at_utc':utc_now(),'type':type(exc).__name__,'message':str(exc)},REPORT/'last_error.json')
        raise
