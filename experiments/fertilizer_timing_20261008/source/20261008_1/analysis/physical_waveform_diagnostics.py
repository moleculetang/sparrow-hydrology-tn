"""Inspect immutable physical/source/plant ledgers without water-quality labels.

Domain sums of incremental local exports are valid; station masses must never
be summed as a basin balance because station supports can be nested. Calendar
centroids and widths are descriptive seasonal shapes, not particle travel times.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from d29_platform.runtime import sha,write_json
from d29_platform.luh3_adapter import LAND_STATES


def shape(values,dates):
    values=np.asarray(values,dtype=np.float64);dates=pd.DatetimeIndex(dates)
    if values.ndim!=1 or len(values)!=len(dates) or not len(values) or not np.isfinite(values).all():
        raise ValueError('FINITE_SAME_DATE_SUPPORT_REQUIRED')
    if not dates.is_unique or not dates.is_monotonic_increasing or len(set(dates.year))!=1:
        raise ValueError('UNIQUE_ORDERED_WITHIN_YEAR_SUPPORT_REQUIRED')
    total=float(values.sum());maximum=float(values.max());minimum=float(values.min())
    out=dict(days=len(values),sum_value=total,mean_value=float(values.mean()),
        std_value=float(values.std(ddof=0)),minimum_value=minimum,maximum_value=maximum,
        p90_minus_p10=float(np.quantile(values,.9)-np.quantile(values,.1)),
        nse_status='not_applicable_physical_shape_or_prediction_only_no_observations')
    resolution=float(8*len(values)*np.finfo(float).eps*np.max(abs(values)))
    out['peak_resolution_value']=resolution
    out['peak_resolution_is_diagnostic_only_no_physical_values_changed']=True
    if maximum>0 and maximum-minimum>resolution:
        ties=np.flatnonzero(values==maximum)
        out.update(peak_date=dates[int(ties[0])].date().isoformat(),peak_tie_count=len(ties),
                   peak_status='resolved_maximum_first_date_with_exact_tie_count')
    else:out.update(peak_date=None,peak_tie_count=len(values) if maximum>0 else None,
                    peak_status='flat_to_float64_summation_resolution' if maximum>0 else 'no_positive_peak')
    if minimum<0:
        out.update(mass_timing_status='negative_remainder_not_clipped_no_mass_quantiles',
                   centroid_day=None,p90_mass_width_days=None)
    elif total==0:
        out.update(mass_timing_status='known_zero_no_mass_timing',centroid_day=None,
                   p90_mass_width_days=None)
    else:
        index=(dates-pd.Timestamp(year=dates[0].year,month=1,day=1)).days.to_numpy(dtype=float)
        cumulative=np.cumsum(values)/total
        lo=int(np.searchsorted(cumulative,.05,side='left'))
        hi=int(np.searchsorted(cumulative,.95,side='left'))
        out.update(mass_timing_status='positive_series_calendar_shape',
                   centroid_origin='Jan1_of_same_calendar_year',
                   centroid_day=float(np.dot(values,index)/total),
                   p90_mass_width_days=float(index[hi]-index[lo]))
    return out


def annual_shapes(series,dates,stage,units,station=None,source=None):
    dates=pd.DatetimeIndex(dates);rows=[]
    for year in sorted(set(dates.year)):
        if not 2016<=year<=2024:continue
        take=dates.year==year
        rows.append(dict(stage=stage,units=units,year=int(year),station_key=station,
                         source=source,sum_value_units='kg N' if units=='kg N/day' else 'm3',
                         **shape(np.asarray(series)[take],dates[take])))
    return rows


def verified(folder):
    physical=json.loads((folder/'physical_ledger.json').read_text(encoding='utf8'))
    replay=folder/'replay.json';freeze=folder/'prediction_freeze.json'
    if replay.exists():
        identity=json.loads(replay.read_text());point=folder/'point.npy'
        if not identity['passed'] or identity.get('evaluation_scores_read') or identity['source_multiplier']!=1.:
            raise RuntimeError('ACCEPTED_LABEL_FREE_REPLAY_REQUIRED')
        if sha(point)!=identity['point_sha256'] or sha(folder/'physical_ledger.json')!=identity['physical_ledger_sha256']:
            raise RuntimeError('REPLAY_POINT_OR_PHYSICAL_RECEIPT_CHANGED')
        kind='fixed_parameter_replay'
    elif freeze.exists():
        identity=json.loads(freeze.read_text());point=folder/'frozen_parameters.npy'
        if not identity['numerically_sufficient'] or not identity['no_evaluation_outcomes_read']:
            raise RuntimeError('SUFFICIENT_FROZEN_PREDICTION_REQUIRED')
        if sha(point)!=identity['parameter_sha256'] or sha(folder/'physical_ledger.json')!=identity['physical_ledger_receipt_sha256']:
            raise RuntimeError('FROZEN_POINT_OR_PHYSICAL_RECEIPT_CHANGED')
        for rel,digest in identity['physical_artifact_sha256'].items():
            p=(folder/rel).resolve()
            if not p.is_relative_to(folder.resolve()) or sha(p)!=digest:
                raise RuntimeError('FROZEN_PHYSICAL_ARTIFACT_CHANGED '+rel)
        kind='sufficient_frozen_fit'
    else:raise RuntimeError('NO_COMPLETED_REPLAY_OR_FROZEN_FIT')
    gate=ROOT/'evidence/formal_gate.json'
    if sha(gate)!=identity['formal_gate_sha256']:
        raise RuntimeError('PHYSICAL_DIAGNOSTIC_FORMAL_GATE_CHANGED')
    if not physical['passed']:raise RuntimeError('PHYSICAL_LEDGER_FAILED')
    raw=np.asarray(np.load(point,allow_pickle=False),dtype='<f8')
    if hashlib.sha256(raw.tobytes()).hexdigest()!=physical['parameter_vector_sha256']:
        raise RuntimeError('PHYSICAL_LEDGER_PARAMETER_VECTOR_CHANGED')
    for name,k in [('full_history_physical_source_ledger.npz','full_history_ledger_sha256'),
                   ('station_source_ledger.npz','station_source_ledger_sha256')]:
        if sha(folder/name)!=physical[k]:raise RuntimeError('SOURCE_LEDGER_BYTES_CHANGED '+name)
    rows=physical['year_ledger']
    if len(rows)!=64 or sorted(r['year'] for r in rows)!=list(range(1961,2025)):
        raise RuntimeError('INCOMPLETE64YEAR_PHYSICAL_LEDGER')
    for row in rows:
        activity=row['plant_activity'];p=(folder/activity['reach_land_file']).resolve()
        if not p.is_relative_to(folder.resolve()) or sha(p)!=activity['reach_land_sha256']:
            raise RuntimeError('PLANT_ACTIVITY_ARTIFACT_CHANGED')
    return physical,identity,kind


def plant_rows(physical,plant_version):
    rows=[];lands=list(LAND_STATES)+['unresolved_surface_ledger']
    for row in physical['year_ledger']:
        a=row['plant_activity']
        for index,land in enumerate(lands):
            planned=np.asarray(a['planned_output_by_land_destination_kg'][index],float)
            actual=np.asarray(a['realized_output_by_land_destination_kg'][index],float)
            unmet=np.asarray(a['unmet_output_by_land_destination_kg'][index],float)
            if not np.isfinite(np.r_[planned,actual,unmet]).all():raise RuntimeError('NONFINITE_PLANT_ACCOUNTING')
            group='cropland' if 5<=index<10 else 'vegetated_noncrop' if index in (0,1,2,3,10,11) else 'passive_surface_ledger'
            denominator=float(planned.sum())
            rows.append(dict(year=int(row['year']),land_class=land,land_group=group,
                planned_total_output_kg=denominator,realized_total_output_kg=float(actual.sum()),
                unmet_total_output_kg=float(unmet.sum()),
                unmet_output_fraction=float(unmet.sum())/denominator if denominator>0 else None,
                external_plant_output_planned_kg=float(planned[0]),
                internal_return_planned_kg=float(planned[1:].sum()),
                external_plant_output_unmet_kg=float(unmet[0]),internal_return_unmet_kg=float(unmet[1:].sum()),
                actual_uptake_kg=a['actual_uptake_by_land_kg'][index],
                potential_capacity_limited_demand_kg=a['potential_daily_demand_with_capacity_by_land_kg'][index],
                supply_shortfall_kg=a['supply_limited_uptake_shortfall_by_land_kg'][index],
                explicit_growth_capacity_kg=a['explicit_growth_capacity_by_land_kg'][index],
                explicit_growth_capacity_applies=bool(group=='cropland' and plant_version=='V1' and row['year']>=2016),
                target_deficit_stock_days_kg_day=a['target_deficit_stock_days_by_land_kg_day'][index],
                end_year_target_deficit_kg=a['end_year_target_deficit_by_land_kg'][index],
                stock_day_deficit_is_harvest_debt=False,nse_status='not_applicable_physical_plant_ledger'))
    return rows


def main(source_folder,destination):
    folder=Path(source_folder).resolve();destination=Path(destination).resolve()
    if not folder.is_relative_to(ROOT.resolve()) or not destination.is_relative_to((ROOT/'outputs').resolve()):
        raise RuntimeError('EXPERIMENT_INPUT_AND_SEPARATE_ANALYSIS_OUTPUT_REQUIRED')
    if destination==folder or destination.is_relative_to(folder):raise RuntimeError('IMMUTABLE_SCIENTIFIC_INPUT_FOLDER')
    physical,identity,kind=verified(folder)
    destination.mkdir(parents=True,exist_ok=True)
    dates=pd.date_range('1961-01-01','2024-12-31');labels=physical['labels'];flows=[]
    with np.load(folder/'full_history_physical_source_ledger.npz',allow_pickle=False) as z:
        for key in ('local_kg_day','local_fast_kg_day','local_slow_kg_day'):
            values=z[key]
            if values.shape!=(len(dates),230):raise RuntimeError('COMPLETE230REACH_HISTORY_REQUIRED')
            flows.extend(annual_shapes(values.sum(axis=1),dates,key,'kg N/day'))
        external=z['external_by_source_kg_day']
        tags=z['local_source_kg_day']
        if external.shape!=(len(dates),len(labels)) or tags.shape!=(len(dates),230,len(labels)):
            raise RuntimeError('SOURCE_HISTORY_IDENTITY')
        for index,label in enumerate(labels):
            flows.extend(annual_shapes(external[:,index],dates,'external_input','kg N/day',source=label))
            flows.extend(annual_shapes(tags[:,:,index].sum(axis=1),dates,'incremental_local_export','kg N/day',source=label))
    station_rows=[]
    with np.load(folder/'station_source_ledger.npz',allow_pickle=False) as z:
        if not np.array_equal(z['labels'],np.asarray(labels,dtype=str)):
            raise RuntimeError('STATION_SOURCE_LABEL_ORDER_CHANGED')
        ti=z['day_index'];keys=z['station_key'];mass=z['total_mass_kg'];water=z['water_m3'];tagmass=z['mass_by_source_kg']
        if ti.ndim!=1 or not np.issubdtype(ti.dtype,np.integer) or (ti<0).any() or (ti>=len(dates)).any():
            raise RuntimeError('INVALID_STATION_DATE_SUPPORT')
        if len(keys)!=len(ti) or mass.shape!=water.shape or mass.shape!=(len(ti),) or tagmass.shape!=(len(ti),len(labels)):
            raise RuntimeError('STATION_SAME_MASS_WATER_SUPPORT_REQUIRED')
        table=pd.DataFrame(dict(station_key=keys,date=dates[ti],mass=mass,water=water))
        if table.duplicated(['station_key','date']).any():raise RuntimeError('DUPLICATE_STATION_DAY_BOUNDARY')
        if not np.isfinite(mass).all() or not np.isfinite(water).all() or (water<0).any():
            raise RuntimeError('INVALID_BOUNDARY_MASS_OR_WATER')
        for station,indices in table.groupby('station_key',sort=True).indices.items():
            positions=np.asarray(indices);order=np.argsort(ti[positions]);positions=positions[order]
            d=dates[ti[positions]]
            station_rows.extend(annual_shapes(mass[positions],d,'OU_boundary_mass','kg N/day',station=station))
            station_rows.extend(annual_shapes(water[positions],d,'OU_boundary_water','m3/day',station=station))
            for index,label in enumerate(labels):
                station_rows.extend(annual_shapes(tagmass[positions,index],d,'OU_boundary_source_mass','kg N/day',station,label))
            for year in range(2016,2025):
                take=positions[dates[ti[positions]].year==year]
                valid=take[water[take]>0]
                if len(valid):
                    descriptor=shape(1000*mass[valid]/water[valid],dates[ti[valid]])
                    descriptor['nse_status']='not_computed_prediction_shape_only_no_evaluation_labels_read'
                    # This is a concentration-weighted calendar shape, not a
                    # conservative mass centroid or a routing transit time.
                    descriptor['mass_timing_status']='positive_concentration_calendar_shape_not_mass_transit'
                    station_rows.append(dict(stage='OU_concentration_prediction_shape',units='mg N/L',year=year,
                        station_key=station,source=None,sum_value_units='mg N day/L_on_positive_water_days',
                        zero_water_days=int(np.sum(water[take]==0)),**descriptor))
    registered=identity.get('registered_job',identity.get('job',folder.name))
    jid=registered['id'] if isinstance(registered,dict) else registered
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']==jid)
    if isinstance(registered,dict) and registered!=job:
        raise RuntimeError('FROZEN_JOB_CONTRACT_CHANGED')
    outputs={}
    for name,rows in [('domain_flux_shapes.csv',flows),('station_boundary_shapes.csv',station_rows),
                      ('plant_activity_annual.csv',plant_rows(physical,job['plant_version']))]:
        p=destination/name;pd.DataFrame(rows).to_csv(p,index=False);outputs[name]=sha(p)
    receipt=dict(passed=True,scope=kind+'_physical_waveform_and_plant_diagnostics_not_TN_accuracy',
        input_folder=str(folder.relative_to(ROOT)),formal_gate_sha256=identity['formal_gate_sha256'],
        physical_receipt_sha256=sha(folder/'physical_ledger.json'),parameter_sha256=identity.get('point_sha256',identity.get('parameter_sha256')),
        source_multiplier=1.,years=[2016,2024],station_count=len(set(keys)),
        observation_data_read=False,model_rerun=False,evaluation_nse_computed=False,
        baseline_improvement_claimed=False,mass_closure_changed=False,negative_values_clipped=False,
        timing_caveats=['centroid is calendar-weighted seasonal shape, not identified transport time',
            'Jan1 year boundary affects cross-year pulses; report this before interpreting phase',
            'station supports can be nested; do not sum their masses as a basin balance',
            'prediction-only concentration shapes do not demonstrate water-quality improvement'],
        output_sha256=outputs,analysis_sha256=sha(Path(__file__)))
    write_json(destination/'complete.json',receipt)
    print(json.dumps(dict(passed=True,scope=kind,station_count=receipt['station_count'],outputs=list(outputs))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source_folder');p.add_argument('destination');a=p.parse_args()
    main(a.source_folder,a.destination)
