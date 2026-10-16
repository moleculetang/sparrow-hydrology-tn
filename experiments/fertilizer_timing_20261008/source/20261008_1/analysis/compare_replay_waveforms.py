"""Same-point physical calendar contrasts; no observation or accuracy claims."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from d29_platform.runtime import write_json,sha


def main():
    folder=ROOT/'outputs/physical_shape_diagnostics_v3'
    proof=json.loads((folder/'complete.json').read_text())
    if not proof['passed'] or proof['count']!=12:raise RuntimeError('COMPLETE12REPLAY_DIAGNOSTICS_REQUIRED')
    cases={r['job']:r for r in proof['cases']};data={};identities={}
    for jid,r in cases.items():
        p=ROOT/r['receipt']
        if sha(p)!=r['sha256']:raise RuntimeError('REPLAY_SHAPE_RECEIPT_CHANGED')
        meta=json.loads(p.read_text())
        for name,digest in meta['output_sha256'].items():
            if sha(p.parent/name)!=digest:raise RuntimeError('REPLAY_SHAPE_DATA_CHANGED')
        data[jid]={n:pd.read_csv(p.parent/n) for n in meta['output_sha256']}
        identities[jid]=meta
    flux=[];station=[];debt=[]
    metrics=['sum_value','mean_value','std_value','p90_minus_p10','centroid_day','p90_mass_width_days']
    for fold in ('F23','F24'):
        for plant in ('V0','V1'):
            for before,after,contrast in [('C0','C1','intermonth_fixed_parameter'),
                                          ('C1','C2','intramonth_fixed_parameter'),
                                          ('C0','C2','complete_timing_fixed_parameter')]:
                base=f'TIM_{fold}_{before}_{plant}_s0';candidate=f'TIM_{fold}_{after}_{plant}_s0'
                if identities[base]['parameter_sha256']!=identities[candidate]['parameter_sha256']:
                    raise RuntimeError('FIXED_PARAMETER_CONTRAST_POINT_CHANGED')
                common=dict(baseline=base,candidate=candidate,contrast=contrast,model_parameters_fixed=True,
                            nse_status='not_applicable_physical_contrast_no_observations_read')
                for filename,target,keys in [('domain_flux_shapes.csv',flux,['stage','year','source']),
                                             ('station_boundary_shapes.csv',station,['stage','year','source','station_key'])]:
                    a=data[base][filename].copy();b=data[candidate][filename].copy()
                    a['source']=a.source.fillna('all_sources');b['source']=b.source.fillna('all_sources')
                    if a.duplicated(keys).any() or b.duplicated(keys).any():raise RuntimeError('DUPLICATE_SHAPE_SUPPORT')
                    joined=a.merge(b,on=keys,how='outer',suffixes=('_base','_candidate'),indicator=True,validate='one_to_one')
                    if not joined['_merge'].eq('both').all():raise RuntimeError('DIFFERENT_PHYSICAL_CONTRAST_SUPPORT')
                    for row in joined.to_dict('records'):
                        result=dict(common,**{k:row[k] for k in keys})
                        for metric in metrics:
                            x=row[metric+'_base'];y=row[metric+'_candidate']
                            result[metric+'_baseline']=float(x) if pd.notna(x) else None
                            result[metric+'_candidate']=float(y) if pd.notna(y) else None
                            result[metric+'_difference']=float(y-x) if pd.notna(x) and pd.notna(y) else None
                        result.update(peak_date_baseline=row['peak_date_base'],peak_date_candidate=row['peak_date_candidate'],
                            peak_status_baseline=row['peak_status_base'],peak_status_candidate=row['peak_status_candidate'])
                        target.append(result)
                def grouped(jid):
                    p=data[jid]['plant_activity_annual.csv']
                    p=p[p.year.between(2016,2024)].copy()
                    return p.groupby(['year','land_group'],as_index=False)[['planned_total_output_kg',
                        'realized_total_output_kg','unmet_total_output_kg','supply_shortfall_kg']].sum()
                a=grouped(base);b=grouped(candidate)
                joined=a.merge(b,on=['year','land_group'],suffixes=('_base','_candidate'),validate='one_to_one')
                for row in joined.to_dict('records'):
                    denominator=row['planned_total_output_kg_base'];other=row['planned_total_output_kg_candidate']
                    same=abs(denominator-other)<=1e-6
                    fraction=lambda which: row['unmet_total_output_kg_'+which]/row['planned_total_output_kg_'+which] if row['planned_total_output_kg_'+which]>0 else None
                    x,y=fraction('base'),fraction('candidate')
                    difference=100*(y-x) if x is not None and y is not None else None
                    debt.append(dict(common,year=row['year'],land_group=row['land_group'],same_planned_output=same,
                        planned_output_baseline_kg=denominator,planned_output_candidate_kg=other,
                        unmet_output_fraction_baseline=x,unmet_output_fraction_candidate=y,
                        unmet_output_difference_percentage_points=difference,
                        one_percentage_point_warning=bool(same and difference>1) if difference is not None else None,
                        warning_scope='conditional_fixed_point_not_final_calibrated_upgrade_decision',
                        supply_shortfall_baseline_kg=row['supply_shortfall_kg_base'],
                        supply_shortfall_candidate_kg=row['supply_shortfall_kg_candidate']))
    paths={}
    for name,rows in [('paired_domain_shapes.csv',flux),('paired_station_shapes.csv',station),('paired_plant_debt.csv',debt)]:
        p=folder/name;pd.DataFrame(rows).to_csv(p,index=False);paths[name]=sha(p)
    write_json(folder/'paired_complete.json',dict(passed=True,contrasts=12,observation_data_read=False,
        model_rerun=False,source_multiplier=1.,scientific_accuracy_improvement_claimed=False,
        shape_analysis_complete_sha256=sha(folder/'complete.json'),output_sha256=paths,
        comparator_sha256=sha(Path(__file__)),
        interpretation='calendar phase/width are seasonal descriptors; no identified particle travel times; all-site predictions are not NSE scores'))
    print(json.dumps(dict(passed=True,contrasts=12,domain_rows=len(flux),station_rows=len(station),plant_rows=len(debt))))


if __name__=='__main__':main()
