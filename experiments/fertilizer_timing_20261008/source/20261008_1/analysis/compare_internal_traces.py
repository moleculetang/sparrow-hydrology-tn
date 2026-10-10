"""Same-point layered seasonal contrasts; no water-quality accuracy inference."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from d29_platform.runtime import sha,write_json


def main():
    proofpath=ROOT/'outputs/replay_internal_traces/complete.json'
    proof=json.loads(proofpath.read_text())
    if not proof['passed'] or proof['count']!=12:raise RuntimeError('ALL12INTERNAL_TRACES_REQUIRED')
    annotationspath=ROOT/'outputs/replay_internal_traces/annotated_shapes_v2/complete.json'
    annotations=json.loads(annotationspath.read_text())
    if not annotations['passed'] or annotations['count']!=12 or annotations['source_complete_sha256']!=sha(proofpath):
        raise RuntimeError('CURRENT12UNIT_ANNOTATIONS_REQUIRED')
    corrected={c['job']:c for c in annotations['cases']}
    cases={}
    for case in proof['completed']:
        path=ROOT/case['receipt']
        if sha(path)!=case['sha256']:raise RuntimeError('COMPLETED_TRACE_CHANGED')
        result=json.loads(path.read_text());job=result['registered_job']
        if not result['passed'] or result['source_point_sha256']!=json.loads((ROOT/result['source_folder']/'replay.json').read_text())['point_sha256']:
            raise RuntimeError('ACCEPTED_TRACE_POINT_CHANGED')
        shapes=path.parent/'internal_and_network_shapes.csv'
        if sha(shapes)!=result['shapes_sha256']:raise RuntimeError('TRACE_SHAPES_CHANGED')
        annotation=corrected[case['job']]
        if annotation['source_receipt_sha256']!=case['sha256'] or annotation['original_shapes_sha256']!=sha(shapes):
            raise RuntimeError('UNIT_ANNOTATION_SOURCE_CHANGED')
        shapes=ROOT/annotation['annotated_file']
        if sha(shapes)!=annotation['annotated_sha256']:raise RuntimeError('ANNOTATED_SHAPES_CHANGED')
        frame=pd.read_csv(shapes,dtype={'source':'string'});frame['source']=frame.source.fillna('domain')
        if frame.duplicated(['stage','year','source']).any():raise RuntimeError('DUPLICATE_STAGE_YEAR_SOURCE')
        cases[(job['fold'],job['plant_version'],job['calendar'])]=(result,frame)
    rows=[];pairs=[]
    for fold in ('F23','F24'):
        for plant in ('V0','V1'):
            for b,c,meaning in [('C0','C1','intermonth_information'),('C1','C2','intramonth_same_monthly_input_mass'),('C0','C2','complete_fixed_timing')]:
                baseline,bf=cases[(fold,plant,b)];candidate,cf=cases[(fold,plant,c)]
                if baseline['source_point_sha256']!=candidate['source_point_sha256']:
                    raise RuntimeError('NONMATCHED_PHYSICAL_PARAMETER_POINT')
                joint=bf.merge(cf,on=['stage','year','source'],suffixes=('_base','_candidate'),how='outer',indicator=True,validate='one_to_one')
                if not joint._merge.eq('both').all():raise RuntimeError('DIFFERENT_INTERNAL_TRACE_SUPPORT')
                for r in joint.to_dict('records'):
                    out=dict(fold=fold,plant_version=plant,baseline=b,candidate=c,comparison=meaning,
                        stage=r['stage'],year=r['year'],source=r['source'],parameters_fixed=True,
                        nse_status='not_applicable_physical_transfer_no_observed_TN',
                        seasonal_centroid_is_particle_travel_time=False)
                    for field in ('units','sum_value_units','quantity_kind','annual_sum_is_external_mass','annual_sum_meaning'):
                        if r[field+'_base']!=r[field+'_candidate']:raise RuntimeError('BASE_CANDIDATE_UNIT_OR_SEMANTIC_MISMATCH')
                        out[field]=r[field+'_base']
                    for name in ('sum_value','std_value','p90_minus_p10','centroid_day','p90_mass_width_days'):
                        v,w=r[name+'_base'],r[name+'_candidate']
                        out[name+'_base']=v;out[name+'_candidate']=w
                        out[name+'_difference']=w-v if pd.notna(v) and pd.notna(w) else np.nan
                        if name in ('std_value','p90_minus_p10'):
                            out[name+'_ratio']=w/v if pd.notna(v) and pd.notna(w) and v!=0 else np.nan
                    out['peak_date_base']=r['peak_date_base'];out['peak_date_candidate']=r['peak_date_candidate']
                    out['peak_status_base']=r['peak_status_base'];out['peak_status_candidate']=r['peak_status_candidate']
                    rows.append(out)
                pairs.append(dict(fold=fold,plant_version=plant,baseline=b,candidate=c,comparison=meaning,
                    point_sha256=baseline['source_point_sha256'],rows=len(joint)))
    folder=ROOT/'outputs/replay_internal_traces';csv=folder/'paired_internal_and_network_shapes_v2.csv'
    pd.DataFrame(rows).to_csv(csv,index=False)
    write_json(folder/'paired_complete_v2.json',dict(passed=True,pairs=pairs,count=len(pairs),rows=len(rows),
        source_complete_sha256=sha(proofpath),csv_sha256=sha(csv),
        unit_annotation_receipt_sha256=sha(annotationspath),
        observations_read=False,conditional_parameter_replay_not_calibration_effect=True,
        comparator_sha256=sha(Path(__file__))))
    print(json.dumps(dict(passed=True,pairs=len(pairs),rows=len(rows))))


if __name__=='__main__':main()
