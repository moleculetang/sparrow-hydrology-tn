"""Evidence inventory and artifact checks. This never marks the thread goal."""
from pathlib import Path
import sys
import json
import subprocess
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME,atomic_json,sha256,utc_now
import pandas as pd
import numpy as np


def main():
    checks=[];details={}
    nested_result=ROOT/'5_Test/20260905_5/reports/nested_validation_summary.json'
    if nested_result.exists() and json.loads(nested_result.read_text(encoding='utf-8'))['status']=='NESTED_VALIDATION_COMPLETE':
        for name in ['audit_nested_preprocessing.py','audit_nested_selection.py']:
            script=ROOT/'5_Test/20260905_5/scripts'/name
            done=subprocess.run([sys.executable,'-B',str(script)],cwd=ROOT)
            if done.returncode:raise RuntimeError(f'Nested audit failed: {name}')
    temporal_auditor=Path(__file__).with_name('audit_temporal_evidence.py')
    done=subprocess.run([sys.executable,'-B',str(temporal_auditor)],cwd=ROOT)
    if done.returncode:raise RuntimeError('Independent temporal evidence audit failed')
    def check(name,condition,evidence):
        checks.append(dict(requirement=name,passed=bool(condition),evidence=evidence))
    def read(relative):
        path=ROOT/'5_Test'/relative
        if not path.exists():return None
        return json.loads(path.read_text(encoding='utf-8'))
    manifest=read('20260905_1/reports/input_manifest.json')
    check('Frozen parent inputs remain exact',manifest is not None and all(sha256(r['path'])==r['sha256'] for r in manifest['files']),'_1 input_manifest.json; actual SHA256 recomputation')
    for filename,status in [('reference_validation','PASS_REFERENCE_NUMERICS_FITTING_PENDING'),('autograd_validation','PASS_EXACT_DAILY_ADJOINT'),('objective_validation','PASS_ASSEMBLED_OBJECTIVES'),('archived_control_parity','PASS_ARCHIVED_CONTROL_PARITY')]:
        report=read(f'20260905_2/reports/{filename}.json')
        passed=report is not None and report['status']==status
        if report and 'code_sha256' in report:
            hashes=report['code_sha256']
            if isinstance(hashes,dict):passed=passed and all(sha256(p)==h for p,h in hashes.items())
            elif filename=='archived_control_parity':passed=passed and sha256(ROOT/'5_Test/20260905_2/scripts/verify_archived_control.py')==hashes
            else:passed=False
        check(filename,passed,f'_2 reports/{filename}.json plus code hashes where recorded')
    queue=read('20260905_3/reports/development_queue.json');good=0
    if queue:
        for job in queue['jobs']:
            path=Path(job['report'])
            if not path.exists():continue
            r=json.loads(path.read_text(encoding='utf-8'))
            good+=int(r['converged'] and r['projected_gradient_max']<=1e-5 and all(sha256(p)==h for p,h in r['identity']['code_sha256'].items()))
    check('All 140 development physical starts stationary',good==140,dict(stationary=good,required=140))
    dev=read('20260905_3/reports/development_summary.json')
    check('Complete common-cohort development OOF and paired comparisons',dev is not None and dev['status'].startswith('DEVELOPMENT_OOF_COMPLETE') and len(dev.get('summaries',{}))==7,'_3 development_summary.json')
    ext=read('20260905_4/reports/extension_validation.json')
    check('Extension derivative, causality and mass validation',ext is not None and ext['status']=='PASS_EXTENSIONS' and all(sha256(p)==h for p,h in ext['code_sha256'].items()) and all(r.get('future_source_crop_causality') for r in ext['checks']),'_4 extension_validation.json')
    inference=read('20260905_4/reports/inference_only_validation.json')
    check('Process parameters decoded without TN histories or station identifiers',inference is not None and inference['status']=='PASS_INFERENCE_WITHOUT_TN_HISTORY' and all(sha256(p)==h for p,h in inference['code_sha256'].items()) and len(inference['checks'])==4 and all(c['all_TN_history_removed'] for c in inference['checks']),'_4 inference_only_validation.json; observation reads forbidden during independent decoding')
    structural=read('20260905_4/reports/structural_summary.json')
    check('All four registered single structural additions evaluated',structural is not None and structural['status']=='STRUCTURAL_OOF_COMPLETE' and {'uniform_daily','calendar_early','calendar_late','contact_dynamic'}<=set(structural['outcomes']),'_4 structural_summary.json')
    residual=read('20260905_4/reports/new_residual_evidence.json')
    temperature_ready=False
    if residual:
        supported=residual['results']['conditional_air_temperature']['probe_supported']
        decision=read('20260905_4/reports/temperature_mechanism_decision.json')
        temperature_ready=not supported or (decision is not None and decision.get('evidence_review_complete',False) and decision.get('required_work_complete',False))
    check('Temperature conditional evidence resolved',temperature_ready,'_4 new_residual_evidence.json; if supported, reviewed mechanism decision and required experiment evidence')
    splits=read('20260905_5/reports/nested_split_registry.json');nested=read('20260905_5/reports/nested_validation_summary.json')
    required_outer=len(splits['outer_reach'])+len(splits['outer_terminal_tree']) if splits else 0
    split_map={}
    if splits:
        for outer in splits['outer_reach']+splits['outer_terminal_tree']:
            split_map[outer['id']]=outer
            for inner in outer['inner']:split_map[inner['id']]=inner
    check('Nested reach and eligible terminal-tree validation',nested is not None and nested['status']=='NESTED_VALIDATION_COMPLETE' and len(nested['selections'])==required_outer and required_outer>=5 and all(sha256(p)==h for p,h in splits['inputs'].items()),'_5 split registry and completed outer selections')
    preprocessing=read('20260905_5/reports/nested_preprocessing_audit.json')
    check('Independent audit of saved fold-local preprocessing',preprocessing is not None and preprocessing['status']=='PASS_COMPLETE_NESTED_PREPROCESSING' and not preprocessing['errors'] and preprocessing['code_sha256']==sha256(ROOT/'5_Test/20260905_5/scripts/audit_nested_preprocessing.py'),'_5 nested_preprocessing_audit.json; actual saved scalers and variance floors recomputed from training exclusions')
    selection_audit=read('20260905_5/reports/nested_selection_audit.json')
    check('Nested selection traced to actual selected predictions',selection_audit is not None and selection_audit['status']=='PASS_COMPLETE_NESTED_SELECTION' and not selection_audit['errors'] and selection_audit['code_sha256']==sha256(ROOT/'5_Test/20260905_5/scripts/audit_nested_selection.py') and all(sha256(p)==h for p,h in selection_audit['sources'].items()),'_5 nested_selection_audit.json; exact OOF lineage and independently recomputed station metrics')
    # Inspect every selected start set; report existence alone is insufficient.
    selected_counts={};selected_errors=[]
    for stage in ['20260905_4','20260905_5','20260905_6']:
        paths=sorted((ROOT/'5_Test'/stage/'reports').glob('*_selected.json'));selected_counts[stage]=len(paths)
        for path in paths:
            s=json.loads(path.read_text(encoding='utf-8'));rows=s['all_starts']
            try:
                assert len(rows)==5 and {r['start'] for r in rows}==set(range(5))
                best=min(rows,key=lambda r:(r['objective'],r['start']));assert best==s['selected']
                for item in rows:
                    r=json.loads(Path(item['path']).read_text(encoding='utf-8'));identity=r['identity'];spec=identity['spec']
                    assert r['converged'] and r['projected_gradient_max']<=1e-5
                    assert identity['start']==item['start'] and r['objective']==item['objective']
                    assert all(sha256(p)==h for p,h in identity['code_sha256'].items())
                    assert sha256(s['spec_path'])==s['spec_sha256']==identity['spec_sha256']
                    if not spec.get('final_fit',False):assert not set(spec['train_ids'])&set(spec['eval_ids'])
                    assert set(r['design']['training_observation_ids'])==set(spec['train_ids'])
                    if spec.get('evaluation_role')=='nested_spatial':
                        expected=split_map[spec['split_id']]
                        assert set(spec['train_ids'])==set(expected['train_ids'])
                        assert set(spec['eval_ids'])==set(expected['eval_ids'])
                        assert set(spec['heldout_reaches'])==set(expected['heldout_reaches'])
                    output=Path(item['path']).parent.parent/'outputs'/f"{item['tag']}_predictions.parquet"
                    p=pd.read_parquet(output)
                    assert len(p)==len(set(spec['eval_ids'])) and set(p.observation_id)==set(spec['eval_ids'])
                    assert np.isfinite(p.prediction_mg_l).all() and p.prediction_mg_l.ge(0).all()
                    if spec['model']!='CONTROL_H7':assert not any(n.startswith(('site_','gamma_head_','beta_low','beta_high','delta_path')) for n in r['parameters'])
            except Exception as error:selected_errors.append(dict(path=str(path),error=repr(error)))
    check('Five-start selection and actual prediction artifacts verified',not selected_errors and selected_counts.get('20260905_4',0)>=16 and selected_counts.get('20260905_5',0)>=required_outer*22+4 and selected_counts.get('20260905_6',0)>=6,dict(counts=selected_counts,errors=selected_errors))
    prior=read('20260905_5/reports/prior_sensitivity.json');ident=read('20260905_5/reports/identifiability.json')
    check('Prior scales 0.5 and 2, in both registered time folds',prior is not None and {(r['prior_scale'],r['year']) for r in prior['results']}=={(.5,2023),(.5,2024),(2.,2023),(2.,2024)},'_5 prior_sensitivity.json and selected fits')
    check('Parameter boundary, curvature and nuisance profiles',ident is not None and ident['status']=='IDENTIFIABILITY_DIAGNOSTICS_COMPLETE' and len(ident['profiles'])==11 and all(r['converged'] and r['projected_gradient_max']<=1e-5 for r in ident['profiles']),'_5 identifiability.json')
    temporal=read('20260905_6/reports/temporal_confirmation.json')
    check('2024 confirmation and 2025 sensitivity evaluated',temporal is not None and set(temporal)=={'2024','2025'},'_6 temporal_confirmation.json with independently selected starts')
    temporal_evidence=read('20260905_6/reports/temporal_evidence_audit.json')
    check('Independent temporal and prior metric/cohort audit',temporal_evidence is not None and temporal_evidence['status']=='PASS_COMPLETE_TEMPORAL_EVIDENCE' and not temporal_evidence['errors'] and len(temporal_evidence['checked'])==6 and all(sha256(p)==h for p,h in temporal_evidence['code_sha256'].items()) and all(sha256(p)==h for p,h in temporal_evidence['sources'].items()),'_6 temporal_evidence_audit.json; exact temporal exclusions, five-start lineage, excluded-domain scores and independent NSE/log-RMSE/correlation')
    excluded_ready=temporal is not None and set(temporal)=={'2024','2025'}
    if excluded_ready:
        observations=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
        for year in [2024,2025]:
            domains=temporal[str(year)].get('excluded_domains',{})
            for name,column,value in [('reach17_position_unresolved','reach_id',17),('tree163_open_lake','terminal_tree_id',163)]:
                domain=domains.get(name,{})
                expected=set(observations.loc[observations.year.eq(year)&observations[column].eq(value),'observation_id'])
                excluded_ready=excluded_ready and set(domain.get('observation_ids',[]))==expected and domain.get('rows')==len(expected)
                for label in ['new','control']:
                    metrics=domain.get(label)
                    excluded_ready=excluded_ready and (metrics is not None and metrics['rows']==len(expected) if expected else label in domain and metrics is None)
    check('Excluded position and open-lake domains separately reported in confirmation',excluded_ready,'_6 temporal_confirmation.json excluded_domains, exact observation cohorts checked against frozen observations')
    for label,rows in [('f24',176640),('f25',179400)]:
        audit=read(f'20260905_6/reports/{label}_export_audit.json');valid=False
        if audit:
            valid=audit['status']=='PASS_EXPERIMENTAL_FULL_DOMAIN_REPLAY' and audit['monthly_rows']==rows and audit['all_reaches']==230
            valid=valid and all(c['passed'] for c in audit['gradient_checks']) and audit['mass']['full_system_relative']<=1e-10
            for artifact in audit['exports']:
                valid=valid and sha256(artifact['path'])==artifact['sha256']
                if 'reach_monthly' in artifact['path']:
                    frame=pd.read_parquet(artifact['path'])
                    valid=valid and len(frame)==rows and frame.reach_id.nunique()==230 and not frame.duplicated(['year','month','reach_id']).any()
        check(f'{label.upper()} full-domain conservative export with exact artifacts',valid,f'_6 {label}_export_audit.json and re-read export files')
    passed=all(c['passed'] for c in checks)
    result=dict(status='REQUIREMENTS_EVIDENCED_REQUIRES_FINAL_SCIENTIFIC_REVIEW' if passed else 'INCOMPLETE',created_utc=utc_now(),runtime=RUNTIME,checks=checks,
        completed_count=sum(c['passed'] for c in checks),requirement_count=len(checks),promotion=False,
        next_action='Review scope, numerical evidence, validation metrics and write final scientific conclusions; never infer success from status alone' if passed else 'Complete the failed or missing requirements')
    atomic_json(result,ROOT/'5_Test/20260905_6/reports/completion_audit.json')
    print('COMPLETION_AUDIT',result['status'],result['completed_count'],result['requirement_count'],flush=True)
    if passed:
        draft=Path(__file__).with_name('write_final_report.py')
        completed=subprocess.run([sys.executable,'-B',str(draft)],cwd=ROOT)
        if completed.returncode:raise RuntimeError('Scientific report draft generation failed')


if __name__=='__main__':main()
