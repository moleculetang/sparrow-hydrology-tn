"""Output-accounting failures cannot be hidden behind complete.json claims."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
from d29_platform.runtime import sha,write_json
from run_approved_evaluation_delivery import verify_outputs


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def fixture(root):
    out=root/'outputs/evaluation';out.mkdir(parents=True)
    labels=['monthly_tn.parquet','hf_daily.parquet','frozen_evaluation_events.parquet']
    for name in labels:
        path=root/'data'/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'synthetic identity only; no observations')
    ids=[f'test_{i}' for i in range(44)];base,candidate=ids[:2]
    manifest=dict(selection_training_only=True,accounted_paths=ids,pairs=[[base,candidate,'test']],
        frozen={candidate:dict(job=dict(fold='F23',space_block=None))})
    put(root/'outputs/campaign_manifest.json',manifest)
    put(out/'summaries.json',[dict(candidate=candidate,baseline=base,scope='monthly',period='2023_all',summary={})])
    put(out/'complete.json',dict(passed=True,accounted_paths=44,NH4_DO=False,seed=1729,
        bootstrap_replicates=1000,pairs=1,selected_configurations=2,
        selection_manifest_sha256=sha(root/'outputs/campaign_manifest.json'),
        evaluation_label_hashes={n:sha(root/'data'/n) for n in labels}))
    files=[]
    for block in (1,2):
        path=out/(candidate+'__vs__'+base)/f'monthly_2023_all_block{block}_centered0.json'
        put(path,dict(seed=1729,replicates=1000,block_months=block,calendar_months=['2023-01'],
            station_support=['synthetic'],metric='NSE',draw_indices=[[0] for _ in range(1000)]))
        path.with_suffix('.csv').write_text('replicate,common_stations,block_months\n'+
            ''.join(f'{i},1,{block}\n' for i in range(1000)),encoding='utf8')
        files.append(path)
    return out,files


def main():
    controls=[];parent=ROOT/'outputs/approved_evaluation_controls';parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        root=Path(temporary);out,files=fixture(root)
        proof=verify_outputs(root)
        assert len(proof['bootstrap'])==2 and all(r['one_month_support_two_month_sensitivity_limited'] for r in proof['bootstrap'])
        controls.append('synthetic_actual1000_rows_and_short_support_identity')
        result=out/'complete.json';original=json.loads(result.read_text())
        for name,edit in [('wrong_seed',lambda v:v.update(seed=0)),
                          ('NH4_DO_enabled',lambda v:v.update(NH4_DO=True)),
                          ('missing_paths',lambda v:v.update(accounted_paths=43)),
                          ('wrong_selection_hash',lambda v:v.update(selection_manifest_sha256='wrong'))]:
            v=copy.deepcopy(original);edit(v);put(result,v)
            try:verify_outputs(root)
            except RuntimeError:controls.append(name+'_rejected')
            else:raise AssertionError(name)
        put(result,original)
        p=files[0];original_receipt=json.loads(p.read_text())
        for name,edit in [('bad_draw_index',lambda v:v['draw_indices'][0].__setitem__(0,1)),
                          ('missing_draw',lambda v:v['draw_indices'].pop()),
                          ('different_station_support',lambda v:v.update(station_support=['wrong']))]:
            v=copy.deepcopy(original_receipt);edit(v);put(p,v)
            try:verify_outputs(root)
            except RuntimeError:controls.append(name+'_rejected')
            else:raise AssertionError(name)
        put(p,original_receipt)
        csv=p.with_suffix('.csv');original_csv=csv.read_text()
        for name,changed in [('999_rows','\n'.join(original_csv.splitlines()[:-1])+'\n'),
                             ('duplicate_replicate',original_csv.replace('999,1,1','998,1,1')),
                             ('fractional_station_count',original_csv.replace('0,1,1','0,0.5,1',1))]:
            csv.write_text(changed)
            try:verify_outputs(root)
            except RuntimeError:controls.append(name+'_rejected')
            else:raise AssertionError(name)
        csv.write_text(original_csv)
        # Removing both sensitivities for a declared primary scope must fail,
        # even if another scope's complete outputs are present.
        summaries=json.loads((out/'summaries.json').read_text())
        summaries.append(dict(summaries[0],scope='daily'))
        put(out/'summaries.json',summaries)
        try:verify_outputs(root)
        except RuntimeError as exc:
            assert 'COMPLETE_PRIMARY_SCOPE' in str(exc);controls.append('entire_primary_metric_missing_rejected')
        else:raise AssertionError('missing metric hidden')
    run=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'analysis/run_approved_evaluation_delivery.py')],
        capture_output=True,text=True,encoding='utf8',cwd=ROOT,timeout=30)
    assert run.returncode and 'FINAL_CORRECTED44DELIVERY_NOT_READY' in run.stderr
    assert not (ROOT/'outputs/approved_evaluation_delivery').exists()
    controls.append('actual_incomplete44_no_labels_no_resource_dispatch')
    proof=dict(passed=True,controls=controls,synthetic_output_checks=True,actual_evaluations=0,
        actual_guard_returncode=run.returncode,actual_guard_stderr=run.stderr,
        source_sha256=sha(ROOT/'analysis/run_approved_evaluation_delivery.py'),test_sha256=sha(Path(__file__)),
        evaluation_labels_read=False,real_scoring_started=False)
    write_json(ROOT/'evidence/approved_evaluation_delivery_controls.json',proof)
    print(json.dumps(dict(passed=True,controls=len(controls),actual_evaluations=0)))


if __name__=='__main__':main()
