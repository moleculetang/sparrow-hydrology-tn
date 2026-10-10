"""Lease-bound full44 evaluation after actual immutable result assembly.

This adds no model selection or physical computation. The existing approved
evaluator remains the sole scoring implementation. Its output bytes and actual
bootstrap rows are checked before signing a resource/evaluation delivery.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis')]
from d29_platform.runtime import sha,write_json
from host_resources import registry,registry_root
from controller_lock import acquire


def verify_outputs(root=ROOT):
    """Verify actual generated files, not only a claim about replicate count."""
    import numpy as np
    import pandas as pd
    evaluation=root/'outputs/evaluation'
    result=json.loads((evaluation/'complete.json').read_text())
    manifest=root/'outputs/campaign_manifest.json'
    if (result.get('passed') is not True or result.get('accounted_paths')!=44 or
            result.get('NH4_DO') is not False or result.get('seed')!=1729 or
            result.get('bootstrap_replicates')!=1000 or
            result.get('selection_manifest_sha256')!=sha(manifest)):
        raise RuntimeError('EXACT_FULL44_EVALUATION_RESULT_REQUIRED')
    selection=json.loads(manifest.read_text())
    if (selection.get('selection_training_only') is not True or
            len(selection.get('accounted_paths',[]))!=44 or len(set(selection['accounted_paths']))!=44 or
            result.get('pairs')!=len(selection.get('pairs',[]))):
        raise RuntimeError('EXACT_FROZEN_TRAINING_ONLY_SELECTION_REQUIRED')
    if result.get('evaluation_label_hashes')!={name:sha(root/'data'/name) for name in
            ('monthly_tn.parquet','hf_daily.parquet','frozen_evaluation_events.parquet')}:
        raise RuntimeError('EVALUATION_LABEL_BYTES_CHANGED')
    summaries=json.loads((evaluation/'summaries.json').read_text())
    expected_bootstrap=set()
    for row in summaries:
        if row['summary'].get('status')=='no_common_support':continue
        job=selection['frozen'][row['candidate']]['job']
        primary=[str(2023 if job['fold']=='F23' else 2024)+'_all'] if job['space_block'] is None else ['2023_held','2024_held']
        if row['period'] not in primary:continue
        for block in (1,2):
            for centered in ((0,1) if row['scope']=='daily' else (0,)):
                name=f"{row['scope']}_{row['period']}_block{block}_centered{centered}.json"
                expected_bootstrap.add(row['candidate']+'__vs__'+row['baseline']+'/'+name)
    actual_bootstrap={p.relative_to(evaluation).as_posix() for p in evaluation.rglob('*_block*_centered*.json')}
    if actual_bootstrap!=expected_bootstrap or not expected_bootstrap:
        raise RuntimeError('COMPLETE_PRIMARY_SCOPE_BOOTSTRAP_ACCOUNTING_REQUIRED')
    bootstrap=[]
    for path in sorted(evaluation.rglob('*_block*_centered*.json')):
        receipt=json.loads(path.read_text());samples=pd.read_csv(path.with_suffix('.csv'))
        months=receipt.get('calendar_months',[]);draws=receipt.get('draw_indices',[])
        if (receipt.get('seed')!=1729 or receipt.get('replicates')!=1000 or
                receipt.get('block_months') not in (1,2) or not months or len(draws)!=1000 or
                len(samples)!=1000 or list(samples.replicate)!=list(range(1000)) or
                samples.block_months.nunique()!=1 or samples.block_months.iloc[0]!=receipt['block_months']):
            raise RuntimeError('ACTUAL1000SYNCHRONOUS_MONTH_DRAWS_REQUIRED '+str(path))
        if any(len(draw)!=len(months) or any(not isinstance(i,int) or not 0<=i<len(months) for i in draw) for draw in draws):
            raise RuntimeError('EXACT_CALENDAR_MONTH_BOOTSTRAP_SUPPORT_REQUIRED')
        partner=path.with_name(path.name.replace('_block1_','_block2_') if receipt['block_months']==1
                               else path.name.replace('_block2_','_block1_'))
        if not partner.exists():raise RuntimeError('BOTH_MONTH_BLOCK_SENSITIVITIES_REQUIRED')
        other=json.loads(partner.read_text())
        if any(other[k]!=receipt[k] for k in ('calendar_months','station_support','metric')):
            raise RuntimeError('SAME_SUPPORT_BLOCK_SENSITIVITY_REQUIRED')
        counts=samples.common_stations.to_numpy(float)
        if not np.isfinite(counts).all() or (counts<0).any() or not np.equal(counts,np.floor(counts)).all():
            raise RuntimeError('DEFINED_BOOTSTRAP_STATION_COUNTS_REQUIRED')
        bootstrap.append(dict(file=path.relative_to(evaluation).as_posix(),replicates=len(samples),
            block_months=receipt['block_months'],station_count=len(receipt['station_support']),
            one_month_support_two_month_sensitivity_limited=len(months)<2))
    if not bootstrap:raise RuntimeError('NO_ACTUAL_BOOTSTRAP_OUTPUTS')
    files={p.relative_to(evaluation).as_posix():sha(p) for p in sorted(evaluation.rglob('*')) if p.is_file()}
    return dict(evaluation_complete_sha256=sha(evaluation/'complete.json'),
        selection_manifest_sha256=sha(manifest),outputs=files,bootstrap=bootstrap,
        actual_paths=44,pairs=result['pairs'],selected_configurations=result['selected_configurations'])


def main():
    if os.name!='nt':raise RuntimeError('LOCAL_APPROVED_EVALUATION_DELIVERY_REQUIRED')
    from approved_epoch_delivery import admitted_folders
    sources=admitted_folders(ROOT)  # Fails before labels, leases or worker creation.
    from assemble_approved_delivery import validate_frozen
    from diagnose_approved_waveforms import approved_gate
    jobs=json.loads((ROOT/'config/jobs.json').read_text())
    support=json.loads((ROOT/'evidence/expected_station_support.json').read_text())
    for job in jobs:validate_frozen(sources[job['id']],job,approved_gate(job),support)
    folder=ROOT/'outputs/approved_evaluation_delivery';folder.mkdir(exist_ok=True)
    lifetime=acquire(folder/'lifetime.lock')
    if (folder/'complete.json').exists():raise RuntimeError('EVALUATION_DELIVERY_ALREADY_COMPLETE_NO_RERUN')
    marker=folder/'owner.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if registry.identity(old['pid'])==old['created']:
            raise RuntimeError('EVALUATION_CONTROLLER_ALREADY_LIVE')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(marker,owner)
    if (ROOT/'outputs/evaluation/complete.json').exists():
        verified=verify_outputs()
        result=dict(passed=True,owner=owner,existing_completed_evaluation_reverified=True,
            new_evaluation_workers=0,**verified)
        write_json(folder/'complete.json',result);print(json.dumps(dict(passed=True,actual_paths=44,new_workers=0)))
        return 0
    if (ROOT/'outputs/evaluation').exists() and any((ROOT/'outputs/evaluation').iterdir()):
        raise RuntimeError('PARTIAL_EVALUATION_REQUIRES_REVIEW_NO_BLIND_REPEAT')
    lease=registry.acquire('20261008_1','analysis/approved_full44_evaluation',
        threads=1,peak_bytes=6500000000,root=registry_root())
    write_json(folder/'resource_decision.json',lease)
    if not lease.get('granted'):
        write_json(folder/'status.json',dict(phase='resource_wait',owner=owner,scientific_signals=0));return 4
    child=None;peak=dict(rss=0,private_commit=0,peak_commit=0);start=time.monotonic()
    try:
        with (folder/'evaluation.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),
                lease['token'],'analysis/evaluate_approved_campaign.py'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            registry.bind(lease['token'],child.pid,root=registry_root())
            write_json(folder/'status.json',dict(phase='approved_full44_evaluation_running',owner=owner,
                child=dict(pid=child.pid,created=registry.identity(child.pid)),lease=lease,scientific_signals=0))
            while child.poll() is None:
                memory=registry.process_memory(child.pid)
                for key in peak:
                    if memory.get(key) is not None:peak[key]=max(peak[key],memory[key])
                time.sleep(.5)
        returned=dict(exit_code=child.returncode,peak=peak,seconds=time.monotonic()-start)
        write_json(folder/'returned.json',returned)
        if child.returncode:
            write_json(folder/'status.json',dict(phase='evaluation_requires_review',owner=owner,**returned));return 2
        verified=verify_outputs()
        result=dict(passed=True,owner=owner,lease=lease,returned=returned,
            source_sha256=sha(Path(__file__)),evaluator_sha256=sha(ROOT/'analysis/evaluate_approved_campaign.py'),
            scientific_signals=0,new_training_calls=0,**verified)
        write_json(folder/'complete.json',result);write_json(folder/'status.json',dict(phase='verified_full44_evaluation_complete',**result))
        print(json.dumps(dict(passed=True,actual_paths=44,pairs=result['pairs'],bootstrap_tables=len(result['bootstrap']))))
        return 0
    finally:
        if child is None or child.poll() is not None:registry.release(lease['token'],root=registry_root())


if __name__=='__main__':sys.exit(main())
