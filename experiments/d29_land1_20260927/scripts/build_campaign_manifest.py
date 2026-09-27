"""Select only independently accepted training points, never load evaluation labels.

Explicit exclusions are a JSON mapping path id -> documented reason. Missing or
live paths are never silently excluded. Predictions must already be frozen.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha
from d29_training.campaign import select_campaign


def build(exclusions_path):
    jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8'))
    exclusions=json.loads(exclusions_path.read_text(encoding='utf-8'))
    records={}
    for job in jobs:
        name=job['id']
        if name in exclusions:continue
        folder=ROOT/'outputs/jobs'/name
        status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
        audit=json.loads((folder/'independent_audit.json').read_text(encoding='utf-8'))
        ledger=json.loads((folder/'physical_ledger.json').read_text(encoding='utf-8'))
        digest=sha(folder/'best.npy')
        if digest!=audit['checkpoint_sha256'] or digest!=ledger['parameter_sha256']:
            raise RuntimeError('AUDIT_CHECKPOINT_MISMATCH '+name)
        value=audit['objective']
        if abs(value-status['best_objective'])>1e-8*(1+abs(value)):
            raise RuntimeError('STATUS_OBJECTIVE_MISMATCH '+name)
        records[name]=dict(status=status['status'],objective=value,
            objective_passed=audit['objective_passed'],physical_passed=ledger['passed'],
            projected_gradient=audit['projected_gradient'],
            numerically_sufficient=audit['numerically_sufficient'],
            checkpoint_sha256=digest,audit_sha256=sha(folder/'independent_audit.json'),
            ledger_sha256=sha(folder/'physical_ledger.json'))
    result=select_campaign(jobs,records,exclusions)
    for name in result['selected_jobs']:
        folder=ROOT/'outputs/jobs'/name
        freeze=json.loads((folder/'prediction_freeze.json').read_text(encoding='utf-8'))
        if freeze['parameter_sha256']!=records[name]['checkpoint_sha256']:
            raise RuntimeError('FROZEN_PARAMETER_MISMATCH '+name)
        for f,key in [('frozen_station_days.parquet','days_sha256'),('frozen_station_months.parquet','months_sha256')]:
            if sha(folder/f)!=freeze[key]:raise RuntimeError('PREDICTION_CHANGED '+name)
    result['exclusion_register_sha256']=sha(exclusions_path)
    result['jobs_sha256']=sha(ROOT/'config/jobs.json')
    result['event_support_contract_sha256']=sha(ROOT/'data/event_support_contract.json')
    target=ROOT/'outputs/campaign_manifest.json'
    if target.exists():raise RuntimeError('MANIFEST_ALREADY_FROZEN')
    write_json(target,result)
    print('Frozen selected configurations:',len(result['selected_jobs']))


if __name__=='__main__':build(Path(sys.argv[1]))
