"""Production stage barriers, without science dispatch or forged acceptance."""
import json
import os
import sys
import tempfile
import hashlib
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import job_lease
import freeze_job
import train_job
from controller_lock import acquire
from d29_platform.runtime import write_json,sha


def rejects(fun,phrase):
    try:fun()
    except RuntimeError as exc:assert phrase in str(exc)
    else:raise AssertionError('production barrier did not reject: '+phrase)


def main():
    checks={};jid='fixture_job'
    with tempfile.TemporaryDirectory(prefix='stage_admission_',dir=ROOT/'cache') as temp:
        base=Path(temp);folder=base/'outputs/jobs'/jid;folder.mkdir(parents=True)
        (base/'config').mkdir();write_json(base/'config/jobs.json',[dict(id=jid)])
        lease=dict(project='20261008_1',job='audit/'+jid,pid=os.getpid(),created='fixture_creation',threads=1)
        with patch.dict(os.environ,{'D29_LEASE_TOKEN':'fixture','D29_WORKER_THREADS':'1'}), \
            patch.object(job_lease.registry,'identity',return_value='fixture_creation'), \
            patch.object(job_lease,'registry_root',return_value=base):
            with patch.object(job_lease.registry,'status',return_value=dict(leases={'fixture':lease})):
                assert job_lease.require(jid,'audit')==lease
                rejects(lambda:job_lease.require(jid,'freeze'),'EXACT_SCIENTIFIC_STAGE')
            checks['stage_job_mismatch_rejected']=True
            with patch.object(job_lease.registry,'status',return_value=dict(leases={'fixture':dict(lease,pid=os.getpid()+1)})):
                rejects(lambda:job_lease.require(jid,'audit'),'EXACT_LIVE_STAGE')
            checks['foreign_pid_rejected']=True
        write_json(folder/'physical_ledger.json',dict(passed=True))
        write_json(folder/'independent_audit.json',dict(objective_passed=True,numerically_sufficient=False,formal_gate_sha256='gate'))
        with patch.object(freeze_job,'ROOT',base),patch.object(freeze_job,'training_only_adapter',return_value=SimpleNamespace(folder=folder)), \
            patch.object(train_job,'validate_admission',return_value=({},'gate')),patch.object(job_lease,'require',return_value=lease):
            rejects(lambda:freeze_job.freeze(jid),'SUFFICIENT_OBJECTIVE_AND_PHYSICAL')
            checks['insufficient_point_cannot_export_frozen_prediction']=True
            write_json(folder/'independent_audit.json',dict(objective_passed=True,numerically_sufficient=True,formal_gate_sha256='old_gate'))
            rejects(lambda:freeze_job.freeze(jid),'AUDIT_FREEZE_GATE_MISMATCH')
            checks['stale_audit_cannot_freeze']=True
        lock=acquire(base/'lifetime.lock')
        rejects(lambda:acquire(base/'lifetime.lock'),'LIFETIME_LOCK_ALREADY_HELD');lock.close()
        lock=acquire(base/'lifetime.lock');lock.close();checks['controller_lifetime_lock_rejects_duplicate']=True
        point=np.array([1.,2.]);physical=dict(parameter_vector_sha256=hashlib.sha256(point.astype('<f8').tobytes()).hexdigest(),year_ledger=[])
        for name,key in [('full_history_physical_source_ledger.npz','full_history_ledger_sha256'),('station_source_ledger.npz','station_source_ledger_sha256')]:
            p=folder/name;p.write_bytes(b'immutable fixture');physical[key]=sha(p)
        for year in range(1961,2025):
            p=folder/'plant_activity_yearly'/f'{year}.npz';p.parent.mkdir(exist_ok=True);p.write_bytes(str(year).encode())
            physical['year_ledger'].append(dict(year=year,plant_activity=dict(reach_land_file=p.relative_to(folder).as_posix(),reach_land_sha256=sha(p))))
        assert len(freeze_job.verify_physical_receipt(folder,point,physical))==66
        rejects(lambda:freeze_job.verify_physical_receipt(folder,point+1.,physical),'PHYSICAL_LEDGER_POINT_MISMATCH')
        checks['physical_ledger_bound_to_actual_frozen_parameter_vector']=True
        (folder/'station_source_ledger.npz').write_bytes(b'changed')
        rejects(lambda:freeze_job.verify_physical_receipt(folder,point,physical),'PHYSICAL_LEDGER_BYTES_CHANGED')
        checks['changed_source_artifact_cannot_be_frozen']=True
    receipt=dict(passed=all(checks.values()),scope='known_error_production_stage_admission_not_physical_export_or_fit',cases=checks,
        scientific_jobs_started=0,identities={p:sha(ROOT/p) for p in ('scripts/job_lease.py','scripts/freeze_job.py','scripts/controller_lock.py')})
    write_json(ROOT/'evidence/stage_admission_known_errors.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
