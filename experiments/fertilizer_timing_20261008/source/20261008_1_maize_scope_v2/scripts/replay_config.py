"""Same-parameter fixed-calendar replay before any fitting or heldout scoring."""
import json
import time
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from timing.adapter import TimingTraining
from train_job import validate_admission
from source_ledger import ledger
from d29_platform.runtime import sha,write_json


def main(jid):
    _,gate_hash=validate_admission()
    from job_lease import require
    require(jid,'replay');start=time.monotonic()
    job=next(j.copy() for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']==jid)
    if job['space_block'] is not None or job['calendar']=='K' or job['entry']!=0:
        raise ValueError('TWELVE_REGISTERED_FIXED_TIME_REPLAYS_ONLY')
    job['id']='replay_'+jid;a=TimingTraining(job);point=a.initial.copy()
    full,_=a.unpack(point)
    if full[a.eta_index]!=0.:raise RuntimeError('REPLAY_MULTIPLIER_NOT_ONE')
    np.save(a.folder/'point.npy',point)
    physical=ledger(a,point)
    np.save(a.folder/'training_support_prediction.npy',a.prediction)
    from host_resources import registry
    if os.name=='posix':
        import resource
        peak=dict(basis='Linux_ru_maxrss',peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    else:peak=dict(basis='Windows_peak_private_commit',**registry.process_memory(os.getpid()))
    result=dict(passed=physical['passed'],scope='full1961_2024_same_parameter_fixed_calendar_replay_not_fit_or_heldout_score',
        registered_job=jid,calendar=job['calendar'],plant_version=job['plant_version'],fold=job['fold'],
        point_sha256=sha(a.folder/'point.npy'),source_multiplier=1.,formal_gate_sha256=gate_hash,
        physical_ledger_sha256=sha(a.folder/'physical_ledger.json'),prediction_sha256=sha(a.folder/'training_support_prediction.npy'),
        training_objective_only=physical['objective'],evaluation_scores_read=False,seconds=time.monotonic()-start,native_peak=peak,
        identities={p:sha(ROOT/p) for p in ('scripts/replay_config.py','scripts/source_ledger.py')})
    write_json(a.folder/'replay.json',result)
    print(json.dumps(result));return 0 if physical['passed'] else 2


if __name__=='__main__':sys.exit(main(sys.argv[1]))
