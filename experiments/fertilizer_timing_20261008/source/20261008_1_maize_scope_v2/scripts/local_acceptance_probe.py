"""First local full-history point; no optimization and no heldout label reads."""
import sys,time,json,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha
import numpy as np
from timing.adapter import TimingTraining

if __name__=='__main__':
    jid=sys.argv[1] if len(sys.argv)>1 else 'TIM_F23_C0_V0_s0'
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']==jid)
    a=TimingTraining(job,acceptance=True);start=time.monotonic()
    v,g=a.value_gradient(a.initial)
    np.save(a.folder/'acceptance_point.npy',a.initial);np.save(a.folder/'acceptance_gradient.npy',g)
    receipt=dict(a.last)
    receipt.update(host=platform.node(),names=a.names,
        prediction_records=len(a.prediction),objective=v,gradient_max=float(np.max(abs(g))),
        total_seconds=time.monotonic()-start,full_history_years=[1961,2024],source_multiplier=1.,
        formal_gate_passed=False,scope='initial_full_history_oracle_only')
    write_json(a.folder/'local_probe.json',receipt)
    print(json.dumps(a.last))
