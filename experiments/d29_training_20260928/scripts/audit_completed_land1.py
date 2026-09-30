"""Finite post-fit LAND1 audits and prediction freezing, no evaluation labels."""
import sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,dispatch_allowed,write_json,sha
configure()


def main():
    jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8'))
    for j in jobs:
        if j['model']!='LAND1':continue
        folder=ROOT/'outputs/jobs'/j['id'];status=folder/'status.json'
        if not status.exists() or not (folder/'best.npy').exists():continue
        s=json.loads(status.read_text(encoding='utf-8'))
        if s['status'] in ('running','resource_checkpoint','continuing_same_path_zero_ftol'):continue
        for script,receipt in [('audit_land1_solution.py','training_block_gradients.json'),('verify_land1_ledger.py','physical_ledger.json'),('freeze_land1_predictions.py','prediction_freeze.json')]:
            if (folder/receipt).exists():continue
            if script=='freeze_land1_predictions.py':
                audit=json.loads((folder/'independent_audit.json').read_text(encoding='utf-8'))
                physical=json.loads((folder/'physical_ledger.json').read_text(encoding='utf-8'))
                digest=sha(folder/'best.npy')
                if not audit['objective_passed'] or not physical['passed'] or digest!=audit['checkpoint_sha256'] or digest!=physical['parameter_sha256']:
                    raise RuntimeError('UNACCEPTED_OR_STALE_LAND1_AUDIT')
            ok,res=dispatch_allowed(reserve_bytes=9_000_000_000)
            if not ok:
                write_json(ROOT/'outputs/land1_audit_pause.json',dict(job=j['id'],script=script,resources=res));return
            with (folder/'audit_console.log').open('ab') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts'/script),j['id']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(j['id'],receipt,flush=True)


if __name__=='__main__':main()
