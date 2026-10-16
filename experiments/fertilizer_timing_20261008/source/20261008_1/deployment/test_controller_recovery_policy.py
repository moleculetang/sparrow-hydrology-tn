import copy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from controller_recovery_policy import resolve_record_lease
from d29_platform.runtime import sha,write_json

def main():
    lease=dict(pid=123,created='native',project='20261008_1',job='train/a',
               cpus=[3],threads=1,reserved_bytes=8775000000,gpu=None)
    row=dict(pid=123,created='native',job='a',stage='train',lease=copy.deepcopy(lease))
    assert resolve_record_lease(row,{'exact':lease})['token']=='exact'
    # A controller's local acquire result can retain its parent PID after bind;
    # canonical record pid and the keyed registry must still match the child.
    row['lease']['pid']=999
    assert resolve_record_lease(row,{'exact':lease})['pid']==123
    failures=[]
    for name,alter in (
        ('missing',lambda r,l:l.clear()),
        ('ambiguous',lambda r,l:l.update(second=copy.deepcopy(l['exact']))),
        ('wrong_creation',lambda r,l:l['exact'].update(created='reused')),
        ('wrong_pid',lambda r,l:l['exact'].update(pid=124)),
        ('wrong_project',lambda r,l:l['exact'].update(project='peer')),
        ('wrong_job',lambda r,l:l['exact'].update(job='train/b')),
        ('wrong_stage',lambda r,l:l['exact'].update(job='audit/a')),
        ('wrong_cpu',lambda r,l:l['exact'].update(cpus=[4])),
        ('wrong_threads',lambda r,l:l['exact'].update(threads=2)),
        ('wrong_reserve',lambda r,l:l['exact'].update(reserved_bytes=1)),
        ('wrong_gpu',lambda r,l:l['exact'].update(gpu=0)),
        ('invalid_explicit_token',lambda r,l:r['lease'].update(token='not-exact')),
    ):
        r=copy.deepcopy(row);l={'exact':copy.deepcopy(lease)};alter(r,l)
        try:resolve_record_lease(r,l)
        except RuntimeError as exc:failures.append(dict(control=name,reason=str(exc)))
        else:raise AssertionError(name)
    r=copy.deepcopy(row);r.update(stage='same_point_audit');l=copy.deepcopy(lease);l.update(job='audit/a')
    assert resolve_record_lease(r,{'audit':l})['token']=='audit'
    write_json(ROOT/'evidence/controller_recovery_policy_known_errors.json',dict(passed=True,
        negative_controls=failures,missing_embedded_token_recovered_by_exact_registry_key=True,
        source_sha256=sha(ROOT/'deployment/controller_recovery_policy.py')))
    print('passed: keyed recovery; 12 rejection controls')

if __name__=='__main__':main()
