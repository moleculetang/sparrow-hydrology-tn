"""Concurrency, hash refusal, restart accounting, and actual label barriers."""
import sys,subprocess,json,time,os
from pathlib import Path
import native_runtime as rt
R=rt.RUN

def main():
    testname='append_test_'+str(time.time_ns())+'.jsonl'
    code="import native_runtime as r,sys;[r.log(sys.argv[1],dict(worker=sys.argv[2],row=i)) for i in range(100)]"
    children=[subprocess.Popen([sys.executable,'-B','-c',code,testname,str(i)],cwd=R/'scripts',creationflags=subprocess.CREATE_NO_WINDOW) for i in range(8)]
    assert all(p.wait()==0 for p in children)
    records=[json.loads(s) for s in (R/'work'/testname).read_text().splitlines()]
    assert len(records)==800 and len({(v['worker'],v['row']) for v in records})==800
    identity={'fixture':'budget_restore'};state=dict(identity=identity,calls=91,active_seconds=231.)
    path=R/'work/infrastructure_fixture';rt.checkpoint(path,state)
    assert rt.restore(path,identity)==state
    ref=rt.read(path/'latest.json');payload=path/ref['payload'];original=payload.read_bytes();payload.write_bytes(original+b'corruption')
    try:rt.restore(path,identity)
    except RuntimeError as e:assert str(e)=='BAD_CHECKPOINT_HASH'
    else:raise AssertionError('BAD_HASH_ACCEPTED')
    payload.write_bytes(original)
    with rt.exclusive('test_duplicate'):
        c=subprocess.run([sys.executable,'-B','-c',"import native_runtime as r;\nwith r.exclusive('test_duplicate'): pass"],cwd=R/'scripts',capture_output=True)
        assert c.returncode!=0 and b'DUPLICATE_LIVE_PROCESS' in c.stderr
    code="""import native_runtime as r
r.label_barrier('F23_G_D')
for p in [r.RUN/'data/folds/T24_G_D_H1/train.parquet',r.RUN/'data/heldout_labels/2023_days.parquet']:
 try:r.sha(p)
 except PermissionError:pass
 else:raise AssertionError('LABEL_READ_ALLOWED')
"""
    c=subprocess.run([sys.executable,'-B','-c',code],cwd=R/'scripts',capture_output=True)
    assert c.returncode==0,c.stderr.decode()
    rt.write(R/'reports/infrastructure_checks.json',dict(status='PASS',concurrent_rows=800,duplicate_refused=True,bad_hash_refused=True,cumulative_budget_restored=True,barrier_before_hash=True))
    print('PASS infrastructure')
if __name__=='__main__':main()
