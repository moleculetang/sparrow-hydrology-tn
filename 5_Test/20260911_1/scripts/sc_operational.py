"""Non-training launch verification of imports, barriers and child-exit events."""
import os
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import ast
import queue
import subprocess
import sys
import threading
import time
import numpy as np
from fc_io import *
from sc_controller import watch_child
from sc_runtime import admission

def main():
    require_environment();checks={}
    for p in (RUN/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    checks['all_scripts_parse']=True
    # Parent is live and another child still runs when first child is audited.
    q=queue.Queue();a=subprocess.Popen([sys.executable,'-c','pass'],creationflags=subprocess.CREATE_NO_WINDOW)
    b=subprocess.Popen([sys.executable,'-c','import time;time.sleep(3)'],creationflags=subprocess.CREATE_NO_WINDOW)
    threading.Thread(target=watch_child,args=(a,'first',q),daemon=True).start()
    threading.Thread(target=watch_child,args=(b,'second',q),daemon=True).start()
    key,code,ended=q.get(timeout=10);assert key=='first' and code==0 and b.poll() is None
    checks['child_exit_visible_while_parent_and_other_child_alive']=True;b.wait()
    resource={'ram':{'used_percent':40.,'available_gib':60.,'total_gib':100.},'cpu':{'used_percent':50.}}
    assert admission(resource,3.,[{'peak':3.,'rss':2.}])
    assert not admission(resource,30.,[{'peak':30.,'rss':1.}])
    resource['cpu']['used_percent']=90.;assert not admission(resource,1.,[])
    resource['cpu']['used_percent']=50.;resource['ram']['used_percent']=90.;assert not admission(resource,1.,[])
    checks['live_growth_reservation_and_CPU_RAM90']=True
    # Exercise installed native solver with a label-free algebraic smoke QP.
    from ty_clarabel_staged import native
    cl=native();from scipy import sparse
    s=cl.DefaultSettings();s.verbose=False;s.max_threads=1;s.max_iter=300;s.time_limit=300.
    solver=cl.DefaultSolver(sparse.csc_matrix([[1.]]),np.array([-1.]),sparse.csc_matrix([[-1.]]),np.array([0.]),[cl.NonnegativeConeT(1)],s)
    r=solver.solve();assert abs(r.x[0]-1)<1e-5
    checks['isolated_Clarabel_label_free_smoke']=True
    import sc_model,sc_physical,sc_report,sc_terminal,sc_worker,sc_audit,sc_capacity
    checks['all_entrypoints_import']=True
    numpath=RUN/'reports/operational_validation_numerics.json'
    if not numpath.exists():raise RuntimeError('AWAITING_INDEPENDENT_NUMERICAL_REVIEW')
    num=read(numpath);checks['independent_numerical_fault_tests']=num.get('status','').startswith('PASS')
    # Worker-level barrier must reject both builtin and pandas native parquet reads.
    code="import sys;sys.path.insert(0,'scripts');from sc_runtime import label_barrier;label_barrier();import pandas as pd\ntry:\n pd.read_parquet('outputs/data/F23_heldout_labels.parquet')\nexcept PermissionError:\n print('BARRIER_PASS')\nelse:\n raise RuntimeError('BARRIER_BYPASSED')"
    test=subprocess.run([sys.executable,'-B','-c',code],cwd=RUN,capture_output=True,text=True)
    assert test.returncode==0 and 'BARRIER_PASS' in test.stdout,test.stderr
    checks['pandas_heldout_label_barrier']=True
    write(RUN/'reports/operational_validation.json',{'status':'PASS_OPERATIONAL_VALIDATION','checks':checks,'utc':now(),
         'capacity_TN_solve_count':0,'predictive_fit_count':0})
    if not all(checks.values()):raise RuntimeError('OPERATIONAL_CHECK_FAILED')
    # Freeze only when all implementations and independent review are settled.
    # Freeze the training implementation separately from presentation/audit
    # plumbing. An independently documented report repair cannot silently
    # change or invalidate the scientific optimizer identity.
    training_names={'sc_model.py','sc_kernel.py','sc_worker.py','sc_runtime.py','gb_model.py','gb_solver.py',
                    'fc_io.py','fc_data.py','fc_legacy.py','fc_control.py','fc_checkpoint.py','fc_resources.py','fc_solvers.py',
                    'fc_features.py','fc_empirical.py','qx_bundle.py','qx_loss.py'}
    codepaths={p for p in (RUN/'scripts').glob('*.py') if p.name in training_names}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).is_file() and Path(path).suffix=='.py' and Path(path).resolve().is_relative_to(ROOT) and not Path(path).resolve().is_relative_to(RUN):codepaths.add(Path(path).resolve())
    proof={'status':'PASS_LAUNCH_VALIDATION','code_sha256':{str(p):sha(p) for p in sorted(codepaths)},
      'fold_config_sha256':{f:sha(RUN/f'configs/{f}.json') for f in ('F23','F24')},
      'registration_sha256':sha(RUN/'configs/campaign.json'),'model_validation_sha256':sha(RUN/'reports/model_validation.json'),
      'operational_validation_sha256':sha(RUN/'reports/operational_validation.json'),
      'operational_code_sha256':{str(p):sha(p) for p in sorted((RUN/'scripts').glob('*.py')) if p not in codepaths},'utc':now()}
    assert read(RUN/'reports/model_validation.json')['status']=='PASS_MODEL_IMPLEMENTATION'
    immutable(RUN/'reports/launch_validation.json',proof);print('PASS_LAUNCH_VALIDATION',flush=True)
if __name__=='__main__':main()
