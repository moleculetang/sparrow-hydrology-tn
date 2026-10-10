"""Run public synthetic controls without access to private experiment roots."""
from pathlib import Path
import os,sys,json,subprocess,time
HERE=Path(__file__).resolve().parent
ROOT=HERE/'source/20261008_1'
TESTS=['../../test_event_profiles_public.py','tests/test_timing_kernel.py','tests/test_timing_edge_cases.py','tests/test_timing_labels.py','tests/test_year_feedback.py','tests/test_plant_activity_ledger.py','work/test_native_bound_adapter_v2.py']
def isolated_child(test):
 def guard(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).resolve();s=str(p).replace('\\','/').lower()
   if s.startswith('e:/sparrow/') and not p.is_relative_to(HERE):raise PermissionError('PRIVATE_EXPERIMENT_READ_BLOCKED')
 sys.addaudithook(guard)
 if test=='work/test_native_bound_adapter_v2.py':
  import tempfile,shutil
  scratch=HERE/'source/20261008_1/outputs/public_controls'
  scratch.mkdir(parents=True,exist_ok=True)
  isolated=Path(tempfile.mkdtemp(prefix='native_',dir=scratch))
  for rel in ('work/native_bound_adapter_v2.py','vendor/legacy22/scripts/serial_solvers.py'):
   dest=isolated/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
  (isolated/'evidence').mkdir()
  os.environ['D29_BOUNDS_TEST_ROOT']=str(isolated)
 (ROOT/'evidence').mkdir(exist_ok=True)
 sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'work')]
 import runpy
 runpy.run_path(str(ROOT/test),run_name='__main__')
def main():
 results=[];env=os.environ.copy()
 for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[n]='1'
 env['PYTHONDONTWRITEBYTECODE']='1';env['NUMBA_CACHE_DIR']=str(ROOT/'cache/numba');env['D29_BOUNDS_TEST_ROOT']=str(ROOT)
 for test in TESTS:
  start=time.monotonic();r=subprocess.run([sys.executable,'-B',str(HERE/'public_tests.py'),'--child',test],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
  results.append(dict(test=test,exit_code=r.returncode,seconds=time.monotonic()-start,stdout=r.stdout,stderr=r.stderr));print(test,r.returncode,flush=True)
  if r.returncode:break
 report=dict(passed=len(results)==len(TESTS) and all(r['exit_code']==0 for r in results),scope='synthetic_only_no_private_data_no_real_fit',tests=results)
 (HERE/'public_test_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 return 0 if report['passed'] else 1
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--child':isolated_child(sys.argv[2])
 else:raise SystemExit(main())
