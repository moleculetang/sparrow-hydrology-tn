"""Data-free checks against the exact published scientific kernels."""
from pathlib import Path
import os,sys,json,unittest
ROOT=Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
sys.path.insert(0,str(ROOT))
def isolation(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        p=os.fsdecode(args[0]).replace('\\','/').lower()
        if 'e:/sparrow/5_test/' in p:raise RuntimeError('PUBLIC_TEST_PRIVATE_EXPERIMENT_ACCESS: '+p)
sys.addaudithook(isolation)
suite=unittest.TestSuite()
for name in sorted(p.name for p in (ROOT/'tests').glob('test_*.py') if p.name!='test_contracts.py'):
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
receipt=dict(tests=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
    source='published exact kernels and synthetic fixtures',private_experiment_reads_forbidden=True,
    real_TN_run_reproduced=False,scientific_acceptance=False)
(ROOT/'public_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
raise SystemExit(0 if result.wasSuccessful() else 1)
