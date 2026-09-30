"""Run the new engineering suite and save a machine-checkable receipt."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import sha,write_json

env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
cmd=[sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_*.py','-q']
run=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
log=ROOT/'outputs/review_engineering_tests.log';log.write_text(run.stdout+run.stderr,encoding='utf-8')
write_json(ROOT/'outputs/review_engineering_tests.json',{'passed':run.returncode==0,'exit_code':run.returncode,
    'test_log_sha256':sha(log),'test_source_hashes':{str(p):sha(p) for p in sorted((ROOT/'tests').glob('test_*.py'))},
    'NSE':'not applicable to engineering tests'})
print(run.stderr[-1500:])
if run.returncode:raise SystemExit(run.returncode)
