"""Serial, recoverable development factorial. No final selection or promotion."""
from pathlib import Path
import sys
import subprocess
import json
import os

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME, atomic_json, sha256, utc_now, memory_guard

RUN=ROOT/'5_Test/20260905_3'
FIT=ROOT/'5_Test/20260905_2/scripts/fit_models.py'


def main():
    proof=json.loads((ROOT/'5_Test/20260905_2/reports/objective_validation.json').read_text(encoding='utf-8'))
    if proof['status']!='PASS_ASSEMBLED_OBJECTIVES' or proof['code_sha256'][str(FIT)]!=sha256(FIT):
        raise RuntimeError('Missing or stale assembled-objective validation')
    code={str(p):sha256(p) for p in [FIT,FIT.parent/'tn_autograd.py',FIT.parent/'tn_reference.py']}
    input_hash=sha256(ROOT/'5_Test/20260905_1/reports/input_manifest.json')
    combinations=[('CONTROL_H7','STUDENT_T4_LOG1P')]
    for model in ['GLOBAL','H7_CONTACT','H7_CONTACT_LIFETIME']:
        combinations.extend((model,loss) for loss in ['STATION_NORMALIZED_MSE','STUDENT_T4_LOG1P'])
    jobs=[]
    # Visit every architecture early, then complete all five starts. This does
    # not authorize model selection before the full paired matrix is available.
    for year in [2020,2021,2022,2023]:
        for start in range(5):
            for model,loss in combinations:
                tag=f'{model.lower()}_{loss.lower()}_t{year}_p1_s{start}'
                stage='20260905_2' if model=='CONTROL_H7' else '20260905_3'
                jobs.append({'tag':tag,'model':model,'loss':loss,'fold':f'T{year}','start':start,
                             'report':str(ROOT/'5_Test'/stage/'reports'/f'{tag}.json')})
    status_path=RUN/'reports/development_queue.json'
    status={'status':'RUNNING','pid':os.getpid(),'runtime':RUNTIME,'started_utc':utc_now(),
            'jobs':jobs,'completed':[],'unresolved':[],'active':None,'code_sha256':code,'input_manifest_sha256':input_hash,
            'remaining_program':['structural_timing_trials','nested_spatial_validation','2024_confirmation','F24_F25_fits','exports','completion_audit']}
    for job in jobs:
        memory_guard()
        if any(sha256(p)!=digest for p,digest in code.items()):raise RuntimeError('Core changed while queue was active')
        report=Path(job['report'])
        if report.exists():
            previous=json.loads(report.read_text(encoding='utf-8'))
            if previous['identity']['code_sha256']!=code or previous['identity']['input_manifest_sha256']!=input_hash:
                raise RuntimeError(f'Stale existing result {report}')
            if previous['converged']:
                status['completed'].append(job['tag']);continue
        status['active']=job['tag'];status['updated_utc']=utc_now()
        atomic_json(status,status_path)
        log=RUN/'logs'/f"{job['tag']}.log"
        log.parent.mkdir(parents=True,exist_ok=True)
        print('QUEUE_JOB_STARTED',len(status['completed']),len(jobs),job['tag'],flush=True)
        with log.open('a',encoding='utf-8') as stream:
            command=[sys.executable,'-B',str(FIT),'--model',job['model'],'--loss',job['loss'],
                     '--fold',job['fold'],'--start',str(job['start']),'--maxiter','700']
            completed=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode!=0:
            status['unresolved'].append({'job':job['tag'],'reason':'process_error','returncode':completed.returncode,'log':str(log)})
        else:
            fitted=json.loads(report.read_text(encoding='utf-8'))
            if fitted['converged']:status['completed'].append(job['tag'])
            else:status['unresolved'].append({'job':job['tag'],'reason':'stationarity_not_reached','report':str(report)})
        status['updated_utc']=utc_now();atomic_json(status,status_path)
        print('QUEUE_JOB_FINISHED',job['tag'],'completed',len(status['completed']),'unresolved',len(status['unresolved']),flush=True)
    status['active']=None
    status['status']='DEVELOPMENT_FITS_COMPLETE' if not status['unresolved'] else 'DEVELOPMENT_FITS_REQUIRE_NUMERICAL_RESOLUTION'
    status['updated_utc']=utc_now();atomic_json(status,status_path)
    print('QUEUE_FINISHED',status['status'],flush=True)


if __name__=='__main__':main()
