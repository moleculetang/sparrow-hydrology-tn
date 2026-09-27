"""Identity plus independent forward/objective re-evaluation of four U paths."""
import sys,subprocess,time,traceback,os
import native_runtime as rt
R=rt.RUN
def main():
    jobs=rt.read(R/'configs/jobs.json');results=[]
    for j in jobs:
        if not j.get('reuse_from'):continue
        tag=j['tag'];old=R.parent/'20260921_2';p=old/'outputs'/tag
        try:
            m=rt.read(p/'model.json');a=rt.read(p/'audit.json')
            assert m['job']['kind']==j['kind']=='SOURCE_UNIFIED'
            for f in ['train.parquet','registry.json']:assert rt.sha(R/'data/folds'/j['fold']/f)==rt.sha(old/'data/folds'/j['fold']/f)
            assert rt.sha(R/'data/domains/FULL24C/arrays.json')==rt.sha(old/'data/domains/FULL24C/arrays.json')
            for name,h in a['files'].items():assert rt.sha(p/name)==h
            for rel in ['scripts/source_corrected.py','scripts/balanced_tags.py','scripts/hf_model.py','scripts/temporal_model.py']:
                assert rt.sha(R/rel)==rt.sha(old/rel)
            nowdesign=rt.read(R/'data/designs'/f"{j['fold']}.json");oldesign=rt.read(old/'data/designs'/f"{j['fold']}.json")
            assert {k:v for k,v in nowdesign.items() if k!='regional_basis'}==oldesign
            rt.resources();time.sleep(.3);res=rt.resources()
            while res['ram_percent']>=85 or res['cpu_percent'] is None or res['cpu_percent']>=85:
                rt.log('resource_events.jsonl',dict(event='REUSE_WAIT',resources=res));time.sleep(2);res=rt.resources()
            rt.write(R/'work/jobs'/tag/'status.json',dict(status='NUMERICALLY_SUFFICIENT' if a['numerical_sufficient'] else 'BUDGET_STOPPED',reason='HISTORICAL_PATH_REUSE_PENDING_INDEPENDENT_AUDIT',new_fit_calls=0,updated=time.time()))
            with (R/'work'/f'reuse_{tag}.log').open('wb') as f:
                child=subprocess.run([sys.executable,'-B',str(R/'scripts/audit_job.py'),tag],cwd=R,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            assert child.returncode==0,'INDEPENDENT_REUSE_AUDIT_FAILED'
            fresh=rt.read(R/'outputs'/tag/'audit.json');assert fresh['physical_reasonable']
            results.append(dict(tag=tag,reused=True,objective_error=abs(fresh['objective']-m['objective']),numerical_sufficient=fresh['numerical_sufficient'],new_fit_calls=0,peak_gib=fresh['process']['peak_gib'],prior_start=m['job'],historical_origin=str(p)))
        except Exception:
            # Conditional fallback is registered before launch; no new scientific start.
            failure=dict(tag=tag,reused=False,error=traceback.format_exc(),fallback='same registered baseline path from original preset')
            archive=R/'evidence/reuse_failures'/tag;archive.mkdir(parents=True,exist_ok=True)
            for folder in [R/'outputs'/tag,R/'work/jobs'/tag]:
                if folder.exists():
                    target=archive/folder.parent.name
                    assert folder.resolve().is_relative_to(R) and target.resolve().is_relative_to(R)
                    os.replace(folder,target)
            j.pop('reuse_from');results.append(failure)
        rt.write(R/'reports/baseline_reuse.json',dict(status='PASS',paths=results))
        print('REUSE',tag,results[-1]['reused'],flush=True)
    rt.write(R/'configs/jobs.json',jobs)
if __name__=='__main__':main()
