"""One-shot completion continuation, with stage receipts and recoverable output archive."""
import os,sys,time,subprocess,datetime,tarfile
from mltn.common import ROOT,read,write,sha

def main():
    while not (ROOT/'outputs/all_training_done.json').exists():
        # The parent continuation may fail; never convert that into a successful wait.
        owner=ROOT/'outputs/remaining_owner.json'
        if owner.exists():
            from mltn.resources import registry
            mod,state=registry()
            if mod is not None and mod.identity(read(owner)['pid']) is None:raise RuntimeError('TRAINING_CONTINUATION_STOPPED_BEFORE_COMPLETION')
        time.sleep(15)
    write(ROOT/'outputs/postprocess_owner.json',dict(pid=os.getpid(),host=__import__('platform').node(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    stages=['coverage.py','baselines.py','history_baseline.py','ensemble.py','freeze_results.py','monthly_readouts.py','full_domain.py','evaluate.py','training_metrics.py','diagnostics.py','feature_reliance.py','feature_collisions.py','independent_readings.py','independent_review.py']
    dependencies={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'mltn').glob('*.py'))}
    def valid_stage(script):
        marker=ROOT/'outputs/postprocess_stages'/script.replace('.py','.json')
        if not marker.exists():return False
        prior=read(marker)
        return prior.get('exit_code')==0 and prior.get('code_sha256')==sha(ROOT/script) and prior.get('source_dependencies')==dependencies
    # A resumed CPU-only tail need not reserve a GPU after all GPU-capable
    # readouts are already verified. No live lease is shrunk or modified.
    device='cpu' if all(valid_stage(s) for s in ['ensemble.py','monthly_readouts.py','full_domain.py']) else 'cuda'
    from mltn.resources import lease
    lease('completion_postprocess',1,device)
    write(ROOT/'outputs/postprocess_execution.json',dict(device=device,threads=1,reason='valid completed readouts reused; GPU only if an unfinished stage can use it',source_dependencies=dependencies))
    records=[]
    for script in stages:
        marker=ROOT/'outputs/postprocess_stages'/script.replace('.py','.json')
        if valid_stage(script):continue
        t=time.monotonic();log=ROOT/'logs'/('postprocess_'+script+'.log')
        env=os.environ.copy()
        for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
        with log.open('ab') as f:code=subprocess.call([sys.executable,'-B',str(ROOT/script)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        r=dict(script=script,exit_code=code,elapsed_s=time.monotonic()-t,code_sha256=sha(ROOT/script),source_dependencies=dependencies);write(marker,r);records.append(r)
        if code:raise RuntimeError('POSTPROCESS_FAILED '+script)
    write(ROOT/'outputs/postprocess_done.json',dict(status='computational_postprocess_done',stages=records,expert_interpretation='pending human-readable evidence-based review',training_missing=read(ROOT/'outputs/result_roles.json')['missing']))
    paths=[]
    for directory in ['jobs','outputs','evidence','logs','config','superseded']:
        if (ROOT/directory).exists():paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
    # Match bytes, including all attempts; original independent arrays remain in their frozen input product.
    write(ROOT/'delivery_manifest.json',{p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths})
    archive=ROOT/'machine_learning_results.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p in [*paths,ROOT/'delivery_manifest.json']:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
    write(ROOT/'delivery_archive_receipt.json',dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),expert_report_complete=False))
if __name__=='__main__':main()
