"""Checked one-time queue reorder deployment; scientific sources must be identical."""
import os,sys,tarfile,shutil,subprocess
from mltn.common import ROOT,read,write,sha

assert os.name=='posix'
archive=ROOT/'transfer/source_update_28.tar.gz'
assert sha(archive)==sys.argv[1],'ARCHIVE_IDENTITY'
assert not (ROOT/'evidence/matched_queue_deployment.json').exists(),'ALREADY_DEPLOYED_INSPECT'
with tarfile.open(archive) as tf:
    members=tf.getmembers()
    for m in members:
        target=(ROOT/m.name).resolve();assert target.is_relative_to(ROOT.resolve()) and m.isfile(),'UNSAFE_MEMBER'
        if m.name.startswith('mltn/') or m.name in ['train.py','joint.py'] or m.name.startswith('data/'):
            assert target.exists() and sha(target)==__import__('hashlib').sha256(tf.extractfile(m).read()).hexdigest(),('SCIENTIFIC_OR_INPUT_CHANGE_FORBIDDEN',m.name)
    for name in ['dispatch_handoff.json','dispatch_handoff_priority.json']:
        old=ROOT/'outputs'/name
        if old.exists():shutil.copyfile(old,ROOT/'evidence'/('matched_queue_previous_'+name))
    for name in ['gpu_priority_handoff_launch.json','gpu_priority_handoff_previous_owner.json']:
        old=ROOT/'evidence'/name
        if old.exists():shutil.copyfile(old,ROOT/'evidence'/('matched_queue_previous_'+name))
    tf.extractall(ROOT,filter='data')
manifest=read(ROOT/'evidence/source_update_28_manifest.json')
for name,digest in manifest.items():assert sha(ROOT/name)==digest,name
write(ROOT/'evidence/source_update_28_remote_verified.json',dict(passed=True,files=len(manifest),scientific_sources_and_inputs_unchanged=True))
from run_remaining import matched_jobs
from controller import jobid
s=read(ROOT/'outputs/frozen_selection.json');best=read(ROOT/'outputs/frozen_joint_selection.json')['selected']
jobs=sorted(map(jobid,matched_jobs(s,best)));assert len(jobs)==5 and len(set(jobs))==5
subprocess.run([sys.executable,'-B',str(ROOT/'launch_gpu_priority_handoff.py')],check=True,cwd=ROOT)
write(ROOT/'evidence/matched_queue_deployment.json',dict(passed=True,source_update=28,existing_matched_job_ids=jobs,scientific_grid_unchanged=True,training_signals_sent=False,activation='original scientific children finish naturally before new supervisor uses merged queue',source_sha256=sha(ROOT/'run_remaining.py')))
print('MATCHED_QUEUE_CHECKED_HANDOFF',jobs,flush=True)
