"""Run only after source transfer; preserve every training PID."""
import os,signal,subprocess,sys,datetime
from pathlib import Path
from mltn.common import ROOT,read,write,sha
version=sys.argv[1]
manifest=read(ROOT/'evidence'/f'source_update_{version}_manifest.json')
for relative,expected in manifest.items():
    p=(ROOT/relative.replace('\\','/')).resolve();assert p.is_relative_to(ROOT.resolve());assert sha(p)==expected,relative
write(ROOT/'evidence'/f'source_update_{version}_remote_verified.json',dict(files=len(manifest),passed=True))
changed=[]
def restart(pid,script,log):
    proc=Path(f'/proc/{pid}')
    if proc.exists():
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert cmd.endswith(str(ROOT/script)+' '),cmd
        assert not (proc/f'task/{pid}/children').read_text().strip(),'unexpected active child'
        os.kill(pid,signal.SIGTERM)
    p=subprocess.Popen([os.sys.executable,'-B',str(ROOT/script)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=(ROOT/'outputs'/log).open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    changed.append(dict(script=script,old_pid=pid,new_pid=p.pid))
if not (ROOT/'outputs/remaining_owner.json').exists() and not (ROOT/'outputs/direct_matrix_done.json').exists():
    prior=ROOT/'outputs/waiter_latest.json'
    pid=read(prior)['pid'] if prior.exists() else read(ROOT/'outputs/waiter_update04_receipt.json')['new_pid']
    restart(pid,'run_remaining.py','remaining_controller.log')
    if changed:write(prior,dict(pid=changed[-1]['new_pid']))
prior=ROOT/'outputs/adoption_latest.json';pid=read(prior)['pid'] if prior.exists() else 1598697
restart(pid,'adopt_current.py','shared_resource_adoption.log')
if changed and changed[-1]['script']=='adopt_current.py':write(prior,dict(pid=changed[-1]['new_pid']))
prior=ROOT/'outputs/postprocess_waiter.json'
if prior.exists() and not (ROOT/'outputs/postprocess_owner.json').exists() and not (ROOT/'outputs/all_training_done.json').exists():
    restart(read(prior)['pid'],'postprocess.py','postprocess_controller.log')
    if changed and changed[-1]['script']=='postprocess.py':write(prior,dict(pid=changed[-1]['new_pid'],kind='one-shot event continuation, not scheduled task'))
receipt=dict(version=version,changed=changed,training_untouched=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(ROOT/'outputs'/f'activation_{version}.json',receipt);print(receipt)
