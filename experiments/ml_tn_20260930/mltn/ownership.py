"""Release only verified dead local ownership; never infer death from SSH loss."""
import os,json,platform,time
from mltn.common import write,read
from mltn.resources import registry

def reclaim(folder,clear_failure=True):
    lock=folder/'owner.lock'
    if not lock.exists():return
    raw=lock.read_text(encoding='utf-8').strip()
    try:
        owner=json.loads(raw);pid=owner['pid'] if isinstance(owner,dict) else int(owner)
    except (ValueError,TypeError):raise RuntimeError('UNPARSEABLE_OWNER '+str(folder))
    if isinstance(owner,dict) and owner.get('host')!=platform.node():raise RuntimeError('OTHER_HOST_OWNER '+str(folder))
    mod,state=registry()
    if mod is None:raise RuntimeError('NO_PROCESS_IDENTITY_CHECKER')
    if mod.identity(pid) is not None:raise RuntimeError('OWNER_STILL_LIVE '+str(pid))
    archive=folder/'recovery';archive.mkdir(exist_ok=True);stamp=str(time.time_ns())
    os.replace(lock,archive/('owner_'+stamp+'.lock'))
    failure=folder/'failure.json'
    if clear_failure and failure.exists():os.replace(failure,archive/('failure_'+stamp+'.json'))
    write(archive/('receipt_'+stamp+'.json'),dict(pid=pid,identity_verified_dead=True,host=platform.node(),action='release dead lock only; epoch checkpoints retained'))
