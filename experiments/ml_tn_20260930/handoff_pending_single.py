"""One pending registered job: acknowledge dispatch hold before changing host ownership."""
import os,time,platform,argparse
from mltn.common import ROOT,read,write,sha
from controller import jobid

def main(jid,host):
    assert os.name=='posix' and host!=platform.node()
    cfgfile=ROOT/'config/execution.json';cfg=read(cfgfile)
    assert not cfg.get('dispatch_hold',False),'ALREADY_HELD_INSPECT_FIRST'
    ownerfile=ROOT/'config/external_ownership.json';owners=read(ownerfile);before=sha(ownerfile)
    assert jid not in owners['jobs'],'ALREADY_EXTERNAL'
    candidates=[j for j in read(ROOT/'config/joint_final_jobs.json') if jobid(j)==jid]
    assert len(candidates)==1,'NOT_REGISTERED_UNIQUE_JOB'
    write(cfgfile,cfg|{'dispatch_hold':True})
    try:
        deadline=time.monotonic()+45
        while True:
            status=read(ROOT/'outputs/controller_status.json')
            if status.get('dispatch_hold') and status['phase']=='joint_final':break
            if time.monotonic()>deadline:raise RuntimeError('DISPATCH_HOLD_NOT_ACKNOWLEDGED')
            time.sleep(1)
        assert jid not in status['active'],'ACTIVE_JOB_CANNOT_MOVE'
        folder=ROOT/'jobs'/jid
        assert not folder.exists(),'PREVIOUS_ATTEMPT_CANNOT_MOVE'
        assert sha(ownerfile)==before,'OWNER_MAP_CHANGED'
        owners['jobs'][jid]=dict(host=host,job=candidates[0]);write(ownerfile,owners)
        write(ROOT/'evidence'/('ownership_'+jid+'.json'),dict(job_id=jid,owner=host,previous_owner_map_sha256=before,new_owner_map_sha256=sha(ownerfile),dispatch_hold_acknowledged=True,verified_no_attempt=True,signals_sent=False,scientific_grid_unchanged=True))
        print(dict(job_id=jid,owner=host,owner_map_sha256=sha(ownerfile)))
    finally:
        current=read(cfgfile);assert current.get('dispatch_hold') is True
        write(cfgfile,current|{'dispatch_hold':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--host',required=True);a=p.parse_args();main(a.job,a.host)
