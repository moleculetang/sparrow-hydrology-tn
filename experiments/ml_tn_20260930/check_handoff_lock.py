"""POSIX-only fake supervisor verifies suspension cannot retain registry.lock."""
import os,time,tempfile,subprocess,signal
from pathlib import Path
from mltn.common import ROOT,write
from mltn.resources import registry
from handoff_controller import freeze_supervisor,state
assert os.name=='posix'
mod,_=registry()
with tempfile.TemporaryDirectory(dir=ROOT/'transfer') as tmp:
    t=Path(tmp);script=t/'fake_supervisor.py'
    script.write_text('import sys,time\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\nimport resource_registry as r\nwith r.transaction(Path(sys.argv[2])) as s:\n s["fixture"]=True\n print("LOCKED",flush=True)\n time.sleep(.6)\ntime.sleep(60)\n')
    base=str(ROOT.parent/'.compute_coordination');p=subprocess.Popen([os.sys.executable,'-B',str(script),base,str(t/'state')],stdout=subprocess.PIPE,text=True)
    identity=mod.identity(p.pid)
    try:
        assert p.stdout.readline().strip()=='LOCKED'
        start=time.monotonic();freeze_supervisor(p.pid,mod,t/'state');elapsed=time.monotonic()-start
        assert state(p.pid)=='T' and elapsed>=.3
        assert mod.status(t/'state')['fixture'] is True
        assert mod.identity(p.pid)==identity
    finally:
        # Only our dedicated fake fixture receives cleanup signals.
        os.kill(p.pid,signal.SIGCONT);p.terminate();p.wait(timeout=10)
write(ROOT/'evidence/handoff_lock_acceptance.json',dict(passed=True,elapsed_s=elapsed,production_signals_sent=False,check='fake owner leaves registry critical section before supervisor suspension'))
print(dict(passed=True,elapsed_s=elapsed))
