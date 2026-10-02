"""Real Windows sharing-denial fixture plus fail-closed atomic publication."""
import sys,os,json,time,tempfile,subprocess,importlib.util
from pathlib import Path
from mltn.common import ROOT,write,sha
source=ROOT/'evidence/resource_registry_atomic_retry_candidate.py'
spec=importlib.util.spec_from_file_location('registry_candidate',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
with tempfile.TemporaryDirectory(dir=ROOT/'transfer') as tmp:
    t=Path(tmp);target=t/'registry.json';target.write_text('{"counter":0}')
    child=t/'lock_reader.py';child.write_text("import ctypes,sys,time\nk=ctypes.windll.kernel32;k.CreateFileW.restype=ctypes.c_void_p\nh=k.CreateFileW(sys.argv[1],0x80000000,3,None,3,0,None)\nassert h not in [None,ctypes.c_void_p(-1).value]\nprint('locked',flush=True)\ntime.sleep(.8)\nk.CloseHandle(ctypes.c_void_p(h))\n")
    p=subprocess.Popen([sys.executable,'-B',str(child),str(target)],stdout=subprocess.PIPE,text=True)
    assert p.stdout.readline().strip()=='locked'
    incoming=t/'incoming.tmp';incoming.write_text('{"counter":1}');start=time.monotonic();m.publish_registry(incoming,target);elapsed=time.monotonic()-start
    assert p.wait()==0 and json.loads(target.read_text())['counter']==1 and elapsed>=.4
    checks.append(dict(check='actual_non_delete_sharing_reader_releases_then_atomic_replace',elapsed_s=elapsed,passed=True))
    real=m.os.replace
    def permanent(*a,**k):raise PermissionError('non-Windows/transient error not classified as retryable')
    m.os.replace=permanent;rejected=False
    try:m.publish_registry(t/'missing',target)
    except PermissionError:rejected=True
    finally:m.os.replace=real
    assert rejected;checks.append(dict(check='unclassified_permission_failure_not_swallowed',passed=True))
    with m.transaction(t/'state') as s:s['pending']['fixture']=dict(count=1,pid=os.getpid(),created=m.identity(os.getpid()))
    assert m.status(t/'state')['pending']['fixture']['count']==1
    checks.append(dict(check='locked_registry_transaction_preserved',passed=True))
write(ROOT/'evidence/registry_atomic_publication_acceptance.json',dict(passed=True,checks=checks,old_sha256=sha(ROOT/'evidence/resource_registry_before_atomic_retry.py'),new_sha256=sha(source),live_leases_released=False,scope='publication only; 90/85, reservations and CPU/GPU APIs unchanged'))
print(dict(passed=True,checks=len(checks)))
