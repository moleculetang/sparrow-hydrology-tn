"""64MiB Windows reserve/commit/touch/free, plus conservative admission fixtures."""
import os,ctypes,time,importlib.util
from pathlib import Path
from mltn.common import ROOT,write,sha
p=ROOT/'evidence/resource_registry_commit_candidate.py'
s=importlib.util.spec_from_file_location('candidate',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert os.name=='nt'
k=ctypes.windll.kernel32;k.VirtualAlloc.restype=ctypes.c_void_p
k.VirtualAlloc.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_ulong,ctypes.c_ulong]
k.VirtualFree.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_ulong]
size=64*1024**2;pid=os.getpid();checks=[];rows={}
def snap(name):rows[name]=dict(process=m.process_memory(pid),system=m.memory())
snap('before');address=k.VirtualAlloc(None,size,0x2000,4);assert address
try:
    snap('reserved');assert k.VirtualAlloc(address,size,0x1000,4)==address
    snap('committed_untouched');ctypes.memset(address,1,size);snap('touched')
finally:assert k.VirtualFree(address,0,0x8000)
snap('freed')
def delta(a,b,key):return rows[b]['process'][key]-rows[a]['process'][key]
assert abs(delta('before','reserved','private_commit'))<2*1024**2
assert abs(delta('reserved','committed_untouched','private_commit')-size)<2*1024**2
assert delta('reserved','committed_untouched','rss')<size//4
assert delta('committed_untouched','touched','rss')>size*.8
assert delta('touched','freed','private_commit')<-size*.8
checks.append(dict(check='native_private_commit_distinguished_from_resident_bytes',passed=True))
lease=dict(pid=pid,created=m.identity(pid),reserved_bytes=1000)
real=m.process_memory
m.process_memory=lambda p:dict(rss=100,private_commit=400,peak_commit=600)
h=m.memory_headrooms([lease]);assert h['physical']==900 and h['commit']==600
m.process_memory=lambda p:dict(rss=100,private_commit=400,peak_commit=2000)
h=m.memory_headrooms([lease]);assert h['physical']==900 and h['commit']==2300
checks.append(dict(check='native_peak_above_reserved_keeps_35_percent_margin',passed=True))
m.process_memory=lambda p:dict(rss=100,private_commit=None,peak_commit=None)
h=m.memory_headrooms([lease]);assert h['commit']==h['physical']==900
checks.append(dict(check='unknown_commit_never_subtracted',passed=True))
wrong=lease|dict(created='wrong_identity')
assert m.memory_headrooms([wrong])['physical']==m.memory_headrooms([wrong])['commit']==1000
checks.append(dict(check='reused_pid_no_commit_credit',passed=True))
m.process_memory=real
# This test does not change the live registry, leases, thresholds, or controllers.
write(ROOT/'evidence/commit_accounting_acceptance.json',dict(passed=True,checks=checks,measurements=rows,candidate_sha256=sha(p),linux_commit_rule_changed=False,live_reservations_changed=False))
print(dict(passed=True,checks=len(checks)))
