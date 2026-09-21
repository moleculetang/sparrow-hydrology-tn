"""Throwaway probe #11: THE PER-WORKER COMMIT RAMP.  DELETED BEFORE DELIVERY.

The 16-worker ramp failed 9 of 16 with `numpy ... Unable to allocate 41.0 MiB` while
Windows still reported 13-16 GB of FREE PHYSICAL memory.  The sampler's `freeCommit`
column explains it: it fell to 300 MB and then 63 MB.  On this machine `CommitLimit` is
63.8 GB with essentially no pagefile, so **commit charge, not physical RAM, is the ceiling**
-- and at that moment ~45 GB of it was held by other processes.

So the number that sizes the worker fleet is neither RSS nor peak RSS but PEAK COMMIT
(`PROCESS_MEMORY_COUNTERS.PeakPagefileUsage`), and the number that sizes how many may be
IN FLIGHT AT ONCE is that peak, because every worker passes through it simultaneously.

This probe prints current and peak commit at each stage of a cache-loaded worker, and
separately prices the two imports a shard worker might be able to avoid (torch) and cannot
(`closures` + `numba`, which the frozen `scan` needs).

Writes only `work\\_probe_commit.log`.
"""
import ctypes
import ctypes.wintypes as wt
import subprocess
import sys
import time

MB = 2**20


class _PMC(ctypes.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


_K32 = ctypes.WinDLL('kernel32', use_last_error=True)
_K32.GetCurrentProcess.restype = ctypes.c_void_p
_K32.GetCurrentProcess.argtypes = []
_PS = ctypes.WinDLL('psapi', use_last_error=True)
_PS.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), wt.DWORD]
_PS.GetProcessMemoryInfo.restype = wt.BOOL
_H = _K32.GetCurrentProcess()


def counters():
    p = _PMC()
    p.cb = ctypes.sizeof(_PMC)
    if not _PS.GetProcessMemoryInfo(_H, ctypes.byref(p), ctypes.sizeof(_PMC)):
        raise OSError('GetProcessMemoryInfo failed')
    return p.PagefileUsage / MB, p.PeakPagefileUsage / MB, p.WorkingSetSize / MB


T0 = time.time()


def stage(name):
    com, pcom, ws = counters()
    print('  %-34s t=%7.1fs  commit=%7.1f MB  PEAKcommit=%7.1f MB  ws=%7.1f MB'
          % (name, time.time() - T0, com, pcom, ws), flush=True)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'imports':
    # what each import costs on its own, in a fresh process of its own
    for label, code in (('numpy', 'import numpy'),
                        ('numba', 'import numba'),
                        ('torch', 'import torch'),
                        ('closures (numba+numpy)', 'import closures'),
                        ('common23 (torch+numba+closures)',
                         'import sys; sys.path.insert(0, "work"); import common23')):
        out = subprocess.run(
            [sys.executable, '-B', '-c',
             'import ctypes,ctypes.wintypes as wt\n'
             'class P(ctypes.Structure):\n'
             ' _fields_=[("cb",wt.DWORD),("pf",wt.DWORD),("pw",ctypes.c_size_t),'
             '("w",ctypes.c_size_t),("qp1",ctypes.c_size_t),("qp2",ctypes.c_size_t),'
             '("qp3",ctypes.c_size_t),("qp4",ctypes.c_size_t),("pf1",ctypes.c_size_t),'
             '("pf2",ctypes.c_size_t)]\n'
             'k=ctypes.WinDLL("kernel32"); k.GetCurrentProcess.restype=ctypes.c_void_p\n'
             's=ctypes.WinDLL("psapi")\n'
             's.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(P),wt.DWORD]\n'
             'import time; t=time.time(); ' + code + '; e=time.time()-t\n'
             'p=P(); p.cb=ctypes.sizeof(P)\n'
             's.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(p),p.cb)\n'
             'print("%.2f %.1f %.1f" % (e, p.pf1/1048576, p.pf2/1048576))'],
            capture_output=True, text=True, cwd='.', timeout=300)
        if out.returncode:
            print('  %-32s FAILED %s' % (label, out.stderr.strip().splitlines()[-1:]), flush=True)
            continue
        sec, com, pcom = out.stdout.split()
        print('  %-32s %6ss   commit=%7.1f MB  PEAKcommit=%7.1f MB'
              % (label, sec, float(com), float(pcom)), flush=True)
    sys.exit(0)


stage('start')
sys.path.insert(0, 'work')
import numpy as np                                                    # noqa: E402
stage('after numpy')
import common23 as C                                                  # noqa: E402
stage('after common23 import (torch+numba+closures)')
z = np.load('work/_probe_cache.npz', mmap_mode='r')
stage('after np.load(mmap_mode) -- untouched file pages')
r = C.solve_k({k.split('.', 1)[1]: z[k] for k in z.files if k.startswith('B.')},
              {k.split('.', 1)[1]: z[k] for k in z.files if k.startswith('G.')},
              'N1', 0.05) if False else None
# the real path: reconstruct + verify + solve, exactly as `load()` does
import json                                                           # noqa: E402
import hashlib                                                        # noqa: E402


def sha(a):
    a = np.ascontiguousarray(a)
    if a.dtype.kind == 'M':
        a = a.view('int64')
    return hashlib.sha256(memoryview(a).cast('B')).hexdigest()


man = json.loads(open('work/_probe_cache_manifest.json', encoding='utf-8').read())
B, G = {}, {}
for name in z.files:
    if sha(z[name]) != man[name]['sha256']:
        raise SystemExit('CACHED_ARRAY_DISAGREES_WITH_THE_MANIFEST %s' % name)
    (B if name.startswith('B.') else G)[name.split('.', 1)[1]] = z[name]
B['shape'] = list(B['h'].shape)
B['cap'] = False
B['k'] = np.asarray(B['k'])
G['S_u'] = float(np.median(G['sd_r']))
stage('after load + sha verify')
t = time.time()
r = C.solve_k(B, G, 'N1', 0.05)
stage('after solve_k N1 +0.05 (%.1fs)' % (time.time() - t))
print('  k_sha=%s' % r['k_sha256'][:16], flush=True)
com, pcom, ws = counters()
print('=== SUMMARY ===', flush=True)
print('  PEAK COMMIT of one cache-loaded worker : %7.1f MB' % pcom, flush=True)
print('  steady commit / working set            : %7.1f / %7.1f MB' % (com, ws), flush=True)
