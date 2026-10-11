"""Compact native live status; no scientific dispatch, SSH or psutil."""
import ctypes
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'compute_coordination'))
import resource_registry as registry


def cpu_seconds(pid):
    if os.name=='posix':
        s=Path(f'/proc/{pid}/stat').read_text();p=s[s.rfind(')')+2:].split()
        return (int(p[11])+int(p[12]))/os.sysconf('SC_CLK_TCK')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:return None
    try:
        values=[ctypes.c_ulonglong() for _ in range(4)]
        if not kernel.GetProcessTimes(ctypes.c_void_p(handle),*[ctypes.byref(v) for v in values]):return None
        return (values[2].value+values[3].value)/1e7
    finally:kernel.CloseHandle(ctypes.c_void_p(handle))


def main():
    current=registry.status();leases=[v for v in current['leases'].values() if v['project']=='20261008_1']
    first={v['pid']:cpu_seconds(v['pid']) for v in leases};start=time.monotonic();time.sleep(.3);dt=time.monotonic()-start
    rows=[]
    for lease in leases:
        live=registry.identity(lease['pid'])==lease['created'];second=cpu_seconds(lease['pid']) if live else None
        a=first[lease['pid']]
        rows.append(dict(job=lease['job'],pid=lease['pid'],created=lease['created'],live=live,cpus=lease['cpus'],
            threads=lease['threads'],cpu_one_core_percent=(second-a)*100/dt if second is not None and a is not None else None,
            reserved_bytes=lease['reserved_bytes'],**registry.process_memory(lease['pid'])))
    m=current['memory'];checks={}
    for name in ('full_P0_compatibility','full_prediction_isolation','production_lifecycle_known_truth','evaluation_known_truth','full_initial_feedback','supported_zero_initial_feedback','full_uptake_support','dual_host_consistency'):
        p=ROOT/'evidence'/f'{name}.json'
        if p.exists():
            r=json.loads(p.read_text(encoding='utf8'));checks[name]=dict(passed=r['passed'],scope=r['scope'],completed_cases=len(r.get('cases',[])))
    audit=ROOT/'outputs/jobs/TIM_F23_C0_V0_s0/prerequisite_audit_progress.json'
    if audit.exists():
        p=json.loads(audit.read_text(encoding='utf8'));checks['C0_V0_coordinates']=dict(completed=len(p['checks']),passed=sum(c['passed'] for c in p['checks']),calls=p['calls'],total=22)
    used={c for v in current['leases'].values() for c in v['cpus']}
    report=dict(host=__import__('platform').node(),physical_cores_configured=len(current['cores']),unreserved_physical_cores=len([c for c in current['cores'] if c not in used]),live_project_workers=rows,
        physical_used_percent=m['used']*100/m['total'],commit_used_percent=m['commit_used']*100/m['commit_total'],
        scientific_fits_started=sum((ROOT/'outputs/jobs'/j['id']/'training_contract.json').exists() for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))),
        fits_with_saved_optimizer=sum((ROOT/'outputs/jobs'/j['id']/'optimizer.pkl').exists() for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))),
        registered_fit_paths=44,checks=checks)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
