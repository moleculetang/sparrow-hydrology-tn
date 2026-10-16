"""Compact native status in the one verified host ledger; no dispatch."""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from host_resources import registry,registry_root


def cpu(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();parts=raw[raw.rfind(')')+2:].split()
        return (int(parts[11])+int(parts[12]))/__import__('os').sysconf('SC_CLK_TCK')
    except FileNotFoundError:return None


def main():
    state=registry.status(root=registry_root());leases=list(state['leases'].values())
    previous={l['pid']:cpu(l['pid']) for l in leases};start=time.monotonic();time.sleep(.3);dt=time.monotonic()-start
    rows=[]
    for l in leases:
        now=cpu(l['pid']);before=previous[l['pid']]
        rows.append(dict(project=l['project'],job=l['job'],pid=l['pid'],created=l['created'],
            live=registry.identity(l['pid'])==l['created'],cpus=l['cpus'],threads=l['threads'],reserved_bytes=l['reserved_bytes'],
            rss_bytes=registry.rss(l['pid']),cpu_one_core_percent=None if before is None or now is None else (now-before)*100/dt))
    slices=[]
    for i in range(6):
        p=ROOT/f'outputs/jobs/prerequisite_K_coordinates_{i}/coordinate_slice.json'
        if not p.exists():continue
        r=json.loads(p.read_text());slices.append(dict(slice=i,complete=r.get('complete'),passed=r.get('passed'),
            count=len(r['checks']),checks=[dict(coordinate=c['coordinate'],passed=c['passed']) for c in r['checks']],calls=r.get('calls')))
    used={c for l in leases for c in l['cpus']};m=state['memory']
    print(json.dumps(dict(host=__import__('platform').node(),physical_core_cap=len(state['cores']),unreserved_cores=len(set(state['cores'])-used),
        physical_used_percent=m['used']*100/m['total'],commit_used_percent=m['commit_used']*100/m['commit_total'],
        leases=rows,coordinate_slices=slices,scientific_fit_paths_started=sum((ROOT/'outputs/jobs'/j['id']/'training_contract.json').exists()
            for j in json.loads((ROOT/'config/jobs.json').read_text())))))


if __name__=='__main__':main()
