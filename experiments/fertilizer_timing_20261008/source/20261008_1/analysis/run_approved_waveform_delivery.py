"""Resource-bound local pure-data extraction, separate from scientific jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'analysis')]
from d29_platform.runtime import sha, write_json
from host_resources import registry, registry_root
from controller_lock import acquire


def verified_case(folder, source, jid, gate):
    path = folder / 'complete.json'
    value = json.loads(path.read_text(encoding='utf8'))
    if (value.get('passed') is not True or value.get('registered_job', {}).get('id') != jid or
            value.get('formal_gate_sha256') != gate or value.get('source_folder') != str(source.resolve()) or
            value.get('source_freeze_sha256') != sha(source / 'prediction_freeze.json') or
            value.get('physical_receipt_sha256') != sha(source / 'physical_ledger.json') or
            value.get('evaluation_labels_read') is not False or value.get('physical_oracle_calls') != 0 or
            value.get('source_sha256') != sha(ROOT / 'analysis/diagnose_approved_waveforms.py') or
            value.get('immutable_shape_helper_sha256') != sha(ROOT / 'analysis/physical_waveform_diagnostics.py')):
        raise RuntimeError('EXACT_APPROVED_WAVEFORM_RECEIPT_REQUIRED')
    outputs = value.get('output_sha256', {})
    if set(outputs) != {'domain_flux_shapes.csv', 'station_boundary_shapes.csv'}:
        raise RuntimeError('EXACT_DIAGNOSTIC_OUTPUTS_REQUIRED')
    for relative, digest in outputs.items():
        if sha(folder / relative) != digest:
            raise RuntimeError('DIAGNOSTIC_OUTPUT_BYTES_CHANGED')
    return dict(job=jid, formal_gate_sha256=gate, complete_sha256=sha(path),
                outputs=outputs, station_count=value['station_count'])


def main(mode):
    if os.name != 'nt' or mode not in ('preflight', 'final'):
        raise RuntimeError('LOCAL_DECLARED_PURE_DATA_DIAGNOSTIC_REQUIRED')
    jobs = json.loads((ROOT / 'config/jobs.json').read_text(encoding='utf8'))
    if mode == 'preflight':
        selected = [j for j in jobs if j['id'] == 'TIM_F24_C0_V0_s0']
        sources = {j['id']: ROOT / 'outputs/jobs' / j['id'] for j in selected}
    else:
        from approved_epoch_delivery import admitted_folders
        sources = admitted_folders(ROOT)  # No resource dispatch before actual44.
        selected = jobs
    folder = ROOT / 'outputs' / ('approved_waveforms_' + mode)
    folder.mkdir(parents=True, exist_ok=True)
    lifetime = acquire(folder / 'lifetime.lock')
    marker = folder / 'owner.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if registry.identity(old['pid']) == old['created']:
            raise RuntimeError('WAVEFORM_CONTROLLER_ALREADY_LIVE')
    if (folder / 'complete.json').exists():
        raise RuntimeError('WAVEFORM_DELIVERY_ALREADY_COMPLETED_REVIEW_NO_RERUN')
    owner = dict(pid=os.getpid(), created=registry.identity(os.getpid()))
    write_json(marker, owner)
    cases = []; events = []; started = time.monotonic()
    from diagnose_approved_waveforms import approved_gate
    for job in selected:
        jid = job['id']; gate = approved_gate(job); destination = folder / jid; source = sources[jid]
        if (destination / 'complete.json').exists():
            cases.append(verified_case(destination, source, jid, gate))
            continue
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError('PARTIAL_DIAGNOSTIC_CASE_REQUIRES_REVIEW')
        lease = registry.acquire('20261008_1', 'analysis/approved_waveforms_' + mode + '/' + jid,
                                 threads=1, peak_bytes=500000000, root=registry_root())
        write_json(folder / 'resource_decision.json', lease)
        if not lease.get('granted'):
            write_json(folder / 'status.json', dict(phase='resource_wait_requires_same_controller_reentry',
                       owner=owner, cases=cases, mode=mode, scientific_signals=0))
            return 4
        child = None; peak = dict(rss=0, private_commit=0, peak_commit=0)
        try:
            with (folder / (jid + '.log')).open('ab') as log:
                child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B',
                    str(ROOT / 'scripts/guarded_worker.py'), lease['token'],
                    'analysis/diagnose_approved_waveforms.py', jid, mode], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                registry.bind(lease['token'], child.pid, root=registry_root())
                write_json(folder / 'status.json', dict(phase='approved_shape_case_running', owner=owner,
                    child=dict(pid=child.pid, created=registry.identity(child.pid)), job=jid,
                    lease=lease, mode=mode, cases=cases, scientific_signals=0, evaluation_labels_read=False))
                while child.poll() is None:
                    memory = registry.process_memory(child.pid)
                    for key in peak:
                        if memory.get(key) is not None: peak[key] = max(peak[key], memory[key])
                    time.sleep(.5)
            event = dict(job=jid, exit_code=child.returncode, peak=peak)
            events.append(event)
            write_json(folder / (jid + '.returned.json'), event)
            if child.returncode:
                write_json(folder / 'status.json', dict(phase='diagnostic_case_requires_review',
                    owner=owner, event=event, cases=cases, mode=mode, scientific_signals=0))
                return 2
            cases.append(verified_case(destination, source, jid, gate))
        finally:
            if child is None or child.poll() is not None:
                registry.release(lease['token'], root=registry_root())
    expected = 1 if mode == 'preflight' else 44
    if len(cases) != expected or len({c['job'] for c in cases}) != expected:
        raise RuntimeError('EXACT_MODE_PATH_ACCOUNTING_REQUIRED')
    result = dict(passed=True, mode=mode, accounted_paths=len(cases), cases=cases, events=events,
        full44_diagnostics_completed=mode == 'final', all44_training_completed_claimed=mode == 'final',
        evaluation_labels_read=False, physical_oracle_calls=0, scientific_signals=0,
        seconds=time.monotonic() - started, source_sha256=sha(Path(__file__)))
    write_json(folder / 'complete.json', result)
    write_json(folder / 'status.json', dict(phase='preflight_complete_not_full44' if mode == 'preflight'
                                           else 'approved44_physical_shapes_complete', owner=owner, **result))
    print(json.dumps(dict(passed=True, mode=mode, cases=len(cases), seconds=result['seconds'])))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
