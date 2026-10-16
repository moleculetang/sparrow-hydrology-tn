"""Verify and import private completed diagnostic receipts, never checkpoints."""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import sha,write_json


def main(archive,expected,kind='baseline'):
    archive=Path(archive)
    if sha(archive)!=expected:raise RuntimeError('RETURN_ARCHIVE_SHA256_MISMATCH')
    specifications={
        'baseline':('imported_workstation_diagnostics',('outputs/jobs/diagnostic_TIM_F23_C0_V0_s0/','outputs/jobs/diagnostic_TIM_F23_K_V1_s0/'),'outputs/remote_prerequisites/status.json','workstation_diagnostic_import'),
        'threads':('imported_workstation_threads',('outputs/jobs/diagnostic_threads2_TIM_F23_K_V1_s0/','outputs/jobs/diagnostic_threads4_TIM_F23_K_V1_s0/'),'outputs/remote_thread_probe/status.json','workstation_thread_import'),
        'coordinates':('imported_workstation_coordinates',tuple(f'outputs/jobs/prerequisite_K_coordinates_{i}/' for i in range(6)),'outputs/remote_coordinates/status.json','workstation_coordinate_import')}
    jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    replay_prefixes=tuple('outputs/jobs/replay_'+j['id']+'/' for j in jobs if j['space_block'] is None and j['calendar']!='K' and j['entry']==0)
    specifications['replays']=('imported_workstation_replays',replay_prefixes,'outputs/replays_workstation/status.json','workstation_replay_import')
    specifications['measurements']=('imported_workstation_measurements',tuple(f'outputs/jobs/measurement_threads{n}_TIM_F23_C0_V0_s0/'
        for n in (1,2,4)),'outputs/measurements_workstation/status.json','workstation_measurement_import')
    specifications['transition']=('imported_workstation_transition',tuple('outputs/jobs/transition_requalification_'+j+'/'
        for j in ('TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0')),
        'outputs/transition_requalification_workstation/status.json','workstation_transition_import')
    specifications['pilot']=('imported_workstation_parallel_targets',
        ('outputs/parallel_target_pilot/',)+tuple(f'outputs/jobs/parallel_target_workers{n}_v1_{i}/'
            for n in (4,8,12) for i in range(n)),
        'outputs/parallel_target_pilot/status.json','workstation_parallel_target_import')
    if kind not in specifications:raise ValueError('REGISTERED_IMPORT_KIND_REQUIRED')
    namespace,allowed,statusfile,receiptfile=specifications[kind]
    target=ROOT/'outputs'/namespace;target.mkdir(parents=True,exist_ok=True)
    files=[]
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers()
        seen=set()
        for member in members:
            if member.name in seen:raise RuntimeError('DUPLICATE_RETURN_ARCHIVE_PATH')
            seen.add(member.name)
            p=PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts or '\\' in member.name or ':' in member.name:
                raise RuntimeError('UNSAFE_RETURN_ARCHIVE_PATH')
            if not (member.isdir() or member.isfile()):raise RuntimeError('LINK_OR_SPECIAL_RETURN_FILE_REJECTED')
            if member.isfile() and not (member.name.startswith(allowed) or member.name==statusfile):
                raise RuntimeError('UNREGISTERED_DIAGNOSTIC_ARTIFACT '+member.name)
        for member in members:
            if not member.isfile():continue
            destination=(target/member.name).resolve()
            if not destination.is_relative_to(target.resolve()):raise RuntimeError('OUTSIDE_IMPORT_ROOT')
            destination.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256()
            with stream.extractfile(member) as source:
                if destination.exists():
                    for block in iter(lambda:source.read(4*1024**2),b''):digest.update(block)
                    if sha(destination)!=digest.hexdigest():raise RuntimeError('EXISTING_IMPORT_BYTE_CONFLICT')
                else:
                    with destination.open('xb') as handle:
                        for block in iter(lambda:source.read(4*1024**2),b''):digest.update(block);handle.write(block)
            files.append(dict(path=member.name,bytes=member.size,sha256=sha(destination)))
    receipt=dict(passed=True,scope='verified_workstation_completed_diagnostic_import_not_fit',
        archive=str(archive),archive_sha256=expected,import_root=str(target),files=files,
        live_checkpoint_imported=False,scientific_fit_paths_dispatched=0)
    receipt['kind']=kind;write_json(ROOT/'evidence'/f'{receiptfile}.json',receipt)
    print(json.dumps(dict(passed=True,files=len(files),bytes=sum(f['bytes'] for f in files))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('sha256');p.add_argument('--kind',choices=['baseline','threads','coordinates','replays','measurements','transition','pilot'],default='baseline')
    a=p.parse_args();main(a.archive,a.sha256,a.kind)
