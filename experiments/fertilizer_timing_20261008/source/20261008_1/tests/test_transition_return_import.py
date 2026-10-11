"""Known-error controls for repaired remote returns; no scientific dispatch."""
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
import import_diagnostics as importer
from d29_platform.runtime import sha,write_json


def archive(path,members):
    with tarfile.open(path,'w:gz') as tar:
        for name,content,kind in members:
            item=tarfile.TarInfo(name);item.type=kind
            if kind==tarfile.REGTYPE:
                item.size=len(content);tar.addfile(item,io.BytesIO(content))
            else:item.linkname='outside';tar.addfile(item)
    return sha(path)


def rejected(call,phrase):
    try:call()
    except RuntimeError as exc:
        if phrase not in str(exc):raise
    else:raise AssertionError('known invalid return accepted: '+phrase)


def main():
    checks={};prefix='outputs/jobs/transition_requalification_TIM_F23_C0_V0_s0/'
    status='outputs/transition_requalification_workstation/status.json'
    with tempfile.TemporaryDirectory(prefix='transition_return_',dir=ROOT/'cache') as temporary:
        base=Path(temporary);file=base/'return.tar.gz'
        write_json(base/'config/jobs.json',[])
        with patch.object(importer,'ROOT',base):
            good=[(prefix+'gradient.npy',b'frozen fixture bytes',tarfile.REGTYPE),(status,b'{"phase":"fixture"}',tarfile.REGTYPE)]
            digest=archive(file,good)
            rejected(lambda:importer.main(file,'0'*64,'transition'),'SHA256_MISMATCH')
            checks['wrong_archive_hash_rejected']=True
            importer.main(file,digest,'transition')
            target=base/'outputs/imported_workstation_transition'/prefix/'gradient.npy'
            assert target.read_bytes()==b'frozen fixture bytes'
            receipt=base/'evidence/workstation_transition_import.json';saved=receipt.read_bytes()
            importer.main(file,digest,'transition');assert target.read_bytes()==b'frozen fixture bytes'
            checks['valid_registered_return_and_idempotent_import']=True
            for name,content,kind,phrase in [
                ('../escaped',b'bad',tarfile.REGTYPE,'UNSAFE_RETURN_ARCHIVE_PATH'),
                ('/escaped',b'bad',tarfile.REGTYPE,'UNSAFE_RETURN_ARCHIVE_PATH'),
                (prefix+'linked',b'',tarfile.SYMTYPE,'LINK_OR_SPECIAL'),
                ('data/monthly_tn.parquet',b'bad',tarfile.REGTYPE,'UNREGISTERED_DIAGNOSTIC')]:
                digest=archive(file,[(name,content,kind)])
                rejected(lambda:importer.main(file,digest,'transition'),phrase)
            checks['traversal_absolute_link_and_unregistered_paths_rejected']=True
            digest=archive(file,[good[0],good[0]])
            rejected(lambda:importer.main(file,digest,'transition'),'DUPLICATE_RETURN')
            checks['duplicate_member_rejected']=True
            digest=archive(file,[(prefix+'gradient.npy',b'changed',tarfile.REGTYPE)])
            rejected(lambda:importer.main(file,digest,'transition'),'EXISTING_IMPORT_BYTE_CONFLICT')
            assert target.read_bytes()==b'frozen fixture bytes' and receipt.read_bytes()==saved
            checks['changed_existing_bytes_rejected_and_original_retained']=True
    write_json(ROOT/'evidence/transition_return_import_known_errors.json',dict(passed=all(checks.values()),
        scope='known_error_remote_transport_controls_not_scientific_derivative_or_fit',cases=checks,
        scientific_jobs_started=0,identities={'deployment/import_diagnostics.py':sha(ROOT/'deployment/import_diagnostics.py')}))
    print(json.dumps(dict(passed=all(checks.values()),cases=checks)))


if __name__=='__main__':main()
