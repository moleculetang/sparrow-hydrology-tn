"""Terminal transport negative controls; no experiment checkpoints are loaded."""
import copy
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from campaign_return import MANIFEST,digest,import_archive,validate_manifest
from d29_platform.runtime import write_json,sha


def archive(path,manifest,payloads,extra=None):
    with tarfile.open(path,'w') as t:
        raw=json.dumps(manifest).encode();m=tarfile.TarInfo(MANIFEST);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        for name,raw in payloads.items():
            m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        if extra:t.addfile(extra,io.BytesIO(b''))


def main():
    checks={};gate='a'*64;ownership='b'*64;jobs=['TIM_FIXTURE']
    payloads={'outputs/campaign_workstation/status.json':b'{"phase":"fixture_terminal_not_convergence"}',
              'outputs/campaign_workstation/owner.json':b'{}',
              'outputs/jobs/TIM_FIXTURE/best.npy':b'explicit_test_payload_not_a_numpy_model'}
    base=dict(version=1,host_role='workstation',owned_paths=jobs,formal_gate_sha256=gate,
        ownership_sha256=ownership,terminal_phase='fixture_terminal_not_convergence',
        files=[dict(path=n,bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in payloads.items()],
        actual_bytes=sum(map(len,payloads.values())))
    with tempfile.TemporaryDirectory(prefix='campaign_transport_',dir=ROOT/'cache') as tmp:
        tmp=Path(tmp);good=tmp/'good.tar';archive(good,base,payloads)
        receipt=import_archive(good,digest(good),tmp/'accepted',jobs,gate,ownership)
        assert receipt['passed'] and receipt['transport_receipt_is_numerical_convergence_certificate'] is False
        again=import_archive(good,digest(good),tmp/'accepted',jobs,gate,ownership);assert again['passed']
        checks['exact_manifest_byte_import_and_identical_resume']=True
        negative=[]
        raw=copy.deepcopy(base);raw['formal_gate_sha256']='c'*64;negative.append(raw)
        raw=copy.deepcopy(base);raw['ownership_sha256']='c'*64;negative.append(raw)
        raw=copy.deepcopy(base);raw['files'].append(raw['files'][0]);raw['actual_bytes']+=raw['files'][0]['bytes'];negative.append(raw)
        for path in ['../outside','data/monthly_tn.parquet','outputs/jobs/FOREIGN/best.npy','outputs/jobs/TIM_FIXTURE/../SECRET',r'outputs\jobs\TIM_FIXTURE\secret','C:/credential']:
            raw=copy.deepcopy(base);raw['files'][2]['path']=path;negative.append(raw)
        for raw in negative:
            try:validate_manifest(raw,jobs,gate,ownership)
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('unsafe or identity-changing manifest accepted')
        checks['foreign_path_credentials_traversal_duplicate_and_identity_changes_rejected']=True
        changed=dict(payloads);changed['outputs/jobs/TIM_FIXTURE/best.npy']=b'x'*len(changed['outputs/jobs/TIM_FIXTURE/best.npy'])
        bad=tmp/'changed.tar';archive(bad,base,changed)
        try:import_archive(bad,digest(bad),tmp/'tampered',jobs,gate,ownership)
        except RuntimeError:pass
        else:raise AssertionError('tampered content passed')
        assert not (tmp/'tampered/outputs/jobs/TIM_FIXTURE/best.npy').exists()
        checks['payload_hash_failure_cannot_promote_partial_file']=True
        extra=tarfile.TarInfo('outputs/jobs/TIM_FIXTURE/best.npy');extra.size=0
        dup=tmp/'duplicate.tar';archive(dup,base,payloads,extra)
        try:import_archive(dup,digest(dup),tmp/'duplicate',jobs,gate,ownership)
        except RuntimeError:pass
        else:raise AssertionError('duplicate tar member passed')
        checks['duplicate_archive_members_rejected']=True
        link=tmp/'symlink.tar'
        with tarfile.open(link,'w') as t:
            raw=json.dumps(base).encode();m=tarfile.TarInfo(MANIFEST);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
            for n,v in payloads.items():
                m=tarfile.TarInfo(n)
                if n.endswith('best.npy'):m.type=tarfile.SYMTYPE;m.linkname='../../outside';t.addfile(m)
                else:m.size=len(v);t.addfile(m,io.BytesIO(v))
        try:import_archive(link,digest(link),tmp/'linked',jobs,gate,ownership)
        except RuntimeError:pass
        else:raise AssertionError('symlink tar member passed')
        checks['archive_symlink_cannot_redirect_import']=True
        old=tmp/'accepted/outputs/jobs/TIM_FIXTURE/best.npy';old.write_bytes(b'preserve prior bytes')
        try:import_archive(good,digest(good),tmp/'accepted',jobs,gate,ownership)
        except RuntimeError:pass
        else:raise AssertionError('conflicting existing immutable import passed')
        assert old.read_bytes()==b'preserve prior bytes'
        checks['immutable_existing_conflict_never_overwritten']=True
    result=dict(passed=all(checks.values()),checks=checks,only_explicit_transport_fixture_used=True,
        original_training_and_remote_processes_untouched=True,optimizer_pickles_deserialized=False,
        identities={n:sha(ROOT/n) for n in ['deployment/campaign_return.py','deployment/test_campaign_return.py']})
    write_json(ROOT/'evidence/campaign_return_transport_controls.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
