"""Verify isolated result staging before importing any checkpoint or prediction."""
import json,tarfile,shutil,datetime,sys
from pathlib import Path
from mltn.common import ROOT,read,write,sha

def main():
    archive=ROOT/'transfer/machine_learning_results.tar.gz';receipt=read(ROOT/'transfer/delivery_archive_receipt.json')
    if sha(archive)!=receipt['sha256']:raise RuntimeError('ARCHIVE_BYTE_IDENTITY')
    stage=ROOT/'transfer/import_verified'
    if stage.exists():raise RuntimeError('STAGING_ALREADY_EXISTS_INSPECT_FIRST')
    stage.mkdir()
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            target=(stage/member.name).resolve()
            if not target.is_relative_to(stage.resolve()) or member.issym() or member.islnk():raise RuntimeError('UNSAFE_ARCHIVE_MEMBER '+member.name)
        tf.extractall(stage,filter='data')
    manifest=read(stage/'delivery_manifest.json')
    for relative,value in manifest.items():
        p=(stage/relative).resolve()
        if not p.is_relative_to(stage.resolve()) or sha(p)!=value['sha256'] or p.stat().st_size!=value['bytes']:raise RuntimeError('RESULT_BYTE_IDENTITY '+relative)
    copied=[];prior=ROOT/'superseded/local_preimport'
    for relative in manifest:
        source=stage/relative;target=ROOT/relative
        if not target.resolve().is_relative_to(ROOT.resolve()):raise RuntimeError('IMPORT_OUTSIDE_EXPERIMENT')
        if target.exists() and sha(target)!=sha(source):
            backup=prior/relative
            if backup.exists():raise RuntimeError('BACKUP_COLLISION '+relative)
            backup.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(target,backup)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);copied.append(relative)
    write(ROOT/'evidence/results_recovery.json',dict(passed=True,files=len(copied),archive_sha256=sha(archive),source='authorized workstation same experiment ID',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),byte_verification='all manifest entries before import; no live training sources overwritten'))
    print(len(copied))
if __name__=='__main__':main()
