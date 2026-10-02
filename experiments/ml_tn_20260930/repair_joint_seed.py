"""One bounded correction of the original joint-tree matrix, no new grid."""
import argparse, shutil, tarfile
from mltn.common import ROOT, read, write, sha
from controller import execute, jobid
from run_remaining import select_joint

def archive_old():
    base=ROOT/'superseded/seed_not_forwarded'
    for folder in list((ROOT/'jobs').glob('joint_*XGBoost*')):
        dest=base/'jobs'/folder.name
        if dest.exists():
            raise RuntimeError('ARCHIVE_ALREADY_EXISTS '+str(dest))
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(folder),str(dest))
    dest=base/'frozen_joint_selection.json'
    if not dest.exists():shutil.copyfile(ROOT/'outputs/frozen_joint_selection.json',dest)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive-only',action='store_true');ap.add_argument('--already-archived',action='store_true');a=ap.parse_args()
    if not a.already_archived:archive_old()
    if a.archive_only:return
    screen=[dict(role='joint',stage='screen',family='XGBoost',config=c,seed=1729) for c in range(8)]
    assert not execute(screen,'seed_repair_screen')
    best=select_joint(read(ROOT/'outputs/frozen_selection.json')['joint_routes'])
    c=best['XGBoost'];jobs=[]
    for stage in ['F23','F24']:
        for seed in [1729,1730,1731]:jobs.append(dict(role='joint',stage=stage,family='XGBoost',config=c,seed=seed))
    for block in [56,113,191]:
        for stage in ['S23','S24']:
            for seed in [1729,1730,1731]:jobs.append(dict(role='joint',stage=stage,family='XGBoost',config=0,seed=seed,block=block))
    jobs.append(dict(role='joint',stage='F24',family='XGBoost',config=0,seed=1729,fixed_recipe=True))
    assert not execute(jobs,'seed_repair_final')
    assert all((ROOT/'jobs'/jobid(j)/'result.json').exists() for j in screen+jobs)
    files=[p for j in screen+jobs for p in (ROOT/'jobs'/jobid(j)).rglob('*') if p.is_file()]
    files += [ROOT/'outputs/frozen_joint_selection.json',ROOT/'outputs/seed_repair_screen_receipt.json',ROOT/'outputs/seed_repair_final_receipt.json']
    write(ROOT/'evidence/joint_seed_repair.json',dict(passed=True,selected=c,fit_calls=24,logical_paths=33,source_sha256=sha(ROOT/'joint.py'),selection='2022 Jan-Sep only; identical preregistered 8 configurations',files={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    files.append(ROOT/'evidence/joint_seed_repair.json')
    with tarfile.open(ROOT/'transfer/joint_seed_repair.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    print('SEED_REPAIR_DONE',c,flush=True)

if __name__=='__main__':main()
