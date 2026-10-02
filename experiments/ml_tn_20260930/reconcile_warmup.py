"""At safe direct-controller completion boundary, rerun only affected input-cohort jobs."""
import shutil
from mltn.common import ROOT,read,write
from controller import execute,screen_jobs,select,final_jobs
SEQUENCE=['LSTM','GRU','TCN','Transformer','GraphTCN']
def main():
    if (ROOT/'outputs/warmup_reconciliation.json').exists():return
    if not (ROOT/'data/reach_features_warmup_corrected.npy').exists():raise RuntimeError('WARMUP_REPAIR_NOT_PRESENT')
    affected=[]
    archive=ROOT/'superseded/warmup_incomplete';archive.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'jobs').glob('*'):
        if (p/'owner.lock').exists():
            from mltn.ownership import reclaim
            reclaim(p,clear_failure=False)
        st=p/'start.json'
        if not st.exists():continue
        s=read(st)
        family=s.get('family',s.get('job_id','').split('_')[2] if len(s.get('job_id','').split('_'))>=3 else '')
        if s.get('task')=='monthly' and family in SEQUENCE:
            if s.get('feature_array_file','reach_features.npy')=='reach_features.npy' or s.get('stage')!='screen':
                dest=archive/p.name
                if dest.exists():raise RuntimeError('ARCHIVE_COLLISION')
                if not p.resolve().is_relative_to(ROOT.resolve()) or not dest.resolve().is_relative_to(ROOT.resolve()):raise RuntimeError('ARCHIVE_OUTSIDE_EXPERIMENT')
                shutil.move(str(p),str(dest));affected.append(p.name)
    # Selection itself can change after fixing affected screen fits; retain old receipt.
    if (ROOT/'outputs/frozen_selection.json').exists():shutil.copyfile(ROOT/'outputs/frozen_selection.json',archive/'prior_selection.json')
    execute(screen_jobs(),'screen_corrected');s=select()
    # Old final paths for sequences using correct input but an old-selected config are
    # descriptive superseded trials. Current matrix only chooses newly frozen configs.
    execute(final_jobs(s),'direct_final_corrected');write(ROOT/'outputs/warmup_reconciliation.json',dict(archived_input_cohort_jobs=affected,reason='available prehistory was needlessly missing in2015 sequence context; all current feature vectors2016+ unchanged',selection='same finite grid; implementation repair repeats are not new scientific configurations',selected=s['selected']))
if __name__=='__main__':main()
