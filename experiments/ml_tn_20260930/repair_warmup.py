"""Restore available pre-2015 H1 history, preserving all original2016+ bytes."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
from pathlib import Path
from mltn.common import ROOT,read,write,sha
def main():
    ident=read(ROOT/'data/feature_identity.json');dates=pd.DatetimeIndex(ident['dates']);names=ident['features'];old=np.load(ROOT/'data/reach_features.npy',mmap_mode='r');x=np.array(old);before=int((~np.isfinite(x)).sum());raw=ROOT.parent/'20260917_5/data/domains/FULL24C';full_dates=pd.DatetimeIndex(np.load(raw/'dates.npy'));warm=dates.year==2015;fullix=full_dates.get_indexer(dates)
    weather=pd.read_parquet(Path(read(ROOT.parent/'20260916_2/data/drivers/feature_registry.json')['weather_source']),columns=['date','reach_id','precipitation_daily_mm','pet_fao56_mm_day']);weather['date']=pd.to_datetime(weather.date);count=0
    for j,n in enumerate(names):
        if '_past_' not in n:continue
        field,w=n.rsplit('_past_',1);field=field.removeprefix('local_');log=field.startswith('log1p_');field=field.removeprefix('log1p_');path=raw/(field+'.npy')
        if path.exists():v=np.asarray(np.load(path,mmap_mode='r'))
        else:v=weather.pivot(index='date',columns='reach_id',values=field).reindex(index=full_dates,columns=np.arange(1,231)).to_numpy()
        if log:v=np.log1p(v)
        v=v.astype('float32');stat=pd.DataFrame(v).shift(1).rolling(int(w),min_periods=int(w)).mean().to_numpy()[fullix].astype('float32');bad=(~np.isfinite(x[:,:,j]))&warm[:,None]
        if not np.isfinite(stat[bad]).all():raise ValueError('PRE2015_DRIVE_UNAVAILABLE '+n)
        count+=int(bad.sum());view=x[:,:,j];view[bad]=stat[bad]
    np.testing.assert_array_equal(x[~warm],old[~warm]);np.save(ROOT/'data/reach_features_warmup_corrected.npy',x)
    write(ROOT/'evidence/warmup_repair.json',dict(before_missing_values=before,restored_cells=count,after_missing_values=int((~np.isfinite(x)).sum()),identical2016plus=True,original_array_sha256=sha(ROOT/'data/reach_features.npy'),corrected_array_sha256=sha(ROOT/'data/reach_features_warmup_corrected.npy'),scope='Only sequence training windows crossing into2015 change; direct current-row features2016+ and all HF windows2020+ byte-identical',original_array_retained=True,prehistory='frozen H1 full history; no TN consulted'))
    ident['array_file']='reach_features_warmup_corrected.npy';ident['warmup_revision']='available pre2015 physical history restores trailing summaries; annual products and2016+ vectors unchanged';write(ROOT/'data/feature_identity.json',ident)
if __name__=='__main__':main()
