"""Actual corrected seed diversity plus same-seed reproducibility control."""
import numpy as np,pandas as pd,xgboost as xgb
from mltn.common import ROOT,read,write,sha
def main():
    x=np.random.default_rng(1).normal(size=(100,4));y=x[:,0]+x[:,1]**2;dm=xgb.DMatrix(x,label=y)
    def fitted(seed):return xgb.train(dict(seed=seed,subsample=.8,colsample_bytree=.8,max_depth=3,nthread=1),dm,num_boost_round=10).predict(dm)
    a=fitted(1729);np.testing.assert_array_equal(a,fitted(1729));assert not np.array_equal(a,fitted(1730))
    c=read(ROOT/'outputs/frozen_joint_selection.json')['selected']['XGBoost'];records=[]
    for fold in ['F23','F24']:
        frames=[]
        for seed in [1729,1730,1731]:
            folder=ROOT/'jobs'/f'joint_{fold}_XGBoost_c{c}_s{seed}';start=read(folder/'start.json')
            assert start['seed']==seed and start['code_sha256']['joint.py']==sha(ROOT/'joint.py')
            q=pd.read_parquet(folder/'prediction_daily_hf.parquet');frames.append(q)
        for q in frames:assert q[['station_key','date','tn_mg_l']].equals(frames[0][['station_key','date','tn_mg_l']])
        differences=[float(np.max(np.abs(frames[i].prediction.to_numpy()-frames[j].prediction.to_numpy()))) for i,j in [(0,1),(0,2),(1,2)]]
        assert min(differences)>0
        records.append(dict(fold=fold,pair_max_abs_differences=differences))
    write(ROOT/'evidence/joint_seed_forwarding_acceptance.json',dict(passed=True,same_seed_repeat_exact=True,different_seed_control=True,actual_paths=records,source_sha256=sha(ROOT/'joint.py')))
    print('ACTUAL_SEED_DIVERSITY_PASSED',flush=True)
if __name__=='__main__':main()
