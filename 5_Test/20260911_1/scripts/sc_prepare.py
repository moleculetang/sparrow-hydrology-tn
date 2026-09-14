import os
from pathlib import Path
os.environ['NUMBA_CACHE_DIR']=str(Path(__file__).resolve().parents[1]/'work/numba_cache')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gc
import numpy as np
import pandas as pd
import torch
from fc_io import *
from fc_data import observations,prediction_metadata
from sc_model import SharedBox,data_cache,reference_scale,CONTRACT,SIGMA

def main():
    require_environment();torch.set_num_threads(1)
    if (RUN/'reports/preparation.json').exists():raise RuntimeError('ALREADY_PREPARED')
    obs=observations();data=data_cache('sensitivity');assert str(data.dates[0])[:10]=='1961-01-01'
    # This cache is the original parameter-free forcing Data, not a T/Y bundle.
    assert not isinstance(data,dict) and data.forcing=='S1'
    policy={'response':CONTRACT,'folds':{'F23':{'train':[2021,2022],'eval':2023,'counts':[2606,1090]},
                                      'F24':{'train':[2021,2023],'eval':2024,'counts':[3696,1138]}},
      'paths':8,'capacity_calls':6,'max_calls':[4000,6000,8000],'max_hours':[4,6,8],
      'campaign_hours':36,'report_hours':2,'threads':1,'dispatch':90,'yield':90,'resume':85,'peak_factor':1.2,
      'projected_gradient':1e-5,'minimum_objective_tolerance':1e-8,'objective':'RAW+MAP',
      'bootstrap':{'n':1000,'seed':1729,'unit':'paired_whole_station_within_water_tree'},'automatic_promotion':False}
    immutable(RUN/'configs/campaign.json',policy)
    cache=read(RUN/'reports/cache_manifest.json')['products']['sensitivity']['data'];assert sha(cache['path'])==cache['sha256']
    register(cache['path'],'parameter_free_frozen_hydrology_and_S1_source_cache')
    register(TEST/'20260906_1/outputs/observations.parquet','historically_seen_original_TN_split_source')
    register(TEST/'20260905_1/outputs/h7_raw_features.parquet','raw_environment_attributes')
    folds={}
    for fold,spec in policy['folds'].items():
        train=obs.loc[obs.primary_gate & obs.year.between(*spec['train'])].sort_values(['station_key','year','month','observation_id']).reset_index(drop=True)
        val=obs.loc[obs.primary_gate & obs.year.eq(spec['eval'])].sort_values(['station_key','year','month','observation_id']).reset_index(drop=True)
        assert [len(train),len(val)]==spec['counts'] and train.station_key.nunique()==val.station_key.nunique()==116
        assert not set(train.observation_id)&set(val.observation_id)
        for typ,frame in [('train',train),('heldout_labels',val),('heldout_metadata',prediction_metadata(val))]:
            frame.to_parquet(local(RUN/f'outputs/data/{fold}_{typ}.parquet'),index=False)
        model=SharedBox(data,train);assert len(model.names)==21
        scale=reference_scale(model)
        sc=SharedBox(data,train,'SC',scale)
        init={'M0_0':model.initial(0).tolist(),'M0_1':model.initial(1).tolist(),
              'SC_1':np.r_[model.initial(1),SIGMA*np.array([1,-1,1,-1,1,-1])].tolist()}
        config={'fold':fold,'scale':scale.tolist(),'designs':{'M0':model.design,'SC':sc.design},'initials':init,
           'names':{'M0':model.names.copy(),'SC':sc.names.copy()},'train_sha256':sha(RUN/f'outputs/data/{fold}_train.parquet'),
           'metadata_sha256':sha(RUN/f'outputs/data/{fold}_heldout_metadata.parquet'),'cache':cache,
           'reference':'preset0 uncalibrated; no old fit/anchor/inversion state'}
        immutable(RUN/f'configs/{fold}.json',config)
        model.loss.ledger.to_parquet(local(RUN/f'outputs/data/{fold}_training_weights.parquet'),index=False)
        folds[fold]={'train_ids':train.observation_id.tolist(),'evaluation_ids':val.observation_id.tolist(),
          'files':{str(p):sha(p) for p in (RUN/'outputs/data').glob(f'{fold}_*.parquet')},'config_sha256':sha(RUN/f'configs/{fold}.json')}
        del model,sc;gc.collect()
    # Algebraic counterexamples: same instantaneous old-P inputs, different antecedent;
    # same wetness/antecedent, different inventory for old eight-term water-only family.
    from sc_kernel import features
    b1,_=features(10.,5.,.6,.1);b2,_=features(10.,5.,.6,-.1);b3,_=features(30.,5.,.6,.1)
    assert b1[2]!=b2[2] and b1[0]!=b3[0]
    write(RUN/'reports/function_equivalence.json',{'status':'PASS_DISTINCT_FUNCTION_CLASSES',
      'old_P_same_W_fastfraction_A_different_pastW':[b1.tolist(),b2.tolist()],
      'old_8wet_same_W_pastW_different_A':[b1.tolist(),b3.tolist()],
      'limitation':'Functional non-equivalence is not evidence of predictive benefit; some subspaces overlap.'})
    for p in [RUN/'PLAN.md',RUN/'configs/campaign.json',RUN/'configs/F23.json',RUN/'configs/F24.json']:
        register(p,'registered_plan_and_fold_preprocessing')
    write(RUN/'reports/preparation.json',{'status':'PREPARED','folds':folds,'utc':now(),'forcing_start':str(data.dates[0]),
        'forcing_end':str(data.dates[-1]),'forcing_attribute_names':sorted(vars(data)),
        'scope':'No predictive fit or T/Y solve yet'})
    print('PREPARED_TWO_ISOLATED_FOLDS',flush=True)
if __name__=='__main__':main()
