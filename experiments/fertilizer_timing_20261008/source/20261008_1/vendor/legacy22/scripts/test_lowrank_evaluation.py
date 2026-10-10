"""Synthetic end-to-end masked mode forecast reporting; no real holdout labels."""
import shutil
import numpy as np,pandas as pd
import native_runtime as rt
import evaluate_lowrank as ev

def main():
    root=rt.RUN/'work/lowrank_evaluation_fixture';source=rt.RUN/'work/regional_reporting_fixture'
    shutil.copytree(source/'data',root/'data',dirs_exist_ok=True);shutil.copytree(source/'outputs/R',root/'outputs/R',dirs_exist_ok=True)
    dates=pd.date_range('2024-01-01','2024-12-31');ss=['s00','s01','s02'];folder=root/'diagnostics/lowrank/F24/daily/anomaly';folder.mkdir(parents=True,exist_ok=True)
    frames=[]
    for name in ['unrestricted','rank1','rank2','rank3']:frames.append(pd.DataFrame(dict(date=np.repeat(dates,3),station_key=np.tile(ss,len(dates)),model=name,predicted_transformed_component=np.log(3.))))
    pd.concat(frames).to_parquet(folder/'forecast.parquet',index=False)
    np.savez_compressed(folder/'training.npz',stations=np.array(ss),scale=np.ones(3),seasonal=np.zeros((5,3)))
    for rank in [1,2,3]:np.savez_compressed(folder/f'rank{rank}.npz',loadings=np.eye(3)[:,:rank])
    rt.write(folder/'identity.json',dict(baseline='R',scale='daily',target='anomaly'))
    rt.write(root/'reports/lowrank_freeze.json',dict(files={p.relative_to(root).as_posix():rt.sha(p) for p in folder.glob('*')}))
    ev.R=root;ev.main();table=pd.read_csv(root/'reports/lowrank_evaluation.csv');assert {'INPUT_ONLY_FORECAST','OBSERVATION_ASSISTED_RECONSTRUCTION_NOT_FORECAST'}==set(table.role)
    assert (table[table.role.eq('INPUT_ONLY_FORECAST')].negative_fraction==0).all()
    rt.write(rt.RUN/'reports/lowrank_evaluation_fixture.json',dict(status='PASS',forecast_and_reconstruction_separate=True,synthetic_only=True))
    print('PASS lowrank evaluation fixture',flush=True)
if __name__=='__main__':main()
