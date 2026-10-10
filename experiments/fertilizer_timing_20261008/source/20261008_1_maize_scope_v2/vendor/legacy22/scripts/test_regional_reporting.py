"""Independent evaluator and spatial support fixture using synthetic observations only."""
import shutil
import numpy as np,pandas as pd
import native_runtime as rt
import evaluation as ev
import audit_regional_evaluation as audit

def main():
    source=rt.RUN/'work/evaluation_pipeline_fixture';root=rt.RUN/'work/regional_reporting_fixture';root.mkdir(exist_ok=True)
    shutil.copytree(source/'data',root/'data',dirs_exist_ok=True)
    for tag in ['R','X']:
        path=root/'outputs'/tag;path.mkdir(parents=True,exist_ok=True);df=pd.read_parquet(source/'outputs'/tag/'daily_station_mass_water.parquet');df['mass_kg_day']=df.concentration_mg_l*df.water_m3_day/1000;df.to_parquet(path/'daily_station_mass_water.parquet',index=False)
    ev.R=root;audit.R=root
    rt.write(root/'data/selected.json',{'F24_U':'R','F24_L3':'X','S56_U':'R','S56_L3':'X'})
    rt.write(root/'data/prediction_freeze.json',{'files':{f'outputs/{tag}/daily_station_mass_water.parquet':rt.sha(root/'outputs'/tag/'daily_station_mass_water.parquet') for tag in ['R','X']}})
    rt.write(root/'evidence/inherited_manifest.json',{})
    for block in [1,2]:
        ev.main({'F24_R':'R','F24_X':'X'},'L3-U',block)
        ev.main({'S56_R':'R','S56_X':'X'},'S56_L3-U',block,scopes=[('S56',2024)],held_stations=['s00','s01'])
    audit.main();result=rt.read(root/'reports/independent_regional_evaluation.json');assert result['status']=='PASS'
    rt.write(rt.RUN/'reports/regional_reporting_fixture.json',dict(status='PASS',synthetic_only=True,spatial_filter=True,independent=result))
    print('PASS regional reporting fixture',flush=True)
if __name__=='__main__':main()
