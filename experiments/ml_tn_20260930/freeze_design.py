"""Freeze finite hyperparameter configurations before validation access."""
from mltn.common import ROOT,write,sha
FAMILIES=['RF','ExtraTrees','XGBoost','LightGBM','CatBoost','MLP','LSTM','GRU','TCN','Transformer','GraphTCN']
def main():
    configs={}
    for family in FAMILIES:
        configs[family]=[]
        for i in range(8):
            # Jointly designed configs; finite list, no held-out dependent expansion.
            c=dict(id=i,width=[32,64,64,128,32,64,128,128][i],window=[90,180,365,90,180,365,180,365][i],dropout=[.1,.2,.3,.2,.3,.1,.3,.2][i],lr=[.001,.001,.0005,.0005,.002,.002,.001,.0005][i],weight_decay=[.0001,.001,.001,.0001,.01,.001,.01,.01][i],epochs=120,batch_size=128,patience=18)
            c.update(depth=[4,6,8,10,5,7,12,6][i],min_leaf=[5,10,20,10,30,5,20,40][i],tree_lr=[.03,.05,.05,.03,.1,.1,.05,.03][i],trees=[400,600,800,600,400,600,800,800][i],max_features=[.7,.8,1.,.7,1.,.8,.7,1.][i])
            configs[family].append(c)
    design=dict(families=FAMILIES,configs=configs,direct_tasks=['monthly','daily'],max_screening_fits=176,selection_metric='station/year/month balanced training-scale standardized SSE on 2022 Jan-Sep',screen_train_end='2021-12-31',selection_start='2022-01-01',selection_end='2022-09-30',ensemble_start='2022-10-01',ensemble_end='2022-12-31',seeds=[1729,1730,1731],output='nonnegative: softplus neural; zero-clamped tree/ridge at readout',prediction='known frozen contemporaneous H1; conditional retrospective',joint_coefficients={'monthly':.8,'hf_anomaly':.2,'data_terms':'each half weighted standardized SSE'},early_stop='internal selection only; final folds use frozen screen epoch/iterations, no evaluation access',neural_limit='120 epochs; stop cause distinct from convergence',historical_tn={'daily_leads':[1,7,30],'monthly_leads':[1],'observations':'strictly before origin; monthly reports available after month end conservative convention','annual_closed_loop':True})
    write(ROOT/'config/design.json',design);write(ROOT/'evidence/design_freeze.json',{'sha256':sha(ROOT/'config/design.json'),'validation_scores_read':False})
if __name__=='__main__':main()
