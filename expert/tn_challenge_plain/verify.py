"""Portable verification. Uses archived references only for parity, never fitting."""
import json,hashlib,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from model import HERE,load_data,Objective,Predictor
def main():
    hashes=json.loads((HERE/'data/manifest.json').read_text(encoding='utf-8'))
    for name,h in hashes.items():
        assert hashlib.sha256((HERE/'data'/name).read_bytes()).hexdigest()==h,name
    d=load_data();train=pd.read_csv(HERE/'data/train.csv');sc=Objective(d,train,'SC');m0=Objective(d,train,'M0')
    x=m0.initial(0);np.testing.assert_allclose(m0.predict(x,m0.meta),sc.predict(np.r_[x,np.zeros(6)],sc.meta),rtol=0,atol=0)
    x=sc.initial(1);x[-6:]=[.01,-.01,.01,-.01,.01,-.01];J,g=sc.value_gradient(x)
    direction=np.random.default_rng(1729).normal(size=len(x));direction/=np.linalg.norm(direction);h=1e-5
    fd=(sc.value_gradient(x+h*direction)[0]-sc.value_gradient(x-h*direction)[0])/(2*h)
    error=abs(fd-g@direction)/max(1,abs(fd),abs(g@direction));assert error<2e-5
    ledger=sc.ledger(x);assert ledger['local_balance_max_kg']<1e-5 and abs(ledger['network_balance_kg'])<1e-3
    for variant in ['M0','SC']:
        s=json.loads((HERE/f'reference_only/F23_{variant}_0.json').read_text(encoding='utf-8'))
        expected=pd.read_csv(HERE/f'reference_only/F23_{variant}_0_predictions.csv')
        meta=expected.drop(columns=['tn_mg_l','prediction_mg_l']);p=Predictor(d,s['design'],variant).predict(s['theta'],meta)
        np.testing.assert_allclose(p,expected.prediction_mg_l,rtol=2e-10,atol=2e-10)
    sets=[set(pd.read_csv(HERE/f'data/{s}.csv').observation_id) for s in ['train','development','evaluation','hindcast']]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    # Evaluation TN can be arbitrary; prediction rejects it, and the train-only
    # objective is unchanged. No evaluation label file is an Objective input.
    evaluation=pd.read_csv(HERE/'data/evaluation.csv');before=sc.value_gradient(x)
    evaluation['tn_mg_l']=evaluation.tn_mg_l*100+123
    try:sc.predict(x,evaluation)
    except ValueError:pass
    else:raise AssertionError('Prediction accepted labels')
    after=sc.value_gradient(x);np.testing.assert_array_equal(before[1],after[1]);assert before[0]==after[0]
    print(json.dumps({'status':'PASS','directional_gradient_relative_error':error,'data_hashes':len(hashes),
      'local_mass_balance_max_kg':ledger['local_balance_max_kg'],'network_balance_kg':ledger['network_balance_kg'],
      'checks':['M0 zero extension','reference predictions','full-history derivative','mass balance','split IDs','label rejection and objective isolation']}))
if __name__=='__main__':main()
