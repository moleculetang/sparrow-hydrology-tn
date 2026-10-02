import pickle,numpy as np,torch
from mltn.models import Network
from train import predict_network,NEURAL
def direct_predict(folder,d,q,device='cpu'):
    ck=pickle.load((folder/'checkpoint.pkl').open('rb'));cfg=ck['cfg'];family=ck['args']['family'];tr=ck['transform']
    if family in NEURAL:
        model=Network(family,len(tr.mean)*2,cfg).to(device);model.load_state_dict(torch.load(folder/'weights.pt',map_location=device,weights_only=True));return predict_network(model,d,q,cfg,family,tr,device)
    return np.maximum(0,ck['model'].predict(tr.apply(d.raw_rows(q,ck['args']['task']=='monthly'))))
