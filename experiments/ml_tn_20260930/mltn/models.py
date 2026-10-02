import numpy as np,torch
from torch import nn
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
class CausalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.pad=2*dilation;self.conv=nn.Conv1d(width,width,3,dilation=dilation);self.norm=nn.LayerNorm(width)
    def forward(self,x):
        y=self.conv(nn.functional.pad(x.transpose(1,2),(self.pad,0))).transpose(1,2)
        return self.norm(x+torch.relu(y))
class Network(nn.Module):
    def __init__(self,family,features,cfg):
        super().__init__();self.family=family;w=cfg['width'];self.proj=nn.Linear(features,w);self.dropout=nn.Dropout(cfg['dropout'])
        if family in ['LSTM','GRU']:self.core=getattr(nn,family)(w,w,batch_first=True)
        elif family in ['TCN','GraphTCN']:self.core=nn.Sequential(*[CausalBlock(w,2**k) for k in range(8)])
        elif family=='Transformer':self.core=nn.TransformerEncoder(nn.TransformerEncoderLayer(w,4,w*2,cfg['dropout'],batch_first=True),2,enable_nested_tensor=False)
        else:self.core=nn.Sequential(nn.Linear(w,w),nn.ReLU(),nn.Dropout(cfg['dropout']),nn.Linear(w,w),nn.ReLU())
        self.head=nn.Linear(w,1)
    def forward(self,x):
        z=self.dropout(torch.relu(self.proj(x)))
        if self.family in ['LSTM','GRU']:z=self.core(z)[0][:,-1]
        elif self.family in ['TCN','GraphTCN']:z=self.core(z)[:,-1]
        elif self.family=='Transformer':
            n=z.shape[1];pos=torch.arange(n,device=z.device,dtype=z.dtype)[:,None];freq=torch.exp(torch.arange(0,z.shape[-1],2,device=z.device,dtype=z.dtype)*(-np.log(10000.)/z.shape[-1]))
            enc=torch.zeros_like(z[0]);enc[:,0::2]=torch.sin(pos*freq);enc[:,1::2]=torch.cos(pos*freq)
            z=self.core(z+enc,mask=nn.Transformer.generate_square_subsequent_mask(n,device=z.device))[:,-1]
        else:z=self.core(z[:,-1] if z.ndim==3 else z)
        return nn.functional.softplus(self.head(z).squeeze(-1))
def tree(family,cfg,seed,threads,device='cpu'):
    if family in ['RF','ExtraTrees']:
        from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
        cls=RandomForestRegressor if family=='RF' else ExtraTreesRegressor
        return cls(n_estimators=cfg['trees'],max_depth=cfg['depth'],min_samples_leaf=cfg['min_leaf'],max_features=cfg['max_features'],random_state=seed,n_jobs=threads)
    if family=='XGBoost':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=cfg['trees'],max_depth=cfg['depth'],min_child_weight=cfg['min_leaf'],learning_rate=cfg['tree_lr'],subsample=.8,colsample_bytree=cfg['max_features'],reg_lambda=1.,random_state=seed,n_jobs=threads,device=device,tree_method='hist',objective='reg:squarederror')
    if family=='LightGBM':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=cfg['trees'],max_depth=cfg['depth'],num_leaves=min(2**cfg['depth'],127),min_child_samples=cfg['min_leaf'],learning_rate=cfg['tree_lr'],colsample_bytree=cfg['max_features'],random_state=seed,n_jobs=threads,verbosity=-1)
    if family=='CatBoost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=cfg['trees'],depth=min(cfg['depth'],10),learning_rate=cfg['tree_lr'],l2_leaf_reg=3,random_seed=seed,thread_count=threads,verbose=False,allow_writing_files=False,task_type='GPU' if device=='cuda' else 'CPU')
    raise ValueError(family)
