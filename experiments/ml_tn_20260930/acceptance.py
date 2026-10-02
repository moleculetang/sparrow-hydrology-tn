import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,copy,platform,pickle,unittest
import numpy as np,pandas as pd,torch
from scipy.sparse import csr_matrix
from mltn.common import ROOT,read,write,sha
from mltn.data import Inputs,Transform,balanced_weights,weights_and_scales
from mltn.objectives import AggregateObjective
from mltn.models import Network
class Gates(unittest.TestCase):
    def test_independent_group_gradient(self):
        b=np.zeros((2,8));b[0,:4]=.25;b[1,4:]=.25;h=np.eye(8)[[0,1,2,4,5,6]]
        o=AggregateObjective(b,[1,2],[.3,.7],h,[1,4,2,3,1,5],[.1,.2,.3,.2,.1,.1],[0,0,0,1,1,1],[1,3,2,2,1,4]);p=np.arange(8)*.2
        j,g=o.value_gradient(p)
        for step in [1e-3,1e-4,1e-5]:
            fd=np.array([(o.value_gradient(p+np.eye(8)[i]*step)[0]-o.value_gradient(p-np.eye(8)[i]*step)[0])/(2*step) for i in range(8)])
            np.testing.assert_allclose(fd,g,rtol=1e-6,atol=1e-9)
        ht=np.column_stack([(o.value_gradient(p+np.eye(8)[i]*1e-4)[1]-o.value_gradient(p-np.eye(8)[i]*1e-4)[1])/(2e-4) for i in range(8)])
        np.testing.assert_allclose(o.diagonal(),np.diag(ht),atol=1e-9)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(np.diag(o.diagonal_majorizer())-ht).min()),-1e-10)
        # An artificial month-common leaf exposes the unsafe exact-diagonal step.
        single=AggregateObjective(np.ones((1,30))/30,[10.],[1.],np.zeros((0,30)),[],[],[],[]);initial=np.ones(30);j0,grad=single.value_gradient(initial)
        old=initial*np.exp(-.1*grad.sum()/single.diagonal().sum());safe=initial*np.exp(-.1*grad.sum()/single.diagonal_majorizer().sum())
        self.assertGreater(single.value_gradient(old)[0],j0);self.assertLess(single.value_gradient(safe)[0],j0)
        t=torch.tensor(p,dtype=torch.float64,requires_grad=True);bb=torch.tensor(b);hh=torch.tensor(h);v=hh@t-torch.tensor(o.yh)
        centered=[]
        for group in np.unique(o.groups):
            ix=np.flatnonzero(o.groups==group);a=torch.tensor(o.counts[ix]/o.counts[ix].sum());z=v[ix];centered.append(z-torch.dot(a,z))
        e=torch.cat(centered);jt=.4*torch.sum(torch.tensor(o.wm)*(bb@t-torch.tensor(o.ym))**2)+.1*torch.sum(torch.tensor(o.wh)*e**2);jt.backward()
        np.testing.assert_allclose(float(jt.detach()),j,atol=1e-12);np.testing.assert_allclose(t.grad,g,atol=1e-12)
    def test_eval_isolation(self):
        d=Inputs();q=d.labels('monthly');train=d.split(q,2021);x=d.raw_rows(train,True);tr=Transform().fit(x);sc,f=weights_and_scales(train,'tn_mg_l')
        altered=q.copy();altered.loc[altered.year>=2022,'tn_mg_l']*=500
        train2=d.split(altered,2021);np.testing.assert_array_equal(train.tn_mg_l,train2.tn_mg_l);np.testing.assert_array_equal(x,d.raw_rows(train2,True));t2=Transform().fit(d.raw_rows(train2,True));np.testing.assert_array_equal(tr.mean,t2.mean)
        self.assertEqual(sc,weights_and_scales(train2,'tn_mg_l')[0]);self.assertFalse(any(n in d.identity['features'] for n in ['TN','source','crop','h_day','h_month']))
    def test_future_sequence(self):
        d=Inputs();q=d.labels('daily').iloc[:1];x=d.sequence(q,90);self.assertEqual(x.shape[1],90)
        ti=int(q.ti.iloc[0]);self.assertTrue(np.isclose(x[0,-1,0],d.x[ti,int(q.ri.iloc[0]),0]))
        class FutureAltered:
            def __init__(self,array,cut):self.array=array;self.shape=array.shape;self.cut=cut
            def __getitem__(self,key):
                first=key[0] if isinstance(key,tuple) else key
                ix=np.arange(self.shape[0])[first] if isinstance(first,slice) else np.asarray(first)
                v=np.asarray(self.array[key]).copy();return v+np.expand_dims(ix>self.cut,-1)*10000 if v.ndim>np.ndim(ix) else v+(ix>self.cut)*10000
        old=d.x;oldw=d.water;before_graph=d.sequence(q,90,True);before_row=d.raw_rows(q)
        d.x=FutureAltered(old,ti);d.water=FutureAltered(oldw,ti)
        np.testing.assert_array_equal(x,d.sequence(q,90));np.testing.assert_array_equal(before_graph,d.sequence(q,90,True));np.testing.assert_array_equal(before_row,d.raw_rows(q))
        d.x=old;d.water=oldw
    def test_duplicate_reading_arithmetic(self):
        from independent_readings import daily_mean
        r=pd.read_parquet(ROOT/'data/hf_readings.parquet').iloc[:100];a=daily_mean(r);b=daily_mean(pd.concat([r,r.iloc[:10]],ignore_index=True));pd.testing.assert_frame_equal(a,b)
    def test_cached_sequence_exact(self):
        d=Inputs();q=d.labels('monthly').iloc[[0,500,1500,3000]]
        for graph in [False,True]:
            tr=Transform().fit(d.sequence(q,365,graph));slow=tr.apply(d.sequence(q,365,graph));fast=d.cached_sequence(q,365,graph,tr)
            np.testing.assert_array_equal(fast,slow)
    def test_graph_direction(self):
        d=Inputs();top=read(ROOT/'data/topology.json');self.assertEqual(d.adj.shape,(230,230));self.assertTrue((d.adj>=0).all());np.testing.assert_array_equal(np.diag(d.adj),0)
        reg=d.reg;self.assertFalse('station_key' in d.identity['features']);self.assertTrue(reg.station_key.is_unique)
        q=d.labels('daily').iloc[:3];window=30;fast=d.sequence(q,window,True);base_features=d.x.shape[-1]
        for j,row in enumerate(q.itertuples()):
            nodes=np.flatnonzero(d.adj[row.ri]);ix=np.arange(row.ti-window+1,row.ti+1)
            slow=np.zeros((window,base_features)) if not len(nodes) else np.mean(d.x[ix][:,nodes],axis=1)
            np.testing.assert_array_equal(fast[j,:,base_features:2*base_features],slow)
    def test_history_origin_and_future_labels(self):
        from mltn.history import HistoryInputs
        d=HistoryInputs('daily',7);q=d.labels('daily');q=q[q.year.eq(2023)].iloc[:10];before=d.feedback(q)
        for s,g in d.history.items():
            g.loc[g.available>=pd.Timestamp('2023-01-01'),'tn_mg_l']*=100
        # Choose strictly pre-2023 origins so changed readings are all unavailable.
        oldq=q.copy();oldq['date']=pd.Timestamp('2023-01-01');clean=HistoryInputs('daily',7)
        np.testing.assert_array_equal(clean.feedback(oldq),d.feedback(oldq))
        d=HistoryInputs('daily',1);s=q.station_key.iloc[0];d.history[s]=pd.DataFrame({'available':[pd.Timestamp('2023-01-01'),pd.Timestamp('2023-01-02')],'observed_at':[pd.Timestamp('2023-01-01'),pd.Timestamp('2023-01-02')],'tn_mg_l':[2.,100.]});z=q.iloc[:1].copy();z['date']=pd.Timestamp('2023-01-03');self.assertEqual(float(d.feedback(z)[0,0]),2.)
        # A delayed old reading cannot replace a newer measurement merely by arriving later.
        d.history[s]=pd.DataFrame({'available':pd.to_datetime(['2023-01-01','2023-01-02']),'observed_at':pd.to_datetime(['2022-12-31','2022-12-25']),'tn_mg_l':[2.,100.]});z['date']=pd.Timestamp('2023-01-05');self.assertEqual(float(d.feedback(z)[0,0]),2.)
        frozen=HistoryInputs('daily',1,freeze_year=2022);self.assertTrue(all(not (g.available>pd.Timestamp('2023-01-01')).any() for g in frozen.history.values()))
        from mltn.history import origins
        monthly=pd.DataFrame({'date':pd.to_datetime(['2023-01-01','2023-02-01','2023-03-01'])});self.assertEqual((origins(monthly,'monthly',1)>=pd.Timestamp('2023-01-02')).tolist(),[False,False,True])
    def test_unknown_upstream_support(self):
        from prepare import upstream_mean
        v=np.array([[np.nan,2,3]]);a=np.ones(3);up=np.array([[1,0,0],[1,1,0],[0,0,1]])
        z=upstream_mean(v,a,up);self.assertTrue(np.isnan(z[0,0]));self.assertTrue(np.isnan(z[0,1]));self.assertEqual(z[0,2],3.)
    def test_zero_variance_and_read_dedup(self):
        from mltn.metrics import basic
        self.assertTrue(np.isnan(basic([1,1],[1,1])['NSE']))
        q=pd.read_parquet(ROOT/'data/hf_daily_accepted.parquet');self.assertFalse(q.duplicated(['station_key','date']).any())
        r=pd.read_parquet(ROOT/'data/hf_readings.parquet');cols=['station_key','monitoring_time']
        if all(c in r for c in cols):self.assertFalse(r.duplicated(cols).any())
        # Do not load the same official station-month twice into the level term.
        m=Inputs().labels('monthly');self.assertFalse(m.duplicated(['station_key','year','month']).any())
    def test_cpu_gpu_fixed_and_recovery(self):
        torch.set_num_threads(1);torch.manual_seed(1729);cfg=dict(width=32,dropout=0.)
        for family in ['MLP','LSTM','GRU','TCN','Transformer','GraphTCN']:
            model=Network(family,12,cfg).eval();x=torch.rand(4,90,12);cpu=model(x).detach();self.assertTrue(torch.isfinite(cpu).all());self.assertTrue((cpu>=0).all())
            restored=Network(family,12,cfg);restored.load_state_dict(model.state_dict());restored.eval();torch.testing.assert_close(cpu,restored(x),rtol=0,atol=0)
            if torch.cuda.is_available():
                gpu=copy.deepcopy(model).cuda();pg=gpu(x.cuda()).cpu().detach();torch.testing.assert_close(cpu,pg,rtol=3e-5,atol=3e-6)
                # Metrics and objectives are independently recomputed in FP64.
                self.assertLess(float(torch.max(torch.abs(cpu.double()**2-pg.double()**2))),1e-4)
    def test_causal_neural(self):
        torch.manual_seed(1729);cfg=dict(width=32,dropout=0.);x=torch.rand(2,100,12)
        for family in ['TCN','GraphTCN']:
            model=Network(family,12,cfg).eval();z=torch.relu(model.proj(x));a=model.core(z);z2=z.clone();z2[:,80:]*=100;b=model.core(z2);torch.testing.assert_close(a[:,:80],b[:,:80],rtol=0,atol=0)
    def test_training_resume_rng_and_optimizer(self):
        from mltn.checkpoint import save_epoch,restore_epoch
        import tempfile
        torch.manual_seed(1729);model=Network('MLP',12,dict(width=16,dropout=.2));opt=torch.optim.AdamW(model.parameters(),lr=.01)
        x=torch.randn(5,4,12);y=torch.randn(5)
        def step(m,o):
            m.train();o.zero_grad();loss=((m(x)-y)**2).mean();loss.backward();o.step();return float(loss.detach())
        step(model,opt)
        with tempfile.TemporaryDirectory(dir=ROOT/'evidence') as temp:
            path=__import__('pathlib').Path(temp)/'resume.pt';identity=dict(job='fixture',data='known')
            save_epoch(path,model,opt,identity,1,[],1.,1,0);expected=step(model,opt)
            restored=Network('MLP',12,dict(width=16,dropout=.2));other=torch.optim.AdamW(restored.parameters(),lr=.01)
            state=restore_epoch(path,restored,other,identity,'cpu');actual=step(restored,other)
            self.assertEqual(state['epoch'],1);self.assertEqual(actual,expected)
            for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            with self.assertRaisesRegex(RuntimeError,'RESUME_IDENTITY'):restore_epoch(path,restored,other,dict(job='other'),'cpu')
    def test_actual_fitter_evaluation_labels_cannot_train(self):
        import tempfile,types,shutil
        from pathlib import Path
        from unittest.mock import patch
        import train as trainer
        dates=pd.date_range('2016-01-01','2023-12-01',freq='MS');rows=[]
        for site in range(3):
            for date in dates:rows.append(dict(station_key=str(site),date=date,year=date.year,month=date.month,tn_mg_l=1+site*.3+date.month*.04,start_ti=0,ti=1))
        labels=pd.DataFrame(rows)
        cfg=dict(width=16,window=90,dropout=.2,lr=.01,weight_decay=.001,epochs=2,batch_size=32,patience=18,trees=4,depth=3,min_leaf=1,max_features=1.,tree_lr=.1)
        def fit(family,altered):
            with tempfile.TemporaryDirectory(dir=ROOT/'evidence') as temp:
                tmp=Path(temp);(tmp/'data').mkdir();(tmp/'config').mkdir();shutil.copyfile(ROOT/'train.py',tmp/'train.py');np.save(tmp/'data/reach_features.npy',np.ones((2,3,3)))
                q=labels.copy()
                if altered:q.loc[q.year.eq(2023),'tn_mg_l']*=500
                q.to_parquet(tmp/'data/monthly_tn_accepted.parquet');write(tmp/'config/design.json',dict(configs={family:[cfg]}))
                class Fixture:
                    identity={'array_file':'reach_features.npy'}
                    def labels(self,task):return q.copy()
                    def split(self,frame,cutoff,block=None):return frame[frame.year<=cutoff].reset_index(drop=True)
                    def raw_rows(self,frame,monthly=False):return np.column_stack([frame.station_key.astype(int),frame.month,frame.year-2016]).astype(float)
                args=types.SimpleNamespace(family=family,stage='F23',task='monthly',config=0,seed=1729,block=None,device='cpu',threads=1,fixed_recipe=False)
                with patch.object(trainer,'ROOT',tmp),patch.object(trainer,'Inputs',Fixture),patch('mltn.resources.lease',return_value=None):trainer.run(args)
                folder=tmp/'jobs'/f'F23_monthly_{family}_c0_s1729';pred=pd.read_parquet(folder/'prediction.parquet').prediction.to_numpy();ck=pickle.load((folder/'checkpoint.pkl').open('rb'));weights=torch.load(folder/'weights.pt',weights_only=True) if family=='MLP' else None
                return pred,ck['transform'].mean,weights
        for family in ['RF','MLP']:
            first=fit(family,False);second=fit(family,True);np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])
            if first[2] is not None:
                for key in first[2]:torch.testing.assert_close(first[2][key],second[2][key],rtol=0,atol=0)
    def test_resume_after_patience_does_not_train_extra_epoch(self):
        import tempfile,types,shutil
        from pathlib import Path
        from unittest.mock import patch
        import train as trainer
        dates=pd.date_range('2016-01-01','2022-12-01',freq='MS');q=pd.DataFrame(dict(station_key=['a']*len(dates),date=dates,year=dates.year,month=dates.month,tn_mg_l=1+dates.month*.04,start_ti=0,ti=1))
        cfg=dict(width=16,window=90,dropout=.2,lr=.01,weight_decay=.001,epochs=8,batch_size=32,patience=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'evidence') as temp:
            tmp=Path(temp);(tmp/'data').mkdir();(tmp/'config').mkdir();shutil.copyfile(ROOT/'train.py',tmp/'train.py');np.save(tmp/'data/reach_features.npy',np.ones((2,1,3)));q.to_parquet(tmp/'data/monthly_tn_accepted.parquet');write(tmp/'config/design.json',dict(configs={'MLP':[cfg]}))
            class Fixture:
                identity={'array_file':'reach_features.npy'}
                def labels(self,task):return q.copy()
                def split(self,frame,cutoff,block=None):return frame[frame.year<=cutoff].reset_index(drop=True)
                def raw_rows(self,frame,monthly=False):return np.column_stack([frame.month,frame.year-2016,np.ones(len(frame))]).astype(float)
            args=types.SimpleNamespace(family='MLP',stage='screen',task='monthly',config=0,seed=1729,block=None,device='cpu',threads=1,fixed_recipe=False)
            original=trainer.atomic_save
            def fail_before_final_weights(value,path):
                if Path(path).name=='weights.pt':raise RuntimeError('SIMULATED_POST_STOP_INTERRUPTION')
                original(value,path)
            with patch.object(trainer,'ROOT',tmp),patch.object(trainer,'Inputs',Fixture),patch('mltn.resources.lease',return_value=None),patch.object(trainer,'predict_network',side_effect=lambda model,d,rows,*a:np.zeros(len(rows))):
                with patch.object(trainer,'atomic_save',side_effect=fail_before_final_weights):
                    with self.assertRaisesRegex(RuntimeError,'SIMULATED_POST_STOP'):trainer.run(args)
                folder=tmp/'jobs/screen_monthly_MLP_c0_s1729';self.assertEqual(len(read(folder/'curve.json')),2);best=torch.load(folder/'best.pt',weights_only=True)
                (folder/'owner.lock').unlink();result=trainer.run(args)
                self.assertEqual(result['epochs_executed'],2);self.assertEqual(result['stop_reason'],'internal_validation_patience')
                final=torch.load(folder/'weights.pt',weights_only=True)
                for key in best:torch.testing.assert_close(best[key],final[key],rtol=0,atol=0)
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Gates);result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(ROOT/'evidence'/f'acceptance_{platform.system()}.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),cuda=torch.cuda.is_available(),torch=torch.__version__,host=platform.node(),dtype_train='FP32',dtype_evaluation='FP64',amp=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
