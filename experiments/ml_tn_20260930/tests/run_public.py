"""Run real source with synthetic fixtures; block private experiment reads."""
import os,sys,io,json,unittest,datetime,platform,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
def barrier(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        s=os.fsdecode(args[0]).replace('\\','/').lower()
        if s.startswith('e:/sparrow/5_test/') or '/home/pc/twy/sparrow/test/20260930_1/' in s:
            raise PermissionError('PRIVATE_EXPERIMENT_READ_BLOCKED')
sys.addaudithook(barrier)
import numpy as np,pandas as pd,torch
torch.set_num_threads(1)
import acceptance
from mltn.metrics import basic,centered,paired_summary
class PublicExtra(unittest.TestCase):
    def test_zero_variance(self):
        self.assertTrue(np.isnan(basic([1.,1.],[2.,3.])['NSE']))
        self.assertTrue(np.isnan(basic([1.,2.],[3.,3.])['correlation']))
    def test_nse_decomposition_and_amplitude(self):
        from diagnostics import decomposition
        y=np.array([1.,2.,4.,2.,1.]);p=np.array([2.,2.2,2.6,2.2,2.])
        d=decomposition(y,p);self.assertAlmostEqual(d['mse'],sum(d[k] for k in ['bias_squared','amplitude_mismatch_squared','decorrelation_error']))
        m=basic(y,p);self.assertAlmostEqual(m['NSE'],2*m['correlation']*m['amplitude_ratio']-m['amplitude_ratio']**2-(m['bias']/y.std())**2)
    def test_month_centered_weighting(self):
        q=pd.DataFrame(dict(month_key=['a']*3+['b']*2,read_count=[1,2,3,4,1],observed=[1.,2.,4.,3.,6.],prediction=[5.,6.,8.,10.,13.]))
        self.assertAlmostEqual(centered(q,'prediction')['NSE'],1.)
    def test_paired_medians_differ(self):
        q=pd.DataFrame(dict(station_key=['a','b','c']*2,configuration=['baseline']*3+['candidate']*3,NSE=[0.,1.,10.,2.,0.,3.]))
        v=paired_summary(q);self.assertEqual(v['common_stations'],3);self.assertNotEqual(v['median_difference'],v['median_paired_difference'])
    def test_tree_seed_repeat_and_diversity(self):
        from mltn.models import tree
        cfg=dict(trees=12,depth=3,min_leaf=1,max_features=.8,tree_lr=.1)
        x=np.random.default_rng(1).normal(size=(100,4));y=x[:,0]+x[:,1]**2
        def fit(s):
            m=tree('XGBoost',cfg,s,1);m.fit(x,y);return m.predict(x)
        np.testing.assert_array_equal(fit(1729),fit(1729));self.assertFalse(np.array_equal(fit(1729),fit(1730)))
    def test_joint_seed_is_forwarded(self):
        import ast
        source=ast.parse((ROOT/'joint.py').read_text(encoding='utf-8-sig'))
        named=[n for n in ast.walk(source) if isinstance(n,ast.keyword) and n.arg=='seed']
        self.assertTrue(named)
    def test_private_read_barrier(self):
        with self.assertRaisesRegex(PermissionError,'PRIVATE_EXPERIMENT_READ_BLOCKED'):open('E:/SPARROW/5_Test/20260930_1/data/feature_identity.json')
names=['test_independent_group_gradient','test_unknown_upstream_support','test_cpu_gpu_fixed_and_recovery','test_causal_neural','test_training_resume_rng_and_optimizer','test_actual_fitter_evaluation_labels_cannot_train','test_resume_after_patience_does_not_train_extra_epoch']
suite=unittest.TestSuite([acceptance.Gates(n) for n in names]);suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(PublicExtra))
result=unittest.TextTestRunner(verbosity=2).run(suite)
versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch','xgboost','lightgbm','catboost','pyarrow']}
receipt=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),private_read_barrier=True,scope='synthetic public package; not real TN reproduction',python=sys.version,platform=platform.system(),packages=versions,cuda_available=torch.cuda.is_available(),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(ROOT/'tests/latest_public_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
raise SystemExit(0 if result.wasSuccessful() else 1)
