"""Source integrity plus a data-free check of the original conservative TN kernel."""
from pathlib import Path
import ast,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
assert os.environ.get('CONDA_DEFAULT_ENV','').lower()=='sparrow','Run with conda run -n sparrow'
assert Path(os.environ.get('CONDA_PREFIX','')).resolve()==Path(sys.prefix).resolve()
manifest=json.loads((ROOT/'publication/source_manifest.json').read_text(encoding='utf-8'))
for item in manifest['files']:
    p=ROOT/item['path'];raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==item['sha256'],item['path']
    if p.suffix=='.py':ast.parse(raw.decode('utf-8-sig'),filename=item['path'])
# Extract the original kernel AST without importing data loaders or bypassing their
# runtime guards. All test inputs below are synthetic; no observed data are read.
path=ROOT/'5_Test/20260905_2/scripts/tn_reference.py'
tree=ast.parse(path.read_text(encoding='utf-8-sig'))
kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='local_daily_kernel')
kernel.decorator_list=[]
import numpy as np
scope={'np':np};exec(compile(ast.Module(body=[kernel],type_ignores=[]),str(path),'exec'),scope)
for uniform in [False,True]:
    probability=np.full((6,2),.2);survival=np.array([.98,.995]);fast=np.full((6,2),.4);release=np.full((6,2),.1)
    source=np.array([[100.,50.],[20.,0.]]);crop=np.array([[10.,5.],[2.,1.]])
    result=scope['local_daily_kernel'](probability,survival,fast,release,source,crop,np.array([0,0,0,1,1,1]),np.array([3,3]),uniform)
    qf,qs,other,uptake,mineral,lower=result
    np.testing.assert_allclose(source.sum(axis=0),qf.sum(axis=0)+qs.sum(axis=0)+other.sum(axis=0)+uptake.sum(axis=0)+mineral[-1]+lower[-1],rtol=1e-12,atol=1e-12)
    assert all(np.isfinite(a).all() and (a>=0).all() for a in result)
print('SOURCE_CHECK_PASS',len(manifest['files']),'manifest files; synthetic N balance: pulse and uniform')
