"""Read-only private archive replay using published algorithms. No optimization."""
import argparse,ast,json,sys,os,hashlib,tempfile
from pathlib import Path
P=Path(__file__).resolve().parents[1];S=P/'snapshot/20260920_6'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--tag',choices=[f'{fold}_{kind}_s{i}' for fold in ['F23','F24'] for kind in ['R','X'] for i in [0,1]],required=True)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();root=args.data_root.resolve();run=root/'5_Test/20260920_6'
    fold='F23_G_D' if args.tag.startswith('F23') else 'T24_G_D_H1'
    # Install label and read-only archive barriers before checking any data identities.
    def barrier(event,a):
        if event!='open' or not isinstance(a[0],(str,bytes,os.PathLike)):return
        q=Path(os.fsdecode(a[0])).resolve();s=q.as_posix().lower();mode=a[1];flags=a[2]
        if q.is_relative_to(root/'5_Test'):
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if write:raise PermissionError('READ_ONLY_ARCHIVE '+str(q))
            if 'heldout_labels' in s or '/data/cohorts/' in s or ('train.parquet' in s and q!=run/'data/folds'/fold/'train.parquet'):raise PermissionError('FOREIGN_LABEL '+str(q))
    sys.addaudithook(barrier)
    required=[run/'configs/jobs.json',run/'configs/folds.json',run/'data/designs'/f'{fold}.json',run/'data/folds'/fold/'train.parquet',run/'outputs'/args.tag/'model.json',run/'data/spatial_support.json']
    missing=[str(q) for q in required if not q.is_file()]
    if missing:print(json.dumps({'status':'MISSING_PRIVATE_DATA','missing':missing},indent=2));return 2
    read=lambda q:json.loads(q.read_text(encoding='utf-8'))
    domain=read(run/'configs/folds.json')[fold]['domain'];layout=run/'data/domains'/domain/'arrays.json'
    if not layout.exists():print('MISSING '+str(layout));return 2
    arrays=read(layout);hydro=root/'5_Test/20260917_5/data/domains/FULL24C'
    required.extend([layout,run/'data/domains'/domain/'topology.json'])
    design=read(run/'data/designs'/f'{fold}.json');required.append(run/design['observation_registry_file'])
    required.extend(hydro/v['file'] for v in arrays.values())
    missing=[str(q) for q in required if not q.is_file()]
    if missing:print(json.dumps({'status':'MISSING_PRIVATE_DATA','missing':missing},indent=2));return 2
    for v in arrays.values():assert digest(hydro/v['file'])==v['sha256'],'ARRAY_IDENTITY_CHANGED'
    # Check fold training, topology and constants against the sealed original launch identity.
    launch=run/'reports/launch_by_fold'/f'{fold}.json'
    if not launch.exists():print('MISSING '+str(launch));return 2
    launch_data=read(launch)
    public_launch=P/'evidence/20260920_6/launch_by_fold'/f'{fold}.json'
    assert digest(launch)==digest(public_launch),'LAUNCH_DIFFERS_FROM_PUBLISHED_IDENTITY'
    archived_model=P/'evidence/20260920_6/parameters'/args.tag/'model.json'
    assert digest(run/'outputs'/args.tag/'model.json')==digest(archived_model),'MODEL_DIFFERS_FROM_PUBLISHED_POINT'
    for rel,h in launch_data.get('frozen_hashes',{}).items():
        rel=rel.replace('\\','/')
        if rel.startswith(('data/','configs/')):
            q=run/rel
            if not q.is_file():print('MISSING '+str(q));return 2
            assert digest(q)==h,'FROZEN_IDENTITY_CHANGED '+rel
    print(json.dumps({'status':'INPUTS_PRESENT_AND_VERIFIED','tag':args.tag,'arrays':len(arrays),'files':len(required)}))
    if args.check_only:return 0
    if args.output_dir is None:parser.error('--output-dir required unless --check-only')
    out=args.output_dir.resolve()
    if out.is_relative_to(root/'5_Test') or out.is_relative_to(P):raise ValueError('OUTPUT_MUST_BE_OUTSIDE_ARCHIVE_AND_PUBLICATION')
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as cache:
        os.environ['NUMBA_CACHE_DIR']=cache;sys.path.insert(0,str(S/'scripts'))
        import campaign_model as cm
        import hf_model,temporal_model,state_modulated
        import numba.core.config as nc
        os.environ['NUMBA_CACHE_DIR']=cache;nc.CACHE_DIR=cache
        for m in [cm,hf_model,temporal_model]:m.RUN=run
        # Only rebase the one hardcoded H1 path in the original loader AST.
        tree=ast.parse((S/'scripts/campaign_model.py').read_text(encoding='utf-8'))
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='load_data')
        class Rebase(ast.NodeTransformer):
            count=0
            def visit_Constant(self,n):
                if n.value=='E:/SPARROW/5_Test/20260917_5/data/domains/FULL24C':
                    self.count+=1;return ast.copy_location(ast.Constant(str(hydro)),n)
                return n
        tr=Rebase();fn=tr.visit(fn);assert tr.count==1
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(S/'scripts/campaign_model.py'),'exec'),cm.__dict__)
        job=next(j for j in read(run/'configs/jobs.json') if j['tag']==args.tag)
        model=cm.for_job(job);record=read(run/'outputs'/args.tag/'model.json')
        x=cm.np.array(record['parameters']);value,g=model.value_gradient(x);a=model.ledger(x)
        from serial_solvers import projected_gradient
        pg=float(cm.np.max(abs(projected_gradient(x,g,model.bounds))))
        scale=float((a['fast']+a['slow']).sum())
        physical=a['local_balance_max_kg']<=1e-6 and abs(a['network_balance_kg'])<=scale*1e-10 and max(a['source_label_sum_errors'].values(),default=0)<=1e-6 and min(a[k].min() for k in ['M','L','fast','slow','uptake','mineral_loss'])>=-1e-7 and cm.np.max(a['uptake']-a['demand'])<=1e-7
        match=abs(value-record['objective'])<=1e-8*(1+abs(value))
        result=dict(status='PASS' if physical and match else 'FAIL',tag=args.tag,objective=value,objective_difference=value-record['objective'],pg=pg,numerically_sufficient=pg<=1e-5,physical=bool(physical),terms=model.last_terms,adapter='one path constant and RUN only; full source functions otherwise unchanged')
        (out/(args.tag+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result));assert physical and match
    return 0
if __name__=='__main__':raise SystemExit(main())
