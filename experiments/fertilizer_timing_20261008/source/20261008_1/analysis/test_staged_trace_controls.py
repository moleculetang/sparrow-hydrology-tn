"""Reject unsafe imports; verify only passive entry admission was changed."""
import ast,io,json,sys,tarfile,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis'),str(ROOT/'deployment')]
from build_staged_trace_worker import derive,SOURCE
from import_staged_frozen import safe_members


def normalized(source):
    tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    # The audited upstream has one declared-mode conditional, before imports.
    first=next(i for i,n in enumerate(main.body) if isinstance(n,ast.If) and
        isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='mode')
    after=next(i for i in range(first+1,len(main.body)) if isinstance(main.body[i],ast.ImportFrom) and
        main.body[i].module=='diagnose_approved_waveforms')
    main.body[first:after]=[]
    return ast.dump(tree,include_attributes=False)


def main():
    source=SOURCE.read_text(encoding='utf8');derived=derive(source)
    assert normalized(source)==normalized(derived)
    count=1
    for changed in (source.replace("if mode == 'preflight':","if mode == 'changed':",1),source+source):
        try:derive(changed)
        except RuntimeError:count+=1
        else:raise AssertionError('changed upstream admitted')
    with tempfile.TemporaryDirectory() as temp:
        target=Path(temp)/'data'
        for names,bad in [(['manifest.json','outputs/jobs/one/file.npy'],False),
            (['../escape'],True),(['/absolute'],True),(['a','a'],True),(['a\\b'],True),(['C:/escape'],True)]:
            data=io.BytesIO()
            with tarfile.open(fileobj=data,mode='w') as stream:
                for name in names:stream.addfile(tarfile.TarInfo(name),io.BytesIO())
            data.seek(0)
            with tarfile.open(fileobj=data,mode='r') as stream:
                try:safe_members(stream,target)
                except RuntimeError:
                    if not bad:raise
                else:
                    if bad:raise AssertionError('unsafe archive admitted')
            count+=1
        for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE):
            data=io.BytesIO();member=tarfile.TarInfo('link');member.type=kind;member.linkname='escape'
            with tarfile.open(fileobj=data,mode='w') as stream:stream.addfile(member)
            data.seek(0)
            with tarfile.open(fileobj=data,mode='r') as stream:
                try:safe_members(stream,target)
                except RuntimeError:count+=1
                else:raise AssertionError('nonregular archive member admitted')
    from d29_platform.runtime import sha,write_json
    result=dict(passed=True,controls=count,numerical_replay_body_AST_identical=True,
        unsafe_archive_controls=True,evaluation_admitted=False,
        upstream_sha256=sha(SOURCE),deriver_sha256=sha(ROOT/'analysis/build_staged_trace_worker.py'),
        importer_sha256=sha(ROOT/'deployment/import_staged_frozen.py'))
    write_json(ROOT/'evidence/staged_trace_controls.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
