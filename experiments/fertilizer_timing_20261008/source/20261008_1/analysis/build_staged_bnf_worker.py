"""Derive a passive staged BNF entry without changing the audited recurrence."""
import ast,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis')]
SOURCE=ROOT/'analysis/trace_approved_bnf_history.py'
TARGET=ROOT/'analysis/trace_staged_bnf_history.py'


def derive(source):
    old="""    if mode == 'preflight' and jid == 'TIM_F24_C0_V0_s0':
        folder = ROOT/'outputs/jobs'/jid
    elif mode == 'final':
        from approved_epoch_delivery import admitted_folders
        folder = admitted_folders(ROOT)[jid]
    else:
        raise RuntimeError('DECLARED_BNF_TRACE_MODE_REQUIRED')
"""
    new="""    if mode != 'staged':
        raise RuntimeError('STAGED_BNF_ONLY_ENTRY_REQUIRED')
    from staged_frozen_sources import resolve
    folder = resolve(jid)
"""
    if source.count(old)!=1:raise RuntimeError('EXACT_AUDITED_BNF_ENTRY_REQUIRED')
    result=source.replace(old,new);compile(result,str(TARGET),'exec')
    return result


def numerical_body(source):
    tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.If) and 'mode' in ast.unparse(n.test))
    end=next(i for i in range(start+1,len(main.body)) if isinstance(main.body[i],ast.ImportFrom) and main.body[i].module=='diagnose_approved_waveforms')
    main.body[start:end]=[]
    return ast.dump(tree,include_attributes=False)


def main():
    from d29_platform.runtime import sha,write_json
    from run_approved_bnf_delivery import require_controls
    controls=json.loads((ROOT/'evidence/approved_bnf_delivery_controls.json').read_text())
    if not controls.get('passed') or controls.get('actual_cases')!=1 or controls.get('files',{}).get('analysis/trace_approved_bnf_history.py')!=sha(SOURCE):
        raise RuntimeError('ACTUAL_AUDITED_BNF_BASELINE_REQUIRED')
    partition=require_controls()
    source=SOURCE.read_text(encoding='utf8');result=derive(source)
    assert numerical_body(source)==numerical_body(result)
    for bad in (source.replace("mode == 'preflight'","mode == 'changed'",1),source+source):
        try:derive(bad)
        except RuntimeError:pass
        else:raise AssertionError('changed BNF body admitted')
    if TARGET.exists():raise RuntimeError('STAGED_BNF_WORKER_ALREADY_BUILT')
    TARGET.write_text(result,encoding='utf8')
    write_json(ROOT/'evidence/staged_bnf_derivation.json',dict(passed=True,controls=3,
        upstream_sha256=sha(SOURCE),staged_worker_sha256=sha(TARGET),numerical_body_AST_identical=True,
        partition_controls_sha256=partition,evaluation_admitted=False,training_ownership_changed=False))
    print(json.dumps(dict(passed=True,controls=3,numerical_body_AST_identical=True)))


if __name__=='__main__':main()
