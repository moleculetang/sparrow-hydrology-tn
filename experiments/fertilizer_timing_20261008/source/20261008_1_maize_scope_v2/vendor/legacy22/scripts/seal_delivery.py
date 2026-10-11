"""Reseal immutable delivery after report-only edits; never reruns fitting."""
import time
import native_runtime as rt
R=rt.RUN
if __name__=='__main__':
    files={str(p.relative_to(R)):rt.sha(p) for folder in ['scripts','configs','reports','outputs','evidence','data'] for p in (R/folder).rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
    files['README.md']=rt.sha(R/'README.md')
    rt.write(R/'data/delivery_manifest.json',dict(files=files,created=time.time(),mutable_logs='work/*.jsonl retained outside immutable seal'))
    reread=rt.read(R/'data/delivery_manifest.json')
    for name,h in reread['files'].items():assert rt.sha(R/name)==h,name
    print('Verified sealed files:',len(files))
