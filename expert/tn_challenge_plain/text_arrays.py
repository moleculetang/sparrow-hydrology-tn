"""Lossless float64 CSV shards; no ZIP, NPZ, pickle or binary download."""
import hashlib,json,re
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_arrays(directory,arrays):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    layout={'format':'uncompressed CSV, comma-separated ASCII, one header row','float_format':'%.17g','order':'C','arrays':{}}
    for name,value in arrays.items():
        if not re.fullmatch(r'[A-Za-z0-9_]+',name):raise ValueError('Invalid array name')
        a=np.asarray(value)
        if a.dtype.kind not in 'bifu' or not np.isfinite(a).all():raise ValueError('Only finite numeric arrays supported')
        rows=a.shape[0] if a.ndim else 1;matrix=a.reshape(rows,-1);cols=matrix.shape[1]
        # Float64 scientific notation fits within 25 characters plus delimiter.
        block=max(1,450000//(cols*26));chunks=[]
        for j,start in enumerate(range(0,rows,block)):
            stop=min(rows,start+block);p=directory/f'{name}_{j:03d}.csv'
            np.savetxt(p,matrix[start:stop],delimiter=',',fmt='%.17g' if a.dtype.kind=='f' else '%d',
                       header=','.join(f'c{i}' for i in range(cols)),comments='',encoding='ascii')
            if p.stat().st_size>=500000:raise ValueError('CSV shard exceeds 500 KB')
            chunks.append({'file':p.name,'start':start,'stop':stop,'bytes':p.stat().st_size,'sha256':sha(p)})
        layout['arrays'][name]={'shape':list(a.shape),'dtype':a.dtype.str,'columns':cols,'chunks':chunks}
    (directory/'array_layout.json').write_text(json.dumps(layout,indent=2),encoding='utf-8')
    return layout

def read_arrays(directory):
    directory=Path(directory);layout=json.loads((directory/'array_layout.json').read_text(encoding='utf-8'));result={}
    for name,spec in layout['arrays'].items():
        dtype=np.dtype(spec['dtype']);parts=[];next_start=0
        for chunk in spec['chunks']:
            p=directory/chunk['file']
            if p.name!=chunk['file'] or p.suffix!='.csv':raise ValueError('Invalid shard path')
            if sha(p)!=chunk['sha256']:raise ValueError(f'Array hash mismatch: {p.name}')
            if chunk['start']!=next_start:raise ValueError('Noncontiguous shard rows')
            x=np.loadtxt(p,delimiter=',',skiprows=1,dtype=np.uint8 if dtype.kind=='b' else dtype,ndmin=2)
            if x.shape!=(chunk['stop']-chunk['start'],spec['columns']):raise ValueError('CSV shape mismatch')
            parts.append(x);next_start=chunk['stop']
        result[name]=np.concatenate(parts,axis=0).astype(dtype,copy=False).reshape(spec['shape']).copy(order='C')
    return result
