"""Candidate provenance calculations on explicitly represented organic stocks."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'d29_platform/tagged_candidate.py').read_text(encoding='utf-8')
s=s.replace('from .land1 import','from .precision_candidate import',1)
s=s.replace('@njit(cache=True)','from .precision_transfer import transfer_expansion\n\n@njit(cache=True)',1)
s=s.replace('total_states, total_fluxes, record, compensated, initial_compensation):','total_states, total_fluxes, record, compensated, initial_compensation, physical_correction):',1)
start=s.index('        if event[t] >= 0:',s.index('def _tag_forward'))
end=s.index('        following =',start)
s=s[:start]+'''        if event[t] >= 0:
            old_low=np.zeros_like(current)
            old_low[:,:,1:3]=-organic_correction
            moved,low,_=transfer_expansion(current.reshape(nr,nl,nk*5),old_low.reshape(nr,nl,nk*5),matrices[event[t]])
            moved=moved.reshape(nu,nk,5);low=low.reshape(nu,nk,5)
            moved_correction=-low[:,:,1:3].copy()
            for r in range(nr):
                sl=slice(r*nl,(r+1)*nl)
                for k in range(nk):
                    err=_accurate_sum(np.concatenate((moved[sl,k].ravel(),-current[sl,k].ravel(),-moved_correction[sl,k].ravel(),organic_correction[sl,k].ravel())))
                    worst=max(worst,abs(err))
            current=moved;organic_correction=moved_correction
''' + s[end:]
start=s.index('                    physical=np.zeros(5')
end=s.index('                else:P,SA,SP,N,L=',start)
s=s[:start]+'''                    all_low=np.zeros_like(total_states[t])
                    all_low[:,1:3]=-physical_correction[t]
                    # Only the current reach is needed for physical replay.
                    r=u//nl;land=u%nl
                    moved,_,_=transfer_expansion(total_states[t,r*nl:(r+1)*nl].reshape(1,nl,5),all_low[r*nl:(r+1)*nl].reshape(1,nl,5),matrices[event[t],r:r+1])
                    P,SA,SP,N,L=moved[0,land]
''' + s[end:]
s=s.replace('                p, a, b, n, lower = current[u, k]','                old_correction=organic_correction[u,k].copy()\n                p, a, b, n, lower = current[u, k]',1)
s=s.replace('err = (following[u, k] - current[u, k]).sum() - source[t, u, k].sum() + values.sum()',
'''err = _accurate_sum(np.concatenate((following[u,k],-current[u,k],-source[t,u,k],values,old_correction,-organic_correction[u,k])))''',1)
s=s.replace('state_error = max(state_error, abs(state_sum[j] - total_states[t + 1, u, j]))',
'''terms=np.concatenate((following[u,:,j],np.array([-total_states[t+1,u,j]])))
                    if j==1 or j==2:
                        terms=np.concatenate((terms,-organic_correction[u,:,j-1],np.array([physical_correction[t+1,u,j-1]])))
                    state_error=max(state_error,abs(_accurate_sum(terms)))''',1)
s=s.replace('    restart_difference = np.abs(initial.sum(axis=-2).reshape(nr * nl, 5)-i["initial"])',
'''    import math
    initial_corr=np.zeros((nr,nl,nk,2)) if initial_compensation is None else np.asarray(initial_compensation).reshape(nr,nl,nk,2)
    physical_corr=i['compensation_history'][0].reshape(nr,nl,2)
    restart_difference=np.zeros((nr,nl,5))
    for r in range(nr):
        for l in range(nl):
            for state in range(5):
                terms=[*initial[r,l,:,state],-i['initial'].reshape(nr,nl,5)[r,l,state]]
                if state in (1,2):terms.extend([*-initial_corr[r,l,:,state-1],physical_corr[r,l,state-1]])
                restart_difference[r,l,state]=abs(math.fsum(terms))
    restart_difference=restart_difference.reshape(nr*nl,5)''',1)
s=s.replace('record_history, compensated, corr)','record_history, compensated, corr, i[\'compensation_history\'])',1)
s=s.replace('    final_err = float(np.max(np.abs(final.sum(axis=-2) - result.final)))',
'''    final_err=0.
    for r in range(nr):
        for l in range(nl):
            for state in range(5):
                terms=[*final[r,l,:,state],-result.final[r,l,state]]
                if state in (1,2):terms.extend([*-ans[7][r*nl+l,:,state-1],result.compensation[r,l,state-1]])
                final_err=max(final_err,abs(math.fsum(terms)))''',1)
s=s.replace('[:,:,1:3]','[:,:,1:5]').replace('[:,1:3]','[:,1:5]')
s=s.replace('(nr,nl,nk,2)','(nr,nl,nk,4)').replace('(nr,nl,2)','(nr,nl,4)').replace('(nr*nl,nk,2)','(nr*nl,nk,4)')
s=s.replace('if state in (1,2):','if state in (1,2,3,4):').replace('if j==1 or j==2:','if j>=1:')
needle='                values = np.array([fast, slow, loss, exp])'
s=s.replace(needle,'''                if compensated:
                    for j,stock,delta,empty in ((2,n,_accurate_sum(np.array([source[t,u,k,3],ka,kb,-take,-e,-loss])),av==0. or pl[t,u]==1.),(3,lower,_accurate_sum(np.array([(1-ff[t,u])*e,-slow])),lr[t,u]==1.)):
                        if empty:
                            following[u,k,j+1]=0.;organic_correction[u,k,j]=0.
                        else:
                            corrected=delta-organic_correction[u,k,j]
                            updated=stock+corrected
                            organic_correction[u,k,j]=(updated-stock)-corrected
                            following[u,k,j+1]=updated
''' + needle,1)
(ROOT/'d29_platform/precision_tags_candidate.py').write_text(s,encoding='utf-8')
