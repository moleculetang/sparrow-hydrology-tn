"""Build a separately auditable candidate; do not change live gradient kernel."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import sha,write_json
source=ROOT/'d29_platform/land1.py'
s=source.read_text(encoding='utf-8')
s=s.replace('STATE_NAMES =', 'from .precision_transfer import transfer_expansion\n\nSTATE_NAMES =',1)
start=s.index('        if event[t] >= 0:',s.index('def _forward('))
end=s.index('        for u in range(nu):',start)
s=s[:start]+'''        if event[t] >= 0:
            old_low=np.zeros_like(before)
            old_low[:,1:3]=-organic_correction
            moved,low,_=transfer_expansion(before.reshape(nr,nl,5),old_low.reshape(nr,nl,5),matrices[event[t]])
            before=moved.reshape(nu,5)
            moved_low=low.reshape(nu,5)
            # Non-plant pools have persistent hi-minus-compensation state. The
            # plant pool retains its ordinary float64 representation; discarded
            # low part remains visible to this strict independent budget check.
            organic_correction=-moved_low[:,1:3].copy()
            represented_low=np.zeros_like(before)
            represented_low[:,1:3]=-organic_correction
            for r in range(nr):
                sl=slice(r*nl,(r+1)*nl)
                err=_accurate_sum(np.concatenate((before[sl].ravel(),represented_low[sl].ravel(),-states[oldi,sl].ravel(),-old_low[sl].ravel())))
                worst=max(worst,abs(err))
''' + s[end:]
s=s.replace('            P, SA, SP, N, L = before[u]\n            ka, kp','            old_correction=organic_correction[u].copy()\n            P, SA, SP, N, L = before[u]\n            ka, kp',1)
s=s.replace('balance = (states[newi, u] - before[u]).sum() - source[t, u].sum() + fast + slow + loss + realized[0]',
'''balance = _accurate_sum(np.concatenate((states[newi,u],-before[u],-source[t,u],flux[t,u],old_correction,-organic_correction[u])))''',1)
s=s.replace('    unmet = np.zeros_like(out)\n','    unmet = np.zeros_like(out)\n    correction_history=np.empty((nt+1 if keep else 0,nu,2),np.float64)\n    if keep:correction_history[0]=organic_correction\n',1)
s=s.replace('return flux, states, worst, t, u, -plant_after, organic_correction, unmet','return flux, states, worst, t, u, -plant_after, organic_correction, unmet, correction_history',1)
s=s.replace('    return flux, states, worst, -1, -1, 0.0, organic_correction, unmet','        if keep:correction_history[t+1]=organic_correction\n    return flux, states, worst, -1, -1, 0.0, organic_correction, unmet, correction_history',1)
s=s.replace('flux, states, err, day, unit, short, correction, unmet = _forward','flux, states, err, day, unit, short, correction, unmet, correction_history = _forward',1)
s=s.replace("    inp['realized_outflows']=inp['outflows']-unmet","    inp['realized_outflows']=inp['outflows']-unmet\n    inp['compensation_history']=correction_history",1)
# Carry available and slow-pool low parts too. Persistent available pools with
# zero configured loss otherwise accumulate independent physical/tag rounding.
s=s.replace('[:,1:3]','[:,1:5]').replace('(nt+1 if keep else 0,nu,2)','(nt+1 if keep else 0,nu,4)')
s=s.replace('np.zeros((nr*nl,2))','np.zeros((nr*nl,4))').replace('reshape(nr*nl,2)','reshape(nr*nl,4)').replace('correction.reshape(nr,nl,2)','correction.reshape(nr,nl,4)')
needle='            flux[t, u, 0], flux[t, u, 1] = fast, slow'
replacement='''            if compensated:
                for j,stock,delta,empty in ((2,N,_accurate_sum(np.array([source[t,u,3],ka,kp,-uptake,-E,-loss])),A==0. or pl[t,u]==1.),(3,L,_accurate_sum(np.array([(1-ff[t,u])*E,-slow])),lr[t,u]==1.)):
                    if empty:
                        states[newi,u,j+1]=0.;organic_correction[u,j]=0.
                    else:
                        corrected=delta-organic_correction[u,j]
                        updated=stock+corrected
                        organic_correction[u,j]=(updated-stock)-corrected
                        states[newi,u,j+1]=updated
''' + needle
s=s.replace(needle,replacement,1)
# Preserve the old adjoint and tag functions as unaccepted candidates. Their
# transfer derivatives and physical branch replay require separate validation.
target=ROOT/'d29_platform/precision_candidate.py'
target.write_text('# EXPERIMENTAL represented-state budget; formal acceptance pending.\n'+s,encoding='utf-8')
write_json(ROOT/'evidence/precision_candidate_identity.json',{'parent_sha256':sha(source),'candidate_sha256':sha(target),'transfer_sha256':sha(ROOT/'d29_platform/precision_transfer.py'),'formal_gate':False,'state_definition':'SON active, SON protected, available and slow stocks = high part minus saved compensation; compensation is required in checkpoint and budget','operator_definition':'largest destination is exact partition remainder; fractions changed at rounding scale only','derivative_status':'not yet accepted'})
