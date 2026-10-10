"""Create the new, separately identified capped kernel from a frozen copy.

V0 is the no-cap specialization. This never edits the parent experiment.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'d29_platform/precision_candidate.py'
s=p.read_text(encoding='utf8')
s=s.replace('# EXPERIMENTAL represented-state budget; formal acceptance pending.',
            '# Timing experiment: capped plant uptake; formal full-scale gate required.')
s=s.replace('initial_compensation, potential_activity):', 'initial_compensation, potential_activity, cap):',1)
s=s.replace('uptake = min(X, need)', 'uptake = min(X, need, cap[t,u])')
s=s.replace('raw_need > 0 and X >= need else', 'raw_need > 0 and X >= need and cap[t,u] >= need else')
s=s.replace('raw_need>0 and X>=need else', 'raw_need>0 and X>=need and cap[t,u]>=need else')
s=s.replace("plant_activity_mode='strict_prescribed', intermediate_roundoff_tolerance=0.0):",
            "plant_activity_mode='strict_prescribed', intermediate_roundoff_tolerance=0.0, plant_growth_capacity=None):")
s=s.replace("inp['plant_activity_mode']=plant_activity_mode", """inp['plant_activity_mode']=plant_activity_mode
    cap=np.full((nt,nr*nl),np.inf) if plant_growth_capacity is None else _field(plant_growth_capacity,(nt,nr,nl),'plant_growth_capacity').reshape(nt,nr*nl)
    inp['plant_growth_capacity']=cap""")
s=s.replace("plant_activity_mode=='potential_with_shortfall')", "plant_activity_mode=='potential_with_shortfall',cap)",1)
start=s.index('@njit(cache=True)\ndef _reverse(')
end=s.index('\n\ndef land1_adjoint',start)
reverse='''@njit(cache=True)
def _reverse(states, source, target, out, probs, event, matrices, nr, nl,
             gflux, gfinal, gstates, potential_activity, cap):
    nt, nu, _ = source.shape
    gi=np.zeros_like(source);go=np.zeros_like(out);gt=np.zeros_like(target)
    gc=np.zeros_like(target);gp=np.zeros((6,nt,nu));gm=np.zeros_like(matrices)
    adj=gfinal.copy();pa,pp,pm,pl,ff,lr=probs
    for t in range(nt-1,-1,-1):
        if gstates.shape[0]>0:adj+=gstates[t+1]
        before=states[t]
        if event[t]>=0:before=_transfer(before,matrices[event[t]],nr,nl)
        prev=np.empty_like(adj)
        for u in range(nu):
            P,SA,SP,N,L=before[u]
            X=N+source[t,u,3]+pa[t,u]*SA+pp[t,u]*SP
            O=out[t,u].sum();raw=target[t,u]+O-P-source[t,u,0]
            need=max(0.,raw);U=min(X,need,cap[t,u]);pre=P+source[t,u,0]+U
            after=target[t,u] if raw>0 and X>=need and cap[t,u]>=need else (P+source[t,u,0]-O)+U
            gP,gSA,gSP,gN,gL=adj[u];gf,gs,gloss,ge=gflux[t,u]
            cots=np.array([ge,gSA,gSP])
            if potential_activity and after<0:
                # A cap-limited shortage may leave X-U>0 for transport.
                gpre=np.dot(cots,out[t,u]/O)
                go[t,u]=(cots-gpre)*pre/O
            else:
                gpre=gP;go[t,u]=cots-gP
            A=X-U;E=pm[t,u]*A;lower=L+(1-ff[t,u])*E;rem=(1-pm[t,u])*A
            glower=gs*lr[t,u]+gL*(1-lr[t,u])
            gp[5,t,u]=(gs-gL)*lower
            gE=gf*ff[t,u]+glower*(1-ff[t,u]);gp[4,t,u]=(gf-glower)*E
            grem=gloss*pl[t,u]+gN*(1-pl[t,u]);gp[3,t,u]=(gloss-gN)*rem
            gA=gE*pm[t,u]+grem*(1-pm[t,u]);gp[2,t,u]=(gE-grem)*A
            gX=gA;gU=gpre-gA;gneed=0.
            # Replay deterministic min order: supply, then demand, then capacity.
            if raw>0:
                if X<=need and X<=cap[t,u]:gX+=gU
                elif need<=cap[t,u]:gneed=gU
                else:gc[t,u]=gU
            gt[t,u]=gneed;go[t,u]+=gneed
            gi[t,u,0]=gpre-gneed;gi[t,u,1]=gSA;gi[t,u,2]=gSP;gi[t,u,3]=gX
            prev[u,0]=gpre-gneed
            prev[u,1]=gSA*(1-pa[t,u])+gX*pa[t,u]
            prev[u,2]=gSP*(1-pp[t,u])+gX*pp[t,u]
            prev[u,3]=gX;prev[u,4]=glower
            gp[0,t,u]=(gX-gSA)*SA;gp[1,t,u]=(gX-gSP)*SP
        if event[t]>=0:
            idx=event[t];adj=np.zeros_like(prev)
            for r in range(nr):
                for old in range(nl):
                    for new in range(nl):
                        for k in range(5):
                            adj[r*nl+old,k]+=matrices[idx,r,old,new]*prev[r*nl+new,k]
                            gm[idx,r,old,new]+=states[t,r*nl+old,k]*prev[r*nl+new,k]
        else:adj=prev
    if gstates.shape[0]>0:adj+=gstates[0]
    return adj,gi,gt,go,gp,gm,gc
'''
s=s[:start]+reverse+s[end:]
s=s.replace("gst,i.get('plant_activity_mode')=='potential_with_shortfall')", "gst,i.get('plant_activity_mode')=='potential_with_shortfall',i['plant_growth_capacity'])")
s=s.replace("transitions={d: ans[5][k]", "plant_growth_capacity=ans[6].reshape(nt,nr,nl),\n                transitions={d: ans[5][k]")
s=s.replace('record, compensated, initial_compensation):', 'record, compensated, initial_compensation, cap):')
s=s.replace('record_history, compensated, corr)', "record_history, compensated, corr, i['plant_growth_capacity'])")
(root/'d29_platform/timing_kernel.py').write_text(s,encoding='utf8')
print('Created independent timing kernel; no acceptance inferred.')
