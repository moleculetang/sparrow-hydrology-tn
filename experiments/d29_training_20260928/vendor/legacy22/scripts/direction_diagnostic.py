"""Local weighted prediction directions at the refitted R optimum, not a fit gate.

Prediction Jacobians use complete-history finite differences (two step sizes for
the new direction), checked against the analytic full-history objective adjoint.
No prior is included in the response-space projection.
"""
import argparse,time
import native_runtime as rt
from campaign_model import *

def main(short):
    jobs=rt.read(RUN/'configs/jobs.json');choices=[j for j in jobs if j['tag'].startswith(short+'_R_') and (RUN/'outputs'/j['tag']/'model.json').exists()]
    legal=[j for j in choices if rt.read(RUN/'outputs'/j['tag']/'audit.json').get('physical_reasonable')]
    if not legal:
        rt.write(RUN/'reports'/f'direction_{short}.json',dict(status='MISSING_LEGAL_BASELINE'));return
    job=min(legal,key=lambda j:rt.read(RUN/'outputs'/j['tag']/'model.json')['objective']);rt.label_barrier(job['fold'])
    m=for_job(dict(job,kind='STATE_MODULATED'));x=np.r_[rt.read(RUN/'outputs'/job['tag']/'model.json')['parameters'],0.]
    y0=m.predict(x,m.meta);jac=[];steps=[];calls=1
    for i in range(31):
        h=2e-5*max(1.,abs(x[i]));lo,hi=m.bounds[i];p=x.copy();n=x.copy()
        if x[i]-h>=lo and x[i]+h<=hi:
            p[i]+=h;n[i]-=h;col=(m.predict(p,m.meta)-m.predict(n,m.meta))/(2*h);scheme='central'
        else:
            sign=1 if x[i]-h<lo else -1;p[i]+=sign*h;n[i]+=sign*2*h
            col=sign*(-3*y0+4*m.predict(p,m.meta)-m.predict(n,m.meta))/(2*h);scheme='one_sided_second_order'
        calls+=2;jac.append(col);steps.append(dict(index=i,step=h,scheme=scheme))
    jac=np.column_stack(jac);weighted=np.sqrt(m.weight)[:,None]*jac
    half=[]
    for i in range(30):
        h=steps[i]['step']/2;lo,hi=m.bounds[i];p=x.copy();n=x.copy()
        if x[i]-h>=lo and x[i]+h<=hi:
            p[i]+=h;n[i]-=h;col=(m.predict(p,m.meta)-m.predict(n,m.meta))/(2*h)
        else:
            sign=1 if x[i]-h<lo else -1;p[i]+=sign*h;n[i]+=sign*2*h;col=sign*(-3*y0+4*m.predict(p,m.meta)-m.predict(n,m.meta))/(2*h)
        half.append(col);calls+=2
    half=np.column_stack(half)
    A=weighted[:,:30];z=weighted[:,30];scale=m.variable_scale()[:30]
    U,s,V=np.linalg.svd(A*scale[None,:],full_matrices=False);cutoff=max(A.shape)*np.finfo(float).eps*s[0];rank=int(np.sum(s>cutoff));projection=U[:,:rank]@(U[:,:rank].T@z)
    curves=[];second=[];first=[]
    for h in [1e-3,5e-4,1e-5]:
        p=x.copy();n=x.copy();p[-1]=h;n[-1]=-h;yp=m.predict(p,m.meta);yn=m.predict(n,m.meta);calls+=2
        first.append((yp-yn)/(2*h));second.append((yp-2*y0+yn)/(h*h))
        curves.append(dict(step=h,first_weighted_norm=float(np.linalg.norm(np.sqrt(m.weight)*first[-1])),second_weighted_norm=float(np.linalg.norm(np.sqrt(m.weight)*second[-1])),positive_change_norm=float(np.linalg.norm(np.sqrt(m.weight)*(yp-y0))),negative_change_norm=float(np.linalg.norm(np.sqrt(m.weight)*(yn-y0)))))
    uncertainty=float(np.linalg.norm(np.sqrt(m.weight)*(first[-1]-jac[:,-1])))
    matrix_step_error=float(np.linalg.norm(np.sqrt(m.weight)[:,None]*(half-jac[:,:30])*scale[None,:],ord=2))
    resolved_cutoff=max(cutoff,5*matrix_step_error);resolved_rank=int(np.sum(s>resolved_cutoff))
    resolved_projection=U[:,:resolved_rank]@(U[:,:resolved_rank].T@z)
    norm=float(np.linalg.norm(z));reliable=norm>max(10*uncertainty,1e-12)
    # Separate level/dynamics/prior gradients in exactly the current D objective.
    t=torch.tensor(x,requires_grad=True);pred=m.tensor_predict(t,m.meta);err=pred-torch.tensor(m.y);w=torch.tensor(m.weight)
    codes=pd.MultiIndex.from_frame(m.train[['station_key','year','month']]).factorize()[0];idx=torch.tensor(codes);ng=int(codes.max()+1)
    sumw=torch.zeros(ng).index_add(0,idx,w);mean=torch.zeros(ng).index_add(0,idx,w*err)/sumw
    level=.5*torch.sum(sumw*mean**2);dynamic=.5*torch.sum(w*(err-mean[idx])**2);prior=.5*m.prior(t).square().sum()
    grads=[torch.autograd.grad(v,t,retain_graph=True)[0].numpy() for v in [level,dynamic,prior]];calls+=1
    obj,analytic=m.value_gradient(x);calls+=1
    decomposition_error=float(np.max(abs(sum(grads)-analytic)))
    assert decomposition_error<1e-9*(1+abs(analytic).max()),decomposition_error
    pg=m.prior(torch.tensor(x,requires_grad=True));tt=torch.tensor(x,requires_grad=True);gp=torch.autograd.grad(.5*m.prior(tt).square().sum(),tt)[0].numpy()
    reconstructed=jac.T@(m.weight*(y0-m.y))+gp
    adjoint_error=float(abs(reconstructed-analytic).max())
    assert adjoint_error<1e-5*(1+abs(analytic).max()),adjoint_error
    angles={}
    for a,b in [(0,1),(0,2),(1,2)]:
        va=grads[a]*m.variable_scale();vb=grads[b]*m.variable_scale();den=np.linalg.norm(va)*np.linalg.norm(vb)
        angles[f'{a}_{b}']=None if den==0 else float(np.dot(va,vb)/den)
    out=RUN/'reports'/f'direction_{short}.npz';np.savez_compressed(out,prediction_jacobian=jac,weighted_jacobian=weighted,weights=m.weight,parameters=x,gradients=np.array(grads),singular_values_scaled=s,singular_values_raw=np.linalg.svd(A,compute_uv=False))
    result=dict(status='DIAGNOSTIC_COMPLETE',baseline=job['tag'],method='Complete-history prediction finite differences, validated against analytic objective adjoint; not an analytic output Jacobian.',rank=rank,rank_cutoff=float(cutoff),weighted_new_direction_norm=norm,step_disagreement=uncertainty,projection_reliable=reliable,explained_fraction=float(projection@projection/(z@z)) if reliable else None,unexplained_fraction=float((z-projection)@(z-projection)/(z@z)) if reliable else None,second_order=curves,steps=steps,gradient_terms={'level':float(level.detach()),'dynamic':float(dynamic.detach()),'prior':float(prior.detach())},scaled_gradient_norms=[float(np.linalg.norm(g*m.variable_scale())) for g in grads],gradient_cosines=angles,adjoint_error=adjoint_error,full_history_calls=calls,diagnostic_only=True)
    result.update(gradient_decomposition_error=decomposition_error,matrix_step_error=matrix_step_error,resolved_rank=resolved_rank,resolved_rank_cutoff=resolved_cutoff,resolved_explained_fraction=float(resolved_projection@resolved_projection/(z@z)) if reliable else None,rank_note='Machine-epsilon rank and an additional 5x step-disagreement rank are both descriptive, never fit admission rules.')
    rt.write(RUN/'reports'/f'direction_{short}.json',result);print(short,'direction complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fold');main(p.parse_args().fold)
