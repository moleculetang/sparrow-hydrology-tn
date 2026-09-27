"""Checkpoint/recompute adjoint including year-start-dependent plant targets.

Builder accepts (year, pre-transition initial state) and returns land1 keyword
fields plus target_initial_mask. For a marked land, every daily plant target
depends with derivative one on the post-transition year-start plant stock.
No observation labels or heldout masks are accepted by this physical chain.
"""
import numpy as np
from d29_platform.land1 import run_land1,land1_adjoint


class AnnualChain:
    def __init__(self,years,builder,intermediate_roundoff_tolerance=0.0):
        self.years=tuple(years);self.builder=builder
        self.intermediate_roundoff_tolerance=intermediate_roundoff_tolerance

    def _run(self,year,initial,eta,correction):
        spec=dict(self.builder(year,initial.copy()))
        mask=np.asarray(spec.pop('target_initial_mask'),bool)
        spec['sources']=np.asarray(spec['sources'])*np.exp(eta)
        optional=({'intermediate_roundoff_tolerance':self.intermediate_roundoff_tolerance}
                  if self.intermediate_roundoff_tolerance else {})
        result=run_land1(initial=initial,keep_history=True,compensated=True,
                         initial_compensation=correction,**optional,**spec)
        return result,mask

    def forward(self,initial,eta=0.):
        state=np.asarray(initial,dtype=float).copy();correction=None
        self.eta=float(eta);self.checkpoints=[];local=[];worst=0.
        self.minimum_year_start_state_kg=float(np.min(state))
        for year in self.years:
            self.minimum_year_start_state_kg=min(self.minimum_year_start_state_kg,float(np.min(state)))
            self.checkpoints.append((state.copy(),None if correction is None else correction.copy()))
            result,_=self._run(year,state,eta,correction)
            local.append(result.fluxes[:,:,:,:2].sum(axis=(2,3)))
            state=result.final.copy();correction=result.compensation.copy()
            worst=max(worst,result.max_local_balance_kg)
        self.lengths=[len(x) for x in local];self.final=state
        self.max_local_balance_kg=worst
        return np.concatenate(local),state.copy()

    def backward(self,grad_local,grad_final=None):
        grad_local=np.asarray(grad_local,dtype=float)
        if len(grad_local)!=sum(self.lengths):raise ValueError('FULL_HISTORY_COTANGENT_REQUIRED')
        adj=np.zeros_like(self.final) if grad_final is None else np.asarray(grad_final).copy()
        geta=0.;end=len(grad_local);response=[]
        for k in range(len(self.years)-1,-1,-1):
            initial,correction=self.checkpoints[k]
            result,mask=self._run(self.years[k],initial,self.eta,correction)
            start=end-self.lengths[k];gf=np.zeros_like(result.fluxes)
            gf[:,:,:,:2]=grad_local[start:end,:,None,None]
            g=land1_adjoint(result,grad_fluxes=gf,grad_final=adj)
            geta+=float(np.sum(g['sources']*result._inputs['sources'].reshape(g['sources'].shape)))
            plant=np.sum(g['plant_target'],axis=0)*mask
            # The annual target builder sees the plant state AFTER day-zero
            # relocation. This path is additional to the state recurrence.
            if 0 in result._inputs['transition_days']:
                matrix=result._inputs['matrices'][0]
                plant=np.einsum('rij,rj->ri',matrix,plant)
            adj=g['initial'];adj[:,:,0]+=plant
            response.append({'year':self.years[k], 'mobilize':g['probabilities']['mobilize'],
                             'fast_fraction':g['probabilities']['fast_fraction']})
            end=start
        return {'initial':adj,'log_external_multiplier':geta,'response':list(reversed(response))}
