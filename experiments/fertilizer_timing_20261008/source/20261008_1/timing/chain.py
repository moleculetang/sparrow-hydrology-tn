"""Full warmup/year-start reverse; yearly response streaming, no gradient truncation."""
import numpy as np
from d29_training.annual_chain import AnnualChain

class TimingChain(AnnualChain):
    def _run(self,year,initial,eta,correction):
        if eta!=0.:raise ValueError('FIXED_MASS_HISTORY_REQUIRES_LOG_MULTIPLIER_ZERO')
        return super()._run(year,initial,eta,correction)

    def backward(self,grad_local,grad_final=None):
        grad_local=np.asarray(grad_local,float)
        if len(grad_local)!=sum(self.lengths):raise ValueError('FULL_HISTORY_COTANGENT_REQUIRED')
        adj=np.zeros_like(self.final) if grad_final is None else np.asarray(grad_final).copy()
        end=len(grad_local);response=[];management=np.zeros(len(self.management_inputs.names))
        for k in range(len(self.years)-1,-1,-1):
            initial,correction=self.checkpoints[k];year=self.years[k]
            result,mask=self._run(year,initial,self.eta,correction)
            start=end-self.lengths[k];gf=np.zeros_like(result.fluxes)
            gf[:,:,:,:2]=grad_local[start:end,:,None,None]
            g=self.kernel.land1_adjoint(result,grad_fluxes=gf,grad_final=adj)
            management+=self.management_inputs.management_vjp(year,g,1.)
            plant=np.sum(g['plant_target'],axis=0)*mask
            if 0 in result._inputs['transition_days']:
                plant=np.einsum('rij,rj->ri',result._inputs['matrices'][0],plant)
            adj=g['initial'];adj[:,:,0]+=plant
            part=dict(year=year,mobilize=g['probabilities']['mobilize'],fast_fraction=g['probabilities']['fast_fraction'])
            sink=getattr(self,'response_sink',None)
            if sink is None:response.append({key:(v.copy() if isinstance(v,np.ndarray) else v) for key,v in part.items()})
            else:sink(part)
            end=start
        return dict(initial=adj,log_external_multiplier=0.,response=list(reversed(response)),management=management)
