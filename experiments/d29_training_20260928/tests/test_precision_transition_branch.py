"""Forward/adjoint regression at low-part stock, transfer and near-depletion."""
import copy
import unittest
import numpy as np

from d29_platform import precision_candidate as kernel
from d29_platform.land1 import PROBABILITY_NAMES


class PrecisionTransitionTests(unittest.TestCase):
    def test_low_parts_transfer_and_near_depletion_gradient(self):
        nt,nr,nl=3,1,2
        initial=np.zeros((nr,nl,5),dtype=float)
        initial[:,:,0]=1.;initial[:,:,1:3]=1e6
        initial[:,:,3]=1.;initial[:,:,4]=.2
        sources=np.zeros((nt,nr,nl,4),dtype=float)
        sources[0,:,:,3]=1e-5
        target=np.full((nt,nr,nl),1.9999)
        outflow=np.zeros((nt,nr,nl,3))
        values=dict(zip(PROBABILITY_NAMES,[0.,0.,.2,0.,.4,.1]))
        probabilities={name:np.full((nt,nr,nl),value) for name,value in values.items()}
        matrix=np.array([[[.9,.1],[.1,.9]]])
        compensation=np.full((nr,nl,4),1e-10)
        spec=dict(initial=initial,sources=sources,plant_target=target,plant_outflows=outflow,
                  probabilities=probabilities,transitions={0:matrix},
                  keep_history=True,compensated=True,initial_compensation=compensation)
        base=kernel.run_land1(**spec)
        self.assertGreater(np.min(base.states[1,:,:,3]),0)
        self.assertLess(np.max(base.states[1,:,:,3]),2e-4)
        self.assertLess(base.max_local_balance_kg,1e-6)
        cotangent=np.zeros_like(base.fluxes);cotangent[...,0]=1.
        gradient=kernel.land1_adjoint(base,grad_fluxes=cotangent)
        for name,index in [('initial',(0,0,3)),('sources',(0,0,0,3)),('probabilities',(0,0,0))]:
            observed=[]
            for step in [1e-6,3e-7]:
                plus=copy.deepcopy(spec);minus=copy.deepcopy(spec)
                if name=='probabilities':
                    plus[name]['mobilize'][index]+=step;minus[name]['mobilize'][index]-=step
                    analytic=gradient[name]['mobilize'][index]
                else:
                    plus[name][index]+=step;minus[name][index]-=step
                    analytic=gradient[name][index]
                vp=kernel.run_land1(**plus).fluxes[...,0].sum()
                vm=kernel.run_land1(**minus).fluxes[...,0].sum()
                observed.append(abs((vp-vm)/(2*step)-analytic))
            self.assertLess(max(observed),1e-6*(1+abs(analytic)),name)


if __name__=='__main__':unittest.main()
