"""23-parameter candidate adapter. Formal calls require the LAND1 mass gate.

Diagnostic use is explicit and cannot produce a formally accepted fit. Annual
sources are reconstructed from independent products; no inverse TN input is read.
"""
import json
import numpy as np
from .u_adapter import UTraining
from .annual_chain import AnnualChain
from d29_platform.runtime import ROOT,sha
from d29_platform.conditional_inputs import ConditionalInputs
from d29_platform.coupling import response_mapping
from d29_platform.mineralization import land1_mineralization_fields
from d29_platform import precision_candidate as precision_kernel


class Land1Training(UTraining):
    def __init__(self,job,diagnostic=False):
        namespace=job.get('output_namespace','jobs')
        gate=ROOT/'outputs/land1_formal_gate.json'
        if not diagnostic and (not gate.exists() or not json.loads(gate.read_text(encoding='utf-8')).get('passed',False)):
            raise RuntimeError('LAND1_ABSOLUTE_MASS_AND_FULL_HISTORY_GATE_NOT_PASSED')
        if not diagnostic:
            receipt=json.loads(gate.read_text(encoding='utf-8'))
            identities=receipt.get('identities',{})
            if not identities or any(sha(path)!=value for path,value in identities.items()):
                raise RuntimeError('LAND1_ACCEPTED_IMPLEMENTATION_CHANGED')
        super().__init__(job)
        self.diagnostic=diagnostic
        self.kernel=precision_kernel
        retired={'log_tau_mineral_days'}|{f'gamma_lifetime_{i}' for i in range(7)}
        self.keep=[i for i,n in enumerate(self.model.names) if n not in retired]
        self.names=[self.model.names[i] for i in self.keep]
        self.initial=self.initial[self.keep];self.bounds=self.bounds[self.keep]
        assert len(self.names)==23
        self.inputs=ConditionalInputs(self.data.dates)
        self.mini=json.loads((ROOT/'config/mineralization_reference.json').read_text(encoding='utf-8'))
        if self.mini['mode']!='fixed' or self.mini['soil_temperature_enabled']:
            raise ValueError('LAND1_TRAINING_REQUIRES_FROZEN_FIXED_MINERALIZATION')
        slots=self.inputs.config['land1_integration']['active_land_slots']
        if len(slots)!=len(set(slots)) or any(not isinstance(i,int) or i<0 or i>=13 for i in slots):
            raise ValueError('INVALID_LAND1_ACTIVE_LAND_SLOTS')
        self.active=np.zeros((1,1,13));self.active[:,:,slots]=1.
        protected_rate=1/(float(self.mini['protected_turnover_years'])*float(self.mini['days_per_reference_year']))
        fields,_=land1_mineralization_fields(float(self.mini['k_active_per_day']),protected_rate,
                                              dt_days=float(self.mini['dt_days']),mode='fixed')
        self.mineralization_probabilities=fields

    def value_gradient(self,x,forward_only=False):
        import torch,time
        from d29_platform.land1 import hazard_to_probability
        from routing import RiverN,boundary_mass
        from temporal_model import aggregate_daily
        start=time.monotonic();t=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        named={name:t[i] for i,name in enumerate(self.names) if name!='log_source_correction_0'}
        h,f=response_mapping(self.model,named,as_numpy=False)
        hn=h.detach().numpy();fn=f.detach().numpy();pm,dp=hazard_to_probability(hn)
        def builder(year,initial):
            ix=np.flatnonzero(self.data.dates.year==year);post=initial[:,:,0].copy()
            if int(ix[0]) in self.inputs.transitions:
                post[:,:12]=np.einsum('ri,rij->rj',post[:,:12],self.inputs.transitions[int(ix[0])])
            a=self.inputs.annual(year,post)
            mask=np.zeros((230,13),bool);mask[:,[0,1,2,3,10,11]]=True
            probs=dict(mineralize_active=self.mineralization_probabilities['mineralize_active']*self.active,
                       mineralize_protected=self.mineralization_probabilities['mineralize_protected']*self.active,
                       mobilize=pm[ix,:,None]*self.active,available_loss=self.inputs.config['available_loss_probability']*self.active,
                       fast_fraction=fn[ix,:,None],lower_release=self.data.lower_release[ix,:,None]*self.active)
            return dict(**{k:a[k] for k in ['sources','plant_target','plant_outflows','transitions']},probabilities=probs,
                        target_initial_mask=mask,plant_activity_mode='potential_with_shortfall')
        chain=AnnualChain(range(1961,2025),builder,
                          intermediate_roundoff_tolerance=0.0,kernel=self.kernel)
        eta_index=self.names.index('log_source_correction_0')
        local,_=chain.forward(self.inputs.initial(),float(x[eta_index]))
        if not self.diagnostic and chain.max_local_balance_kg>1e-6:
            raise RuntimeError('LAND1_TRIAL_POINT_ABSOLUTE_MASS_FAILURE')
        mass=torch.tensor(local,requires_grad=True)
        vf=t[self.names.index('v_f')]
        inlet,outlet,releases=RiverN.apply(mass,vf,self.model.daily_data,'monthly',True)
        c,records,weights=self.model.daily_metadata(self.meta)
        station=boundary_mass(inlet,outlet,releases,mass,vf,c)
        p=aggregate_daily(station,c['water'],records,weights,len(self.meta),'MATCH')
        value,dobs,terms,_=self.objective.evaluate(p.detach().numpy())
        # Retired lifetime coordinates are at their zero-prior reference only;
        # they are neither fitted nor passed to LAND1 transport.
        reference=self.model.initial(0)
        full=[]
        for i in range(len(reference)):
            full.append(t[self.keep.index(i)] if i in self.keep else torch.tensor(reference[i]))
        prior=self.model.prior(torch.stack(full));R=.5*torch.dot(prior,prior)
        if forward_only:
            self.calls+=1
            self.last={'objective':float(value+R.detach()),'data_terms':terms,'prior':float(R.detach()),'seconds':time.monotonic()-start,'calls':self.calls,'diagnostic_only':self.diagnostic,'local_balance_kg':chain.max_local_balance_kg,'minimum_year_start_state_kg':chain.minimum_year_start_state_kg,'forward_only':True}
            return self.last['objective'],None
        torch.autograd.backward([p,R],[torch.tensor(dobs),None])
        g=chain.backward(mass.grad.detach().numpy());gh=np.zeros_like(hn);gf=np.zeros_like(fn)
        for part in g['response']:
            ix=np.flatnonzero(self.data.dates.year==part['year'])
            gh[ix]=np.sum(part['mobilize']*self.active,axis=2)*dp[ix]
            gf[ix]=np.sum(part['fast_fraction'],axis=2)
        torch.autograd.backward([h,f],[torch.tensor(gh),torch.tensor(gf)])
        gradient=t.grad.detach().numpy().copy();gradient[eta_index]+=g['log_external_multiplier']
        self.calls+=1
        self.last={'objective':float(value+R.detach()),'data_terms':terms,'prior':float(R.detach()),'seconds':time.monotonic()-start,'calls':self.calls,'diagnostic_only':self.diagnostic,'local_balance_kg':chain.max_local_balance_kg,'minimum_year_start_state_kg':chain.minimum_year_start_state_kg}
        return self.last['objective'],gradient
