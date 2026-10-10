"""All crop budgets are allocated once; unsupported dates retain explicit scenarios."""
import json
import numpy as np
from d29_platform.conditional_inputs import ConditionalInputs
from d29_platform.runtime import ROOT
from .profiles import profile,monthly_partition

class TimingInputs(ConditionalInputs):
    def __init__(self,dates,calendar='C0',plant_version='V0',parameters=None):
        super().__init__(dates)
        if calendar not in ('C0','C1','C2','K') or plant_version not in ('V0','V1'):
            raise ValueError('UNREGISTERED_TIMING_CONFIG')
        self.calendar=calendar;self.plant_version=plant_version
        self.profiles=json.loads((ROOT/'inputs/calendar_support/reach_profiles.json').read_text(encoding='utf8'))
        self.specs=json.loads((ROOT/'config/timing_evidence.json').read_text(encoding='utf8'))
        self.names=[];self.bounds=[];self.sigmas=[]
        if calendar=='K':
            for spec in self.specs['fertilizer_profiles']:
                bound=spec.get('basal_delta_bounds')
                if bound and spec['enabled']:
                    self.names.append(f"{spec['crop']}_{spec['source']}_basal")
                    self.bounds.append(bound);self.sigmas.append((bound[1]-bound[0])/2)
        if len(self.names)>8:raise ValueError('TOO_MANY_MANAGEMENT_COORDINATES')
        self.theta=np.zeros(len(self.names)) if parameters is None else np.asarray(parameters,float)
        if self.theta.shape!=(len(self.names),):raise ValueError('MANAGEMENT_COORDINATES')
        if self.names and np.any((self.theta<np.array(self.bounds)[:,0])|(self.theta>np.array(self.bounds)[:,1])):
            raise ValueError('MANAGEMENT_OUTSIDE_BOUNDS')
        self._profiles={};self.last_allocation=[]
        self._year_dates={y:self.dates[self.dates.year==y] for y in range(1961,2025)}

    def _daily_profile(self,year,crop,source,rid):
        key=(year,crop,source,rid,tuple(self.theta))
        if key in self._profiles:return self._profiles[key]
        n=len(self._year_dates[year]);u=np.full(n,1/n)
        spec=next((s for s in self.specs['fertilizer_profiles'] if s['crop']==crop and s['source']==source),None)
        reach=self.profiles.get(crop,{}).get(str(rid))
        admitted=year>=2016 and spec and spec['enabled'] and reach
        d=np.zeros(n);identity='uniform_research_scenario_date_unknown'
        if admitted:
            name=f'{crop}_{source}_basal';delta=self.theta[self.names.index(name)] if name in self.names else 0.
            w,d,_,_=profile(year,reach['seasons'],delta)
            identity=spec['identity']
            if self.calendar=='C0':w=u;d=np.zeros(n)
            elif self.calendar=='C1':
                _,w,_=monthly_partition(w,self._year_dates[year].month)
                d=np.zeros(n)
        else:w=u
        ans=(w,d,identity,bool(admitted))
        if admitted:self._profiles[key]=ans
        return ans

    def annual(self,year,initial_plant):
        self._profiles.clear()  # One year only; never retain a giant historical profile tensor.
        base=super().annual(year,initial_plant);n=len(base['indices']);wl=self.land_weights(year,[5,6,7,8,9])
        # The parent already aggregates every crop budget exactly once for its
        # uniform allocation. Reuse this exact specialization in the common
        # warmup and C0, rather than rebuilding an equivalent tensor each call.
        if year<2016:
            base['ledger'].update(source_multiplier=1.,timing=self.calendar,plant_version=self.plant_version,
                                 preintervention_exact_uniform=True)
            return base
        tags=base['tags']
        if self.calendar!='C0':tags[:,:,:,1:3,:]=0.
        self.last_allocation=[]
        mf=self.config['manure']['mineral_fraction'];ma=self.config['manure']['organic_active_fraction']
        rows=self.ag[self.ag.year==year]
        if rows.duplicated(['reach_id','crop','field']).any():raise ValueError('DUPLICATE_ANNUAL_CROP_BUDGET')
        if rows.mass_kg.isna().any() or (rows.mass_kg<0).any():raise ValueError('UNKNOWN_OR_NEGATIVE_BUDGET')
        months=self.dates[base['indices']].month
        for row in rows.itertuples():
            r=int(row.reach_id)-1;B=float(row.mass_kg)
            w,_,identity,admitted=self._daily_profile(year,row.crop,row.field,r+1)
            daily=B*w;daily[int(np.argmax(w))]+=B-daily.sum()
            if daily.min()<0:raise ValueError('NEGATIVE_DAILY_BUDGET')
            if self.calendar!='C0':
                field=daily[:,None]*wl[r][None,:]
                if row.field=='Nfer':tags[:,r,:,1,3]+=field
                elif row.field=='Nmanure':
                    tags[:,r,:,2,3]+=field*mf
                    tags[:,r,:,2,1]+=field*(1-mf)*ma
                    tags[:,r,:,2,2]+=field*(1-mf)*(1-ma)
                else:raise ValueError('UNKNOWN_AGRICULTURAL_SOURCE')
            monthly=np.array([daily[months==m].sum() for m in range(1,13)])
            self.last_allocation.append(dict(year=year,reach_id=r+1,crop=row.crop,source=row.field,
                budget_kg=B,monthly_kg=monthly.tolist(),day_sum_kg=float(daily.sum()),
                max_annual_error_kg=abs(float(daily.sum())-B),date_identity=identity,
                admitted_date_evidence=admitted,original_area_year=row.original_area_year,
                rate_year=row.rate_year,rate_donor_identity=row.rate_donor_identity))
        base['sources']=tags.sum(axis=3)
        if self.plant_version=='V1' and year>=2016:
            out=base['plant_outflows'];target=base['plant_target']
            # Infinity is an absent capacity on frozen noncrop units, not a missing input.
            cap=np.full(target.shape,np.finfo(float).max)
            cap[:,:,5:10]=0.;growth=np.zeros_like(target);cropout=np.zeros_like(out);shape=np.zeros_like(target)
            ret=self.config['crop_residue']['returned'];exp=self.config['crop_residue']['exported_unresolved_fates'];af=self.config['crop_residue']['active_return_fraction']
            for row in self.activity[self.activity.year==year].itertuples():
                r=int(row.reach_id)-1;rp=self.profiles.get(row.crop,{}).get(str(r+1))
                if rp:_,_,g,o,shape_crop=profile(year,rp['seasons'],return_target_shape=True)
                else:g=np.full(n,1/n);o=g;shape_crop=np.zeros(n)
                exported=row.harvest_n_kg+row.residue_n_kg*exp;returned=row.residue_n_kg*ret
                growth[:,r]+=(exported+returned)*g[:,None]*wl[r]
                shape[:,r]+=(exported+returned)*shape_crop[:,None]*wl[r]
                cropout[:,r,:,0]+=exported*o[:,None]*wl[r]
                cropout[:,r,:,1]+=returned*af*o[:,None]*wl[r]
                cropout[:,r,:,2]+=returned*(1-af)*o[:,None]*wl[r]
            out[:,:,5:10]=cropout[:,:,5:10]
            cap[:,:,5:10]=growth[:,:,5:10]
            target[:,:,5:10]=initial_plant[:,5:10][None,:,:]+shape[:,:,5:10]
            if target.min()<-1e-7:raise ValueError('NEGATIVE_STAGE_TARGET_REQUIRES_INPUT_REVIEW')
            # Tiny endpoint cancellation is handled by exact cumulative identities,
            # never by clipping a physical stock or introducing a nitrogen source.
            if target.min()<0:raise ValueError('STAGE_TARGET_FLOAT64_IDENTITY_REQUIRES_REPAIR')
            base['plant_growth_capacity']=cap
        base['ledger'].update(source_multiplier=1.,timing=self.calendar,plant_version=self.plant_version,
            crop_budget_rows=len(rows),date_evidence_mass_kg=sum(r['budget_kg'] for r in self.last_allocation if r['admitted_date_evidence']))
        return base

    def target_mask(self,year):
        mask=np.zeros((230,13),bool);mask[:,[0,1,2,3,10,11]]=True
        if self.plant_version=='V1' and year>=2016:mask[:,5:10]=True
        return mask

    def management_vjp(self,year,g,source_multiplier):
        if source_multiplier!=1:raise ValueError('SOURCE_MULTIPLIER_MUST_BE_ONE')
        ans=np.zeros(len(self.names))
        if year<2016 or not self.names:return ans
        wl=self.land_weights(year,[5,6,7,8,9]);cot=g['sources']
        mf=self.config['manure']['mineral_fraction'];af=self.config['manure']['organic_active_fraction']
        for row in self.ag[self.ag.year==year].itertuples():
            name=f'{row.crop}_{row.field}_basal'
            if name not in self.names:continue
            r=int(row.reach_id)-1;_,dw,_,_=self._daily_profile(year,row.crop,row.field,r+1)
            c=cot[:,r,:,3] if row.field=='Nfer' else mf*cot[:,r,:,3]+(1-mf)*(af*cot[:,r,:,1]+(1-af)*cot[:,r,:,2])
            ans[self.names.index(name)]+=row.mass_kg*float(np.sum(dw*np.sum(c*wl[r],axis=1)))
        return ans
