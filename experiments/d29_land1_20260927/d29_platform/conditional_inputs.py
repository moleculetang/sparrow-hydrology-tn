"""Explicit research adapters. Physical generation has no observations or masks.

The thirteenth land slot is a passive unresolved surface ledger, not soil.
Urban is also passive; this is a declared retention bound, not urban chemistry.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd
from .luh3_adapter import load_area_transfers,LAND_STATES
from .runtime import ROOT

class ConditionalInputs:
    def __init__(self,dates):
        self.dates=pd.DatetimeIndex(dates);self.parent=ROOT.parent/'20260926_1'
        self.config=json.loads((ROOT/'config/conditional_reference.json').read_text(encoding='utf-8'))
        self.area,self.transitions=load_area_transfers(self.parent/'outputs/luh3_reach/luh3_reach_state_1961_2024.csv',self.parent/'outputs/luh3_reach/luh3_land_area_transfer_operator_1961_2023.csv',dates)
        self.ag=pd.read_parquet(ROOT/'outputs/agriculture_reference/annual_crop_source_reference.parquet')
        self.activity=pd.read_parquet(ROOT/'outputs/agriculture_reference/annual_crop_activity_reference.parquet')
        self.dep=pd.read_parquet(self.parent/'outputs/deposition_landclass/monthly_reach_clcd_class_deposition_1961_2024.parquet')
        self.npp=pd.read_csv(self.parent/'outputs/mod17_gee_reachcover/reach_clcd_class_annual_npp_qc_2001_2025.csv')
        self.npp=self.npp[self.npp.year.between(2001,2024)&self.npp.clcd_class.isin([2,3,4,9])].copy()
        if self.npp.npp_negative_area_m2.sum()!=0:raise ValueError('NEGATIVE_NPP_REQUIRES_NEW_POLICY')
        self.npp['group']=np.where(self.npp.clcd_class==2,'forest','nonforest')
        n=self.npp.groupby(['year','reach_id','group'])[['npp_numeric_kg_C_year','npp_numeric_area_m2']].sum()
        n['density']=n.npp_numeric_kg_C_year/n.npp_numeric_area_m2
        donor=self.npp.groupby(['year','group'])[['npp_numeric_kg_C_year','npp_numeric_area_m2']].sum()
        donor['density']=donor.npp_numeric_kg_C_year/donor.npp_numeric_area_m2
        self.densities={};self.npp_donors=[]
        for y in range(2001,2025):
            for group in ('forest','nonforest'):
                v=n.xs((y,group),level=('year','group')).density.reindex(range(1,231)).to_numpy(copy=True)
                bad=~np.isfinite(v)
                replacement=float(donor.loc[(y,group),'density'])
                if not np.isfinite(replacement):raise ValueError('MISSING_NPP_GROUP_DONOR')
                for rid in np.flatnonzero(bad):self.npp_donors.append(dict(year=y,reach_id=int(rid+1),group=group,reference_kg_c_m2=replacement))
                v[bad]=replacement;self.densities[y,group]=v
        for group in ('forest','nonforest'):
            v=np.mean([self.densities[y,group] for y in range(2001,2021)],axis=0)
            for y in range(1961,2001):self.densities[y,group]=v
        self.labels=['deposition','fertilizer','field_manure','crop_BNF','initial_research_state']

    def land_weights(self,year,eligible):
        a=np.zeros((230,13));a[:,eligible]=self.area[year-1961,:,eligible].T
        total=a.sum(-1);bad=total==0
        a[bad,12]=1.;total[bad]=1.
        return a/total[:,None]

    def vegetation(self,year):
        """Annual N allocation, returns and retained wood, all kg N; no external N."""
        gross=np.zeros((230,13));returned=gross.copy();wood=gross.copy()
        for group,lands in [('forest',[0,2]),('nonforest',[1,3,10,11])]:
            c=self.densities[year,group];area=self.area[year-1961][:,lands]
            if group=='forest':
                a1,a2,a4=self.config['plant']['tree_allometry'];a3=2.7/(1+np.exp(-.004*(c*1000-300)))-.4
                leaf=c/(1+a1+a3*(1+a2));root=leaf*a1;woody=leaf*a3*(1+a2)
                cn=self.config['plant']['tree_cn'];ln=leaf/cn[0];rn=root/cn[1];wn=woody*(a4/cn[2]+(1-a4)/cn[3])
            else:
                a1=self.config['plant']['grass_root_leaf_carbon_ratio'];leaf=c/(1+a1);root=leaf*a1
                cn=self.config['plant']['grass_cn'];ln=leaf/cn[0];rn=root/cn[1];wn=np.zeros_like(c)
            # NPP already net primary production; do not subtract growth respiration twice.
            ret=ln*(1-self.config['plant']['leaf_resorption'])+rn
            gross[:,lands]=(ln+rn+wn)[:,None]*area
            returned[:,lands]=ret[:,None]*area;wood[:,lands]=wn[:,None]*area
        return gross,returned,wood

    def initial(self):
        s=pd.read_csv(ROOT/'outputs/soil_reference/son_soc14_reference.csv')
        density=s[s.depth==self.config['son']['depth']].set_index('reach_id').reindex(range(1,231)).son_reference_kg_m2.to_numpy()
        if not np.isfinite(density).all():raise ValueError('INITIAL_SON_REFERENCE_MISSING')
        active_fraction=float(self.config['son']['active_fraction'])
        if not 0 <= active_fraction <= 1:
            raise ValueError('INVALID_INITIAL_SON_ACTIVE_FRACTION')
        initial=np.zeros((230,13,5));initial[:,:12,1]=self.area[0]*density[:,None]*active_fraction
        initial[:,:12,2]=self.area[0]*density[:,None]*(1-active_fraction)
        # Urban initial soil is an inert stored initial-reference amount, not active soil.
        gross,ret,wood=self.vegetation(1961);initial[:,:,0]=gross-wood
        return initial

    def annual(self,year,initial_plant):
        idx=np.flatnonzero(self.dates.year==year);dates=self.dates[idx];nd=len(idx)
        tags=np.zeros((nd,230,13,5,4));weights=self.land_weights(year,[5,6,7,8,9])
        # Four deposition components remain traceable in the input table; entry is mineral.
        dep=self.dep[self.dep.year==year];quarantine=0.;dep_total=0.;dep_water=0.
        groups={1:[5,6,7,8,9],2:[0,2],3:[1,3,10,11],4:[1,3,10,11],9:[1,3,10,11]}
        for (month,cl),g in dep.groupby(['month','clcd_class']):
            v=g.set_index('reach_id').reindex(range(1,231)).total_kg_n
            # Sparse class rows: absence must be accompanied by absent class records,
            # not a NaN mass in a present record.
            if g.total_kg_n.isna().any():raise ValueError('DEPOSITION_CLASS_MISSING_VALUE')
            v=v.fillna(0).to_numpy();di=np.flatnonzero(dates.month==month)
            if int(cl)==5:dep_water+=float(v.sum());continue
            if int(cl) in groups:w=self.land_weights(year,groups[int(cl)])
            else:
                w=np.zeros((230,13));w[:,12]=1.
            mapped=v[:,None]*w;quarantine+=float(mapped[:,12].sum());dep_total+=float(v.sum())
            tags[di,:,:,0,3]+=mapped[None,:,:]/len(di)
        ag=self.ag[self.ag.year==year]
        mass=ag.groupby(['reach_id','field']).mass_kg.sum().unstack().reindex(range(1,231))
        if mass.isna().any().any():raise ValueError('AGRICULTURAL_REACH_NOT_COVERED')
        tags[:,:,:,1,3]=(mass.Nfer.to_numpy()[:,None]*weights)[None,:,:]/nd
        manure=mass.Nmanure.to_numpy()[:,None]*weights;mf=self.config['manure']['mineral_fraction'];af=self.config['manure']['organic_active_fraction']
        tags[:,:,:,2,3]=manure[None,:,:]*mf/nd;tags[:,:,:,2,1]=manure[None,:,:]*(1-mf)*af/nd;tags[:,:,:,2,2]=manure[None,:,:]*(1-mf)*(1-af)/nd
        a=self.activity[self.activity.year==year].groupby('reach_id')[['harvest_n_kg','residue_n_kg','bnf_plant_kg','bnf_soil_kg']].sum().reindex(range(1,231))
        if a.isna().any().any():raise ValueError('CROP_ACTIVITY_REACH_NOT_COVERED')
        tags[:,:,:,3,0]=(a.bnf_plant_kg.to_numpy()[:,None]*weights)[None,:,:]/nd
        tags[:,:,:,3,3]=(a.bnf_soil_kg.to_numpy()[:,None]*weights)[None,:,:]/nd
        crop_ret=a.residue_n_kg.to_numpy()[:,None]*weights*self.config['crop_residue']['returned']
        crop_exp=(a.harvest_n_kg+a.residue_n_kg*self.config['crop_residue']['exported_unresolved_fates']).to_numpy()[:,None]*weights
        gross,veg_ret,wood=self.vegetation(year)
        out=np.zeros((nd,230,13,3));af=self.config['crop_residue']['active_return_fraction']
        out[:,:,:,0]=crop_exp[None,:,:]/nd;out[:,:,:,1]=(crop_ret+veg_ret)[None,:,:]*af/nd;out[:,:,:,2]=(crop_ret+veg_ret)[None,:,:]*(1-af)/nd
        # Annual activity proxy: uniform harvest/return and accumulating annual wood.
        # No claimed observed harvest date or static minimum tissue pool on cropland.
        target=np.zeros((nd,230,13));vegetated=[0,1,2,3,10,11]
        target[:,:,vegetated]=initial_plant[:,vegetated][None,:,:]
        target+=np.arange(1,nd+1)[:,None,None]/nd*wood[None,:,:]
        # A source that has no crop land support is not allowed to fabricate a plant in quarantine.
        if np.any(out[:,:,12]>0) or np.any(tags[:,:,12,1:4]>0):raise ValueError('CROP_ACTIVITY_WITHOUT_LUH_CROPLAND')
        trans={}
        if int(idx[0]) in self.transitions:
            mat=np.zeros((230,13,13));mat[:,:12,:12]=self.transitions[int(idx[0])];mat[:,12,12]=1.;trans[0]=mat
        return dict(tags=tags,sources=tags.sum(axis=3),plant_target=target,plant_outflows=out,transitions=trans,indices=idx,
                    ledger=dict(year=year,deposition_land_kg=dep_total,deposition_water_excluded_kg=dep_water,quarantined_deposition_kg=quarantine,
                        fertilizer_kg=float(mass.Nfer.sum()),field_manure_kg=float(mass.Nmanure.sum()),bnf_plant_kg=float(a.bnf_plant_kg.sum()),bnf_soil_kg=float(a.bnf_soil_kg.sum()),
                        planned_crop_export_kg=float(crop_exp.sum()),planned_internal_crop_return_kg=float(crop_ret.sum()),planned_internal_vegetation_return_kg=float(veg_ret.sum()),
                        vegetation_gross_N_allocation_kg=float(gross.sum()),vegetation_wood_increment_kg=float(wood.sum())))
